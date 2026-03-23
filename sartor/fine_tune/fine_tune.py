"""Fine-tune BLIP-2 on SAR captioning + VQA with discriminative learning rates."""

import os
import functools
import multiprocessing as mp

import hydra
from omegaconf import DictConfig

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from sartor.modules.blip2_dataset import (
    Blip2CaptionDataset,
    Blip2VQADataset,
    blip2_collate_fn,
)
from sartor.modules.compute_metrics import compute_metrics


class Blip2Trainer(Seq2SeqTrainer):
    """Trainer with discriminative learning rates for vision encoder vs Q-Former."""

    def __init__(self, vision_lr, *args, **kwargs):
        self.vision_lr = vision_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        vision_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("vision_model"):
                vision_params.append(param)
            else:
                other_params.append(param)

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        base_lr = optimizer_kwargs.pop("lr", self.args.learning_rate)

        param_groups = [{"params": other_params, "lr": base_lr}]
        if vision_params:
            param_groups.append({"params": vision_params, "lr": self.vision_lr})

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer


def load_vqa_data(vqa_dir, dataset_name):
    """Load VQA CSV data from the specified dataset."""
    csv_path = os.path.join(vqa_dir, f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    return df


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config["blip2"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")

    num_workers = min(mp.cpu_count(), 8)
    orig_cwd = hydra.utils.get_original_cwd()

    caps_dir = os.path.join(orig_cwd, cfg["ft_caps"])
    imgs_dir = os.path.join(orig_cwd, cfg["ft_imgs"])
    vqa_train_dir = os.path.join(orig_cwd, cfg["ft_vqa_train"])
    output_model = os.path.join(orig_cwd, cfg["output_finetuned"])

    # Load from pretrained BLIP-2 (ours or Salesforce)
    pretrained_path = os.path.join(orig_cwd, cfg["output_pretrained"])
    model_path = pretrained_path if os.path.exists(pretrained_path) else cfg["model"]
    print(f"Loading BLIP-2 from: {model_path}")

    processor = Blip2Processor.from_pretrained(
        pretrained_path if os.path.exists(pretrained_path) else cfg["model"]
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
    )

    # Freeze LLM, selectively unfreeze vision encoder
    model.language_model.requires_grad_(False)
    model.qformer.requires_grad_(True)
    model.language_projection.requires_grad_(True)

    # Unfreeze last N layers of vision encoder
    model.vision_model.requires_grad_(False)
    n_unfreeze = cfg["unfreeze_vision_layers"]
    if n_unfreeze > 0:
        encoder_layers = model.vision_model.encoder.layers
        for layer in encoder_layers[-n_unfreeze:]:
            layer.requires_grad_(True)
        model.vision_model.post_layernorm.requires_grad_(True)
        print(f"Unfreeze last {n_unfreeze}/{len(encoder_layers)} vision encoder layers")

    # Upcast trainable parts to float32 for stable gradients
    model.qformer.float()
    model.language_projection.float()
    for param in model.vision_model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # --- Caption data ---
    cap_df = pd.read_csv(caps_dir)
    cap_df.columns = [col.strip() for col in cap_df.columns]
    if "Caption Type" in cap_df.columns:
        cap_df["Caption Type"] = cap_df["Caption Type"].str.strip()
        cap_df = cap_df[cap_df["Caption Type"] == "complex caption"]
    print(f"Caption samples: {len(cap_df)}")

    cap_train, cap_val = train_test_split(
        cap_df, train_size=cfg["train_pct"], random_state=cfg["seed"]
    )
    train_cap_ds = Blip2CaptionDataset(
        cap_train, imgs_dir, processor, cfg["max_target_length"],
        prompt="Describe this SAR image in detail.",
    )
    val_cap_ds = Blip2CaptionDataset(
        cap_val, imgs_dir, processor, cfg["max_target_length"],
        prompt="Describe this SAR image in detail.",
    )

    # --- VQA data ---
    vqa_name = cfg["vqa_dataset"]
    vqa_train_name = f"{vqa_name}_train"
    vqa_df = load_vqa_data(vqa_train_dir, vqa_train_name)
    max_vqa = cfg["vqa_max_samples"]
    if max_vqa > 0 and len(vqa_df) > max_vqa:
        vqa_df = vqa_df.sample(n=max_vqa, random_state=cfg["seed"])
    print(f"VQA samples ({vqa_name}): {len(vqa_df)}")

    vqa_train, vqa_val = train_test_split(
        vqa_df, train_size=cfg["train_pct"], random_state=cfg["seed"]
    )
    train_vqa_ds = Blip2VQADataset(
        vqa_train, imgs_dir, processor, cfg["max_input_length"], cfg["max_target_length"]
    )
    val_vqa_ds = Blip2VQADataset(
        vqa_val, imgs_dir, processor, cfg["max_input_length"], cfg["max_target_length"]
    )

    # --- Combine datasets ---
    train_dataset = ConcatDataset([train_cap_ds, train_vqa_ds])
    val_dataset = ConcatDataset([val_cap_ds, val_vqa_ds])
    print(f"Total training samples: {len(train_dataset)} "
          f"(caption: {len(train_cap_ds)}, VQA: {len(train_vqa_ds)})")

    collate_fn = functools.partial(
        blip2_collate_fn, pad_token_id=processor.tokenizer.pad_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["val_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        save_strategy="steps",
        logging_strategy="steps",
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        warmup_steps=cfg["ft_warmup_steps"],
        learning_rate=cfg["ft_lr"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["ft_epochs"],
        dataloader_num_workers=num_workers,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        save_total_limit=1,
        bf16=True,
        optim="adamw_8bit",
        generation_max_length=cfg["max_new_tokens"],
    )

    trainer = Blip2Trainer(
        vision_lr=cfg["ft_vision_lr"],
        model=model,
        args=training_args,
        processing_class=processor.tokenizer,
        compute_metrics=functools.partial(
            compute_metrics, tokenizer=processor.tokenizer
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    trainer.model_accepts_loss_kwargs = False

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_model)


if __name__ == "__main__":
    main()
