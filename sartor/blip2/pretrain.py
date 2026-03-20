"""Pretrain BLIP-2 Q-Former on RSICD remote sensing captions."""

import os
import functools
import multiprocessing as mp

import hydra
from omegaconf import DictConfig

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from sartor.modules.blip2_dataset import Blip2CaptionDataset, blip2_collate_fn
from sartor.modules.compute_metrics import compute_metrics
from sartor.modules.utils import json2csv


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

    caps_dir = os.path.join(orig_cwd, cfg["pretrain_caps"])
    imgs_dir = os.path.join(orig_cwd, cfg["pretrain_imgs"])
    output_model = os.path.join(orig_cwd, cfg["output_pretrained"])

    print(f"Loading BLIP-2 model: {cfg['model']}")
    processor = Blip2Processor.from_pretrained(cfg["model"])
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16,
    )

    # Freeze everything except Q-Former
    model.vision_model.requires_grad_(False)
    model.language_model.requires_grad_(False)
    model.language_projection.requires_grad_(True)
    model.qformer.requires_grad_(True)

    # Upcast Q-Former to float32 for stable gradients
    # (language_projection stays in bfloat16 to match Q-Former output during generate())
    model.qformer.float()
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Load RSICD data
    df = json2csv(caps_dir)
    train_df, val_df = train_test_split(
        df, train_size=cfg["train_pct"], random_state=cfg["seed"]
    )

    train_dataset = Blip2CaptionDataset(
        train_df, imgs_dir, processor, cfg["max_target_length"],
        prompt="Describe this remote sensing image.",
    )
    val_dataset = Blip2CaptionDataset(
        val_df, imgs_dir, processor, cfg["max_target_length"],
        prompt="Describe this remote sensing image.",
    )

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
        eval_strategy="steps",
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        warmup_steps=cfg["pretrain_warmup_steps"],
        learning_rate=cfg["pretrain_lr"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["pretrain_epochs"],
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

    trainer = Seq2SeqTrainer(
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
