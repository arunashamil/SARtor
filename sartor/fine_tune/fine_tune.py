import os
import hydra
from omegaconf import DictConfig

import multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import functools

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from transformers import AutoTokenizer, default_data_collator

from sartor.modules.dataset import ImgDataset
from sartor.modules.compute_metrics import compute_metrics


class SARTrainer(Seq2SeqTrainer):
    def __init__(self, encoder_lr, *args, **kwargs):
        self.encoder_lr = encoder_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        encoder_params = []
        decoder_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder"):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        base_lr = optimizer_kwargs.pop("lr", self.args.learning_rate)

        self.optimizer = optimizer_cls(
            [
                {"params": encoder_params, "lr": self.encoder_lr},
                {"params": decoder_params, "lr": base_lr},
            ],
            **optimizer_kwargs,
        )
        return self.optimizer


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    num_workers = min(mp.cpu_count(), 8)
    orig_cwd = hydra.utils.get_original_cwd()

    caps_dir = os.path.join(orig_cwd, config["fine_tune"]["caps_dir"])
    imgs_dir = os.path.join(orig_cwd, config["fine_tune"]["imgs_dir"])
    output_model = os.path.join(orig_cwd, config["fine_tune"]["output_model"])
    pretrained_model = os.path.join(orig_cwd, config["pretrain"]["output_model"])

    feature_extractor = AutoImageProcessor.from_pretrained(pretrained_model, use_fast=False)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    df = pd.read_csv(caps_dir)
    df.columns = [col.strip() for col in df.columns]

    if "Caption Type" in df.columns:
        df["Caption Type"] = df["Caption Type"].str.strip()
        df = df[df["Caption Type"] == "complex caption"]
        print(f"Filtered to complex captions: {len(df)} samples")

    train_df, val_df = train_test_split(
        df,
        train_size=config["fine_tune"]["train_pct"],
        random_state=config["fine_tune"]["seed"])

    train_dataset = ImgDataset(
        train_df,
        root_dir=imgs_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_length=config["fine_tune"]["max_length"],
        )

    val_dataset = ImgDataset(
        val_df,
        root_dir = imgs_dir,
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_length=config["fine_tune"]["max_length"],
        )

    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model)

    # Freeze only early encoder stages — let later stages learn SAR features
    encoder_frozen_stages = config["fine_tune"]["encoder_frozen_stages"]
    frozen_encoder_prefixes = (
        tuple(f"encoder.layers.{i}." for i in range(encoder_frozen_stages))
        + ("embeddings.",)
    )

    for name, param in model.encoder.named_parameters():
        if name.startswith(frozen_encoder_prefixes):
            param.requires_grad = False

    # Freeze early decoder layers, keep wte unfrozen (tied to lm_head)
    frozen_layers = config["fine_tune"]["decoder_frozen_layers"]
    frozen_prefixes = tuple(f"transformer.h.{i}." for i in range(frozen_layers))

    for name, param in model.decoder.named_parameters():
        if name.startswith(frozen_prefixes) or "wpe" in name:
            param.requires_grad = False

    def unshifted_loss(logits, labels, vocab_size, **kwargs):
        return F.cross_entropy(logits.float().view(-1, vocab_size), labels.view(-1), ignore_index=-100)

    model.loss_function = unshifted_loss

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    gen = model.generation_config

    gen.eos_token_id = tokenizer.eos_token_id
    gen.pad_token_id = tokenizer.pad_token_id
    gen.decoder_start_token_id = tokenizer.bos_token_id

    gen.max_new_tokens = 60
    gen.min_new_tokens = 8
    gen.num_beams = 6
    gen.num_beam_groups = 3
    gen.diversity_penalty = 1.0
    gen.early_stopping = True
    gen.no_repeat_ngram_size = 3
    gen.length_penalty = 1.2

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=config["fine_tune"]["train_batch_size"],
        per_device_eval_batch_size=config["fine_tune"]["val_batch_size"],
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        save_strategy="steps",
        logging_strategy="steps",
        eval_strategy=config["fine_tune"]["eval_strategy"],
        eval_steps=config["fine_tune"]["eval_steps"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=config["fine_tune"]["logging_steps"],
        save_steps=config["fine_tune"]["save_steps"],
        warmup_steps=config["fine_tune"]["warmup_steps"],
        gradient_accumulation_steps=config["fine_tune"]["grad_accum_steps"],
        learning_rate=config["fine_tune"]["lr"],
        weight_decay=config["fine_tune"]["weight_decay"],
        num_train_epochs=config["fine_tune"]["epochs"],
        dataloader_num_workers=num_workers,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        save_total_limit=1,
        fp16=True,
        optim="adamw_8bit",
        label_smoothing_factor=config["fine_tune"]["label_smoothing_factor"],
    )

    trainer = SARTrainer(
        encoder_lr=config["fine_tune"]["encoder_lr"],
        processing_class=tokenizer,
        model=model,
        args=training_args,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model()
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
