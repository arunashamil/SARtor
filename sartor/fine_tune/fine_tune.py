import hydra
from omegaconf import DictConfig

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import functools

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from transformers import AutoTokenizer, default_data_collator

from sartor.modules.dataset import ImgDataset
from sartor.modules.compute_metrics import compute_metrics


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    feature_extractor = AutoImageProcessor.from_pretrained(config["pretrain"]["encoder"], use_fast=False)
    
    tokenizer = AutoTokenizer.from_pretrained(config["pretrain"]["decoder"])
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(config["fine_tune"]["caps_dir"])
    df.columns = [col.strip() for col in df.columns]
    
    train_df, val_df = train_test_split(
        df,
        train_size=config["fine_tune"]["train_pct"],
        random_state=config["fine_tune"]["seed"])

    train_dataset = ImgDataset(
        train_df, 
        root_dir=config["fine_tune"]["imgs_dir"],
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_length=config["fine_tune"]["max_length"],
        )
    
    val_dataset = ImgDataset(
        val_df, 
        root_dir = config["fine_tune"]["imgs_dir"],
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_length=config["fine_tune"]["max_length"],
        )

    model = VisionEncoderDecoderModel.from_pretrained(config["pretrain"]["output_model"])
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.eos_token_id

    gen = model.generation_config
    
    gen.eos_token_id = tokenizer.eos_token_id
    gen.pad_token_id = tokenizer.pad_token_id
    gen.decoder_start_token_id = tokenizer.eos_token_id

    gen.max_new_tokens = 60
    gen.min_new_tokens = 12
    gen.num_beams = 4
    gen.early_stopping = True
    gen.no_repeat_ngram_size = 3
    gen.length_penalty = 1.2

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["fine_tune"]["output_model"],
        per_device_train_batch_size=config["fine_tune"]["train_batch_size"],
        per_device_eval_batch_size=config["fine_tune"]["val_batch_size"],
        predict_with_generate=True,
        label_smoothing_factor = config["fine_tune"]["label_smoothing_factor"],
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
        learning_rate= config["fine_tune"]["lr"],
        num_train_epochs=config["fine_tune"]["epochs"],
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
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