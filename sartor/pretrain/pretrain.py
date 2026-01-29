import hydra
from omegaconf import DictConfig

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTImageProcessor, AutoModel
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

from sartor.modules.dataset import ImgDataset
from sartor.modules.compute_metrics import compute_metrics
from sartor.modules.utils import json2csv


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    num_workers = mp.cpu_count()
    feature_extractor = ViTImageProcessor.from_pretrained(config["pretrain"]["encoder"])
    tokenizer = AutoTokenizer.from_pretrained(config["pretrain"]["decoder"])
    tokenizer.pad_token = tokenizer.unk_token

    transformations = transforms.Compose(
        [
            transforms.Resize((config["pretrain"]["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["pretrain"]["mean"],
                std=config["pretrain"]["std"]
            )
        ]
    )
    
    df = json2csv(config["pretrain"]["caps_dir"])

    train_df, val_df = train_test_split(df, 
                                        test_size=1-config["pretrain"]["train_pct"], 
                                        random_state=config["pretrain"]["seed"])

    train_dataset = ImgDataset(
        train_df, 
        root_dir=config["pretrain"]["imgs_dir"],
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_length=config["pretrain"]["max_length"],
        )
    
    val_dataset = ImgDataset(
        val_df, 
        root_dir = config["pretrain"]["imgs_dir"],
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_length=config["pretrain"]["max_length"],
        )

    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        config["pretrain"]["encoder"], 
        config["pretrain"]["decoder"]
        )
    
    tokenizer.pad_token = tokenizer.eos_token

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.config.vocab_size = model.config.decoder.vocab_size    
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["pretrain"]["output_model"],
        per_device_train_batch_size=config["pretrain"]["train_batch_size"],
        per_device_eval_batch_size=config["pretrain"]["val_batch_size"],
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=config["pretrain"]["logging_steps"],
        save_steps=config["pretrain"]["save_steps"],
        warmup_steps=config["pretrain"]["warmup_steps"],
        learning_rate= config["pretrain"]["lr"],
        num_train_epochs=config["pretrain"]["epochs"],
        overwrite_output_dir=True,
        save_total_limit=1,
        fp16=True,
        optim="adamw_8bit",
    )

    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # trainer.train()
    

if __name__ == "__main__":
    main()