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
from transformers import VisionEncoderDecoderModel , ViTImageProcessor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

from sartor.modules.dataset import ImgDataset


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
    feature_extractor = ViTImageProcessor.from_pretrained(config["train"]["encoder"])
    tokenizer = AutoTokenizer.from_pretrained(config["train"]["decoder"])
    tokenizer.pad_token = tokenizer.unk_token

    transformations = transforms.Compose(
        [
            transforms.Resize((config["train"]["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0.5,
                std=0.5
            )
        ]
    )

    train_df = pd.read_csv(config["train"]["caps_dir"])
    
    train_dataset = ImgDataset(
        train_df, 
        root_dir=config["train"]["imgs_dir"],
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        transform=transformations
        )
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        config["train"]["encoder"], 
        config["train"]["decoder"]
        )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="VIT_large_gpt2",
        per_device_train_batch_size=config["train"]["train_batch_size"],
        per_device_eval_batch_size=config["train"]["val_batch_size"],
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=config["train"]["logging_steps"],
        save_steps=config["train"]["save_steps"],
        warmup_steps=config["train"]["warmup_steps"],
        learning_rate= config["train"]["lr"],
        num_train_epochs=config["train"]["epochs"],
        overwrite_output_dir=True,
        save_total_limit=1
    )

if __name__ == "__main__":
    main()