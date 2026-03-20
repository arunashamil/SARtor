import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class Blip2CaptionDataset(Dataset):
    """Dataset for BLIP-2 image captioning (pretrain on RSICD, fine-tune on SAR)."""

    def __init__(self, df, root_dir, processor, max_target_length,
                 prompt="Describe this satellite image in detail."):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.prompt = prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df["Image Name"].iloc[idx]
        img_path = os.path.join(self.root_dir, image_name)
        img = Image.open(img_path).convert("RGB")

        caption = self.df["Caption"].iloc[idx].strip()

        inputs = self.processor(
            images=img,
            text=self.prompt,
            return_tensors="pt",
            max_length=self.max_target_length,
            truncation=True,
        )

        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids[0]

        return {
            "pixel_values": inputs.pixel_values[0],
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": labels,
        }


class Blip2VQADataset(Dataset):
    """Dataset for BLIP-2 visual question answering on SAR images."""

    def __init__(self, df, root_dir, processor, max_input_length, max_target_length):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.processor = processor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df["Image Name"].iloc[idx]
        img_path = os.path.join(self.root_dir, image_name)
        img = Image.open(img_path).convert("RGB")

        question = self.df["Question"].iloc[idx].strip()
        answer = self.df["Answer"].iloc[idx].strip()

        inputs = self.processor(
            images=img,
            text=question,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )

        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids[0]

        return {
            "pixel_values": inputs.pixel_values[0],
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": labels,
        }


def blip2_collate_fn(batch, pad_token_id):
    """Dynamic padding collator for BLIP-2 datasets."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
