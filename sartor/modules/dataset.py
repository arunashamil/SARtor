import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, max_length, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = self.df["Image Name"].iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")

        caption = self.df["Caption"].iloc[idx].strip()
        caption_ids = self.tokenizer(caption, add_special_tokens=False)["input_ids"]
        caption_ids = caption_ids[:self.max_length - 1]

        pixel_values = self.feature_extractor(
            images=img, 
            return_tensors='pt',
            do_normalize=True
            ).pixel_values[0] # single image is passed

        input_ids = caption_ids + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        labels = caption_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }