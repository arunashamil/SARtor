import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, max_length=50, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.df["Caption"].iloc[idx]
        image = self.df["Image Name"].iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            img = torch.from_numpy(np.array(img)).float()
            img = (img + 1.0) / 2.0

        pixel_values = self.feature_extractor(
            images=img, 
            return_tensors='pt',
            do_normalize=True).pixel_values

        captions = self.tokenizer(caption, 
                                  padding='max_length',
                                  max_length=self.max_length).input_ids
        
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]

        encoding = {'pixel_values': pixel_values.squeeze(), 'labels': torch.tensor(captions)}

        return encoding