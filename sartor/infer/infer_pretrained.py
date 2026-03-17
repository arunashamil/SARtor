import textwrap

import fire
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor
from sartor.modules.generate import generate
from sartor.modules.utils import json2csv

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_PATH = str(PROJECT_ROOT / "models" / "sartor-pretrained")
CAPS_DIR = str(PROJECT_ROOT / "data" / "pretrain" / "dataset_rsicd.json")
IMGS_DIR = str(PROJECT_ROOT / "data" / "pretrain" / "RSICD_images")

COLS = 4


def main(max_new_tokens: int = 64, num_beams: int = 4, n_samples: int = 20) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    df = json2csv(CAPS_DIR)
    test_df = df.drop_duplicates(subset="Image Name").sample(n=n_samples, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH)

    model = VisionEncoderDecoderModel.from_pretrained(MODELS_PATH).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(MODELS_PATH)

    print(f"Loaded pretrained model. Running inference on {n_samples} samples...\n")

    results = []
    for _, row in test_df.iterrows():
        img_path = f"{IMGS_DIR}/{row['Image Name']}"
        pred = generate(
            model,
            processor,
            tokenizer,
            img_path,
            device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        results.append((img_path, row["Image Name"], row["Caption"], pred))
        print(f"[{row['Image Name']}]  PRED: {pred}")

    rows = (len(results) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 4, rows * 5))
    axes = axes.flatten()

    for idx, (img_path, name, gt, pred) in enumerate(results):
        ax = axes[idx]
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(name, fontsize=9, fontweight="bold")
        caption = f"GT: {textwrap.fill(gt, 40)}\nPRED: {textwrap.fill(pred, 40)}"
        ax.set_xlabel(caption, fontsize=7, ha="left", x=0)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(results), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
