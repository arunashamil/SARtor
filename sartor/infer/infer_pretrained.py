"""BLIP-2 pretrained model inference on RSICD dataset."""

import fire
import torch
from PIL import Image
from pathlib import Path

from transformers import Blip2ForConditionalGeneration, Blip2Processor
from sartor.modules.utils import json2csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_PATH = str(PROJECT_ROOT / "models" / "blip2-pretrained")
CAPS_DIR = str(PROJECT_ROOT / "data" / "pretrain" / "dataset_rsicd.json")
IMGS_DIR = str(PROJECT_ROOT / "data" / "pretrain" / "RSICD_images")


def main(max_new_tokens: int = 64, num_beams: int = 4, n_samples: int = 20) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")

    processor = Blip2Processor.from_pretrained(MODELS_PATH)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODELS_PATH, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    df = json2csv(CAPS_DIR)
    test_df = df.drop_duplicates(subset="Image Name").sample(n=n_samples, random_state=42)

    print("Successfully loaded BLIP-2 pretrained model")
    print("Prediction started ...\n")

    for i, row in test_df.iterrows():
        img_path = f"{IMGS_DIR}/{row['Image Name']}"
        img = Image.open(img_path).convert("RGB")

        inputs = processor(
            images=img,
            text="Describe this remote sensing image.",
            return_tensors="pt",
        ).to(device, dtype=torch.bfloat16)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"{i:05d}  {row['Image Name']}")
        print(f"       TRUE: {row['Caption']}")
        print(f"       PRED: {pred}\n")


if __name__ == "__main__":
    fire.Fire(main)
