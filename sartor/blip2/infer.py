"""BLIP-2 inference: captioning, VQA, and image retrieval."""

import os
import fire
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from transformers import Blip2ForConditionalGeneration, Blip2Processor


MODELS_PATH = "models/blip2-finetuned"
CAPS_DIR = "data/fine_tune/Text/Caption/test/Caption_test.csv"
VQA_DIR = "data/fine_tune/Text/VQA/test"
IMGS_DIR = "data/fine_tune/SARimages_preprocessed/SARimages"


def load_model(model_path, device):
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
    ).to(device).eval()
    return model, processor


def generate_caption(model, processor, img_path, device,
                     prompt="Describe this SAR image in detail.",
                     max_new_tokens=64, num_beams=4):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(
        device, dtype=torch.float16
    )
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def caption(model_path=MODELS_PATH, max_new_tokens=64, num_beams=4):
    """Generate captions for test images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_path, device)

    test_df = pd.read_csv(CAPS_DIR)
    test_df.columns = [col.strip() for col in test_df.columns]

    for i, row in test_df.iterrows():
        img_path = f"{IMGS_DIR}/{row['Image Name']}"
        pred = generate_caption(
            model, processor, img_path, device,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
        )
        print(f"{i:05d}  {row['Image Name']}")
        print(f"       TRUE: {row['Caption']}")
        print(f"       PRED: {pred}\n")


def vqa(model_path=MODELS_PATH, dataset="SARVQA2", n_samples=100,
        max_new_tokens=32, num_beams=4):
    """Answer questions about SAR images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_path, device)

    csv_path = os.path.join(VQA_DIR, f"{dataset}_test.csv")
    test_df = pd.read_csv(csv_path)
    test_df.columns = [col.strip() for col in test_df.columns]

    if n_samples > 0 and len(test_df) > n_samples:
        test_df = test_df.sample(n=n_samples, random_state=42)

    correct = 0
    total = 0
    for i, row in test_df.iterrows():
        img_path = f"{IMGS_DIR}/{row['Image Name']}"
        question = row["Question"]

        pred = generate_caption(
            model, processor, img_path, device,
            prompt=question, max_new_tokens=max_new_tokens, num_beams=num_beams,
        )

        is_correct = pred.lower().strip() == str(row["Answer"]).lower().strip()
        correct += is_correct
        total += 1

        print(f"{total:04d}  {row['Image Name']}")
        print(f"       Q: {question}")
        print(f"       A: {row['Answer']}")
        print(f"       PRED: {pred}  {'OK' if is_correct else 'X'}\n")

    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")


def retrieve(model_path=MODELS_PATH, query="bridge", top_k=10):
    """Find images matching a text query using BLIP-2 image-text matching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    img_dir = Path(IMGS_DIR)
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.jpg"))

    print(f"Building index for {len(img_files)} images...")
    print(f"Query: '{query}'")

    # Generate captions for all images, then rank by query relevance
    # (simple approach: generate caption, check if query appears)
    # For production: use BLIP-2 image-text matching scores
    results = []
    batch_size = 16

    for i in tqdm(range(0, len(img_files), batch_size)):
        batch_files = img_files[i:i + batch_size]
        images = [Image.open(f).convert("RGB") for f in batch_files]

        inputs = processor(
            images=images,
            text=[f"Does this image contain a {query}? Answer:"] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(device, dtype=torch.float16)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=8)

        answers = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for f, ans in zip(batch_files, answers):
            ans_lower = ans.strip().lower()
            if "yes" in ans_lower:
                results.append((f.name, ans.strip()))

    print(f"\nFound {len(results)} images matching '{query}':")
    for name, answer in results[:top_k]:
        print(f"  {name}  ({answer})")

    if len(results) > top_k:
        print(f"  ... and {len(results) - top_k} more")

    return results


def main():
    fire.Fire({
        "caption": caption,
        "vqa": vqa,
        "retrieve": retrieve,
    })


if __name__ == "__main__":
    main()
