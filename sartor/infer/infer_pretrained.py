import fire
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor
from sartor.modules.generate import generate
from sartor.modules.utils import json2csv

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_PATH = str(PROJECT_ROOT / "models" / "sartor-pretrained")
CAPS_DIR = str(PROJECT_ROOT / "data" / "pretrain" / "dataset_rsicd.json")
IMGS_DIR = str(PROJECT_ROOT / "data" / "pretrain" / "RSICD_images")


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

    print("Successfully loaded pretrained model and test dataset")
    print("Prediction started ...")

    for i, row in test_df.iterrows():
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

        print(f"{i:05d}  {row['Image Name']}")
        print(f"       TRUE: {row['Caption']}")
        print(f"       PRED: {pred}\n")


if __name__ == "__main__":
    fire.Fire(main)
