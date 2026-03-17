import fire
import torch
import pandas as pd
from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor
from sartor.modules.constants import MODELS_PATH, CAPS_DIR, IMGS_DIR
from sartor.modules.generate import generate

def main(max_new_tokens: int = 64, num_beams: int = 4) -> None:
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    test_df = pd.read_csv(CAPS_DIR)
    test_df.columns = [col.strip() for col in test_df.columns]
    tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH)

    model = VisionEncoderDecoderModel.from_pretrained(MODELS_PATH).to(device).eval()
    print("Successfully loaded fine-tuned model and test dataset")

    processor = AutoImageProcessor.from_pretrained(MODELS_PATH)

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
            num_beams=num_beams
            )

        print(f"{i:05d}  {row['Image Name']}")
        print(f"       TRUE: {row['Caption']}")
        print(f"       PRED: {pred}\n")


if __name__ == "__main__":
    fire.Fire(main)