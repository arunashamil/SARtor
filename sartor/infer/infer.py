import fire
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor, Seq2SeqTrainer, default_data_collator
from sartor.modules.constants import MODELS_PATH, CAPS_DIR, IMGS_DIR, IMG_SIZE, ENCODER
from sartor.modules.dataset import ImgDataset
from sartor.modules.compute_metrics import compute_metrics
from torchvision import transforms

def main(checkpoint_name: str) -> None:
    if torch.cuda.is_available():    

        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    test_df = pd.read_csv(CAPS_DIR)
    test_df.columns = [col.strip() for col in test_df.columns]

    tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_PATH}/{checkpoint_name}")
    feature_extractor = ViTImageProcessor.from_pretrained(ENCODER)
    model = VisionEncoderDecoderModel.from_pretrained(f"{MODELS_PATH}/{checkpoint_name}")
    testing_args = torch.load(f"{MODELS_PATH}/{checkpoint_name}/{"training_args.bin"}", weights_only=False)

    transformations = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0.5,
                std=0.5
            )
        ]
    )

    test_dataset = ImgDataset(
        test_df, 
        root_dir=IMGS_DIR,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        transform=transformations
        )
    
    print("Successfully loaded fine-tuned model and test dataset")
    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Seq2SeqTrainer(
        processing_class=tokenizer,
        model=model,
        args=testing_args,
        data_collator=default_data_collator
    )

    print("Prediction started ...")
    test_predictions = trainer.predict(test_dataset=test_dataset)

    test_results = trainer.predict(test_dataset)
    predictions = test_results.predictions

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = test_results.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for i, (pred, label) in enumerate(zip(decoded_predictions, decoded_labels)):
        print(f"Example {i+1}:")
        print(f"  Prediction: {pred}")
        print(f"  Label: {label}")
        print()


if __name__ == "__main__":
    fire.Fire(main)