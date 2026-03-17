from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_PATH = str(PROJECT_ROOT / "models" / "sartor-finetuned")
CAPS_DIR = str(PROJECT_ROOT / "data" / "fine_tune" / "Text" / "Caption" / "test" / "Caption_test.csv")
IMGS_DIR = str(PROJECT_ROOT / "data" / "fine_tune" / "SARimages_preprocessed" / "SARimages")
