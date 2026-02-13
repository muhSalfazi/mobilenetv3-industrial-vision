from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
JSON_STATS_DIR = INPUT_DIR

RAW_DIR = BASE_DIR / "dataset" / "dataset_classification" / "train"
RESIZE_DIR = OUTPUT_DIR / "resize"
FILTERED_DIR = OUTPUT_DIR / "filtered"
NORMALIZED_DIR = OUTPUT_DIR / "normalized"
AUGMENTED_DIR = OUTPUT_DIR / "augmented"
STATS_DIR = OUTPUT_DIR / "stats"

IMG_SIZE = (224, 224)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
