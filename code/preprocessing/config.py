#!/usr/bin/env python3
"""Configuration file untuk preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path

# Direktori utama
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Direktori input (raw images dari video)
RAW_DIR = DATA_DIR / "input"

# Direktori output untuk setiap stage preprocessing
RESIZE_DIR = DATA_DIR / "output" / "resize"
FILTERED_DIR = DATA_DIR / "output" / "filtered"
NORMALIZED_DIR = DATA_DIR / "output" / "normalized"
AUGMENTED_DIR = DATA_DIR / "output" / "augmented"

# Direktori untuk statistik dan visualisasi
STATS_DIR = DATA_DIR / "output" / "stats"
JSON_STATS_DIR = STATS_DIR / "cctv_specs"

# Konfigurasi image
IMG_SIZE = (224, 224)  # MobileNetV3Large memerlukan 224x224
NUM_CLASSES = 3
CLASS_NAMES = ["bekerja", "idle", "meninggalkan_area"]

# Seed untuk reproducibility
SEED = 42

# Dataset classification (untuk training)
DATASET_CLASSIFICATION_DIR = DATASET_DIR / "dataset_classification"
TRAIN_DIR = DATASET_CLASSIFICATION_DIR / "train"
VALID_DIR = DATASET_CLASSIFICATION_DIR / "valid"
TEST_DIR = DATASET_CLASSIFICATION_DIR / "test"

# Model paths
MODEL_DIR = DATASET_DIR / "model"
BEST_MODEL_PATH = MODEL_DIR / "mobilenet_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "mobilenetv3_final_finetuned.keras"

# Batch size untuk training
BATCH_SIZE = 32

# Konfigurasi preprocessing stages
PREPROCESS_STAGES = {
    "resize": {"enabled": True, "size": IMG_SIZE},
    "filter": {"enabled": True, "method": "nlmeans", "strength": 10},
    "normalize": {"enabled": False, "mode": "minmax"},  # Di-skip karena rescale=1/255 handle di training
    "augment": {"enabled": True, "per_image": 2},
}
