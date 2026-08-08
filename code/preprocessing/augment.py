#!/usr/bin/env python3
"""
Data augmentation menggunakan Keras ImageDataGenerator.
Matches training notebook approach dengan rescale=1/255 dan spatial+color augmentasi.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import time

import numpy as np

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    HAS_TF = True
except ImportError:
    HAS_TF = False

from config import AUGMENTED_DIR, FILTERED_DIR, CLASS_NAMES, SEED, STATS_DIR
from utils import class_from_path, ensure_dir, iter_images, read_image, save_preview_grid, write_csv, write_image, write_json


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augment_per_image: int,
    seed: int,
    preview: bool = False,
    preview_count: int = 6,
    preview_style: str = "image",
) -> dict:
    """
    Generate augmented images menggunakan ImageDataGenerator.
    Setiap input image akan di-augment sebanyak augment_per_image kali.
    """
    if not HAS_TF:
        print("ERROR: TensorFlow tidak tersedia. Install: pip install tensorflow")
        return {"total": 0, "skipped": 0, "seconds": 0, "rows": []}
    
    ensure_dir(output_dir)
    augment_stats_dir = STATS_DIR / "augment"
    ensure_dir(augment_stats_dir)
    
    image_paths = list(iter_images(input_dir))
    
    if not image_paths:
        print(f"  ⚠️  Tidak ada images di: {input_dir}")
        return {"total": 0, "skipped": 0, "seconds": 0, "rows": []}
    
    # ======== SETUP AUGMENTATION (MATCH TRAINING NOTEBOOK) ========
    # rescale=1/255 untuk normalize ke [0, 1]
    # Spatial augmentation: rotation, shift, shear, zoom
    # Color augmentation: brightness, channel_shift
    # horizontal_flip untuk augmentation
    augmentation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        # Spatial  augmentation
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        # Color augmentation
        brightness_range=(0.75, 1.25),
        channel_shift_range=20.0,
        fill_mode="nearest",
    )
    
    # ======== PROCESS IMAGES ========
    rows = []
    total = 0
    skipped = 0
    t0 = time()
    preview_pairs = []
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}
    aug_index = 0
    
    for img_path in image_paths:
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue
        
        # ======== Read original image ========
        img_array = img.astype(np.float32)
        # Add batch dimension untuk ImageDataGenerator (format: [1, H, W, 3])
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Maintain class folder structure
        rel = img_path.relative_to(input_dir)
        class_name = class_from_path(input_dir, img_path)
        out_class_dir = output_dir / class_name
        ensure_dir(out_class_dir)
        
        # Get original filename without extension
        original_stem = rel.stem
        
        # ======== GENERATE AUGMENTED VERSIONS ========
        aug_count = 0
        for aug_data in augmentation_datagen.flow(
            img_batch,
            batch_size=1,
            seed=seed,
            shuffle=False,
        ):
            if aug_count >= augment_per_image:
                break
            
            # aug_data is already rescaled to [0, 1] by rescale=1/255
            augmented = aug_data[0]  # Get first (only) image from batch
            
            # Convert back to [0, 255] uint8 untuk save sebagai JPG
            augmented_uint8 = np.clip(augmented * 255.0, 0, 255).astype(np.uint8)
            
            # Generate output filename: original_aug{counter}.jpg
            out_name = f"{original_stem}_aug{aug_count}.jpg"
            out_path = out_class_dir / out_name
            
            write_image(out_path, augmented_uint8)
            
            # ======== COLLECT PREVIEW ========
            if preview:
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                if class_counts[class_name] < per_class and len(preview_pairs) < preview_count:
                    # Store as float [0, 1] untuk preview
                    preview_pairs.append((
                        img_array / 255.0,
                        augmented,  # Already in [0, 1]
                        f"{class_name}/{aug_count}"
                    ))
                    class_counts[class_name] += 1
            
            rows.append({
                "original": str(img_path),
                "output": str(out_path),
                "class": class_name,
                "augmentation_index": aug_count,
            })
            
            aug_count += 1
            total += 1
            aug_index += 1
    
    elapsed = time() - t0
    
    # Save preview
    if preview and preview_pairs:
        save_preview_grid(
            preview_pairs,
            augment_stats_dir / "preview_augment.jpg",
            matrix_view=(preview_style == "matrix"),
        )
    
    # Save augmentation config
    aug_config = {
        "augment_per_image": augment_per_image,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_augmented": total,
        "seed": seed,
        "augmentation_params": {
            "rescale": "1./255",
            "rotation_range": 20,
            "width_shift_range": 0.15,
            "height_shift_range": 0.15,
            "shear_range": 0.10,
            "zoom_range": 0.20,
            "horizontal_flip": True,
            "brightness_range": [0.75, 1.25],
            "channel_shift_range": 20.0,
            "fill_mode": "nearest",
        },
    }
    write_json(augment_stats_dir / "augmentation_config.json", aug_config)
    
    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment dataset menggunakan ImageDataGenerator")
    parser.add_argument("--input", type=Path, default=FILTERED_DIR)
    parser.add_argument("--output", type=Path, default=AUGMENTED_DIR)
    parser.add_argument("--augment-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    args = parser.parse_args()
    
    stats = augment_dataset(
        args.input,
        args.output,
        args.augment_per_image,
        args.seed,
        args.preview,
        args.preview_count,
        args.preview_style,
    )
    
    print(f"\n✅ Augmentasi selesai!")
    print(f"  Total augmented: {stats['total']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Time: {stats['seconds']}s")


if __name__ == "__main__":
    main()
