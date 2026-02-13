#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from config import RAW_DIR, RESIZE_DIR, FILTERED_DIR, NORMALIZED_DIR, AUGMENTED_DIR, IMG_SIZE
from resize import resize_images
from filter_denoise import denoise_images
from normalize import normalize_images
from augment import augment_dataset
from utils import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline preprocessing TA: resize -> filter -> normalize -> augment")
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument("--resize-dir", type=Path, default=RESIZE_DIR)
    parser.add_argument("--filtered-dir", type=Path, default=FILTERED_DIR)
    parser.add_argument("--normalized-dir", type=Path, default=NORMALIZED_DIR)
    parser.add_argument("--augmented-dir", type=Path, default=AUGMENTED_DIR)
    parser.add_argument("--width", type=int, default=IMG_SIZE[0])
    parser.add_argument("--height", type=int, default=IMG_SIZE[1])
    parser.add_argument("--filter-method", choices=["nlmeans", "gaussian", "median", "bilateral"], default="nlmeans")
    parser.add_argument("--filter-strength", type=int, default=10)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--norm-mode", choices=["minmax", "zscore", "mobilenetv3"], default="minmax")
    parser.add_argument("--save-norm-image", action="store_true")
    parser.add_argument("--augment-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-resize", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    parser.add_argument("--demo-rank", type=int, default=1)
    parser.add_argument("--demo-image", type=str, default=None)
    args = parser.parse_args()

    if not args.skip_resize:
        resize_stats = resize_images(
            args.input,
            args.resize_dir,
            (args.width, args.height),
            args.preview,
            args.preview_count,
            args.preview_style,
            args.demo_rank,
            args.demo_image,
        )
        write_csv(args.resize_dir / "resize_manifest.csv", resize_stats["rows"])
        print(f"Resize selesai: {resize_stats['total']} images")

    if not args.skip_filter:
        filter_stats = denoise_images(
            args.resize_dir,
            args.filtered_dir,
            args.filter_method,
            args.filter_strength,
            args.clahe,
            args.preview,
            args.preview_count,
            args.preview_style,
        )
        write_csv(args.filtered_dir / "filter_manifest.csv", filter_stats["rows"])
        print(f"Filter selesai: {filter_stats['total']} images")

    if not args.skip_normalize:
        norm_stats = normalize_images(
            args.resize_dir,
            args.normalized_dir,
            args.norm_mode,
            args.save_norm_image,
            args.preview,
            args.preview_count,
            args.preview_style,
        )
        write_csv(args.normalized_dir / "normalize_manifest.csv", norm_stats["rows"])
        print(f"Normalisasi selesai: {norm_stats['total']} images")

    if not args.skip_augment:
        aug_stats = augment_dataset(
            args.filtered_dir,
            args.augmented_dir,
            args.augment_per_image,
            args.seed,
            preview=args.preview,
            preview_count=args.preview_count,
            preview_style=args.preview_style,
        )
        write_csv(args.augmented_dir / "augment_manifest.csv", aug_stats["rows"])
        print(f"Augmentasi selesai: {aug_stats['total']} images")


if __name__ == "__main__":
    main()
