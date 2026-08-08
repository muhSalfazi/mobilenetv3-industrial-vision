#!/usr/bin/env python3
"""
Preprocessing Pipeline untuk dataset CCTV TA.

Pipeline: resize -> filter -> (optional: normalize) -> (optional: augment)

Catatan Penting:
- Training notebook menggunakan rescale=1/255 di ImageDataGenerator
- Jadi normalisasi/augmentasi di pipeline ini hanya opsional untuk pre-processing
- Untuk training, gunakan filtered/ atau resize/ folder langsung dengan rescale=1/255
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import RAW_DIR, RESIZE_DIR, FILTERED_DIR, NORMALIZED_DIR, AUGMENTED_DIR, IMG_SIZE, SEED
from resize import resize_images
from filter_denoise import denoise_images
from normalize import normalize_images
from augment import augment_dataset
from utils import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline preprocessing: resize -> filter -> (normalize) -> (augment)",
        epilog="Contoh: python pipeline.py --input /path/to/raw --preview"
    )
    parser.add_argument("--input", type=Path, default=RAW_DIR,
                        help="Input directory dengan raw images (default: data/input)")
    parser.add_argument("--resize-dir", type=Path, default=RESIZE_DIR,
                        help="Output directory untuk resize")
    parser.add_argument("--filtered-dir", type=Path, default=FILTERED_DIR,
                        help="Output directory untuk filter")
    parser.add_argument("--normalized-dir", type=Path, default=NORMALIZED_DIR,
                        help="Output directory untuk normalize (optional)")
    parser.add_argument("--augmented-dir", type=Path, default=AUGMENTED_DIR,
                        help="Output directory untuk augment (optional)")
    
    # Resize parameters
    parser.add_argument("--width", type=int, default=IMG_SIZE[0],
                        help="Target image width")
    parser.add_argument("--height", type=int, default=IMG_SIZE[1],
                        help="Target image height")
    
    # Filter parameters
    parser.add_argument("--filter-method", choices=["nlmeans", "gaussian", "median", "bilateral"],
                        default="nlmeans", help="Filtering method")
    parser.add_argument("--filter-strength", type=int, default=10,
                        help="Filter strength (param h untuk nlmeans)")
    parser.add_argument("--clahe", action="store_true",
                        help="Apply CLAHE for contrast enhancement")
    
    # Normalize parameters (optional)
    parser.add_argument("--norm-mode", choices=["minmax", "zscore", "mobilenetv3"],
                        default="minmax", help="Normalization mode")
    parser.add_argument("--save-norm-image", action="store_true",
                        help="Save normalized images as JPG (besides .npy)")
    
    # Augment parameters (optional)
    parser.add_argument("--augment-per-image", type=int, default=2,
                        help="Number of augmentations per image")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    
    # Skip options
    parser.add_argument("--skip-resize", action="store_true",
                        help="Skip resizing stage")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Skip filtering stage")
    parser.add_argument("--skip-normalize", action="store_true",
                        help="Skip normalization (default: True, recommended for training)")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip augmentation (default: True, prefer on-the-fly in training)")
    
    # Preview options
    parser.add_argument("--preview", action="store_true",
                        help="Generate preview images and statistics")
    parser.add_argument("--preview-count", type=int, default=6,
                        help="Number of preview samples per class")
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image",
                        help="Preview layout style")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input directory tidak ada: {args.input}")
        sys.exit(1)
    
    print("=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Input : {args.input}")
    print(f"Output: {args.filtered_dir}")
    if not args.preview:
        print("💡 Tambahkan --preview untuk melihat statistik visual per tahap")
    print()
    
    # ========== RESIZE ==========
    if not args.skip_resize:
        print("🔄 TAHAP 1: RESIZE")
        print("-" * 70)
        resize_stats = resize_images(
            args.input,
            args.resize_dir,
            (args.width, args.height),
            args.preview,
            args.preview_count,
            args.preview_style,
        )
        write_csv(args.resize_dir / "resize_manifest.csv", resize_stats["rows"])
        print(f"✅ Resize selesai: {resize_stats['total']} images")
        if args.preview:
            print(f"   Preview: {args.resize_dir.parent / 'stats' / 'resize' / 'preview_resize.jpg'}")
        print()
    else:
        print("⏭️  RESIZE: skipped")
        print()
    
    # ========== FILTER ==========
    if not args.skip_filter:
        print("🔄 TAHAP 2: FILTER & DENOISE")
        print("-" * 70)
        filter_input = args.resize_dir if not args.skip_resize else args.input
        filter_stats = denoise_images(
            filter_input,
            args.filtered_dir,
            args.filter_method,
            args.filter_strength,
            args.clahe,
            args.preview,
            args.preview_count,
            args.preview_style,
        )
        write_csv(args.filtered_dir / "filter_manifest.csv", filter_stats["rows"])
        print(f"✅ Filter selesai: {filter_stats['total']} images ")
        print(f"   Method: {args.filter_method} (strength={args.filter_strength})")
        if args.clahe:
            print(f"   CLAHE: enabled")
        if args.preview:
            print(f"   Preview: {args.filtered_dir.parent / 'stats' / 'filter' / 'preview_filter.jpg'}")
        print()
    else:
        print("⏭️  FILTER: skipped")
        print()
    
    # ========== NORMALIZE (optional) ==========
    if not args.skip_normalize:
        print("🔄 TAHAP 3: NORMALIZE")
        print("-" * 70)
        print("⚠️  CATATAN: Training notebook menggunakan rescale=1/255 di ImageDataGenerator")
        print("   Normalisasi .npy ini opsional untuk pre-processing")
        normalize_input_dir = args.filtered_dir if not args.skip_filter else args.resize_dir
        norm_stats = normalize_images(
            normalize_input_dir,
            args.normalized_dir,
            args.norm_mode,
            args.save_norm_image,
            args.preview,
            args.preview_count,
            args.preview_style,
        )
        write_csv(args.normalized_dir / "normalize_manifest.csv", norm_stats["rows"])
        print(f"✅ Normalize selesai: {norm_stats['total']} images")
        print(f"   Mode: {args.norm_mode}")
        if args.preview:
            print(f"   Preview: {args.normalized_dir.parent / 'stats' / 'normalize' / 'preview_normalize.jpg'}")
        print()
    else:
        print("⏭️  NORMALIZE: skipped (recommended for training)")
        print()
    
    # ========== AUGMENT (optional) ==========
    if not args.skip_augment:
        print("🔄 TAHAP 4: AUGMENT")
        print("-" * 70)
        print("⚠️  CATATAN: Training notebook melakukan augmentasi on-the-fly dengan ImageDataGenerator")
        print("   Augmentasi .jpg ini opsional untuk pre-processing")
        augment_input = args.filtered_dir if not args.skip_filter else args.resize_dir
        aug_stats = augment_dataset(
            augment_input,
            args.augmented_dir,
            args.augment_per_image,
            args.seed,
            preview=args.preview,
            preview_count=args.preview_count,
            preview_style=args.preview_style,
        )
        write_csv(args.augmented_dir / "augment_manifest.csv", aug_stats["rows"])
        print(f"✅ Augment selesai: {aug_stats['total']} images")
        print(f"   Per image: {args.augment_per_image} augmentations")
        if args.preview:
            print(f"   Preview: {args.augmented_dir.parent / 'stats' / 'augment' / 'preview_augment.jpg'}")
        print()
    else:
        print("⏭️  AUGMENT: skipped (recommended for training)")
        print()
    
    # ========== SUMMARY ==========
    print("=" * 70)
    print("📌 SUMMARY")
    print("=" * 70)
    final_output = args.augmented_dir if not args.skip_augment else (
        args.normalized_dir if not args.skip_normalize else args.filtered_dir
    )
    print(f"✅ Pipeline selesai!")
    print(f"Output folder: {final_output}")
    print()
    print("🚀 Siap untuk training dengan notebook:")
    print(f"   BASE_PATH = '{final_output}'")
    print(f"   train_datagen = ImageDataGenerator(rescale=1./255, ...augmentation...)")
    print(f"   train_gen = train_datagen.flow_from_directory(f'{{BASE_PATH}}/train', ...)")
    print()


if __name__ == "__main__":
    main()
    main()
