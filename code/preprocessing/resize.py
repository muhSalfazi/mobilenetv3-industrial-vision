#!/usr/bin/env python3
"""Resize images ke target size untuk training model."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import time

import cv2
import numpy as np

from config import RESIZE_DIR, IMG_SIZE, STATS_DIR
from utils import class_from_path, ensure_dir, iter_images, read_image, save_preview_grid, write_csv, write_image, write_json


def resize_images(
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int],
    preview: bool,
    preview_count: int,
    preview_style: str,
    demo_rank: int = 1,
    demo_image: str | None = None,
) -> dict:
    """
    Resize all images ke target size menggunakan cv2.resize dengan interpolasi linear.
    Simpan ke output_dir dengan struktur class yang sama.
    """
    ensure_dir(output_dir)
    resize_stats_dir = STATS_DIR / "resize"
    ensure_dir(resize_stats_dir)
    
    image_paths = list(iter_images(input_dir))
    
    if not image_paths:
        print(f"  ⚠️  Tidak ada images di: {input_dir}")
        return {"total": 0, "skipped": 0, "seconds": 0, "rows": []}
    
    rows = []
    total = 0
    skipped = 0
    t0 = time()
    preview_pairs = []
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}
    
    for img_path in image_paths:
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue
        
        # Resize dengan cv2.INTER_LINEAR (default interpolation)
        resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        
        # Maintain class folder structure
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        ensure_dir(out_path.parent)
        write_image(out_path, resized)
        
        # Collect preview samples
        if preview:
            label = class_from_path(input_dir, img_path)
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < per_class and len(preview_pairs) < preview_count:
                # Convert to float [0, 1] untuk preview
                preview_pairs.append((
                    img.astype(np.float32) / 255.0,
                    resized.astype(np.float32) / 255.0,
                    label
                ))
                class_counts[label] += 1
        
        rows.append({
            "input": str(img_path),
            "output": str(out_path),
            "original_size": f"{img.shape[1]}x{img.shape[0]}",
            "resized_size": f"{resized.shape[1]}x{resized.shape[0]}",
        })
        total += 1
    
    elapsed = time() - t0
    
    # Save preview
    if preview and preview_pairs:
        save_preview_grid(
            preview_pairs,
            resize_stats_dir / "preview_resize.jpg",
            matrix_view=(preview_style == "matrix"),
        )
    
    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize images untuk training")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RESIZE_DIR)
    parser.add_argument("--width", type=int, default=IMG_SIZE[0])
    parser.add_argument("--height", type=int, default=IMG_SIZE[1])
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    args = parser.parse_args()
    
    if args.input is None:
        print("ERROR: --input diperlukan")
        return
    
    stats = resize_images(
        args.input,
        args.output,
        (args.width, args.height),
        args.preview,
        args.preview_count,
        args.preview_style,
    )
    
    print(f"\n✅ Resize selesai!")
    print(f"  Total: {stats['total']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Time: {stats['seconds']}s")


if __name__ == "__main__":
    main()
