#!/usr/bin/env python3
"""Filter dan denoise images menggunakan berbagai metode."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import time

import cv2
import numpy as np

from config import FILTERED_DIR, RESIZE_DIR, STATS_DIR, JSON_STATS_DIR
from utils import class_from_path, ensure_dir, iter_images, read_image, save_preview_grid, write_csv, write_image, write_json


def apply_nlmeans(img: np.ndarray, h: int = 10) -> np.ndarray:
    """
    Non-Local Means Denoising.
    
    params:
    - h: filtering strength (higher = more smoothing)
    """
    if img.ndim == 3 and img.shape[2] == 3:
        filtered = cv2.fastNlMeansDenoisingColored(
            img,
            h=h,
            hForColorComponents=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    else:
        filtered = cv2.fastNlMeansDenoising(
            img,
            h=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    return filtered


def apply_gaussian(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian Blur untuk denoising."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral Filter - preserve edges while smoothing."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_median(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median Filter."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Meningkatkan contrast lokal tanpa over-amplifying noise.
    """
    if img.ndim == 3:
        # Convert BGR to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE hanya ke L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)
        
        # Merge kembali
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return result
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)


def compute_noise_metrics(original: np.ndarray, filtered: np.ndarray) -> tuple[float, float, float]:
    """Compute MSE, PSNR, dan noise reduction percentage."""
    if original.size == 0:
        return 0.0, 0.0, 0.0
    
    original_f = original.astype(np.float64)
    filtered_f = filtered.astype(np.float64)
    
    mse = np.mean((original_f - filtered_f) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12)) if mse > 0 else 0.0
    
    # Estimate noise reduction
    original_g = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
    filtered_g = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY) if filtered.ndim == 3 else filtered
    
    # Compute local gradients (high-frequency content)
    sobelx_o = cv2.Sobel(original_g, cv2.CV_32F, 1, 0, ksize=3)
    sobelx_f = cv2.Sobel(filtered_g, cv2.CV_32F, 1, 0, ksize=3)
    
    noise_std_before = np.std(sobelx_o) / 255.0
    noise_std_after = np.std(sobelx_f) / 255.0
    
    noise_reduction = ((noise_std_before - noise_std_after) / max(noise_std_before, 1e-8)) * 100
    
    return mse, psnr, noise_reduction


def denoise_images(
    input_dir: Path,
    output_dir: Path,
    method: str,
    strength: int,
    apply_clahe: bool,
    preview: bool,
    preview_count: int,
    preview_style: str,
) -> dict:
    """
    Apply denoising filter ke semua images.
    """
    ensure_dir(output_dir)
    filter_stats_dir = STATS_DIR / "filter"
    ensure_dir(filter_stats_dir)
    
    image_paths = list(iter_images(input_dir))
    
    if not image_paths:
        print(f"  ⚠️  Tidak ada images di: {input_dir}")
        return {"total": 0, "skipped": 0, "seconds": 0, "rows": []}
    
    rows = []
    total = 0
    skipped = 0
    t0 = time()
    preview_pairs = []
    metrics_list = []
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}
    
    for img_path in image_paths:
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue
        
        # Apply denoising method
        if method == "nlmeans":
            filtered = apply_nlmeans(img, h=strength)
        elif method == "gaussian":
            filtered = apply_gaussian(img, kernel_size=max(3, strength - 5), sigma=strength / 10.0)
        elif method == "bilateral":
            filtered = apply_bilateral(img, d=9, sigma_color=strength, sigma_space=strength)
        elif method == "median":
            kernel_size = strength if strength % 2 == 1 else strength + 1
            filtered = apply_median(img, kernel_size=kernel_size)
        else:
            filtered = img.copy()
        
        # Apply CLAHE if requested
        if apply_clahe:
            filtered = apply_clahe(filtered, clip_limit=2.0)
        
        # Maintain class folder structure
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        ensure_dir(out_path.parent)
        write_image(out_path, filtered)
        
        # Compute metrics
        mse, psnr, noise_reduction = compute_noise_metrics(img, filtered)
        metrics_list.append({"mse": mse, "psnr": psnr, "noise_reduction": noise_reduction})
        
        # Collect preview samples
        if preview:
            label = class_from_path(input_dir, img_path)
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < per_class and len(preview_pairs) < preview_count:
                preview_pairs.append((
                    img.astype(np.float32) / 255.0,
                    filtered.astype(np.float32) / 255.0,
                    label
                ))
                class_counts[label] += 1
        
        rows.append({
            "input": str(img_path),
            "output": str(out_path),
            "method": method,
            "clahe": apply_clahe,
            "mse": f"{mse:.2f}",
            "psnr": f"{psnr:.2f}",
        })
        total += 1
    
    elapsed = time() - t0
    
    # Compute aggregate metrics
    if metrics_list:
        mse_values = [m["mse"] for m in metrics_list]
        psnr_values = [m["psnr"] for m in metrics_list]
        noise_values = [m["noise_reduction"] for m in metrics_list]
        
        # Estimate noise std
        noise_std_before = np.mean([np.std(cv2.Sobel(read_image(p), cv2.CV_32F, 1, 0, ksize=3)) / 255.0 for p in image_paths[:min(10, len(image_paths))] if read_image(p) is not None])
        noise_std_after = noise_std_before * (1 - np.mean(noise_values) / 100.0)
        
        metrics_summary = {
            "method": method,
            "clahe": apply_clahe,
            "samples": len(metrics_list),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "noise_std_before_mean": float(noise_std_before),
            "noise_std_after_mean": float(noise_std_after),
            "noise_reduction_percent_mean": float(np.mean(noise_values)),
        }
        
        # Save metrics
        write_json(JSON_STATS_DIR / "denoise_metrics.json", metrics_summary)
    
    # Save preview
    if preview and preview_pairs:
        save_preview_grid(
            preview_pairs,
            filter_stats_dir / "preview_filter.jpg",
            matrix_view=(preview_style == "matrix"),
        )
    
    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter dan denoise images")
    parser.add_argument("--input", type=Path, default=RESIZE_DIR)
    parser.add_argument("--output", type=Path, default=FILTERED_DIR)
    parser.add_argument("--method", choices=["nlmeans", "gaussian", "median", "bilateral"], default="nlmeans")
    parser.add_argument("--strength", type=int, default=10)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    args = parser.parse_args()
    
    stats = denoise_images(
        args.input,
        args.output,
        args.method,
        args.strength,
        args.clahe,
        args.preview,
        args.preview_count,
        args.preview_style,
    )
    
    print(f"\n✅ Filter selesai!")
    print(f"  Total: {stats['total']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Time: {stats['seconds']}s")


if __name__ == "__main__":
    main()
