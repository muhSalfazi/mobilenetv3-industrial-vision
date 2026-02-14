#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import time

import cv2
import numpy as np

from config import FILTERED_DIR, RESIZE_DIR, STATS_DIR, JSON_STATS_DIR
from utils import class_from_path, ensure_dir, iter_images, read_image, save_preview_grid, write_csv, write_image, write_json


def _odd_kernel(strength: int) -> int:
    k = max(3, int(strength))
    if k % 2 == 0:
        k += 1
    return min(k, 31)


def apply_denoise(img: np.ndarray, method: str, strength: int, clahe: bool) -> np.ndarray:
    if method == "gaussian":
        k = _odd_kernel(max(3, strength))
        out = cv2.GaussianBlur(img, (k, k), 0)
    elif method == "median":
        k = _odd_kernel(max(3, strength))
        out = cv2.medianBlur(img, k)
    elif method == "bilateral":
        d = max(5, min(25, int(strength)))
        sigma = max(20, int(strength * 5))
        out = cv2.bilateralFilter(img, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    elif method == "nlmeans":
        h = max(3, int(strength))
        out = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    else:
        raise SystemExit(f"Method tidak dikenal: {method}")

    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_fn = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe_fn.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    return out


def save_denoise_proof_charts(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    out_dir: Path,
    method: str,
    clahe: bool,
) -> dict:
    ensure_dir(out_dir)
    if not pairs:
        return {}

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "Peringatan: chart filter tidak dibuat karena matplotlib belum tersedia "
            f"({exc.__class__.__name__})."
        )
        return {}
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        sns = None
        print("Peringatan: seaborn belum tersedia, memakai fallback chart matplotlib untuk filter.")

    before_int = []
    after_int = []
    diff_maps = []
    mse_values = []
    psnr_values = []
    before_noise_std = []
    after_noise_std = []
    reduction_percent = []
    labels = []

    rng = np.random.default_rng(42)
    sample_pixels = 25000

    for i, (before, after, label) in enumerate(pairs, start=1):
        b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
        a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(b - a)

        mse = float(np.mean((b - a) ** 2))
        psnr = float("inf") if mse < 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        b_noise = b - cv2.GaussianBlur(b, (5, 5), 0)
        a_noise = a - cv2.GaussianBlur(a, (5, 5), 0)
        b_std = float(np.std(b_noise))
        a_std = float(np.std(a_noise))
        red = ((b_std - a_std) / max(b_std, 1e-8)) * 100.0

        mse_values.append(mse)
        psnr_values.append(psnr)
        before_noise_std.append(b_std)
        after_noise_std.append(a_std)
        reduction_percent.append(red)
        labels.append(f"{i}-{label}")
        diff_maps.append(diff)

        bf = b.flatten()
        af = a.flatten()
        if bf.size > sample_pixels:
            idx = rng.choice(bf.size, sample_pixels, replace=False)
            bf = bf[idx]
        if af.size > sample_pixels:
            idx = rng.choice(af.size, sample_pixels, replace=False)
            af = af[idx]
        before_int.append(bf)
        after_int.append(af)

    before_int = np.concatenate(before_int)
    after_int = np.concatenate(after_int)
    mean_diff = np.mean(np.stack(diff_maps, axis=0), axis=0)

    subtitle = f"Method: {method.upper()} | CLAHE: {'ON' if clahe else 'OFF'}"

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.histplot(before_int, bins=80, stat="density", color="#1f77b4", alpha=0.55, ax=ax)
    else:
        ax.hist(before_int, bins=80, density=True, color="#1f77b4", alpha=0.55)
    ax.set_title("Histogram Pixel Intensity (Before)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.text(0.98, 0.95, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_histogram_before.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.histplot(after_int, bins=80, stat="density", color="#2ca02c", alpha=0.55, ax=ax)
    else:
        ax.hist(after_int, bins=80, density=True, color="#2ca02c", alpha=0.55)
    ax.set_title("Histogram Pixel Intensity (After)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.text(0.98, 0.95, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_histogram_after.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.6, 5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mean_diff, cmap="inferno")
    ax.set_title("Difference Map (Original - Denoised)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_difference_map.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.6, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(mse_values) + 1), mse_values, marker="o", color="#d62728")
    ax.set_title("MSE Graph per Sample")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("MSE")
    if len(labels) <= 12:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_mse_graph.jpg"), dpi=220)
    plt.close(fig)

    psnr_plot = [v if np.isfinite(v) else 100.0 for v in psnr_values]
    fig = plt.figure(figsize=(7.6, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(psnr_plot) + 1), psnr_plot, marker="o", color="#1f77b4")
    ax.set_title("PSNR Graph per Sample")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("PSNR (dB)")
    if len(labels) <= 12:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.text(
        0.98,
        0.93,
        "Inf dipotong ke 100 dB untuk visualisasi",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_psnr_graph.jpg"), dpi=220)
    plt.close(fig)

    x = np.arange(1, len(before_noise_std) + 1)
    mean_before = float(np.mean(before_noise_std))
    mean_after = float(np.mean(after_noise_std))
    mean_reduction = float(np.mean(reduction_percent))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(x, before_noise_std, marker="o", color="#d62728", linewidth=2.0, label="Noise STD (Before)")
    ax.axhline(mean_before, color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean: {mean_before:.3f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Noise STD")
    ax.set_title("Noise Level Before Filtering")
    ax.grid(alpha=0.25)
    if len(labels) <= 12:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.text(0.98, 0.95, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#444444")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_noise_before_graph.jpg"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(x, after_noise_std, marker="o", color="#2ca02c", linewidth=2.0, label="Noise STD (After)")
    ax.axhline(mean_after, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_after:.3f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Noise STD")
    ax.set_title("Noise Level After Filtering")
    ax.grid(alpha=0.25)
    if len(labels) <= 12:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.text(0.98, 0.95, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#444444")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_noise_after_graph.jpg"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x, reduction_percent, color="#1f77b4", alpha=0.8, label="Reduction (%)")
    ax.axhline(mean_reduction, color="#0d3b66", linestyle="--", linewidth=1.5, label=f"Mean: {mean_reduction:.2f}%")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Noise Reduction (%)")
    ax.set_title("Noise Reduction After Filtering")
    ax.grid(axis="y", alpha=0.25)
    if len(labels) <= 12:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.text(0.98, 0.95, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#444444")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(out_dir / "denoise_noise_reduction_graph.jpg"), dpi=220)
    plt.close(fig)

    stats = {
        "method": method,
        "clahe": clahe,
        "samples": len(pairs),
        "mse_mean": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "psnr_mean": float(np.mean(psnr_plot)),
        "psnr_std": float(np.std(psnr_plot)),
        "noise_std_before_mean": float(np.mean(before_noise_std)),
        "noise_std_after_mean": float(np.mean(after_noise_std)),
        "noise_reduction_percent_mean": float(np.mean(reduction_percent)),
    }
    json_stats_dir = JSON_STATS_DIR / "filter"
    ensure_dir(json_stats_dir)
    write_json(json_stats_dir / "denoise_metrics.json", stats)
    return stats


def denoise_images(
    input_dir: Path,
    output_dir: Path,
    method: str,
    strength: int,
    clahe: bool,
    preview: bool,
    preview_count: int,
    preview_style: str,
) -> dict:
    ensure_dir(output_dir)
    filter_stats_dir = STATS_DIR / "filter"
    ensure_dir(filter_stats_dir)

    rows = []
    total = 0
    skipped = 0
    t0 = time()
    preview_pairs = []
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}

    for img_path in iter_images(input_dir):
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue
        denoised = apply_denoise(img, method, strength, clahe)
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".jpg")
        write_image(out_path, denoised)
        rows.append({
            "input": str(img_path),
            "output": str(out_path),
            "method": method,
            "strength": int(strength),
            "clahe": bool(clahe),
        })
        if preview:
            label = class_from_path(input_dir, img_path)
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < per_class and len(preview_pairs) < preview_count:
                preview_pairs.append((img, denoised, label))
                class_counts[label] += 1
        total += 1

    elapsed = time() - t0
    if preview and preview_pairs:
        save_preview_grid(
            preview_pairs,
            filter_stats_dir / "preview_filter.jpg",
            matrix_view=(preview_style == "matrix"),
        )
        save_denoise_proof_charts(
            preview_pairs,
            filter_stats_dir,
            method,
            clahe,
        )

    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter denoise images (Gaussian/NLMeans/Median/Bilateral + CLAHE)")
    parser.add_argument("--input", type=Path, default=RESIZE_DIR)
    parser.add_argument("--output", type=Path, default=FILTERED_DIR)
    parser.add_argument("--method", choices=["nlmeans", "gaussian", "median", "bilateral"], default="gaussian")
    parser.add_argument("--strength", type=int, default=5)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=8)
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
    write_csv(args.output / "filter_manifest.csv", stats["rows"])
    print(f"Filter selesai. Total: {stats['total']}, Skipped: {stats['skipped']}, Time: {stats['seconds']}s")
    if not args.preview:
        print("Catatan: statistik visual filter dibuat saat menjalankan command dengan flag `--preview`.")


if __name__ == "__main__":
    main()
