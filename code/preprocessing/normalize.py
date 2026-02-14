#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import time

import cv2
import numpy as np

from config import NORMALIZED_DIR, RESIZE_DIR, STATS_DIR, JSON_STATS_DIR
from utils import class_from_path, ensure_dir, iter_images, read_image, save_preview_grid, write_csv, write_image, write_json


def compute_mean_std(image_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)

    for p in image_paths:
        img = read_image(p)
        if img is None:
            continue
        img = img.astype(np.float64) / 255.0
        count += img.shape[0] * img.shape[1]
        channel_sum += img.reshape(-1, 3).sum(axis=0)
        channel_sumsq += (img.reshape(-1, 3) ** 2).sum(axis=0)

    mean = channel_sum / max(count, 1)
    var = (channel_sumsq / max(count, 1)) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


def to_unit_range(normalized: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minmax":
        unit = np.clip(normalized, 0.0, 1.0)
    elif mode == "mobilenetv3":
        unit = np.clip((normalized + 1.0) / 2.0, 0.0, 1.0)
    else:
        unit = np.clip((normalized + 3.0) / 6.0, 0.0, 1.0)
    return unit


def save_academic_normalization_charts(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    out_dir: Path,
    title_prefix: str = "Normalisasi",
) -> None:
    if not pairs:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "Peringatan: chart normalisasi tidak dibuat karena matplotlib belum tersedia "
            f"({exc.__class__.__name__})."
        )
        return
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        sns = None
        print("Peringatan: seaborn belum tersedia, memakai fallback chart matplotlib untuk normalisasi.")

    before_intensity_pixels = []
    after_intensity_pixels = []
    diff_maps = []
    before_frames = []
    after_frames = []
    mse_values = []
    psnr_values = []
    labels = []

    rng = np.random.default_rng(42)
    sample_pixels = 20000

    for idx, (before_unit, after_unit, label) in enumerate(pairs, start=1):
        before_u8 = np.clip(before_unit * 255.0, 0, 255).astype(np.uint8)
        after_u8 = np.clip(after_unit * 255.0, 0, 255).astype(np.uint8)

        if before_u8.ndim == 3:
            before_gray = cv2.cvtColor(before_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            before_gray = before_u8.astype(np.float32) / 255.0
        if after_u8.ndim == 3:
            after_gray = cv2.cvtColor(after_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            after_gray = after_u8.astype(np.float32) / 255.0

        diff = np.abs(before_gray - after_gray)
        mse = float(np.mean((before_gray - after_gray) ** 2))
        psnr = float("inf") if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))
        diff_maps.append(diff)
        before_frames.append(before_gray)
        after_frames.append(after_gray)
        mse_values.append(mse)
        psnr_values.append(psnr)
        labels.append(f"{idx}-{label}")

        bpx = before_gray.flatten()
        apx = after_gray.flatten()
        if bpx.size > sample_pixels:
            idx = rng.choice(bpx.size, sample_pixels, replace=False)
            bpx = bpx[idx]
        if apx.size > sample_pixels:
            idx = rng.choice(apx.size, sample_pixels, replace=False)
            apx = apx[idx]
        before_intensity_pixels.append(bpx.astype(np.float32))
        after_intensity_pixels.append(apx.astype(np.float32))

    if not diff_maps:
        return

    before_intensity_pixels = np.concatenate(before_intensity_pixels)
    after_intensity_pixels = np.concatenate(after_intensity_pixels)
    mean_before_map = np.mean(np.stack(before_frames, axis=0), axis=0)
    mean_after_map = np.mean(np.stack(after_frames, axis=0), axis=0)
    mean_diff_map = np.mean(np.stack(diff_maps, axis=0), axis=0)

    fig = plt.figure(figsize=(6.8, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.histplot(before_intensity_pixels, bins=80, stat="density", color="#1f77b4", alpha=0.55, ax=ax)
    else:
        ax.hist(before_intensity_pixels, bins=80, density=True, color="#1f77b4", alpha=0.55)
    ax.set_title(f"{title_prefix} - Histogram Intensitas (Before)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_histogram_before.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.8, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.histplot(after_intensity_pixels, bins=80, stat="density", color="#2ca02c", alpha=0.55, ax=ax)
    else:
        ax.hist(after_intensity_pixels, bins=80, density=True, color="#2ca02c", alpha=0.55)
    ax.set_title(f"{title_prefix} - Histogram Intensitas (After)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_histogram_after.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mean_before_map, cmap="viridis")
    ax.set_title(f"{title_prefix} - Before Map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_before_map.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mean_after_map, cmap="viridis")
    ax.set_title(f"{title_prefix} - After Map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_after_map.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mean_diff_map, cmap="inferno")
    ax.set_title(f"{title_prefix} - Difference Map (Original - Normalized)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_difference_map.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(mse_values) + 1), mse_values, marker="o", color="#d62728")
    ax.set_title(f"{title_prefix} - MSE per Sampel")
    ax.set_xlabel("Index Sampel")
    ax.set_ylabel("MSE")
    if len(labels) <= 12:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_mse_graph.jpg"), dpi=220)
    plt.close(fig)

    psnr_plot = [v if np.isfinite(v) else 100.0 for v in psnr_values]
    fig = plt.figure(figsize=(7.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(psnr_plot) + 1), psnr_plot, marker="o", color="#1f77b4")
    ax.set_title(f"{title_prefix} - PSNR per Sampel")
    ax.set_xlabel("Index Sampel")
    ax.set_ylabel("PSNR (dB)")
    if len(labels) <= 12:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.text(
        0.98,
        0.95,
        "Inf dipotong ke 100 dB untuk visualisasi",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "normalize_psnr_graph.jpg"), dpi=220)
    plt.close(fig)


def normalize_images(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    save_image: bool,
    preview: bool,
    preview_count: int,
    preview_style: str,
) -> dict:
    ensure_dir(output_dir)
    normalize_stats_dir = STATS_DIR / "normalize"
    ensure_dir(normalize_stats_dir)
    image_paths = list(iter_images(input_dir))
    rows = []
    total = 0
    skipped = 0
    t0 = time()
    preview_pairs = []
    noise_pairs = []
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}

    mean = std = None
    if mode == "zscore":
        mean, std = compute_mean_std(image_paths)
        json_stats_dir = JSON_STATS_DIR / "normalize"
        ensure_dir(json_stats_dir)
        write_json(json_stats_dir / "zscore_stats.json", {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source": str(input_dir),
        })

    if mode == "mobilenetv3":
        try:
            from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        except Exception as e:
            raise SystemExit(f"TensorFlow tidak tersedia untuk mobilenetv3 preprocess: {e}")
    else:
        preprocess_input = None

    for img_path in image_paths:
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue

        img_f = img.astype(np.float32)

        if mode == "minmax":
            normalized = img_f / 255.0
        elif mode == "zscore":
            normalized = (img_f / 255.0 - mean) / std
        elif mode == "mobilenetv3":
            normalized = preprocess_input(img_f)
        else:
            raise SystemExit(f"Mode tidak dikenal: {mode}")

        rel = img_path.relative_to(input_dir)
        out_npy = output_dir / rel.with_suffix(".npy")
        ensure_dir(out_npy.parent)
        np.save(out_npy, normalized.astype(np.float32))

        preview_img = None
        if save_image or preview:
            preview_arr = to_unit_range(normalized, mode)
            if mode in {"zscore", "mobilenetv3"}:
                preview_arr = (preview_arr - preview_arr.min()) / (preview_arr.max() - preview_arr.min() + 1e-8)
            preview_img = np.clip(preview_arr * 255.0, 0, 255).astype(np.uint8)
            if save_image:
                out_img = output_dir / rel.with_suffix(".jpg")
                write_image(out_img, preview_img)

        rows.append({
            "input": str(img_path),
            "output_npy": str(out_npy),
            "mode": mode,
        })
        if preview and preview_img is not None:
            label = class_from_path(input_dir, img_path)
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < per_class and len(preview_pairs) < preview_count:
                preview_pairs.append((img, preview_img, label))
                noise_pairs.append((img.astype(np.float32) / 255.0, to_unit_range(normalized, mode), label))
                class_counts[label] += 1
        total += 1

    elapsed = time() - t0
    if preview and preview_pairs:
        legacy_panel = normalize_stats_dir / "normalize_before_after_difference_panel.jpg"
        if legacy_panel.exists():
            legacy_panel.unlink()
        save_preview_grid(
            preview_pairs,
            normalize_stats_dir / "preview_normalize.jpg",
            matrix_view=(preview_style == "matrix"),
        )
        save_academic_normalization_charts(
            noise_pairs,
            normalize_stats_dir,
            "Normalisasi",
        )
    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize images for model training")
    parser.add_argument("--input", type=Path, default=RESIZE_DIR)
    parser.add_argument("--output", type=Path, default=NORMALIZED_DIR)
    parser.add_argument("--mode", choices=["minmax", "zscore", "mobilenetv3"], default="minmax")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    args = parser.parse_args()

    stats = normalize_images(
        args.input,
        args.output,
        args.mode,
        args.save_image,
        args.preview,
        args.preview_count,
        args.preview_style,
    )
    write_csv(args.output / "normalize_manifest.csv", stats["rows"])
    print(f"Normalisasi selesai. Total: {stats['total']}, Skipped: {stats['skipped']}, Time: {stats['seconds']}s")
    if not args.preview:
        print("Catatan: statistik visual normalisasi dibuat saat menjalankan command dengan flag `--preview`.")


if __name__ == "__main__":
    main()
