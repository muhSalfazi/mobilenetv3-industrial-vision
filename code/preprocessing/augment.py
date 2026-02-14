#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import time

import cv2
import numpy as np

from config import AUGMENTED_DIR, FILTERED_DIR, STATS_DIR, JSON_STATS_DIR
from utils import (
    class_from_path,
    ensure_dir,
    iter_images,
    read_image,
    save_histogram_chart,
    save_preview_grid,
    save_professional_panels,
    write_csv,
    write_image,
    write_json,
)


@dataclass(frozen=True)
class AugmentConfig:
    rotation_max_deg: float = 15.0
    contrast_delta: float = 0.2
    brightness_min: float = 0.7
    brightness_max: float = 1.3
    translate_max_pct: float = 0.1
    flip_prob: float = 0.5


def _sample_params(
    rng: np.random.Generator,
    width: int,
    height: int,
    cfg: AugmentConfig,
) -> dict[str, float | bool]:
    angle_deg = float(rng.uniform(-cfg.rotation_max_deg, cfg.rotation_max_deg))
    do_flip = bool(rng.random() < cfg.flip_prob)
    contrast_alpha = float(rng.uniform(1.0 - cfg.contrast_delta, 1.0 + cfg.contrast_delta))
    brightness_factor = float(rng.uniform(cfg.brightness_min, cfg.brightness_max))
    tx_pct = float(rng.uniform(-cfg.translate_max_pct, cfg.translate_max_pct))
    ty_pct = float(rng.uniform(-cfg.translate_max_pct, cfg.translate_max_pct))
    tx_px = float(tx_pct * width)
    ty_px = float(ty_pct * height)
    return {
        "angle_deg": angle_deg,
        "flip_horizontal": do_flip,
        "contrast_alpha": contrast_alpha,
        "brightness_factor": brightness_factor,
        "translate_x_pct": tx_pct,
        "translate_y_pct": ty_pct,
        "translate_x_px": tx_px,
        "translate_y_px": ty_px,
    }


def _apply_augment(img: np.ndarray, params: dict[str, float | bool]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img
    if params["flip_horizontal"]:
        out = cv2.flip(out, 1)

    m = cv2.getRotationMatrix2D((w / 2, h / 2), params["angle_deg"], 1.0)
    m[0, 2] += params["translate_x_px"]
    m[1, 2] += params["translate_y_px"]
    out = cv2.warpAffine(
        out,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Kontras sekitar titik tengah, lalu brightness sebagai faktor skala.
    out_f = out.astype(np.float32) / 255.0
    out_f = (out_f - 0.5) * params["contrast_alpha"] + 0.5
    out_f = out_f * params["brightness_factor"]
    out = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)
    return out


def _summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _save_augment_charts(
    stats_dir: Path,
    rows: list[dict],
    before_gray_values: list[float],
    after_gray_values: list[float],
    class_before: Counter,
    class_after: Counter,
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "Peringatan: chart augment tidak dibuat karena matplotlib belum tersedia "
            f"({exc.__class__.__name__})."
        )
        return

    angles = np.array([float(r["angle_deg"]) for r in rows], dtype=np.float32)
    contrasts = np.array([float(r["contrast_alpha"]) for r in rows], dtype=np.float32)
    brightness = np.array([float(r["brightness_factor"]) for r in rows], dtype=np.float32)
    tx = np.array([float(r["translate_x_pct"]) * 100.0 for r in rows], dtype=np.float32)
    ty = np.array([float(r["translate_y_pct"]) * 100.0 for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    plots = [
        (angles, "Distribusi Rotasi (deg)", "Angle (deg)", "#1f77b4"),
        (contrasts, "Distribusi Kontras", "Contrast alpha", "#2ca02c"),
        (brightness, "Distribusi Brightness", "Brightness factor", "#ff7f0e"),
        (tx, "Distribusi Translasi X (%)", "Translate X (%)", "#9467bd"),
        (ty, "Distribusi Translasi Y (%)", "Translate Y (%)", "#d62728"),
    ]
    for idx, (vals, title, xlabel, color) in enumerate(plots):
        ax = axes[idx]
        ax.hist(vals, bins=24, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Jumlah")
        ax.grid(axis="y", alpha=0.25)
    flip_ratio = float(np.mean([1.0 if bool(r["flip_horizontal"]) else 0.0 for r in rows]) * 100.0)
    axes[5].axis("off")
    axes[5].text(
        0.05,
        0.7,
        "Ringkasan Augmentasi",
        fontsize=12,
        weight="bold",
    )
    axes[5].text(
        0.05,
        0.48,
        f"Flip horizontal aktif: {flip_ratio:.2f}%",
        fontsize=10.5,
    )
    axes[5].text(
        0.05,
        0.30,
        f"Total sampel augment: {len(rows)}",
        fontsize=10.5,
    )
    fig.suptitle("Distribusi Parameter Augmentasi", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(stats_dir / "augment_parameter_distributions.jpg"), dpi=220)
    plt.close(fig)

    split_specs = [
        (angles, "Distribusi Rotasi (deg)", "Angle (deg)", "#1f77b4", "augment_rotation_distribution.jpg"),
        (contrasts, "Distribusi Kontras", "Contrast alpha", "#2ca02c", "augment_contrast_distribution.jpg"),
        (brightness, "Distribusi Brightness", "Brightness factor", "#ff7f0e", "augment_brightness_distribution.jpg"),
        (tx, "Distribusi Translasi X (%)", "Translate X (%)", "#9467bd", "augment_translate_x_distribution.jpg"),
        (ty, "Distribusi Translasi Y (%)", "Translate Y (%)", "#d62728", "augment_translate_y_distribution.jpg"),
    ]
    for vals, title, xlabel, color, filename in split_specs:
        fig = plt.figure(figsize=(6.2, 4.2))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(vals, bins=24, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Jumlah")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(str(stats_dir / filename), dpi=220)
        plt.close(fig)

    fig = plt.figure(figsize=(5.0, 2.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.03, 0.78, "Ringkasan Augmentasi", fontsize=12, weight="bold")
    ax.text(0.03, 0.48, f"Flip horizontal aktif: {flip_ratio:.2f}%", fontsize=10.5)
    ax.text(0.03, 0.24, f"Total sampel augment: {len(rows)}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(str(stats_dir / "augment_parameter_summary.jpg"), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    classes = sorted(set(class_before.keys()) | set(class_after.keys()))
    x = np.arange(len(classes))
    width = 0.35
    before_vals = [int(class_before.get(c, 0)) for c in classes]
    after_vals = [int(class_after.get(c, 0)) for c in classes]
    ax.bar(x - width / 2, before_vals, width, label="Before", color="#1f77b4", alpha=0.9)
    ax.bar(x + width / 2, after_vals, width, label="After", color="#2ca02c", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylabel("Jumlah Gambar")
    ax.set_title("Perbandingan Class Balance (Before vs After Augmentasi)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(stats_dir / "augment_class_balance.jpg"), dpi=220)
    plt.close(fig)

    if before_gray_values and after_gray_values:
        fig = plt.figure(figsize=(7.4, 4.2))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(before_gray_values, bins=50, density=True, alpha=0.5, label="Before", color="#1f77b4")
        ax.hist(after_gray_values, bins=50, density=True, alpha=0.5, label="After", color="#ff7f0e")
        ax.set_title("Distribusi Intensitas Mean Gray (Before vs After)")
        ax.set_xlabel("Mean Gray Intensity")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(str(stats_dir / "augment_intensity_shift.jpg"), dpi=220)
        plt.close(fig)

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(tx, ty, c=angles, cmap="coolwarm", alpha=0.65, s=20)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Translate X (%)")
    ax.set_ylabel("Translate Y (%)")
    ax.set_title("Sebaran Translasi (warna = sudut rotasi)")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Angle (deg)")
    fig.tight_layout()
    fig.savefig(str(stats_dir / "augment_translation_scatter.jpg"), dpi=220)
    plt.close(fig)


def _save_split_mean_heatmaps(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    stats_dir: Path,
) -> None:
    if not pairs:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "Peringatan: heatmap augment tidak dibuat karena matplotlib belum tersedia "
            f"({exc.__class__.__name__})."
        )
        return

    before_imgs = []
    after_imgs = []
    target_shape = None

    for a, b, _ in pairs:
        if a is None or b is None:
            continue
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if target_shape is None:
            target_shape = a.shape
        if a.shape != target_shape:
            a = cv2.resize(a, (target_shape[1], target_shape[0]))
        if b.shape != target_shape:
            b = cv2.resize(b, (target_shape[1], target_shape[0]))
        before_imgs.append(a.astype(np.float32))
        after_imgs.append(b.astype(np.float32))

    if not before_imgs or not after_imgs:
        return

    mean_before = np.mean(np.stack(before_imgs, axis=0), axis=0)
    mean_after = np.mean(np.stack(after_imgs, axis=0), axis=0)
    diff = np.abs(mean_after - mean_before)
    heatmaps = [
        (mean_before, "Augmentasi - Mean Heatmap Before", "viridis", "chart_augment_heatmap_before.jpg"),
        (mean_after, "Augmentasi - Mean Heatmap After", "viridis", "chart_augment_heatmap_after.jpg"),
        (diff, "Augmentasi - Mean Heatmap Difference", "inferno", "chart_augment_heatmap_difference.jpg"),
    ]

    for data, title, cmap, filename in heatmaps:
        fig = plt.figure(figsize=(5.6, 4.8))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(stats_dir / filename), dpi=220)
        plt.close(fig)


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    per_image: int,
    seed: int,
    preview: bool = False,
    preview_count: int = 12,
    preview_style: str = "image",
    stats_dir: Path | None = None,
    cfg: AugmentConfig | None = None,
) -> dict:
    ensure_dir(output_dir)
    if stats_dir is None:
        stats_dir = STATS_DIR / "augment"
    ensure_dir(stats_dir)
    cfg = cfg or AugmentConfig()

    rows = []
    total = 0
    skipped = 0
    t0 = time()
    rng = np.random.default_rng(seed)
    preview_pairs = []
    before_gray_values = []
    after_gray_values = []
    class_before = Counter()
    class_after = Counter()
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}

    for img_path in iter_images(input_dir):
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        cls = class_from_path(input_dir, img_path)
        class_before[cls] += 1
        base_gray_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        before_gray_values.append(base_gray_mean)
        rel = img_path.relative_to(input_dir)

        for i in range(per_image):
            params = _sample_params(rng, w, h, cfg)
            aug = _apply_augment(img, params)
            out_path = output_dir / rel.parent / f"{rel.stem}_aug{i+1:02d}.jpg"
            write_image(out_path, aug)
            aug_gray_mean = float(cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY).mean())
            after_gray_values.append(aug_gray_mean)
            class_after[cls] += 1

            rows.append({
                "input": str(img_path),
                "output": str(out_path),
                "class_label": cls,
                "augment_index": i + 1,
                "angle_deg": round(float(params["angle_deg"]), 4),
                "flip_horizontal": bool(params["flip_horizontal"]),
                "contrast_alpha": round(float(params["contrast_alpha"]), 4),
                "brightness_factor": round(float(params["brightness_factor"]), 4),
                "translate_x_pct": round(float(params["translate_x_pct"]), 5),
                "translate_y_pct": round(float(params["translate_y_pct"]), 5),
                "translate_x_px": round(float(params["translate_x_px"]), 3),
                "translate_y_px": round(float(params["translate_y_px"]), 3),
                "gray_mean_before": round(base_gray_mean, 4),
                "gray_mean_after": round(aug_gray_mean, 4),
            })
            if preview:
                if cls not in class_counts:
                    class_counts[cls] = 0
                if class_counts[cls] < per_class and len(preview_pairs) < preview_count:
                    preview_pairs.append((img.copy(), aug.copy(), cls))
                    class_counts[cls] += 1
            total += 1

    elapsed = time() - t0
    _save_augment_charts(stats_dir, rows, before_gray_values, after_gray_values, class_before, class_after)
    if preview and preview_pairs:
        save_preview_grid(
            preview_pairs,
            stats_dir / "preview_augment.jpg",
            matrix_view=(preview_style == "matrix"),
        )
        save_histogram_chart(
            preview_pairs,
            stats_dir / "chart_augment_before_after.jpg",
            "Augmentasi: Before vs After",
        )
        save_professional_panels(
            preview_pairs,
            stats_dir / "chart_augment_professional",
            "Augmentasi",
        )
        _save_split_mean_heatmaps(preview_pairs, stats_dir)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_input_images": int(sum(class_before.values())),
        "total_augmented_images": int(total),
        "per_image": int(per_image),
        "seed": int(seed),
        "seconds": round(elapsed, 2),
        "class_counts_before": dict(sorted(class_before.items())),
        "class_counts_after": dict(sorted(class_after.items())),
        "rotation_angle_deg": _summarize_numeric([float(r["angle_deg"]) for r in rows]),
        "contrast_alpha": _summarize_numeric([float(r["contrast_alpha"]) for r in rows]),
        "brightness_factor": _summarize_numeric([float(r["brightness_factor"]) for r in rows]),
        "translate_x_pct": _summarize_numeric([float(r["translate_x_pct"]) for r in rows]),
        "translate_y_pct": _summarize_numeric([float(r["translate_y_pct"]) for r in rows]),
        "gray_mean_before": _summarize_numeric(before_gray_values),
        "gray_mean_after": _summarize_numeric(after_gray_values),
        "flip_horizontal_ratio_percent": round(
            float(np.mean([1.0 if bool(r["flip_horizontal"]) else 0.0 for r in rows]) * 100.0),
            4,
        ) if rows else 0.0,
        "transform_config": {
            "rotation_deg_range": [-cfg.rotation_max_deg, cfg.rotation_max_deg],
            "contrast_alpha_range": [1.0 - cfg.contrast_delta, 1.0 + cfg.contrast_delta],
            "brightness_factor_range": [cfg.brightness_min, cfg.brightness_max],
            "translate_pct_range": [-cfg.translate_max_pct, cfg.translate_max_pct],
            "flip_probability": cfg.flip_prob,
        },
    }
    json_stats_dir = JSON_STATS_DIR / "augment"
    ensure_dir(json_stats_dir)
    write_json(json_stats_dir / "augment_summary.json", summary)

    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
        "summary": summary,
        "stats_dir": str(stats_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentasi data latih dengan statistik akademik")
    parser.add_argument("--input", type=Path, default=FILTERED_DIR)
    parser.add_argument("--output", type=Path, default=AUGMENTED_DIR)
    parser.add_argument("--per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    parser.add_argument("--stats-dir", type=Path, default=STATS_DIR / "augment")
    args = parser.parse_args()

    stats = augment_dataset(
        args.input,
        args.output,
        args.per_image,
        args.seed,
        preview=args.preview,
        preview_count=args.preview_count,
        preview_style=args.preview_style,
        stats_dir=args.stats_dir,
    )
    write_csv(args.output / "augment_manifest.csv", stats["rows"])
    print(
        "Augmentasi selesai. "
        f"Total: {stats['total']}, Skipped: {stats['skipped']}, "
        f"Time: {stats['seconds']}s, Stats: {stats['stats_dir']}"
    )
    if not args.preview:
        print("Catatan: preview tambahan augment dibuat saat menjalankan command dengan flag `--preview`.")


if __name__ == "__main__":
    main()
