#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path
from time import time

import cv2
import numpy as np

from config import IMG_SIZE, RAW_DIR, RESIZE_DIR, STATS_DIR
from utils import class_from_path, ensure_dir, iter_images, read_image, save_class_stats_charts, save_histogram_chart, save_preview_grid, save_professional_panels, write_csv, write_image


def _to_binary_matrix(
    img: np.ndarray,
    grid_shape: tuple[int, int] = (16, 16),
    keep_aspect: bool = False,
) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    target_h, target_w = grid_shape
    if keep_aspect:
        src_h, src_w = gray.shape[:2]
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small = np.zeros((target_h, target_w), dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        small[y0:y0 + new_h, x0:x0 + new_w] = resized
    else:
        small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def _save_binary_matrix_figure(
    binary_matrix: np.ndarray,
    title: str,
    caption: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows, cols = binary_matrix.shape
    fig_w = max(6.2, cols * 0.38)
    fig_h = max(6.8, rows * 0.38 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(binary_matrix, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title, fontsize=15, weight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#999999", linestyle="-", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for r in range(rows):
        for c in range(cols):
            v = int(binary_matrix[r, c])
            color = "white" if v == 1 else "#2f2f2f"
            ax.text(c, r, str(v), ha="center", va="center", fontsize=8.5, color=color)

    fig.text(0.5, 0.035, caption, ha="center", va="bottom", fontsize=11, color="#333333")
    fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.97])
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def _save_resize_example_figure(
    before_img: np.ndarray,
    after_img: np.ndarray,
    label: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    b_h, b_w = before_img.shape[:2]
    a_h, a_w = after_img.shape[:2]
    before_rgb = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    axes[0].imshow(before_rgb)
    axes[0].set_title(f"Sebelum Resize ({b_w} x {b_h})", fontsize=12, weight="bold")
    axes[0].axis("off")
    axes[1].imshow(after_rgb)
    axes[1].set_title(f"Sesudah Resize ({a_w} x {a_h})", fontsize=12, weight="bold")
    axes[1].axis("off")

    fig.suptitle("Contoh Dampak Resize (Before vs After)", fontsize=14, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.03,
        f"Kelas: {label} | Catatan: perbedaan rasio aspek dapat menyebabkan objek tampak meregang/menyusut.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.93])
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def resize_images(
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int],
    preview: bool,
    preview_count: int,
    preview_style: str,
    demo_rank: int,
    demo_image: str | None,
) -> dict:
    ensure_dir(output_dir)
    rows = []
    total = 0
    skipped = 0
    t0 = time()
    preview_pairs = []
    top_resize_demos = []
    rank_index = max(1, demo_rank)
    rank_counter = 0
    demo_image_norm = demo_image.strip().replace("\\", "/").lstrip("./") if demo_image else None
    forced_demo = None
    class_dirs = [d.name for d in input_dir.iterdir() if d.is_dir()] if input_dir.exists() else []
    per_class = max(1, math.ceil(preview_count / max(len(class_dirs), 1)))
    class_counts = {c: 0 for c in class_dirs}

    for img_path in iter_images(input_dir):
        img = read_image(img_path)
        if img is None:
            skipped += 1
            continue
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".jpg")
        write_image(out_path, resized)
        if preview:
            rel_jpg = rel.with_suffix(".jpg").as_posix()
            rel_src = rel.as_posix()
            if demo_image_norm and (rel_jpg == demo_image_norm or rel_src == demo_image_norm):
                forced_demo = (img.copy(), resized.copy(), class_from_path(input_dir, img_path))
            src_h, src_w = img.shape[:2]
            target_ratio = size[0] / max(size[1], 1)
            src_ratio = src_w / max(src_h, 1)
            aspect_shift = abs(math.log((src_ratio + 1e-8) / (target_ratio + 1e-8)))
            label = class_from_path(input_dir, img_path)
            candidate = (aspect_shift, rank_counter, img.copy(), resized.copy(), label)
            rank_counter += 1
            if len(top_resize_demos) < rank_index:
                heapq.heappush(top_resize_demos, candidate)
            elif aspect_shift > top_resize_demos[0][0]:
                heapq.heapreplace(top_resize_demos, candidate)
        rows.append({
            "input": str(img_path),
            "output": str(out_path),
            "width": size[0],
            "height": size[1],
        })
        if preview:
            label = class_from_path(input_dir, img_path)
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < per_class and len(preview_pairs) < preview_count:
                preview_pairs.append((img, resized, label))
                class_counts[label] += 1
        total += 1

    elapsed = time() - t0
    demo_source = "auto"
    if preview and preview_pairs:
        resize_stats_dir = STATS_DIR / "resize"
        ensure_dir(resize_stats_dir)
        legacy_binary_path = resize_stats_dir / "preview_resize_binary.jpg"
        if legacy_binary_path.exists():
            legacy_binary_path.unlink()
        if forced_demo is not None:
            before_img, after_img, sample_label = forced_demo
            demo_source = "forced"
        elif top_resize_demos:
            ranked = sorted(top_resize_demos, key=lambda x: x[0], reverse=True)
            pick_idx = min(rank_index, len(ranked)) - 1
            _, _, before_img, after_img, sample_label = ranked[pick_idx]
            demo_source = f"rank-{pick_idx + 1}"
        else:
            before_img, after_img, sample_label = preview_pairs[0]
            demo_source = "preview-first"
        before_matrix = _to_binary_matrix(before_img, keep_aspect=True)
        after_matrix = _to_binary_matrix(after_img)
        before_size = f"{before_img.shape[1]} x {before_img.shape[0]} px"
        after_size = f"{after_img.shape[1]} x {after_img.shape[0]} px"
        _save_binary_matrix_figure(
            before_matrix,
            "Citra Biner Sebelum Resize (0/1)",
            f"Kelas: {sample_label} | Ukuran Awal: {before_size}",
            resize_stats_dir / "preview_resize_binary_before.jpg",
        )
        _save_binary_matrix_figure(
            after_matrix,
            "Citra Biner Sesudah Resize (0/1)",
            f"Kelas: {sample_label} | Ukuran Sesudah Resize: {after_size}",
            resize_stats_dir / "preview_resize_binary_after.jpg",
        )
        _save_resize_example_figure(
            before_img,
            after_img,
            sample_label,
            resize_stats_dir / "preview_resize_example.jpg",
        )
        save_preview_grid(
            preview_pairs,
            resize_stats_dir / "preview_resize.jpg",
            matrix_view=(preview_style == "matrix"),
        )
        save_histogram_chart(
            preview_pairs,
            resize_stats_dir / "chart_resize_before_after.jpg",
            "Resize: Before vs After",
        )
        save_class_stats_charts(
            preview_pairs,
            resize_stats_dir / "chart_resize_class",
            "Resize (Per Class)",
        )   
        save_professional_panels(
            preview_pairs,
            resize_stats_dir / "chart_resize_professional",
            "Resize",
        )
    return {
        "total": total,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "rows": rows,
        "demo_source": demo_source,
        "demo_image": demo_image_norm if demo_image_norm else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize images for dataset preprocessing")
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=RESIZE_DIR)
    parser.add_argument("--width", type=int, default=IMG_SIZE[0])
    parser.add_argument("--height", type=int, default=IMG_SIZE[1])
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-style", choices=["image", "matrix"], default="image")
    parser.add_argument("--demo-rank", type=int, default=1, help="Pilih sampel demo resize berdasarkan peringkat distorsi rasio aspek (1 = paling tinggi)")
    parser.add_argument("--demo-image", type=str, default=None, help="Path relatif sampel demo (contoh: bekerja/2.jpg)")
    args = parser.parse_args()

    stats = resize_images(
        args.input,
        args.output,
        (args.width, args.height),
        args.preview,
        args.preview_count,
        args.preview_style,
        args.demo_rank,
        args.demo_image,
    )
    write_csv(args.output / "resize_manifest.csv", stats["rows"])
    if args.preview and args.demo_image and stats.get("demo_source") != "forced":
        print(f"Demo image '{args.demo_image}' tidak ditemukan, fallback ke {stats.get('demo_source')}.")
    print(f"Resize selesai. Total: {stats['total']}, Skipped: {stats['skipped']}, Time: {stats['seconds']}s")


if __name__ == "__main__":
    main()
