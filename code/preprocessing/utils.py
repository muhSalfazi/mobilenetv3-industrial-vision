from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from config import SUPPORTED_EXT


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        return []
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return sorted(files)


def class_from_path(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except Exception:
        pass
    return path.parent.name


def read_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def write_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), image)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def to_matrix_view(img: np.ndarray, scale: int = 2) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    gray = (gray * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    if scale > 1:
        heat = cv2.resize(heat, (heat.shape[1] * scale, heat.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return heat


def _unpack_pair(pair):
    if len(pair) == 3:
        return pair[0], pair[1], pair[2]
    return pair[0], pair[1], None


def save_preview_grid(
    pairs: list[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, str]],
    out_path: Path,
    tile_cols: int = 3,
    matrix_view: bool = False,
) -> None:
    if not pairs:
        return
    tiles = []
    for pair in pairs:
        a, b, _ = _unpack_pair(pair)
        if a is None or b is None:
            continue
        if matrix_view:
            a = to_matrix_view(a)
            b = to_matrix_view(b)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        tiles.append(np.hstack([a, b]))
    if not tiles:
        return

    rows = []
    for i in range(0, len(tiles), tile_cols):
        row = tiles[i:i + tile_cols]
        max_h = max(t.shape[0] for t in row)
        row_fixed = []
        for t in row:
            if t.shape[0] < max_h:
                pad = max_h - t.shape[0]
                t = cv2.copyMakeBorder(t, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            row_fixed.append(t)
        rows.append(np.hstack(row_fixed))

    if not rows:
        return
    max_w = max(r.shape[1] for r in rows)
    rows_fixed = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = max_w - r.shape[1]
            r = cv2.copyMakeBorder(r, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        rows_fixed.append(r)

    grid = np.vstack(rows_fixed)
    write_image(out_path, grid)


def _histogram_image(gray: np.ndarray, width: int = 256, height: int = 200) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    if hist.max() > 0:
        hist = hist / hist.max()
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for x in range(256):
        val = int(hist[x] * (height - 10))
        cv2.line(canvas, (x, height - 5), (x, height - 5 - val), (60, 120, 200), 1)
    return canvas


def save_histogram_chart(
    pairs: list[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, str]],
    out_path: Path,
    title: str,
    width: int = 640,
    height: int = 360,
) -> None:
    if not pairs:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        return

    before = []
    after = []
    for pair in pairs:
        a, b, _ = _unpack_pair(pair)
        if a is None or b is None:
            continue
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        before.append(a.flatten())
        after.append(b.flatten())
    if not before or not after:
        return
    before = np.concatenate(before)
    after = np.concatenate(after)

    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(before, bins=50, stat="density", color="#1f77b4", alpha=0.45, label="Before", ax=ax)
    sns.histplot(after, bins=50, stat="density", color="#2ca02c", alpha=0.45, label="After", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def save_class_stats_charts(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    out_prefix: Path,
    title_prefix: str,
    sample_pixels: int = 5000,
) -> None:
    if not pairs:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        return

    classes = sorted({p[2] for p in pairs if p[2] is not None})
    if not classes:
        return

    rng = np.random.default_rng(42)
    before_means = []
    after_means = []
    before_stds = []
    after_stds = []
    before_samples = []
    after_samples = []

    for cls in classes:
        cls_pairs = [p for p in pairs if p[2] == cls]
        if not cls_pairs:
            continue
        b_means = []
        a_means = []
        b_stds = []
        a_stds = []
        b_pix = []
        a_pix = []
        for a, b, _ in cls_pairs:
            if a is None or b is None:
                continue
            if a.ndim == 3:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            if b.ndim == 3:
                b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            b_means.append(float(a.mean()))
            a_means.append(float(b.mean()))
            b_stds.append(float(a.std()))
            a_stds.append(float(b.std()))

            flat_a = a.flatten()
            flat_b = b.flatten()
            if flat_a.size > sample_pixels:
                idx = rng.choice(flat_a.size, sample_pixels, replace=False)
                flat_a = flat_a[idx]
            if flat_b.size > sample_pixels:
                idx = rng.choice(flat_b.size, sample_pixels, replace=False)
                flat_b = flat_b[idx]
            b_pix.append(flat_a)
            a_pix.append(flat_b)

        before_means.append(np.mean(b_means) if b_means else 0)
        after_means.append(np.mean(a_means) if a_means else 0)
        before_stds.append(np.mean(b_stds) if b_stds else 0)
        after_stds.append(np.mean(a_stds) if a_stds else 0)
        before_samples.append(np.concatenate(b_pix) if b_pix else np.array([0]))
        after_samples.append(np.concatenate(a_pix) if a_pix else np.array([0]))

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, before_means, width, label="Before")
    plt.bar(x + width / 2, after_means, width, label="After")
    plt.xticks(x, classes, rotation=20)
    plt.title(f"{title_prefix} - Mean Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_prefix.with_name(out_prefix.name + "_mean.jpg")))
    plt.close()

    # Heatmap: class (rows) vs before/after (columns)
    if classes:
        heat = np.vstack([before_means, after_means]).T
        plt.figure(figsize=(6, 0.6 * len(classes) + 1.5))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            yticklabels=classes,
            xticklabels=["Before", "After"],
            cbar=True,
        )
        plt.title(f"{title_prefix} - Mean Heatmap")
        plt.tight_layout()
        plt.savefig(str(out_prefix.with_name(out_prefix.name + "_mean_heatmap.jpg")))
        plt.close()


def save_professional_charts(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    out_prefix: Path,
    title_prefix: str,
    sample_pixels: int = 20000,
) -> None:
    if not pairs:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rng = np.random.default_rng(42)
    before_pixels = []
    after_pixels = []
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
        a = a.astype(np.float32)
        b = b.astype(np.float32)

        before_imgs.append(a)
        after_imgs.append(b)

        fa = a.flatten()
        fb = b.flatten()
        if fa.size > sample_pixels:
            idx = rng.choice(fa.size, sample_pixels, replace=False)
            fa = fa[idx]
        if fb.size > sample_pixels:
            idx = rng.choice(fb.size, sample_pixels, replace=False)
            fb = fb[idx]
        before_pixels.append(fa)
        after_pixels.append(fb)

    if not before_imgs or not after_imgs:
        return

    before_pixels = np.concatenate(before_pixels)
    after_pixels = np.concatenate(after_pixels)
    mean_before = np.mean(np.stack(before_imgs, axis=0), axis=0)
    mean_after = np.mean(np.stack(after_imgs, axis=0), axis=0)
    diff = np.abs(mean_after - mean_before)

    plt.figure(figsize=(7, 4))
    plt.hist(before_pixels, bins=50, alpha=0.6, label="Before", color="#1f77b4", density=True)
    plt.hist(after_pixels, bins=50, alpha=0.6, label="After", color="#2ca02c", density=True)
    plt.title(f"{title_prefix} - Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_prefix.with_name(out_prefix.name + "_hist.jpg")))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.boxplot([before_pixels, after_pixels], labels=["Before", "After"], patch_artist=True)
    plt.title(f"{title_prefix} - Boxplot Intensity")
    plt.ylabel("Pixel Intensity")
    plt.tight_layout()
    plt.savefig(str(out_prefix.with_name(out_prefix.name + "_boxplot.jpg")))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(mean_before, cmap="viridis")
    plt.title(f"{title_prefix} - Mean Heatmap (Before)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_prefix.with_name(out_prefix.name + "_heatmap_before.jpg")))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(mean_after, cmap="viridis")
    plt.title(f"{title_prefix} - Mean Heatmap (After)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_prefix.with_name(out_prefix.name + "_heatmap_after.jpg")))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(diff, cmap="inferno")
    plt.title(f"{title_prefix} - Heatmap Difference")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_prefix.with_name(out_prefix.name + "_heatmap_diff.jpg")))
    plt.close()


def save_professional_panels(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    out_prefix: Path,
    title_prefix: str,
    sample_pixels: int = 30000,
) -> None:
    if not pairs:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except Exception:
        return

    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        sns = None

    rng = np.random.default_rng(42)
    before_pixels = []
    after_pixels = []
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
        a = a.astype(np.float32)
        b = b.astype(np.float32)

        before_imgs.append(a)
        after_imgs.append(b)

        fa = a.flatten()
        fb = b.flatten()
        if fa.size > sample_pixels:
            idx = rng.choice(fa.size, sample_pixels, replace=False)
            fa = fa[idx]
        if fb.size > sample_pixels:
            idx = rng.choice(fb.size, sample_pixels, replace=False)
            fb = fb[idx]
        before_pixels.append(fa)
        after_pixels.append(fb)

    if not before_imgs or not after_imgs:
        return

    before_pixels = np.concatenate(before_pixels)
    after_pixels = np.concatenate(after_pixels)
    mean_before = np.mean(np.stack(before_imgs, axis=0), axis=0)
    mean_after = np.mean(np.stack(after_imgs, axis=0), axis=0)
    diff = np.abs(mean_after - mean_before)

    # Panel heatmap (Before/After/Diff) with colorbars
    fig = plt.figure(figsize=(10, 3.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    im1 = ax1.imshow(mean_before, cmap="viridis")
    ax1.set_title("Before")
    ax1.axis("off")
    im2 = ax2.imshow(mean_after, cmap="viridis")
    ax2.set_title("After")
    ax2.axis("off")
    im3 = ax3.imshow(diff, cmap="inferno")
    ax3.set_title("Difference")
    ax3.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title_prefix} - Mean Heatmap", y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_prefix.with_name(out_prefix.name + "_panel_heatmap.jpg")), dpi=200)
    plt.close(fig)

    # Histogram (professional)
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.histplot(before_pixels, bins=50, stat="density", color="#1f77b4", alpha=0.5, label="Before", ax=ax)
        sns.histplot(after_pixels, bins=50, stat="density", color="#2ca02c", alpha=0.5, label="After", ax=ax)
    else:
        ax.hist(before_pixels, bins=50, density=True, alpha=0.5, label="Before", color="#1f77b4")
        ax.hist(after_pixels, bins=50, density=True, alpha=0.5, label="After", color="#2ca02c")
    ax.set_title(f"{title_prefix} - Intensity Distribution")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_prefix.with_name(out_prefix.name + "_panel_hist.jpg")), dpi=200)
    plt.close(fig)

    # Boxplot (professional)
    fig = plt.figure(figsize=(5.5, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.boxplot(
            data=[before_pixels, after_pixels],
            palette=["#1f77b4", "#2ca02c"],
            ax=ax,
        )
        ax.set_xticklabels(["Before", "After"])
    else:
        ax.boxplot([before_pixels, after_pixels], labels=["Before", "After"], patch_artist=True)
    ax.set_title(f"{title_prefix} - Boxplot Intensity")
    ax.set_ylabel("Pixel Intensity")
    fig.tight_layout()
    fig.savefig(str(out_prefix.with_name(out_prefix.name + "_panel_boxplot.jpg")), dpi=200)
    plt.close(fig)

def save_noise_previews(
    pairs: list[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, str]],
    out_heatmap: Path,
    out_hist: Path,
    tile_cols: int = 3,
) -> None:
    if not pairs:
        return
    heatmaps = []
    hists = []
    for pair in pairs:
        a, b, _ = _unpack_pair(pair)
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        if a.ndim == 3:
            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        else:
            a_gray = a
        if b.ndim == 3:
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        else:
            b_gray = b
        diff = cv2.absdiff(a_gray, b_gray)
        heat = to_matrix_view(diff)
        heatmaps.append(heat)
        hists.append(_histogram_image(diff))

    if heatmaps:
        rows = []
        for i in range(0, len(heatmaps), tile_cols):
            row = heatmaps[i:i + tile_cols]
            max_h = max(t.shape[0] for t in row)
            row_fixed = []
            for t in row:
                if t.shape[0] < max_h:
                    pad = max_h - t.shape[0]
                t = cv2.copyMakeBorder(t, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
                row_fixed.append(t)
            rows.append(np.hstack(row_fixed))
        max_w = max(r.shape[1] for r in rows)
        rows_fixed = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = max_w - r.shape[1]
            r = cv2.copyMakeBorder(r, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            rows_fixed.append(r)
        write_image(out_heatmap, np.vstack(rows_fixed))

    if hists:
        rows = []
        for i in range(0, len(hists), tile_cols):
            row = hists[i:i + tile_cols]
            max_h = max(t.shape[0] for t in row)
            row_fixed = []
            for t in row:
                if t.shape[0] < max_h:
                    pad = max_h - t.shape[0]
                    t = cv2.copyMakeBorder(t, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
                row_fixed.append(t)
            rows.append(np.hstack(row_fixed))
        max_w = max(r.shape[1] for r in rows)
        rows_fixed = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = max_w - r.shape[1]
                r = cv2.copyMakeBorder(r, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            rows_fixed.append(r)
        write_image(out_hist, np.vstack(rows_fixed))
