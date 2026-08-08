#!/usr/bin/env python3
"""Utility functions untuk preprocessing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists, create if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_images(root_dir: Path, extensions: list[str] | None = None) -> list[Path]:
    """Iterate over all image files in directory (recursively by class)."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    
    if not root_dir.exists():
        return []
    
    images = []
    for ext in extensions:
        images.extend(root_dir.rglob(f"*{ext}"))
        images.extend(root_dir.rglob(f"*{ext.upper()}"))
    
    return sorted(set(images))


def read_image(path: Path | str) -> np.ndarray | None:
    """
    Read image from file using OpenCV (BGR format).
    Returns None jika file corrupt atau tidak dapat dibaca.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  ⚠️  Tidak dapat membaca: {path}")
            return None
        return img
    except Exception as e:
        print(f"  ✗ Error membaca {path}: {e}")
        return None


def write_image(path: Path | str, img: np.ndarray) -> bool:
    """Write image to file using OpenCV."""
    try:
        ensure_dir(Path(path).parent)
        success = cv2.imwrite(str(path), img)
        if not success:
            print(f"  ✗ Gagal menulis: {path}")
        return success
    except Exception as e:
        print(f"  ✗ Error menulis {path}: {e}")
        return False


def write_image_png(path: Path | str, img: np.ndarray) -> bool:
    """Write image as PNG."""
    try:
        ensure_dir(Path(path).parent)
        success = cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return success
    except Exception as e:
        print(f"  ✗ Error menulis PNG {path}: {e}")
        return False


def class_from_path(root_dir: Path, img_path: Path) -> str:
    """
    Extract class name dari path image.
    Assumes structure: root_dir/class_name/image_file
    """
    try:
        rel = img_path.relative_to(root_dir)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]  # Class folder name
        return "unknown"
    except:
        return "unknown"


def write_csv(path: Path | str, rows: list[dict]) -> bool:
    """Write list of dicts as CSV."""
    try:
        path = Path(path)
        ensure_dir(path.parent)
        
        if not rows:
            return True
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return True
    except Exception as e:
        print(f"  ✗ Error menulis CSV {path}: {e}")
        return False


def write_json(path: Path | str, data: dict | list) -> bool:
    """Write dict/list as JSON."""
    try:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"  ✗ Error menulis JSON {path}: {e}")
        return False


def read_json(path: Path | str) -> dict | list | None:
    """Read JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ✗ Error membaca JSON {path}: {e}")
        return None


def save_preview_grid(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: Path | str,
    matrix_view: bool = False,
    cols: int = 3,
) -> bool:
    """
    Save side-by-side comparison grid dari before/after images.
    
    Parameters:
    - pairs: List of (before_img, after_img, label) tuples
    - output_path: Output file path
    - matrix_view: If True, arrange in matrix; if False, side-by-side
    - cols: Number of columns
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib tidak tersedia, skip preview grid")
        return False
    
    if not pairs:
        return False
    
    try:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        n = len(pairs)
        
        if matrix_view:
            rows = 2 * ((n + cols - 1) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
            if n == 1:
                axes = axes.reshape(1, -1)
            elif axes.ndim == 1:
                axes = axes.reshape(-1, cols)
            
            idx = 0
            for i in range(0, rows, 2):
                for j in range(cols):
                    if idx >= n:
                        break
                    before, after, label = pairs[idx]
                    
                    # Before image
                    if before.ndim == 3 and before.shape[2] == 3:
                        before_rgb = cv2.cvtColor((before * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    else:
                        before_rgb = (before * 255).astype(np.uint8)
                    
                    axes[i, j].imshow(before_rgb, cmap="gray" if before_rgb.ndim == 2 else None)
                    axes[i, j].set_title(f"{label} (before)", fontsize=8)
                    axes[i, j].axis("off")
                    
                    # After image
                    if after.ndim == 3 and after.shape[2] == 3:
                        after_rgb = cv2.cvtColor((after * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    else:
                        after_rgb = (after * 255).astype(np.uint8)
                    
                    axes[i + 1, j].imshow(after_rgb, cmap="gray" if after_rgb.ndim == 2 else None)
                    axes[i + 1, j].set_title(f"{label} (after)", fontsize=8)
                    axes[i + 1, j].axis("off")
                    
                    idx += 1
                
                if idx >= n:
                    for k in range(j + 1, cols):
                        axes[i, k].axis("off")
                        axes[i + 1, k].axis("off")
                    break
        else:
            rows = n
            fig, axes = plt.subplots(rows, 2, figsize=(8, 3 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (before, after, label) in enumerate(pairs):
                # Before
                if before.ndim == 3 and before.shape[2] == 3:
                    before_rgb = cv2.cvtColor((before * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    before_rgb = (before * 255).astype(np.uint8)
                
                axes[idx, 0].imshow(before_rgb, cmap="gray" if before_rgb.ndim == 2 else None)
                axes[idx, 0].set_title(f"{label} (before)")
                axes[idx, 0].axis("off")
                
                # After
                if after.ndim == 3 and after.shape[2] == 3:
                    after_rgb = cv2.cvtColor((after * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    after_rgb = (after * 255).astype(np.uint8)
                
                axes[idx, 1].imshow(after_rgb, cmap="gray" if after_rgb.ndim == 2 else None)
                axes[idx, 1].set_title(f"{label} (after)")
                axes[idx, 1].axis("off")
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  ✗ Error membuat preview: {e}")
        return False
