#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TIMESTAMP_PATTERNS = [
    re.compile(r"(?P<y>\d{4})[-/](?P<m>\d{2})[-/](?P<d>\d{2})\s+(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})"),
    re.compile(r"(?P<d>\d{2})[-/](?P<m>\d{2})[-/](?P<y>\d{4})\s+(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})"),
    re.compile(r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def try_load_pytesseract():
    if shutil.which("tesseract") is None:
        return None
    try:
        import pytesseract  # type: ignore
    except Exception:
        return None
    return pytesseract


def try_load_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def try_load_ocr_backend() -> tuple[str, Any]:
    pytesseract_module = try_load_pytesseract()
    if pytesseract_module is not None:
        return "pytesseract", pytesseract_module
    rapidocr_engine = try_load_rapidocr()
    if rapidocr_engine is not None:
        return "rapidocr", rapidocr_engine
    return "none", None


def parse_timestamp_text(text: str) -> str | None:
    clean = " ".join((text or "").replace("\n", " ").split())
    for pattern in TIMESTAMP_PATTERNS:
        m = pattern.search(clean)
        if not m:
            continue
        g = m.groupdict()
        if {"y", "m", "d", "h", "mi", "s"}.issubset(g):
            return f"{g['y']}-{g['m']}-{g['d']} {g['h']}:{g['mi']}:{g['s']}"
        if {"d", "m", "y", "h", "mi", "s"}.issubset(g):
            return f"{g['y']}-{g['m']}-{g['d']} {g['h']}:{g['mi']}:{g['s']}"
        if {"h", "mi", "s"}.issubset(g):
            return f"{g['h']}:{g['mi']}:{g['s']}"
    return None


def fourcc_to_str(fourcc_val: float) -> str:
    code = int(fourcc_val)
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()


def preprocess_for_ocr(img: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return [gray, th1, th2]


def _ocr_timestamp_from_frame_tesseract(frame: np.ndarray, pytesseract_module) -> dict[str, Any]:
    h, w = frame.shape[:2]
    regions = {
        "top_left": frame[0:int(h * 0.22), 0:int(w * 0.52)],
        "top_right": frame[0:int(h * 0.22), int(w * 0.48):w],
        "bottom_left": frame[int(h * 0.78):h, 0:int(w * 0.6)],
        "bottom_right": frame[int(h * 0.78):h, int(w * 0.4):w],
        "top_bar": frame[0:int(h * 0.16), :],
        "bottom_bar": frame[int(h * 0.84):h, :],
    }
    best = {"raw_text": "", "parsed": None, "conf": -1.0, "region": ""}
    config = "--psm 7 -c tessedit_char_whitelist=0123456789:/-. "

    for region_name, crop in regions.items():
        if crop.size == 0:
            continue
        for candidate in preprocess_for_ocr(crop):
            text = pytesseract_module.image_to_string(candidate, config=config)
            parsed = parse_timestamp_text(text)
            if not parsed:
                continue
            data = pytesseract_module.image_to_data(candidate, config=config, output_type=pytesseract_module.Output.DICT)
            conf_values = []
            for c in data.get("conf", []):
                try:
                    v = float(c)
                except Exception:
                    continue
                if v >= 0:
                    conf_values.append(v)
            conf = float(np.mean(conf_values)) if conf_values else 0.0
            if conf > best["conf"]:
                best = {"raw_text": text.strip(), "parsed": parsed, "conf": conf, "region": region_name}

    return best


def _ocr_timestamp_from_frame_rapidocr(frame: np.ndarray, rapidocr_engine) -> dict[str, Any]:
    h, w = frame.shape[:2]
    regions = {
        "top_left": frame[0:int(h * 0.22), 0:int(w * 0.52)],
        "top_right": frame[0:int(h * 0.22), int(w * 0.48):w],
        "bottom_left": frame[int(h * 0.78):h, 0:int(w * 0.6)],
        "bottom_right": frame[int(h * 0.78):h, int(w * 0.4):w],
    }
    best = {"raw_text": "", "parsed": None, "conf": -1.0, "region": ""}

    for region_name, crop in regions.items():
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates = [gray, th1]
        for candidate in candidates:
            in_img = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR) if candidate.ndim == 2 else candidate
            if in_img.shape[1] > 960:
                scale = 960.0 / in_img.shape[1]
                nh = max(1, int(in_img.shape[0] * scale))
                in_img = cv2.resize(in_img, (960, nh), interpolation=cv2.INTER_AREA)
            out = rapidocr_engine(in_img)
            if not out:
                continue
            rec_res = out[0] if isinstance(out, (tuple, list)) else None
            if not rec_res:
                continue
            texts = []
            confs = []
            for item in rec_res:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                texts.append(str(item[1]))
                try:
                    confs.append(float(item[2]))
                except Exception:
                    pass
            if not texts:
                continue
            raw_text = " ".join(texts)
            parsed = parse_timestamp_text(raw_text)
            if not parsed:
                continue
            conf = float(np.mean(confs)) if confs else 0.0
            if conf > best["conf"]:
                best = {"raw_text": raw_text, "parsed": parsed, "conf": conf, "region": region_name}
            if best["parsed"] and best["conf"] >= 0.9:
                return best

    return best


def ocr_timestamp_from_frame(frame: np.ndarray, backend_name: str, backend_obj: Any) -> dict[str, Any]:
    if backend_name == "pytesseract":
        return _ocr_timestamp_from_frame_tesseract(frame, backend_obj)
    if backend_name == "rapidocr":
        return _ocr_timestamp_from_frame_rapidocr(frame, backend_obj)
    return {"raw_text": "", "parsed": None, "conf": -1.0, "region": ""}


def save_clock_panel(frame: np.ndarray, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    h, w = frame.shape[:2]
    tl = frame[0:int(h * 0.22), 0:int(w * 0.52)]
    tr = frame[0:int(h * 0.22), int(w * 0.48):w]
    bl = frame[int(h * 0.78):h, 0:int(w * 0.6)]
    br = frame[int(h * 0.78):h, int(w * 0.4):w]

    def fit(crop: np.ndarray, width: int, height: int) -> np.ndarray:
        if crop.size == 0:
            return np.full((height, width, 3), 255, dtype=np.uint8)
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

    tile_w, tile_h = 500, 120
    tiles = [
        ("Top-Left", fit(tl, tile_w, tile_h)),
        ("Top-Right", fit(tr, tile_w, tile_h)),
        ("Bottom-Left", fit(bl, tile_w, tile_h)),
        ("Bottom-Right", fit(br, tile_w, tile_h)),
    ]
    rows = []
    for i in range(0, len(tiles), 2):
        block = []
        for label, tile in tiles[i:i + 2]:
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (80, 80, 80), 1)
            cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            block.append(tile)
        rows.append(np.hstack(block))
    panel = np.vstack(rows)
    header = np.full((52, panel.shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(header, title, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    out = np.vstack([header, panel])
    cv2.imwrite(str(out_path), out)


def read_frame_at_second(cap: cv2.VideoCapture, sec: float, fps: float, total_frames: int) -> np.ndarray | None:
    if total_frames <= 0:
        return None
    if fps <= 0:
        idx = 0
    else:
        idx = int(round(sec * fps))
    idx = max(0, min(total_frames - 1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def compute_clock_delta(start_text: str | None, end_text: str | None) -> tuple[float | None, str]:
    if not start_text or not end_text:
        return None, "timestamp_cctv_tidak_lengkap"

    dt_format = "%Y-%m-%d %H:%M:%S"
    t_format = "%H:%M:%S"
    try:
        if len(start_text) == 19 and len(end_text) == 19:
            start_dt = datetime.strptime(start_text, dt_format)
            end_dt = datetime.strptime(end_text, dt_format)
            return (end_dt - start_dt).total_seconds(), "berdasarkan_datetime_overlay"
        if len(start_text) == 8 and len(end_text) == 8:
            base = datetime.strptime("2000-01-01 " + start_text, "%Y-%m-%d " + t_format)
            end = datetime.strptime("2000-01-01 " + end_text, "%Y-%m-%d " + t_format)
            if end < base:
                end = end + timedelta(days=1)
            return (end - base).total_seconds(), "berdasarkan_jam_overlay"
    except Exception:
        return None, "format_timestamp_overlay_tidak_valid"
    return None, "format_timestamp_overlay_tidak_dikenal"


def analyze_video(video_path: Path, out_dir: Path, ocr_backend_name: str, ocr_backend_obj: Any) -> dict[str, Any]:
    file_stat = video_path.stat()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_file": video_path.name,
            "video_path": str(video_path),
            "error": "Video tidak bisa dibuka",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    codec = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    duration_sec = (total_frames / fps) if fps > 0 else 0.0

    samples = [("start", 0.0), ("middle", duration_sec / 2.0 if duration_sec > 0 else 0.0), ("end", max(0.0, duration_sec - (1.0 / max(fps, 1.0))))]
    cctv_times: dict[str, str | None] = {"start": None, "middle": None, "end": None}
    ocr_regions: dict[str, str] = {"start": "", "middle": "", "end": ""}
    ocr_conf: dict[str, float] = {"start": -1.0, "middle": -1.0, "end": -1.0}

    for label, sec in samples:
        frame = read_frame_at_second(cap, sec, fps, total_frames)
        if frame is None:
            continue
        panel_path = out_dir / "clock_panels" / f"{video_path.stem}_{label}.jpg"
        save_clock_panel(frame, panel_path, f"{video_path.name} | {label.upper()} @ {format_hms(sec)}")
        if ocr_backend_name != "none":
            ocr = ocr_timestamp_from_frame(frame, ocr_backend_name, ocr_backend_obj)
            cctv_times[label] = ocr["parsed"]
            ocr_regions[label] = ocr["region"]
            ocr_conf[label] = round(float(ocr["conf"]), 2)

    cap.release()

    clock_delta, clock_note = compute_clock_delta(cctv_times["start"], cctv_times["end"])

    row = {
        "video_file": video_path.name,
        "video_path": str(video_path),
        "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
        "file_modified_time": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": f"{width}x{height}",
        "width_px": width,
        "height_px": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 3),
        "duration_hms": format_hms(duration_sec),
        "codec_fourcc": codec,
        "cctv_time_start": cctv_times["start"] or "",
        "cctv_time_middle": cctv_times["middle"] or "",
        "cctv_time_end": cctv_times["end"] or "",
        "cctv_clock_duration_seconds": round(clock_delta, 3) if clock_delta is not None else "",
        "cctv_clock_duration_hms": format_hms(clock_delta) if clock_delta is not None else "",
        "cctv_clock_note": clock_note,
        "ocr_region_start": ocr_regions["start"],
        "ocr_region_middle": ocr_regions["middle"],
        "ocr_region_end": ocr_regions["end"],
        "ocr_conf_start": ocr_conf["start"] if ocr_conf["start"] >= 0 else "",
        "ocr_conf_middle": ocr_conf["middle"] if ocr_conf["middle"] >= 0 else "",
        "ocr_conf_end": ocr_conf["end"] if ocr_conf["end"] >= 0 else "",
        "ocr_backend": ocr_backend_name,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Laporan spesifikasi video CCTV (durasi, resolusi, fps, codec, dan jam overlay CCTV jika OCR tersedia)."
    )
    parser.add_argument(
        "--video-glob",
        default="dataset/videos/rekamanCCTV-video*.mp4",
        help="Pola file video (default: dataset/videos/rekamanCCTV-video*.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/stats/cctv_specs"),
        help="Folder output laporan",
    )
    args = parser.parse_args()

    videos = sorted(Path(".").glob(args.video_glob))
    ensure_dir(args.output_dir)
    ocr_backend_name, ocr_backend_obj = try_load_ocr_backend()

    rows = []
    for video in videos:
        print(f"Analisis: {video}", flush=True)
        row = analyze_video(video, args.output_dir, ocr_backend_name, ocr_backend_obj)
        rows.append(row)

    csv_path = args.output_dir / "cctv_video_specs.csv"
    json_path = Path("data/input/cctv_specs/cctv_video_specs.json")
    ensure_dir(json_path.parent)
    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nSelesai. Total video: {len(rows)}")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Panel jam CCTV (crop): {args.output_dir / 'clock_panels'}")
    if ocr_backend_name == "none":
        print("Catatan: OCR jam CCTV belum aktif (install `pytesseract` + binary `tesseract`, atau `rapidocr-onnxruntime`).")
    else:
        print(f"OCR aktif dengan backend: {ocr_backend_name}")


if __name__ == "__main__":
    main()
