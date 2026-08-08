#!/usr/bin/env python3
"""
Export video dengan hanya frame yang terdeteksi sebagai 'Idle' oleh model MobileNetV3.

Script ini:
1. Load model mobilenet_best.keras
2. Process video frame per frame
3. Filter frame yang diprediksi sebagai 'Idle'
4. Buat video baru dengan hanya frame idle
5. Simpan ke folder demo/
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from tqdm import tqdm
import subprocess
import tempfile
import os

# ============= KONFIGURASI =============
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Naik ke Project-TA
MODEL_PATH = PROJECT_ROOT / "dataset" / "model" / "mobilenet_best.keras"
VIDEO_INPUT_DIR = PROJECT_ROOT / "dataset" / "videos"
DEMO_OUTPUT_DIR = PROJECT_ROOT / "demo"
DEMO_OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
CLASSES = ["Bekerja", "Idle", "Meninggalkan Area"]

# Koordinat ROI MURNI sesuai dengan web.py (Wajib Tepat untuk Validitas TA)
ROI_LEFT = 545
ROI_RIGHT = 1090
ROI_TOP = 527
ROI_BOTTOM = 991


# ============= FUNGSI UTILITY =============
def load_model_safe():
    """Load model dengan error handling."""
    if not MODEL_PATH.exists():
        print(f"❌ Model tidak ditemukan: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"✅ Model loaded: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_frame(frame):
    """
    Preprocess frame untuk model MobileNetV3.
    PENTING: Match training preprocessing (rescale=1./255, NOT preprocess_input)
    BGR -> RGB -> Resize -> normalize [0,1]
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    img_array = resized.astype(np.float32) / 255.0  # Match training rescale
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_activity(frame, model):
    """
    Predict aktivitas dari frame dengan ROI yang sama dengan web.py.
    Returns: (label, confidence)
    """
    try:
        h_vid, w_vid = frame.shape[:2]
        r_top = max(0, min(ROI_TOP, h_vid - 1))
        r_bot = max(r_top + 1, min(ROI_BOTTOM, h_vid))
        r_left = max(0, min(ROI_LEFT, w_vid - 1))
        r_right = max(r_left + 1, min(ROI_RIGHT, w_vid))

        roi_frame = frame[r_top:r_bot, r_left:r_right]
        if roi_frame.size == 0:
            return "Meninggalkan Area", 1.0

        processed = preprocess_frame(roi_frame)
        preds = model.predict(processed, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]
        label = CLASSES[idx]
        return label, confidence
    except Exception as e:
        print(f"⚠️  Error predicting: {e}")
        return None, 0.0


def export_idle_frames_from_video(
    video_path, model, output_path, min_confidence=0.5, sample_rate=1, max_seconds=30
):
    """
    Extract frame yang diprediksi sebagai 'Idle' dan buat video baru.
    Menggunakan MJPG codec (Motion JPEG) - lebih reliable tanpa FFmpeg.

    Args:
        video_path: Path ke input video
        model: Loaded TensorFlow model
        output_path: Path untuk output video (akan save sebagai .avi dengan MJPG)
        min_confidence: Minimum confidence threshold
        sample_rate: Proses setiap N frame (untuk speed up)
        max_seconds: Batas MAXIMUM durasi video yang dihasilkan.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Tidak bisa buka video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fallback if FPS is 0
    if fps <= 0:
        fps = 30.0

    MAX_OUTPUT_FRAMES = int(fps * max_seconds)

    print(f"\n📹 Processing: {video_path.name}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Maksimum Output: {max_seconds} Detik ({MAX_OUTPUT_FRAMES} Frame)")

    # Output akan disimpan sebagai AVI dengan MJPEG codec
    # MJPEG adalah format paling reliable di OpenCV
    output_avi_path = output_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(
        str(output_avi_path), fourcc, fps, (width, height), isColor=True
    )

    if not out.isOpened():
        print(f"❌ Tidak bisa buat video writer untuk {output_avi_path}!")
        cap.release()
        return False

    frame_idx = 0
    idle_count = 0
    total_processed = 0

    print("⏳ Memproses frame & menulis video (Filter IDLE, Max 30 detik)...")
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idle_count >= MAX_OUTPUT_FRAMES:
                print(
                    f"\n🛑 Batas durasi {max_seconds} detik tercapai. Ekspor dihentikan dengan sukses."
                )
                break

            # Process setiap N-th frame
            if frame_idx % sample_rate == 0:
                label, confidence = predict_activity(frame, model)
                total_processed += 1

                # Jika hasil adalah 'Idle' dan confidence cukup
                if label == "Idle" and confidence >= min_confidence:
                    # Gambarkan letak ROI hijau yang dipakai agar video kelihatan profesional
                    cv2.rectangle(
                        frame,
                        (ROI_LEFT, ROI_TOP),
                        (ROI_RIGHT, ROI_BOTTOM),
                        (0, 255, 0),
                        2,
                    )

                    # Add info text ke frame
                    text = f"IDLE - Confidence: {confidence:.2%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        frame,
                        text,
                        (ROI_LEFT + 5, max(30, ROI_TOP - 10)),
                        font,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                    # Write frame ke output video
                    if not out.write(frame):
                        print(f"⚠️  Warning: Error writing frame {frame_idx}")
                    else:
                        idle_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    # Force flush ke disk
    import subprocess

    subprocess.run(["sync"], check=False)

    print(f"\n✅ Export selesai!")
    print(f"   Total frame diproses: {total_processed}")
    print(f"   Frame IDLE ditemukan: {idle_count}")
    print(f"   Output video: {output_avi_path}")

    # Verifikasi output
    if output_avi_path.exists():
        file_size = output_avi_path.stat().st_size / (1024 * 1024)  # MB
        cap_check = cv2.VideoCapture(str(output_avi_path))
        if cap_check.isOpened():
            check_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            check_fps = cap_check.get(cv2.CAP_PROP_FPS)
            cap_check.release()
            print(
                f"   ✓ Verifikasi: {check_frames} frames, {check_fps:.1f} fps, {file_size:.1f} MB"
            )
        else:
            print(
                f"⚠️  Warning: Output file tidak bisa dibaca (kemungkinan masih ditulis)"
            )

    if idle_count == 0:
        print("⚠️  Tidak ada frame IDLE yang ditemukan!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export video dengan hanya frame 'Idle' dari model MobileNetV3"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path ke input video (default: scan semua video di dataset/videos/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Nama output video (default: <input_name>_Idle_Export.avi)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=1,
        help="Process setiap N frame (default: 1 = semua frame)",
    )

    args = parser.parse_args()

    # ============= LOAD MODEL =============
    model = load_model_safe()
    if model is None:
        return

    # ============= PROCESS VIDEO(S) =============
    if args.input:
        # Process single video
        video_path = Path(args.input)
        if not video_path.exists():
            print(f"❌ Video tidak ditemukan: {video_path}")
            return

        output_name = args.output or f"{video_path.stem}_Idle_Export.avi"
        output_path = DEMO_OUTPUT_DIR / output_name

        export_idle_frames_from_video(
            video_path,
            model,
            output_path,
            min_confidence=args.confidence,
            sample_rate=args.sample_rate,
        )
    else:
        # Process semua video di dataset/videos/
        if not VIDEO_INPUT_DIR.exists():
            print(f"❌ Input directory tidak ditemukan: {VIDEO_INPUT_DIR}")
            return

        video_files = list(VIDEO_INPUT_DIR.glob("*.mp4")) + list(
            VIDEO_INPUT_DIR.glob("*.avi")
        )

        if not video_files:
            print(f"⚠️  Tidak ada video ditemukan di {VIDEO_INPUT_DIR}")
            return

        print(f"🎬 Ditemukan {len(video_files)} video(s)")

        for video_path in video_files:
            output_name = f"{video_path.stem}_Idle_Export.avi"
            output_path = DEMO_OUTPUT_DIR / output_name

            export_idle_frames_from_video(
                video_path,
                model,
                output_path,
                min_confidence=args.confidence,
                sample_rate=args.sample_rate,
            )

    print("\n" + "=" * 60)
    print("✅ Semua video telah diproses!")
    print(f"📁 Output disimpan di: {DEMO_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
