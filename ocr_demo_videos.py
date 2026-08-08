import cv2
import os
import sys
from rapidocr_onnxruntime import RapidOCR
import numpy as np


def ocr_video(video_path, sample_rate_seconds=2.0):
    print(f"Analyzing: {os.path.basename(video_path)}", flush=True)
    try:
        engine = RapidOCR()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = int(fps * sample_rate_seconds)
        frame_idx = 0
        all_text = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                print(f"  Frame {frame_idx}/{total_frames}...", end="\r", flush=True)
                result, _ = engine(frame)
                if result:
                    for res in result:
                        text = res[1]
                        if len(text.strip()) > 2:
                            all_text.add(text.strip())

            frame_idx += 1

        cap.release()
        print(f"\n  Found {len(all_text)} text snippets.", flush=True)
        return sorted(list(all_text))
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}", flush=True)
        return []


def main():
    demo_dir = "/media/Windows/project-python/Project-TA/demo"
    video_files = [f for f in os.listdir(demo_dir) if f.endswith(".mp4")]

    results = {}
    # Run only for the first two videos for now to check
    for video in video_files:
        path = os.path.join(demo_dir, video)
        results[video] = ocr_video(path)

    print("\n" + "=" * 50, flush=True)
    print("OCR RESULTS SUMMARY", flush=True)
    print("=" * 50, flush=True)
    for video, texts in results.items():
        print(f"\nVIDEO: {video}", flush=True)
        if texts:
            # Group similar texts or just print them
            for t in texts:
                print(f"  - {t}", flush=True)
        else:
            print("  (No text detected)", flush=True)


if __name__ == "__main__":
    main()
