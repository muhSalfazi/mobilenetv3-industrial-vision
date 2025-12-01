import cv2
import os

def extract_frames(video_dir, out_dir, interval=10):
    os.makedirs(out_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_name = os.path.splitext(video_file)[0].replace(" ", "_").lower()

        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        print(f"[INFO] Processing: {video_file}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # Save name: <videoName>_frame_00001.jpg
                save_name = f"{video_name}_frame_{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(out_dir, save_name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"[DONE] Saved {saved_count} frames from {video_file}")
