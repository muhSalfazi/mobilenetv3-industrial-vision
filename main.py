# main.py
from pipeline.extract_frames import extract_frames
from pipeline.filter_frames import filter_frames
from pipeline.preprocess_augment import resize_and_augment
from pipeline.split_dataset import split_dataset
import subprocess
import os

VIDEO_DIR = "videos/"
RAW_DIR = "frames_raw/"
FILTER_DIR = "frames_filtered/"
AUG_DIR = "dataset_augmented/"
DATASET_DIR = "dataset/"
OUT_DIR = "output/"

def run_labeler():
    # jalankan labeler sebagai proses terpisah (GUI blocking)
    subprocess.run(["python3", "labeling_gui.py"], check=True)

def main():
    print("\n=== DATA PIPELINE START ===")
    # print("\n[1] Extracting frames...")
    # extract_frames(VIDEO_DIR, RAW_DIR, interval=10)

    # print("\n[2] Filtering frames...")
    # filter_frames(RAW_DIR, FILTER_DIR)

    print("\n[3] Start labeling GUI (manual/semi-auto)...")
    run_labeler()

    print("\n[4] Resize + augment (OpenCV)...")
    # proses augment akan membaca dataset/working, dataset/idle, dataset/out_area
    resize_and_augment(DATASET_DIR, AUG_DIR)

    print("\n[5] Split dataset...")
    split_dataset(AUG_DIR, OUT_DIR)

    print("\n=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
