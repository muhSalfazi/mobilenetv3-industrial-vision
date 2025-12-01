import cv2
import os
import numpy as np

def fast_dhash(image, size=8):
    # Resize kecil untuk hashing (sangat cepat)
    img = cv2.resize(image, (size + 1, size))
    diff = img[:, 1:] > img[:, :-1]
    return diff.astype(np.uint8).tobytes()  # jauh lebih cepat dari sha1

def filter_frames(in_dir, out_dir, blur_thr=100.0, bright_min=40, bright_max=220):
    os.makedirs(out_dir, exist_ok=True)

    seen_hashes = set()
    save_index = 0  # index file output

    print("[INFO] Filtering started...")

    for file in sorted(os.listdir(in_dir)):
        if not file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(in_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Blur check (Laplacian variance)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thr:
            continue

        # 2. Brightness check
        brightness = np.mean(gray)
        if brightness < bright_min or brightness > bright_max:
            continue

        # 3. Duplicate check (super fast)
        h = fast_dhash(gray)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # 4. Save immediately (light memory use)
        out_name = f"filtered_{save_index:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), img)
        save_index += 1

    print(f"[DONE] {save_index} frames saved after filtering.")
