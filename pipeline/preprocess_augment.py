# pipeline/preprocess_augment.py
import os
import cv2
import numpy as np
import random

def random_brightness(img, factor_range=(0.8, 1.2)):
    factor = random.uniform(*factor_range)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def random_flip(img):
    if random.random() < 0.5:
        return cv2.flip(img, 1)
    return img

def random_rotate(img, angle_range=(-5, 5)):
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_shift(img, shift_range=0.05):
    h, w = img.shape[:2]
    max_shift_x = int(w * shift_range)
    max_shift_y = int(h * shift_range)
    tx = random.randint(-max_shift_x, max_shift_x)
    ty = random.randint(-max_shift_y, max_shift_y)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_zoom(img, zoom_range=0.05):
    h, w = img.shape[:2]
    zoom = 1 + random.uniform(-zoom_range, zoom_range)
    nh, nw = int(h * zoom), int(w * zoom)
    resized = cv2.resize(img, (nw, nh))
    if zoom > 1:
        start_x = (nw - w) // 2
        start_y = (nh - h) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        canvas = np.zeros_like(img)
        start_x = (w - nw) // 2
        start_y = (h - nh) // 2
        canvas[start_y:start_y+nh, start_x:start_x+nw] = resized
        return canvas

def augment_image(img):
    img = random_brightness(img)
    img = random_flip(img)
    img = random_rotate(img)
    img = random_shift(img)
    img = random_zoom(img)
    return img

def resize_and_augment(in_dir, out_dir, target_size=(224, 224), aug_count=3):
    os.makedirs(out_dir, exist_ok=True)
    for cls in os.listdir(in_dir):
        cls_path = os.path.join(in_dir, cls)
        out_cls = os.path.join(out_dir, cls)
        os.makedirs(out_cls, exist_ok=True)
        for file in os.listdir(cls_path):
            if not file.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            resized = cv2.resize(img, target_size)
            base_name = os.path.splitext(file)[0]
            cv2.imwrite(os.path.join(out_cls, f"{base_name}.jpg"), resized)
            for i in range(aug_count):
                aug = augment_image(resized)
                cv2.imwrite(os.path.join(out_cls, f"{base_name}_aug{i}.jpg"), aug)
    print("[DONE] Resize + augment complete (OpenCV).")
