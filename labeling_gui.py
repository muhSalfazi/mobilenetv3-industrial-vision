import os
import shutil
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
import json
import cv2
import numpy as np

SOURCE_DIR = "frames_filtered"
OUTPUT_DIR = "dataset"
ROI_CONFIG = ".roi.json"

LABELS = {"1": "berkerja", "2": "idle", "3": "out_area"}

# ========== Create Output Folders ==========
for lbl in LABELS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, lbl), exist_ok=True)

files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]
files.sort()

index = 0
img_tk = None
polygon = []
polygon_draw_mode = False
prev_frame = None
auto_suggestion = ""
worker_point = None


# =====================================================
# ROI SAVE / LOAD
# =====================================================
def save_roi(polygon_points):
    with open(ROI_CONFIG, "w") as f:
        json.dump({"polygon": polygon_points}, f)

def load_roi():
    if os.path.exists(ROI_CONFIG):
        with open(ROI_CONFIG, "r") as f:
            return json.load(f).get("polygon", [])
    return []


# =====================================================
# POLYGON CHECK
# =====================================================
def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and \
            (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


# =====================================================
# AUTO LABEL MOTION
# =====================================================
def compute_motion(current_img):
    global prev_frame
    if prev_frame is None:
        prev_frame = current_img
        return 0

    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(g1, g2)
    score = np.sum(diff) / 10000
    prev_frame = current_img

    return score

def auto_label_suggestion(current_img):
    global worker_point
    roi = load_roi()

    motion = compute_motion(current_img)

    # ROI check
    if worker_point and roi:
        if not point_in_polygon(worker_point[0], worker_point[1], roi):
            return "out_area", "Auto: keluar area kerja"

    # Motion thresholds
    if motion > 8:
        return "berkerja", f"Auto: BEKERJA (motion={motion:.1f})"
    elif motion < 2:
        return "idle", f"Auto: IDLE (motion={motion:.1f})"
    return "", f"Auto: tidak pasti (motion={motion:.1f})"


# =====================================================
# LOAD IMAGE
# =====================================================
def load_image():
    global index, img_tk, auto_suggestion

    if index >= len(files):
        messagebox.showinfo("Selesai", "Semua gambar telah selesai dilabeli.")
        root.quit()
        return

    img_path = os.path.join(SOURCE_DIR, files[index])

    # ---- Load PIL preview (900×600) ----
    pil_img = Image.open(img_path)
    pil_img = pil_img.resize((900, 600))
    img_tk = ImageTk.PhotoImage(pil_img)
    img_label.config(image=img_tk)

    status.config(text=f"{index+1}/{len(files)} - {files[index]}")

    # ---- Auto Label ----
    cv_img = cv2.imread(img_path)
    cv_img = cv2.resize(cv_img, (900, 600))

    suggested, text = auto_label_suggestion(cv_img)
    auto_suggestion = suggested
    auto_label_box.config(text=text)

    redraw_overlay()


# =====================================================
# SAVE LABELED IMAGE (224×224)
# =====================================================
def label_image(label_name):
    global index

    fname = files[index]
    src = os.path.join(SOURCE_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, label_name, fname)

    img = cv2.imread(src)
    img224 = cv2.resize(img, (224, 224))
    cv2.imwrite(dst, img224)

    print(f"[SAVED] {fname} → {label_name}")

    index += 1
    load_image()


def accept_auto_label():
    if auto_suggestion:
        label_image(auto_suggestion)

def skip_image():
    global index
    index += 1
    load_image()

def undo_last():
    global index
    prev = index - 1
    if prev < 0:
        return

    for lbl in LABELS.values():
        path = os.path.join(OUTPUT_DIR, lbl, files[prev])
        if os.path.exists(path):
            shutil.move(path, os.path.join(SOURCE_DIR, files[prev]))
            index = prev
            load_image()
            print(f"[UNDO] restored {files[prev]}")
            break


# =====================================================
# EVENTS
# =====================================================
def on_key(event):
    key = event.char
    if key in LABELS:
        label_image(LABELS[key])
    elif key == " ":
        skip_image()
    elif key.lower() == "a":
        accept_auto_label()
    elif key.lower() == "p":
        toggle_polygon_mode()
    elif key.lower() == "s":
        save_roi(polygon)
    elif key.lower() == "u":
        undo_last()

def on_click(event):
    global polygon, worker_point

    x, y = event.x, event.y

    if polygon_draw_mode:
        polygon.append((x, y))
        redraw_overlay()
    else:
        worker_point = (x, y)
        auto_label_box.config(text=f"Worker point: {worker_point}")


def toggle_polygon_mode():
    global polygon_draw_mode
    polygon_draw_mode = not polygon_draw_mode
    instruction.config(
        text="Polygon Mode ON — klik titik ROI"
        if polygon_draw_mode else
        "Polygon Mode OFF"
    )


# =====================================================
# DRAW OVERLAY
# =====================================================
def redraw_overlay():
    canvas.delete("all")
    poly = polygon if polygon else load_roi()

    if poly and len(poly) >= 2:
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            canvas.create_line(x1, y1, x2, y2, fill="red", width=2)


# =====================================================
# GUI SETUP — FIXED FOR UBUNTU
# =====================================================
root = tk.Tk()
root.geometry("960x780")
root.title("Labeling GUI + Auto Pre-Label | 224×224 Output")

instruction = Label(root, text="1=Bekerja | 2=Idle | 3=Out Area | A=Auto | P=Polygon | U=Undo | Space=Skip",
                    font=("Arial", 12))
instruction.pack()

auto_label_box = Label(root, text="Auto: ...", font=("Arial", 12), fg="blue")
auto_label_box.pack()

status = Label(root, text="", font=("Arial", 12))
status.pack()

frame_view = tk.Frame(root)
frame_view.pack()

# ---- Gambar ----
img_label = Label(frame_view)
img_label.grid(row=0, column=0)

# ---- Canvas Overlay Tidak Menutupi Gambar ----
canvas = tk.Canvas(frame_view, width=900, height=600,
                   highlightthickness=0)
canvas.grid(row=0, column=0)

# Bind events
root.bind("<Key>", on_key)
canvas.bind("<Button-1>", on_click)

# Load ROI + first image
polygon = load_roi()
load_image()

root.mainloop()
