import cv2
import os
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
demo_dir = "demo"
videos = sorted([f for f in os.listdir(demo_dir) if f.endswith(".mp4")])

print(f"{'Video File':<40} | {'Detected Text'}")
print("-" * 80)

for v in videos:
    path = os.path.join(demo_dir, v)
    cap = cv2.VideoCapture(path)

    # Check middle frame as well to be sure
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2 if frame_count > 0 else 0)

    ret, frame = cap.read()
    cap.release()

    found_texts = []
    if ret:
        result, _ = engine(frame)
        if result:
            for res in result:
                txt = res[1]
                if len(txt) > 2:
                    found_texts.append(txt)

    text_display = " | ".join(found_texts) if found_texts else "None"
    print(f"{v:<40} | {text_display}")
