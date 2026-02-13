
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# CONFIG
VIDEOS = [
    str(BASE_DIR / "dataset" / "videos" / "rekamanCCTV-video1.mp4"),
    str(BASE_DIR / "dataset" / "videos" / "rekamanCCTV-video2.mp4"),
    str(BASE_DIR / "dataset" / "videos" / "rekamanCCTV-video3.mp4"),
]
MODEL_PATH = str(BASE_DIR / "dataset" / "model" / "mobilenetv3_final_finetuned.keras")
OUTPUT_DIR = str(BASE_DIR / "customer_demos")
CLASSES = ['Bekerja', 'Idle', 'Meninggalkan Area']
ROI_TOP, ROI_BOTTOM = 436, 935
ROI_LEFT, ROI_RIGHT = 450, 1282
IMG_SIZE = (224, 224)
SAMPLE_INTERVAL = 300 # Check every 10 seconds (approx) for speed
CLIP_DURATION_SEC = 15 
BATCH_SIZE = 32

def preprocess_frame_simple(frame):
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
    if roi.size == 0: roi = frame
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    img_array = resized.astype(np.float32)
    return img_array

def scan_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    events = [] 
    
    print(f"Scanning {os.path.basename(video_path)} ({total_frames} frames)...")
    
    batch_frames = []
    batch_indices = []
    
    start_time = time.time()
    
    for i in range(0, total_frames, SAMPLE_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        preprocessed = preprocess_frame_simple(frame)
        batch_frames.append(preprocessed)
        batch_indices.append(i)
        
        if len(batch_frames) >= BATCH_SIZE:
             batch_input = preprocess_input(np.array(batch_frames))
             preds = model.predict(batch_input, verbose=0)
             
             for j, pred in enumerate(preds):
                 idx = np.argmax(pred)
                 label = CLASSES[idx]
                 conf = pred[idx]
                 frame_idx = batch_indices[j]
                 
                 events.append({
                    'video': video_path,
                    'frame_idx': frame_idx,
                    'time_sec': frame_idx/fps,
                    'label': label,
                    'conf': conf,
                    'fps': fps
                 })
             
             batch_frames = []
             batch_indices = []
             
             elapsed = time.time() - start_time
             speed = (len(events)) / elapsed if elapsed > 0 else 0
             print(f"Scanned {i}/{total_frames} frames... ({speed:.1f} checks/sec)", end='\r')
    
    if batch_frames:
         batch_input = preprocess_input(np.array(batch_frames))
         preds = model.predict(batch_input, verbose=0)
         for j, pred in enumerate(preds):
             idx = np.argmax(pred)
             label = CLASSES[idx]
             conf = pred[idx]
             frame_idx = batch_indices[j]
             events.append({
                'video': video_path,
                'frame_idx': frame_idx,
                'time_sec': frame_idx/fps,
                'label': label,
                'conf': conf,
                'fps': fps
             })

    print(f"\nFinished {os.path.basename(video_path)}")
    cap.release()
    return events

def create_multi_view_clip(best_clips, output_path):
    # expect best_clips to be a dict {'Bekerja': clip, 'Idle': clip, 'Meninggalkan Area': clip}
    # We will arrange them in a layout.
    # Since we have 3, we can do 1 big left, 2 small right, or 3 vertical/horizontal columns.
    # Let's do 3 Horizontal columns.
    
    print("\nCreating Multi-View Compilation...")
    
    caps = []
    labels = []
    
    target_order = ['Bekerja', 'Idle', 'Meninggalkan Area']
    
    for lbl in target_order:
        if lbl in best_clips:
            clip = best_clips[lbl]
            cap = cv2.VideoCapture(clip['video'])
            fps = clip['fps']
            start_time = max(0, clip['time_sec'] - (CLIP_DURATION_SEC/2))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
            caps.append(cap)
            labels.append(lbl)
            print(f" - Added panel: {lbl} from {os.path.basename(clip['video'])}")
        else:
            print(f" - Missing clip for {lbl}")
            caps.append(None)
            labels.append(lbl)
            
    if all(c is None for c in caps):
        print("No clips to combined.")
        return

    # Assuming all videos have similar resolution, otherwise we resize
    # We'll target a standardized panel height
    TARGET_H = 480
    
    # Setup Output
    fps = 30 # standard output fps
    # Calculate width
    # 3 panels. 16:9 => 480p height => 854 width approx.
    # Total width = 854 * 3 = ~2560. Might be too wide.
    # Let's use smaller width per panel, say 400px.
    PANEL_W = 500
    PANEL_H = int(PANEL_W * 9 / 16) # 281
    
    TOTAL_W = PANEL_W * 3
    TOTAL_H = PANEL_H + 60 # +60 for labels header
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TOTAL_W, TOTAL_H))
    
    # Overlay colors
    COLORS = {
        'Bekerja': (0, 255, 0), # Green
        'Idle': (0, 215, 255), # Gold-ish (BGR)
        'Meninggalkan Area': (0, 0, 255) # Red
    }
    
    frames_to_write = int(CLIP_DURATION_SEC * fps)
    
    for i in range(frames_to_write):
        panel_imgs = []
        
        for j, cap in enumerate(caps):
            frame_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            label = labels[j]
            
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Resize
                    frame_img = cv2.resize(frame, (PANEL_W, PANEL_H))
                    
                    # Add border based on class
                    color = COLORS.get(label, (255, 255, 255))
                    cv2.rectangle(frame_img, (0,0), (PANEL_W-2, PANEL_H-2), color, 4)
                else:
                    # Loop if finished early? Or black.
                    pass
            
            panel_imgs.append(frame_img)
            
        # Stack images horizontally
        # Add Header area
        header = np.zeros((60, TOTAL_W, 3), dtype=np.uint8)
        header[:] = (30, 30, 30) # Dark gray background
        
        # Draw labels on header
        for k, lbl in enumerate(labels):
             text_color = COLORS.get(lbl, (255, 255, 255))
             x_pos = k * PANEL_W + 20
             cv2.putText(header, lbl, (x_pos, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Combine panels
        row = np.hstack(panel_imgs)
        final_frame = np.vstack([header, row])
        
        out.write(final_frame)
        
    for c in caps: 
        if c: c.release()
    out.release()
    
    if os.path.exists(output_path):
        print(f"Created Multi-View Clip: {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    all_events = []
    
    for v in VIDEOS:
        if not os.path.exists(v):
            print(f"Video not found: {v}")
            continue
        events = scan_video(v, model)
        all_events.extend(events)
        
    targets = ['Bekerja', 'Idle', 'Meninggalkan Area']
    selected_clips = []
    best_clips_dict = {}
    
    # 1. Best per class
    for cls in targets:
        candidates = [e for e in all_events if e['label'] == cls]
        candidates.sort(key=lambda x: x['conf'], reverse=True)
        if candidates:
            best = candidates[0]
            selected_clips.append(best)
            best_clips_dict[cls] = best
            print(f"Best {cls}: {best['conf']:.2f}")

    # 2. Fill to 5
    sorted_all = sorted(all_events, key=lambda x: x['conf'], reverse=True)
    
    for e in sorted_all:
        if len(selected_clips) >= 5: break
        
        is_overlapping = False
        for s in selected_clips:
            if s['video'] == e['video'] and abs(s['time_sec'] - e['time_sec']) < 30:
                is_overlapping = True
                break
        
        if not is_overlapping:
            selected_clips.append(e)

    # Extract Individual Clips
    print(f"\nExtracting {len(selected_clips)} clips...")
    for idx, clip in enumerate(selected_clips):
        v_path = clip['video']
        fps = clip['fps']
        
        start_time = max(0, clip['time_sec'] - (CLIP_DURATION_SEC/2))
        start_frame = int(start_time * fps)
        end_frame = start_frame + int(CLIP_DURATION_SEC * fps)
        
        cap = cv2.VideoCapture(v_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        w = int(cap.get(3))
        h = int(cap.get(4))
        
        safe_label = clip['label'].replace(" ", "")
        out_name = f"Demo_{idx+1}_{safe_label}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        curr = start_frame
        while curr < end_frame:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            curr += 1
            
        cap.release()
        out.release()
    
    # 3. Create Multi-View Compilation
    if len(best_clips_dict) >= 3:
        create_multi_view_clip(best_clips_dict, os.path.join(OUTPUT_DIR, "Demo_Multi_Compilation.mp4"))
    else:
        print("Not enough classes found for multi-view (Need Bekerja, Idle, Meninggalkan Area)")

if __name__ == "__main__":
    main()
