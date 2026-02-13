
import cv2
import numpy as np
import os

OUTPUT_DIR = "/media/Windows/project-python/Pre-Processing-ProjectTA/demo"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Demo_Multi_Compilation.mp4")

# Files we found in the directory
CLIPS = {
    'Bekerja': os.path.join(OUTPUT_DIR, "Final_Bekerja.mp4"),
    'Idle': os.path.join(OUTPUT_DIR, "Final_Idle_Weak.mp4"),
    'Meninggalkan Area': os.path.join(OUTPUT_DIR, "Final_MeninggalkanArea.mp4")
}


def create_compilation():
    print("Stitching clips sequentially (Timeline mode)...")
    
    # We will play them one after another: Bekerja -> Idle -> Meninggalkan Area
    playlist = ['Bekerja', 'Idle', 'Meninggalkan Area']
    
    # 1. Get properties from the first available video to set the output format
    first_cap = None
    width, height, fps = 0, 0, 30.0
    
    valid_clips = []
    
    for lbl in playlist:
        path = CLIPS.get(lbl)
        if path and os.path.exists(path):
            valid_clips.append((lbl, path))
        else:
            print(f"Skipping missing clip: {lbl}")
            
    if not valid_clips:
        print("No clips found.")
        return

    # Read first clip to get dims
    cap0 = cv2.VideoCapture(valid_clips[0][1])
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    cap0.release()
    
    output_filename = "Demo_Sequential_Compilation.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Output: {output_path} ({width}x{height} @ {fps}fps)")
    
    total_frames = 0
    
    for lbl, path in valid_clips:
        print(f"Adding clip: {lbl}...")
        cap = cv2.VideoCapture(path)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Ensure uniform size just in case
            if frame.shape[1] != width or frame.shape[0] != height:
                 frame = cv2.resize(frame, (width, height))
            
            # Optional: Add a label overlay for debugging? 
            # User probably wants raw CCTV look, but let's add a subtle text for the demo feel?
            # actually user said "buat dalam satu rekaman video" implies they want to simulate the stream.
            # I will leave it clean.
            
            out.write(frame)
            total_frames += 1
            if total_frames % 100 == 0:
                print(f"Encoded {total_frames} frames...", end='\r')
                
        cap.release()
        
        # Add 1 second of black screen transition? OR just direct cut.
        # Direct cut is better for continuous CCTV feel.
        
    out.release()
    print(f"\nDone! Sequential video saved to: {output_path}")

if __name__ == "__main__":
    create_compilation()
