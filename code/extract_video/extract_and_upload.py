#!/usr/bin/env python3
"""
PIPELINE LENGKAP: Ekstrak frames dari 3 video + Upload ke Roboflow
Otomatis dari awal sampai selesai - tinggal tidur!
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from roboflow import Roboflow

# ============= KONFIGURASI =============
BASE_DIR = Path(__file__).resolve().parents[2]

# Video paths
VIDEOS = [
    str(BASE_DIR / "dataset" / "videos" / "rekamanCCTV-video1.mp4"),
    str(BASE_DIR / "dataset" / "videos" / "rekamanCCTV-video2.mp4"),
    str(BASE_DIR / "dataset" / "videos" / "rekamanCCTV-video3.mp4"),
]

# Output
OUTPUT_DIR = str(BASE_DIR / "frames_filtered")
MAX_FRAMES_PER_VIDEO = 5000  # 5000 frames per video = 15000 total

# Filter settings
FRAME_INTERVAL = 10
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 250
SIMILARITY_THRESHOLD = 0.95

# Roboflow
API_KEY = "vy5tXdqMJ0rpEfil3l6z"
WORKSPACE = "salman-fauzi"
PROJECT = "object-operatormesin"
BATCH_SIZE = 50

# ============= FUNGSI HELPER =============
def calculate_blur(image):
    """Hitung blur score menggunakan Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    """Hitung rata-rata brightness"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_similarity(img1, img2):
    """Hitung similarity antara 2 images untuk detect duplicate"""
    if img1 is None or img2 is None:
        return 0.0
    
    # Resize untuk perbandingan cepat
    size = (64, 64)
    img1_small = cv2.resize(img1, size)
    img2_small = cv2.resize(img2, size)
    
    # Convert ke grayscale
    gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
    
    # Hitung histogram
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Compare
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# ============= STEP 1: EKSTRAKSI =============
def extract_frames():
    """Ekstrak frames dari semua video"""
    print("\n" + "="*70)
    print("STEP 1: EKSTRAKSI FRAMES DARI 3 VIDEO")
    print("="*70)
    
    # Bersihkan folder output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(OUTPUT_DIR, "*.jpg"))
    if existing:
        print(f"\n🗑️  Menghapus {len(existing)} file lama...")
        for f in existing:
            os.remove(f)
        print("✅ Folder dibersihkan!")
    
    total_saved = 0
    
    # Proses tiap video
    for video_idx, video_path in enumerate(VIDEOS, 1):
        video_name = os.path.basename(video_path).replace(".mp4", "")
        
        print(f"\n{'='*70}")
        print(f"VIDEO {video_idx}/3: {video_name}")
        print(f"{'='*70}")
        
        # Check video exists
        if not os.path.exists(video_path):
            print(f"⚠️  Video tidak ditemukan: {video_path}")
            continue
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Tidak bisa membuka video: {video_path}")
            continue
        
        # Info video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Total frames: {total_frames:,}")
        print(f"⏱️  Durasi: {duration/60:.1f} menit")
        print(f"🎯 Target: {MAX_FRAMES_PER_VIDEO:,} frames")
        
        # Process frames
        frame_count = 0
        saved_count = 0
        skipped = {'blur': 0, 'brightness': 0, 'duplicate': 0}
        prev_frame = None
        
        pbar = tqdm(total=total_frames, desc=f"Extracting", unit="frame")
        
        while saved_count < MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process setiap N frame
            if frame_count % FRAME_INTERVAL == 0:
                # Check blur
                blur = calculate_blur(frame)
                if blur < BLUR_THRESHOLD:
                    skipped['blur'] += 1
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                # Check brightness
                brightness = calculate_brightness(frame)
                if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
                    skipped['brightness'] += 1
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                # Check duplicate
                if prev_frame is not None:
                    similarity = calculate_similarity(frame, prev_frame)
                    if similarity > SIMILARITY_THRESHOLD:
                        skipped['duplicate'] += 1
                        frame_count += 1
                        pbar.update(1)
                        continue
                
                # Save frame
                filename = f"video{video_idx}_frame_{saved_count:05d}.jpg"
                output_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                saved_count += 1
                prev_frame = frame.copy()
                pbar.set_description(f"Saved: {saved_count}/{MAX_FRAMES_PER_VIDEO}")
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Summary untuk video ini
        print(f"\n✅ {video_name}: {saved_count:,} frames tersimpan")
        print(f"   Filtered - Blur: {skipped['blur']}, Brightness: {skipped['brightness']}, Duplicate: {skipped['duplicate']}")
        
        total_saved += saved_count
    
    print(f"\n{'='*70}")
    print(f"📊 TOTAL FRAMES DARI SEMUA VIDEO: {total_saved:,}")
    print(f"📁 Tersimpan di: {OUTPUT_DIR}/")
    print(f"{'='*70}")
    
    return total_saved

# ============= STEP 2: UPLOAD =============
def upload_to_roboflow():
    """Upload semua frames ke Roboflow"""
    print(f"\n{'='*70}")
    print("STEP 2: UPLOAD KE ROBOFLOW")
    print(f"{'='*70}")
    
    # Get semua gambar
    all_images = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
    
    if not all_images:
        print("❌ Tidak ada gambar untuk diupload!")
        return 0
    
    print(f"📂 Total gambar: {len(all_images):,}")
    
    # Initialize Roboflow
    print(f"\n🔗 Connecting to Roboflow...")
    try:
        rf = Roboflow(api_key=API_KEY)
        workspace = rf.workspace(WORKSPACE)
        project = workspace.project(PROJECT)
        print(f"✅ Connected to project: {PROJECT}")
    except Exception as e:
        print(f"❌ Koneksi gagal: {e}")
        print(f"\n💡 Pastikan:")
        print(f"   - API Key benar: {API_KEY}")
        print(f"   - Workspace ada: {WORKSPACE}")
        print(f"   - Project ada: {PROJECT}")
        return 0
    
    # Upload dalam batch
    total_batches = (len(all_images) + BATCH_SIZE - 1) // BATCH_SIZE
    uploaded = 0
    failed = 0
    
    print(f"\n🚀 Uploading {len(all_images):,} images in {total_batches} batches...")
    
    for batch_idx in range(0, len(all_images), BATCH_SIZE):
        batch = all_images[batch_idx:batch_idx + BATCH_SIZE]
        current_batch = (batch_idx // BATCH_SIZE) + 1
        
        print(f"\n📦 Batch {current_batch}/{total_batches} ({len(batch)} images)")
        
        pbar = tqdm(batch, desc=f"  Uploading", unit="img")
        for img_path in pbar:
            try:
                project.upload(img_path, num_retry_uploads=1)
                uploaded += 1
            except Exception as e:
                failed += 1
                # Tidak print per error biar tidak spam
        pbar.close()
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 UPLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Berhasil: {uploaded:,}/{len(all_images):,}")
    print(f"❌ Gagal:    {failed:,}/{len(all_images):,}")
    print(f"📁 Project:  https://app.roboflow.com/{WORKSPACE}/{PROJECT}")
    print(f"{'='*70}")
    
    return uploaded

# ============= MAIN =============
def main():
    print("\n" + "="*70)
    print("🤖 PIPELINE OTOMATIS: 3 VIDEO → ROBOFLOW")
    print("="*70)
    print(f"📹 Videos: {len(VIDEOS)}")
    for i, v in enumerate(VIDEOS, 1):
        print(f"   {i}. {os.path.basename(v)}")
    print(f"🎯 Target: {MAX_FRAMES_PER_VIDEO:,} frames per video")
    print(f"☁️  Upload to: {PROJECT}")
    print("="*70)
    
    # Step 1: Extract
    total_frames = extract_frames()
    
    if total_frames == 0:
        print("\n❌ Tidak ada frame yang berhasil diekstrak!")
        return
    
    # Step 2: Upload
    uploaded = upload_to_roboflow()
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 SELESAI!")
    print("="*70)
    print(f"✅ Total frames extracted: {total_frames:,}")
    print(f"✅ Total frames uploaded:  {uploaded:,}")
    print("="*70)
    print("\n💤 Proses selesai! Selamat tidur!")

if __name__ == "__main__":
    main()
