# 🎬 Video Frame Extraction & Roboflow Upload

Pipeline otomatis untuk ekstraksi frames dari 3 video CCTV dan upload ke Roboflow.

## 📋 Requirements

```bash
pip install opencv-python numpy tqdm roboflow
```

## 🚀 Quick Start

Jalankan satu file ini untuk proses lengkap:

```bash
python extract_and_upload.py
```

Script akan otomatis:

1. ✅ Ekstrak 5000 frames per video (total 15,000 frames)
2. ✅ Filter frames (blur, brightness, duplicate)
3. ✅ Upload semua ke Roboflow

## ⚙️ Konfigurasi

Edit di `extract_and_upload.py`:

```python
# Video paths
VIDEOS = [
    "videos/rekamanCCTV-video1.mp4",
    "videos/rekamanCCTV-video2.mp4",
    "videos/rekamanCCTV-video3.mp4"
]

# Target frames per video
MAX_FRAMES_PER_VIDEO = 5000

# Roboflow config
API_KEY = "your-api-key"
WORKSPACE = "your-workspace"
PROJECT = "your-project"
```

## 📁 Struktur

```
.
├── videos/                      # Folder video input
│   ├── rekamanCCTV-video1.mp4
│   ├── rekamanCCTV-video2.mp4
│   └── rekamanCCTV-video3.mp4
├── frames_filtered/             # Output frames (auto-generated)
├── extract_and_upload.py        # Main script
└── requirements.txt             # Dependencies
```

## 🎯 Features

- **Auto extraction**: Ambil frames dengan interval cerdas
- **Quality filters**:
  - Blur detection (Laplacian variance)
  - Brightness check (min/max threshold)
  - Duplicate removal (histogram comparison)
- **Batch upload**: Upload efficient ke Roboflow
- **Progress tracking**: Real-time progress bar

## 📊 Output

Frames akan tersimpan dengan format:

```
video1_frame_00000.jpg
video1_frame_00001.jpg
...
video2_frame_00000.jpg
...
video3_frame_00000.jpg
```

## 💤 Automation

Script berjalan full otomatis - tinggal tidur! 🌙

Proses bisa memakan waktu beberapa jam tergantung ukuran video.

## 📝 License

MIT
