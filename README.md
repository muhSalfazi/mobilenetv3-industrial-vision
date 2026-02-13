Aplikasi Kecerdasan Buatan (AI) untuk mengidentifikasi aktivitas operator mesin pada rekaman CCTV menggunakan model Deep Learning **MobileNetV3**.

Dikembangkan untuk Tugas Akhir oleh **Muhamad Salman Fauzi**.

---

## Cara Menjalankan Aplikasi

Berikut adalah langkah-langkah singkat untuk menjalankan aplikasi ini di komputer Anda:

### 1. Prasyarat

Pastikan Anda sudah menginstal **Python** (versi 3.9 s/d 3.11 disarankan).

### 2. Buat & Aktifkan Virtual Environment (Disarankan)

Supaya library tidak bentrok dengan sistem lain, sebaiknya gunakan virtual environment.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Tips:** Untuk keluar dari virtual environment, cukup ketik perintah `deactivate`.

### 3. Install Dependencies

Buka terminal/command prompt di folder project ini, lalu jalankan:

```bash
pip install -r requirements.txt
```

_(Pastikan file `requirements.txt` sudah ada)_

### 4. Jalankan Aplikasi

Ketik perintah berikut di terminal:

```bash
streamlit run camera_app.py
```

Browser akan otomatis terbuka menampilkan aplikasi.

---

## 🛠️ Fitur Utama

1.  **Upload Video**: Mendukung format `.mp4` dan `.avi`.
2.  **Klasifikasi Real-time**: Mendeteksi 3 status: _Idle_, _Bekerja_, _Meninggalkan Area_.
3.  **Smart Validation**: Menggunakan _Motion Detection_ untuk memvalidasi aktivitas "Meninggalkan Area" agar lebih akurat.
4.  **Laporan Statistik**: Di akhir video, ditampilkan grafik ringkasan aktivitas.

---

