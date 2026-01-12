# Dokumentasi Pelatihan & Evaluasi Model AI (MobileNetV3)

Dokumen ini menjelaskan alur teknis, logika, dan tata cara evaluasi model MobileNetV3 yang digunakan untuk klasifikasi aktivitas operator (Bekerja, Idle, Meninggalkan Area).

---

## 1. Konsep Dasar: Transfer Learning

Kita tidak melatih model dari nol (from scratch) karena membutuhkan jutaan gambar dan waktu berbulan-bulan. Kita menggunakan teknik **Transfer Learning**.

- **Analogi**: Kita merekrut seorang sarjana yang sudah pintar matematika (Model Pre-trained ImageNet). Kita tidak perlu mengajarinya berhitung dari nol, kita hanya perlu memberinya "kursus singkat" (Training) tentang cara mengenali aktivitas pabrik.
- **Base Model**: MobileNetV3 Large (dipilih karena ringan, cepat, dan akurat untuk CPU/Mobile device).
- **Weights**: ImageNet (sudah kenal 1000 jenis objek umum).

---

## 2. Strategi Pelatihan 2-Fase (PENTING)

Kode training yang kita gunakan menerapkan strategi **2-Stage Training** untuk hasil maksimal. Banyak mahasiswa salah karena hanya melakukan tahap 1 lalu berhenti.

### **Fase 1: Feature Extraction (Frozen Head)**

- **Tujuan**: Melatih "Otak Baru" (Classifier Layer) agar bisa membedakan 3 kelas kita, tanpa merusak pengetahuan lama model.
- **Teknis**:
  - `base_model.trainable = False` (Otak lama dibekukan).
  - Hanya layer `Dense` (Output) yang belajar.
  - Learning Rate: `0.001` (Standar).
- **Durasi**: 15 Epoch.

### **Fase 2: Fine-Tuning (Adaptasi Spesifik)**

- **Tujuan**: Mengizinkan model untuk sedikit mengubah "cara pandang" terhadap tekstur/bentuk yang spesifik ada di pabrik (misal: bentuk mesin, seragam operator).
- **Teknis**:
  - `base_model.trainable = True` (Otak lama dicairkan).
  - Kita hanya membuka kunci **50 layer terakhir** (layer awal biarkan tetap beku karena fitur dasar seperti garis/sudut sudah sempurna).
  - Learning Rate: **`1e-5` (Sangat Kecil)**. _Kenapa?_ Agar perubahan bobot pelan-pelan dan tidak merusak memori lama model.
- **Durasi**: 30 Epoch (Total training +/- 45 Epoch).

---

## 3. Data Augmentation (Trik CCTV)

CCTV seringkali memiliki tantangan: posisi orang berubah, pencahayaan berubah, atau orangnya terlihat kecil. Kita mengatasi ini di `ImageDataGenerator`:

| Parameter               | Fungsi Logis         | Relevansi CCTV                                          |
| :---------------------- | :------------------- | :------------------------------------------------------ |
| `zoom_range=[0.8, 1.2]` | Simulasi zoom in/out | Orang kadang dekat, kadang jauh dari kamera.            |
| `rotation_range=20`     | Rotasi gambar        | Kamera mungkin miring atau postur orang miring.         |
| `brightness_range`      | Terang/Gelap         | Simulasi perubahan cahaya pagi/siang/sore di pabrik.    |
| `horizontal_flip`       | Cermin Kiri-Kanan    | Orang bisa berjalan dari kiri ke kanan atau sebaliknya. |

---

## 4. Metrik Evaluasi (Cara Baca Hasil)

Setelah training selesai, kita mengevaluasi menggunakan Data Test (Data yang belum pernah dilihat model sama sekali).

### **A. Classification Report**

1.  **Precision**: "Seberapa tepat tebakan model?"
    - _Jika Precision 'Idle' rendah_: Berarti model sering menuduh orang 'Bekerja' padahal sedang 'Idle' (Banyak False Positive).
2.  **Recall (Sensitivitas)**: "Seberapa peka model?"
    - _Jika Recall 'Meninggalkan Area' rendah_: Berarti ada kejadian 'Meninggalkan Area' yang lolos/dilewatkan oleh model.
3.  **F1-Score**: Rata-rata harmonis Precision & Recall.
    - Target TA biasanya **> 80% (0.80)**.

### **B. Confusion Matrix**

Tabel kejujuran model.

- **Diagonal Utama (Kiri Atas ke Kanan Bawah)**: Prediksi Benar. Warnanya harus paling gelap.
- **Kotak Lain (Off-Diagonal)**: Kesalahan.
  - _Contoh Error_: Jika kotak baris 'Idle' kolom 'Bekerja' ada angkanya, berarti model bingung membedakan orang diam dan orang kerja. Kemungkinan karena hanya duduk tapi tangan bergerak.

---

## 5. Cheat Sheet Pertanyaan Sidang

**Q: Mengapa pakai MobileNetV3? Kenapa bukan ResNet atau VGG?**

> **A:** Karena sistem ini dirancang untuk _Real-Time Monitoring_ di perangkat dengan daya komputasi terbatas (Laptop/Edge Device). MobileNetV3 jauh lebih ringan (low latency) dibanding ResNet, tapi akurasinya masih sangat kompetitif untuk kasus ini.

**Q: Apa itu Fine-Tuning yang kamu lakukan?**

> **A:** Saya membuka kunci (unfreeze) 50 layer terakhir dari MobileNetV3 dan melatihnya ulang dengan _learning rate_ sangat kecil (1e-5). Tujuannya agar model bisa belajar mengenali fitur spesifik lingkungan kerja saya (seragam, mesin, latar belakang) yang mungkin tidak ada di dataset standar ImageNet.

**Q: Kenapa Inputnya di-Crop (ROI)? Apakah itu tidak curang?**

> **A:** Tidak, Pak/Bu. Itu adalah standar _Computer Vision_ yang disebut _Region Proposal_. Perspektif CCTV sangat luas (wide), sehingga objek manusia menjadi sangat kecil (<5% frame). Jika tidak di-crop, model akan bias mempelajari background (lantai/tembok) daripada aktivitas manusianya. ROI memastikan input network relevan dengan apa yang ingin dideteksi.

**Q: Bagaimana jika akurasi 'Idle' rendah?**

> **A:** Aktivitas 'Idle' dan 'Bekerja' seringkali mirip secara visual (sama-sama duduk). Perbedaannya seringkali hanya pada gerakan tangan minor. Model gambar diam (CNN) memiliki keterbatasan dalam menangkap konteks gerakan (temporal). Solusi kedepannya bisa menggunakan model berbasis video (LSTM/3D-CNN).
