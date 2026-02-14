BISMILLAH WISUDA!!!

Dikembangkan untuk Tugas Akhir oleh **Muhamad Salman Fauzi**.

---

### 2. Buat & Aktifkan Virtual Environment

Supaya library tidak bentrok dengan sistem lain, sebaiknya gunakan virtual environment.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Tips:** Untuk keluar dari virtual environment, cukup ketik perintah `deactivate`.

### 3. Install Dependencies

Buka terminal/command prompt di folder project ini, lalu jalankan:

```bash
pip install -r requirements.txt
```
### 4. Jalankan Aplikasi

Ketik perintah berikut di terminal:

```bash
streamlit run streamlit/web.py
```

Browser akan otomatis terbuka menampilkan aplikasi.

---

## 🛠️ Fitur Utama

1.  **Upload Video**: Mendukung format `.mp4` dan `.avi`.
2.  **Klasifikasi Real-time**: Mendeteksi 3 status: _Idle_, _Bekerja_, _Meninggalkan Area_.

