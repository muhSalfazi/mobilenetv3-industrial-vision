import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
import tempfile
import os
import time
import altair as alt
from pathlib import Path
from streamlit_cropper import st_cropper
from PIL import Image as PILImage

# ============= 1. KONFIGURASI HALAMAN & STYLE =============
st.set_page_config(
    page_title="Activity Classifier AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input file styling */
    .stFileUploader > div > div {
        background-color: #161b22;
        border: 1px dashed #30363d;
        border-radius: 12px;
    }
    
    /* Cards */
    .pred-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .badge {
        background-color: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 24px;
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    .title-activity {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .desc-activity {
        font-size: 0.95rem;
        color: #8b949e;
        text-align: center;
        margin-bottom: 30px;
    }
    .conf-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
    }
    .prob-bar-container {
        height: 12px;
        background-color: #21262d;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 35px;
    }
    .prob-bar-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .dist-title {
        font-size: 0.8rem; 
        color: #8b949e; 
        margin-bottom: 15px; 
        letter-spacing: 1.5px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .dist-row {
        display: flex;
        align-items: center;
        margin-bottom: 14px;
        font-size: 0.9rem;
    }
    .dist-label {
        width: 140px;
        color: #8b949e;
    }
    .dist-icon {
        margin-right: 8px;
    }
    .dist-bar-wrap {
        flex-grow: 1;
        height: 8px;
        background-color: #21262d;
        border-radius: 4px;
        margin: 0 15px;
        overflow: hidden;
    }
    .dist-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .dist-val {
        width: 55px;
        text-align: right;
        color: #c9d1d9;
        font-weight: bold;
    }
    
    /* Header Style */
    .main-header {
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(180deg, rgba(31,111,235,0.05) 0%, rgba(11,15,25,0) 100%);
        border-bottom: 1px solid #30363d;
        margin-bottom: 2.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============= 2. KONFIGURASI MODEL =============
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(BASE_DIR / "dataset" / "model" / "mobilenet_best.keras")
IMG_SIZE = (224, 224)

CLASSES_ORIGINAL = ["bekerja", "idle", "meninggalkan_area"]
CLASS_NOT_IDENTIFIED = "bukan_area_monitor"


def map_model_output_to_class(preds, similarity_score=1.0, threshold=0.4):
    """
    Map model output ke class dengan Verifikasi Area
    """
    preds_corrected = np.array([preds[0], preds[1], preds[2]])
    label_idx = np.argmax(preds_corrected)
    conf = preds_corrected[label_idx]

    # LOGIKA SIDANG: Cek apakah area video sesuai dengan referensi training
    if similarity_score < threshold:
        return CLASS_NOT_IDENTIFIED, 0.0, np.array([0.0, 0.0, 0.0])

    label = CLASSES_ORIGINAL[label_idx]
    return label, conf, preds_corrected


# ============= 3. FUNGSI UTILITY =============
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    return None


def preprocess_frame(frame):
    """
    Preprocessing sesuai training:
    BGR -> RGB -> Resize -> rescale 1/255 (PURE PREPROCESSING, NO MOTION)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    img_array = resized.astype(np.float32) / 255.0  # Match training rescale=1./255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@st.cache_resource
def get_master_area_histogram():
    """
    Mengambil Master Frame dari data training sebagai benchmark area monitor asli.
    """
    ref_dir = (
        BASE_DIR / "dataset" / "dataset_classification" / "train" / "meninggalkan_area"
    )
    if not ref_dir.exists():
        return None

    files = list(ref_dir.glob("*.jpg"))
    if not files:
        return None

    # Ambil 1000.jpg atau file pertama jika tidak ada
    target_file = ref_dir / "1000.jpg"
    ref_path = target_file if target_file.exists() else files[0]

    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        return None

    # Gunakan HSV agar lebih tahan terhadap perubahan cahaya (Brightness)
    hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def calculate_area_similarity(frame, master_hist):
    if master_hist is None or frame is None or frame.size == 0:
        return 1.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # Korelasi HISTOGRAM: 1.0 (Identik) s/d -1.0 (Bedatotal)
    score = cv2.compareHist(master_hist, hist, cv2.HISTCMP_CORREL)
    return max(0, score)  # Pastikan tidak negatif


def get_card_html(label, conf, preds):
    desc_map = {
        "bekerja": "Operator sedang berada di area dan melakukan pekerjaannya dengan aktif.",
        "idle": "Operator terlihat berada di tempat kerjanya namun tidak ada aktivitas signifikan.",
        "meninggalkan_area": "Berdasarkan pantauan, operator tidak terdeteksi di area kerjanya.",
        "bukan_area_monitor": "Struktur visual area tidak cocok dengan Master Data training (Area Asing).",
    }
    color_map_css = {
        "bekerja": "#2ea043",  # Hijau
        "idle": "#e3b341",  # Kuning
        "meninggalkan_area": "#f85149",  # Merah
        "bukan_area_monitor": "#8b949e",  # Abu-abu
    }
    # Convert label to title case for display
    label_display = label.replace("_", " ").title()
    color = color_map_css.get(label, "#2ea043")

    prob_bekerja = preds[0] * 100
    prob_idle = preds[1] * 100
    prob_meninggalkan = preds[2] * 100

    html = f"""
    <div class="pred-card">
        <center><div class="badge" style="color: {color}; border-color: {color}; background-color: {color}1A;">MOBILENETV3 PREDICTION</div></center>
        <div class="title-activity" style="color: {color};">{label_display}</div>
        <div class="desc-activity">{desc_map.get(label, '')}</div>
        <div class="conf-label">
            <span>Tingkat Keyakinan</span>
            <span style="color: {color};">{conf*100:.2f}%</span>
        </div>
        <div class="prob-bar-container">
            <div class="prob-bar-fill" style="width: {conf*100}%; background-color: {color};"></div>
        </div>
        <div class="dist-title">CONFIDENCE MODEL</div>
        <div class="dist-row">
            <div class="dist-label"><span class="dist-icon">🔨</span> Bekerja</div>
            <div class="dist-bar-wrap"><div class="dist-bar" style="width: {prob_bekerja}%; background-color: #2ea043;"></div></div>
            <div class="dist-val">{prob_bekerja:.1f}%</div>
        </div>
        <div class="dist-row">
            <div class="dist-label"><span class="dist-icon">⏳</span> Idle</div>
            <div class="dist-bar-wrap"><div class="dist-bar" style="width: {prob_idle}%; background-color: #e3b341;"></div></div>
            <div class="dist-val">{prob_idle:.1f}%</div>
        </div>
        <div class="dist-row">
            <div class="dist-label"><span class="dist-icon">🚶</span> Meninggalkan</div>
            <div class="dist-bar-wrap"><div class="dist-bar" style="width: {prob_meninggalkan}%; background-color: #f85149;"></div></div>
            <div class="dist-val">{prob_meninggalkan:.1f}%</div>
        </div>
    </div>
    """
    return html


def safe_remove_file(file_path, retries=5, delay=0.2):
    """Best-effort file removal for Windows where handles may release slightly late."""
    if not file_path:
        return

    for _ in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return
        except PermissionError:
            time.sleep(delay)
        except FileNotFoundError:
            return

    # Final attempt to surface unexpected issues while still avoiding app crash.
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.warning(f"Gagal menghapus file sementara: {e}")


# ============= 4. APP UTAMA =============
def main():
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; padding:0; font-weight:800; color:#c9d1d9; letter-spacing:-0.5px;">IDENTIFIKASI AKTIVITAS OPERATOR MESIN PADA REKAMAN CCTV MENGGUNAKAN<span style="color:#58a6ff;">MOBILENETV3</span></h1>
            <p style="color:#8b949e; margin-top:10px; font-size:1.1rem;">Muhamad Salman fauzi - 22416255201063</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # ===== SIDEBAR CALIBRATION =====
    with st.sidebar:
        st.header("⚙️ Kalibrasi Deteksi")

        # Tambahkan Toggle untuk aktif/nonaktifkan fitur ini
        enable_area_check = st.toggle(
            "Aktifkan Verifikasi Area",
            value=True,
            help="Jika aktif, sistem akan menolak video yang bukan berasal dari area kerja asli.",
        )

        area_sensitivity = st.slider(
            "Sensitivitas Verifikasi Area",
            0.0,
            1.0,
            0.05,
            0.01,
            disabled=not enable_area_check,
            help="Geser ke kiri jika area asli terdeteksi sebagai 'Bukan Area Monitor'.",
        )

        st.divider()

        # Opsi Baru: Analisis Global vs ROI
        analysis_mode = st.radio(
            "Target Analisis",
            ["Fokus ROI (Kotak)", "Seluruh Frame (Global)"],
            index=0,
            help="Pilih apakah model harus melihat kotak tertentu atau seluruh tampilan CCTV.",
        )

        st.info(
            "Gunakan 'Seluruh Frame' agar sistem tidak langsung mendeteksi keluar saat operator bergeser dari kotaknya."
        )

        show_roi_selector = st.toggle(
            "Tampilkan Pengaturan Area Operator",
            value=True,
            help="Tampilkan/sembunyikan panel 'Tentukan Area Operator' di area utama.",
        )

    # Removed manual ROI sliders. Replaced with interactive cropper after upload.
    # ===== FILE UPLOAD =====
    tab_video = st.container()

    model = load_model()
    if not model:
        st.error(f"❌ Model tidak ditemukan di `{MODEL_PATH}`")
        return

    # ===== TAB 1: VIDEO =====
    with tab_video:
        st.markdown(
            "<h3 style='margin-bottom:0px;'>Upload Video CCTV</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#8b949e; margin-top:0px;'>Format yang didukung: MP4, AVI</p>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Video",
            type=["mp4", "avi"],
            label_visibility="collapsed",
            key="video_upload",
        )

        if uploaded_file:
            temp_video_path = None
            cap = None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name

            cap = cv2.VideoCapture(temp_video_path)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            # Ambil frame pertama untuk interactive cropper
            ret, first_frame = cap.read()
            if not ret:
                st.error("Gagal membaca frame video untuk preview.")
                if cap is not None:
                    cap.release()
                safe_remove_file(temp_video_path)
                return

            # Reset frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            h_video = first_frame.shape[0]
            w_video = first_frame.shape[1]

            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            pil_first_frame = PILImage.fromarray(first_frame_rgb)

            # default_coords = (x_left, x_right, y_top, y_bottom)
            default_coords = (545, 1090, 527, 991)
            if w_video < 1150:  # fallback for smaller videos
                default_coords = (
                    int(w_video * 0.3),
                    int(w_video * 0.7),
                    int(h_video * 0.2),
                    int(h_video * 0.8),
                )

            if "roi_rect" not in st.session_state:
                st.session_state["roi_rect"] = None

            if st.session_state["roi_rect"]:
                cached_rect = st.session_state["roi_rect"]
                default_coords_for_cropper = (
                    int(cached_rect["left"]),
                    int(cached_rect["left"] + cached_rect["width"]),
                    int(cached_rect["top"]),
                    int(cached_rect["top"] + cached_rect["height"]),
                )
            else:
                default_coords_for_cropper = default_coords

            if show_roi_selector:
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                st.markdown(
                    "<h4 style='color:#c9d1d9;'>Tentukan Area Operator (Tarik Kotak Hijau)</h4>",
                    unsafe_allow_html=True,
                )
                st.info(
                    "Geser dan sesuaikan kotak hijau muda di bawah hingga mengepas tepat di bagian tubuh/area operator. Biarkan sisa frame gelap."
                )

                # Interactive cropper UI
                rect = st_cropper(
                    pil_first_frame,
                    realtime_update=True,
                    box_color="#00FF00",
                    return_type="box",
                    default_coords=default_coords_for_cropper,
                )
                st.session_state["roi_rect"] = rect
            else:
                if st.session_state["roi_rect"]:
                    rect = st.session_state["roi_rect"]
                else:
                    rect = {
                        "left": int(default_coords[0]),
                        "top": int(default_coords[2]),
                        "width": int(default_coords[1] - default_coords[0]),
                        "height": int(default_coords[3] - default_coords[2]),
                    }

            roi_left = int(rect["left"])
            roi_top = int(rect["top"])
            roi_w = int(rect["width"])
            roi_h = int(rect["height"])

            roi_right = roi_left + roi_w
            roi_bottom = roi_top + roi_h

            if show_roi_selector:
                st.markdown(
                    f"<p style='color:#3fb950; font-size:0.9rem;'><b>Info Area Terpilih:</b> Lebar={roi_w}px, Tinggi={roi_h}px (Kiri:{roi_left}, Kanan:{roi_right}, Atas:{roi_top}, Bawah:{roi_bottom})</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "Panel area operator disembunyikan. Sistem memakai ROI terakhir/default."
                )
            st.markdown("<br>", unsafe_allow_html=True)

            # Release preview capture. Re-open a fresh handle only when analysis starts.
            if cap is not None:
                cap.release()
                cap = None

            if st.button(
                "🚀 Mulai Analisis Video", type="primary", use_container_width=True
            ):
                cap = cv2.VideoCapture(temp_video_path)
                st.divider()

                # Layout Utama Saat Memproses
                col_vid, col_space, col_realtime = st.columns([1.4, 0.1, 1])
                with col_vid:
                    st.markdown(
                        "<h4 style='color:#c9d1d9;'>📹 Monitoring Playback</h4>",
                        unsafe_allow_html=True,
                    )
                    video_display = st.empty()
                    st.markdown("<br>", unsafe_allow_html=True)
                    progress_bar = st.progress(0)

                with col_realtime:
                    st.markdown(
                        "<h4 style='color:#c9d1d9;'>Hasil Klasifikasi</h4>",
                        unsafe_allow_html=True,
                    )
                    status_placeholder = st.empty()

                results = []
                frame_idx = 0
                skip_frames = 5  # Fix value untuk realtime inference tanpa lemot
                prev_roi_frame = None

                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_idx % skip_frames == 0:
                            roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

                            # --- LAYER 1: VERIFIKASI LOKASI AREA ---
                            master_hist = get_master_area_histogram()
                            sim_score = calculate_area_similarity(roi_frame, master_hist)

                            # --- LAYER 2: KLASIFIKASI AI ---
                            if roi_frame.size > 0:
                                preds_raw = model.predict(
                                    preprocess_frame(roi_frame), verbose=0
                                )[0]

                                # Cek status verifikasi dari Sidebar
                                sim_val = sim_score if enable_area_check else 1.0
                                thresh_val = (
                                    area_sensitivity if enable_area_check else 0.0
                                )

                                label, conf, preds = map_model_output_to_class(
                                    preds_raw, sim_val, thresh_val
                                )
                            else:
                                label = "meninggalkan_area"
                                conf = 1.0
                                preds = np.array([0.0, 0.0, 1.0])

                            prev_roi_frame = roi_frame.copy()

                            timestamp = frame_idx / fps if fps > 0 else 0
                            results.append(
                                {
                                    "Waktu": timestamp,
                                    "Aktivitas": label,
                                    "Confidence": float(conf),
                                }
                            )

                            # Update Realtime UI (CARD)
                            card_html = get_card_html(label, conf, preds)
                            status_placeholder.markdown(card_html, unsafe_allow_html=True)

                            # Update Video Frame Visual
                            color_roi = (
                                (0, 255, 0)
                                if label != "bukan_area_monitor"
                                else (139, 148, 158)
                            )
                            cv2.rectangle(
                                frame,
                                (roi_left, roi_top),
                                (roi_right, roi_bottom),
                                color_roi,
                                3,
                            )

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            bgr_colors = {
                                "bekerja": (0, 255, 0),
                                "idle": (0, 215, 255),
                                "meninggalkan_area": (0, 0, 255),
                                "bukan_area_monitor": (158, 148, 139),
                            }

                            # display_text = label.replace("_", " ").upper()
                            # cv2.putText(
                            #     frame,
                            #     f"{display_text} (SIM: {sim_score:.2f})",
                            #     (roi_left + 10, roi_top + 45),
                            #     font,
                            #     1.1,
                            #     bgr_colors.get(label, (255, 255, 255)),
                            #     3,
                            # )

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Resize slightly for rendering performance in streamlit
                            frame_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.7, fy=0.7)

                            # Add ROI zoom inset (top-right corner)
                            roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                            roi_zoomed = cv2.resize(roi_rgb, (150, 150))
                            h, w = frame_rgb.shape[:2]
                            end_x = min(w, 150 + 10)
                            end_y = min(h, 150 + 10)
                            frame_rgb[10 : 10 + 150, w - 160 : w - 10] = roi_zoomed

                            video_display.image(
                                frame_rgb, channels="RGB", use_container_width=True
                            )

                        progress_pct = min((frame_idx + 1) / total_frames_video, 1.0)
                        progress_bar.progress(
                            progress_pct,
                            text=f"Sedang menganalisis... {int(progress_pct*100)}%",
                        )
                        frame_idx += 1
                finally:
                    if cap is not None:
                        cap.release()
                    safe_remove_file(temp_video_path)

                # --- CLEANUP & RINGKASAN AKHIR ---
                st.success("Analisis Selesai.")
                if results:
                    df = pd.DataFrame(results)
                    st.divider()
                    st.markdown(
                        "<h3 style='margin-bottom:20px;'>Ringkasan Laporan</h3>",
                        unsafe_allow_html=True,
                    )

                    c1, c2 = st.columns(2)
                    c1.metric("Total Sampel Frame", len(df))

                    count_df = df["Aktivitas"].value_counts().reset_index()
                    count_df.columns = ["Aktivitas", "Jumlah"]

                    if not count_df.empty:
                        c2.metric(
                            "Aktivitas Paling Dominan", count_df.loc[0, "Aktivitas"]
                        )

                        col_chart, col_breakdown = st.columns([1, 1])

                        with col_chart:
                            color_scale = alt.Scale(
                                domain=["bekerja", "idle", "meninggalkan_area"],
                                range=["#2ea043", "#e3b341", "#f85149"],
                            )

                            base = alt.Chart(count_df).encode(
                                theta=alt.Theta("Jumlah", stack=True)
                            )
                            pie = base.mark_arc(outerRadius=130, innerRadius=70).encode(
                                color=alt.Color("Aktivitas", scale=color_scale),
                                order=alt.Order("Jumlah", sort="descending"),
                                tooltip=["Aktivitas", "Jumlah"],
                            )
                            st.altair_chart(pie, use_container_width=True)

                        with col_breakdown:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            for _, row in count_df.iterrows():
                                pct = (row["Jumlah"] / len(df)) * 100
                                st.markdown(
                                    f"""
                                <div style="background-color:#161b22; padding:15px; border-radius:10px; border:1px solid #30363d; margin-bottom:10px; display:flex; justify-content:space-between;">
                                    <div><strong style="color:#c9d1d9;">{row['Aktivitas']}</strong></div>
                                <div><span style="color:#8b949e;">{row['Jumlah']} frame</span> &nbsp; <strong style="color:#58a6ff;">{pct:.1f}%</strong></div>
                            </div>
                            """,
                                    unsafe_allow_html=True,
                                )
                else:
                    st.warning("Tidak ada aktivitas yang terdeteksi.")


if __name__ == "__main__":
    main()
