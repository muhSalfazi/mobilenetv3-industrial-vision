import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
import tempfile
import os
import altair as alt
from pathlib import Path
from PIL import Image as PILImage

# ============= 1. KONFIGURASI HALAMAN & STYLE =============
st.set_page_config(
    page_title="Activity Classifier AI - Full Frame",
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

# Original class order dari training (match training_Model_MobileNetV3.ipynb)
CLASSES_ORIGINAL = ["bekerja", "idle", "meninggalkan_area"]


def map_model_output_to_class(preds):
    """
    Map model output ke class - tanpa swap, pakai langsung
    """
    # Gunakan predictions langsung tanpa swap
    preds_corrected = np.array([preds[0], preds[1], preds[2]])
    label_idx = np.argmax(preds_corrected)
    conf = preds_corrected[label_idx]
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


def preprocess_frame(frame, aspect_ratio_preserve=True):
    """
    Preprocessing sesuai training dengan aspect ratio preservation:
    BGR -> RGB -> Preserve Aspect Ratio (pad dengan black) -> rescale 1/255
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if aspect_ratio_preserve:
        # Preserve aspect ratio dengan padding (CENTER)
        h, w = rgb.shape[:2]
        scale = min(IMG_SIZE[0] / h, IMG_SIZE[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding ke 224x224 di tengah
        top = (IMG_SIZE[0] - new_h) // 2
        bottom = IMG_SIZE[0] - new_h - top
        left = (IMG_SIZE[1] - new_w) // 2
        right = IMG_SIZE[1] - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # Resize langsung (fast, tapi bisa distort)
        padded = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    
    img_array = padded.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_card_html(label, conf, preds):
    desc_map = {
        "bekerja": "Operator sedang berada di area dan melakukan pekerjaannya dengan aktif.",
        "idle": "Operator terlihat berada di tempat kerjanya namun tidak ada aktivitas signifikan.",
        "meninggalkan_area": "Berdasarkan pantauan, operator tidak terdeteksi di area kerjanya.",
    }
    color_map_css = {
        "bekerja": "#2ea043",  # Hijau Soft
        "idle": "#e3b341",  # Kuning Soft
        "meninggalkan_area": "#f85149",  # Merah Soft
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


# ============= 4. APP UTAMA =============
def main():
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; padding:0; font-weight:800; color:#c9d1d9; letter-spacing:-0.5px;">IDENTIFIKASI AKTIVITAS OPERATOR MESIN PADA REKAMAN CCTV MENGGUNAKAN<span style="color:#58a6ff;">MOBILENETV3</span></h1>
            <p style="color:#8b949e; margin-top:10px; font-size:1.1rem;">Mode Analisis: Full Frame (Tanpa ROI)</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # ===== FILE UPLOAD =====
    tab_video = st.container()

    model = load_model()
    if not model:
        st.error(f"❌ Model tidak ditemukan di `{MODEL_PATH}`")
        return

    # ===== VIDEO UPLOAD & ANALYSIS =====
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
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            # Ambil frame pertama untuk preview
            ret, first_frame = cap.read()
            if not ret:
                st.error("Gagal membaca frame video untuk preview.")
                return

            # Reset frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            h_video = first_frame.shape[0]
            w_video = first_frame.shape[1]

            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            st.info("📸 Mode Analisis: Auto-Crop Center - Fokus pada area tengah untuk deteksi lebih akurat")
            
            # Slider untuk adjust auto-crop margins
            col_slider_w, col_slider_h = st.columns(2)
            with col_slider_w:
                crop_pct_w = st.slider(
                    "Crop Margin Lebar (%)",
                    min_value=5, max_value=40, value=20, step=5,
                    help="Semakin besar = semakin fokus ke tengah (lebih crop dari sisi kiri-kanan)"
                )
            with col_slider_h:
                crop_pct_h = st.slider(
                    "Crop Margin Tinggi (%)",
                    min_value=5, max_value=40, value=25, step=5,
                    help="Semakin besar = semakin fokus ke tengah (lebih crop dari sisi atas-bawah)"
                )
            
            # Auto-crop center region dengan custom margins
            crop_margin_w = int(w_video * (crop_pct_w / 100))
            crop_margin_h = int(h_video * (crop_pct_h / 100))
            
            roi_left = crop_margin_w
            roi_right = w_video - crop_margin_w
            roi_top = crop_margin_h
            roi_bottom = h_video - crop_margin_h
            roi_w = roi_right - roi_left
            roi_h = roi_bottom - roi_top
            
            st.markdown(
                f"<p style='color:#58a6ff; font-size:0.85rem;'><b>Auto-Crop Area:</b> {roi_w}×{roi_h}px (margin: {crop_margin_w}px W, {crop_margin_h}px H)</p>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🚀 Mulai Analisis Video", type="primary", use_container_width=True
            ):
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

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % skip_frames == 0:
                        # Process dengan auto-crop area tengah (bukan full frame)
                        frame_to_process = frame[roi_top:roi_bottom, roi_left:roi_right].copy()

                        # Process with model ONLY (Pure AI Classification)
                        if frame_to_process.size > 0:
                            try:
                                preds_raw = model.predict(
                                    preprocess_frame(frame_to_process, aspect_ratio_preserve=True), 
                                    verbose=0
                                )[0]
                                
                                # Debug: Check predictions
                                if np.isnan(preds_raw).any():
                                    st.warning(f"⚠️ NaN detected in predictions at frame {frame_idx}")
                                    preds_raw = np.array([1/3, 1/3, 1/3])
                                
                                label, conf, preds = map_model_output_to_class(preds_raw)
                            except Exception as e:
                                st.error(f"❌ Inference error at frame {frame_idx}: {str(e)}")
                                label = "error"
                                conf = 0.0
                                preds = np.array([0.0, 0.0, 0.0])
                        else:
                            label = "meninggalkan_area"
                            conf = 1.0
                            preds = np.array([0.0, 0.0, 1.0])

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
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bgr_colors = {
                            "bekerja": (0, 255, 0),
                            "idle": (0, 215, 255),
                            "meninggalkan_area": (0, 0, 255),
                        }
                        
                        # Draw auto-crop area boundary
                        cv2.rectangle(
                            frame,
                            (roi_left, roi_top),
                            (roi_right, roi_bottom),
                            (100, 100, 100),  # Gray border untuk auto-crop area
                            2,
                        )
                        
                        cv2.putText(
                            frame,
                            f"{label} ({conf*100:.1f}%)",
                            (roi_left + 10, roi_top + 45),
                            font,
                            1.5,
                            bgr_colors.get(label, (255, 255, 255)),
                            4,
                        )

                        # Add frame counter
                        cv2.putText(
                            frame,
                            f"Frame: {frame_idx}",
                            (30, h_video - 30),
                            font,
                            1,
                            (200, 200, 200),
                            2,
                        )

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize slightly for rendering performance in streamlit
                        frame_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.7, fy=0.7)

                        video_display.image(
                            frame_rgb, channels="RGB", use_container_width=True
                        )

                    progress_pct = min((frame_idx + 1) / total_frames_video, 1.0)
                    progress_bar.progress(
                        progress_pct,
                        text=f"Sedang menganalisis... {int(progress_pct*100)}%",
                    )
                    frame_idx += 1

                cap.release()
                os.remove(tfile.name)

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
                        
                        # Tabel detail hasil
                        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
                        st.markdown("<h4 style='color:#c9d1d9;'>Detail Hasil Analisis</h4>", unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Tidak ada aktivitas yang terdeteksi.")


if __name__ == "__main__":
    main()
