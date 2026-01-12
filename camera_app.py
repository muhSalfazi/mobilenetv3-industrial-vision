import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import tempfile
import os
import time
import altair as alt

# ============= 1. KONFIGURASI HALAMAN & STYLE =============
st.set_page_config(
    page_title="Activity Classifier AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan Laporan Akhir yang bersih
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .big-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .sub-label {
        font-size: 1.0rem;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# ============= 2. KONFIGURASI MODEL =============
# Menggunakan model hasil Fine-Tuning terakhir
MODEL_PATH = "mobilenetv3_final_finetuned.keras"
IMG_SIZE = (224, 224)
# Urutan sesuai training generator (alphabetical order)
CLASSES = ['Bekerja', 'Idle', 'Meninggalkan Area']

# Warna UI & Grafik
COLOR_MAP = {
    'Idle': '#FFD700',           # Pure Yellow (Gold)
    'Bekerja': '#00FF00',        # Bright Green
    'Meninggalkan Area': '#FF0000' # Bright Red
}
# Warna Text di Video (BGR)
CV2_COLORS = {
    'Idle': (0, 215, 255),       # Yellow/Gold (BGR)
    'Bekerja': (0, 255, 0),      # Bright Green (BGR)
    'Meninggalkan Area': (0, 0, 255) # Bright Red (BGR)
}

# ============= 3. FUNGSI UTILITY =============
# NOTE: Cache dimatikan sementara agar jika file model baru masuk, langsung terbaca tanpa restart app
# @st.cache_resource
def load_model():
    # Debugging Info (Akan muncul di sidebar jika error)
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"❌ DEBUG: File '{MODEL_PATH}' TIDAK DITEMUKAN!")
        st.sidebar.warning(f"Lokasi pencarian: {os.getcwd()}")
        try:
            files = os.listdir('.')
            st.sidebar.info(f"File yang ada di folder ini: {files}")
        except:
            pass
            
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    return None

# Import preprocess_input from keras implementation
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def preprocess_frame(frame):
    # PURE STANDARD MobileNetV3 Preprocessing
    # 1. BGR ke RGB (Wajib karena OpenCV baca BGR, Model train RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Resize ke 224x224 (Wajib sesuai input shape model)
    resized = cv2.resize(rgb, IMG_SIZE)
    
    # 3. Convert ke Array
    img_array = resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dim (1, 224, 224, 3)
    
    # 4. Keras Official Preprocessing (Handle normalization -1 to 1 automatically)
    final_input = preprocess_input(img_array)
    
    return final_input

# ============= 4. APP UTAMA =============
def main():
    # HEADER
    st.title("🏭 Identifikasi Aktivitas Operator Mesin pada Rekaman CCTV")
    st.markdown("Muhamad Salman Fauzi | 22416255201063")
    st.divider()

    # --- MAIN FILE UPLOADER ---
    with st.container():
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            st.markdown("### 📂 Input Data Video")
            st.markdown("Silakan unggah rekaman CCTV (format `.mp4` atau `.avi`) yang ingin dianalisis.")
            uploaded_file = st.file_uploader("", type=['mp4', 'avi'], label_visibility="collapsed")
        
        with col_up2:
            st.info("💡 **Panduan Singkat**:\n\n1. Upload video di sebelah kiri.\n2. Sesuaikan **ROI (Kotak Hijau)** di sidebar agar fokus ke operator.\n3. Klik tombol **'Mulai Identifikasi'**.")

    if not uploaded_file:
        st.warning("⚠️ Belum ada video yang diupload. Silakan upload video untuk memulai.")


    # SIDEBAR CONFIGURATION
    with st.sidebar:
        st.header("⚙️ Panel Kontrol")
        # uploader removed from here
        
        st.subheader("Pengaturan")
        skip_frames = st.slider("Interval Sampling Frame", 1, 30, 5, help="Memproses setiap frame ke-n agar realtime")
        
        st.divider()
        st.subheader("🔧 Konfigurasi Area Kerja")
        # st.info("🎯 **Fokus AI**: Wajib sesuaikan kotak hijau ke area operator. Ini membantu MobileNetV3 'fokus' melihat detail aktivitas (karena input model hanya 224x224 pixel).")
        
        # Initialize session state for ROI if not exists
        if 'roi_top' not in st.session_state: st.session_state['roi_top'] = 436
        if 'roi_bottom' not in st.session_state: st.session_state['roi_bottom'] = 935
        if 'roi_left' not in st.session_state: st.session_state['roi_left'] = 450
        if 'roi_right' not in st.session_state: st.session_state['roi_right'] = 1282

        # Reset Button (Default ROI - Adjust based on your typical CCTV view)
        if st.button("↺ Reset ROI ke Default"):
            st.session_state['roi_top'] = 436
            st.session_state['roi_bottom'] = 935
            st.session_state['roi_left'] = 450
            st.session_state['roi_right'] = 1282
            st.rerun()

        # Sliders with Session State
        roi_top = st.slider("Batas Atas (Top)", 0, 1080, key='roi_top')
        roi_bottom = st.slider("Batas Bawah (Bottom)", 0, 1080, key='roi_bottom')
        roi_left = st.slider("Batas Kiri (Left)", 0, 1920, key='roi_left')
        roi_right = st.slider("Batas Kanan (Right)", 0, 1920, key='roi_right')
        
        st.divider()
        st.subheader("🛠️ Debugging")
        show_debug = st.checkbox("Tampilkan AI Vision (Apa yang dilihat Robot)", value=False)
    
    # Load Model
    model = load_model()
    if not model:
        st.error(f"❌ Model `{MODEL_PATH}` tidak ditemukan!\n\n**Solusi:**\n1. Download file `mobilenetv3_final_finetuned.keras` dari Google Drive.\n2. Pindahkan file tersebut ke folder projek ini: `{os.getcwd()}`")
        # Retry button
        if st.button("🔄 Coba Reload Model"):
            st.rerun()
        return

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = total_frames_video / fps if fps > 0 else 0
        w_video = int(cap.get(3))
        h_video = int(cap.get(4))
        
        # Validasi ROI agar tidak error
        roi_top = max(0, min(roi_top, h_video-1))
        roi_bottom = max(roi_top+1, min(roi_bottom, h_video))
        roi_left = max(0, min(roi_left, w_video-1))
        roi_right = max(roi_left+1, min(roi_right, w_video))
        
        # INFO VIDEO
        st.markdown(f"### 📹 Video Input")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Frame", f"{total_frames_video:,}")
        c2.metric("Durasi", f"{duration:.1f} detik")
        c3.metric("FPS Asli", f"{fps:.1f}")
        c4.metric("Resolusi", f"{w_video}x{h_video}")

        if st.button("🚀 Mulai Identifikasi", type="primary", use_container_width=True):
            
            # --- COMPONENT LAYOUT ---
            st.divider()
            col_vid, col_realtime = st.columns([1.5, 1])
            
            with col_vid:
                st.subheader("Monitor CCTV")
                video_display = st.image([])
                progress_bar = st.progress(0)
            
            with col_realtime:
                st.subheader("Klasifikasi Real-time")
                # STATIC PLACEHOLDERS
                status_placeholder = st.empty()
                metrics_placeholder = st.empty()
                history_placeholder = st.empty()
                
            # Storage results
            results = []
            frame_counts = {'Idle': 0, 'Bekerja': 0, 'Meninggalkan Area': 0}
            
            # --- PROCESSING LOOP ---
            frame_idx = 0
        
            # ROI Variables (Mapped from Sliders)
            ROI_TOP, ROI_BOTTOM = roi_top, roi_bottom
            ROI_LEFT, ROI_RIGHT = roi_left, roi_right

            # Default vars initialization
            label = 'Idle'
            display_text = "Init..."
            raw_preds = np.array([0.0, 0.0, 0.0])
            
            # Motion Detector untuk visualisasi "Meninggalkan Area"
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                h_frame, w_frame, _ = frame.shape
                
                # Update background model strictly for every frame to keep it adapting
                fgmask = fgbg.apply(frame)
                
                # --- PROCESSING EVERY N-th FRAME ---
                if frame_idx % skip_frames == 0:
                    
                    # 1. Crop sesuai kotak hijau (Region of Interest)
                    # MobileNetV3 dilatih pada gambar orang close-up, jadi cropping wajib.
                    roi_frame = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
                    
                    # Handle jika ROI 0 size
                    if roi_frame.size == 0: roi_frame = frame
                    
                    # 2. Preprocess & Predict
                    processed_input = preprocess_frame(roi_frame)
                    
                    # 3. Model Prediction (PURE)
                    preds = model.predict(processed_input, verbose=0)
                    raw_preds = preds[0]  # [score_bekerja, score_idle, score_meninggalkan]
                    
                    # 4. Get Result
                    label_idx = np.argmax(raw_preds)
                    conf = raw_preds[label_idx]
                    label = CLASSES[label_idx]

                    # --- SPATIAL & MOTION VALIDATION (AUTO-CORRECT) ---
                    # Logic: Jika AI Prediksi "Meninggalkan Area", validasi dengan posisi bounding box.
                    # Jika orangnya masih di tengah (aman), force label jadi "Idle".
                    
                    is_leaving_spatially = False
                    detected_box = None
                    
                    if label == 'Meninggalkan Area':
                         # Post-process mask (Only need needed calculation here)
                        _, mask_thresh = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY)
                        kernel = np.ones((5,5), np.uint8)
                        mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
                        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
                        
                        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(c) > 1000:
                                detected_box = cv2.boundingRect(c)
                                x, y, w, h = detected_box
                                
                                # Check ROI overlap
                                margin = 20
                                is_touching_left = x <= (ROI_LEFT + margin)
                                is_touching_right = (x + w) >= (ROI_RIGHT - margin)
                                is_touching_top = y <= (ROI_TOP + margin)
                                is_touching_bottom = (y + h) >= (ROI_BOTTOM - margin)
                                
                                is_leaving_spatially = is_touching_left or is_touching_right or is_touching_top or is_touching_bottom

                        # FINAL DECISION: Check Correctness
                        if detected_box:
                            x, y, w, h = detected_box

                            # Validate 1: Apakah gerakan ini valid (beririsan dengan ROI)?
                            # Jika kotak gerakannya 100% di luar ROI, berarti itu noise/mesin lain.
                            is_completely_outside = (x > ROI_RIGHT) or (x + w < ROI_LEFT) or \
                                                    (y > ROI_BOTTOM) or (y + h < ROI_TOP)
                            
                            if is_completely_outside:
                                label = 'Idle' # Force Idle untuk gerakan diluar area
                                
                            # Validate 2: Jika AI bilang Leave, tapi secara spasial TIDAK leave (masih di tengah)
                            elif not is_leaving_spatially:
                                 # Cek apakah dia benar-benar ada di dalam ROI (center point)
                                cx = x + w//2
                                cy = y + h//2
                                if (ROI_LEFT < cx < ROI_RIGHT) and (ROI_TOP < cy < ROI_BOTTOM):
                                    label = 'Idle' # Override
                                    display_text = f"Idle (Auto-Corrected) ({conf:.1%})"
                    
                    # Track Stats (Now using the potentially corrected 'label')
                    if label in frame_counts:
                         frame_counts[label] += 1
                    
                    # Simpan data log
                    timestamp = frame_idx / fps if fps > 0 else 0
                    results.append({
                        "Waktu": round(timestamp, 2),
                        "Aktivitas": label,
                        "Confidence": round(conf, 3)
                    })
                
                # --- VISUALIZATION (Draw on Frame) ---
                color = CV2_COLORS.get(label, (200, 200, 200))
                
                # DEBUG: Visualisasi Bounding Box jika ada (Tech Style Only if Leaving is confirmed)
                if 'detected_box' in locals() and detected_box and label == 'Meninggalkan Area':
                     x, y, w, h = detected_box
                     # Gambar Tech Box
                     l_len = int(min(w, h) * 0.2)
                     top_left = (x, y)
                     bottom_right = (x+w, y+h)
                     box_color = CV2_COLORS['Meninggalkan Area']
                     
                     # Rect tipis
                     cv2.rectangle(frame, top_left, bottom_right, box_color, 1)
                     
                     # Corners tebal
                     cv2.line(frame, (x, y), (x + l_len, y), box_color, 4)
                     cv2.line(frame, (x, y), (x, y + l_len), box_color, 4)
                     cv2.line(frame, (x+w, y), (x+w - l_len, y), box_color, 4)
                     cv2.line(frame, (x+w, y), (x+w, y + l_len), box_color, 4)
                     cv2.line(frame, (x, y+h), (x + l_len, y+h), box_color, 4)
                     cv2.line(frame, (x, y+h), (x, y+h - l_len), box_color, 4)
                     cv2.line(frame, (x+w, y+h), (x+w - l_len, y+h), box_color, 4)
                     cv2.line(frame, (x+w, y+h), (x+w, y+h - l_len), box_color, 4)
                     
                     cv2.putText(frame, "Meninggalkan Area", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # ROI Box (Hijau - Area Kerja)
                cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
                
                # Overlay Info
                cv2.rectangle(frame, (0, 0), (w_frame, 60), color, -1)
                cv2.putText(frame, f"AI PREDICTION: {label}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                # --- UPDATE UI ---
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_display.image(frame_rgb, use_container_width=True)
                
                # Debug View (Show what AI sees)
                if show_debug and 'roi_frame' in locals():
                    st.sidebar.image(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB), caption="Input Model (224x224)", use_container_width=True)
                
                # Status Card
                with status_placeholder.container():
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid {COLOR_MAP.get(label, '#ccc')};">
                        <h2 class="big-label" style="color: {COLOR_MAP.get(label, '#fff')};">{label}</h2>
                        <p class="sub-label">Confidence: {raw_preds[np.argmax(raw_preds)]:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Chart Probs
                with metrics_placeholder.container():
                    df_probs = pd.DataFrame({'Activity': CLASSES, 'Probability': raw_preds})
                    c = alt.Chart(df_probs).mark_bar().encode(
                        x=alt.X('Probability', scale=alt.Scale(domain=[0,1])),
                        y=alt.Y('Activity', sort=None),
                        color=alt.Color('Activity', scale=alt.Scale(domain=CLASSES, range=[COLOR_MAP[c] for c in CLASSES]), legend=None)
                    ).properties(height=200)
                    st.altair_chart(c, use_container_width=True)
                
                # Progress Update
                progress_bar.progress(min(frame_idx / total_frames_video, 1.0))
                frame_idx += 1

            cap.release()
            
            # --- FINAL SUMMARY ---
            st.success("✅ Analisis Video Selesai.")
            if results:
                df = pd.DataFrame(results)
                
                st.divider()
                st.subheader("📊 Laporan Statistik Akhir")
                
                total_frames_processed = len(df)
                if total_frames_processed > 0:
                    c1, c2 = st.columns(2)
                    c1.metric("Total Sampel", total_frames_processed)
                    if not df['Aktivitas'].empty:
                        c2.metric("Dominant Activity", df['Aktivitas'].mode()[0])
                        
                        # Simple Pie Chart
                        count_df = df['Aktivitas'].value_counts().reset_index()
                        count_df.columns = ['Aktivitas', 'Jumlah']
                        
                        base = alt.Chart(count_df).encode(theta=alt.Theta("Jumlah", stack=True))
                        pie = base.mark_arc(outerRadius=120).encode(
                            color=alt.Color("Aktivitas", scale=alt.Scale(domain=CLASSES, range=[COLOR_MAP[c] for c in CLASSES])),
                            order=alt.Order("Jumlah", sort="descending"),
                            tooltip=["Aktivitas", "Jumlah"]
                        )
                        text = base.mark_text(radius=140).encode(
                            text=alt.Text("Jumlah"),
                            order=alt.Order("Jumlah", sort="descending"),
                            color=alt.value("white")
                        )
                        st.altair_chart(pie + text, use_container_width=True)
                    else:
                        st.warning("Tidak ada aktivitas terdeteksi.")
                        
                    with st.expander("Lihat Data Log Detik-ke-Detik"):
                        st.dataframe(df)
            
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
