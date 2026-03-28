import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import pandas as pd
from collections import Counter
import tempfile
import os
import time
import altair as alt
from pathlib import Path

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
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(BASE_DIR / "dataset" / "model" / "mobilenetv3_final_finetuned.keras")
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
# ENABLE CACHING UNTUK KECEPATAN
@st.cache_resource
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


def split_roi_into_zones(roi_frame, num_zones=2, orientation='Horizontal'):
    """
    Split ROI frame menjadi multiple zones (Left-Right atau Top-Bottom).
    Returns: list of (zone_frame, zone_bbox_relative)
    """
    h, w = roi_frame.shape[:2]
    zones = []
    
    if num_zones == 1:
        zones.append((roi_frame, (0, 0, w, h)))
    elif orientation == 'Horizontal':
        # Split left-right (Horizontal division)
        zone_w = w // num_zones
        for i in range(num_zones):
            x1 = i * zone_w
            x2 = (i + 1) * zone_w if i < num_zones - 1 else w
            zones.append((roi_frame[:, x1:x2], (x1, 0, x2, h)))
    else:
        # Split top-bottom (Vertical division)
        zone_h = h // num_zones
        for i in range(num_zones):
            y1 = i * zone_h
            y2 = (i + 1) * zone_h if i < num_zones - 1 else h
            zones.append((roi_frame[y1:y2, :], (0, y1, w, y2)))
    
    return zones

def detect_multiple_persons(fgmask, roi_top, roi_bottom, roi_left, roi_right, min_area=1000):
    """
    DEPRECATED - Using zone-based detection instead of motion contours.
    Keep for backward compatibility but not used.
    """
    return []

def preprocess_frame(frame):
    """
    Standard Preprocessing: BGR -> RGB -> Resize -> preprocess_input.
    (Cocok dengan ImageDataGenerator di Colab yang TIDAK menggunakan rescale=1/255)
    """
    # 1. BGR ke RGB (OpenCV read BGR, Model trained on RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Resize ke 224x224
    resized = cv2.resize(rgb, IMG_SIZE)
    
    # 3. Keras Preprocessing (Handles channel scaling correctly for MobileNetV3)
    # MobileNetV3 di Keras biasanya identity atau internal rescaling.
    img_array = resized.astype(np.float32)
    img_array = preprocess_input(img_array)
    
    # 4. Add batch dim (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

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
        
        st.subheader("⚡ Mode Kecepatan")
        quality_mode = st.radio(
            "Pilih mode processing:",
            options=['🚀 Cepat', '⚖️ Seimbang', '🎯 Akurat'],
            help="Cepat = lebih banyak frame skip, Akurat = proses semua frame"
        )
        
        # Dynamic skip_frames based on quality mode
        mode_skip = {'🚀 Cepat': 15, '⚖️ Seimbang': 5, '🎯 Akurat': 1}
        default_skip = mode_skip.get(quality_mode, 5)
        
        st.subheader("Pengaturan")
        skip_frames = st.slider("Interval Sampling Frame", 1, 30, default_skip, help="Memproses setiap frame ke-n agar realtime")
        
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

        # Sliders with Session State (FIX: Add default values untuk slider)
        roi_top = st.slider("Batas Atas (Top)", 0, 1080, value=st.session_state.get('roi_top', 436), key='roi_top')
        roi_bottom = st.slider("Batas Bawah (Bottom)", 0, 1080, value=st.session_state.get('roi_bottom', 935), key='roi_bottom')
        roi_left = st.slider("Batas Kiri (Left)", 0, 1920, value=st.session_state.get('roi_left', 450), key='roi_left')
        roi_right = st.slider("Batas Kanan (Right)", 0, 1920, value=st.session_state.get('roi_right', 1282), key='roi_right')
        
        st.divider()
        st.subheader("👥 Konfigurasi Operator")
        num_operators = st.slider("Jumlah Operator/Zona", 1, 4, 1, help="Jumlah area mandiri yang akan dideteksi dalam ROI")
        split_orientation = st.selectbox("Orientasi Pembagian", ["Horizontal (Kiri-Kanan)", "Vertical (Atas-Bawah)"], index=0)
        orientation_val = 'Horizontal' if "Horizontal" in split_orientation else 'Vertical'

        st.divider()
        st.subheader("🛠️ Debugging")
        show_debug = st.checkbox("Tampilkan AI Vision (Apa yang dilihat Robot)", value=False)
        
        st.divider()
        st.info(f"""
        **🎯 Status Deteksi:**
        - **Mode:** {'⚡ Per Frame (Accurate)' if skip_frames == 1 else '⏩ Optimized Inference'}
        - **Operator:** {num_operators} Zona ({orientation_val})
        - **Model:** MobileNetV3 (Fine-tuned)
        
        **Klasifikasi:**
        - Tiap operator dideteksi secara mandiri tanpa menggunakan sensor gerakan (PURE AI).
        - Hasil disimpan per frame untuk laporan akhir.
        """)

    
    # Load Model
    model = load_model()
    if not model:
        st.error(
            f"❌ Model `{MODEL_PATH}` tidak ditemukan!\n\n"
            "**Solusi:**\n"
            "1. Download file `mobilenetv3_final_finetuned.keras` dari Google Drive.\n"
            f"2. Pindahkan file tersebut ke folder: `{BASE_DIR / 'dataset' / 'model'}`"
        )
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
                progress_text = st.empty()
                progress_bar = st.progress(0)
            
            with col_realtime:
                st.subheader("Klasifikasi Real-time")
                # STATIC PLACEHOLDERS
                status_placeholder = st.empty()
                metrics_placeholder = st.empty()
                history_placeholder = st.empty()
                
            # Storage results - with zone-based detection
            results = [] 
            # Per-zone results (to show separate metrics for each operator)
            zone_results = {i: [] for i in range(num_operators)}
            
            frame_counts = {'Idle': 0, 'Bekerja': 0, 'Meninggalkan Area': 0}
            zone_activities = {}  # Track aktivitas per zone {zone_idx: {'label', 'conf', ...}}
            
            # --- PROCESSING LOOP ---
            frame_idx = 0
        
            # ROI Variables (Mapped from Sliders)
            ROI_TOP, ROI_BOTTOM = roi_top, roi_bottom
            ROI_LEFT, ROI_RIGHT = roi_left, roi_right

            # Default vars initialization
            label = 'Idle'
            display_text = "Init..."
            raw_preds = np.array([0.0, 0.0, 0.0])
            
            # OPTIMIZATION: Reduce display resolution untuk kecepatan
            display_scale = 0.5 if quality_mode == '🚀 Cepat' else (0.75 if quality_mode == '⚖️ Seimbang' else 1.0)
            
            
            # Number of zones per ROI
            NUM_ZONES = num_operators

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                h_frame, w_frame, _ = frame.shape
                
                
                # --- PROCESSING EVERY N-th FRAME ---
                if frame_idx % skip_frames == 0:
                    
                    # 1. Crop ROI (Kotak Hijau - Area Operator)
                    roi_frame = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
                    
                    # Handle jika ROI 0 size
                    if roi_frame.size == 0: 
                        roi_frame = frame
                        zone_activities = {}
                    else:
                        # 2. Split ROI menjadi multiple zones
                        zones = split_roi_into_zones(roi_frame, NUM_ZONES, orientation_val)
                        zone_activities = {}  # Reset per frame
                        
                    # 3. Classify TIAP ZONE dengan MobileNetV3 (PURE MODEL OUTPUT)
                        for zone_idx, (zone_frame, zone_bbox_rel) in enumerate(zones):
                            if zone_frame.size == 0:
                                continue
                            
                            try:
                                # Preprocess & Predict
                                processed_input = preprocess_frame(zone_frame)
                                preds = model.predict(processed_input, verbose=0)
                                raw_preds_zone = preds[0]
                                
                                # Get result - PURE MODEL OUTPUT
                                label_idx = np.argmax(raw_preds_zone)
                                conf = raw_preds_zone[label_idx]
                                label = CLASSES[label_idx]
                                
                                # Store zone activity - NO MOTION OVERRIDE
                                zone_activities[zone_idx] = {
                                    'label': label,
                                    'conf': conf,
                                    'raw_preds': raw_preds_zone,
                                    'bbox_rel': zone_bbox_rel  # Relative to ROI
                                }
                            except Exception as e:
                                continue
                    
                    # Determine final activity (PURE MODEL-BASED)
                    if zone_activities:
                        # Aggregate dari semua zones (majority voting)
                        all_labels = [info['label'] for info in zone_activities.values()]
                        most_common = Counter(all_labels).most_common(1)
                        final_label = most_common[0][0] if most_common else 'Idle'
                        
                        # Average confidence dari zones
                        avg_conf = np.mean([info['conf'] for info in zone_activities.values()])
                        final_conf = avg_conf
                    else:
                        # Fallback jika ROI tidak bisa dibaca sama sekali
                        final_label = 'Meninggalkan Area'
                        final_conf = 0.0
                    
                    # Store result (Total & Per Zone)
                    timestamp = float(frame_idx) / fps if fps > 0 else 0.0
                    
                    # Store per zone for the report
                    if zone_activities:
                        for z_idx, z_info in zone_activities.items():
                            zone_results[z_idx].append({
                                "Waktu": round(float(timestamp), 2),
                                "Aktivitas": z_info['label'],
                                "Confidence": round(float(z_info['conf']), 3)
                            })
                    else:
                        # If no zones (leaving area), add to all? Or just skip
                        for z_idx in range(NUM_ZONES):
                            zone_results[z_idx].append({
                                "Waktu": round(float(timestamp), 2),
                                "Aktivitas": "Meninggalkan Area",
                                "Confidence": 0.95
                            })

                    results.append({
                        "Waktu": round(float(timestamp), 2),
                        "Aktivitas": final_label,
                        "Confidence": round(float(final_conf), 3)
                    })
                    
                    if final_label in frame_counts:
                        frame_counts[final_label] += 1
                
                # --- VISUALIZATION (Draw on Frame) ---
                # DRAW MAIN ROI BOX
                color_roi = (0, 255, 0)  # Green untuk ROI
                cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), color_roi, 3)
                
                # Draw zone boundaries INSIDE ROI
                if zone_activities:
                    roi_h = ROI_BOTTOM - ROI_TOP
                    roi_w = ROI_RIGHT - ROI_LEFT
                    
                    # Draw vertical/horizontal lines for split
                    if NUM_ZONES > 1:
                        for i in range(1, NUM_ZONES):
                            if orientation_val == 'Horizontal':
                                mid_x = ROI_LEFT + (i * roi_w // NUM_ZONES)
                                cv2.line(frame, (mid_x, ROI_TOP), (mid_x, ROI_BOTTOM), (100, 100, 100), 2)
                            else:
                                mid_y = ROI_TOP + (i * roi_h // NUM_ZONES)
                                cv2.line(frame, (ROI_LEFT, mid_y), (ROI_RIGHT, mid_y), (100, 100, 100), 2)
                    
                    # Draw label untuk TIAP ZONE
                    for zone_idx, zone_info in zone_activities.items():
                        bbox_rel = zone_info['bbox_rel']
                        x1_rel, y1_rel, x2_rel, y2_rel = bbox_rel
                        
                        # Convert to global frame coords
                        x1 = int(ROI_LEFT) + int(x1_rel)
                        y1 = int(ROI_TOP) + int(y1_rel)
                        x2 = int(ROI_LEFT) + int(x2_rel)
                        y2 = int(ROI_TOP) + int(y2_rel)
                        
                        label_zone = zone_info['label']
                        conf_zone = zone_info['conf']
                        color = CV2_COLORS.get(label_zone, (200, 200, 200))
                        
                        # Draw zone label text
                        label_text = f"Z{zone_idx}: {label_zone} ({conf_zone:.0%})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 1
                        
                        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                        text_x = x1 + 5
                        text_y = y1 + 20 + (zone_idx * 25)
                        
                        # Background text
                        cv2.rectangle(frame,
                                     (text_x - 3, text_y - text_size[1] - 3),
                                     (text_x + text_size[0] + 3, text_y + 3),
                                     color, -1)
                        
                        # Draw text
                        cv2.putText(frame, label_text, (text_x, text_y),
                                   font, font_scale, (255, 255, 255), thickness)
                
                # --- UPDATE UI ---
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # OPTIMIZATION: Resize untuk display
                if display_scale < 1.0:
                    h_display = int(frame_rgb.shape[0] * display_scale)
                    w_display = int(frame_rgb.shape[1] * display_scale)
                    frame_rgb = cv2.resize(frame_rgb, (w_display, h_display), interpolation=cv2.INTER_LINEAR)
                
                video_display.image(frame_rgb, use_container_width=True)
                
                # Status Card - Show all zone activities
                with status_placeholder.container():
                    if zone_activities:
                        for zone_idx, info in sorted(zone_activities.items()):
                            zone_title = "Operator" if NUM_ZONES == 1 else f"Zona {zone_idx}"
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 5px solid {COLOR_MAP.get(info['label'], '#ccc')};">
                                <h3>📍 {zone_title}</h3>
                                <h2 class="big-label" style="color: {COLOR_MAP.get(info['label'], '#fff')};">{info['label']}</h2>
                                <p class="sub-label">Confidence: {info['conf']:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2 class="big-label">Menunggu Data...</h2>
                            <p class="sub-label">Sesuaikan ROI jika tidak muncul hasil</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Chart Probs - Aggregate dari semua zones
                with metrics_placeholder.container():
                    if zone_activities:
                        # Average predictions dari all zones
                        avg_preds = np.zeros(3)
                        for info in zone_activities.values():
                            avg_preds += info['raw_preds']
                        avg_preds /= max(len(zone_activities), 1)
                        
                        df_probs = pd.DataFrame({'Activity': CLASSES, 'Probability': avg_preds})
                    else:
                        df_probs = pd.DataFrame({
                            'Activity': CLASSES,
                            'Probability': [0.0, 0.0, 1.0]  # Meninggalkan Area
                        })
                    
                    c = alt.Chart(df_probs).mark_bar().encode(
                        x=alt.X('Probability', scale=alt.Scale(domain=[0,1])),
                        y=alt.Y('Activity', sort=None),
                        color=alt.Color('Activity', scale=alt.Scale(domain=CLASSES, range=[COLOR_MAP[c] for c in CLASSES]), legend=None)
                    ).properties(height=200)
                    st.altair_chart(c, use_container_width=True)
                
                # Progress Update
                progress_pct = min((frame_idx + 1) / total_frames_video, 1.0)
                progress_bar.progress(progress_pct)
                progress_text.text(f"⏳ {int(progress_pct * 100)}% - Frame {frame_idx + 1}/{total_frames_video} | Zones Detected: {len(zone_activities)}")
                frame_idx += 1

            cap.release()
            
            # --- CLEANUP UI ---
            # Menghapus status real-time agar fokus ke laporan akhir
            status_placeholder.empty()
            metrics_placeholder.empty()
            
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
                        c2.metric("Aktivitas Dominan", df['Aktivitas'].mode()[0])
                        
                        # Activity counts
                        count_df = df['Aktivitas'].value_counts().reset_index()
                        count_df.columns = ['Aktivitas', 'Jumlah']
                        
                        st.subheader("📈 Breakdown Aktivitas")
                        for _, row in count_df.iterrows():
                            pct = (row['Jumlah'] / total_frames_processed) * 100
                            st.write(f"  • **{row['Aktivitas']}**: {row['Jumlah']} frame ({pct:.1f}%)")
                        
                        # Pie chart
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

                    # --- DETAILED PER-ZONE REPORT ---
                    st.divider()
                    st.subheader("👥 Analisis Detail per Operator")
                    
                    tabs = st.tabs([f"Operator/Zona {i}" for i in range(NUM_ZONES)])
                    for i, tab in enumerate(tabs):
                        with tab:
                            if zone_results[i]:
                                zdf = pd.DataFrame(zone_results[i])
                                total_z = len(zdf)
                                mode_z = zdf['Aktivitas'].mode()[0]
                                
                                col1, col2 = st.columns(2)
                                col1.metric(f"Total Frame Z{i}", total_z)
                                col2.metric(f"Aktivitas Utama Z{i}", mode_z)
                                
                                # Counts per zone
                                z_counts = zdf['Aktivitas'].value_counts().reset_index()
                                z_counts.columns = ['Aktivitas', 'Jumlah']
                                
                                st.markdown(f"**Breakdown Aktivitas Operator {i}:**")
                                for _, row in z_counts.iterrows():
                                    st.write(f"- {row['Aktivitas']}: {row['Jumlah']} frame ({row['Jumlah']/total_z:.1%})")
                                
                                # Mini Chart per zone
                                z_chart = alt.Chart(z_counts).mark_bar().encode(
                                    x='Jumlah',
                                    y=alt.Y('Aktivitas', sort='-x'),
                                    color=alt.Color('Aktivitas', scale=alt.Scale(domain=CLASSES, range=[COLOR_MAP[c] for c in CLASSES]))
                                ).properties(height=150)
                                st.altair_chart(z_chart, use_container_width=True)
                            else:
                                st.info(f"Belum ada data untuk Zona {i}")
            
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
