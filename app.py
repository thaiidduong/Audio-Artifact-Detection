import streamlit as st
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder
import os
from pydub import AudioSegment

def compress_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)

    # export sang MP3 bitrate thấp
    audio.export(output_path, format="mp3", bitrate="8k")
# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Audio Artifact Detector Pro", layout="wide")

# =========================
# CSS + UI ENHANCEMENTS
# =========================
st.markdown("""
<style>
                 
            /* BUTTON STYLE (glass) */
button {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Hover */
button:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}
    .stApp { background: linear-gradient(135deg, #1a1c35, #29648a); color: #E0E0E0; }
    
            /* ANIMATED SOUND WAVE */
    .visualizer-container {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        height: 40px;
        gap: 4px;
        margin-bottom: -10px;
        padding: 0 20px;
    }

    .bar {
        width: 6px;
        height: 5px;
        background: linear-gradient(to top, #4facfe, #00f2fe);
        border-radius: 3px;
        animation: sound-wave 1.5s infinite ease-in-out;
    }

     /* Tạo hiệu ứng ngẫu nhiên cho 20 thanh bar để trải dài hơn */
    .bar:nth-child(2n) { animation-duration: 1.2s; }
    .bar:nth-child(3n) { animation-duration: 1.8s; }
    .bar:nth-child(4n) { animation-duration: 1.4s; }
    
    /* Delay từng nhóm để tạo hiệu ứng lượn sóng */
    .bar:nth-child(odd) { animation-delay: 0.2s; }
    .bar:nth-child(even) { animation-delay: 0.4s; }
    .bar:nth-child(3n+1) { animation-delay: 0.6s; }

    @keyframes sound-wave {
        0%, 100% { height: 5px; opacity: 0.3; }
        50% { height: 40px; opacity: 1; }
    }
            
    /* GLASS CARD */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

/* Chuyển Box Upload sang màu trong suốt */
    [data-testid="stFileUploaderDropzone"] {
        background-color: transparent !important; /* Trong suốt hoàn toàn */
        border: 1px solid rgba(255, 255, 255, 0.2) !important; /* Viền xám mảnh */
        border-radius: 10px !important;
        transition: none !important;
    }

    /* Khóa trạng thái khi di chuột để không bị hiện màu trắng */
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploaderDropzone"]:active {
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important; /* Viền sáng hơn một chút khi hover */
    }

    /* Giữ chữ và icon luôn trắng rõ nét */
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    /* Nút bấm bên trong cũng nên để trong suốt nhẹ */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: none !important;
    }
            
    /* CUSTOM ALERTS */
    .stAlert { color: white !important; }
    
    /* TEXT COLORS */
    p, li, span, label, .stMarkdown { color: #E0E0E0 !important; }
    .title-glow {
        font-size: 70px; font-weight: 900; text-align: center;
        background: linear-gradient(90deg, #fff, #4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* HEADER */
    .custom-header {
        position: fixed; top: 0; left: 0; width: 100%; height: 60px;
        background: rgba(20, 30, 48, 0.9); display: flex;
        justify-content: space-between; align-items: center; padding: 0 40px; z-index: 9999;
    }
</style>
<div class="custom-header">
    <div style="color:white; font-weight:600;">Audio Artifact Detection Pro</div>
    <div style="color:#aaa; font-size:13px;">Nguyen Anh Thai Duong • HUST</div>
</div>
<div style="margin-top:80px;"></div>
""", unsafe_allow_html=True)

# =========================
# MODEL DEFINITION
# =========================
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,1)),
        )
        self.fc_time = nn.Linear(64, 2)
    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        x = x.permute(0, 2, 1)
        return self.fc_time(x)

@st.cache_resource
def load_model():
    model = BetterCNN()
    if os.path.exists("models/model.pth"):
        model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# MAIN APP HEADING
# =========================
# Tạo danh sách các thanh bar (ở đây mình dùng 28 thanh để trải dài)
bars_html = "".join(['<div class="bar"></div>' for _ in range(28)])

st.markdown(f"""
<div class="visualizer-container">
    {bars_html}
</div>
<div class='title-glow'>Audio Artifact Detection</div>
""", unsafe_allow_html=True)

col_input, col_result = st.columns([1, 2])

with col_input:
    st.markdown("### Input Control")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # 1. Tạo hàm dọn dẹp bộ nhớ
    def clear_old_results():
        if "analysis_results" in st.session_state:
            st.session_state.analysis_results = None

    # 2. Thêm lựa chọn nguồn đầu vào (Gắn kèm hàm dọn dẹp vào on_change)
    input_mode = st.radio(
        "Select Source:", 
        ["File Upload", "Live Recording"],
        on_change=clear_old_results
    )

    # Biến để lưu đường dẫn file sẽ xử lý
    source_audio_path = "temp_raw.wav" 
    ready_to_analyze = False

    if input_mode == "File Upload":
        uploaded_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
        if uploaded_file:
            with open(source_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.audio(uploaded_file)
            ready_to_analyze = True
    else:
        st.write("Record your clean audio:")
        # Widget ghi âm trực tiếp
        audio_record = mic_recorder(start_prompt="⏺ Start Recording", stop_prompt="Stop", key='recorder')
        if audio_record:
            with open(source_audio_path, "wb") as f:
                f.write(audio_record['bytes'])
            st.audio(audio_record['bytes'])
            ready_to_analyze = True

     # Thêm nút bấm Nén và Phân tích để tạo luồng "Real-time"
    if ready_to_analyze:
        if input_mode == "File Upload":
            button_label = " ANALYZE"
        else:
            button_label = " COMPRESS & ANALYZE"

        analyze_btn = st.button(button_label, use_container_width=True)
    else:
        analyze_btn = False

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PHẦN XỬ LÝ LOGIC & LƯU TRẠNG THÁI (SESSION STATE)
# ==========================================

# 1. Khởi tạo bộ nhớ tạm để giữ kết quả không bị mất khi ấn Toggle
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if analyze_btn:
    with st.spinner("Deep Analysis in progress..."):
        # ===== CHỌN FILE =====
        if input_mode == "Live Recording":
            compressed_path = "temp_compressed.mp3"
            compress_audio(input_path=source_audio_path, output_path=compressed_path)
            audio_path_to_use = compressed_path
            st.write("Compressed audio:")
            st.audio(compressed_path)
        else:
            audio_path_to_use = source_audio_path

        # ===== LOAD AUDIO & FEATURE =====
        y, sr = librosa.load(audio_path_to_use, sr=22050, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.stack([mfcc, delta, delta2])
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        # ===== PREDICT =====
        with torch.no_grad():
            output = model(X)
            probs = torch.softmax(output, dim=-1)[0].numpy()

        # 2. LƯU TẤT CẢ KẾT QUẢ VÀO SESSION STATE
        st.session_state.analysis_results = {
            "artifact_probs_series": probs[:, 1],
            "avg_score": float(np.mean(probs[:, 1])),
            "y": y,
            "sr": sr,
            "mfcc_var": np.var(mfcc)
        }

# ==========================================
# PHẦN HIỂN THỊ GIAO DIỆN (Lấy từ Session State)
# ==========================================

# Chỉ hiển thị kết quả nếu đã có dữ liệu trong bộ nhớ
if st.session_state.analysis_results is not None:
    # Lấy dữ liệu ra
    res = st.session_state.analysis_results
    artifact_probs_series = res["artifact_probs_series"]
    avg_score = res["avg_score"]
    y = res["y"]
    sr = res["sr"]
    mfcc_var = res["mfcc_var"]

    is_compressed = avg_score > 0.5
    theme_color = "#FF4B4B" if is_compressed else "#00FF7F" 
    status_text = "COMPRESSED AUDIO DETECTED" if is_compressed else "CLEAN AUDIO VERIFIED"

    with col_result:
        if is_compressed:
            display_icon = "🚨"
            bg_box = "rgba(255, 75, 75, 0.2)"
            border_box = "#FF4B4B"
            desc = "The system detected signs of significant lossy compression or high-frequency degradation."
        else:
            display_icon = "🟢"
            bg_box = "rgba(0, 255, 127, 0.2)"
            border_box = "#00FF7F"
            desc = "High-fidelity audio signal. No characteristic compression artifacts were detected."

        st.markdown(f"### Analysis Result: <span style='color:{theme_color}'>{status_text}</span>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Overview", "Technical Metrics"])
        
        with tab1:
            r1, r2 = st.columns([1, 1.2])
            with r1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=avg_score*100,
                    number={'suffix': "%", 'font': {'color': theme_color}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"},
                           'bar': {'color': theme_color},
                           'steps': [{'range': [0, 50], 'color': "rgba(0, 255, 127, 0.1)"},
                                     {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.1)"}]}
                ))
                fig_g.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_g, use_container_width=True)
            
            with r2:
                st.markdown(f"""
                    <div style="background-color:{bg_box}; border: 1px solid {border_box}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <span style="font-size: 1.2rem; margin-right: 8px;">{display_icon}</span>
                        <strong style="color:{border_box};">{status_text}</strong><br>
                        <div style="margin-top: 8px; font-size: 0.85rem; color: #E0E0E0; line-height: 1.4;">{desc}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.metric("Probability Score", f"{avg_score:.4f}")

        with tab2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Sample Rate", f"{sr/1000} kHz")
            m2.metric("Feature Variance", f"{mfcc_var:.2f}")
            m3.metric("Confidence Level", "High" if abs(avg_score-0.5)>0.2 else "Low")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # VISUALIZATIONS
    # =========================
    st.markdown("### Deep Signal Analysis")
    
    # Chỉ cho phép hiện Highlight nếu kết quả dự đoán trung bình > threshold (tức là máy báo lỗi)
    if avg_score > 0.5:
        highlight_compress = st.toggle(" Highlight compressed area", value=False)
    else:
        # Nếu file sạch, tắt highlight và hiện dòng thông báo nhẹ cho người dùng
        highlight_compress = False
        st.markdown("<p style='font-size: 0.85rem; color: #00FF7F; opacity: 0.8;'> No significant artifacts detected to highlight.</p>", unsafe_allow_html=True)

    threshold = 0.5 

    v1, v2 = st.columns(2)
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white", "figure.facecolor": (0,0,0,0), "axes.facecolor": (0,0,0,0)})

    with v1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption("Artifact Probability Over Time")
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(y=artifact_probs_series, mode='lines', line=dict(color=theme_color, width=2), fill='tozeroy', fillcolor=f'rgba{tuple(list(int(theme_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'))
        
        if highlight_compress:
            start_idx = None
            for i, prob in enumerate(artifact_probs_series):
                if prob > threshold and start_idx is None:
                    start_idx = i
                elif prob <= threshold and start_idx is not None:
                    fig_l.add_vrect(x0=start_idx, x1=i, fillcolor="red", opacity=0.25, line_width=0)
                    start_idx = None
            if start_idx is not None:
                fig_l.add_vrect(x0=start_idx, x1=len(artifact_probs_series)-1, fillcolor="red", opacity=0.25, line_width=0)

        fig_l.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, xaxis_title="Time Frames", yaxis_title="Probability")
        st.plotly_chart(fig_l, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption("Power Spectrogram (dB)")
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
        
        if highlight_compress:
            times = librosa.frames_to_time(np.arange(len(artifact_probs_series)), sr=sr)
            start_idx = None
            for i, prob in enumerate(artifact_probs_series):
                if prob > threshold and start_idx is None:
                    start_idx = i
                elif prob <= threshold and start_idx is not None:
                    ax.axvspan(times[start_idx], times[i], color='red', alpha=0.3)
                    start_idx = None
            if start_idx is not None:
                ax.axvspan(times[start_idx], times[-1], color='red', alpha=0.3)

        plt.colorbar(img, ax=ax, format="%+2.f dB").ax.yaxis.set_tick_params(color='white')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown(f"<div style='text-align:center; padding: 30px; color: #4facfe; font-size: 11px; letter-spacing: 1px; opacity: 0.6;'>NGUYEN ANH THAI DUONG • HUST • 2026</div>", unsafe_allow_html=True)