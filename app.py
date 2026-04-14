import streamlit as st
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Audio Artifact Detector Pro", layout="wide")

# =========================
# CSS + UI ENHANCEMENTS
# =========================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1c35, #29648a); color: #E0E0E0; }
    
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
        font-size: 50px; font-weight: 800; text-align: center;
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
# MAIN APP
# =========================
st.markdown("<div class='title-glow'>Audio Artifact Detection</div>", unsafe_allow_html=True)

col_input, col_result = st.columns([1, 2])

with col_input:
    st.markdown("### Input Audio")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    if file:
        st.audio(file)
    st.markdown('</div>', unsafe_allow_html=True)

if file:
    with st.spinner("Deep Analysis in progress..."):
        # Load and extract features
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        feat = np.stack([mfcc, delta, delta2])
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            output = model(X)
            probs = torch.softmax(output, dim=-1)[0].numpy()
            artifact_probs_series = probs[:, 1]
            avg_score = float(np.mean(artifact_probs_series))

    is_compressed = avg_score > 0.5
    theme_color = "#FF4B4B" if is_compressed else "#00FF7F" 
    status_text = "COMPRESSED AUDIO DETECTED" if is_compressed else "CLEAN AUDIO VERIFIED"

    with col_result:
        # Xác định icon và thông điệp dựa trên kết quả
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

        # Header kết quả
        st.markdown(f"### Analysis Result: <span style='color:{theme_color}'>{status_text}</span>", unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Overview", "Technical Metrics"])
        
        with tab1:
            r1, r2 = st.columns([1, 1.2])
            with r1:
                # Gauge Chart
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
                # Box thông báo tùy chỉnh với Icon theo yêu cầu
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
            m2.metric("Feature Variance", f"{np.var(mfcc):.2f}")
            m3.metric("Confidence Level", "High" if abs(avg_score-0.5)>0.2 else "Low")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # VISUALIZATIONS
    # =========================
    st.markdown("### Deep Signal Analysis")
    v1, v2 = st.columns(2)

    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white", "figure.facecolor": (0,0,0,0), "axes.facecolor": (0,0,0,0)})

    with v1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption("Artifact Probability Over Time")
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(y=artifact_probs_series, mode='lines', line=dict(color=theme_color, width=2), fill='tozeroy', fillcolor=f'rgba{tuple(list(int(theme_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'))
        fig_l.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, xaxis_title="Time Frames", yaxis_title="Probability")
        st.plotly_chart(fig_l, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption("Power Spectrogram (dB)")
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
        plt.colorbar(img, ax=ax, format="%+2.f dB").ax.yaxis.set_tick_params(color='white')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown(f"<div style='text-align:center; padding: 30px; color: #4facfe; font-size: 11px; letter-spacing: 1px; opacity: 0.6;'>NGUYEN ANH THAI DUONG • HUST • 2026</div>", unsafe_allow_html=True)