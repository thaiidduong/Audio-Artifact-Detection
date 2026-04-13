import streamlit as st
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Audio Artifact Detector Pro", layout="wide")

# =========================
# GLOBAL CSS (UI đẹp)
# =========================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #25274D, #29648A);
    color: white;
}

/* ===== TITLE STYLE ===== */
.title {
    font-size: 64px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #EDEDED, #2E9CCA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
}

/* ===== SUBTITLE ===== */
.subtitle {
    text-align:center;
    color:#AAABB8;
    font-size:18px;
}

/* ===== HEADINGS ===== */
h2, h3 {
    background: linear-gradient(90deg, #EDEDED, #2E9CCA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight:600;
}

/* ===== TEXT ===== */
p, label, span {
    color: #EDEDED !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div {
    background: linear-gradient(90deg, #2E9CCA, #ffffff);
    height: 12px;
    border-radius: 10px;
}

 /* ===== FIX TEXT UPLOAD BUTTON ===== */
[data-testid="stFileUploader"] button {
    color: black !important;
    font-weight: 600;
}

[data-testid="stFileUploader"] button span {
    color: black !important;
}

            /* Force toàn bộ text trong button thành đen */
[data-testid="stFileUploader"] button * {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<div class='title'>Audio Artifact Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning System • MFCC + Temporal Features</div>", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
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
    path = "models/model.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    return None

model = load_model()

# =========================
# UPLOAD
# =========================
st.markdown("## Upload Audio")
file = st.file_uploader("Choose WAV/MP3", type=["wav", "mp3"])

if file:

    if model is None:
        st.error("Model not found! Train first.")
        st.stop()

    # Save file
    with open("temp_test.wav", "wb") as f:
        f.write(file.getbuffer())

    st.audio("temp_test.wav")

    # =========================
    # FEATURE
    # =========================
    y, sr = librosa.load("temp_test.wav", sr=22050, duration=5.0)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.stack([mfcc, delta, delta2])
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)

    X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    # =========================
    # PREDICT
    # =========================
    with torch.no_grad():
        output = model(X)
        probs = torch.softmax(output, dim=-1)[0].numpy()
        artifact_prob = probs[:, 1]

    score = float(np.mean(artifact_prob))

    # =========================
    # RESULT
    # =========================
    st.markdown("## Result")

    if score > 0.5:
        st.markdown(f"""
        <div style='
            background: linear-gradient(90deg, #ff4b5c, #8b0000);
            padding: 20px;
            border-radius: 12px;
            text-align:center;
            font-weight:600;
            font-size:20px;
            color:white;
            box-shadow: 0 0 25px rgba(255,0,0,0.5);
        '>
        COMPRESSED AUDIO DETECTED
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            background: linear-gradient(90deg, #00e676, #007f5f);
            padding: 20px;
            border-radius: 12px;
            text-align:center;
            font-weight:600;
            font-size:20px;
            color:white;
            box-shadow: 0 0 20px rgba(0,255,150,0.5);
        '>
        CLEAN AUDIO (HIGH QUALITY)
        </div>
        """, unsafe_allow_html=True)

    # ===== Confidence đẹp =====
    st.markdown(f"""
    <p style='
        font-size:18px;
        font-weight:500;
    '>
    Confidence Score: <span style='color:#2E9CCA; font-weight:700;'>{score:.4f}</span>
    </p>
    """, unsafe_allow_html=True)

    st.progress(score)

    # =========================
    # VISUAL
    # =========================
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Spectrogram")

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)

        times = np.linspace(0, len(y)/sr, len(artifact_prob))
        hop = times[1] - times[0] if len(times) > 1 else 0

        for i, p in enumerate(artifact_prob):
            if p > 0.5:
                ax.axvspan(times[i], times[i] + hop, color='red', alpha=0.3)

        plt.colorbar(img)
        st.pyplot(fig)

    with col2:
        st.markdown("### Delta MFCC")

        fig, ax = plt.subplots(figsize=(10, 6))
        librosa.display.specshow(delta, sr=sr, x_axis='time', ax=ax)
        st.pyplot(fig)