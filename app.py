import streamlit as st
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG & UI
# =========================
st.set_page_config(page_title="Audio Artifact Detector Pro", layout="wide")
st.title("🎧 Advanced Audio Artifact Detection")
st.caption("Deep Learning 3-Channel (MFCC + Delta + Delta2) | Trained on ESC-50")

# =========================
# MODEL STRUCTURE (3-Channel)
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

# LOAD MODEL
@st.cache_resource
def load_my_model():
    m = BetterCNN()
    model_path = "models/model.pth"
    if os.path.exists(model_path):
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        m.eval()
        return m
    return None

model = load_my_model()

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("📂 Upload file âm thanh để kiểm tra (WAV/MP3)", type=["wav", "mp3"])

if file:
    if model is None:
        st.error("❌ Không tìm thấy models/model.pth. Hãy chạy train.py trước!")
    else:
        with open("temp_test.wav", "wb") as f:
            f.write(file.getbuffer())

        # 1. TRÍCH XUẤT ĐẶC TRƯNG 3 KÊNH (GIỐNG HỆT TRAIN.PY)
        y, sr = librosa.load("temp_test.wav", sr=22050, duration=5.0)
        st.audio("temp_test.wav")

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        feat = np.stack([mfcc, delta, delta2])
        # Chuẩn hóa per-sample
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        
        # Chuyển thành tensor (Batch=1, Channel=3, Freq=13, Time=T)
        X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        # 2. DỰ ĐOÁN
        with torch.no_grad():
            output = model(X) # Output shape: (1, T, 2)
            probs = torch.softmax(output, dim=-1)[0].numpy()
            artifact_prob = probs[:, 1] # Xác suất bị nén (Class 1)

        # 3. HIỂN THỊ KẾT QUẢ
        # Lấy giá trị trung bình hoặc max để đánh giá tổng thể file
        final_score = np.mean(artifact_prob)
        
        st.subheader("🎯 Prediction Result")
        if final_score > 0.5:
            st.error(f"**Status: Compressed (Artifacts Detected)**")
        else:
            st.success(f"**Status: Clean (High Quality)**")
        
        st.write(f"Artifact Confidence Score: **{final_score:.4f}**")
        st.progress(float(final_score))

        # 4. VISUALIZATION (HIGHLIGHTS)
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 Spectrogram with Highlights")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            fig, ax = plt.subplots(figsize=(10, 6))
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            
            # Vẽ vùng highlight đỏ nếu xác suất tại khung thời gian đó > 0.5
            times = np.linspace(0, len(y)/sr, len(artifact_prob))
            hop_time = times[1] - times[0] if len(times) > 1 else 0
            for i, p in enumerate(artifact_prob):
                if p > 0.5:
                    ax.axvspan(times[i], times[i] + hop_time, color='red', alpha=0.3)
            
            plt.colorbar(img, format="%+2.0f dB")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 3-Channel Feature View")
            # Hiển thị kênh Delta để thấy sự biến thiên
            fig, ax = plt.subplots(figsize=(10, 6))
            librosa.display.specshow(delta, sr=sr, x_axis='time', ax=ax)
            st.subheader("Delta MFCC (Motion Features)")
            st.pyplot(fig)
