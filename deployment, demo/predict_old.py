import os
import torch
import numpy as np
import librosa

# ===== CONFIG =====
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_LEN = 130

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== LOAD MODEL =====
class AudioCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 3 * 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = AudioCNN()
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "model.pth")))
model.eval()

# ===== EXTRACT MFCC =====
def extract(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# ===== TEST FILE =====
file = "../data/raw/voice.wav"  # đổi file nếu muốn

mfcc = extract(file)
mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ===== PREDICT =====
with torch.no_grad():
    output = model(mfcc)
    pred = torch.argmax(output, dim=1).item()

if pred == 1:
    print("🔴 Compressed audio")
else:
    print("🟢 Original audio")
