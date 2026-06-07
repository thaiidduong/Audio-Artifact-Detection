import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ==========================================
# 1. TRÍCH XUẤT ĐẶC TRƯNG 3 KÊNH (PRO)
# ==========================================
def extract_features(directory, label):
    data = []
    labels = []
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    for f in files:
        path = os.path.join(directory, f)
        # Load audio chuẩn 22050Hz, lấy 5 giây đầu
        y, sr = librosa.load(path, sr=22050, duration=5.0)
        
        # Tạo 3 lớp đặc trưng
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack thành (3, 13, T)
        feat = np.stack([mfcc, delta, delta2])
        
        # Chuẩn hóa từng mẫu (Per-sample Normalization)
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        
        data.append(feat)
        labels.append(label)
    return np.array(data), np.array(labels)

print("🔍 Đang trích xuất đặc trưng từ 200 file...")
X_clean, y_clean = extract_features("data/clean", 0)
X_comp, y_comp = extract_features("data/compressed", 1)

X = np.concatenate([X_clean, X_comp], axis=0)
y = np.concatenate([y_clean, y_comp], axis=0)

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU TRAIN
# ==========================================
# Cắt nhãn theo khung thời gian (T) để đồng bộ với App
T_frames = X.shape[3]
y_frames = np.repeat(y[:, None], T_frames, axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y_frames, test_size=0.2, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

# ==========================================
# 3. KIẾN TRÚC BetterCNN (3-CHANNEL)
# ==========================================
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), # Nhận 3 kênh vào
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
        x = torch.mean(x, dim=2) # GAP theo Freq
        x = x.permute(0, 2, 1) # (B, T, 64)
        return self.fc_time(x)

# ==========================================
# 4. HUẤN LUYỆN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetterCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"🚀 Bắt đầu Train trên {device}...")
for epoch in range(30):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.view(-1, 2), batch_y.view(-1))
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/30 - Loss: {loss.item():.4f}")

# Lưu model xịn
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
print("✅ Đã lưu Model chuẩn chỉ tại models/model.pth!")
