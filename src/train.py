import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ===== LOAD DATA =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(BASE_DIR, "features")

X = np.load(os.path.join(FEATURE_PATH, "mfcc_features.npy"))
y = np.load(os.path.join(FEATURE_PATH, "labels.npy"))

print("Loaded data:", X.shape, y.shape)

# ===== SPLIT =====
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ===== CONVERT TO TENSOR =====
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# ===== MODEL =====
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 32, 64),  # adjust if needed
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = AudioCNN()

# ===== TRAIN =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ===== VALIDATION =====
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, preds = torch.max(val_outputs, 1)
        acc = (preds == y_val).float().mean()

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={acc:.4f}")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "model.pth"))

print("Model saved!")
from sklearn.metrics import accuracy_score

model.eval()
with torch.no_grad():
    outputs = model(X_val)
    preds = torch.argmax(outputs, dim=1)

acc = accuracy_score(y_val.numpy(), preds.numpy())
print("Final Accuracy:", acc)
from sklearn.metrics import classification_report

print(classification_report(y_val.numpy(), preds.numpy()))
