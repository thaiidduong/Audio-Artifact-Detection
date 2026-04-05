import os
import numpy as np

# ===== PATH =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MFCC_PATH = os.path.join(BASE_DIR, "data", "mfcc")
FEATURE_PATH = os.path.join(BASE_DIR, "features")

os.makedirs(FEATURE_PATH, exist_ok=True)

# ===== CONFIG =====
MAX_LEN = 130  # phải giống extract_mfcc

# ===== LOAD DATA =====
X = []
y = []

print("Loading MFCC files from:", MFCC_PATH)

for file in os.listdir(MFCC_PATH):
    if file.endswith(".npy"):
        file_path = os.path.join(MFCC_PATH, file)

        mfcc = np.load(file_path)

        # ===== FIX SHAPE =====
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        X.append(mfcc)

        # ===== LABEL =====
        if "compressed" in file.lower():
            y.append(1)
        else:
            y.append(0)

        print(f"Loaded: {file} | shape: {mfcc.shape}")

# ===== CONVERT =====
X = np.array(X)
y = np.array(y)

print("\nFINAL DATASET:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ===== SAVE =====
np.save(os.path.join(FEATURE_PATH, "mfcc_features.npy"), X)
np.save(os.path.join(FEATURE_PATH, "labels.npy"), y)

print("\nSaved dataset to /features/")
