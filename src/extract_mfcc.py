import os
import numpy as np
import librosa

input_dir = "../data/processed"
output_dir = "../data/mfcc"

os.makedirs(output_dir, exist_ok=True)

files = os.listdir(input_dir)
print("FILES FOUND:", files)

for file in files:
    print("Checking file:", file)

    if file.endswith(".wav"):
        print("Processing:", file)

        path = os.path.join(input_dir, file)

        y, sr = librosa.load(path, sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        save_path = os.path.join(output_dir, file.replace(".wav", ".npy"))
        np.save(save_path, mfcc)

        print("Saved:", save_path)

print("DONE")
