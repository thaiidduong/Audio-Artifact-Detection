import librosa
import soundfile as sf
import os

input_folder = "data/raw"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        path = os.path.join(input_folder, file)

        print(f"Processing: {file}")

        y, sr = librosa.load(path, sr=22050, mono=True)

        output_path = os.path.join(output_folder, file)
        sf.write(output_path, y, 22050)

print("✅ Done preprocessing!")
