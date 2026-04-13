import os

input_dir = r"C:\Users\DELL\OneDrive\Desktop\main.cpp\AUDIO\audio_clean"
output_dir = r"C:\Users\DELL\OneDrive\Desktop\main.cpp\AUDIO\audio_compressed"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".wav"):
        input_path = os.path.join(input_dir, file)

        output_file = file.replace(".wav", ".mp3")
        output_path = os.path.join(output_dir, output_file)

        print(f"Compressing: {file}")

        os.system(f'ffmpeg -y -i "{input_path}" -b:a 8k "{output_path}"')

print("DONE ALL 🔥")