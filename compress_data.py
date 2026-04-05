import os
import subprocess

# Tạo thư mục đích nếu chưa có
os.makedirs("data/compressed", exist_ok=True)

clean_dir = "data/clean"
comp_dir = "data/compressed"
files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]

print(f"🛠️ Đang nén nát {len(files)} file xuống 8kbps...")

for f in files:
    input_path = os.path.join(clean_dir, f)
    output_path = os.path.join(comp_dir, f)
    
    # Nén xuống 8k sau đó đưa về định dạng wav 22050Hz để model dễ học
    cmd = f"ffmpeg -y -i {input_path} -codec:a libmp3lame -b:a 8k -ar 22050 {output_path} -loglevel quiet"
    subprocess.run(cmd, shell=True)

print("🎉 Hoàn thành! Thư mục data/compressed đã sẵn sàng.")
