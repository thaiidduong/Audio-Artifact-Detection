import os
import urllib.request
import zipfile
import shutil
import librosa
import soundfile as sf

# 1. Tạo thư mục cấu trúc
os.makedirs("data/clean", exist_ok=True)
os.makedirs("temp_data", exist_ok=True)

print("🚀 Đang tải bộ dữ liệu ESC-50 (nhẹ và chuẩn)...")
url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
zip_path = "esc50.zip"

# Tải file
urllib.request.urlretrieve(url, zip_path)

print("📦 Đang giải nén...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("temp_data")

# 2. Lấy 100 file đầu tiên, convert về chuẩn 22050Hz
src_dir = "temp_data/ESC-50-master/audio"
files = [f for f in os.listdir(src_dir) if f.endswith('.wav')][:100]

print(f"🎵 Đang xử lý 100 file vào data/clean...")
for i, f in enumerate(files):
    # Load và resample về 22050Hz cho đồng nhất
    y, sr = librosa.load(os.path.join(src_dir, f), sr=22050)
    sf.write(f"data/clean/sample_{i}.wav", y, sr)

# 3. Dọn dẹp rác
shutil.rmtree("temp_data")
os.remove(zip_path)

print("🎉 Xong! 100 file sạch đã sẵn sàng trong thư mục data/clean")
