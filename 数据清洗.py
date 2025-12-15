# filter_wav_by_length.py
import os, glob, shutil, librosa

IN_WAV_DIR   = r"E:\cat的方法\所有音频集合"      # 原始 wav 路径
IN_LABEL_DIR = r"E:\cat的方法\所有音频集合_label" # 原始标签路径

OUT_WAV_DIR   = r"/T192_mel163/所有音频集合_filtered"
OUT_LABEL_DIR = r"/T192_mel163/所有音频集合_label_filtered"

os.makedirs(OUT_WAV_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# 设定保留区间（单位：梅尔帧数）
low, high = 32, 200

# 梅尔谱参数（和你训练保持一致）
SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 163
FMIN = 50
FMAX = 14000

kept, removed = 0, 0

for wav_path in glob.glob(os.path.join(IN_WAV_DIR, "*.wav")):
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        # 计算梅尔谱，取时间帧数
        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX
        )
        T = S.shape[1]  # 时间帧数

        base = os.path.splitext(os.path.basename(wav_path))[0]
        label_path = os.path.join(IN_LABEL_DIR, base + ".txt")

        if low <= T <= high:
            shutil.copy(wav_path, os.path.join(OUT_WAV_DIR, base + ".wav"))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(OUT_LABEL_DIR, base + ".txt"))
            kept += 1
        else:
            removed += 1
    except Exception as e:
        print(f"[ERR] {wav_path}: {e}")
        removed += 1

print(f"保留 {kept} 条，剔除 {removed} 条 (区间: {low}–{high})")

