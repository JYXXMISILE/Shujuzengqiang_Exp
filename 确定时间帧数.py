# check_max_frames_fixed.py
import os, glob, math
import numpy as np
from collections import Counter
import librosa

# ==============================
# 在这里改成你的 wav 文件夹路径
WAV_DIR = r"E:\cat的方法\所有音频集合"
# 采样和梅尔参数（需与你的预处理保持一致）
SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 163   # 要和模型的 feature_dim 对齐
FMIN = 50
FMAX = 14000
# ==============================

def scan_wav(wav_root):
    frames = []
    for p in glob.glob(os.path.join(wav_root, "**", "*.wav"), recursive=True):
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
            S = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0, center=True
            )
            T = S.shape[1]  # 时间帧数
            frames.append(T)
        except Exception as e:
            print(f"[ERR] {p}: {e}")
    return frames

def summarize(frames):
    if not frames:
        print("未找到任何 wav 文件")
        return
    arr = np.array(frames, dtype=int)
    minT, maxT = int(arr.min()), int(arr.max())
    meanT = float(arr.mean())
    medT = int(np.median(arr))
    print(f"文件数: {len(arr)}")
    print(f"帧数最小/最大/均值/中位数: {minT} / {maxT} / {meanT:.1f} / {medT}")
    print(f"建议 prior.max_len = ceil(max_T/32) = {math.ceil(maxT/32)}")
    # 输出 Top-10 常见帧数
    from collections import Counter
    cnt = Counter(arr.tolist()).most_common(100)
    print("最常见帧数 Top-10：")
    for v, c in cnt:
        print(f"  T={v:<5}  count={c}")

if __name__ == "__main__":
    frames = scan_wav(WAV_DIR)
    summarize(frames)

