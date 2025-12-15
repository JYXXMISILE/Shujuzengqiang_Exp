# scan_fmax.py
import os, glob
import librosa
import numpy as np
from collections import Counter

# ========= 配置 =========
IN_DIR = r"E:\cat的方法\所有音频集合_filtered"  # 你的 wav 文件夹
SR     = 16000                        # 采样率，保持和预处理一致
N_FFT  = 1024                         # 和 mel 里一致
HOP    = 512
THR_DB = -40                          # 能量阈值，低于主能量 40dB 以下忽略
# ========================

def get_file_list(in_dir):
    return glob.glob(os.path.join(in_dir, "**/*.wav"), recursive=True)

def highest_freq(y, sr, n_fft=1024, hop=512, thr_db=-40):
    # STFT 幅度谱
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mag_db = librosa.amplitude_to_db(S, ref=np.max)
    # 在时间上取最大值
    max_db = mag_db.max(axis=1)
    # 找超过阈值的频率 bin
    valid = np.where(max_db > thr_db)[0]
    if len(valid) == 0:
        return 0.0
    return freqs[valid[-1]]  # 最后一个满足条件的频率

def main():
    files = get_file_list(IN_DIR)
    if not files:
        print("未找到 wav 文件"); return
    print(f"发现 {len(files)} 个音频文件，开始统计…")

    fmax_list = []
    for i, f in enumerate(files, 1):
        try:
            y, _ = librosa.load(f, sr=SR, mono=True)
            fmax = highest_freq(y, SR, N_FFT, HOP, THR_DB)
            fmax_list.append(fmax)
        except Exception as e:
            print(f"[错误] {f}: {e}")
        if i % 200 == 0:
            print(f"[{i}/{len(files)}]")

    fmax_arr = np.array(fmax_list)
    print("\n==== 统计结果 ====")
    print(f"文件数: {len(fmax_list)}")
    print(f"最高频率 min / max / mean / median: "
          f"{fmax_arr.min():.1f} / {fmax_arr.max():.1f} / {fmax_arr.mean():.1f} / {np.median(fmax_arr):.1f}")

    # 取最常见的 fmax 档位
    bins = (fmax_arr // 100).astype(int) * 100
    top = Counter(bins).most_common(20)
    print("\n最常见的最高频率范围 Top-20：")
    for val, cnt in top:
        print(f" {val:4d} Hz ~ {val+100:4d} Hz: {cnt}")

if __name__ == "__main__":
    main()
