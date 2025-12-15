import librosa, numpy as np, os, glob
from skimage.transform import resize

IN_DIR  = r"E:\cat的方法\所有音频集合_filtered"
OUT_DIR = r"E:\cat的方法\T192_mel163\163_NPY"

SR, SEC   = 32000, 9.0          # ← 32 kHz（与 Cnn14 预训练一致）
N_MELS    = 163                 # ← 模型的 feature_dim=163
TARGET_T  = 192                 # ← 统一到主程序常用帧长（你之前常见 961）
HOP       = 320                 # ← 与 Cnn14 一致
N_FFT     = 1024
FMIN, FMAX = 50, 14000          # ← 与 Cnn14 一致

os.makedirs(OUT_DIR, exist_ok=True)
L = int(SR * SEC)

for wav in glob.glob(os.path.join(IN_DIR, "**/*.wav"), recursive=True):
    y, _ = librosa.load(wav, sr=SR, mono=True)
    y = np.pad(y, (0, max(0, L - len(y))))[:L]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT,
        fmin=FMIN, fmax=FMAX
    )
    logmel = librosa.power_to_db(mel, ref=np.max)      # [F, T]
    logmel = resize(logmel, (N_MELS, TARGET_T), mode='constant', anti_aliasing=False)
    logmel = logmel.T.astype(np.float32)               # [T, F]

    # 可选：做稳健归一化与裁剪，减弱异常样本影响
    m, s = logmel.mean(), logmel.std()
    logmel = (logmel - m) / (s + 1e-6)
    logmel = np.clip(logmel, -5, 5)

    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(wav))[0] + ".npy")
    np.save(out_path, logmel)
    print("saved", out_path)

