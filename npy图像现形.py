import os
import numpy as np
import matplotlib.pyplot as plt

def plot_triplet(orig, vq, trad, save_path=None, show=True, title_prefix="sample"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    data_list = [orig, vq, trad]
    titles = ["Original", "VQ-VAE Aug", "Traditional Aug"]

    for ax, data, t in zip(axes, data_list, titles):
        im = ax.imshow(
            data.T,           # 假设是 [T, F]，转置后 [F, T]
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    # 这里改成你的三个文件
    orig_path = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio\ZC_4.npy"
    vq_path   = r"E:\cat的方法\generated_by_class20k_copy\mels\class0_sample0_mel.npy"
    trad_path = r"E:\cat的方法\传统增强方式测试\offline_aug_mels\ZC_4_aug0.npy"  # 如果是npy

    # ---- 加载数据 ----
    orig = np.load(orig_path)  # [T, F]

    vq   = np.load(vq_path)

    # 如果传统增强也是 npy：
    trad = np.load(trad_path)

    # ⚠️ 如果你的 SL_1_concat.txt 其实是文本形式的谱图，可以改成：
    # trad = np.loadtxt(trad_path)  # 看你保存的格式

    # 对齐时间长度
    # 对齐时间长度
    T_min = min(orig.shape[0], vq.shape[0], trad.shape[0])

    # 想要展示的最大时间帧数，比如 200、300 自己调
    max_T = 50
    T_use = min(T_min, max_T)

    orig = orig[:T_use]
    vq = vq[:T_use]
    trad = trad[:T_use]

    out_dir = r"E:\cat的方法\npy图像现形"
    save_path = os.path.join(out_dir, "compare_ZC_1.png")

    plot_triplet(orig, vq, trad, save_path=save_path, show=True, title_prefix="ZC comparison")

