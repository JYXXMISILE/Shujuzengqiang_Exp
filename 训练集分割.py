import os
import shutil
import random

def split_dataset(
    feat_dir, label_dir,
    out_root,
    val_ratio=0.2,
    seed=42
):
    """
    将一个完整的数据集自动划分为 train / val
    feat_dir: 原始特征（.npy）目录
    label_dir: 原始标签（.txt）目录
    out_root: 输出根目录，例如 "E:/cat的方法/split_dataset"
    val_ratio: 验证集比例
    seed: 随机种子，保证可复现
    """
    random.seed(seed)

    # 输出目录
    train_feat = os.path.join(out_root, "train_audio")
    val_feat   = os.path.join(out_root, "val_audio")
    train_lab  = os.path.join(out_root, "train_label")
    val_lab    = os.path.join(out_root, "val_label")
    for d in [train_feat, val_feat, train_lab, val_lab]:
        os.makedirs(d, exist_ok=True)

    # 所有文件名（只看特征，标签跟着走）
    files = [f for f in os.listdir(feat_dir) if f.endswith(".npy")]
    random.shuffle(files)

    n_total = len(files)
    n_val   = int(n_total * val_ratio)
    val_files = set(files[:n_val])
    train_files = set(files[n_val:])

    # 复制文件
    for fn in train_files:
        # 复制特征
        shutil.copy(os.path.join(feat_dir, fn), os.path.join(train_feat, fn))
        # 复制标签
        label_fn = fn.replace(".npy", ".txt")
        shutil.copy(os.path.join(label_dir, label_fn), os.path.join(train_lab, label_fn))

    for fn in val_files:
        shutil.copy(os.path.join(feat_dir, fn), os.path.join(val_feat, fn))
        label_fn = fn.replace(".npy", ".txt")
        shutil.copy(os.path.join(label_dir, label_fn), os.path.join(val_lab, label_fn))

    print(f"总共 {n_total} 个样本，其中 {len(train_files)} 个训练，{len(val_files)} 个验证。")
    print(f"新数据集保存在: {out_root}")


if __name__ == "__main__":
    feat_dir  = r"E:\cat的方法\T192_mel163\163_NPY"  # 原始特征路径
    label_dir = r"E:\cat的方法\T192_mel163\所有音频集合_label_filtered"  # 原始标签路径
    out_root  = r"E:\cat的方法\T192_mel163\所有音频集合_dataset"  # 输出根目录
    split_dataset(feat_dir, label_dir, out_root, val_ratio=0.2)
