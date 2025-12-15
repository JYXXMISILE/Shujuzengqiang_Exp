# normalize_npy_only.py
import os, glob
import numpy as np
from pathlib import Path

# ========= 配置区（不想用命令行时，直接改这里）=========
INPUT_PATH = r"E:\cat的方法\T192_mel163\163_NPY"      # 单个文件(.npy)或文件夹
OUTPUT_DIR = r"E:\cat的方法\T192_mel163\npy_norm" # 输出文件夹
EPS = 1e-6
# =====================================================

def is_npy_file(p: str) -> bool:
    return os.path.isfile(p) and p.lower().endswith(".npy")

def list_npy(p: str):
    if os.path.isdir(p):
        return [f for f in glob.glob(os.path.join(p, "**", "*.npy"), recursive=True)]
    elif is_npy_file(p):
        return [p]
    else:
        raise ValueError("INPUT_PATH 既不是 .npy 文件也不是包含 .npy 的文件夹。")

def out_path_for(in_path: str, input_root: str, out_root: str) -> str:
    """当输入是文件夹时，保留相对目录；当输入是单文件时，直接放到 out_root。"""
    if os.path.isdir(input_root):
        rel = os.path.relpath(in_path, input_root)
        dst = os.path.join(out_root, rel)
    else:
        dst = os.path.join(out_root, os.path.basename(in_path))
    Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
    return dst

def normalize_per_file(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    # 逐文件标准化，不改变形状与 dtype（保存为 float32）
    m = x.mean()
    s = x.std()
    return ((x - m) / (s + eps)).astype(np.float32)

def main():
    files = list_npy(INPUT_PATH)
    if not files:
        print("未找到 .npy 文件。"); return
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, 1):
        x = np.load(f)
        if x.ndim != 2:
            print(f"[跳过] 维度非2（{x.ndim}）：{f}")
            continue
        x_norm = normalize_per_file(x)
        dst = out_path_for(f, INPUT_PATH, OUTPUT_DIR)
        np.save(dst, x_norm)
        if i % 100 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] 已输出：{dst}")

    print(f"完成！共处理 {len(files)} 个文件。输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
