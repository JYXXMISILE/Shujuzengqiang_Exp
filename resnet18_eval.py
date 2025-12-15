# -*- coding: utf-8 -*-
# resnet18_eval.py —— 二分类（正常/湿罗音）+ 支持合成数据混训
# 用 ResNet18 替代 CNN14，直接吃 [T, F] 梅尔谱 .npy

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ========== 可视化依赖（无则降级为只打印） ==========
_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")  # 后端设为无界面，方便服务器/脚本运行
    import matplotlib.pyplot as plt
    # 尝试让中文正常显示；没装对应字体也不影响运行
    matplotlib.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS","DejaVu Sans","sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    _HAS_MPL = False

try:
    from torchvision.models import resnet18
    # 兼容新老版本权重写法
    try:
        from torchvision.models import ResNet18_Weights
        _HAS_TORCHVISION_WEIGHTS = True
    except Exception:
        _HAS_TORCHVISION_WEIGHTS = False
except Exception as e:
    raise RuntimeError("需要安装 torchvision 才能使用 ResNet18。pip install torchvision") from e

# ============ 基础设置 ============
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 标签映射（保持与 cnn14.py 一致）
LABEL_MAP  = {"正常":0, "湿罗音":1}
IDX2NAME   = {0:"正常", 1:"湿罗音"}
ALL_LABELS = [0,1]
TARGET_NAMES = [IDX2NAME[i] for i in ALL_LABELS]
LABEL_ALIASES = {
    "正常":0, "0":0, "normal":0,
    "湿罗音":1, "1":1, "wheeze":1, "ronchus":1, "rhonchi":1
}

# ============ 路径（按需改） ============
REAL_TRAIN_FEAT = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio"
REAL_TRAIN_LABEL= r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_label"
REAL_TEST_FEAT  = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_audio"
REAL_TEST_LABEL = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_label"

SYN_MEL_DIR     = r"E:\cat的方法\generated_by_class\mels"
SYN_LABEL_DIR   = r"E:\cat的方法\generated_by_class\labels"

# ============ 超参 ============
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
SYN_PER_REAL = 0.1    # 合成/真实数量上限比
WEIGHT_REAL  = 1.0
WEIGHT_SYN   = 0.25
MIN_ENERGY   = 1e-4
MIN_VAR      = 1e-6
N_MELS       = 163      # 梅尔频点

USE_IMAGENET_PRETRAIN = True   # 是否使用 ImageNet 预训练并适配为单通道

# ============ Dataset ============
class MelNPYDataset(Dataset):
    def __init__(self, feat_dir, label_dir, is_synth=False, quality_filter=True):
        self.feat_dir = feat_dir
        self.label_dir= label_dir
        self.is_synth = is_synth
        files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        self.samples = []; dropped = 0
        for fn in files:
            base = os.path.splitext(fn)[0]
            feat_path = os.path.join(feat_dir, fn)
            label_path= os.path.join(label_dir, base.replace("_mel","").replace("_idx","") + ".txt")
            if not os.path.exists(label_path): continue
            try:
                mel = np.load(feat_path)
                if mel.ndim != 2:
                    if mel.shape[0] < mel.shape[1]: mel = mel.T
                if quality_filter:
                    if (not np.isfinite(mel).all()) or np.mean(mel**2) < MIN_ENERGY or np.var(mel) < MIN_VAR:
                        dropped += 1; continue
                with open(label_path, "r", encoding="utf-8") as f:
                    txt = f.readline().strip().lower()
                if txt not in LABEL_ALIASES: continue
                y = LABEL_ALIASES[txt]
                if y not in ALL_LABELS: continue
                self.samples.append((feat_path, y))
            except Exception:
                dropped += 1
                continue
        print(f"[{'SYN' if is_synth else 'REAL'}] Loaded {len(self.samples)} samples from {feat_dir} (dropped {dropped})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        npy_path, y = self.samples[idx]
        mel = np.load(npy_path)
        if mel.ndim != 2 and mel.shape[0] < mel.shape[1]:
            mel = mel.T
        mel = np.asarray(mel, dtype=np.float32)
        # 每样本 z-score
        m, s = mel.mean(), mel.std() + 1e-6
        mel = (mel - m) / s
        x = torch.from_numpy(mel)  # [T,F]
        return x, y, (1 if self.is_synth else 0)

# ============ 模型 ============
class AudioClassifierResNet18(nn.Module):
    """
    ResNet18 骨干，吃 [B, T, F] 梅尔谱；首层改为 1 通道。
    如 USE_IMAGENET_PRETRAIN=True，会载入 ImageNet 权重并把首层权重按通道均值适配到 1 通道。
    """
    def __init__(self, num_classes=2, f_bins=N_MELS, use_pretrain=USE_IMAGENET_PRETRAIN):
        super().__init__()
        if use_pretrain and _HAS_TORCHVISION_WEIGHTS:
            m = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif use_pretrain and (not _HAS_TORCHVISION_WEIGHTS):
            m = resnet18(pretrained=True)
        else:
            m = resnet18(weights=None) if _HAS_TORCHVISION_WEIGHTS else resnet18(pretrained=False)

        # 将首层改为单通道，并尽量保留预训练能力
        old_conv1 = m.conv1
        m.conv1 = nn.Conv2d(1, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                            stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        if use_pretrain:
            with torch.no_grad():
                if old_conv1.weight.shape[1] == 3:
                    m.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

        # 分类头
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.backbone = m

    def forward(self, x_tf):  # x_tf: [B, T, F]
        x = x_tf.transpose(1, 2).unsqueeze(1)  # [B, 1, F, T]
        logits = self.backbone(x)
        return logits

# ============ 数据构建 ============
def make_real_datasets():
    train_real = MelNPYDataset(REAL_TRAIN_FEAT, REAL_TRAIN_LABEL, is_synth=False)
    test_real  = MelNPYDataset(REAL_TEST_FEAT,  REAL_TEST_LABEL,  is_synth=False, quality_filter=False)
    return train_real, test_real

def make_mixed_train(train_real: MelNPYDataset):
    syn_full = MelNPYDataset(SYN_MEL_DIR, SYN_LABEL_DIR, is_synth=True)
    max_syn = int(len(train_real) * SYN_PER_REAL)
    use_syn = min(len(syn_full), max_syn)
    if len(syn_full) > max_syn:
        idx = np.random.permutation(len(syn_full))[:max_syn]
        syn = Subset(syn_full, idx.tolist())
    else:
        syn = syn_full
    mixed = ConcatDataset([train_real, syn])
    print(f"[MIX] real={len(train_real)}, syn_all={len(syn_full)}, cap={max_syn}, use_syn={use_syn}, mixed_total={len(mixed)}")
    return mixed

def make_weighted_sampler(dataset):
    cnt = Counter()
    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        cnt[int(y)] += 1
    class_count  = np.array([cnt.get(c,0) for c in ALL_LABELS], dtype=np.float32)
    class_weight = 1.0 / np.maximum(class_count, 1.0)
    sample_weight = np.array([class_weight[int(dataset[i][1])] for i in range(len(dataset))], dtype=np.float32)
    return WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True), class_weight

# ============ 训练与评估 ============
def train_and_eval(tag, train_set, test_set):
    model = AudioClassifierResNet18(num_classes=len(ALL_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 采样器 + 类权重（供损失重加权）
    train_sampler, cls_w_np = make_weighted_sampler(train_set)
    class_weight = torch.tensor(cls_w_np/ (cls_w_np.sum()/len(cls_w_np)), dtype=torch.float32, device=device)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss=0.0; correct=0; total=0
        synth_seen=0; total_seen=0

        for xb, yb, synth_flag in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_vec = criterion(out, yb)                 # [B]
            cw = class_weight[yb]                         # [B]
            rw = (synth_flag.to(device).float()*WEIGHT_SYN + (1 - synth_flag.to(device).float())*WEIGHT_REAL)
            loss = (loss_vec * cw * rw).mean()

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            running_loss += loss.item()*xb.size(0)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)

            synth_seen += int(synth_flag.sum().item())
            total_seen += synth_flag.numel()

        train_loss = running_loss / max(total,1)
        train_acc  = correct / max(total,1)

        # 验证
        model.eval()
        all_preds=[]; all_labels=[]
        with torch.no_grad():
            for xb, yb, _ in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(1)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(yb.cpu().numpy().tolist())

        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1m = f1_score(all_labels, all_preds, average='macro',
                       labels=ALL_LABELS, zero_division=0) if all_labels else 0.0
        ratio = synth_seen / max(total_seen,1)
        print(f"[{tag}] Epoch {epoch:02d}/{EPOCHS}  TrainLoss {train_loss:.4f}  "
              f"TrainAcc {train_acc:.4f}  TestAcc {acc:.4f}  F1-macro {f1m:.4f}  "
              f"synth_used {synth_seen}/{total_seen} ({ratio:.1%})")

    # 最终报告
    cm = confusion_matrix(all_labels, all_preds, labels=ALL_LABELS) \
         if all_labels else np.zeros((len(ALL_LABELS),len(ALL_LABELS)), dtype=int)
    report = classification_report(all_labels, all_preds,
                                   labels=ALL_LABELS, target_names=TARGET_NAMES,
                                   zero_division=0) if all_labels else "EMPTY"
    return model, acc, f1m, cm, report

def save_csv(rows, out_csv="compare_results.csv"):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    header = ["tag","model","acc","f1_macro","syn_per_real","w_real","w_syn","epochs"]
    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file: w.writerow(header)
        for r in rows: w.writerow(r)

# ============ 新增：行归一化工具（不影响原流程） ============
def row_normalize_confmat(cm: np.ndarray) -> np.ndarray:
    """
    对混淆矩阵做“按行归一化”（每个真实类别一行，除以该行总数）。
    若某一行总数为 0，则保持为 0。
    """
    cm = cm.astype(np.float64, copy=False)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return cm / row_sum  # 每行加和为 1

# ============ 新增：绘制/保存混淆矩阵 ============
def _annot_text(v, normalize):
    return f"{v:.2f}" if normalize else f"{int(v)}"

def plot_and_save_confmat(cm: np.ndarray, labels, title: str, out_png: str, normalize: bool):
    if not _HAS_MPL:
        print(f"[WARN] matplotlib 未安装，无法保存图像：{out_png}")
        return
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0,
                   vmax=(1.0 if normalize else cm.max() if cm.size else 1.0))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("预测标签 (pred)")
    ax.set_ylabel("真实标签 (true)")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    # 方格内标注
    thresh = (0.5 if normalize else cm.max()/2.0 if cm.size else 0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, _annot_text(cm[i, j], normalize),
                    ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_png}")

def save_cm_plots(tag: str, cm_count: np.ndarray, cm_row: np.ndarray, labels):
    # 计数版
    plot_and_save_confmat(cm_count, labels,
                          title=f"{tag} - 混淆矩阵（计数）",
                          out_png=os.path.join("figs", f"{tag}_cm_count.png"),
                          normalize=False)
    # 行归一化版
    plot_and_save_confmat(cm_row, labels,
                          title=f"{tag} - 混淆矩阵（行归一化）",
                          out_png=os.path.join("figs", f"{tag}_cm_row_norm.png"),
                          normalize=True)

if __name__ == "__main__":
    # 数据
    train_real, test_real = make_real_datasets()

    # A) 只用真实
    model1, acc1, f1m1, cm1, rep1 = train_and_eval(
        tag="train_only_original",
        train_set=train_real,
        test_set=test_real
    )

    # B) 真实 + 合成
    mixed_train = make_mixed_train(train_real)
    model2, acc2, f1m2, cm2, rep2 = train_and_eval(
        tag="train_with_generated",
        train_set=mixed_train,
        test_set=test_real
    )

    # CSV
    rows = [
        ["train_only_original","ResNet18",f"{acc1:.6f}",f"{f1m1:.6f}",SYN_PER_REAL,WEIGHT_REAL,WEIGHT_SYN,EPOCHS],
        ["train_with_generated","ResNet18",f"{acc2:.6f}",f"{f1m2:.6f}",SYN_PER_REAL,WEIGHT_REAL,WEIGHT_SYN,EPOCHS],
    ]
    save_csv(rows, out_csv="compare_results.csv")

    # ===== 原有输出（计数版 + 分类报告）=====
    print("\n=== Confusion Matrix (train_only_original) ==="); print(cm1)
    print("\n=== Classification Report (train_only_original) ==="); print(rep1)
    print("\n=== Confusion Matrix (train_with_generated) ==="); print(cm2)
    print("\n=== Classification Report (train_with_generated) ==="); print(rep2)

    # ===== 新增：行归一化版本 =====
    np.set_printoptions(precision=3, suppress=True)
    cm1_row = row_normalize_confmat(cm1)
    cm2_row = row_normalize_confmat(cm2)

    print("\n[说明] 混淆矩阵行列含义：行 = 真实类别 (true)，列 = 预测类别 (pred)")
    print(f"类别顺序：{TARGET_NAMES}  ← 对应索引 {ALL_LABELS}")

    print("\n=== Row-Normalized Confusion Matrix (train_only_original) ===")
    print(cm1_row)

    print("\n=== Row-Normalized Confusion Matrix (train_with_generated) ===")
    print(cm2_row)

    # ===== 新增：保存两套图片 =====
    save_cm_plots("train_only_original", cm1, cm1_row, TARGET_NAMES)
    save_cm_plots("train_with_generated", cm2, cm2_row, TARGET_NAMES)

    print("\n→ 结果已追加到 compare_results.csv；混淆矩阵图片输出至 ./figs/")

