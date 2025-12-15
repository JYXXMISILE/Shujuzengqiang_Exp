# -*- coding: utf-8 -*-
# cnn14_eval.py —— 二分类（正常/湿罗音）+ 支持合成数据混训

import os, csv, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")            # 纯保存图片，兼容无显示环境
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ============ 基础设置 ============
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 标签映射（保持与你一致）
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

SYN_MEL_DIR     = r"E:\cat的方法\generated_by_class20k\mels"
SYN_LABEL_DIR   = r"E:\cat的方法\generated_by_class20k\labels"

# 预训练 CNN14 权重（PANNs 官方发布的 Audioset 权重）
CNN14_CKPT      = r"E:\cat的方法\VAE\Cnn14_mAP=0.431.pth"  # 改成你的路径

# ============ 超参 ============
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
SYN_PER_REAL = 3.0     # eg. 0.25 / 0.5 也可
WEIGHT_REAL  = 1.0
WEIGHT_SYN   = 1.0
MIN_ENERGY   = 1e-4
MIN_VAR      = 1e-6
N_MELS       = 163      # 你的特征频点数

# ============ 导入 PANNs Cnn14 ============
# 你项目里若已有 third_party/audioset_tagging_cnn，可像这样动态加入路径
repo_pytorch = r"E:\cat的方法\third_party\audioset_tagging_cnn\pytorch"
if repo_pytorch not in sys.path:
    sys.path.insert(0, repo_pytorch)
models = r'E:\cat的方法\third_party\audioset_tagging_cnn\pytorch\models.py'
try:
    from models import Cnn14
except Exception as e:
    raise RuntimeError(
        "未找到 PANNs 的 Cnn14 实现。请把官方仓库放到 third_party/audioset_tagging_cnn/pytorch，"
        "或自行修改此处 import 路径。"
    ) from e

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
class AudioClassifierCNN14(nn.Module):
    """
    使用 PANNs Cnn14 作为特征骨干，去掉前端与原分类头，输入 [B, T, F]。
    """
    def __init__(self, num_classes=2, n_mels=N_MELS, ckpt_path=CNN14_CKPT):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=n_mels, fmin=50, fmax=14000, classes_num=527
        )
        # 移除不需要的声学前端和分类层
        self.backbone.fc_audioset          = nn.Identity()
        self.backbone.spectrogram_extractor= nn.Identity()
        self.backbone.logmel_extractor     = nn.Identity()
        self.backbone.spec_augmenter       = nn.Identity()
        self.backbone.bn0                  = nn.Identity()

        # 加载预训练
        if ckpt_path and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = sd.get("model", sd)
            # 丢弃与前端/分类头相关的权重（若存在）
            for k in list(sd.keys()):
                if k.startswith('logmel_extractor.') or k.startswith('fc_audioset.'):
                    sd.pop(k)
            self.backbone.load_state_dict(sd, strict=False)
            print(f"[CNN14] Loaded pretrained: {ckpt_path}")
        else:
            print("[CNN14] Warning: pretrained weights not found. Using random init.")

        # 将 conv_block5→conv_block6→时轴特征做全局池化 + 线性分类
        self.proj = nn.Conv1d(2048, 512, kernel_size=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # over time
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_tf):  # x_tf: [B, T, F]
        x = x_tf.unsqueeze(1)  # [B,1,T,F] —— 与 Cnn14 的 conv_block 接口对齐
        x1 = self.backbone.conv_block1(x, pool_size=(2,2), pool_type='avg')
        x2 = self.backbone.conv_block2(x1, pool_size=(2,2), pool_type='avg')
        x3 = self.backbone.conv_block3(x2, pool_size=(2,2), pool_type='avg')
        x4 = self.backbone.conv_block4(x3, pool_size=(2,2), pool_type='avg')
        x5 = self.backbone.conv_block5(x4, pool_size=(2,2), pool_type='avg')
        feat_4d = self.backbone.conv_block6(x5, pool_size=(1,1), pool_type='avg')  # [B,2048,T/32,F']
        feat = feat_4d.mean(dim=-1)  # 频率均值 → [B,2048, T/32]
        feat = self.proj(feat)       # [B,512, T/32]
        logits = self.head(feat)     # [B, num_classes]
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
    class_count = np.array([cnt.get(c,0) for c in ALL_LABELS], dtype=np.float32)
    class_weight = 1.0 / np.maximum(class_count, 1.0)
    sample_weight = np.array([class_weight[int(dataset[i][1])] for i in range(len(dataset))], dtype=np.float32)
    return WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True), class_weight

def save_cm_figure(cm: np.ndarray, labels, out_path: str, normalize: bool = False):
    """
    保存混淆矩阵图：
      - normalize=False：计数版（整型，values_format='d'）
      - normalize=True ：行归一化（浮点，values_format='.2f'）
    """
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    if normalize:
        cm_plot = cm.astype(np.float32)
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        # 避免除以0：对全零行保持为0
        cm_plot = np.divide(cm_plot, row_sum, out=np.zeros_like(cm_plot), where=row_sum != 0)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (row-wise)'
    else:
        cm_plot = cm.astype(np.int64)   # 计数版保持整型
        fmt = 'd'
        title = 'Confusion Matrix'

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_plot, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    disp.plot(cmap='Blues', ax=ax, values_format=fmt, colorbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"--> saved confusion matrix to {out_path}")



# ============ 训练与评估 ============
def train_and_eval(tag, train_set, test_set):
    model = AudioClassifierCNN14(num_classes=len(ALL_LABELS)).to(device)
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
        ["train_only_original","CNN14",f"{acc1:.6f}",f"{f1m1:.6f}",SYN_PER_REAL,WEIGHT_REAL,WEIGHT_SYN,EPOCHS],
        ["train_with_generated","CNN14",f"{acc2:.6f}",f"{f1m2:.6f}",SYN_PER_REAL,WEIGHT_REAL,WEIGHT_SYN,EPOCHS],
    ]
    save_csv(rows, out_csv="compare_results.csv")

    print("\n=== Confusion Matrix (train_only_original) ==="); print(cm1)
    print("\n=== Classification Report (train_only_original) ==="); print(rep1)
    print("\n=== Confusion Matrix (train_with_generated) ==="); print(cm2)
    print("\n=== Classification Report (train_with_generated) ==="); print(rep2)
    print("\n→ Results appended to compare_results.csv")

    # 现有打印…
    print("\n=== Confusion Matrix (train_only_original) ===");
    print(cm1)
    print("\n=== Classification Report (train_only_original) ===");
    print(rep1)
    print("\n=== Confusion Matrix (train_with_generated) ===");
    print(cm2)
    print("\n=== Classification Report (train_with_generated) ===");
    print(rep2)

    # 新增：把两组实验的混淆矩阵各导出“计数版”和“行归一化版”
    save_cm_figure(cm1, TARGET_NAMES, 'cm_train_only_original_counts.png', normalize=False)
    save_cm_figure(cm1, TARGET_NAMES, 'cm_train_only_original_norm.png', normalize=True)
    save_cm_figure(cm2, TARGET_NAMES, 'cm_train_with_generated_counts.png', normalize=False)
    save_cm_figure(cm2, TARGET_NAMES, 'cm_train_with_generated_norm.png', normalize=True)

    print("\n→ Results appended to compare_results.csv")

