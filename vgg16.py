# -*- coding: utf-8 -*-
# compare_generated_vs_real.py —— 二分类（正常/湿罗音）+ 防塌缩版
import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ========== 可视化依赖（无则降级为只打印） ==========
_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端
    import matplotlib.pyplot as plt
    # 尝试中文字体；没装也不影响运行
    matplotlib.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS","DejaVu Sans","sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    _HAS_MPL = False

# ============ 基础设置 ============
SEED=42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 仅二分类
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
EPOCHS     = 30            # 建议 >=20
LR         = 1e-3
SYN_PER_REAL = 0.1          # 先从 0.5 或 0.25 试起
WEIGHT_REAL  = 1.0
WEIGHT_SYN   = 0.25
MIN_ENERGY = 1e-4
MIN_VAR    = 1e-6

# ============ Dataset ============
class MelNPYDataset(Dataset):
    def __init__(self, feat_dir, label_dir, is_synth=False, transform=None, quality_filter=True):
        self.feat_dir = feat_dir
        self.label_dir= label_dir
        self.transform= transform
        self.is_synth = is_synth
        files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        self.samples = []; dropped=0
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
        if mel.ndim != 2:
            if mel.shape[0] < mel.shape[1]: mel = mel.T
        # --- 关键：每样本 z-score ---
        mel = np.asarray(mel, dtype=np.float32)
        m, s = mel.mean(), mel.std() + 1e-6
        mel = (mel - m) / s

        x = np.stack([mel, mel, mel], axis=0).astype(np.float32)
        x = torch.from_numpy(x)
        if self.transform: x = self.transform(x)
        return x, y, (1 if self.is_synth else 0)

# torchvision 张量变换
data_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ============ Model ============
def build_vgg16(num_classes=2, local_ckpt=None):
    model = models.vgg16(weights=None)
    cand=[]
    if local_ckpt: cand.append(local_ckpt)
    cand += [
        r"C:\Users\JYXXMISILE\.cache\torch\hub\checkpoints\vgg16-397923af.pth",
        os.path.join(os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch")),
                     "hub","checkpoints","vgg16-397923af.pth"),
    ]
    loaded=False
    for p in cand:
        if os.path.isfile(p):
            try:
                sd=torch.load(p, map_location="cpu")
                model.load_state_dict(sd)
                print(f"[VGG16] Loaded local pretrained: {p}"); loaded=True; break
            except Exception as e:
                print(f"[VGG16] Found but failed to load: {p} ({e})")
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, num_classes)
    # ✅ 始终解冻：端到端微调
    for p in model.features.parameters(): p.requires_grad = True
    print("[VGG16] UNFREEZE backbone (fine-tune end-to-end)")
    return model

# ============ 数据构建 ============
def make_real_datasets():
    train_real = MelNPYDataset(REAL_TRAIN_FEAT, REAL_TRAIN_LABEL, is_synth=False, transform=data_tf)
    test_real  = MelNPYDataset(REAL_TEST_FEAT,  REAL_TEST_LABEL,  is_synth=False, transform=data_tf, quality_filter=False)
    return train_real, test_real

def make_mixed_train(train_real: MelNPYDataset):
    syn_full = MelNPYDataset(SYN_MEL_DIR, SYN_LABEL_DIR, is_synth=True, transform=data_tf)
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

# ============ 训练与评估 ============
def train_and_eval(tag, train_set, test_set):
    model = build_vgg16(num_classes=len(ALL_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 采样器 + 类权重（供损失重加权）
    train_sampler, cls_w_np = make_weighted_sampler(train_set)
    class_weight = torch.tensor(cls_w_np/ (cls_w_np.sum()/len(cls_w_np)), dtype=torch.float32, device=device)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 分组学习率：backbone 小，head 大
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR * 0.1},  # e.g., 1e-4
        {"params": head_params,     "lr": LR},        # e.g., 1e-3
    ])

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss=0.0; correct=0; total=0
        synth_seen=0; total_seen=0

        for xb, yb, synth_flag in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_vec = criterion(out, yb)                 # [B]
            cw = class_weight[yb]                         # [B] 类别权重
            rw = (synth_flag.to(device).float()*WEIGHT_SYN + (1 - synth_flag.to(device).float())*WEIGHT_REAL)  # [B]
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

# ============ 新增：行归一化与绘图（不改动原逻辑） ============
def row_normalize_confmat(cm: np.ndarray) -> np.ndarray:
    """按行归一化（每个真实类一行）；若该行总数为0则保持为0。"""
    cm = cm.astype(np.float64, copy=False)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return cm / row_sum

def _annot_text(v, normalize):
    return f"{v:.2f}" if normalize else f"{int(v)}"

def plot_and_save_confmat(cm: np.ndarray, labels, title: str, out_png: str, normalize: bool):
    if not _HAS_MPL:
        print(f"[WARN] matplotlib 未安装，无法保存图像：{out_png}")
        return
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                   vmin=0.0, vmax=(1.0 if normalize else (cm.max() if cm.size else 1.0)))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("预测标签 (pred)")
    ax.set_ylabel("真实标签 (true)")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    thresh = (0.5 if normalize else (cm.max()/2.0 if cm.size else 0.5))
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
    plot_and_save_confmat(cm_count, labels,
                          title=f"{tag} - 混淆矩阵（计数）",
                          out_png=os.path.join("figs", f"{tag}_cm_count.png"),
                          normalize=False)
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
        ["train_only_original","VGG16",f"{acc1:.6f}",f"{f1m1:.6f}",SYN_PER_REAL,WEIGHT_REAL,WEIGHT_SYN,EPOCHS],
        ["train_with_generated","VGG16",f"{acc2:.6f}",f"{f1m2:.6f}",SYN_PER_REAL,WEIGHT_REAL,WEIGHT_SYN,EPOCHS],
    ]
    save_csv(rows, out_csv="compare_results.csv")

    print("\n=== Confusion Matrix (train_only_original) ==="); print(cm1)
    print("\n=== Classification Report (train_only_original) ==="); print(rep1)
    print("\n=== Confusion Matrix (train_with_generated) ==="); print(cm2)
    print("\n=== Classification Report (train_with_generated) ==="); print(rep2)

    # ===== 新增：行归一化 + 保存四张图 =====
    np.set_printoptions(precision=3, suppress=True)
    cm1_row = row_normalize_confmat(cm1)
    cm2_row = row_normalize_confmat(cm2)

    save_cm_plots("train_only_original", cm1, cm1_row, TARGET_NAMES)
    save_cm_plots("train_with_generated", cm2, cm2_row, TARGET_NAMES)

    print("\n→ Results appended to compare_results.csv")
    print("→ 混淆矩阵图片输出至 ./figs/")




