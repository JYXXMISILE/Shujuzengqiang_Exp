# -*- coding: utf-8 -*-
# cnn14_eval_offline_aug.py —— 二分类（正常/湿罗音）+ 支持合成数据混训 + 离线传统增强

import os, csv, random, sys, time
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

SYN_MEL_DIR     = r""
SYN_LABEL_DIR   = r""

# 离线增强输出目录（原始+生成一起增强后放这里）
AUG_MEL_DIR     = r"E:\cat的方法\传统增强方式测试\offline_aug_mels"
AUG_LABEL_DIR   = r"E:\cat的方法\传统增强方式测试\offline_aug_labels"

# 预训练 CNN14 权重（PANNs 官方发布的 Audioset 权重）
CNN14_CKPT      = r"E:\cat的方法\VAE\Cnn14_mAP=0.431.pth"  # 改成你的路径

# ============ 超参 ============
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3

SYN_PER_REAL = 50.0 # 真实样本 : 合成样本 = 1 : SYN_PER_REAL（用于之前的对比实验，这里主要看全量版本）
WEIGHT_REAL  = 1.0
WEIGHT_SYN   = 1.0

MIN_ENERGY   = 1e-4
MIN_VAR      = 1e-6
N_MELS       = 163      # 你的特征频点数

# 离线增强超参
RUN_OFFLINE_AUG   = True     # 是否执行离线增强
OFFLINE_AUG_TIMES = 3        # 每条样本生成多少个增强样本
OFFLINE_AUG_CFG   = {
    "time_shift_max": 10,    # 时间轴平移最大帧数
    "noise_std": 0.01,       # 高斯噪声标准差
    "time_mask_max_width": 20,
    "freq_mask_max_width": 8,
}


# ============ 导入 PANNs Cnn14 ============
repo_pytorch = r"E:\cat的方法\third_party\audioset_tagging_cnn\pytorch"
if repo_pytorch not in sys.path:
    sys.path.insert(0, repo_pytorch)
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
        if not os.path.isdir(feat_dir):
            raise RuntimeError(f"Feat dir not found: {feat_dir}")
        files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        self.samples = []; dropped = 0
        for fn in files:
            base = os.path.splitext(fn)[0]
            feat_path = os.path.join(feat_dir, fn)
            # 标签规则：去掉 "_mel" / "_idx"，再加 .txt
            label_path= os.path.join(label_dir, base.replace("_mel","").replace("_idx","") + ".txt")
            if not os.path.exists(label_path):
                continue
            try:
                mel = np.load(feat_path)
                if mel.ndim != 2:
                    if mel.shape[0] < mel.shape[1]:
                        mel = mel.T
                if quality_filter:
                    if (not np.isfinite(mel).all()) or np.mean(mel**2) < MIN_ENERGY or np.var(mel) < MIN_VAR:
                        dropped += 1; continue
                with open(label_path, "r", encoding="utf-8") as f:
                    txt = f.readline().strip().lower()
                if txt not in LABEL_ALIASES:
                    continue
                y = LABEL_ALIASES[txt]
                if y not in ALL_LABELS:
                    continue
                self.samples.append((feat_path, y))
            except Exception:
                dropped += 1
                continue
        print(f"[{'SYN' if is_synth else 'REAL'}] Loaded {len(self.samples)} samples from {feat_dir} (dropped {dropped})")

    def __len__(self):
        return len(self.samples)

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
        feat = feat_4d.mean(dim=-1)          # 频率均值 → [B,2048, T/32]
        feat = self.proj(feat)               # [B,512, T/32]
        logits = self.head(feat)             # [B, num_classes]
        return logits


# ============ 离线传统增强（在 mel 空间） ============
def _time_shift(mel: np.ndarray, max_shift: int) -> np.ndarray:
    if max_shift <= 0:
        return mel
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return mel
    return np.roll(mel, shift=shift, axis=0)  # 时间轴


def _add_noise(mel: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return mel
    noise = np.random.normal(0.0, std, size=mel.shape).astype(np.float32)
    return mel + noise


def _time_mask(mel: np.ndarray, max_width: int) -> np.ndarray:
    T, F = mel.shape
    if max_width <= 0 or T <= 1:
        return mel
    w = np.random.randint(1, max_width + 1)
    w = min(w, T - 1)
    t0 = np.random.randint(0, T - w + 1)
    mel2 = mel.copy()
    mel2[t0:t0+w, :] = 0.0
    return mel2


def _freq_mask(mel: np.ndarray, max_width: int) -> np.ndarray:
    T, F = mel.shape
    if max_width <= 0 or F <= 1:
        return mel
    w = np.random.randint(1, max_width + 1)
    w = min(w, F - 1)
    f0 = np.random.randint(0, F - w + 1)
    mel2 = mel.copy()
    mel2[:, f0:f0+w] = 0.0
    return mel2


def _apply_offline_aug(mel: np.ndarray, cfg: dict) -> np.ndarray:
    """对一条 mel 做一套固定强度的增强，用于离线复制新样本。"""
    mel_aug = mel.copy()
    mel_aug = _time_shift(mel_aug, cfg.get("time_shift_max", 0))
    mel_aug = _add_noise(mel_aug, cfg.get("noise_std", 0.0))
    mel_aug = _time_mask(mel_aug, cfg.get("time_mask_max_width", 0))
    mel_aug = _freq_mask(mel_aug, cfg.get("freq_mask_max_width", 0))
    return mel_aug


def _list_mel_files(feat_dir: str):
    if not os.path.isdir(feat_dir):
        return []
    return [f for f in os.listdir(feat_dir) if f.endswith('.npy')]


def offline_augment_all():
    """
    把【真实训练集 + 生成数据】这两份 mel npy 混在一起做“集中离线增强”，
    结果写到 AUG_MEL_DIR / AUG_LABEL_DIR。
    """
    if not RUN_OFFLINE_AUG:
        print("[OfflineAug] RUN_OFFLINE_AUG=False, 跳过离线增强")
        return

    os.makedirs(AUG_MEL_DIR, exist_ok=True)
    os.makedirs(AUG_LABEL_DIR, exist_ok=True)

    orig_files = _list_mel_files(REAL_TRAIN_FEAT)
    synth_files = _list_mel_files(SYN_MEL_DIR)

    entries = []
    entries += [(REAL_TRAIN_FEAT, REAL_TRAIN_LABEL, f) for f in orig_files]
    entries += [(SYN_MEL_DIR,     SYN_LABEL_DIR,     f) for f in synth_files]

    print(f"[OfflineAug] 原始 train mel 数量: {len(orig_files)}")
    print(f"[OfflineAug] 生成 mel 数量     : {len(synth_files)}")
    print(f"[OfflineAug] 总计参与增强样本 : {len(entries)}")
    print(f"[OfflineAug] 每条生成 {OFFLINE_AUG_TIMES} 条增强样本")

    total_new = 0
    t0 = time.perf_counter()

    for idx, (feat_dir, label_dir, fn) in enumerate(entries):
        feat_path = os.path.join(feat_dir, fn)
        base = os.path.splitext(fn)[0]               # e.g. class0_sample0_mel
        # 原标签名：去掉 "_mel"/"_idx"
        label_base = base.replace("_mel", "").replace("_idx", "")
        label_path = os.path.join(label_dir, label_base + ".txt")
        if not os.path.exists(label_path):
            continue

        mel = np.load(feat_path).astype(np.float32)
        if mel.ndim != 2 and mel.shape[0] < mel.shape[1]:
            mel = mel.T

        # 标签内容（原文）
        with open(label_path, "r", encoding="utf-8") as f:
            lab_str = f.readline().strip()

        for k in range(OFFLINE_AUG_TIMES):
            mel_aug = _apply_offline_aug(mel, OFFLINE_AUG_CFG)
            new_base = f"{base}_aug{k}"               # e.g. class0_sample0_mel_aug0
            new_feat_path = os.path.join(AUG_MEL_DIR, new_base + ".npy")
            # 对应标签名：去掉 "_mel" / "_idx"
            new_label_name = new_base.replace("_mel", "").replace("_idx", "") + ".txt"
            new_label_path = os.path.join(AUG_LABEL_DIR, new_label_name)

            np.save(new_feat_path, mel_aug)
            with open(new_label_path, "w", encoding="utf-8") as f:
                f.write(lab_str + "\n")
            total_new += 1

        if (idx + 1) % 200 == 0:
            print(f"[OfflineAug] 已处理 {idx+1}/{len(entries)} 条, 当前生成 {total_new} 条增强样本")

    t1 = time.perf_counter()
    print(f"[OfflineAug] 完成！总共生成 {total_new} 条增强样本，用时 {t1 - t0:.2f} 秒")
    print(f"[OfflineAug] 增强样本存放于: {AUG_MEL_DIR}")


# ============ 数据构建 ============
def make_real_datasets():
    train_real = MelNPYDataset(REAL_TRAIN_FEAT, REAL_TRAIN_LABEL, is_synth=False)
    test_real  = MelNPYDataset(REAL_TEST_FEAT,  REAL_TEST_LABEL,  is_synth=False, quality_filter=False)
    return train_real, test_real


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
    if normalize:
        cm_plot = cm.astype(np.float32)
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sum, out=np.zeros_like(cm_plot), where=row_sum != 0)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (row-wise)'
    else:
        cm_plot = cm.astype(np.int64)
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
    class_weight = torch.tensor(cls_w_np / (cls_w_np.sum()/len(cls_w_np)),
                                dtype=torch.float32, device=device)

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
            rw = (synth_flag.to(device).float()*WEIGHT_SYN
                  + (1 - synth_flag.to(device).float())*WEIGHT_REAL)
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
        if new_file:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


# ============ 关键：三源合并训练集 ============

def make_train_all_sources():
    """
    训练集 = 真实 + VAE 生成 + 离线增强样本
    这里不再对 VAE/增强做上限裁剪，多少全吃。
    """
    ds_list = []

    # 1. 真实训练集
    train_real = MelNPYDataset(REAL_TRAIN_FEAT, REAL_TRAIN_LABEL, is_synth=False)
    ds_list.append(train_real)
    print(f"[ALL] real train: {len(train_real)}")

    # 2. VAE 生成数据
    if os.path.isdir(SYN_MEL_DIR):
        ds_syn = MelNPYDataset(SYN_MEL_DIR, SYN_LABEL_DIR, is_synth=True)
        ds_list.append(ds_syn)
        print(f"[ALL] vae synth: {len(ds_syn)}")
    else:
        print(f"[ALL] SYN_MEL_DIR 不存在，跳过 VAE 生成数据: {SYN_MEL_DIR}")

    # 3. 离线增强样本（刚才生成的 7 万多条就从这里进来）
    if os.path.isdir(AUG_MEL_DIR):
        # 这里关掉 quality_filter，避免增强样本被一股脑 drop 掉
        ds_aug = MelNPYDataset(AUG_MEL_DIR, AUG_LABEL_DIR,
                               is_synth=True, quality_filter=False)
        ds_list.append(ds_aug)
        print(f"[ALL] offline aug: {len(ds_aug)}")
    else:
        print(f"[ALL] AUG_MEL_DIR 不存在，跳过离线增强数据: {AUG_MEL_DIR}")

    if not ds_list:
        raise RuntimeError("没有任何训练数据源，请检查路径配置。")

    train_all = ConcatDataset(ds_list)
    print(f"[ALL] 最终训练集总样本数: {len(train_all)}")
    return train_all


# ============ 主程序：只跑“原始+VAE+离线增强”这一组 ============

if __name__ == "__main__":
    # 1) 先做一次离线增强（真实 + VAE 一起增强，生成 ~7.5 万条）
    offline_augment_all()

    # 2) 构建 “三合一” 训练集：real + vae + offline_aug
    train_all = make_train_all_sources()

    # 3) 测试集仍然用真实 val 集
    _, test_real = make_real_datasets()

    # 4) 在三合一训练集上训练 CNN14
    model_all, acc_all, f1m_all, cm_all, rep_all = train_and_eval(
        tag="train_real+vae+offline_aug",
        train_set=train_all,
        test_set=test_real
    )

    # 5) 记录结果
    rows = [
        ["train_real+vae+offline_aug",
         "CNN14",
         f"{acc_all:.6f}",
         f"{f1m_all:.6f}",
         SYN_PER_REAL, WEIGHT_REAL, WEIGHT_SYN, EPOCHS],
    ]
    save_csv(rows, out_csv="compare_results_real_vae_offline_aug.csv")

    print("\n=== Confusion Matrix (train_real+vae+offline_aug) ===")
    print(cm_all)
    print("\n=== Classification Report (train_real+vae+offline_aug) ===")
    print(rep_all)

    save_cm_figure(cm_all, TARGET_NAMES,
                   'cm_train_real+vae+offline_aug_counts.png',
                   normalize=False)
    save_cm_figure(cm_all, TARGET_NAMES,
                   'cm_train_real+vae+offline_aug_norm.png',
                   normalize=True)

    print("\n→ Results appended to compare_results_real_vae_offline_aug.csv")




