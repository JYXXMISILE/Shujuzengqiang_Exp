import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.conditional_vqvae import ConditonalVQVAE
from data.data_loader import RespiratoryDataset  # 确保路径正确
# ----------------------------------------
# 1. 配置区
# ----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes    = 2
batch_size     = 16
num_workers    = 0
vqvae_ckpt     = r"E:\cat的方法\checkpoints\best_epoch50.pth"
output_root    = "class_all_skips"
os.makedirs(output_root, exist_ok=True)

train_feat_dir  = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio"
train_label_dir = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_label"


def transform_fn(x):
    x = torch.from_numpy(x).float()
    return (x - x.mean()) / (x.std() + 1e-6)

# 用位置参数，不要带 features_dir=... labels_dir=...
train_ds = RespiratoryDataset(
    train_feat_dir,
    train_label_dir,
    transform_fn
)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True
)
# ----------------------------------------
# 3. 加载 VQ-VAE，只用 encoder 和 label_embed
# ----------------------------------------
model = ConditonalVQVAE(
    num_labels=num_classes,
    feature_dim=163,
    latent_dim=64,
    codebook_size=128,
    label_embed_dim=16
).to(device)
model.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
model.eval()

encoder     = model.encoder
label_embed = model.label_embed

# ----------------------------------------
# 4. 按类别收集所有样本的 skips
# ----------------------------------------
# skip_all[c] 是一个列表，列表中的每一项又是一个“skip list”：
#   skip_all[c][i] = [skip_layer0, skip_layer1, ..., skip_layerL]
skip_all = {c: [] for c in range(num_classes)}

with torch.no_grad():
    for mel_batch, label_batch in tqdm(train_loader, desc="Collecting skips"):
        # 假设 mel_batch: [B, T_in, F]
        mel_batch   = mel_batch.to(device)
        label_batch = label_batch.to(device)

        # 计算 label embedding
        y_emb = label_embed(label_batch)  # [B, label_embed_dim]

        # 前向 encoder
        _, skips_list = encoder(mel_batch, y_emb)
        # skips_list: List[L] of [B, C_j, T_j]

        B = mel_batch.size(0)
        # 按样本逐一提取
        for i in range(B):
            c = int(label_batch[i].item())
            # 取出第 i 个样本的那一组 skips（List[L] of [C_j, T_j]）
            sample_skips = [layer_feat[i].detach().cpu() for layer_feat in skips_list]
            skip_all[c].append(sample_skips)

# 打印每个类别收集了多少条 skips
for c in range(num_classes):
    print(f"Class {c}: collected {len(skip_all[c])} skip-sets")

# ----------------------------------------
# 5. 保存到磁盘
# ----------------------------------------
for c, skips_list in skip_all.items():
    if not skips_list:
        print(f"Warning: class {c} has no samples, skipping save.")
        continue
    save_path = os.path.join(output_root, f"class{c}_all_skips.pt")
    # 保存一个 Python list of list of tensors
    torch.save(skips_list, save_path)
    print(f"→ saved {len(skips_list)} samples' skips for class {c} → {save_path}")

print("Done.")
