import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.data_loader import RespiratoryDataset
from model.conditional_vqvae import ConditonalVQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

train_feat_dir  = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio"
train_label_dir = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_label"


def transform_fn(x):
    """
    将 numpy.ndarray [T, F] 转为归一化的 torch.Tensor [T, F]
    """
    x = torch.from_numpy(x).float()
    return (x - x.mean()) / (x.std() + 1e-6)

train_dataset = RespiratoryDataset(train_feat_dir, train_label_dir, transform=transform_fn)

# 1) 实例化并加载训练好的模型权重
model = ConditonalVQVAE(
    num_labels=2,        # 必须跟原训练时一致
    feature_dim=163,     # 必须跟原训练时的输入维度一致
    latent_dim=64,       # 如果原来训练也是 64，就保持不变
    codebook_size=128,
    label_embed_dim=16
).to(device)
model.load_state_dict(torch.load(r"E:\cat的方法\checkpoints\best_epoch50.pth"))
model.eval()

# 2) 构建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

all_indices = []  # 用来积累每个样本的 [T_lat] 索引
all_labels  = []  # 如果需要 y，也一并积累

with torch.no_grad():
    for x, y in train_loader:
        x = x.to(device)  # [B, T_in, feature_dim]
        y = y.to(device)  # [B]

        # 3) 前向计算：拿到 model.last_indices
        _x_recon, _logits, _vq_loss = model(x, y)
        indices_batch = model.last_indices  # [B, T_lat]

        # 4) 存到列表里
        all_indices.append(indices_batch.cpu())  
        all_labels.append(y.cpu())

# 5) 拼接所有 batch，得到 [N, T_lat]
all_indices = torch.cat(all_indices, dim=0)
all_labels  = torch.cat(all_labels,  dim=0)

# 6) 保存到磁盘
torch.save(all_indices, "VAE/train_discrete_indices.pt")
torch.save(all_labels, "VAE/train_labels.pt")

print("已生成索引文件：", all_indices.shape, all_labels.shape)
