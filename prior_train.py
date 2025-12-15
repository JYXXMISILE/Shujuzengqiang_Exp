import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model.small_transformer_prior import SmallTransformerPrior
from data.prior_dataset import PriorDataset  

# 超参数（根据你的任务调整）
num_embeddings = 128     # Codebook 大小
emb_dim = 64             # Token/Pos/Cond 嵌入维度
condition_dim = 2        # 条件标签种类数
n_layers = 2
n_heads = 4
ff_hidden = 128
max_len = 6
batch_size = 32
lr = 1e-3
num_epochs = 25
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 读取离散索引数据集
dataset = PriorDataset(
    indices_path=r"E:\cat的方法\VAE\train_discrete_indices.pt",  # [N, T]
    labels_path=r"E:\cat的方法\VAE\train_labels.pt"               # [N]
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 2) 实例化并移动到设备
model = SmallTransformerPrior(
    num_embeddings=num_embeddings,
    emb_dim=emb_dim,
    condition_dim=condition_dim,
    n_layers=n_layers,
    n_heads=n_heads,
    ff_hidden=ff_hidden,
    max_len=max_len
).to(device)

# 3) 优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 4) 开始训练
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for idx_seq_batch, label_batch in loader:
        # idx_seq_batch: [B, T], label_batch: [B]
        idx_seq_batch = idx_seq_batch.to(device)   # LongTensor
        label_batch = label_batch.to(device)       # LongTensor

        # 预测每个位置下一个索引：input = idx[:, :-1], target = idx[:, 1:]
        input_seq = idx_seq_batch[:, :-1]  # [B, T-1]
        target_seq = idx_seq_batch[:, 1:]  # [B, T-1]

        # forward
        logits = model(input_seq, label_batch)  # [B, T-1, num_embeddings]

        # reshape 以匹配 CrossEntropyLoss: [B*(T-1), num_embeddings]
        B, Tm1, K = logits.shape
        logits_flat = logits.reshape(-1, K)
        target_flat = target_seq.reshape(-1)

        loss = criterion(logits_flat, target_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

# 训练完毕后保存模型
torch.save(model.state_dict(), "VAE/small_transformer_prior.pth")
print("Training complete. Model saved to small_transformer_prior.pth.")
