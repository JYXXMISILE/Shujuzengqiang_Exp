# train.py

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.data_loader import RespiratoryDataset
from model.conditional_vqvae import ConditonalVQVAE

def transform_fn(x):
    """
    将 numpy.ndarray [T, F] 转为归一化的 torch.Tensor [T, F]
    """
    x = torch.from_numpy(x).float()
    return (x - x.mean()) / (x.std() + 1e-6)

def train(
    train_feat_dir: str,
    train_label_dir: str,
    val_feat_dir: str,
    val_label_dir: str,
    num_labels: int = 2,
    feature_dim: int = 163,
    latent_dim: int = 64,
    codebook_size: int = 128,
    label_embed_dim: int = 16,
    commitment_cost: float = 0.25,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 50,
    lambda_cls: float = 0.01,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    # —— DataLoader —— #
    train_ds = RespiratoryDataset(train_feat_dir, train_label_dir, transform=transform_fn)
    val_ds   = RespiratoryDataset(val_feat_dir,   val_label_dir,   transform=transform_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    import collections
    # 在构建 val_loader 之后
    val_labels_all = []
    for _, y in val_loader:
        val_labels_all.extend(y.tolist())
    print(collections.Counter(val_labels_all))

    # —— Model & Optim —— #
    model = ConditonalVQVAE(
        num_labels=num_labels,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        label_embed_dim=label_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses   = []

    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # --- 训练 --- #
        model.train()
        running_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
        for x_batch, y_batch in train_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_recon, logits, vq_loss = model(x_batch, y_batch)

            recon_loss = F.mse_loss(x_recon, x_batch)
            cls_loss   = F.cross_entropy(logits, y_batch)
            loss       = recon_loss + vq_loss + lambda_cls * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * x_batch.size(0)
            train_bar.set_postfix(train_loss=loss.item())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- 验证 --- #
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total   = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for x_val, y_val in val_bar:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_recon, logits, vq_loss = model(x_val, y_val)

                recon_loss = F.mse_loss(x_recon, x_val, reduction='sum')
                cls_loss   = F.cross_entropy(logits, y_val, reduction='sum')
                loss       = recon_loss + vq_loss * x_val.size(2) + 0.01 * cls_loss

                running_val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total   += y_val.size(0)

                val_bar.set_postfix(val_loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(epoch_val_loss)

        # 打印本 epoch 总结
        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Train Loss: {epoch_train_loss:.4f}  "
              f"Val  Loss: {epoch_val_loss:.4f}  "
              f"Val  Acc:  {val_acc:.4f}")

        # 保存最优模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join("checkpoints", f"best_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Saved best model to {ckpt_path}")

    # —— 绘制损失曲线 —— #
    epochs = list(range(1, num_epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("--> Saved loss curve to loss_curve.png")
    plt.show()

if __name__ == "__main__":
    # 请根据实际路径修改
    train_feat_dir  = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio"
    train_label_dir = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_label"
    val_feat_dir    = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_audio"
    val_label_dir   = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_label"

    train(
        train_feat_dir,
        train_label_dir,
        val_feat_dir,
        val_label_dir,
        num_labels=2,
        feature_dim=163,
        latent_dim=64,
        codebook_size=128,
        label_embed_dim=16,
        commitment_cost=0.25,
        batch_size=32,
        lr=1e-3,
        num_epochs=50,
        lambda_cls=1.0
    )
