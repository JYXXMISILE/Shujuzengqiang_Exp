import os
import numpy as np
import torch
import torch.nn.functional as F
from model.small_transformer_prior import SmallTransformerPrior
from model.conditional_vqvae import ConditonalVQVAE

# ----------------------------------------
# 配置区：根据你自己训练时的参数修改
# ----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prior 模型（小 Transformer）的参数
K = 128               # Codebook 大小（和你 VQ-VAE 一致）
emb_dim = 64          # Prior 中 token/pos/cond embedding 维度
condition_dim = 2     # 类别数（这里我们有 3 个类别）
n_layers = 2
n_heads = 4
ff_hidden = 128
max_len = 6       # Prior 训练/采样时允许的最大序列长度
T = 6            # 要生成的索引序列长度

# Decoder（ConditionalVQVAE）参数
latent_dim = 64
feature_dim = 163     # Decoder 输出维度（例如梅尔谱的频率 bins 数）
label_embed_dim = 16  # VQ-VAE 中 label embedding 维度
target_length = 192   # Decoder 输出的时间帧数（要和训练时一致）

# 采样时使用的温度
temperature = 0.8

# 每个类别生成多少条索引（这里示例设为 1 条；可改成 >1）
samples_per_class = 10000

# Prior 模型 checkpoint 路径
prior_ckpt = r"E:\cat的方法\VAE\small_transformer_prior.pth"

# VQ-VAE 模型 checkpoint 路径
vqvae_ckpt = r"E:\cat的方法\checkpoints\best_epoch50.pth"

# “平均 skip 模板” 存放目录
skip_template_dir = "class_all_skips"  # 请确保该目录下已有 class{c}_avg_skips.pt

# 输出目录：下面会在此目录下创建 3 个子目录
out_dir = "generated_by_class20k"
indices_dir = os.path.join(out_dir, "indices")
mels_dir    = os.path.join(out_dir, "mels")
labels_dir  = os.path.join(out_dir, "labels")

# 确保输出目录及子目录存在
os.makedirs(indices_dir, exist_ok=True)
os.makedirs(mels_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 类别索引到文本的映射，请根据实际任务自行修改
label_map = {
    0: "正常",
    1: "湿罗音"
}
assert set(label_map.keys()) == set(range(condition_dim)), "请确保 label_map 和 condition_dim 匹配"


# ----------------------------------------
# 1. 定义并行采样函数
# ----------------------------------------
def sample_prior_for_class(
    cond_id: int,
    B
) -> np.ndarray:
    """
    并行生成 B 条长度为 T 的离散索引序列，均属于类别 cond_id。
    返回一个 numpy array，形状 (B, T)，dtype=int
    """
    # 1.1) 实例化并加载 Prior 模型
    model = SmallTransformerPrior(
        num_embeddings=K,
        emb_dim=emb_dim,
        condition_dim=condition_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_hidden=ff_hidden,
        max_len=max_len
    ).to(device)
    model.load_state_dict(torch.load(prior_ckpt, map_location=device))
    model.eval()

    # 1.2) 批量生成的张量 [B, T]
    generated = torch.zeros((B, T), dtype=torch.long, device=device)
    # 1.3) 批量的类别标签 [B]
    cond = torch.full((B,), cond_id, dtype=torch.long, device=device)

    for t in range(T):
        if t == 0:
            # t=0：没有历史，用均匀分布采样
            logits_t = torch.zeros((B, K), device=device)  # [B, K]
            probs = F.softmax(logits_t / temperature, dim=-1)  # [B, K]
            idxs = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            generated[:, 0] = idxs
        else:
            # t>0：用前 t 步生成的索引和 cond 去预测第 t 步
            input_seq = generated[:, :t]       # [B, t]
            logits = model(input_seq, cond)    # [B, t, K]
            logits_t = logits[:, -1, :]        # [B, K]
            logits_t = logits_t / temperature
            probs = F.softmax(logits_t, dim=-1)  # [B, K]
            idxs = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            generated[:, t] = idxs

    return generated.cpu().numpy()  # (B, T) ndarray


# ----------------------------------------
# 2. 定义解码函数：离散索引 → 梅尔谱特征
# ----------------------------------------
def decode_indices_to_mel(
    gen_indices: np.ndarray,
    cond_id: int
) -> np.ndarray:
    """
    把一批 [B, T] 的离散索引序列映射到 Decoder 上，得到 [B, target_length, feature_dim] 的特征。
    会根据 cond_id 自动加载对应类别的平均 skips 模板（class{cond_id}_avg_skips.pt）。
    """
    B = gen_indices.shape[0]

    # 1) 实例化并加载 VQ-VAE 模型
    model = ConditonalVQVAE(
        num_labels=condition_dim,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        codebook_size=K,
        label_embed_dim=label_embed_dim
    ).to(device)
    model.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    model.eval()

    # 2) 把索引序列转成 Tensor: [B, T]
    idx_seq = torch.from_numpy(gen_indices).long().to(device)  # [B, T]

    # 3) 从 quantizer 拿到 codebook 嵌入: [K, D]
    codebook = model.quantizer.embeddings.weight  # [K, D]

    # 4) 把 idx_seq 映射回潜在向量: [B, T, D]
    z_q = codebook[idx_seq]                        # [B, T, D]

    # 5) 转成 Decoder 需要的格式: [B, D, T]
    z_q = z_q.permute(0, 2, 1).contiguous()        # [B, D, T]

    # 6) 准备条件 embedding: [B, label_embed_dim]
    cond = torch.full((B,), cond_id, dtype=torch.long, device=device)  # [B]
    y_emb = model.label_embed(cond)  # [B, label_embed_dim]

   # —— 7) 从 class_all_skips 根目录加载该类的所有 skips 列表 —— #
    all_skips_path = os.path.join(skip_template_dir, f"class{cond_id}_all_skips.pt")
    if not os.path.exists(all_skips_path):
        raise FileNotFoundError(f"未找到类别 {cond_id} 的 skip 列表: {all_skips_path}")

    # all_skips: List of N_c items, each item is a List of L torch.Tensor
    all_skips: list = torch.load(all_skips_path, map_location="cpu")

    # 8) 对批量中的每条样本随机抽一个 skip-list
    #    selected_skips[i] 是第 i 条生成样本对应的 L 层 skip-list
    import random
    selected_skips = [ random.choice(all_skips) for _ in range(B) ]

    # 9) 按层堆叠，把 selected_skips 从 List[B][List[L][C_j,T_j]] 
    #    变成 skips_for_dec: List[L] of [B,C_j,T_j]
    num_layers = len(selected_skips[0])
    skips_for_dec = []
    for layer_idx in range(num_layers):
        # 收集这一层所有 B 条样本的 skip 张量，并在第 0 维堆叠
        layer_feats = [ sample[layer_idx] for sample in selected_skips ]  # list of [C_j,T_j]
        layer_stack = torch.stack(layer_feats, dim=0).to(device)          # [B, C_j, T_j]
        skips_for_dec.append(layer_stack)

    # 8) 调用 decoder，传入平均 skips
    with torch.no_grad():
        x_recon = model.decoder(
            z_q,
            y_emb,
            skips_for_dec,        # 传入同类别的平均 skip
            target_length=target_length
        )  # [B, target_length, feature_dim]

    return x_recon.detach().cpu().numpy()  # 返回 numpy: (B, target_length, feature_dim)


# ----------------------------------------
# 3. 主流程：对每个类别生成并保存到三个子目录
# ----------------------------------------
if __name__ == "__main__":
    for cond_id in range(condition_dim):  # cond_id = 0, 1, 2
        # 3.1) 并行生成 samples_per_class 条索引序列 (shape=(samples_per_class, T))
        gen_indices = sample_prior_for_class(
            cond_id=cond_id,
            B=samples_per_class
        )

        # 3.2) 解码为梅尔谱特征，shape = (samples_per_class, target_length, feature_dim)
        mel_features = decode_indices_to_mel(
            gen_indices=gen_indices,
            cond_id=cond_id
        )

        # 3.3) 保存每条索引、梅尔谱和标签文本到对应子目录
        for i in range(samples_per_class):
            # 当前索引序列：[T,]
            idx_seq = gen_indices[i]
            # 当前梅尔特征： [target_length, feature_dim]
            mel = mel_features[i]

            # 文件名前缀： class{cond_id}_sample{i}
            prefix = f"class{cond_id}_sample{i}"

            # --- 3.3.1) 保存索引 npy 到 indices/ 子目录 ---
            idx_npy_fn = prefix + "_idx.npy"
            idx_npy_path = os.path.join(indices_dir, idx_npy_fn)
            np.save(idx_npy_path, idx_seq)

            # --- 3.3.2) 保存梅尔特征 npy 到 mels/ 子目录 ---
            mel_npy_fn = prefix + "_mel.npy"
            mel_npy_path = os.path.join(mels_dir, mel_npy_fn)
            np.save(mel_npy_path, mel)

            # --- 3.3.3) 保存标签 txt 到 labels/ 子目录 ---
            txt_fn = prefix + "_label.txt"
            txt_path = os.path.join(labels_dir, txt_fn)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{label_map[cond_id]}\n")

            print(f"[类别 {cond_id} - 样本 {i}] 已保存：")
            print(f"  • 索引 → {idx_npy_path}")
            print(f"  • 梅尔谱 → {mel_npy_path}")
            print(f"  • 标签 → {txt_path}（{label_map[cond_id]}）\n")

    print(f"全部保存完成，目录结构为：\n"
          f"{out_dir}/\n"
          f"  ├─ indices/   (所有索引 *.npy)\n"
          f"  ├─ mels/      (所有梅尔特征 *.npy)\n"
          f"  └─ labels/    (所有标签 *.txt)\n")
