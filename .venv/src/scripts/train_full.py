import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model.model import TransformerEncoder
from data.data_set import TextDataset
from data.download_data import download_wikitext2

# ------------------------
# 配置参数
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 64
num_layers = 2
num_heads = 4
d_ff = 256
max_len = 128
dropout = 0.1

batch_size = 16
epochs = 30
lr = 1e-3
weight_decay = 1e-4
grad_clip = 1.0
save_path = "../results/small_transformer.pth"

# ------------------------
# 下载并加载数据集
# ------------------------
download_wikitext2("data")  # 下载小数据集（可选截取前5万字符）
train_dataset = TextDataset("data/train.txt", seq_len=32)
val_dataset = TextDataset("data/valid.txt", seq_len=32)
vocab_size = len(train_dataset.vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ------------------------
# 模型
# ------------------------
model = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    max_len=max_len,
    dropout=dropout,
    learned_pe=False
).to(device)

# ------------------------
# 参数统计
# ------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# ------------------------
# 损失、优化器、调度器
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# ------------------------
# 可选：加载已保存模型
# ------------------------
if os.path.exists(save_path):
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    print("Loaded saved model from", save_path)

# ------------------------
# 训练循环
# ------------------------
loss_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)  # (B, T, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        # 梯度裁剪
        clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ------------------------
    # 每 5 个 epoch 保存一次模型
    # ------------------------
    if (epoch + 1) % 5 == 0:
        os.makedirs("../results", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch + 1
        }, save_path)
        print(f"Model saved to {save_path}")

# ------------------------
# 绘制训练曲线
# ------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("../results/loss_curve.png")
plt.show()
