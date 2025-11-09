# scripts/test_transformer_challenge.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from model.model_2 import SmallTransformer

# ------------------------ 小数据集示例 ------------------------
class ToyDataset(Dataset):
    """随机生成的分类任务，vocab_size=50"""
    def __init__(self, seq_len=20, vocab_size=50, dataset_size=2000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dataset_size = dataset_size
        self.data = torch.randint(0, vocab_size, (dataset_size, seq_len))
        self.labels = torch.randint(0, vocab_size, (dataset_size, seq_len))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloader(batch_size=32):
    train_dataset = ToyDataset()
    valid_dataset = ToyDataset(dataset_size=500)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader

# ------------------------ 配置 ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 50
seq_len = 20

# 可调超参，用于敏感性分析
d_model = 64
num_layers = 2
num_heads = 4
d_ff = 256
dropout = 0.1
epochs = 10
batch_size = 32
lr = 1e-3
weight_decay = 1e-4
grad_clip = 1.0

# 相对位置编码 / 局部稀疏注意力开关
relative_position = True
local_window = 5

save_path = "../results/challenge_transformer.pth"

# ------------------------ 数据 ------------------------
train_loader, valid_loader = get_dataloader(batch_size=batch_size)

# ------------------------ 模型 ------------------------
model = SmallTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    max_len=seq_len,
    dropout=dropout,
    relative_position=relative_position,
    local_window=local_window
).to(device)

# ------------------------ 参数统计 ------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}, Trainable: {trainable_params}")

# ------------------------ 优化器、调度器 ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# ------------------------ 训练 ------------------------
train_loss_history, valid_loss_history = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, x)  # src=tgt=toy数据
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # ----------------- 验证 -----------------
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x_val, y_val in valid_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits_val = model(x_val, x_val)
            loss_val = criterion(logits_val.view(-1, vocab_size), y_val.view(-1))
            valid_loss += loss_val.item()
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_loss_history.append(avg_valid_loss)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存模型
    if (epoch + 1) % 5 == 0:
        os.makedirs("../results", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# ------------------------ 绘制训练曲线 ------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, epochs+1), train_loss_history, label="Train")
plt.plot(range(1, epochs+1), valid_loss_history, label="Valid")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Challenge Transformer Training Curve")
plt.grid(True)
plt.legend()
plt.savefig("../results/challenge_loss_curve.png")
plt.show()
