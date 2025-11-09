import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from model.model import TransformerEncoder
from data.dataset import SmallTextDataset

# ------------------- 配置 -------------------
vocab_size = 100
seq_len = 16
batch_size = 32
epochs = 20
d_model = 64
num_heads = 4
d_ff = 256
num_layers = 2
lr = 1e-3
grad_clip = 1.0
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

# ------------------- 数据 -------------------
dataset = SmallTextDataset(vocab_size=vocab_size, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------- 模型 -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerEncoder(vocab_size=vocab_size,
                           d_model=d_model,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           d_ff=d_ff,
                           max_len=seq_len,
                           dropout=0.1).to(device)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# ------------------- 损失与优化器 -------------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ------------------- 训练 -------------------
loss_history = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # ---------------- 修正这里 ----------------
        logits, _, _ = model(x)  # TransformerEncoder 返回 (logits, attn_list, enc_output)

        # logits: (B, T, V), y: (B, T)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, f"transformer_epoch{epoch+1}.pt"))

# ------------------- 可视化训练曲线 -------------------
plt.figure()
plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.show()

