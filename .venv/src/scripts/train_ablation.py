# scripts/train_module_ablation.py
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_2 import SmallTransformer  # 支持 relative_position/local_window 参数

# ------------------------ ToyDataset ------------------------
class ToyDataset(Dataset):
    """随机生成的序列预测任务"""
    def __init__(self, seq_len=16, vocab_size=50, dataset_size=1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dataset_size = dataset_size
        self.data = torch.randint(0, vocab_size, (dataset_size, seq_len))
        self.labels = torch.randint(0, vocab_size, (dataset_size, seq_len))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloader(batch_size=16):
    train_dataset = ToyDataset(seq_len=16, vocab_size=50, dataset_size=1000)
    valid_dataset = ToyDataset(seq_len=16, vocab_size=50, dataset_size=200)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader

# ------------------------ 训练函数 ------------------------
def train_model(model, loader, device, epochs, vocab_size, use_scheduler=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    if use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    loss_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, x)  # src=tgt
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_list

# ------------------------ 主流程 ------------------------
def train_module_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50
    seq_len = 16
    batch_size = 16
    epochs = 20

    train_loader, valid_loader = get_dataloader(batch_size=batch_size)

    results = {}

    # ---------------- 1. 全功能模型 ----------------
    print("==== Training Full Model ====")
    model_full = SmallTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=seq_len,
        dropout=0.1,
        relative_position=True,
        local_window=5
    ).to(device)
    results['Full Model'] = train_model(model_full, train_loader, device, epochs, vocab_size)

    # ---------------- 2. 去掉相对位置编码 ----------------
    print("==== Training Without Relative Position ====")
    model_no_relpos = SmallTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=seq_len,
        dropout=0.1,
        relative_position=False,  # 禁用相对位置编码
        local_window=5
    ).to(device)
    results['No Relative Position'] = train_model(model_no_relpos, train_loader, device, epochs, vocab_size)

    # ---------------- 3. 去掉局部稀疏注意力 ----------------
    print("==== Training Without Sparse Attention ====")
    model_no_sparse = SmallTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=seq_len,
        dropout=0.1,
        relative_position=True,
        local_window=None  # 禁用局部窗口 -> 全局注意力
    ).to(device)
    results['No Sparse Attention'] = train_model(model_no_sparse, train_loader, device, epochs, vocab_size)

    # ---------------- 4. 去掉学习率调度器 ----------------
    print("==== Training Without LR Scheduler ====")
    model_no_scheduler = SmallTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=seq_len,
        dropout=0.1,
        relative_position=True,
        local_window=5
    ).to(device)
    results['No LR Scheduler'] = train_model(model_no_scheduler, train_loader, device, epochs, vocab_size, use_scheduler=False)

    # ---------------- 绘制消融曲线 ----------------
    os.makedirs("../results", exist_ok=True)
    plt.figure(figsize=(7,5))
    for key, loss in results.items():
        plt.plot(range(1, epochs+1), loss, marker='o', label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Module Ablation Study (ToyDataset)")
    plt.grid(True)
    plt.legend()
    plt.savefig("../results/module_ablation_loss.png")
    plt.show()

if __name__ == "__main__":
    train_module_ablation()
