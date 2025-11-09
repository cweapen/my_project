import torch
from torch.utils.data import Dataset

class SmallTextDataset(Dataset):
    """
    简单示例数据集，输入输出都是 token ids
    实际可以替换成文本 tokenizer 后的 id
    """
    def __init__(self, vocab_size=100, seq_len=16, dataset_size=500):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.data = torch.randint(0, vocab_size, (dataset_size, seq_len), dtype=torch.long)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]  # 简单预测下一个 token
        return x, y
