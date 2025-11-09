from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, path, seq_len=32):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().lower()
        self.chars = sorted(list(set(text)))
        self.vocab = {c: i for i, c in enumerate(self.chars)}
        self.inv_vocab = {i: c for c, i in self.vocab.items()}
        self.ids = [self.vocab[c] for c in text]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y
