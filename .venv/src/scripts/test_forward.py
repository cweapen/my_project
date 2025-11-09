# test_forward.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.model_1 import TransformerEncoder

def simple_test():
    vocab_size = 100
    batch_size = 2
    seq_len = 16
    # create random token ids
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    model = TransformerEncoder(vocab_size=vocab_size,
                               d_model=64,
                               num_layers=2,
                               num_heads=4,
                               d_ff=256,
                               max_len=128,
                               dropout=0.1,
                               learned_pe=False)
    logits, attn_list = model(x)  # logits: (B, T, V)
    print("logits shape:", logits.shape)  # expect (2, 16, 100)
    for i, a in enumerate(attn_list):
        print(f"layer {i} attn shape:", a.shape)  # expect (B, h, T, T)
    # quick sanity: sum of attn weights across last dim should be 1 (softmax)
    print("attn sum (layer0, first head, first sample, first query):",
          attn_list[0][0,0,0,:].sum().item())

if __name__ == "__main__":
    simple_test()
