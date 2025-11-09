import torch

def subsequent_mask(size):
    """Mask out subsequent positions for decoder self-attention (causal)."""
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return ~mask  # True for allowed, False for masked
