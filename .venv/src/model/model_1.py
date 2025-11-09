# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (non-learnable) or learned if learned=True.
    Returns a tensor of shape (1, max_len, d_model) registered as buffer or parameter.
    """
    def __init__(self, d_model, max_len=5000, learned=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        if learned:
            pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(pe, mean=0.0, std=0.02)
            self.pe = pe
            self.learned = True
        else:
            # create sinusoidal encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
            self.learned = False

    def forward(self, x):
        # x shape: (B, T, d_model)
        T = x.size(1)
        if self.learned:
            return self.pe[:, :T, :]  # parameter
        else:
            return self.pe[:, :T, :]  # buffer

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    Input: x (B, T, d_model)
    Output: out (B, T, d_model), attn_weights (B, num_heads, T, T)
    """
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # combined linear projections for QKV can be used; here use separate for clarity
        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (B, T, d_model)
        mask: (B, 1, 1, T) or (B, 1, T, T) or None. mask==0 indicates positions to mask out.
        """
        B, T, _ = x.size()

        # Linear projections
        Q = self.wq(x)  # (B, T, d_model)
        K = self.wk(x)
        V = self.wv(x)

        # Split heads
        # reshape to (B, T, num_heads, d_k) then transpose to (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product
        # scores: (B, h, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask should be broadcastable to (B, h, T, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, h, T, T)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (B, h, T, d_k)

        # combine heads
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        out = self.wo(context)  # (B, T, d_model)
        return out, attn

class PositionwiseFFN(nn.Module):
    """
    Feed-forward network applied position-wise (independently per time step).
    FFN(x) = W2 (GELU(W1 x + b1)) + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.0, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'gelu':
            self.act = F.gelu
        else:
            raise ValueError("Unsupported activation")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderBlock(nn.Module):
    """
    Single transformer encoder block: MultiHeadSelfAttention -> Add&LayerNorm -> FFN -> Add&LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, activation='relu'):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout, activation=activation)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention sublayer
        attn_out, attn_weights = self.self_attn(x, mask=mask)  # (B, T, d_model)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # FFN sublayer
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)
        return x, attn_weights

class TransformerEncoder(nn.Module):
    """
    A small Transformer Encoder stack.
    """
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, dropout=0.1, learned_pe=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, learned=learned_pe)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        # optional output projection for e.g. LM / classification
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        x: (B, T) token ids (long)
        mask: optional mask for attention (B, 1, 1, T) or (B, 1, T, T)
        returns logits (B, T, vocab_size) and list of attn weights per layer
        """
        B, T = x.shape
        x = self.token_emb(x) * self.scale  # (B, T, d_model)
        x = x + self.pos_enc(x)  # broadcasting positional encoding
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attn_list.append(attn)
        x = self.ln(x)
        logits = self.output_layer(x)
        return logits, attn_list
