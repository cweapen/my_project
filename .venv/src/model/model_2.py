# model_2.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Positional Encoding ----------------
class PositionalEncoding(nn.Module):
    """支持绝对和学习位置编码"""
    def __init__(self, d_model, max_len=512, learned=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        if learned:
            pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(pe, mean=0.0, std=0.02)
            self.pe = pe
            self.learned = True
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.learned = False

    def forward(self, x):
        T = x.size(1)
        return self.pe[:, :T, :]

# ---------------- Relative Position Bias ----------------
class RelativePositionBias(nn.Module):
    """相对位置编码 bias"""
    def __init__(self, num_heads, max_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.relative_bias = nn.Embedding(2*max_len-1, num_heads)

    def forward(self, T):
        # 生成相对位置索引
        range_vec = torch.arange(T)
        distance_mat = range_vec[None, :] - range_vec[:, None]  # (T,T)
        distance_mat_clipped = distance_mat + self.max_len - 1  # shift index
        bias = self.relative_bias(distance_mat_clipped)  # (T,T,H)
        return bias.permute(2,0,1)  # (H,T,T)

# ---------------- Multi-Head Attention ----------------
class MultiHeadSelfAttention(nn.Module):
    """支持局部稀疏和相对位置 bias"""
    def __init__(self, d_model, num_heads, dropout=0.0, relative_position=False, local_window=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.relative_position = relative_position
        self.local_window = local_window

        # 一次性投影 Q,K,V
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if relative_position:
            self.rel_pos_bias = RelativePositionBias(num_heads)

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        qkv = self.qkv(x)  # (B,T,3*d_model)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1,2)  # (B,H,T,d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)  # (B,H,T,T)

        # 相对位置 bias
        if self.relative_position:
            bias = self.rel_pos_bias(T).to(x.device)  # (H,T,T)
            scores = scores + bias.unsqueeze(0)  # broadcast B

        # 局部稀疏注意力
        if self.local_window is not None:
            mask_local = torch.zeros_like(scores, dtype=torch.bool)
            for i in range(T):
                start = max(0, i-self.local_window)
                end = min(T, i+self.local_window+1)
                mask_local[:,:,i,start:end] = 1
            scores = scores.masked_fill(mask_local==0, float('-inf'))

        # 传统 mask
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1,2).contiguous().view(B,T,self.d_model)
        out = self.wo(context)
        return out, attn

# ---------------- Feed Forward ----------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = F.relu if activation=='relu' else F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------- Encoder Block ----------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, activation='relu', relative_position=False, local_window=None):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout, relative_position, local_window)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.self_attn(x, mask)
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x, attn_weights

# ---------------- Decoder Block ----------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, activation='relu', relative_position=False, local_window=None):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout, relative_position, local_window)
        self.enc_dec_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_out, self_attn_weights = self.self_attn(x, mask=tgt_mask)
        x = self.ln1(x + self.dropout(self_attn_out))
        enc_dec_out, enc_dec_weights = self.enc_dec_attn(x, mask=memory_mask)
        x = self.ln2(x + self.dropout(enc_dec_out))
        x = self.ln3(x + self.dropout(self.ffn(x)))
        return x, self_attn_weights, enc_dec_weights

# ---------------- Transformer Encoder ----------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, dropout=0.1, learned_pe=False, relative_position=False, local_window=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, learned_pe)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout, relative_position=relative_position, local_window=local_window)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        B, T = x.shape
        x = self.token_emb(x) * self.scale
        x = x + self.pos_enc(x)
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attn_list.append(attn)
        x = self.ln(x)
        logits = self.output_layer(x)
        return logits, attn_list

# ---------------- Transformer Decoder ----------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, dropout=0.1, learned_pe=False, relative_position=False, local_window=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, learned_pe)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout, relative_position=relative_position, local_window=local_window)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        B, T = x.shape
        x = self.token_emb(x) * self.scale
        x = x + self.pos_enc(x)
        self_attn_list, enc_dec_attn_list = [], []
        for layer in self.layers:
            x, self_attn, enc_dec_attn = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
            self_attn_list.append(self_attn)
            enc_dec_attn_list.append(enc_dec_attn)
        x = self.ln(x)
        logits = self.output_layer(x)
        return logits, self_attn_list, enc_dec_attn_list

# ---------------- Full Transformer ----------------
class SmallTransformer(nn.Module):
    """Encoder + Decoder 全量模型"""
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, dropout=0.1, relative_position=False, local_window=None):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, relative_position=relative_position, local_window=local_window)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, relative_position=relative_position, local_window=local_window)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output, enc_attn = self.encoder(src, mask=src_mask)
        logits, self_attn, enc_dec_attn = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits
