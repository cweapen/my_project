import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Positional Encoding ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, learned=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        if learned:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, mean=0.0, std=0.02)
            self.learned = True
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.learned = False

    def forward(self, x):
        T = x.size(1)
        return self.pe[:, :T, :]


# ---------------- Multi-Head Attention ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Tq, _ = query.size()
        Tk = key.size(1)

        Q = self.wq(query).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(key).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(value).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.num_heads * self.d_k)
        out = self.wo(context)
        return out, attn


# ---------------- Feed Forward ----------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'gelu':
            self.act = F.gelu
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ---------------- Encoder Block ----------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x, attn_weights


# ---------------- Decoder Block ----------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.ln1(x + self.dropout(self_attn_out))
        cross_out, cross_weights = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.ln2(x + self.dropout(cross_out))
        ffn_out = self.ffn(x)
        x = self.ln3(x + self.dropout(ffn_out))
        return x, self_attn_weights, cross_weights


# ---------------- Encoder ----------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_emb(x) * math.sqrt(self.token_emb.embedding_dim)
        x = x + self.pos_enc(x)
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_list.append(attn)
        x = self.ln(x)
        logits = self.output_layer(x)
        return logits, attn_list, x  # x 是 encoder 输出


# ---------------- Decoder ----------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.token_emb(x) * math.sqrt(self.token_emb.embedding_dim)
        x = x + self.pos_enc(x)
        self_attn_list, cross_attn_list = [], []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, tgt_mask, memory_mask)
            self_attn_list.append(self_attn)
            cross_attn_list.append(cross_attn)
        x = self.ln(x)
        logits = self.output_layer(x)
        return logits, self_attn_list, cross_attn_list


# ---------------- Small Transformer (Encoder / Encoder-Decoder) ----------------
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512,
                 max_len=512, dropout=0.1, decoder=True):
        """
        参数 decoder:
            True  -> 使用 Encoder + Decoder
            False -> 仅使用 Encoder（用于消融实验）
        """
        super().__init__()
        self.decoder_enabled = decoder
        self.encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

        if decoder:
            self.decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        else:
            self.decoder = None  # 不创建解码器

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        _, _, enc_output = self.encoder(src, src_mask)

        # 如果不使用解码器，直接输出 encoder 的分类结果
        if not self.decoder_enabled:
            logits = self.encoder.output_layer(enc_output)
            return logits

        # 使用 encoder-decoder 结构
        logits, _, _ = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits
