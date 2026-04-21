# Transformer Evolution

Transformer 架構的演進與優化，從原始架構到各種變體。

---

## 1. 原始 Transformer

### 架構概述

```
原始 Transformer:
┌─────────────────────────────────────┐
│           Input Embedding           │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         Positional Encoding         │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         N x Encoder Layer           │
│  ┌─────────────────────────────┐   │
│  │   Multi-Head Attention      │   │
│  │   (Self-Attention)          │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │   Add & Norm                │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │   Feed Forward (FFN)        │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │   Add & Norm                │   │
│  └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         N x Decoder Layer           │
│  (類似結構 + Masked Self-Attn +      │
│   Encoder-Decoder Attention)         │
└─────────────────────────────────────┘
```

### 核心計算

```python
class TransformerBlock(nn.Module):
    """原始 Transformer 塊"""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        # Self Attention + Add & Norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
```

---

## 2. 注意力優化

### 多查詢注意力 (MQA)

```python
class MultiQueryAttention(nn.Module):
    """多查詢注意力 - 共享 K V"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # 獨立的 Q，共享的 K 和 V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)  # 共享
        self.W_v = nn.Linear(d_model, self.d_k)  # 共享
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Q 分割成多個頭
        Q = self.W_q(q).view(-1, self.num_heads, self.d_k)

        # K 和 V 不分割（共享）
        K = self.W_k(k)  # shape: (batch, d_k)
        V = self.W_v(v)

        # 廣播到每個頭
        Q = Q.view(-1, self.num_heads, 1, self.d_k)
        K = K.view(-1, 1, self.d_k).expand(-1, self.num_heads, -1, -1)
        V = V.view(-1, 1, self.d_k).expand(-1, self.num_heads, -1, -1)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # 合併頭
        output = output.contiguous().view(-1, self.d_model)

        return self.W_o(output)
```

### 分組查詢注意力 (GQA)

```python
class GroupedQueryAttention(nn.Module):
    """分組查詢注意力 - K V 組"""

    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, self.num_groups * self.d_k)
```

---

## 3. 位置編碼演進

### RoPE

```python
class RotaryPositionEmbedding(nn.Module):
    """旋轉位置編碼"""

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # 預計算頻率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        # 生成位置
        positions = torch.arange(seq_len, device=device).float()

        # 計算頻率
        freqs = torch.outer(positions, self.inv_freq)

        # 旋轉
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos(), emb.sin()

    def rotate_half(x):
        """旋轉一半"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin):
        """應用 RoPE"""
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k
```

---

## 4. 架構變體

### Encoder-Only (BERT)

```python
class BERTModel(nn.Module):
    """BERT 模型"""

    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        # 任務頭
        self.cls_head = nn.Linear(d_model, d_model)
        self.token_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # 嵌入
        x = self.embedding(x)
        x = self.position(x)

        # Transformer
        for layer in self.transformer:
            x = layer(x, mask)

        # CLS 和 Token 預測
        cls_output = self.cls_head(x[:, 0])
        token_logits = self.token_head(x)

        return cls_output, token_logits
```

### Decoder-Only (GPT)

```python
class GPTModel(nn.Module):
    """GPT 模型"""

    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)

        self.transformer = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, mask=None):
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = self.position(x)

        for layer in self.transformer:
            x = layer(x, mask)

        return self.lm_head(x)
```

---

## 5. 效率優化

### 稀疏注意力

```python
class SparseAttention(nn.Module):
    """稀疏注意力"""

    def __init__(self, d_model, num_heads, sparsity=0.9):
        super().__init__()
        self.sparsity = sparsity
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        # 局部窗口
        window_size = int(x.shape[1] * (1 - self.sparsity))

        # 只計算窗口內的注意力
        # ...

        return output
```

---

## 6. 架構比較

| 架構 | 特性 | 用途 |
|------|------|------|
| **Original** | 完整 Encoder-Decoder | 翻譯、Seq2Seq |
| **BERT** | Encoder-only, 雙向 | 理解任務 |
| **GPT** | Decoder-only, 自迴歸 | 生成任務 |
| **T5** | Encoder-Decoder | 統一框架 |
| **LLaMA** | Decoder, RoPE, SwiGLU | 高效生成 |

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **RoPE** | 位置編碼 |
| **Flash Attention** | 注意力優化 |
| **MoE** | 模型擴展 |

---

## 延伸閱讀

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)