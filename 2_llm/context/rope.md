# RoPE

旋轉位置編碼，透過旋轉矩陣將位置資訊編碼到每個 token 的表示中，能夠自然地處理相對位置關係。

---

## 1. 什麼是？

### 簡單範例

```
訓練長度: 2K tokens
測試長度: 100K tokens

傳統位置編碼:
  - 看不見的位置 → 性能下降

RoPE:
  - 使用旋轉矩陣編碼
  - 任意長度都能推廣
  - 相對位置自然編碼
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **相對位置編碼** | 直接建模 token 之間的相對距離 |
| **可擴展到更長上下文** | 是長上下文模型的關鍵技術 |
| **無需額外學習** | 旋轉矩陣是確定的，不需要訓練 |
| **主流採用** | LLaMA, Falcon, Pawn 等模型都使用 RoPE |

---

## 3. 核心原理

### RoPE 核心思想

```python
def rotate_half(x):
    """旋轉半部分"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """應用 RoPE"""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
```

### 2D RoPE（簡化說明）

```
對於每對 token (位置 i, j):

RoPE(Q_i, K_j) ∝ cos((i-j)θ)

這意味著：
- 只依賴相對位置 (i-j)
- 不需要知道絕對位置
- 自然推廣到未見過的長度
```

### 多維度 RoPE (LLaMA 採用)

```python
# 頻率計算
def get_rope_freqs(dim, base=10000, seq_len=None):
    # θ_i = base^(-2i/d), i = 0, 1, ..., d/2-1
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    if seq_len:
        freqs = freqs[:seq_len]
    return freqs
```

```
維度分配:
  維度 0-1: 低頻 (θ₀) → 編碼遠距離位置
  維度 2-3: 較低頻 (θ₁) → 
  ...
  維度 d-2~d-1: 高頻 (θ_{d/2-1}) → 編碼近距離位置
```

### 為何需要不同頻率？

```
低頻維度: 編碼粗粒度位置 (遠距離)
高頻維度: 編碼細粒度位置 (近距離)

這樣設計的好處:
  - 短距離精確區分
  - 長距離穩定表達
```

---

## 4. 位置編碼外推

### 問題

```
訓練長度: 2048
測試長度: 4096

直接使用會導致:
  - 旋轉角超過訓練範圍
  - 性能下降 (extrapolation 問題)
```

### 解決方案

| 方法 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| **位置線性插值** | 將位置除以 scaling factor | 簡單 | 短 context 效果下降 |
| **NTK-aware Scaling** | 改變 RoPE 頻率 base | 保持長距離能力 | 需要調整 base |
| **YaRN** | 動態調整頻率 |效果好 | 需額外訓練 |
| **Hotpot** | 混合多個 RoPE | 靈活 | 複雜 |

### NTK-aware Scaling 詳解

```python
# 原始 RoPE: θ_i = base^(-2i/d)
# 縮放後: θ_i = (base * α)^(-2i/d)

# α > 1 讓頻率更低
# 低頻維度影響小，高頻維度影響大
# 結果: 長距離位置更穩定
```

---

## 5. RoPE vs 其他位置編碼

| 特性 | Sinusoidal | RoPE | ALiBi |
|------|------------|------|-------|
| 絕對/相對 | 絕對 | 相對 | 相對 |
| 可學習 | 可選 | 否 | 否 |
| 長上下文 | 一般 | 較好 | 較好 |
| 實現複雜度 | 低 | 中 | 中 |
| 主流採用 | BERT | LLaMA | BLOOM |
| 對話應用 | 普通 | 優秀 | 優秀 |

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Ring Attention** | 需要 RoPE 處理超長上下文 |
| **Context Window** | RoPE 決定了上下文長度 |
| **ALiBi** | 另一種位置編碼 |
| **Flash Attention** | 與 RoPE 兼容 |

---

## 延伸閱讀

- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [LLaMA RoPE 實現](https://github.com/meta-llama/llama)
- [位置編碼外推綜述](https://arxiv.org/abs/2309.00071)
- [NTK-aware Scaling](https://arxiv.org/abs/2309.00071)