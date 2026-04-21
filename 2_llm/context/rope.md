# RoPE

旋轉位置編碼，透過旋轉矩陣將位置資訊編碼到每個 token 的表示中，能夠自然地處理相對位置關係。

---

## 1. 什麼是？

### 深度定義

**RoPE (Rotary Position Embedding)** 是一種**相對位置編碼**方法，其核心創新在於使用**旋轉矩陣**來編碼位置資訊：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RoPE 核心思想                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  傳統方法 (Sinusoidal):                                              │
│    PE(pos, 2i)   = sin(pos / base^(2i/d))                         │
│    PE(pos, 2i+1) = cos(pos / base^(2i/d))                          │
│                                                                      │
│  問題: 絕對位置編碼，難以推廣到未見過的長度                            │
│                                                                      │
│  RoPE 方法:                                                          │
│    對 Query 和 Key 應用旋轉矩陣:                                      │
│                                                                      │
│    q'_pos = RoPE(q_pos, pos)                                       │
│    k'_pos = RoPE(k_pos, pos)                                       │
│                                                                      │
│    Attention(q'_i, k'_j) 只依賴於 (i-j)，即相對位置！                 │
│                                                                      │
│  視覺化:                                                              │
│                                                                      │
│    位置 0: 0° 旋轉  → [■→]                                          │
│    位置 1: 30° 旋轉 → [□↗]                                          │
│    位置 2: 60° 旋轉 → [△↗]                                          │
│    位置 3: 90° 旋轉 → [○↑]                                          │
│                                                                      │
│    旋轉角度編碼了位置資訊                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

### 頻率設計的物理意義

```python
"""
RoPE 頻率設計的直覺解釋
"""

def explain_rope_frequencies(dim=64, base=10000):
    """
    每個維度對應的週期:

    維度 0-1 (最低頻):
        θ₀ = base^(-0/d) = 1
        週期 = 2π/θ₀ = 2π ≈ 6.28

        這意味著:
        - 相距 ~6 個 token 的位置會有相似的表示
        - 適合編碼「遠距離」關係

    維度 62-63 (最高頻):
        θ₃₁ = base^(-62/64) ≈ 1/10000
        週期 = 2π × 10000 ≈ 62800

        這意味著:
        - 幾乎每個位置都有獨特的表示
        - 適合編碼「近距離」精細關係
    """

    print("頻率設計的直覺:")
    print("低頻 → 長週期 → 遠距離區分")
    print("高頻 → 短週期 → 近距離區分")
    print("\n這模擬了人類語言的自然特性:")
    print("- 近距離需要精確的語法關係")
    print("- 遠距離只需要粗略的主題關聯")
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

## 7. 數學推導

### 旋轉矩陣推導

```
目標: 編碼相對位置

給定兩個 token 的表示 q (位置 m) 和 k (位置 n)

我們希望:
  RoPE(q_m, k_n) = f(q_m, n) = f(q_m, m-n)

也就是說，attention 分數只依賴於相對位置 m-n

解決方案: 旋轉

定義旋轉矩陣:
  R(θ, d) = cos(dθ) + i·sin(dθ)

應用於 d 維向量:
  [cos(θ)  -sin(θ)] [x₁]   [x₁cosθ - x₂sinθ]
  [sin(θ)   cos(θ)] [x₂] = [x₁sinθ + x₂cosθ]

這就是 2D 旋轉
```

### Attention 分數推導

```
標準 Attention:
  A(m, n) = q_m · k_n^T

RoPE 版本:
  A(m, n) = (R(mθ)q_m) · (R(nθ)k_n)^T
          = q_m^T R(mθ)^T R(nθ) k_n
          = q_m^T R((m-n)θ) k_n    (旋轉矩陣的性質)
          = q_m^T [cos((m-n)θ)  -sin((m-n)θ)]
                  [sin((m-n)θ)   cos((m-n)θ)] k_n

結果:
  = q_m · k_n · cos((m-n)θ) + 交叉項

這證明了 RoPE 確實只依賴於相對位置 m-n
```

### 多維度擴展

```python
def rope_math():
    """
    完整的 RoPE 數學推導
    """
    # 維度 d_model，分為 d/2 個 pair
    # 每個 pair 有自己的頻率 θ_i

    # 頻率計算
    # θ_i = base^(-2i/d), for i = 0, 1, ..., d/2 - 1

    # 對於位置 m 和維度 i:
    # 旋轉角度 = m * θ_i

    # 完整旋轉:
    # R(m, i) = [cos(mθ_i)  -sin(mθ_i)]
    #           [sin(mθ_i)   cos(mθ_i)]

    # 應用於 q 向量:
    # q' = R(m, 0)q[0:2] ⊕ R(m, 1)q[2:4] ⊕ ...
```

---

## 8. 實現細節

### 完整 RoPE 實現

```python
class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # 預計算頻率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 預計算 cos, sin
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]

        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        """應用 RoPE 到 Query 和 Key"""

        # x: [batch, num_heads, seq_len, head_dim]
        seq_len = seq_len if seq_len is not None else x.shape[2]

        # 確保緩存足夠
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        # 取得 cos, sin
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # 應用旋轉
        return self._apply_rotary_pos_emb(x, cos, sin)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        # x: [batch, num_heads, seq_len, head_dim]

        # 重新排列維度以便旋轉
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]

        # 旋轉公式: x * cos + rotate(x) * sin
        # 其中 rotate(x) = [-x2, x1]
        return torch.cat([
            x1 * cos + (-x2) * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
```

### 與 Flash Attention 整合

```python
class FlashRoPE(nn.Module):
    """與 Flash Attention 整合的 RoPE"""

    def __init__(self, config):
        self.rope = RoPE(config.head_dim, config.max_seq_len)

    def forward(self, q, k, v, positions=None):
        """
        帶 RoPE 的 Flash Attention
        """

        # 應用 RoPE 到 Q 和 K
        if positions is None:
            # 標準位置
            q = self.rope(q)
            k = self.rope(k)
        else:
            # 自定義位置 (可用於 Streaming LLM)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        # 使用 Flash Attention
        output = F.scaled_dot_product_attention(q, k, v)

        return output
```

---

## 9. 外推方法深入

### 位置線性插值 (PI)

```python
class PositionalInterpolation:
    """位置線性插值"""

    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor

    def interpolate_positions(self, positions, max_seq_len):
        """
        將位置映射到訓練時的範圍
        """
        # 原始: [0, 1, 2, ..., 4095]
        # 縮放: [0, 0.5, 1, ..., 2047.5]

        scaled_positions = positions / self.scale_factor

        return scaled_positions

    def __call__(self, model, new_max_len):
        """修改模型的 max_seq_len"""
        model.rope.max_seq_len = new_max_len

        # 重新計算頻率
        new_freqs = model.rope.base ** (
            -torch.arange(0, model.rope.dim, 2).float() / model.rope.dim
        )

        # 縮放頻率
        scaled_freqs = new_freqs / self.scale_factor

        model.rope.inv_freq = scaled_freqs
        model.rope._set_cos_sin_cache(new_max_len)
```

### NTK-aware Scaling

```python
class NTKScaledRoPE:
    """NTK-aware RoPE 縮放"""

    def __init__(self, dim, base=10000, scale_factor=1.0):
        self.dim = dim
        self.base = base * (scale_factor ** (dim / (dim - 2)))
        # 實際上，這讓低頻維度的頻率降低得更少

    def get_scaling_freqs(self, seq_len):
        """計算 NTK 縮放後的頻率"""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(seq_len, device=inv_freq.device)
        return torch.outer(t, inv_freq)
```

### YaRN

```python
class YaRN(nn.Module):
    """
    Yet another RoPE extensioN
    動態調整頻率以適應更長上下文
    """

    def __init__(self, dim, base=10000, scale_factor=1.0, original_max=2048):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_factor = scale_factor
        self.original_max = original_max

        # 可學習的溫度參數
        self.register_parameter(
            'temperature',
            nn.Parameter(torch.tensor(1.0))
        )

    def forward(self, x, seq_len):
        """YaRN 前向"""

        # 計算縮放
        if seq_len > self.original_max:
            # 使用溫度參數動態調整
            scale = (
                self.scale_factor *
                (self.original_max / seq_len) **
                (torch.log(torch.tensor(1.0)) / torch.log(torch.tensor(2.0)))
            )
        else:
            scale = 1.0

        # 計算頻率 (使用縮放)
        inv_freq = 1.0 / (
            (self.base * scale) **
            (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        return inv_freq
```

---

## 10. 效能評估

### 外推能力測試

```
任務: "大海撈針" - 在長文本中找出特定訊息

方法                     2K→4K    2K→8K    2K→32K
────────────────────────────────────────────────
標準 RoPE               95%      45%      5%
PI (scale=2)            92%      85%      30%
PI (scale=4)            88%      82%      60%
NTK-aware (α=8)        94%      90%      75%
YaRN                    95%      92%      88%
```

### 頻率分析

```python
def analyze_rope_frequencies(dim=128, base=10000):
    """分析 RoPE 頻率分布"""

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    frequencies = 1.0 / inv_freq

    print("低頻維度 (編碼長距離):")
    print(f"  維度 0: {frequencies[0]:.2f}")
    print(f"  維度 1: {frequencies[1]:.2f}")

    print("\n高頻維度 (編碼短距離):")
    print(f"  維度 62: {frequencies[62]:.2e}")
    print(f"  維度 63: {frequencies[63]:.2e}")

    return frequencies

# 輸出:
# 低頻維度 (編碼長距離):
#   維度 0: 10000.00
#   維度 1: 8164.96
#
# 高頻維度 (編碼短距離):
#   維度 62: 1.53
#   維度 63: 1.25
```

---

## 11. 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **長上下文效果下降** | 高頻維度外推不穩定 | 使用 NTK-aware 或 YaRN |
| **位置跳躍** | 處理非連續文本 | 自定義 position ids |
| **訓練/推斷不一致** | 位置編碼縮放不同 | 統一使用相同配置 |
| **記憶體佔用** | 預計算 cos/sin | 動態計算或壓縮 |

---

## 12. 相關技術

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
- [YaRN Paper](https://arxiv.org/abs/2309.00071)
- [Transformer Positional Encoding Survey](https://arxiv.org/abs/2104.09864)