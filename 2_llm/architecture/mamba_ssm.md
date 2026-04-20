# Mamba / SSM

基於 State Space Model（狀態空間模型）的新型模型架構，旨在替代 Transformer 的 Attention 機制，實現更高效的長序列處理。

---

## 1. 什麼是？

### 簡單範例

```
處理 100K tokens 的序列：

Transformer:
  - 計算量: O(n²) = 100K² = 10¹⁰
  - 延遲: 數十秒
  - KV Cache: 數十 GB

Mamba/SSM:
  - 計算量: O(n) = 100K = 10⁵
  - 延遲: < 1 秒
  - 狀態空間: 數百 MB

→ 10-100x 加速
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **線性推理複雜度** | O(n) vs Transformer 的 O(n²) |
| **長上下文優勢** | 在超長序列上效率更高 |
| **選擇性狀態空間** | 能夠根據輸入動態篩選資訊 |
| **硬體友好** | 避免了 Attention 的 KV Cache 記憶體瓶頸 |

---

## 3. 核心原理

### Transformer 的瓶頸

```
輸入序列長度: n
維度: d
頭數: h

Attention 計算:
- 時間複雜度: O(n² × d)
- 記憶體: O(n²) (KV Cache)

問題：
- 序列越長，計算量二次方增長
- KV Cache 記憶體巨大，無法處理超長上下文
```

### 狀態空間模型 (SSM)

#### 連續時間視角

```
狀態方程:
  h'(t) = A × h(t) + B × x(t)
  y(t) = C × h(t) + D × x(t)

其中:
- h(t) = 隱藏狀態 (N 維)
- x(t) = 輸入信號
- y(t) = 輸出信號
- A, B, C, D = 可學習矩陣 (N×N, N×1, 1×N, 1×1)
```

#### 離散化 (Zero-Order Hold)

```
將連續方程轉為離散形式 (時間步長 Δ):

hₜ = Ā × hₜ₋₁ + B̄ × xₜ
yₜ = C × hₜ

其中:
- Ā = exp(AΔ)
- B̄ = (exp(AΔ) - I) × A⁻¹ × B × Δ

關鍵：可用卷積實現高效計算！
```

#### 卷積視角

```
SSM 可以轉換為離散卷積：

y = Conv(K, x)

卷積核 K:
K = [C·B̄, C·Ā·B̄, C·Ā²·B̄, ..., C·Āⁿ⁻¹·B̄]
```

### SSM 公式推導

```python
# 簡化的 SSM 前向傳播
def ssm_forward(x, A, B, C):
    N = A.shape[0]  # 狀態維度
    T = len(x)

    h = torch.zeros(N)  # 初始狀態
    outputs = []

    for t in range(T):
        # 更新狀態
        h = A @ h + B @ x[t]
        # 產生輸出
        y = C @ h
        outputs.append(y)

    return outputs
```

### Mamba 的創新

#### 1. 選擇性 SSM (Selective State Space)

```
傳統 SSM: 所有輸入同等處理
  → 無法根據輸入動態調整

Mamba: 根據輸入動態決定處理方式
  s(z) = σ(Linear(z))  # 選擇性投影
  → 可以篩選重要的 KV
```

```python
# Mamba 的選擇性機制
def selective_scan(x, ssm_params):
    # 動態生成 SSM 參數
    B = ssm_params.B * s(x)  # 選擇性門控
    C = ssm_params.C * s(x)

    # 標準 SSM 計算
    return ssm(x, A, B, C)
```

#### 2. 並行掃描 (Parallel Scan)

```
計算 hidden states 時的依賴問題：
  h₀ = A₀h₀₋₁ + B₀x₀
  h₁ = A₁h₀ + B₁x₁  ← 依賴 h₀
  h₂ = A₂h₁ + B₂x₂  ← 依賴 h₁
  ...

傳統: 順序計算，O(n)
並行掃描: 分而治之，O(log n)
```

#### 3. 硬體優化

```
- 融合核心運算 (Kernel Fusion)
- 減少記憶體訪問 (FlashAttention 思想)
- 使用 GPU Tensor Core 加速
- 分塊計算避免記憶體瓶頸
```

### 與 Transformer 比較

| 特性 | Transformer | Mamba/SSM |
|------|-------------|-----------|
| 時間複雜度 | O(n²) | O(n) |
| 空間複雜度 | O(n²) | O(n) |
| 推理速度 | 慢（長序列） | 快 |
| 訓練速度 | 快 | 中等 |
| 上下文長度 | 受限（記憶體） | 更長 |
| 並行訓練 | 易 | 較難 |
| 推理記憶體 | 高（KV Cache） | 低 |

---

## 4. 知名 SSM/Mamba 模型

| 模型 | 開發者 | 特色 | 參數 |
|------|--------|------|------|
| **Mamba** | Stanford | 選擇性 SSM | 2.7B-7B |
| **Mamba-2** | Stanford | 結構化狀態空間 | 2.7B-7B |
| **Jamba** | AI21 Labs | Mamba + Transformer 混合 | 12B |
| **Grok-1** | xAI | 傳言使用 MoE + SSM | 314B |
| **Griffin** | Google | 混合 RNN + Attention | - |

### 混合架構趨勢

```
純 Transformer: GPT-4, Claude, Gemini

混合架構:
  - Mamba + Transformer: Jamba
  - MoE + SSM: Grok-1
  - RNN + Attention: Griffin

→ 結合兩者優點
```

---

## 5. 挑戰與解決方案

### 挑戰 1：表達能力

**問題**：SSM 在某些任務上不如 Transformer

**解決方案**：
- 與 Transformer 混合
- 增加狀態維度 N
- 選擇性機制

### 挑戰 2：並行訓練

**問題**：SSM 的順序依賴限制了訓練並行度

**解決方案**：
- 並行掃描算法
- 近似方法
- 與 Transformer 分層使用

### 挑戰 3：實現複雜度

**問題**：高效實現困難

**解決方案**：
- 使用 CUDA Kernel 融合
- 借助 FlashAttention 經驗
- 開源實現（Mamba 官方）

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Attention** | SSM 要替代的技術 |
| **RNN/LSTM** | SSM 的理論基礎 |
| **S4/S4D** | Mamba 的前身 |
| **Long Context** | Mamba 的主要應用場景 |
| **KV Cache** | SSM 不需要大量的 KV Cache |

---

## 7. 離散化方法詳解

### Zero-Order Hold (ZOH)

```
最常用的離散化方法，假設輸入在時間間隔內保持常數：

連續: x(t) 在 [tₖ, tₖ₊₁] 區間內為常數 xₖ

離散化公式:
  Ā = exp(AΔ)
  B̄ = A⁻¹(Ā - I)B  (當 A 可逆)
  B̄ ≈ Δ × B         (當 Δ 很小的近似)

其中 Δ 為時間步長，可為固定值或可學習參數
```

### 雙線性變換 (Tustin Method)

```
將連續系統轉為離散的另一種方法：

s = (2/Δ) × (z - 1)/(z + 1)

優點:
  - 保持穩定性
  - 頻率響應更好

缺點:
  - 計算更複雜
```

### 離散化實現

```python
def discretize_ss(A, B, delta):
    """
    Zero-Order Hold 離散化

    A: [N, N] 連續系統矩陣
    B: [N, D] 輸入矩陣
    delta: 時間步長

    返回: Ā, B̄
    """
    N = A.shape[0]

    # 使用矩陣指數實現 Ā = exp(AΔ)
    A_discrete = matrix_exp(A * delta)

    # B̄ = A⁻¹(Ā - I)B
    # 使用近似: B̄ ≈ Δ × B (當 Δ 小時)
    if delta < 0.1:
        B_discrete = B * delta
    else:
        A_inv = torch.inverse(A)
        B_discrete = A_inv @ (A_discrete - torch.eye(N)) @ B

    return A_discrete, B_discrete
```

---

## 8. 卷積視角深入

### SSM → 卷積的數學推導

```
離散 SSM:
  hₜ = Āhₜ₋₁ + B̄xₜ
  yₜ = Chₜ

展開:
  h₀ = B̄x₀
  h₁ = ĀB̄x₀ + B̄x₁
  h₂ = Ā²B̄x₀ + ĀB̄x₁ + B̄x₂
  ...

輸出:
  y₀ = C B̄x₀
  y₁ = CĀB̄x₀ + C B̄x₁
  y₂ = CĀ²B̄x₀ + CĀB̄x₁ + C B̄x₂

卷積核:
  K = [C B̄, CĀB̄, CĀ²B̄, ..., CĀⁿ⁻¹B̄]

y = x * K
```

### 卷積計算實現

```python
def ssm_as_convolution(x, A, B, C):
    """將 SSM 轉為卷積計算"""

    T = len(x)
    N = A.shape[0]

    # 計算卷積核
    K = []
    h = torch.zeros(N)
    for t in range(T):
        K.append(C @ h)
        h = A @ h + B @ x[t] if t < len(x) else A @ h

    K = torch.stack(K)

    # 使用卷積
    y = F.conv1d(
        x.unsqueeze(0),
        K.unsqueeze(0),
        padding=T-1
    ).squeeze(0)

    return y
```

### 卷積視角的優勢

| 方面 | 遞迴形式 | 卷積形式 |
|------|----------|----------|
| **訓練** | 需 Parallel Scan | 高度並行 |
| **推理** | 狀態遞進 | 可用 KV Cache |
| **記憶體** | O(N) | O(N×T) |
| **實現** | 複雜 | 簡單 (PyTorch Conv1d) |

---

## 9. 選擇性 SSM 深入

### 選擇性門控機制

```python
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, state_dim=16):
        self.state_dim = state_dim
        self.d_model = d_model

        # 投影矩陣
        self.x_proj = nn.Linear(d_model, state_dim * 2)
        self.dt_proj = nn.Linear(state_dim, d_model)

        # 可學習的 A 矩陣 (HiPPO 初始化)
        self.A = nn.Parameter(
            torch.randn(state_dim, state_dim)
        )

    def forward(self, x):
        """
        x: [batch, seq, d_model]
        """
        # 生成選擇性投影
        s = self.x_proj(x)  # [B, T, 2N]
        B, C = s.chunk(2, dim=-1)

        # Sigmoid 門控
        B = F.sigmoid(B)
        C = F.sigmoid(C)

        # 動態時間步長
        dt = F.softplus(self.dt_proj(x))

        # 標準 SSM 計算 (使用選擇性參數)
        return ssm_selective(x, self.A, B, C, dt)
```

### 選擇性機制的直觀理解

```
Transformer: 所有 token 都有相同的 KV
  →  無論輸入是什麼，都處理所有資訊

SSM: 動態選擇性
  →  根據輸入決定哪些資訊需要保留/遺忘
  
範例:
  用戶: "首先寫一個函數，然後優化它的效能"

  SSM 能夠:
  - 記住 "首先" 的指令
  - 動態更新 "然後" 的上下文
  - 在 "優化" 時回顧前面的函數內容
```

---

## 10. 硬體優化實作

### CUDA Kernel 融合

```python
# 原始實現 (多次 kernel 调用)
h = A @ h + B @ x  # Kernel 1: 矩陣乘法
h = F.relu(h)      # Kernel 2: 激活函數
y = C @ h          # Kernel 3: 矩陣乘法

# 融合後 (單次 kernel 调用)
y = fused_ssm_kernel(A, B, C, x)  # 一次完成
```

### FlashSSM 優化

```
借鑒 FlashAttention 的思想：

1. 分塊計算 (Tiling)
   - 將長序列分為多個 chunk
   - 每個 chunk 獨立計算

2. 減少記憶體訪問
   - 避免從 global memory 頻繁讀寫
   - 使用 shared memory

3. 融合核心運算
   - 將多個運算合併為一個 kernel
   - 減少記憶體傳輸
```

### 效能比較

```
處理 128K tokens:

Transformer (FlashAttention-2):
  - 延遲: ~30 秒
  - 記憶體: ~80 GB

Mamba (選擇性 SSM):
  - 延遲: ~2 秒
  - 記憶體: ~8 GB

→ 15x 加速，10x 記憶體節省
```

---

## 11. 與 RNN/LSTM 比較

### 數學對比

```
RNN: h_t = tanh(W_h h_{t-1} + W_x x_t)
     - 單一hidden state
     - 梯度消失問題
     - 難以處理長距離依賴

LSTM: h_t = f_t ⊙ h_{t-1} + i_t ⊙ tanh(W_h h_{t-1} + W_x x_t)
     - 門控機制
     - 緩解梯度消失
     - 但仍是 O(n) 順序計算

SSM: h_t = A h_{t-1} + B x_t
     - 連續狀態空間
     - 可轉為並行卷積
     - 更強的表達能力
```

### 表達能力比較

| 特性 | RNN | LSTM | SSM |
|------|-----|------|-----|
| 短距離依賴 | ✓ | ✓ | ✓ |
| 長距離依賴 | ✗ | △ | ✓ |
| 並行訓練 | ✗ | ✗ | ✓ |
| 選擇性 | ✗ | ✓ | ✓ |
| 狀態壓縮 | 低 | 中 | 高 |

### 為何 SSM 比 RNN 強

```
1. 連續表示:
   - RNN: 離散時間步
   - SSM: 連續時間模擬

2. 矩陣驅動:
   - RNN: 權重共享有限
   - SSM: 可學習的狀態轉換矩陣

3. 卷積表示:
   - SSM 可以轉為卷積
   - 利用 FFT 加速
```

---

## 12. 推理優化

### 推理時的狀態管理

```python
class SSMInference:
    def __init__(self, model):
        self.model = model
        self.current_state = None

    def step(self, token):
        """遞進式推理"""

        if self.current_state is None:
            # 初始化狀態
            self.current_state = torch.zeros(
                model.state_dim
            )

        # 更新狀態 (SSM遞迴)
        self.current_state = (
            model.A @ self.current_state +
            model.B @ token
        )

        # 產生輸出
        output = model.C @ self.current_state

        return output

    def generate(self, prompt, max_len):
        """自迴歸生成"""

        tokens = [prompt]

        for _ in range(max_len):
            # 計算下一個 token
            next_token = self.step(tokens[-1])

            # 採樣
            token = sample(next_token)
            tokens.append(token)

        return tokens
```

### KV Cache 優化對比

```
Transformer 推理:
  - 每個 token 需要完整 Attention
  - KV Cache 持續增長
  - 序列越長，記憶體越大

Mamba 推理:
  - 只需要狀態向量 (N 維)
  - 記憶體恆定
  - 不受序列長度影響
```

---

## 13. 基準測試

### 各項任務效能

| 任務 | Transformer | Mamba | 差異 |
|------|-------------|-------|------|
| **語言建模** | 基准 | 相近 | - |
| **大海撈針** | 中等 | 優秀 | Mamba 更強 |
| **程式碼生成** | 优秀 | 良好 | Transformer 更強 |
| **長上下文理解** | 受限 | 優秀 | Mamba 更強 |
| **推理任務** | 优秀 | 中等 | Transformer 更強 |

### 推理速度對比

```
序列長度: 8K tokens

LLaMA 7B (Transformer):
  - 首 token: 50 ms
  - 續 tokne: 15 ms
  - 記憶體: 14 GB

Mamba 2.8B:
  - 首 token: 30 ms
  - 續 tokne: 5 ms
  - 記憶體: 6 GB

→ 首 token 1.7x 快，續 token 3x 快
```

---

## 14. 相關主題

| 技術 | 關係 |
|------|------|
| **Attention** | SSM 要替代的技術 |
| **RNN/LSTM** | SSM 的理論基礎 |
| **S4/S4D** | Mamba 的前身 |
| **Long Context** | Mamba 的主要應用場景 |
| **KV Cache** | SSM 不需要大量的 KV Cache |

---

## 延伸閱讀

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)
- [SSM 基礎講解](https://arxiv.org/abs/2208.04933)
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [S4 Paper](https://arxiv.org/abs/2111.00396)
- [HiPPO Paper](https://arxiv.org/abs/2208.02446)