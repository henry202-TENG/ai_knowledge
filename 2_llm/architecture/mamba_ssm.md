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

## 延伸閱讀

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)
- [SSM 基礎講解](https://arxiv.org/abs/2208.04933)
- [Mamba GitHub](https://github.com/state-spaces/mamba)