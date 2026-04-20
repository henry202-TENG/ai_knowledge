# Mamba / SSM (State Space Model)

## 1. 什麼是？
Mamba 是一種新型的模型架構，基於 State Space Model（狀態空間模型），旨在替代 Transformer 的 Attention 機制，實現更高效的長序列處理。

## 2. 為什麼重要？
- **線性推理複雜度**：O(n) vs Transformer 的 O(n²)
- **長上下文優勢**：在超長序列上效率更高
- **選擇性狀態空間**：能夠根據輸入動態篩選資訊
- **硬體友好**：避免了 Attention 的記憶體瓶頸

## 3. 核心原理

### Transformer 的瓶頸
```
輸入序列長度: n
維度: d

Attention 計算:
- 時間複雜度: O(n² × d)
- 記憶體: O(n²)

問題：
- 序列越長，計算量暴漲
- KV Cache 記憶體巨大
```

### 狀態空間模型 (SSM)

#### 連續時間視角
```
狀態方程:
  h'(t) = A × h(t) + B × x(t)
  y(t) = C × h(t) + D × x(t)

其中:
- h(t) = 隱藏狀態
- x(t) = 輸入
- A, B, C, D = 可學習矩陣
```

#### 離散化 (Zero-Order Hold)
```
將連續方程轉為離散:
  hₜ = Ā × hₜ₋₁ + B̄ × xₜ
  yₜ = C × hₜ

可用卷積實現高效計算:
  y = Conv(K, x)
```

### Mamba 的創新

#### 1. 選擇性 SSM
```
傳統 SSM: 所有輸入同等處理
Mamba: 根據輸入動態決定處理方式

s(z) = σ(Linear(z))  # 選擇性投影
```

#### 2. 並行掃描 (Parallel Scan)
```
計算hidden states時:
- 傳統: 順序依賴，O(n)
- Mamba: 並行掃描，O(log n)
```

#### 3. 硬體優化
```
- 使用 FlashAttention 類似的技術
- 融合核心運算
- 減少記憶體訪問
```

### 與 Transformer 比較
| 特性 | Transformer | Mamba/SSM |
|------|-------------|-----------|
| 時間複雜度 | O(n²) | O(n) |
| 空間複雜度 | O(n²) | O(n) |
| 推理速度 | 慢（長序列） | 快 |
| 訓練速度 | 快 | 中等 |
| 上下文長度 | 受限 | 更長 |
| 並行訓練 | 易 | 較難 |

## 4. 知名 SSM/Mamba 模型

| 模型 | 開發者 | 特色 |
|------|--------|------|
| **Mamba-2** | Stanford | 結構化狀態空間 |
| **Jamba** | AI21 Labs | Mamba + Transformer 混合 |
| **Grok-1** | xAI | 傳言使用 MoE + SSM |
| **Griffin** | Google | 混合 RNN + Attention |

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **Attention** | SSM 要替代的技術 |
| **RNN/LSTM** | SSM 的理論基礎 |
| **S4/S4D** | Mamba 的前身 |
| **Long Context** | Mamba 的主要應用場景 |

## 6. 延伸閱讀
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)
- [SSM 基礎講解](https://arxiv.org/abs/2208.04933)

---

*待補充...*