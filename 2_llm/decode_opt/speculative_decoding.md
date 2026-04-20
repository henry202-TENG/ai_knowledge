# Speculative Decoding

推論加速技術，利用較小的「投機模型」快速生成多個 tokens，然後用大模型一次性驗證，大幅減少延遲。

---

## 1. 什麼是？

### 簡單範例

```
生成 "The cat sat on the mat"

傳統 (大模型單獨):
  The → cat → sat → on → the → mat
  每個 token 1 次大模型推理
  = 6 次推理

投機解碼:
  投機階段 (小模型): 一次生成 ["the", "cat", "sat", "on", "the", "mat"]
  驗證階段 (大模型): 一次驗證全部 6 個 tokens
  = 2 次推理

→ 3x 加速
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **降低延遲** | 可達 2-3 倍加速 |
| **保持質量** | 輸出與標準解碼完全一致 |
| **實用性強** | 已被多個推理框架採用 |
| **通用性** | 可應用於任何自迴歸模型 |

---

## 3. 核心原理

### 標準 Speculative Decoding 流程

```
步驟 1: 投機階段（小模型）
┌──────────────────────────────────────────────┐
│  輸入: "The cat sat on"                       │
│  小模型連續生成 K 個 tokens:                   │
│  ["the", "mat", ".", "It", "was", "cute"]    │
└──────────────────────────────────────────────┘
                              ↓
步驟 2: 驗證階段（大模型）
┌──────────────────────────────────────────────┐
│  一次輸入: "The cat sat on the mat. It was cute"│
│  大模型驗證每個 position                        │
│  - position 0: "the" ✓                        │
│  - position 1: "mat" ✓                        │
│  - position 2: "." → "?" (修正)               │
│  - position 3: "It" ✗ (reject, 從這裡重新)    │
└──────────────────────────────────────────────┘
                              ↓
步驟 3: 採用結果
┌──────────────────────────────────────────────┐
│  - "the", "mat" → 採用                       │
│  - 從 position 2 重新用大模型解碼              │
└──────────────────────────────────────────────┘
```

### 拒絕採樣算法

```python
def speculative_verify(proposal_tokens, small_probs, large_probs):
    """驗證投機結果"""
    accepted = []
    for i, token in enumerate(proposal_tokens):
        # 計算接受概率
        p_small = small_probs[i, token]
        p_large = large_probs[i, token]

        accept_prob = min(1, p_large / p_small)

        if random.random() < accept_prob:
            accepted.append(token)
        else:
            # 拒絕，從大模型輸出
            new_token = sample_from_large()
            accepted.append(new_token)
            break

    return accepted
```

### 加速原理

```
標準: K 個 token → K 次大模型推理
投機: K 個 token → 1 次小模型 + 1 次大模型

加速比 ≈ K × 命中率 / (1 + K × 命中率)

例如 K=6, 命中率=80%:
  加速比 ≈ 6 × 0.8 / 1.8 ≈ 2.67x
```

---

## 4. 變體和優化

### Medusa

```
投機解碼: 1 個小模型 → 多個 tokens
Medusa: 多個「預測頭」→ 同時預測多個 tokens

每個 head 預測 future tokens:
  Head 1: 預測 t+1
  Head 2: 預測 t+2
  Head 3: 預測 t+3
  ...
```

### Eagle

```
投機解碼: 小模型生成完整 tokens
Eagle: 使用大模型的中間層特徵 + 早 Exit

改進:
  - 使用大模型最後幾層的特徵
  - 更準確的投機
  - 更少的參數
```

### Lookahead Decoding

```
不使用獨立小模型
使用 n-gram 投機:
  - 從已生成的 tokens 提取 n-gram
  - 假設未來會重複這些模式
  - 大模型驗證
```

### Self-Speculative

```
不使用小模型
使用同一模型的不同配置:
  - 使用較少 layers 的版本投機
  - 用完整版本驗證
```

| 方法 | 優點 | 缺點 |
|------|------|------|
| **標準 Speculative** | 簡單直接 | 需要小模型 |
| **Medusa** | 不需小模型 | 需要訓練預測頭 |
| **Eagle** | 效率高 | 需要特殊訓練 |
| **Lookahead** | 無需額外模型 | 依賴文本模式 |

---

## 5. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **KV Cache** | Speculative Decoding 的基礎 |
| **PagedAttention** | 管理投機階段的 KV Cache |
| **DistilBERT** | 投機模型的典型選擇 |
| **Early Exit** | 類似思想，提前輸出 |

---

## 延伸閱讀

- [Speculative Decoding Paper](https://arxiv.org/abs/2302.01318)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)