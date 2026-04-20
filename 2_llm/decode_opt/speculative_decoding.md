# Speculative Decoding (投機解碼)

## 1. 什麼是？
Speculative Decoding 是一種推論加速技術，利用較小的「投機模型」快速生成多個 tokens，然後用大模型一次性驗證，大幅減少延遲。

## 2. 為什麼重要？
- **降低延遲**：可達 2-3 倍加速
- **保持質量**：輸出與標準解碼完全一致
- **實用性強**：已被多個推理框架採用

## 3. 核心原理

### 傳統自迴歸解碼的問題
```
輸入: "Hello"
輸出: " world"

Token-by-token 生成:
1. "H" → 計算 Attention → "e"
2. "He" → 計算 Attention → "l"
3. "Hel" → 計算 Attention → "l"
4. "Hell" → 計算 Attention → "o"

問題：每個 token 都需要完整 Attention 計算
     長序列時延遲線性增長
```

### Speculative Decoding 流程
```
步驟 1: 投機階段（小模型）
輸入: "The cat sat on"
投機模型連續生成 K 個 tokens: ["the", "mat", ".", "It", "was"]

步驟 2: 驗證階段（大模型）
一次輸入: "The cat sat on the mat. It was"
大模型驗證並修正

步驟 3: 採用結果
- 投機正確 → 直接採用
- 投機錯誤 → 從錯誤點重新開始
```

### 關鍵算法

#### 標準 Speculative Decoding
```
Algorithm:
1. 小模型生成 K 個候選 tokens
2. 大模型一次處理所有 K 個 tokens
3. 採用「接受」直到第一個拒絕位置
4. 從該位置繼續大模型解碼
```

#### 拒絕採樣
```
每個 token 的接受概率:
accept_prob = min(1, P_large(token) / P_small(token))

如果隨機數 < accept_prob → 接受
否則 → 拒絕並用大模型輸出替換
```

### 加速原理
```
標準: K 個 token 需要 K 次大模型推理
投機: K 個 token 只需要 1 次大模型推理

加速比 ≈ K（假設投機命中率 > 80%）
```

## 4. 變體和優化

| 方法 | 改進 |
|------|------|
| **Medusa** | 多個頭同時預測 |
| **Eagle** | 早exit + 特徵重用 |
| **Lookahead Decoding** | 自迴歸生成多個 n-gram |
| **Self-Speculative** | 用同一模型的不同層 |

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **KV Cache** | Speculative Decoding 的基礎 |
| **PagedAttention** | 管理投機階段的 KV Cache |
| **DistilBERT** | 投機模型的典型選擇 |
| **Early Exit** | 類似思想，提前輸出 |

## 6. 延伸閱讀
- [Speculative Decoding Paper](https://arxiv.org/abs/2302.01318)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

---

*待補充...*