# KV Cache

## 1. 什麼是？
Key-Value Cache (KV Cache) 是一種推論優化技術，、快取已計算過的 Key 和 Value 矩陣，避免重複計算。

## 2. 為什麼重要？
- **大幅降低延遲**：避免對已處理的 tokens 重複計算 Attention
- **節省計算資源**：Transformer 推理的主要瓶頸是 Attention 計算
- **支援更長上下文**：是長上下文推理的基礎技術

## 3. 核心原理

### Transformer 推理流程
```
輸入: "The cat sat on the"
輸出: "mat"

第一步：處理 "The"
- 計算 Q, K, V
- Self-Attention
- 輸出 "cat"

第二步：處理 "cat"
- 重新計算 ALL tokens 的 K, V ← 浪費！
- 計算新的 Q
- Self-Attention
- 輸出 "sat"
```

### KV Cache 優化
```
使用 KV Cache 後：

第一步：處理 "The"
- 計算 Q₀, K₀, V₀
- Cache K₀, V₀

第二步：處理 "cat"
- 只計算新的 Q₁
- 從 Cache 取 K₀, V₀
- 計算 Attention
- Cache K₁, V₁
```

**公式比較**：
- 無 KV Cache：O(n²d) per token
- 有 KV Cache：O(nd) per token（n 為序列長度，d 為維度）

## 4. 實現方案

### PagedAttention (vLLM)
- 分頁管理 KV Cache
- 支援並發請求
- 減少記憶體碎片

### FlashDecoding
- 將 KV Cache 分塊載入
- 支援更長序列

### KV Cache 量化
- FP8 KV Cache
- 4-bit 量化
- 壓縮記憶體使用

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **Speculative Decoding** | 依賴 KV Cache 加速驗證 |
| **Context Window** | 需要更大的 KV Cache |
| **PagedAttention** | KV Cache 的記憶體管理方案 |
| **Flash Attention** | 與 KV Cache 互補的計算優化 |

## 6. 延伸閱讀
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [FlashDecoding Paper](https://arxiv.org/abs://2311.06683)
- [LLaMA2 Inference Optimization](https://ai.meta.com/research/publications/llama-2/)

---

*待補充...*