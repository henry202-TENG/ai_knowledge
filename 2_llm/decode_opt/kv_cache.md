# KV Cache

推論優化技術，快取已計算過的 Key 和 Value 矩陣，避免重複計算，顯著降低延遲。

---

## 1. 什麼是？

### 簡單範例

```
處理序列 "The cat sat on the mat"

無 KV Cache:
  Step 1: 計算 "The" 的 K, V → Cache 遺失
  Step 2: 重新計算 "The" + "cat" 的 K, V  ← 浪費！
  Step 3: 重新計算 "The" + "cat" + "sat" 的 K, V  ← 浪費！
  ...

有 KV Cache:
  Step 1: 計算 K₀, V₀ → Cache 儲存
  Step 2: 只計算新的 K₁, V₁ → Cache 新增
  Step 3: 只計算新的 K₂, V₂ → Cache 新增
  ...
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **大幅降低延遲** | 避免對已處理 tokens 重複計算 Attention |
| **節省計算資源** | Transformer 推理的主要瓶頸是 Attention 計算 |
| **支援更長上下文** | 是長上下文推理的基礎技術 |
| **支撐其他優化** | Speculative Decoding、FlashDecoding 的基礎 |

---

## 3. 核心原理

### Prefill 與 Decode 階段

```
┌─────────────────────────────────────────────────────────────────┐
│  Prefill 階段                                                   │
│  處理輸入 prompt                                                │
│  - 計算所有輸入 tokens 的 Q, K, V                                │
│  - 計算 Attention                                              │
│  - 快取所有 K, V                                               │
│  - 輸出第一個 token                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Decode 階段                                                    │
│  自迴歸生成新 token                                             │
│  - 使用快取的 K, V                                             │
│  - 只計算新 token 的 Q                                          │
│  - 計算 Attention → 輸出新 token                               │
│  - 快取新的 K, V                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 複雜度分析

| 階段 | 無 KV Cache | 有 KV Cache |
|------|-------------|-------------|
| **每個 Token** | O(n² × d) | O(n × d) |
| **n=1024, d=128** | 134M 運算 | 131K 運算 |
| **節約** | - | ~1000x |

> n = 序列長度，d = 隱藏維度

---

## 4. 實現方案

### PagedAttention (vLLM)

```python
# 傳統 KV Cache: 連續記憶體
# [The][,][cat][,][sat][,][on][,][the][,][mat]

# PagedAttention: 分頁管理
# Page 0: [The][,][cat][,][sat]
# Page 1: [on][,][the][,][mat]
# Page 2: [...] (新 page)
```

**優點**：
- 減少記憶體碎片
- 支援並發請求共享 Page
- 支援 Swapping (磁碟)

```python
# vLLM 使用範例
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

# 自動使用 PagedAttention
outputs = llm.generate(prompts, sampling_params)
```

### FlashDecoding

```
問題：KV Cache 太長時，單次 Attention 計算量大

解決：分塊載入 + 最終聚合

Step 1: 將 KV Cache 分成 N 塊
Step 2: 每塊與 Query 計算 partial attention
Step 3: 聚合所有 partial results
```

```python
def flash_decoding(q, kv_cache, block_size=512):
    num_blocks = kv_cache.num_blocks
    outputs = []

    for i in range(num_blocks):
        # 分塊載入
        kv_block = kv_cache.get_block(i)
        # 計算 partial attention
        partial = attention(q, kv_block)
        outputs.append(partial)

    # 聚合
    return sum(outputs)
```

### KV Cache 量化

| 方法 | 精度 | 壓縮比 | 效能影響 |
|------|------|--------|----------|
| **FP8** | 8-bit | 4x | 最小 |
| **INT8** | 8-bit | 4x | 低 |
| **INT4** | 4-bit | 8x | 中等 |
| **Binary** | 1-bit | 32x | 高 |

---

## 5. 挑戰與解決方案

### 挑戰 1：記憶體佔用

**問題**：長上下文需要大量 KV Cache

**解決方案**：
- PagedAttention 減少碎片
- KV Cache 量化
- 蒸餾壓縮

### 挑戰 2：Prefill-Decode 效率不均

**問題**：Prefill 很慢，Decode 很快，體驗不均

**解決方案**：
- Split Prefill：分段處理
- Chunked Prefill：小塊處理
- 持續批處理

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Speculative Decoding** | 依賴 KV Cache 加速驗證 |
| **Context Window** | 需要更大的 KV Cache |
| **PagedAttention** | KV Cache 的記憶體管理方案 |
| **Flash Attention** | 與 KV Cache 互補的計算優化 |

---

## 延伸閱讀

- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [FlashDecoding Paper](https://arxiv.org/abs/2311.06683)
- [LLaMA2 Inference Optimization](https://ai.meta.com/research/publications/llama-2/)