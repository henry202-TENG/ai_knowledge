# Ring Attention

讓 KV Cache 分布到多個 GPU 裝置上，實現百萬級 token 上下文的分散式長上下文注意力機制。

---

## 1. 什麼是？

### 簡單範例

```
傳統單 GPU:
  序列長度 100K tokens
  → KV Cache 記憶體爆了 → 無法處理

Ring Attention (4 GPU):
  GPU 0: tokens 0-25K
  GPU 1: tokens 25K-50K
  GPU 2: tokens 50K-75K
  GPU 3: tokens 75K-100K

透過環形通訊，每個 GPU 只持有 1/4 的 KV Cache
  → 100K tokens 正常處理 ✓
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **突破單 GPU 記憶體限制** | 標準 Attention 需要 O(n²) 記憶體，分散後每個 GPU 只 O(n/N) |
| **支援超長上下文** | 百萬級 token 上下文 |
| **線性通訊成本** | 相比傳統 All-to-All 分散式 Attention，通訊量大幅降低 |
| **與現有優化相容** | 可結合 PagedAttention、FlashAttention |

---

## 3. 核心原理

### 傳統問題

```
單 GPU 處理長序列：
- 序列長度 n
- KV Cache 記憶體：O(n) per layer
- Attention 計算：O(n²)
- 受限於 GPU 記憶體大小 (通常 80GB)

例如 n=100K, d=128, heads=32:
- KV Cache: 100K × 128 × 32 × 2 × 2 bytes ≈ 1.6 GB (per layer)
- 80GB GPU 只能處理 ~50 layers
```

### Ring Attention 解決方案

```
多 GPU 環形分布：

GPU 0: [tokens 0-1000] ─┐
GPU 1: [tokens 1001-2000] ─┼─→ 環形通訊
GPU 2: [tokens 2001-3000] ─┤
GPU 3: [tokens 3001-4000] ─┘

環形通訊流程：
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
  ↑                                   ↓
  └───────────────────────────────────┘
```

### 關鍵技術

#### 1. 分塊計算 (Chunked Computation)

```python
# 將 Attention 按 chunk 分開計算
def ring_attention(query, kv_chunks, num_gpus):
    results = []
    for i, kv_chunk in enumerate(kv_chunks):
        # 每個 GPU 計算自己持有的 KV chunk
        chunk_attention = attention(query, kv_chunk)
        results.append(chunk_attention)

        # 環形傳遞 query 到下一個 GPU
        query = send_to_next_gpu(query)

    return sum(results)
```

#### 2. 流水線傳遞 (Pipelined KV Transfer)

```
時間 →

t0: GPU 0 計算 chunk 0，同時接收 chunk 1
t1: GPU 0 計算 chunk 1，同時接收 chunk 2
t2: GPU 0 計算 chunk 2，同時接收 chunk 3
...

計算與通訊重疊 → 隱藏延遲
```

#### 3. FlashAttention 相容

```python
# Ring Attention 可以使用 FlashAttention 加速每個 chunk
def ring_flash_attention(q, kv_chunks):
    for i, kv in enumerate(kv_chunks):
        # 每個 chunk 使用 FlashAttention
        o_i = flash_attn_func(q, kv.k, kv.v)
        outputs.append(o_i)

        # 環形傳遞
        q = ring_send(q, next_rank)
```

### 通訊複雜度分析

| 方法 | 通訊量 | 延遲 |
|------|--------|------|
| All-to-All (傳統) | O(n) per GPU | 高 |
| Ring Attention | O(n/N) per step | 低 (流水線) |

---

## 4. 挑戰與解決方案

### 挑戰 1：負載不均衡

**問題**：某些 GPU 的 KV chunk 可能被更頻繁訪問

**解決方案**：
- 資料重排：將熱門 key 均勻分布
- 快取策略：本地快取熱門 KV

### 挑戰 2：通訊瓶頸

**問題**：跨節點 Ring Attention 頻寬不足

**解決方案**：
- NVLink + RDMA 組合
- 計算與通訊重疊
- 調整 chunk 大小

### 挑戰 3：同步开销

**問題**：每個 token 需要等待所有 GPU

**解決方案**：
- 非同步 Ring Attention
- 前瞻 (Lookahead) 技術

---

## 5. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **KV Cache** | Ring Attention 的基礎，每個 GPU 快取部分 KV |
| **PagedAttention** | 可與 Ring Attention 結合，進一步優化記憶體 |
| **FlashAttention** | 每個 chunk 計算時可使用 FlashAttention 加速 |
| **Pipeline Parallelism** | 可與 Ring Attention 結合 (層級 + 序列) |
| **RoPE** | 需要位置編碼外推配合長上下文 |

---

## 6. 知名實現

| 框架 | 說明 |
|------|------|
| **Ring Attention Paper** | 原始論文 |
| **FlashAttention Ring** | FlashAttention 團隊的 Ring 實現 |
| **vLLM** | 推論時支援 Ring Attention |
| **DeepSpeed Ulysses** | DeepSpeed 的 Ring Attention 實現 |

---

## 延伸閱讀

- [Ring Attention Paper](https://arxiv.org/abs/2310.01889)
- [Long Context Training](https://arxiv.org/abs/2402.17762)
- [DeepSpeed Ulysses](https://www.microsoft.com/en-us/research/publication/ulysses-deconstructing-sequence-parallelism-in-light-of-transformers/)
- [FlashDecoding](https://arxiv.org/abs/2311.06683)