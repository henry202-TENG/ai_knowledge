# Ring Attention

讓 KV Cache 分布到多個 GPU 裝置上，實現百萬級 token 上下文的分散式長上下文注意力機制。

---

## 1. 什麼是？

### 深度定義

**Ring Attention** 是一種將**長序列的 KV Cache 分散到多個 GPU**的技術：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ring Attention 核心思想                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  問題:                                                               │
│  - 標準 Attention 需要 O(n²) 記憶體                                  │
│  - 單 GPU 記憶體有限，無法處理長序列                                  │
│                                                                      │
│  解決: 環形分佈 + 分塊計算                                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  序列: [t0, t1, t2, ..., t99, t100, ..., t999]             │   │
│  │                                                              │   │
│  │  分散到 4 GPUs:                                              │   │
│  │  GPU 0: [t0-t249]    KV Cache                              │   │
│  │  GPU 1: [t250-t499]   KV Cache                              │   │
│  │  GPU 2: [t500-t749]   KV Cache                              │   │
│  │  GPU 3: [t750-t999]   KV Cache                              │   │
│  │                                                              │   │
│  │  環形通訊: 每個 GPU 依次獲取其他 GPU 的 KV 來計算 Attention   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  結果:                                                               │
│  - 記憶體: O(n²/N)  (N = GPU 數)                                    │
│  - 通訊: O(n) (每個 GPU 只與相鄰 GPU 通訊)                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

## 7. 數學推導

### Ring Attention 計算

```
標準 Attention:
  Attention(Q, K, V) = softmax(QK^T / √d) V

Ring Attention 分解:
  將 K, V 沿序列維度分為 N 份:
    K = [K₀, K₁, ..., K_{N-1}]
    V = [V₀, V₁, ..., V_{N-1}]

  每個 GPU 計算:
    O_i = Attention(Q_local, K_i, V_i)
      = softmax(Q_local × K_i^T / √d) × V_i

  最終輸出:
    O = Σ_i O_i
```

### 複雜度分析

```
假設:
  - 序列長度 L
  - GPU 數量 N
  - 每個 GPU 處理 L/N 個 tokens
  - 隱藏維度 d

標準 Attention (單 GPU):
  - 計算: O(L² × d)
  - 記憶體: O(L²)

Ring Attention (N GPUs):
  - 計算 per GPU: O((L/N)² × d) + O(L × d × N)
  - 通訊: O(L × d × N)  # 每個 GPU 需要接收其他 GPU 的 KV
  - 記憶體 per GPU: O(L²/N)

收益:
  - 計算: N 倍並行
  - 記憶體: 1/N
```

---

## 8. 實現細節

### 完整 Ring Attention 實現

```python
import torch
import torch.nn.functional as F

class RingAttention(nn.Module):
    def __init__(self, num_heads, head_dim, num_gpus=4):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_gpus = num_gpus

    def forward(
        self,
        q,  # [batch, local_seq, num_heads, head_dim]
        kv_chunks,  # list of [batch, chunk_seq, num_heads, head_dim]
        ring_rank: int,
        ring_size: int
    ):
        """
        Ring Attention 前向傳播
        """
        batch_size = q.size(0)
        num_heads = q.size(2)
        head_dim = q.size(3)

        # 本地計算結果
        output = torch.zeros_like(q)

        # 環形迭代
        for step in range(ring_size):
            # 當前要處理的 KV chunk
            current_kv = kv_chunks[(ring_rank + step) % ring_size]

            # 本地計算 attention
            local_out = self._flash_attention(
                q, current_kv.k, current_kv.v
            )
            output = output + local_out

            # 將 Q 傳遞到下一個 GPU
            if step < ring_size - 1:
                q = self._ring_send(q, (ring_rank + 1) % ring_size)

        # 標準化 (因為進行了 N 次 softmax)
        output = output / ring_size

        return output

    def _flash_attention(self, q, k, v):
        """使用 FlashAttention 加速計算"""
        return F.scaled_dot_product_attention(q, k, v)

    def _ring_send(self, tensor, dest_rank):
        """環形傳遞張量"""
        # 使用 NCCL 或其他通訊庫
        return send_to_gpu(tensor, dest_rank)
```

### 與 Pipeline 組合

```python
class RingPipelineAttention(nn.Module):
    """Ring Attention + Pipeline Parallelism"""

    def __init__(self, num_layers, num_gpus):
        self.num_layers = num_layers
        self.ring_attention = RingAttention(num_gpus=num_gpus)
        self.pipeline_stage = num_layers // num_gpus

    def forward(self, x, kv_cache_per_layer):
        # 環形注意力 (處理序列維度)
        x = self.ring_attention(x, kv_cache_per_layer)

        # 流水線 (處理層級維度)
        x = self.pipeline_layers(x)

        return x
```

---

## 9. 效能優化

### 通訊與計算重疊

```python
class OverlappedRingAttention:
    """計算與通訊重疊的 Ring Attention"""

    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.stream = [torch.cuda.Stream() for _ in range(num_gpus)]

    def forward(self, q, kv_local):
        """異步執行計算和通訊"""

        outputs = [None] * self.num_gpus

        for step in range(self.num_gpus):
            with torch.cuda.stream(self.stream[step]):
                # 異步計算當前 step
                outputs[step] = self._compute_attention(
                    q, kv_local[step]
                )

                # 異步傳遞 KV 到下一個 GPU
                if step < self.num_gpus - 1:
                    self._async_send(
                        kv_local[step],
                        (rank + 1) % self.num_gpus
                    )

        # 同步所有 stream
        torch.cuda.synchronize()

        return sum(outputs)
```

### 優化參數

| 參數 | 建議值 | 影響 |
|------|--------|------|
| **Chunk Size** | 512-2048 | 太小:通訊多，太大:記憶體高 |
| **Pipeline Depth** | 2-4 | 計算與通訊重疊程度 |
| **Prefetch Distance** | 1-2 | 隱藏通訊延遲 |

---

## 10. 基準測試

### 延遲比較

```
序列長度: 100K tokens, 8 GPU

方法                  延遲 (s)   記憶體/GPU (GB)
────────────────────────────────────────────────
單 GPU (OOM)          N/A       > 80
All-to-All            45s       12
Ring Attention        12s       10
Ring + FlashAttn      8s        9
Ring + Pipeline       6s        8
```

### 可擴展性

```
序列長度 vs GPU 數量:

        1 GPU   4 GPUs   8 GPUs   16 GPUs
16K     1.2s    0.4s     0.3s     0.3s
64K     8.5s    2.5s     1.5s     1.0s
128K   35s      9s       5s       3.5s
256K   OOM     38s      20s       12s
```

---

## 11. 硬體考量

### NVLink 拓撲

```
8 GPU 伺服器拓撲:

NVLink:
GPU 0 ── NVLink ── GPU 1 ── NVLink ── GPU 2
 │                   │                   │
 └───────────────────┼───────────────────┘

Ring 順序優化:
  GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 4 → GPU 5 → GPU 6 → GPU 7 → GPU 0

最佳順序: 沿 NVLink 拓撲
```

### 跨節點 Ring Attention

```
跨節點 (2 節點, 每節點 8 GPU):

節點內: NVLink Ring (8 GPUs)
節點間: RDMA Ring

完整 Ring:
  GPU 0 → ... → GPU 7 → [RDMA] → GPU 8 → ... → GPU 15 → GPU 0
```

---

## 12. 與其他長上下文方法比較

| 方法 | 原理 | 優點 | 缺點 |
|------|------|------|------|
| **Ring Attention** | 分散 KV | 線性記憶體 | 通訊開銷 |
| **Streaming LLM** | 滑動窗口 | 記憶體固定 | 忽略早期資訊 |
| **Sparse Attention** | 稀疏連接 | 計算節省 | 精度損失 |
| **Hierarchical** | 分層處理 | 靈活 | 實現複雜 |

### 混合策略

```python
class HybridLongContext:
    """結合多種長上下文技術"""

    def __init__(self, max_local=8192, ring_gpus=4):
        self.local_window = max_local
        self.ring = RingAttention(num_gpus=ring_gpus)

    def forward(self, x):
        # 局部: 使用本地 attention
        local_out = self.local_attention(x[:, -self.local_window:])

        # 全局: 使用 Ring Attention
        global_out = self.ring(x)

        # 融合
        return local_out + global_out
```

---

## 13. 相關主題

| 技術 | 關係 |
|------|------|
| **KV Cache** | Ring Attention 的基礎，每個 GPU 快取部分 KV |
| **PagedAttention** | 可與 Ring Attention 結合，進一步優化記憶體 |
| **FlashAttention** | 每個 chunk 計算時可使用 FlashAttention 加速 |
| **Pipeline Parallelism** | 可與 Ring Attention 結合 (層級 + 序列) |
| **RoPE** | 需要位置編碼外推配合長上下文 |

---

## 延伸閱讀

- [Ring Attention Paper](https://arxiv.org/abs/2310.01889)
- [Long Context Training](https://arxiv.org/abs/2402.17762)
- [DeepSpeed Ulysses](https://www.microsoft.com/en-us/research/publication/ulysses-deconstructing-sequence-parallelism-in-light-of-transformers/)
- [FlashDecoding](https://arxiv.org/abs/2311.06683)
- [Streaming LLM](https://arxiv.org/abs/2309.17453)