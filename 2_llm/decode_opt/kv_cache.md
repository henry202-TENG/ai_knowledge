# KV Cache

推論優化技術，快取已計算過的 Key 和 Value 矩陣，避免重複計算，顯著降低延遲。

---

## 1. 什麼是？

### 深度定義

**KV Cache** 是 Transformer 推論優化的基石技術，其核心價值在於利用 Transformer 的**自迴歸性質**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Transformer 自迴歸生成示意                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  時刻 t:                                                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Input: [Token₀, Token₁, ..., Token_{t-1}]                   │ │
│  │                              ↓                                  │ │
│  │  計算每個位置對最後位置的 Attention 贡献                        │ │
│  │                              ↓                                  │ │
│  │  Output: Token_t                                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  關鍵觀察:                                                            │
│  - [Token₀, Token₁, ..., Token_{t-1}] 的 K, V 已經計算過             │
│  - 這些 K, V 在後續所有時刻都會被重複使用                             │
│  - 我們只需要計算新 token 的 K, V                                    │
│                                                                      │
│  這就是 KV Cache 要保存的內容！                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

### 深度挑戰與解決方案

#### 挑戰 3：長上下文下的 Attention 計算瓶頸

**問題描述**：
```
當序列長度達到 128K+ 時，Attention 計算成為瓶頸

複雜度: O(n² × d)
n = 128K 時:
- Attention 計算量 ≈ 128K² × 128 ≈ 2×10¹² 運算
- 即使使用 GPU，也需要數秒鐘
```

**解決方案 - Flash Attention 原理**：
```python
def flash_attention(Q, K, V, block_size=256):
    """
    Flash Attention: IO-aware exact attention

    核心思想: 將 O(N²) 的 memory complexity 降低到 O(N)
    通過分塊計算 + online softmax
    """

    # Q, K, V 形狀: [batch, heads, seq, head_dim]

    num_blocks = (Q.shape[2] + block_size - 1) // block_size

    # 分塊計算
    output = torch.zeros_like(Q)

    for i in range(num_blocks):
        # 載入 Q 塊
        q_block = Q[:, :, i*block_size:(i+1)*block_size, :]

        # 計算這塊的 attention
        # 使用 online softmax 技巧
        for j in range(num_blocks):
            # 載入 K, V 塊
            k_block = K[:, :, j*block_size:(j+1)*block_size, :]
            v_block = V[:, :, j*block_size:(j+1)*block_size, :]

            # 計算 partial attention
            attn_block = torch.matmul(q_block, k_block.transpose(-2, -1))

            # 分塊 softmax
            block_max = attn_block.max(dim=-1, keepdim=True)[0]
            block_exp = torch.exp(attn_block - block_max)

            # 累積到輸出
            output[:, :, i*block_size:(i+1)*block_size, :] += \
                torch.matmul(block_exp, v_block)

    return output
```

#### 挑戰 4：多請求下的 KV Cache 衝突

**問題描述**：
```
多個用戶請求同時進行時:
- 每個請求需要獨立的 KV Cache
- 記憶體碎片化
- 請求之間的隔離困難
```

**解決方案 - vLLM 的 PagedAttention**：
```python
class PagedKVCache:
    """
    vLLM 的分頁 KV Cache 管理

    核心設計:
    - 將 KV Cache 劃分為固定大小的頁面
    - 頁面不需要物理連續
    - 頁面可以被多個請求共享
    """

    def __init__(self, page_size=16):
        self.page_size = page_size
        self.pages = {}  # page_id -> physical_memory
        self.page_table = {}  # sequence_id -> [page_ids]

    def allocate(self, sequence_id, max_length):
        """為序列分配頁面"""

        num_pages = (max_length + self.page_size - 1) // self.page_size

        # 分配物理頁面
        allocated_pages = []
        for _ in range(num_pages):
            page_id = self._allocate_free_page()
            allocated_pages.append(page_id)

        # 建立頁表映射
        self.page_table[sequence_id] = allocated_pages

        return allocated_pages

    def append_token(self, sequence_id, token_id):
        """追加新 token 到序列"""

        # 找到當前位置對應的頁面
        current_len = self.get_length(sequence_id)
        page_idx = current_len // self.page_size
        offset = current_len % self.page_size

        # 如果需要新頁面
        if offset == 0:
            new_page = self._allocate_free_page()
            self.page_table[sequence_id].append(new_page)

        # 計算並存儲 KV
        page_id = self.page_table[sequence_id][page_idx]
        self._write_kv(page_id, offset, token_id)
```

#### 挑戰 5：KV Cache 預取延遲

**問題描述**：
```
Decode 階段每生成一個 token 都需要:
1. 讀取現有 KV Cache
2. 計算新 token 的 K, V
3. 寫入 KV Cache
4. 計算 Attention

當 GPU 計算太快時，記憶體讀取成為瓶頸
```

**解決方案 - 預取優化**：
```python
class PrefetchAwareScheduler:
    """預取感知排程"""

    def __init__(self, model):
        self.model = model
        self.prefetch_queue = asyncio.Queue()

    async def schedule_with_prefetch(self, requests):
        """調度 + 預取"""

        # 預取下一批請求需要的 KV
        next_batch = self._get_next_batch(requests)

        prefetch_task = asyncio.create_task(
            self._prefetch_kv(next_batch)
        )

        # 同時處理當前批次
        current_batch = self._get_current_batch(requests)
        results = await self._process_batch(current_batch)

        # 等待預取完成
        prefetched_kv = await prefetch_task

        # 將預取的 KV 注入到下一批
        self._inject_prefetched(next_batch, prefetched_kv)

        return results

    async def _prefetch_kv(self, batch):
        """預先計算即將需要的 KV"""
        prefetched = {}

        for seq in batch:
            # 預測下一批要處理的 token
            next_tokens = self._predict_next(seq)

            for token in next_tokens:
                # 異步計算 K, V
                kv = await self._compute_kv_async(token)
                prefetched[(seq.id, token)] = kv

        return prefetched
```

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Speculative Decoding** | 依賴 KV Cache 加速驗證 |
| **Context Window** | 需要更大的 KV Cache |
| **PagedAttention** | KV Cache 的記憶體管理方案 |
| **Flash Attention** | 與 KV Cache 互補的計算優化 |

---

## 7. 記憶體計算詳細

### KV Cache 記憶體佔用

```
單層 Transformer 的 KV Cache 大小:

假設:
  - 序列長度 n = 4096
  - 隱藏維度 d = 4096
  - 頭數 h = 32
  - 每頭維度 d_head = 128
  - 數據類型: FP16 (2 bytes)

每層的 KV Cache:
  KV = 2 × n × h × d_head × 2 (K + V)
     = 2 × 4096 × 32 × 128 × 2
     = 64 MB

整個模型 (32 層):
  Total = 64 MB × 32 = 2 GB

多請求並發:
  - 10 個請求: 20 GB
  - 100 個請求: 200 GB
```

### 記憶體優化公式

```python
def calculate_kv_cache_memory(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    dtype_bytes: int = 2
):
    """計算 KV Cache 記憶體"""

    # 每個 token 的 K/V 大小
    per_token = num_heads * head_dim * dtype_bytes  # bytes

    # K + V
    per_token *= 2

    # 完整序列
    full_cache = per_token * max_seq_len

    # 考慮所有層
    total = full_cache * num_layers

    # 轉換為 MB
    return total / (1024 ** 2)
```

---

## 8. 注意力機制深入

### 多頭注意力計算

```python
def multi_head_attention(
    Q, K, V,
    num_heads: int,
    head_dim: int
):
    """
    Q: [batch, seq_q, d_model]
    K, V: [batch, seq_kv, d_model]
    """

    batch_size = Q.size(0)

    # 投影到 Q, K, V
    Q = Q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    # 計算 Attention
    # [batch, heads, seq_q, seq_kv]
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 輸出
    output = torch.matmul(attn_weights, V)
    output = output.transpose(1, 2).contiguous().view(
        batch_size, -1, num_heads * head_dim
    )

    return output
```

### KV Cache 下的推理

```python
def decode_with_cache(
    model,
    input_ids: torch.Tensor,  # [batch, seq_len]
    kv_cache: dict,
    layer_idx: int
):
    """使用 KV Cache 進行解碼"""

    # 取得最後一個 token
    last_token = input_ids[:, -1:]

    # 計算 Q (只需要最後一個位置)
    Q = model.embed(last_token)
    Q = model.layers[layer_idx].attention.W_q(Q)
    Q = Q.view(Q.size(0), 1, model.num_heads, model.head_dim)

    # 取得快取的 K, V
    cached_K = kv_cache[f"layer_{layer_idx}_K"]
    cached_V = kv_cache[f"layer_{layer_idx}_V"]

    # 計算新的 K, V
    new_K = model.layers[layer_idx].attention.W_k(last_token)
    new_V = model.layers[layer_idx].attention.W_v(last_token)
    new_K = new_K.view(new_K.size(0), 1, model.num_heads, model.head_dim)
    new_V = new_V.view(new_V.size(0), 1, model.num_heads, model.head_dim)

    # 追加到快取
    K = torch.cat([cached_K, new_K], dim=2)
    V = torch.cat([cached_V, new_V], dim=2)

    # 更新快取
    kv_cache[f"layer_{layer_idx}_K"] = K
    kv_cache[f"layer_{layer_idx}_V"] = V

    # 計算 Attention
    attn = torch.matmul(Q, K.transpose(-2, -1)) / (model.head_dim ** 0.5)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, V)

    return output.squeeze(1), kv_cache
```

---

## 9. 進階快取策略

### 分層 KV Cache

```python
class HierarchicalKVCache:
    """分層 KV Cache: GPU + CPU + 磁碟"""

    def __init__(self, gpu_capacity_mb=10000, cpu_capacity_mb=100000):
        self.gpu_cache = {}   # 熱數據
        self.cpu_cache = {}   # 溫數據
        self.disk_cache = {}  # 冷數據
        self.gpu_capacity = gpu_capacity_mb
        self.cpu_capacity = cpu_capacity_mb

    def get(self, key):
        """分層獲取"""
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        elif key in self.cpu_cache:
            data = self.cpu_cache.pop(key)
            self.gpu_cache[key] = data  # 升級到 GPU
            return data
        elif key in self.disk_cache:
            data = self.disk_cache.pop(key)
            self.cpu_cache[key] = data  # 升級到 CPU
            return data
        return None

    def put(self, key, value):
        """分層存放"""
        size_mb = value.numel() * value.element_size() / (1024 ** 2)

        if self.get_cache_size_mb() + size_mb > self.gpu_capacity:
            # 驅逐到 CPU
            self._evict_to_cpu()

        self.gpu_cache[key] = value
```

### 主動快取策略

```python
class PredictiveKVCache:
    """預測性 KV Cache"""

    def __init__(self, model):
        self.model = model
        self.access_history = {}
        self.prefetch_threshold = 0.8

    def predict_and_prefetch(self, prompt):
        """預測可能訪問的內容並預先載入"""

        # 簡單預測: 根據歷史
        likely_next = self._predict_next_tokens(prompt)

        # 預先計算這些 token 的 K, V
        prefetch_kv = {}
        for token in likely_next:
            kv = self.model.compute_kv(token)
            prefetch_kv[token] = kv

        return prefetch_kv

    def _predict_next_tokens(self, prompt, top_k=4):
        """簡單的下一 token 預測"""
        with torch.no_grad():
            logits = self.model(prompt)
            probs = F.softmax(logits, dim=-1)
            _, top_indices = probs.topk(top_k)
            return top_indices.tolist()
```

---

## 10. Prefill-Decode 排程

### Split Prefill 策略

```
問題: Prefill 階段計算量大，會阻塞 Decode

解決: 將 Prefill 分成多個小塊，與 Decode 交替執行

時間線 (傳統):
  [Prefill 5s][Decode 0.1s][Decode 0.1s]...
                ↑ 阻塞很長時間

時間線 (Split Prefill):
  [P1 1s][Decode][P2 1s][Decode][P3 1s][Decode]...
  ↑ 可交互響應
```

### 實現

```python
class SplitPrefillScheduler:
    """Split Prefill 排程器"""

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def schedule(self, requests):
        """
        混合排程 Prefill 和 Decode 請求
        """
        prefill_queue = [r for r in requests if r.is_prefill]
        decode_queue = [r for r in requests if r.is_decode]

        schedule = []

        # 交替處理
        while prefill_queue or decode_queue:
            # 先處理 Decode (低延遲)
            if decode_queue:
                schedule.append(("decode", decode_queue.pop(0)))

            # 每處理 N 個 Decode，執行一次 Prefill
            if prefill_queue and len(schedule) % 3 == 0:
                chunk = prefill_queue[0][:self.chunk_size]
                schedule.append(("prefill", chunk))

        return schedule
```

### Continuous Batching

```python
class ContinuousBatching:
    """持續批處理: 動態加入新請求"""

    def __init__(self, max_batch_size=16):
        self.max_batch_size = max_batch_size
        self.running_requests = []

    def add_request(self, request):
        """加入新請求"""
        self.running_requests.append(request)

        # 如果超過容量，等待
        while len(self.running_requests) > self.max_batch_size:
            time.sleep(0.01)

    def step(self):
        """執行一步"""

        # 收集所有請求的下一 token
        batch = [req.current_tokens for req in self.running_requests]

        # 批量推理
        next_tokens = self.model.generate_batch(batch)

        # 更新每個請求
        for req, token in zip(self.running_requests, next_tokens):
            req.append(token)

            # 完成的請求移除
            if req.is_done():
                self.running_requests.remove(req)

        return next_tokens
```

---

## 11. 量化深入

### KV Cache 量化實現

```python
class KVCacheQuantizer:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def quantize(self, tensor):
        """量化 KV Cache"""

        if self.num_bits == 8:
            # INT8 量化
            scale = tensor.abs().max() / 127
            quantized = (tensor / scale).round().to(torch.int8)
            return quantized, scale

        elif self.num_bits == 4:
            # INT4 量化 (使用 group)
            group_size = 64
            shape = tensor.shape
            tensor = tensor.reshape(-1, group_size)

            scale = tensor.abs().max(dim=1, keepdim=True)[0] / 7
            quantized = (tensor / scale).round().to(torch.int8)

            return quantized.reshape(shape), scale

    def dequantize(self, quantized, scale):
        """反量化"""
        return quantized.float() * scale
```

### 量化精度權衡

| 方法 | 困惑度變化 | 記憶體節省 |
|------|-----------|-----------|
| FP16 | 0 (baseline) | 1x |
| INT8 | +0.5% | 2x |
| INT4 | +2% | 4x |
| NF4 | +1% | 4x |

---

## 12. 相關技術

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
- [Continuous Batching](https://arxiv.org/abs/2308.12669)
- [KV Cache Quantization](https://arxiv.org/abs/2312.02288)