# LLM Performance Optimization

LLM 系統的全面性能優化技術，包括推理延遲、吞吐量、成本優化。

---

## 1. 什麼是？

### 深度定義

**LLM Performance Optimization** 是系統性**提升推理速度、降低成本、提高吞吐量**的技術集合：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM 優化維度                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  優化目標:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  延遲 (Latency):  首Token產生時間 (TTFT) + 總生成時間        │   │
│  │    - 目標: < 1秒 for 交互式應用                              │   │
│  │    - 瓶頸: Prefill 階段計算密集                              │   │
│  │                                                              │   │
│  │  吞吐量 (Throughput):  每秒處理 tokens 數                   │   │
│  │    - 目標: 高並發場景                                        │   │
│  │    - 瓶頸: Decode 階段串行生成                               │   │
│  │                                                              │   │
│  │  成本 (Cost):  每千tokens成本                                │   │
│  │    - 目標: 最低化資源消耗                                    │   │
│  │    - 瓶頸: GPU 計算和記憶體                                  │   │
│  │                                                              │   │
│  │  記憶體 (Memory):  VRAM 使用                                 │   │
│  │    - 目標: 最大化 batch size                                 │   │
│  │    - 瓶頸: KV Cache 大小                                     │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  優化層次:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. 系統層: CUDA、核心庫優化                                 │   │
│  │     - Flash Attention v2/v3                                  │   │
│  │     - CUDA Graph                                             │   │
│  │     - TensorRT-LLM                                           │   │
│  │                                                              │   │
│  │  2. 框架層: 推理引擎                                         │   │
│  │     - vLLM: PagedAttention                                   │   │
│  │     - Text Generation Inference (TGI)                       │   │
│  │     - DeepSpeed Inference                                    │   │
│  │                                                              │   │
│  │  3. 模型層: 模型結構優化                                     │   │
│  │     - Quantization: FP16 → INT8 → INT4                      │   │
│  │     - Pruning: 剪枝不必要的權重                              │   │
│  │     - Distillation: 知識蒸餾到小模型                         │   │
│  │                                                              │   │
│  │  4. 應用層: 請求處理優化                                     │   │
│  │     - Dynamic Batching                                       │   │
│  │     - Semantic Caching                                       │   │
│  │     - Speculative Decoding                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  優化權衡:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  速度 ↑ = 成本 ↓ = 質量可能 ↓                                │   │
│  │                                                              │   │
│  │  量化: 4x 加速 but 5-10% 質量損失                            │   │
│  │  蒸餾: 10x 小模型 but 需要額外訓練                           │   │
│  │  快取: 0 成本 but 只適用重複請求                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何重要**:
1. **用戶體驗**: 低延遲是交互式應用的關鍵
2. **成本控制**: 優化可直接降低運營成本
3. **競爭力**: 高吞吐量支撐更多用戶
4. **可擴展性**: 優化後可服務更多並發

### 簡單範例

```
優化前:
  輸入問題 → 等待 30 秒 → 輸出回答
  成本: $0.05/請求

優化後:
  輸入問題 → 等待 2 秒 → 輸出回答
  成本: $0.002/請求

提升: 15x 速度，25x 成本降低
```

---

## 2. 延遲優化

### Pipelining

```python
class PipelineOptimizer:
    """流水線優化"""

    def __init__(self, model):
        self.model = model
        self.preprocess_queue = []
        self.infer_queue = []
        self.postprocess_queue = []

    def process_stream(self, input_stream):
        """流式處理"""

        for input_data in input_stream:
            # 1. 預處理 (非阻塞)
            preprocessed = self._preprocess(input_data)

            # 2. 推理
            result = self.model.generate(preprocessed)

            # 3. 後處理
            output = self._postprocess(result)

            yield output

    def _preprocess(self, data):
        """預處理"""
        return self.model.tokenizer(data)
```

### 請求合併

```python
class RequestBatching:
    """請求合併"""

    def __init__(self, max_batch_size=32, timeout_ms=100):
        self.queue = []
        self.max_batch = max_batch_size
        self.timeout = timeout_ms / 1000

    async def add_request(self, prompt):
        """添加請求"""

        future = asyncio.Future()
        self.queue.append({
            "prompt": prompt,
            "future": future,
            "arrival": time.time()
        })

        # 觸發處理
        if len(self.queue) >= self.max_batch:
            await self.process_batch()

        return await future

    async def process_batch(self):
        """處理批量"""

        batch = self.queue[:self.max_batch]
        self.queue = self.queue[self.max_batch:]

        # 批量推理
        prompts = [r["prompt"] for r in batch]
        outputs = await self.model.generate_batch(prompts)

        # 分發結果
        for req, output in zip(batch, outputs):
            req["future"].set_result(output)
```

---

## 3. 吞吐量優化

### Prefill 優化

```python
class PrefillOptimization:
    """預填充優化"""

    @staticmethod
    def speculative_prefill(model, prompt, num_speculations=5):
        """投機預填充"""

        # 快速生成 draft tokens
        draft = model.fast_generate(prompt, num_speculations)

        # 驗證 draft
        verified = model.verify_batch(prompt, draft)

        # 接受正確的
        accepted = []
        for i, token in enumerate(verified):
            if token == draft[i]:
                accepted.append(token)
            else:
                break

        return accepted
```

### 緩存優化

```python
class KVCacheManager:
    """KV Cache 優化"""

    def __init__(self, max_cache_size_gb=40):
        self.cache = {}
        self.max_size = max_cache_size_gb * 1024**3

    def get_cache(self, prompt_hash):
        """獲取缓存"""

        return self.cache.get(prompt_hash)

    def store_cache(self, prompt_hash, kv_cache):
        """存儲缓存"""

        # 檢查大小
        cache_size = self._estimate_size(kv_cache)

        if cache_size > self.max_size:
            self._evict()

        self.cache[prompt_hash] = kv_cache

    def _evict(self):
        """淘汰策略 - LRU"""

        # 移除最舊的
        oldest = min(
            self.cache.items(),
            key=lambda x: x[1]["last_used"]
        )

        del self.cache[oldest[0]]
```

---

## 4. 內存優化

### 量化配置

```python
QUANTIZATION_CONFIGS = {
    "fp8": {
        "description": "8-bit 浮點",
        "speedup": "1.5x",
        "accuracy_loss": "< 1%",
        "memory_reduction": "2x"
    },
    "int8": {
        "description": "8-bit 整數",
        "speedup": "2x",
        "accuracy_loss": "2-3%",
        "memory_reduction": "2x"
    },
    "int4": {
        "description": "4-bit 整數",
        "speedup": "4x",
        "accuracy_loss": "5-10%",
        "memory_reduction": "4x"
    }
}

# vLLM 配置
llm = LLM(
    model="meta-llama/Llama-2-7b",
    quantization="fp8",  # or "awq", "gptq"
    tensor_parallel_size=2
)
```

### 模型蒸餾

```python
class ModelDistillation:
    """模型蒸餾"""

    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.temperature = 4.0

    def distill(self, dataset):
        """蒸餾訓練"""

        for batch in dataset:
            # Teacher 推理
            with torch.no_grad():
                teacher_logits = self.teacher(batch["input"])

            # Student 推理
            student_logits = self.student(batch["input"])

            # 蒸餾 Loss
            loss = self._distillation_loss(
                student_logits,
                teacher_logits,
                batch["labels"]
            )

            loss.backward()
            self.optimizer.step()
```

---

## 5. 成本優化

### 成本計算

```python
class CostOptimizer:
    """成本優化器"""

    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075}
    }

    def calculate_cost(self, model_name, prompt_tokens, completion_tokens):
        """計算成本"""

        prices = self.PRICING.get(model_name, {})

        input_cost = prompt_tokens / 1000 * prices["input"]
        output_cost = completion_tokens / 1000 * prices["output"]

        return input_cost + output_cost

    def optimize_model_selection(self, task_complexity):
        """優化模型選擇"""

        if task_complexity == "simple":
            return "gpt-3.5-turbo"
        elif task_complexity == "moderate":
            # 成本 vs 質量權衡
            return "claude-3-sonnet"
        else:
            return "gpt-4"
```

### 快取策略

```python
class SemanticCache:
    """語意緩存"""

    def __init__(self, similarity_threshold=0.95):
        self.cache = {}
        self.threshold = similarity_threshold

    def get_cached_response(self, prompt, embedding_fn):
        """檢查緩存"""

        prompt_emb = embedding_fn(prompt)

        best_match = None
        best_score = 0

        for cached_prompt, (cached_emb, response) in self.cache.items():
            score = self._cosine_similarity(prompt_emb, cached_emb)

            if score > best_score:
                best_score = score
                best_match = (cached_prompt, response)

        if best_score >= self.threshold:
            return best_match[1], best_score

        return None, 0

    def cache_response(self, prompt, embedding_fn, response):
        """緩存回覆"""

        prompt_emb = embedding_fn(prompt)

        self.cache[prompt] = (prompt_emb, response)
```

---

## 6. 監控指標

### 關鍵指標

```python
PERFORMANCE_METRICS = {
    "latency": {
        "p50": "50% 請求延遲",
        "p95": "95% 請求延遲",
        "p99": "99% 請求延遲"
    },
    "throughput": {
        "tps": "Tokens per second",
        "rps": "Requests per second"
    },
    "efficiency": {
        "tokens_per_dollar": "每美元處理 tokens",
        "gpu_utilization": "GPU 利用率"
    },
    "reliability": {
        "success_rate": "成功率",
        "error_rate": "錯誤率"
    }
}

class MetricsCollector:
    """指標收集器"""

    def __init__(self):
        self.latencies = []
        self.tokens = []
        self.errors = []

    def record(self, latency, tokens, error=None):
        self.latencies.append(latency)
        self.tokens.append(tokens)

        if error:
            self.errors.append(error)

    def get_stats(self):
        import numpy as np

        return {
            "latency_p50": np.percentile(self.latencies, 50),
            "latency_p95": np.percentile(self.latencies, 95),
            "latency_p99": np.percentile(self.latencies, 99),
            "throughput": sum(self.tokens) / sum(self.latencies),
            "success_rate": 1 - len(self.errors) / max(len(self.latencies), 1)
        }
```

---

## 7. 調優清單

### 快速優化清單

```
優先順序:

1. [高] 啟用 KV Cache
2. [高] 使用批量推理
3. [高] 啟用 FP8 量化
4. [中] 啟用 Flash Attention
5. [中] 啟用 prefix caching
6. [中] 優化 max_tokens
7. [低] 使用更小的模型
8. [低] 實現 semantic caching
```

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **Quantization** | 內存/速度優化 |
| **Speculative Decoding** | 吞吐量優化 |
| **Caching** | 成本優化 |

---

## 延伸閱讀

- [vLLM Performance](https://docs.vllm.ai/en/latest/)
- [NVIDIA Inference](https://docs.nvidia.com/deeplearning/)
- [LLM Optimization](https://hamel.dev/notes/llm/optimization/)