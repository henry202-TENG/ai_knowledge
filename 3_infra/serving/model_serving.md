# Model Serving

將訓練好的模型部署為服務的技術，是 LLM 應用的關鍵基礎設施。

---

## 1. 什麼是？

### 簡單範例

```
本地模型:
  載入模型 → 處理輸入 → 輸出結果
  (單機、單用戶)

模型服務:
  API 接口 ← 請求隊列 ← 用戶
       ↓
  模型池 ← 負載均衡
       ↓
  GPU 集群
  (多用户、高可用、可擴展)
```

---

## 2. 服務框架

### vLLM

```python
from vllm import LLM, SamplingParams

# 初始化引擎
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4,  # 4 GPU
    gpu_memory_utilization=0.9,
    max_num_seqs=256
)

# 推斷
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512
)

outputs = llm.generate(
    prompts=["Hello, world!"],
    sampling_params=sampling_params
)

for output in outputs:
    print(output.outputs[0].text)
```

### FastAPI + Transformers

```python
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# 載入模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )

    return {
        "text": tokenizer.decode(outputs[0])
    }
```

### Triton Inference Server

```python
import tritonclient.http as httpclient

# 客戶端調用
client = httpclient.InferenceServerClient(url="localhost:8000")

# 準備輸入
inputs = [
    httpclient.InferInput(
        "INPUT_TEXT",
        [1, 512],
        "BYTES"
    )
]

# 執行推斷
results = client.infer(
    model_name="llama_model",
    inputs=inputs
)
```

---

## 3. 部署架構

### 單機部署

```
┌─────────────────┐
│   FastAPI      │
│   (HTTP Server) │
└────────┬────────┘
         │
┌────────▼────────┐
│   PyTorch      │
│   Runtime      │
└────────┬────────┘
         │
┌────────▼────────┐
│   GPU 0        │
│   (Llama 7B)   │
└─────────────────┘
```

### 集群部署

```
                    ┌──────────────┐
                    │ Load Balancer│
                    │   (nginx)    │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐        ┌────▼────┐
   │GPU 0-1  │       │GPU 2-3  │        │GPU 4-5  │
   │Node 1   │       │Node 2   │        │Node 3   │
   │(LLama 7B)│       │(LLama 7B)│       │(LLama 7B)│
   └─────────┘       └─────────┘        └─────────┘
```

### Kubernetes 部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-server
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "40Gi"
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-server
  ports:
  - port: 80
    targetPort: 8000
```

---

## 4. 性能優化

### 批量處理

```python
class BatchingManager:
    """動態批量處理"""

    def __init__(self, max_batch_size=32, timeout=0.1):
        self.queue = []
        self.max_batch = max_batch_size
        self.timeout = timeout

    async def add_request(self, prompt):
        """添加請求到隊列"""

        future = asyncio.Future()
        self.queue.append({
            "prompt": prompt,
            "future": future
        })

        # 觸發批量處理
        if len(self.queue) >= self.max_batch:
            await self.process_batch()

        return await future

    async def process_batch(self):
        """處理批量請求"""

        batch = self.queue[:self.max_batch]
        self.queue = self.queue[self.max_batch:]

        # 批量推理
        outputs = self.model.generate(batch["prompts"])

        # 結果分發
        for i, output in enumerate(outputs):
            batch[i]["future"].set_result(output)
```

### 連續批處理 (Continuous Batching)

```python
# vLLM 連續批處理配置
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # 動態批處理
    enforce_eager=False,
    max_num_seqs=256,
    max_seq_len_to_capture=4096,
)
# vLLM 自動:
# 1. 接收新請求加入批處理
# 2. 完成的請求移出
# 3. 動態調整批大小
```

### 緩存優化

```python
class KVCacheManager:
    """KV Cache 管理"""

    def __init__(self, cache_size_gb=20):
        self.cache = {}
        self.max_size = cache_size_gb * 1024**3

    def get_or_compute(self, prompt_hash, compute_fn):
        """緩存或計算"""

        if prompt_hash in self.cache:
            return self.cache[prompt_hash]

        # 計算並緩存
        result = compute_fn()
        self.cache[prompt_hash] = result

        # 內存管理
        if self._exceeds_limit():
            self._evict_lru()

        return result
```

---

## 5. 高可用性

### 負載均衡

```python
import httpx

class LLMGateway:
    """LLM 網關 - 請求分發"""

    def __init__(self, backends):
        self.backends = backends
        self.current = 0

    async def forward(self, request):
        """請求分發"""

        # 簡單輪詢
        backend = self.backends[self.current]
        self.current = (self.current + 1) % len(self.backends)

        try:
            return await self._call_backend(backend, request)
        except Exception as e:
            # 故障轉移
            return await self._failover(request)

    async def health_check(self):
        """健康檢查"""

        for backend in self.backends:
            try:
                response = await httpx.AsyncClient().get(
                    f"{backend}/health",
                    timeout=5
                )
                backend["healthy"] = response.status == 200
            except:
                backend["healthy"] = False
```

### 熔斷器

```python
class CircuitBreaker:
    """熔斷器"""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"
        self.last_failure_time = None

    async def call(self, func):
        """熔斷保護"""

        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError()

        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
```

---

## 6. 監控指標

### 關鍵指標

```python
class ServingMetrics:
    """服務指標"""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "latency_sum": 0,
            "tokens_generated": 0,
            "gpu_utilization": [],
            "queue_length": []
        }

    def record_request(self, latency, success, tokens):
        """記錄請求"""

        self.metrics["requests_total"] += 1

        if success:
            self.metrics["requests_success"] += 1
        else:
            self.metrics["requests_failed"] += 1

        self.metrics["latency_sum"] += latency
        self.metrics["tokens_generated"] += tokens

    def get_stats(self):
        """獲取統計"""

        total = self.metrics["requests_total"]

        return {
            "qps": total / 60,  # 每分鐘
            "success_rate": (
                self.metrics["requests_success"] / max(total, 1)
            ),
            "avg_latency_ms": (
                self.metrics["latency_sum"] / max(total, 1)
            ),
            "tokens_per_second": (
                self.metrics["tokens_generated"] / max(total, 1)
            )
        }
```

### Prometheus 整合

```python
from prometheus_client import Counter, Histogram, Gauge

# 定義指標
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_duration_seconds',
    'Request latency',
    ['model']
)

GPU_MEMORY = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory usage',
    ['device']
)

# 在請求處理中使用
@app.middleware("http")
async def track_metrics(request, call_next):
    start = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(
        model=request.state.model,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        model=request.state.model
    ).observe(time.time() - start)

    return response
```

---

## 7. 挑戰與解決

| 挑戰 | 解決方案 |
|------|----------|
| **延遲高** | 批量處理、連續批處理、緩存 |
| **GPU 記憶體不足** | 量化、Streaming、Partition |
| **高並發** | 負載均衡、隊列管理 |
| **故障恢復** | 多副本、熔斷器、健康檢查 |
| **成本高** | 按需擴縮、GPU 共享 |

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **vLLM** | 高效推理引擎 |
| **Kubernetes** | 容器編排 |
| **Quantization** | 降低記憶體需求 |
| **Speculative Decoding** | 加速推斷 |

---

## 延伸閱讀

- [vLLM Documentation](https://docs.vllm.ai/)
- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [KServe](https://kserve.github.io/website/)