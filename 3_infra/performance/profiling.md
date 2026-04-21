# LLM Profiling & Benchmarking

LLM 系統的性能分析和基準測試，識別瓶頸和優化方向。

---

## 1. 性能指標

### 延遲測量

```python
import time
from contextlib import contextmanager

class LatencyProfiler:
    """延遲分析器"""

    def __init__(self):
        self.measurements = {}

    @contextmanager
    def measure(self, operation_name):
        """測量操作延遲"""

        start = time.perf_counter()

        yield

        end = time.perf_counter()
        latency = (end - start) * 1000  # ms

        if operation_name not in self.measurements:
            self.measurements[operation_name] = []

        self.measurements[operation_name].append(latency)

    def get_stats(self):
        """獲取統計"""

        import numpy as np

        stats = {}
        for op, latencies in self.measurements.items():
            stats[op] = {
                "mean": np.mean(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "count": len(latencies)
            }

        return stats


# 使用
profiler = LatencyProfiler()

with profiler.measure("tokenize"):
    tokens = tokenizer.encode(prompt)

with profiler.measure("llm_forward"):
    outputs = model(tokens)

with profiler.measure("decode"):
    result = tokenizer.decode(outputs)

print(profiler.get_stats())
```

### 吞吐量測量

```python
class ThroughputBenchmark:
    """吞吐量基準測試"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def benchmark(self, num_requests=100, batch_size=1):
        """基準測試"""

        prompts = self._generate_test_prompts(num_requests)

        start_time = time.time()
        total_tokens = 0

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # 編碼
            inputs = self.tokenizer(batch, return_tensors="pt")

            # 推斷
            outputs = self.model.generate(**inputs, max_new_tokens=100)

            # 計數 tokens
            total_tokens += outputs.shape[1]

        end_time = time.time()
        duration = end_time - start_time

        return {
            "requests": num_requests,
            "total_tokens": total_tokens,
            "duration_seconds": duration,
            "requests_per_second": num_requests / duration,
            "tokens_per_second": total_tokens / duration,
            "avg_latency": duration / num_requests * 1000
        }

    def _generate_test_prompts(self, num):
        """生成測試 prompt"""
        base_prompt = "Write a short story about"
        return [f"{base_prompt} topic {i}" for i in range(num)]
```

---

## 2. 火焰圖分析

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, schedule

class TorchProfiler:
    """PyTorch 性能分析"""

    def __init__(self, model):
        self.model = model

    def profile_forward(self, input_data):
        """分析前向傳播"""

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            schedule=schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(10):
                self.model(input_data)
                prof.step()

        return prof

    def _trace_handler(self, prof):
        """處理結果"""
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))
```

### GPU 利用率分析

```python
class GPUAnalyzer:
    """GPU 分析"""

    @staticmethod
    def get_gpu_stats():
        """獲取 GPU 統計"""

        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        stats = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            stats.append({
                "device": i,
                "memory_used_mb": memory.used / 1024**2,
                "memory_total_mb": memory.total / 1024**2,
                "memory_percent": memory.used / memory.total * 100,
                "gpu_utilization": utilization.gpu,
                "memory_utilization": utilization.memory
            })

        pynvml.nvmlShutdown()
        return stats
```

---

## 3. 內存分析

### 內存追蹤

```python
class MemoryTracker:
    """內存追蹤"""

    def __init__(self):
        self.snapshots = []

    def snapshot(self, label):
        """記錄內存快照"""

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self.snapshots.append({
            "label": label,
            "timestamp": time.time(),
            "cpu_memory_mb": self._get_cpu_memory(),
            "gpu_memory_mb": self._get_gpu_memory() if torch.cuda.is_available() else 0
        })

    def _get_cpu_memory(self):
        """CPU 內存"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024**2

    def _get_gpu_memory(self):
        """GPU 內存"""
        return torch.cuda.memory_allocated() / 1024**2

    def report(self):
        """生成報告"""

        report = []
        for i, snapshot in enumerate(self.snapshots):
            if i == 0:
                report.append(f"{snapshot['label']}: CPU {snapshot['cpu_memory_mb']:.1f}MB")
            else:
                delta_cpu = snapshot['cpu_memory_mb'] - self.snapshots[i-1]['cpu_memory_mb']
                delta_gpu = snapshot['gpu_memory_mb'] - self.snapshots[i-1]['gpu_memory_mb']
                report.append(
                    f"{snapshot['label']}: CPU {snapshot['cpu_memory_mb']:.1f}MB "
                    f"(Δ{delta_cpu:+.1f}MB) GPU {snapshot['gpu_memory_mb']:.1f}MB "
                    f"(Δ{delta_gpu:+.1f}MB)"
                )

        return "\n".join(report)
```

---

## 4. 瓶頸分析

### 瓶頸識別

```python
class BottleneckAnalyzer:
    """瓶頸分析器"""

    def __init__(self, model):
        self.model = model

    def analyze(self, input_data):
        """分析瓶頸"""

        # 測量各部分時間
        times = {}

        # 1. 編碼
        start = time.perf_counter()
        tokens = self.model.tokenize(input_data)
        times["tokenize"] = time.perf_counter() - start

        # 2. Embedding
        start = time.perf_counter()
        embeds = self.model.embed(tokens)
        times["embedding"] = time.perf_counter() - start

        # 3. Transformer 層
        start = time.perf_counter()
        hidden = self.model.transformer(embeds)
        times["transformer"] = time.perf_counter() - start

        # 4. 解碼
        start = time.perf_counter()
        output = self.model.decode(hidden)
        times["decode"] = time.perf_counter() - start

        # 計算百分比
        total = sum(times.values())
        percentages = {k: v/total*100 for k, v in times.items()}

        # 識別瓶頸
        bottleneck = max(percentages, key=percentages.get)

        return {
            "times_ms": {k: v*1000 for k, v in times.items()},
            "percentages": percentages,
            "bottleneck": bottleneck
        }
```

---

## 5. 基準測試套件

### 完整基準

```python
class LLMBenchmark:
    """LLM 完整基準測試"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_full_benchmark(self):
        """運行完整基準"""

        results = {}

        # 1. 延遲測試
        results["latency"] = self._benchmark_latency()

        # 2. 吞吐量測試
        results["throughput"] = self._benchmark_throughput()

        # 3. 內存測試
        results["memory"] = self._benchmark_memory()

        # 4. 長上下文測試
        results["long_context"] = self._benchmark_long_context()

        return results

    def _benchmark_latency(self):
        """延遲基準"""

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            self._generate("Test prompt")
            latencies.append(time.perf_counter() - start)

        import numpy as np
        return {
            "p50_ms": np.percentile(latencies, 50) * 1000,
            "p95_ms": np.percentile(latencies, 95) * 1000,
            "p99_ms": np.percentile(latencies, 99) * 1000
        }

    def _benchmark_throughput(self):
        """吞吐量基準"""
        # ...
        pass

    def _benchmark_memory(self):
        """內存基準"""
        import torch
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3
        }

    def _benchmark_long_context(self):
        """長上下文基準"""
        # ...
        pass
```

---

## 6. 監控儀表板

### 關鍵指標

```
Performance Metrics:

1. Latency
   - Time to First Token (TTFT)
   - Time Per Output Token (TPOT)
   - Total End-to-End Latency

2. Throughput
   - Requests per Second
   - Tokens per Second
   - Concurrent Users

3. Resource Utilization
   - GPU Utilization %
   - GPU Memory Used
   - CPU Utilization %

4. Cost
   - Cost per 1K Input Tokens
   - Cost per 1K Output Tokens
   - Cost per Request
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **Optimization** | 性能優化 |
| **Monitoring** | 持續監控 |
| **Serving** | 部署服務 |

---

## 延伸閱讀

- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight](https://developer.nvidia.com/nsight-systems)
- [MLPerf](https://mlperf.org/)