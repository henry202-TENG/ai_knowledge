# Efficient LLM Models

高效 LLM 模型設計壓縮蒸餾等技術實現模型輕量化的完整指南。

---

## 1. 什麼是？

### 深度定義

**高效 LLM 模型**指透過各種壓縮技術，在保持能力的同時大幅降低推理成本的模型：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    模型壓縮技術全景                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    輸入: 70B 大模型                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                        │
│         ┌────────────────────┼────────────────────┐                 │
│         ↓                    ↓                    ↓                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │  量化       │     │  蒸餾       │     │  剪枝       │          │
│  │  Quantize  │     │  Distill   │     │  Pruning   │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│         │                    │                    │                 │
│         ↓                    ↓                    ↓                  │
│  FP32→INT4 (8x)        70B→7B (10x)         移除50%權重           │
│                              │                                       │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  輸出: 7B 高效模型                                          │   │
│  │  - 記憶體: 4GB (原本 140GB)                                 │   │
│  │  - 速度: 10x 加速                                           │   │
│  │  - 能力: 保留 80-90%                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**核心目標**:
1. **減少記憶體**: 讓大模型能在消費級 GPU 運行
2. **加速推理**: 降低延遲，提高吞吐量
3. **保持能力**: 精度損失最小化

---

## 2. 模型壓縮概述

### 壓縮方法比較

```
方法對比:

1. 量化 (Quantization)
   - 減少權重精度: FP32 → FP16 → INT8 → INT4
   - 壓縮比: 2x → 4x → 8x
   - 速度提升: 1.5x → 4x
   - 精度損失: 0-2%

2. 蒸餾 (Distillation)
   - 大模型 → 小模型
   - 壓縮比: 10x+
   - 速度提升: 10x+
   - 精度保留: 80-95%

3. 剪枝 (Pruning)
   - 移除不重要權重
   - 壓縮比: 2-5x
   - 速度提升: 1.5-3x

4. 知識蒸餾
   - 中間層蒸餾
   - 注意力蒸餾
```

---

## 2. 量化技術

### 訓練後量化 (PTQ)

```python
class PostTrainingQuantization:
    """訓練後量化"""

    @staticmethod
    def dynamic_quantization(model):
        """動態量化"""

        import torch.quantization

        # 動態量化 (權重 + 激活)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.LSTMCell},
            dtype=torch.qint8
        )

        return quantized_model

    @staticmethod
    def static_quantization(model, calibration_data):
        """靜態量化"""

        # 1. 準備模型
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # 2. 準備數據
        torch.quantization.prepare(model, inplace=True)

        # 3. 校準
        with torch.no_grad():
            for data in calibration_data:
                model(data)

        # 4. 轉換
        quantized_model = torch.quantization.convert(model, inplace=False)

        return quantized_model


# 使用 INT4 量化
# 使用 AWQ or GPTQ
```

### GGML 量化

```python
# llama.cpp 量化命令
# 原始: 7B 模型 ~13GB
# Q4_K_M: ~4GB (量化到 4-bit)
# Q5_K_S: ~4.5GB
# Q8_0: ~7GB

# 使用 ctransformers
from ctransformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q4_K_M.gguf",
    model_type="llama"
)
```

---

## 3. 蒸餾技術

### 層蒸餾

```python
class LayerDistillation:
    """層級蒸餾"""

    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student

    def distill_hidden_layers(self, inputs):
        """蒸餾隱藏層"""

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs, output_hidden_states=True)

        student_outputs = self.student(inputs, output_hidden_states=True)

        # 匹配對應層
        teacher_hidden = teacher_outputs.hidden_states
        student_hidden = student_outputs.hidden_states

        loss = 0
        for t_layer, s_layer in zip(teacher_hidden, student_hidden):
            # MSE Loss
            loss += F.mse_loss(s_layer, t_layer)

        return loss

    def distill_attention(self, inputs):
        """蒸餾注意力"""

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs, output_attentions=True)

        student_outputs = self.student(inputs, output_attentions=True)

        # 注意力蒸餾
        loss = 0
        for t_attn, s_attn in zip(
            teacher_outputs.attentions,
            student_outputs.attentions
        ):
            # KL Divergence
            loss += F.kl_div(
                s_attn.log(),
                t_attn,
                reduction='batchmean'
            )

        return loss
```

---

## 4. 剪枝

### 結構化剪枝

```python
class StructuredPruning:
    """結構化剪枝"""

    @staticmethod
    def prune_weights(weights, sparsity=0.5):
        """權重剪枝"""

        # 計算閾值
        threshold = np.percentile(
            np.abs(weights.detach().numpy()),
            sparsity * 100
        )

        # 創建掩碼
        mask = torch.abs(weights) > threshold

        # 應用剪枝
        pruned_weights = weights * mask.float()

        return pruned_weights, mask

    @staticmethod
    def prune_heads(attention, head_importance):
        """注意力頭剪枝"""

        # 根據重要性排序
        sorted_heads = sorted(
            head_importance.items(),
            key=lambda x: x[1]
        )

        # 移除不重要的頭
        num_to_keep = int(len(head_importance) * 0.7)
        heads_to_keep = [h[0] for h in sorted_heads[-num_to_keep:]]

        # 創建掩碼
        head_mask = torch.ones(attention.shape[1])
        for i in range(attention.shape[1]):
            if i not in heads_to_keep:
                head_mask[i] = 0

        return attention * head_mask.unsqueeze(0).unsqueeze(0)
```

---

## 5. 效能測試

### 基準測試

```python
class EfficiencyBenchmark:
    """效率基準測試"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def benchmark_all(self):
        """完整基準測試"""

        return {
            "memory": self.measure_memory(),
            "latency": self.measure_latency(),
            "throughput": self.measure_throughput(),
            "accuracy": self.measure_accuracy()
        }

    def measure_memory(self):
        """內存測量"""

        import torch
        torch.cuda.reset_peak_memory_stats()

        # 加載模型
        mem_before = torch.cuda.memory_allocated()

        # 推理
        _ = self.generate_test()

        mem_after = torch.cuda.memory_allocated()

        return {
            "model_memory_mb": (mem_after - mem_before) / 1024**2,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2
        }

    def measure_latency(self):
        """延遲測量"""

        import time

        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            _ = self.generate_test()
            latencies.append(time.perf_counter() - start)

        import numpy as np
        return {
            "mean_ms": np.mean(latencies) * 1000,
            "p95_ms": np.percentile(latencies, 95) * 1000
        }

    def generate_test(self):
        """測試生成"""
        inputs = self.tokenizer("Hello", return_tensors="pt")
        return self.model.generate(**inputs, max_new_tokens=50)
```

---

## 6. 最佳實踐

### 選擇指南

```
壓縮方法選擇:

1. 需要快速部署
   → 選擇 INT4 量化 (GGML/llama.cpp)

2. 需要保持精度
   → 選擇蒸餾 + INT8

3. 需要特定加速
   → 選擇剪枝 + 量化

4. 資源極度受限
   → 選擇蒸餾後量化 (4-bit)
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **Quantization** | 精度壓縮 |
| **Distillation** | 知識遷移 |
| **Pruning** | 結構簡化 |

---

## 延伸閱讀

- [LLM Compress](https://arxiv.org/abs/2309.04281)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)