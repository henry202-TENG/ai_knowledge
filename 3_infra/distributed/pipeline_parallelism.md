# Pipeline Parallelism

將模型的不同層分配到不同的 GPU 上，讓資料像工廠管線一樣在各個 stage 間流動。

---

## 1. 什麼是？

### 簡單範例

```
模型: L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8

4 GPU Pipeline:

GPU 0: [L1, L2] ──┐
GPU 1: [L3, L4] ──┼──→ 資料流
GPU 2: [L5, L6] ──┤
GPU 3: [L7, L8] ──┘

輸入: "今天天氣很好"
  ↓ GPU 0 → L1 → L2 → 傳遞
  ↓ GPU 1 → L3 → L4 → 傳遞
  ↓ GPU 2 → L5 → L6 → 傳遞
  ↓ GPU 3 → L7 → L8 → 輸出
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **支援超大模型** | 單 GPU 放不下的模型 |
| **簡單直觀** | 按層切分，邏輯清晰 |
| **節點擴展** | 輕鬆擴展到多節點 |
| **通訊較少** | 只在相鄰 stage 間傳遞 |

---

## 3. 核心原理

### Pipeline Bubbles 問題

```
時間 →

GPU 0: [F0][F1][F2][F3][F4][F5]
GPU 1:     [W0][W1][W2][W3][W4][W5]
GPU 2:         [W0][W1][W2][W3][W4][W5]
GPU 3:             [B0][B1][B2][B3][B4][B5]

[F: Forward, W: Wait, B: Backward]

問題：每個 GPU 有大量空閒時間 (bubbles)
```

### 解決方案：Micro-batching

```python
# GPipe: 將 batch 切成 micro-batches

batch_size = 16
micro_batch_size = 4
num_microbatches = 4

# 資料流:
# micro_batch 1 → GPU 0 → GPU 1 → GPU 2 → GPU 3
# micro_batch 2 → GPU 0 → GPU 1 → GPU 2 → GPU 3
# micro_batch 3 → ...
# micro_batch 4 → ...

# Bubble 大幅減少
```

### Pipeline Schedule 比較

| Schedule | 特色 | 優點 | 缺點 |
|----------|------|------|------|
| **Forward only** | 簡單 | 實現容易 | Memory 高 |
| **1F1B** | 交替 F/B | 記憶體低 | 控制複雜 |
| **Interleaved** | 多 stage F/B | Bubble 小 | 通訊多 |

### 1F1B 實現

```python
def schedule_1f1b(model, microbatches, num_stages):
    """1F1B 排程"""
    forward_microbatches = microbatches[:num_microbatches//2]
    backward_microbatches = microbatches[num_microbatches//2:]

    # 前半段: 全部 Forward
    for mb in forward_microbatches:
        stage_forward(mb)

    # 後半段: 交替 Forward/Backward
    for mb in backward_microbatches:
        stage_forward(mb)
        stage_backward(mb)

    # 最後Backward
    for mb in reversed(forward_microbatches):
        stage_backward(mb)
```

### Bubble 效率計算

```
原始實現 Bubble: (P-1)/P (P = stages 數量)
micro-batch 後 Bubble: (P-1)/(P+M) (M = micro-batches 數量)

例如 P=4, M=8:
  原始: 75%
  Micro-batch: 27%
```

---

## 4. 常見問題與解決

### 梯度同步問題

```python
# 不同 stage 的梯度需要同步
# 使用 all_reduce 跨 pipeline 同步
def sync_grads_across_pipelines(model):
    for param in model.parameters():
        dist.all_reduce(param.grad)
```

### 記憶體優化

```
問題: 每個 stage 需要保存所有 micro-batch 的 activation

解決:
  1. Gradient Checkpointing: 只保存部分
  2. Offload: 將暫時不需要的移到 CPU
  3. 減小 micro-batch size
```

---

## 5. 知名實現

| 框架 | 說明 |
|------|------|
| **PyTorch DDP** | 原生 Pipeline 支援 |
| **Megatron-LM** | 優化的 PP 實現 |
| **DeepSpeed** | 3D Parallelism |
| **Fairscale** | PyTorch PP wrapper |

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Tensor Parallelism** | 常與 PP 結合 (3D Parallelism) |
| **GPipe** | 經典 PP 論文 |
| **1F1B Schedule** | 常用排程算法 |
| **Gradient Accumulation** | 增大 effective batch size |

---

## 7. 數學分析

### Bubble 效率公式

```
定義:
  P = pipeline stages 數量
  M = micro-batches 數量
  F = 每個 micro-batch 的 forward 時間
  B = 每個 micro-batch 的 backward 時間

標準 GPipe Schedule:
  總時間 = P × F + M × (F + B) + P × B
         ≈ M × (F + B) + 2P × F  (忽略 B 相對於 F)

Bubble 時間:
  Bubble = 2P × F

效率:
  Efficiency = 實際計算 / 總時間
             = M × (F + B) / (M × (F + B) + 2P × F)
             ≈ 1 / (1 + 2P/M)

例如:
  P=8, M=32: Efficiency ≈ 1/(1+0.5) = 67%
  P=8, M=128: Efficiency ≈ 1/(1+0.125) = 89%
```

### 記憶體分析

```
每個 Stage 的記憶體需求:

Activation Memory:
  M_act = micro_batch_size × seq_len × hidden_size × num_layers × 4 bytes

Gradient Memory:
  M_grad = M_act × 1 (大約相等)

Parameter Memory:
  M_param = hidden_size² × num_layers × 4 bytes

Total ≈ M_act + M_grad + M_param
```

---

## 8. 進階排程算法

### Interleaved 1F1B

```python
def schedule_interleaved_1f1b(
    model_stages,
    microbatches,
    num_warmup=3,
    num_cooldown=3
):
    """
    Interleaved 1F1B: 進一步減少 bubble
    """

    num_mb = len(microbatches)
    num_stages = len(model_stages)

    # Warmup 階段
    for i in range(num_warmup):
        stage_id = i % num_stages
        forward(microbatches[i], model_stages[stage_id])

    # 交替階段
    forward_idx = num_warmup
    backward_idx = num_mb - num_cooldown

    while forward_idx < num_mb and backward_idx >= 0:
        # Forward
        stage_id = forward_idx % num_stages
        forward(microbatches[forward_idx], model_stages[stage_id])
        forward_idx += 1

        # Backward
        stage_id = backward_idx % num_stages
        backward(microbatches[backward_idx], model_stages[stage_id])
        backward_idx -= 1

    # Cooldown 階段
    for i in range(num_cooldown):
        stage_id = (num_warmup + i) % num_stages
        backward(microbatches[backward_idx - i], model_stages[stage_id])
```

### Virtual Pipeline Parallelism

```python
class VirtualPipelineParallelism:
    """
    將每個 physical stage 進一步劃分為多個 virtual stages
    減少 bubble 但增加通訊
    """

    def __init__(self, num_layers, num_gpus, num_microbatches):
        self.num_layers = num_layers
        self.num_gpus = num_gpus
        self.num_microbatches = num_microbatches

        # 每個 GPU 分配的 virtual stages
        self.virtual_stages_per_gpu = num_layers // num_gpus

    def partition_model(self):
        """分配模型到 GPU"""
        partitions = []

        for gpu_id in range(self.num_gpus):
            start_layer = gpu_id * self.virtual_stages_per_gpu
            end_layer = start_layer + self.virtual_stages_per_gpu

            partitions.append({
                "gpu": gpu_id,
                "layers": list(range(start_layer, end_layer))
            })

        return partitions
```

---

## 9. 梯度同步

### 跨 Stage 梯度同步

```python
import torch.distributed as dist

class PipelineGradientSync:
    def __init__(self, pipeline_group):
        self.pipeline_group = pipeline_group
        self.world_size = dist.get_world_size(pipeline_group)

    def sync_gradients(self, model):
        """
        同步所有 pipeline stage 的梯度
        """

        # 遍歷所有參數
        for name, param in model.named_parameters():
            if param.grad is not None:
                # All-reduce 梯度
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.pipeline_group
                )

                # 平均
                param.grad /= self.world_size

    def async_sync(self, param):
        """異步同步"""
        handle = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            group=self.pipeline_group,
            async_op=True
        )
        return handle
```

### 分片梯度同步

```python
class ChunkedGradientSync:
    """分片同步以減少記憶體"""

    def __init__(self, chunk_size=1024 * 1024):
        self.chunk_size = chunk_size

    def sync_in_chunks(self, param):
        """分塊同步大參數"""
        grad_data = param.grad
        numel = grad_data.numel()

        handles = []
        for start in range(0, numel, self.chunk_size):
            end = min(start + self.chunk_size, numel)
            chunk = grad_data.view(-1)[start:end]

            handle = dist.all_reduce(
                chunk,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            handles.append(handle)

        # 等待所有同步完成
        for handle in handles:
            handle.wait()
```

---

## 10. 與其他並行技術結合

### 3D Parallelism

```python
class ThreeDimensionalParallelism:
    """
    Tensor Parallelism × Pipeline Parallelism × Data Parallelism
    """

    def __init__(self, tp_size, pp_size, dp_size):
        self.tp_size = tp_size    # Tensor Parallelism
        self.pp_size = pp_size    # Pipeline Parallelism
        self.dp_size = dp_size    # Data Parallelism

        # 總 GPU 數 = tp × pp × dp
        self.world_size = tp_size * pp_size * dp_size

    def get_rank_coordinates(self, rank):
        """根據 rank 計算 (tp, pp, dp) 坐標"""
        dp = rank % self.dp_size
        pp = (rank // self.dp_size) % self.pp_size
        tp = rank // (self.dp_size * self.pp_size)

        return tp, pp, dp

    def partition_model(self, model):
        """3D 分區模型"""

        # 首先按 Pipeline Stage 分割
        layers_per_stage = model.num_layers // self.pp_size

        # 然後每個 Stage 內按 Tensor Parallelism 分割
        # 每個 TP group 處理 layers_per_stage / tp_size

        return {
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "dp_rank": self.dp_rank,
            "layers": assigned_layers
        }
```

### PP + TP 組合效能

```
配置:
  - 8 GPUs
  - 模型: 80 層, 175B 參數

方案 1: 純 PP
  - 8 stages, 每 stage 10 層
  - Bubble: ~12%

方案 2: PP + TP
  - 2 TP groups × 4 PP stages
  - 每個 TP group 內 2-way Tensor Parallelism
  - 每個 stage 5 層 × 2
  - Bubble: ~8%
  - 通訊增加
```

---

## 11. 故障處理

### Checkpoint 恢復

```python
class PipelineCheckpointManager:
    """Pipeline 檢查點管理"""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, model, optimizer, epoch, step):
        """保存檢查點"""

        # 每個 rank 保存自己的部分
        rank = dist.get_rank()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        path = f"{self.checkpoint_dir}/ckpt_rank{rank}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, model, optimizer):
        """載入檢查點"""

        rank = dist.get_rank()
        path = f"{self.checkpoint_dir}/ckpt_rank{rank}.pt"

        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["epoch"], checkpoint["step"]
```

### 節點故障處理

```python
class PipelineFaultTolerance:
    """Pipeline 容錯機制"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def handle_stage_failure(self, failed_rank, model_stages):
        """
        處理 stage 故障

        策略:
        1. 重新初始化失敗的 stage
        2. 從最近的 checkpoint 恢復
        3. 重新分配數據
        """

        for attempt in range(self.max_retries):
            try:
                # 重新初始化
                model_stages[failed_rank].reset()

                # 從 checkpoint 恢復
                self.restore_from_checkpoint(failed_rank)

                # 測試連接
                self.test_pipeline_connectivity()

                return True

            except Exception as e:
                logger.warning(f"Recovery attempt {attempt} failed: {e}")
                continue

        return False
```

---

## 12. 監控與調優

### 關鍵指標

```python
class PipelineMonitor:
    def __init__(self):
        self.metrics = {
            "forward_time": [],
            "backward_time": [],
            "comm_time": [],
            "bubble_time": []
        }

    def record_step(self, forward_time, backward_time, comm_time):
        """記錄每步的時間"""

        self.metrics["forward_time"].append(forward_time)
        self.metrics["backward_time"].append(backward_time)
        self.metrics["comm_time"].append(comm_time)

        # 計算 bubble 時間
        total_time = forward_time + backward_time + comm_time
        ideal_time = forward_time + backward_time
        bubble = max(0, total_time - ideal_time)
        self.metrics["bubble_time"].append(bubble)

    def get_efficiency(self):
        """計算 pipeline 效率"""

        avg_total = np.mean([
            f + b + c for f, b, c in zip(
                self.metrics["forward_time"],
                self.metrics["backward_time"],
                self.metrics["comm_time"]
            )
        ])

        avg_useful = np.mean([
            f + b for f, b in zip(
                self.metrics["forward_time"],
                self.metrics["backward_time"]
            )
        ])

        return avg_useful / max(avg_total, 1e-6)
```

### 調優參數

| 參數 | 建議值 | 影響 |
|------|--------|------|
| **num_microbatches** | 4-16 | 太小:bubble大，太大:記憶體高 |
| **gradient_accumulation** | 視 GPU 記憶體 | 增大 effective batch |
| **async_forward** | True | 隱藏通訊延遲 |
| **partition_method** | "uniform" | 均勻分區 |

---

## 13. 相關主題

| 技術 | 關係 |
|------|------|
| **Tensor Parallelism** | 常與 PP 結合 (3D Parallelism) |
| **GPipe** | 經典 PP 論文 |
| **1F1B Schedule** | 常用排程算法 |
| **Gradient Accumulation** | 增大 effective batch size |

---

## 延伸閱讀

- [GPipe Paper](https://arxiv.org/abs/1811.06965)
- [PipeDream Paper](https://arxiv.org/abs/1907.13257)
- [Megatron-LM PP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)
- [PipeDream 2DP](https://arxiv.org/abs/2104.05273)
- [DAPPRE](https://arxiv.org/abs/2205.00398)