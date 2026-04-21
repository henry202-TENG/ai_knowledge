# Distributed Training

大型語言模型的分散式訓練技術，支援數十億參數模型的訓練。

---

## 1. 什麼是？

### 深度定義

**Distributed Training (分散式訓練)** 是將**大型模型**分散到**多個硬體設備**進行訓練的技術：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    分散式訓練架構                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  單機 vs 分散式:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  單機訓練:                                                    │   │
│  │                                                              │   │
│  │         ┌──────────────┐                                     │   │
│  │         │    GPU 0     │                                     │   │
│  │         │  完整模型    │                                     │   │
│  │         │  70B params  │                                     │   │
│  │         └──────────────┘                                     │   │
│  │         需要 140GB+ VRAM                                     │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  分散式訓練:                                                  │   │
│  │                                                              │   │
│  │   GPU 0      GPU 1      GPU 2      GPU 3                     │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                     │   │
│  │  │層0-20│  │層21-40│  │層41-60│  │層61-80│                    │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘                     │   │
│  │       ↓        ↓        ↓        ↓                            │   │
│  │   [通訊層: NCCL/RDMA]                                        │   │
│  │                                                              │   │
│  │   記憶體需求: 70B/4 = 17.5GB per GPU                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  並行策略:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  數據並行 (Data Parallel):                                  │   │
│  │    每個 GPU 有完整模型，處理不同數據                          │   │
│  │    優化器狀態複製，梯度 All-Reduce                           │   │
│  │    適用: 數據量大，模型可放入單卡                             │   │
│  │                                                              │   │
│  │  模型並行 (Model Parallel):                                  │   │
│  │    模型分割到多卡，每卡負責部分層                            │   │
│  │    需跨設備傳遞激活值                                         │   │
│  │    適用: 超大模型，無法放入單卡                               │   │
│  │                                                              │   │
│  │  流水線並行 (Pipeline Parallel):                            │   │
│  │    將模型分為多個 stage，流水線處理                          │   │
│  │    減少設備空閒時間                                          │   │
│  │    減少通訊開銷                                              │   │
│  │                                                              │   │
│  │  張量並行 (Tensor Parallel):                                 │   │
│  │    單層內部並行 (如 Attention 線性層)                        │   │
│  │    需要 NCCL 高速互聯                                       │   │
│  │                                                              │   │
│  │  ZeRO:                                                       │   │
│  │    分片存儲: 優化器狀態/梯度/參數分片                        │   │
│  │    減少記憶體需求至 O(1)                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  通訊優化:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  NCCL: NVIDIA 集合通訊庫，高頻寬                             │   │
│  │  RDMA: 遠程直接內存訪問，極低延遲                           │   │
│  │  Gradient Compression: 梯度壓縮減少通訊量                   │   │
│  │  Mixed Precision: FP16/BF16 減少記憶體和通訊              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  核心挑戰:                                                           │
│  1. 設備同步: 多設備狀態一致性                                      │
│  2. 通訊瓶頸: 梯度同步延遲                                          │
│  3. 負載均衡: 確保各設備工作量均衡                                   │
│  4. 故障恢復: 單點故障不會中斷整個訓練                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何重要**:
1. **突破記憶體限制**: 訓練超出單卡容量的模型
2. **加速訓練**: 並行處理減少訓練時間
3. **成本效益**: 使用多個小型 GPU 而非巨型 GPU
4. **可擴展性**: 隨數據和模型增長線性擴展

### 簡單範例

```
單機訓練:
  1 GPU → 70B 模型 → 需要 140GB 記憶體
  不可行

分散式訓練:
  8 GPU (每張 80GB) → 分散模型 → 可訓練
  速度: 8x 加速
```

---

## 2. 訓練架構

### 數據並行

```python
class DataParallel:
    """數據並行 - 每個 GPU 有完整模型"""

    def __init__(self, model, devices):
        self.model = model
        self.devices = devices

        # 複製模型到每個設備
        self.replicas = [model.to(d) for d in devices]

    def train_step(self, batch):
        """數據並行訓練"""

        # 分割數據
        batches = self._split_batch(batch)

        # 每個設備前向傳播
        losses = []
        for device, batch_slice in zip(self.devices, batches):
            with torch.device(device):
                output = self.replicas[device](batch_slice)
                losses.append(output.loss)

        # 計算平均梯度
        avg_loss = sum(losses) / len(losses)

        # 反向傳播
        avg_loss.backward()

        # 同步參數
        self._sync_parameters()

        return avg_loss

    def _sync_parameters(self):
        """同步參數"""
        # All-Reduce
        for param in self.model.parameters():
            torch.distributed.all_reduce(param.grad)
            param.grad /= len(self.devices)
```

### 模型並行

```python
class ModelParallel:
    """模型並行 - 分割模型到多個 GPU"""

    def __init__(self, model, devices):
        self.devices = devices

        # 分割模型
        self.layers = model.layers

        # 分配層
        self.layer分配 = self._distribute_layers(
            self.layers,
            len(devices)
        )

    def _distribute_layers(self, layers, num_devices):
        """分配層到設備"""

        layers_per_device = len(layers) // num_devices

        allocation = {}
        for i, device in enumerate(self.devices):
            start = i * layers_per_device
            end = start + layers_per_device if i < num_devices - 1 else len(layers)
            allocation[device] = layers[start:end]

        return allocation

    def forward(self, x):
        """跨設備前向傳播"""

        current = x

        for device, layers in self.layer_allocation.items():
            with torch.device(device):
                for layer in layers:
                    current = layer(current)

        return current

    def backward(self, grad):
        """跨設備反向傳播"""
        # 反向遍歷
        pass
```

### 流水線並行

```python
class PipelineParallel:
    """流水線並行"""

    def __init__(self, model, devices, num_stages=4):
        self.devices = devices
        self.num_stages = num_stages

        # 分割模型成 stages
        self.stages = self._create_stages(model, num_stages)

    def _create_stages(self, model, num_stages):
        """創建 stages"""

        layers = list(model.children())
        stage_size = len(layers) // num_stages

        stages = []
        for i in range(num_stages):
            start = i * stage_size
            end = start + stage_size if i < num_stages - 1 else len(layers)

            stage = nn.Sequential(*layers[start:end])
            stage = stage.to(self.devices[i])

            stages.append(stage)

        return stages

    def forward_backward(self, batch):
        """流水線前向和反向"""

        # 前向傳播 (micro-batches)
        outputs = []
        for micro_batch in self._split_micro(batch):
            x = micro_batch

            for stage in self.stages:
                x = stage(x)

            outputs.append(x)

        # 反向傳播
        # ...
```

---

## 3. 通訊優化

### 通訊原語

```python
class NCCLCommunicator:
    """NCCL 通訊"""

    def __init__(self):
        self.rank = torch.distributed.get_rank()
        self.size = torch.distributed.get_world_size()

    def all_reduce(self, tensor):
        """All-Reduce"""

        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM
        )

    def all_gather(self, tensor):
        """All-Gather"""

        tensor_list = [torch.zeros_like(tensor) for _ in range(self.size)]

        torch.distributed.all_gather(
            tensor_list,
            tensor
        )

        return tensor_list

    def broadcast(self, tensor, src=0):
        """Broadcast"""

        torch.distributed.broadcast(
            tensor,
            src=src
        )
```

### Gradient Compression

```python
class GradientCompression:
    """梯度壓縮"""

    @staticmethod
    def topk_compress(grad, compression_ratio=0.01):
        """Top-K 壓縮"""

        # 只保留最大的 k% 元素
        k = int(grad.numel() * compression_ratio)

        _, indices = torch.topk(
            torch.abs(grad).flatten(),
            k
        )

        # 壓縮
        compressed = {
            "values": grad.flatten()[indices],
            "indices": indices,
            "shape": grad.shape
        }

        # 解壓縮
        decompressed = torch.zeros(grad.numel())
        decompressed[indices] = compressed["values"]

        return decompressed.view(grad.shape)

    @staticmethod
    def quantization_compress(grad, bits=8):
        """量化壓縮"""

        # 量化梯度
        max_val = grad.abs().max()
        scale = (2 ** bits - 1) / max_val

        quantized = (grad * scale).round().to(torch.uint8)

        # 解量化
        decompressed = quantized.float() / scale

        return decompressed
```

---

## 4. 優化器

### 分片優化器

```python
class ShardedOptimizer:
    """分片優化器 - ZeRO"""

    def __init__(self, model, devices):
        self.model = model
        self.devices = devices
        self.world_size = len(devices)

        # 參數分片
        self.param_shards = self._shard_parameters(model)

    def _shard_parameters(self, model):
        """分片參數"""

        all_params = list(model.parameters())
        shard_size = len(all_params) // self.world_size

        shards = []
        for i in range(self.world_size):
            start = i * shard_size
            end = start + shard_size if i < self.world_size - 1 else len(all_params)
            shards.append(all_params[start:end])

        return shards

    def step(self):
        """優化器步驟"""

        my_shard = self.param_shards[self.rank]

        for param in my_shard:
            # 本地更新
            param -= self.lr * param.grad

        # 同步
        self._sync_across_shards()

    def _sync_across_shards(self):
        """跨分片同步"""
        # All-Gather 參數片段
        pass
```

---

## 5. 混合精度

```python
class MixedPrecisionTrainer:
    """混合精度訓練"""

    def __init__(self, model):
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, batch):
        """混合精度訓練步驟"""

        with torch.cuda.amp.autocast():
            output = self.model(batch["input"])
            loss = F.cross_entropy(
                output,
                batch["labels"]
            )

        # 縮放損失，執行反向傳播
        self.scaler.scale(loss).backward()

        # 反縮放梯度
        self.scaler.unscale_(self.optimizer)

        # 梯度裁剪
        self.scaler.clip_grad_norm_(
            self.model.parameters(),
            1.0
        )

        # 更新參數
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

---

## 6. 檢查點

```python
class DistributedCheckpoint:
    """分散式檢查點"""

    def save_checkpoint(self, model, optimizer, path):
        """保存檢查點"""

        # 只保存本地分片
        checkpoint = {
            "model_state": self._get_local_state(model),
            "optimizer_state": self._get_local_state(optimizer),
            "epoch": self.epoch
        }

        torch.save(checkpoint, f"{path}/rank_{self.rank}.pt")

        # 同步
        torch.distributed.barrier()

    def load_checkpoint(self, model, optimizer, path):
        """加載檢查點"""

        # 從對應分片加載
        checkpoint = torch.load(f"{path}/rank_{self.rank}.pt")

        self._set_local_state(model, checkpoint["model_state"])
        self._set_local_state(optimizer, checkpoint["optimizer_state"])
```

---

## 7. 集群調度

### Job 提交

```bash
# Slurm 提交
#!/bin/bash
#SBATCH --job-name=llm-training
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

srun python train.py \
    --nnodes=8 \
    --nproc_per_node=8 \
    --master_addr=$SLURM_JOB_NODELIST
```

### Kubernetes Training

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training
spec:
  parallelism: 8
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0-cuda11.7
        command:
        - torchrun
        - --nproc_per_node=8
        - --nnodes=8
        - train.py
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: "256Gi"
```

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **Tensor Parallelism** | 模型並行 |
| **Pipeline Parallelism** | 流水線並行 |
| **Gradient Checkpointing** | 記憶體優化 |

---

## 延伸閱讀

- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch DDP](https://pytorch.org/tutorials/)