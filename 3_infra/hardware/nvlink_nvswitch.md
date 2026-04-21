# NVLink / NVSwitch

NVIDIA 開發的高速 GPU 互連技術，提供比傳統 PCIe 更快的 GPU 之間通訊頻寬，是大規模 AI 訓練的硬體基礎。

---

## 1. 什麼是？

### 簡單範例

```
DGX A100 (8 GPU):

PCIe 連接:
  GPU 0 ── PCIe ── GPU 1 (16 GB/s)
  瓶頸: 頻寬不足，AllReduce 慢

NVLink + NVSwitch 連接:
  GPU 0 ⇄ NVLink ⇄ GPU 1 (600 GB/s)
  任意 GPU 可直接高速通訊
  訓練速度提升 2-3x
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **突破頻寬瓶頸** | GPU 通訊不再受限於 PCIe |
| **支撐大規模訓練** | 多節點訓練的關鍵技術 |
| **降低延遲** | P2P 通訊延遲極低 |
| **必備基礎** | Tensor/Parallelism 的硬體支撐 |

---

## 3. NVLink

### 技術規格

| 世代 | GPU | 每條 link | 最大 links | 總頻寬 |
|------|-----|-----------|------------|--------|
| **NVLink 1.0** | Pascal | 40 GB/s | 4 | 160 GB/s |
| **NVLink 2.0** | Volta | 50 GB/s | 6 | 300 GB/s |
| **NVLink 3.0** | Ampere | 50 GB/s | 12 | 600 GB/s |
| **NVLink 4.0** | Hopper | 50 GB/s | 18 | 900 GB/s |

### 拓撲結構

```
4-GPU NVLink 拓撲 (A100):

    GPU 0 ── NVLink ── GPU 1
     │                      │
     │        X             │
     │                      │
    GPU 3 ── NVLink ── GPU 2

每個 GPU 有 12 個 NVLink
形成高度互連的網路
```

### 對比 PCIe

| 特性 | NVLink 4.0 | PCIe Gen5 |
|------|-----------|-----------|
| 每 link 頻寬 | 50 GB/s | 32 GB/s |
| 延遲 | ~1 µs | ~1-2 µs |
| 拓撲 | 點對點 | 共享匯流排 |
| CPU 參與 | 可選 | 必需 |

---

## 4. NVSwitch

### 什麼是？

NVSwitch 是構建在 NVLink 基礎上的交換晶片，允許更多 GPU 高頻寬互聯。

### 架構

```
傳統 NVLink (4 GPU):
GPU 0 ─┐
GPU 1 ─┼─ 有限連接
GPU 2 ─┤
GPU 3 ─┘

NVSwitch (8 GPU):

GPU 0 ──┬── NVSwitch 0 ── GPU 4
GPU 1 ──┤              ├── GPU 5
GPU 2 ──┤              ├── GPU 6
GPU 3 ──┘              └── GPU 7

任意 GPU 可直接通訊
```

### 硬體規格

| 世代 | 交換晶片頻寬 | 支援 GPU 數 |
|------|-------------|------------|
| **第一代** | 2.4 TB/s | 8 |
| **第二代** | 4.8 TB/s | 8 |
| **第三代** | 7.2 TB/s | 8+ |

---

## 5. H100/H200 新規格

### H100 NVLink

```
- 每 GPU 18 NVLink links
- 總頻寬: 900 GB/s (雙向)
- NVSwitch: 第三代
- 支援 NVLink Network (節點互聯)
```

### H200

```
- 與 H100 相同的 NVLink
- 升級: 141GB HBM3e 記憶體
- 更適合大模型 Inference
```

---

## 6. 對 AI 訓練的影響

### 應用場景

| 場景 | 效益 |
|------|------|
| **Multi-GPU Training** | 8-GPU 以上訓練必備 |
| **Tensor Parallelism** | 需要高頻寬跨 GPU 通訊 |
| **AllReduce** | 集合通訊的關鍵路徑 |
| **RDMA 準備** | NVSwitch + RDMA 實現節點互聯 |

### 效能影響

```
訓練時間比較 (64 A100s, 175B 模型):

PCIe:          ████████████████████████████ 100%
NVLink + Switch: ████████████████ 70%

頻寬提升帶來:
- Tensor Parallelism 效率提升 30-50%
- 梯度同步時間減少 50%
- 整體訓練速度提升 30%
```

---

## 7. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Tensor Parallelism** | 依賴 NVLink 高頻寬 |
| **RDMA** | NVSwitch + RDMA 實現節點互聯 |
| **AllReduce** | 通訊瓶頸受益於 NVLink |
| **InfiniBand** | 與 NVLink 互補的網路技術 |

---

## 8. 數學分析

### 頻寬計算

```
NVLink 頻寬公式:

單-link 頻寬 = 50 GB/s (雙向)
             = 25 GB/s (單向)

8-link 總頻寬 = 50 × 8 = 400 GB/s (雙向)

範例:
  H100 有 18 links
  理論峰值 = 50 × 18 = 900 GB/s (雙向)
  
實際頻寬 (考慮協議開銷):
  實際 ≈ 理論 × 0.9 ≈ 810 GB/s
```

### 延遲分析

```
延遲組成:

1. 傳輸延遲 (L1):
   - 短封包: ~100ns
   - 長封包: ~500ns

2. 軟體堆疊:
   - CUDA IPC: ~1-2μs
   - NCCL: ~2-5μs

3. GPU Core 處理:
   - 記憶體訪問: ~100ns
   - 計算: ~10ns per FP32

NVLink vs PCIe 延遲比較:
  | 操作 | NVLink | PCIe | 加速 |
  |------|--------|------|------|
  | 1KB 傳輸 | 0.5μs | 5μs | 10x |
  | 1MB 傳輸 | 20μs | 100μs | 5x |
  | AllReduce 8 GPU | 15μs | 150μs | 10x |
```

---

## 9. NVSwitch 架構深入

### 架構圖

```
第一代 NVSwitch (Tesla V100 / DGX-2):

           ┌─────────────────────────────────────────┐
           │              NVSwitch 晶片                │
           │  18 ports, 100 GB/s each                 │
           └──────────────────┬────────────────────────┘
                              │
    ┌──────────┬───────────────┼───────────────┬──────────┐
    ▼          ▼               ▼               ▼          ▼
  GPU 0      GPU 1           GPU 2           GPU 3      GPU 4
  (HBM)      (HBM)           (HBM)           (HBM)      (HBM)
  
  全連接拓撲: 任意 GPU 可直接通訊
```

### 交換架構

```python
class NVSwitchRouting:
    """NVSwitch 路由邏輯"""

    def __init__(self, num_gpus=8):
        self.num_gpus = num_gpus

    def compute_route(self, src_gpu, dst_gpu):
        """
        計算最佳路由路徑
        """

        # 單節點內: 直接傳輸
        if self._is_same_node(src_gpu, dst_gpu):
            return self._direct_path(src_gpu, dst_gpu)

        # 跨節點: 透過網路
        return self._network_path(src_gpu, dst_gpu)

    def _direct_path(self, src, dst):
        """直接 NVLink 路徑"""
        return {
            "type": "nvlink",
            "path": [src, dst],
            "bandwidth": "400 GB/s"
        }
```

---

## 10. 效能優化

### 通訊優化

```python
class NVLinkTuner:
    """NVLink 效能調優"""

    @staticmethod
    def optimize_buffer_sizes():
        """優化緩衝區大小"""

        # 增大 NCCL buffer 提高頻寬
        import os
        os.environ['NCCL_BUFFSIZE'] = '33554432'  # 32MB

        # 啟用 CUDA IPC
        os.environ['CUDA_VISIBLE_DEVICES'] = 'all'

    @staticmethod
    def enable_gpu_direct():
        """啟用 GPU Direct RDMA"""

        # 確認 GPU Direct 支援
        import torch.cuda

        if torch.cuda.is_available():
            # 自動使用 GPU Direct
            pass
```

### 拓撲優化

```bash
# 檢查 NVLink 拓撲
nvidia-smi topo -m

# 輸出範例:
#        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity
# GPU0   X     NV1   NV1   NV1   NV2   NV2   NV2   NV2  0-39
# GPU1   NV1   X     NV1   NV1   NV2   NV2   NV2   NV2  0-39
# GPU2   NV1   NV1   X     NV1   NV2   NV2   NV2   NV2  0-39
# ...

# NV1: 1 NVLink
# NV2: 2 NVLinks
```

---

## 11. 在分散式訓練中的應用

### AllReduce 優化

```python
import torch.distributed as nccl

def optimized_allreduce(tensor, group):
    """使用 NVLink 優化的 AllReduce"""

    # 確保使用 GPU Direct
    torch.cuda.set_device(tensor.get_device())

    # NCCL 自動使用 NVLink
    dist.all_reduce(tensor, op=nccl.ReduceOp.SUM, group=group)

def benchmark_nvlink_allreduce(size_mb=100, num_iterations=100):
    """基準測試 NVLink AllReduce"""

    # 建立測試張量
    size_bytes = size_mb * 1024 * 1024
    num_elements = size_bytes // 4  # FP32
    tensor = torch.randn(num_elements, device='cuda')

    # 預熱
    for _ in range(10):
        torch.distributed.all_reduce(tensor)

    # 基準測試
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        torch.distributed.all_reduce(tensor)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    bandwidth_gbps = (size_mb * num_iterations * 1000) / elapsed_ms

    return {
        "elapsed_ms": elapsed_ms,
        "bandwidth_gbps": bandwidth_gbps,
        "efficiency": bandwidth_gbps / 900  # 相對於理論峰值
    }
```

### 集合通訊模式

```
AllReduce 模式 (8 GPU, NVSwitch):

GPU 0: [Data 0] ─┐
GPU 1: [Data 1] ─┼─→ AllReduce → [Sum 0-7]
GPU 2: [Data 2] ─┤
...              │
GPU 7: [Data 7] ─┘

NVLink 優勢:
- 硬體支援 reduce
- 點對點傳輸
- 延遲極低
```

---

## 12. 與其他互聯技術比較

| 特性 | NVLink 4 | PCIe Gen5 | InfiniBand HDR |
|------|-----------|-----------|----------------|
| 每 link 頻寬 | 50 GB/s | 32 GB/s | 50 GB/s |
| 延遲 | ~1 μs | ~1-2 μs | ~0.6 μs |
| 拓撲 | 點對點 | 共享匯流排 | 交換式 |
| CPU 參與 | 可選 | 必需 | 可選 |
| 成本 | 高 | 低 | 中 |

### 混合架構

```
典型資料中心配置:

節點內: NVLink/NVSwitch
節點間: InfiniBand 或 RoCE

         節點 1                    節點 2
    ┌──────────────┐          ┌──────────────┐
    │ GPU 0 ─ NVLink ─ GPU 1 │ GPU 4 ─ NVLink ─ GPU 5 │
    │       ╲    ╱          │       ╲    ╱          │
    │        ╲  ╱           │        ╲  ╱           │
    │       NVSwitch         │       NVSwitch        │
    │        ╱  ╲           │        ╲  ╱           │
    │       ╱    ╲          │       ╱    ╲          │
    │ GPU 2 ─ NVLink ─ GPU 3 │ GPU 6 ─ NVLink ─ GPU 7 │
    └──────────────┘          └──────────────┘
            │  RDMA                  │  RDMA
            └──────────┬─────────────┘
                       ▼
                   Switch
```

---

## 13. 故障診斷

### 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **頻寬低於預期** | NVLink 未全速運行 | 檢查拓撲和驅動版本 |
| **通訊超時** | NCCL 緩衝區不足 | 增大緩衝區大小 |
| **GPU 看不到彼此** | NVLink 連接問題 | 檢查硬體連接 |
| **效能不穩定** | 溫度過高 | 檢查散熱 |

### 診斷命令

```bash
# 檢查 NVLink 狀態
nvidia-smi nvlink -s

# 檢查 GPU 連接
nvidia-smi topo -m

# 監控 NVLink 流量
nvidia-smi dmon -s u

# 錯誤日誌
dmesg | grep -i nvlink
```

---

## 14. 相關主題

| 技術 | 關係 |
|------|------|
| **Tensor Parallelism** | 依賴 NVLink 高頻寬 |
| **RDMA** | NVSwitch + RDMA 實現節點互聯 |
| **AllReduce** | 通訊瓶頸受益於 NVLink |
| **InfiniBand** | 與 NVLink 互補的網路技術 |

---

## 延伸閱讀

- [NVLink/NVSwitch 白皮書](https://resources.nvidia.com/en-us-tensor-core)
- [Hopper Architecture](https://resources.nvidia.com/en-us-hopper-architecture)
- [DGX Systems](https://www.nvidia.com/en-us/data-center/dgx-systems/)
- [NVLink Technology](https://www.nvidia.com/en-us/data-center/nvlink/)