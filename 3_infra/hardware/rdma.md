# RDMA

繞過 CPU 直接在兩台伺服器的記憶體之間傳輸資料的網路技術，讓 GPU 可以直接讀寫另一台伺服器的記憶體，實現超低延遲的高效能運算。

---

## 1. 什麼是？

### 深度定義

**RDMA (Remote Direct Memory Access)** 是一種讓資料傳輸**繞過作業系統**的網路技術：

```
┌─────────────────────────────────────────────────────────────────────┐
│                 傳統 TCP/IP vs RDMA 傳輸路徑                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  傳統 TCP/IP:                                                        │
│                                                                      │
│  [應用] → [系統緩衝區] → [CPU處理] → [網卡驅動] → [網卡] → [網路]    │
│                                                                      │
│  步驟:                                                               │
│  1. 資料從網卡到達                                                    │
│  2. CPU 中斷處理                                                     │
│  3. CPU 複製資料到用戶空間                                            │
│  4. CPU 處理協議棧                                                   │
│  5. CPU 複製資料到目標緩衝區                                          │
│                                                                      │
│  問題:                                                               │
│  - CPU 每次傳輸都要參與                                              │
│  - 多次記憶體拷貝                                                    │
│  - 延遲: 數十微秒                                                    │
│                                                                      │
│  RDMA:                                                               │
│                                                                      │
│  [GPU A] ──────────────────────────────── [GPU B]                   │
│     ↓                                            ↓                   │
│  [RDMA NIC] ────── 直接記憶體傳輸 ──────── [RDMA NIC]                │
│     ↓                                            ↓                   │
│  [遠端記憶體寫入]                             [接收通知]            │
│                                                                      │
│  優勢:                                                               │
│  - CPU 只參與初始化                                                  │
│  - 零拷貝                                                           │
│  - 延遲: < 1 微秒                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 簡單範例

```
節點 1 的 GPU 0 需要獲取節點 2 的 GPU 3 的梯度

傳統路徑 (TCP/IP):
  GPU 3 → CPU → 網卡 → 網路 → 網卡 → CPU → GPU 0
  延遲: 數十微秒
  CPU 參與: 每次傳輸都需要

RDMA 路徑:
  GPU 3 ── RDMA ──→ GPU 0
  延遲: < 1 微秒
  CPU 參與: 僅初始化
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **零拷貝** | 資料直接從網路到記憶體，繞過 CPU |
| **超低延遲** | 微秒級傳輸延遲 |
| **CPU 卸载** | 不佔用 CPU 資源 |
| **大規模訓練的基石** | 多節點 GPU 集群必備 |

---

## 3. 核心原理

### 傳統 TCP/IP 問題

```
請求流程:
1. 資料從網卡到達
2. CPU 中斷處理
3. CPU 複製資料到用戶空間
4. CPU 處理協議棧
5. CPU 複製資料到目標緩衝區

問題:
- CPU 參與每個 packet
- 多次記憶體拷貝
- 高延遲 (數十微秒)
```

### RDMA 工作方式

```
RDMA 流程:
1. 應用程式註冊記憶體區域 (MR)
2. 發送端直接寫入遠端記憶體
3. 接收端直接從記憶體讀取
4. 通知應用程式完成

關鍵：繞過作業系統和 CPU
```

### RDMA Verbs

| 操作 | 說明 | 應用場景 |
|------|------|----------|
| **Send/Receive** | 基本訊息傳遞 | Point-to-Point |
| **Write** | 直接寫入遠端記憶體 | 梯度同步 |
| **Read** | 直接讀取遠端記憶體 | 參數讀取 |
| **Atomic** | 原子操作 | 分散式計數 |

### 深度技術：記憶體註冊 (Memory Region)

```python
class RDMAMemoryRegion:
    """
    RDMA 記憶體註冊 - 核心概念
    """

    def __init__(self, pd):
        self.pd = pd
        self.memory_regions = []

    def register_memory(self, buffer, access_flags):
        """
        註冊記憶體區域 (Memory Region - MR)

        這是 RDMA 的核心:
        - 讓網卡能直接訪問這塊記憶體
        - 註冊後會獲得 remote key (rkey)
        - 其他節點可以用 rkey 寫入這塊記憶體
        """

        mr = self.pd.reg_mr(
            buffer=buffer,
            length=len(buffer),
            access=access_flags
        )

        # 保存註冊信息
        mr_info = {
            "addr": mr.buf,
            "length": mr.length,
            "lkey": mr.lkey,  # 本地 key
            "rkey": mr.rkey   # 遠程 key - 給別人用來寫入
        }

        self.memory_regions.append(mr_info)

        return mr_info

    def get_remote_key(self, index=0):
        """取得遠程訪問 key"""
        mr = self.memory_regions[index]
        return {
            "addr": mr["addr"],
            "rkey": mr["rkey"],
            "length": mr["length"]
        }
```

### 傳輸模式

| 模式 | 特色 | 應用 |
|------|------|------|
| **RC (Reliable Connection)** | 可靠連接，保證送達 | 常見 AI 訓練 |
| **UC (Unreliable Connection)** | 不可靠，效能高 | 較少使用 |
| **UD (Unreliable Datagram)** | 無連接，類似 UDP | 需要時 |

---

## 4. RDMA 技術比較

### 主要 RDMA 技術

| 技術 | 開發者 | 頻寬 | 延遲 | 特色 |
|------|--------|------|------|------|
| **InfiniBand** | IBTA | 400 Gbps | ~0.5 µs | 專用網路，低延遲 |
| **RoCE v2** | Cisco/Broadcom | 200 Gbps | ~1 µs | Ethernet 上運行 |
| **iWARP** | IETF | 100 Gbps | ~2 µs | 基於 TCP |

### InfiniBand vs RoCE

```
InfiniBand:
- 專用交換機 (Mellanox)
- 延遲 ~0.5-1 µs
- 需要專門硬體
- 成本較高 (但穩定高效)

RoCE v2:
- 可用標準 Ethernet 交換機
- 延遲 ~1-2 µs
- 相容現有網路
- 成本較低
```

---

## 5. 在 AI 訓練中的應用

### 分散式訓練通訊

```
節點 1                    節點 2
GPU 0 ─┐                  GPU 4 ─┐
GPU 1 ─┼─ NVSwitch ── RDMA ─┼─ NVSwitch
GPU 2 ─┤                  GPU 5 ─┤
GPU 3 ─┘                  GPU 6 ─┘
       ↓                        ↓
    梯度同步 ── RDMA Write ── 梯度聚合
```

### AllReduce 的 RDMA 優化

```
傳統:
  梯度 → CPU → 網卡 ── TCP ──→ 網卡 → CPU → 梯度

RDMA:
  梯度 ── RDMA Write ──→ 遠端記憶體 (繞過 CPU)

效益:
  - 延遲降低 10x
  - 頻寬提升 2-4x
  - CPU 使用率降為 0
```

---

## 6. NCCL 整合

```python
import torch.distributed as nccl

# 使用 NCCL + RDMA
nccl.init_process_group(
    backend='nccl',
    init_method='env://',
    env={
        'NCCL_IB_DISABLE': '0',
        'NCCL_NET_GDR_LEVEL': '2'  # 啟用 RDMA
    }
)

# AllReduce 使用 RDMA
torch.distributed.all_reduce(tensor)
```

### NCCL RDMA 調優

```bash
# 環境變數
NCCL_IB_DISABLE=0          # 啟用 InfiniBand
NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA
NCCL_IB_CUDA_SUPPORT=1     # 支援 CUDA
NCCL_SOCKET_IFNAME=eth0    # 指定網卡
```

---

## 7. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **NVLink/NVSwitch** | 節點內 GPU 互連 |
| **InfiniBand** | RDMA 的主要實現 |
| **Tensor Parallelism** | 受益於 RDMA 的通訊 |
| **NCCL** | NVIDIA 通訊庫，支援 RDMA |

---

## 8. 數學分析

### 延遲構成

```
RDMA 延遲組成:

1. 連接建立 (首次):
   - QP 建立: ~10-50μs
   - MR 註冊: ~100-500μs

2. 數據傳輸:
   - 純傳輸延遲: ~0.5-1μs
   - 網路傳播: ~0.1μs/m (光纖)
   
3. 總結:
   - 單次 RDMA Write: ~1-2μs
   - 傳統 TCP/IP: ~50-100μs
   - 加速: 50-100x
```

### 頻寬計算

```
頻寬公式:

BW = link_speed × encoding × lanes

範例:
  HDR InfiniBand:
  - link_speed = 200 Gbps (每 lane)
  - encoding = 64/66 (16b 編碼)
  - lanes = 4
  - 理論頻寬 = 200 × 64/66 × 4 / 8 ≈ 77 GB/s

實際頻寬:
  - 典型效率: 80-90%
  - 實際頻寬 ≈ 60-70 GB/s
```

---

## 9. 程式碼實作

### RDMA 基本操作

```python
import pyverbs.cm as cm  # RDMA Connection Manager
import pyverbs.device as d

class RDMAConnection:
    def __init__(self, device_name='mlx5_0'):
        # 開啟設備
        self.ctx = d.Context(name=device_name)

        # 建立Protection Domain
        self.pd = self.ctx.alloc_pd()

        # 建立 Completion Queue
        self.cq = self.ctx.create_cq(1000)

    def create_queue_pair(self):
        """建立 Queue Pair"""

        # QP 屬性
        qp_init_attr = rdp.QPInitAttr(
            qp_type=rdp.QP_RC,  # Reliable Connection
            send_cq=self.cq,
            recv_cq=self.cq,
            cap=rdp.QPCap(
                max_send_wr=1000,
                max_recv_wr=1000,
                max_send_sge=10,
                max_recv_sge=10
            )
        )

        self.qp = self.pd.create_qp(qp_init_attr)

    def register_memory(self, buffer):
        """註冊記憶體區域"""

        mr = self.pd.reg_mr(
            buffer=buffer,
            access=rdp.IBV_ACCESS_LOCAL_WRITE |
                   rdp.IB V_ACCESS_REMOTE_WRITE |
                   rdp.IB V_ACCESS_REMOTE_READ
        )

        return mr

    def post_send(self, mr, remote_addr, rkey):
        """發送 RDMA Write"""

        # 建立 work request
        wr = rdp.SendWR(
            opcode=rdp.IBV_WR_RDMA_WRITE,
            num_sge=1,
            sg_list=[
                rdp.SGE(
                    lkey=mr.lkey,
                    addr=mr.buf,
                    length=mr.length
                )
            ],
            wr_id=1
        )

        # 發送
        self.qp.post_send(wr)

        # 等待完成
        self.cq.poll_completion()
```

### GPU Direct RDMA

```python
import cupy as cp
import pyverbs

class GPURDMA:
    def __init__(self):
        # 初始化 CUDA
        self.stream = cp.cuda.Stream()

        # 開啟 RDMA 設備
        self.ctx = pyverbs.Context('mlx5_0')

        # 建立 PD
        self.pd = self.ctx.alloc_pd()

    def register_gpu_memory(self, gpu_array):
        """註冊 GPU 記憶體"""

        # 取得 GPU 記憶體指標
        gpu_ptr = gpu_array.data.ptr
        size = gpu_array.nbytes

        # 註冊為 RDMA MR
        mr = self.pd.reg_mr(
            addr=gpu_ptr,
            length=size,
            access=pyverbs.IBV_ACCESS_LOCAL_WRITE |
                   pyverbs.IBV_ACCESS_REMOTE_WRITE
        )

        return mr

    def remote_write(self, local_mr, remote_mr, size):
        """RDMA Write 從 GPU 到遠端"""

        wr = pyverbs.RDMASendWR(
            opcode=pyverbs.IBV_WR_RDMA_WRITE,
            remote_addr=remote_mr.addr,
            rkey=remote_mr.rkey,
            num_sge=1,
            sg_list=[
                pyverbs.SGE(
                    lkey=local_mr.lkey,
                    addr=local_mr.addr,
                    length=size
                )
            ]
        )

        self.qp.post_send(wr)
        self.cq.poll_completion()
```

---

## 10. NCCL 整合深入

### 自訂 RDMA 傳輸

```python
import torch.distributed as dist

# 環境變數配置
os.environ['NCCL_IB_DISABLE'] = '0'      # 啟用 InfiniBand
os.environ['NCCL_IB_GID_INDEX'] = '3'    # GID 索引
os.environ['NCCL_NET_PLUGIN'] = 'ibnetrdma'  # RDMA 外掛

# 自訂通訊
class RDMACollective:
    def __init__(self, group):
        self.group = group
        self.stream = torch.cuda.Stream()

    def allreduce_rdma(self, tensor):
        """使用 RDMA 優化的 AllReduce"""

        # 確保張量在 GPU 上
        assert tensor.is_cuda

        # NCCL 自動使用 RDMA
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)

        return tensor

    def broadcast_rdma(self, tensor, src):
        """使用 RDMA 廣播"""

        dist.broadcast(tensor, src=src, group=self.group)
```

### 效能監控

```python
class NCCLRDMAMonitor:
    def __init__(self):
        self.stats = {
            "allreduce_latency": [],
            "broadcast_latency": [],
            "bytes_sent": []
        }

    @staticmethod
    def enable_profiling():
        """啟用 NCCL 效能分析"""

        import os
        os.environ['NCCL_DEBUG'] = 'WARN'
        os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    def profile_allreduce(self, tensor_size_mb, num_iterations=100):
        """測試 AllReduce 效能"""

        tensor = torch.randn(
            tensor_size_mb * 1024 * 1024 // 4,
            device='cuda'
        )

        # 預熱
        for _ in range(10):
            torch.distributed.all_reduce(tensor)

        torch.cuda.synchronize()

        # 基準測試
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            torch.distributed.all_reduce(tensor)
        end.record()

        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        avg_latency = elapsed_ms / num_iterations
        bandwidth_gbps = (tensor_size_mb * num_iterations * 1000) / elapsed_ms

        return {
            "latency_ms": avg_latency,
            "bandwidth_gbps": bandwidth_gbps,
            "efficiency": bandwidth_gbps / 400  # 假設 400 GB/s
        }
```

---

## 11. 網路拓撲

### 典型叢集拓撲

```
典型 AI 訓練叢集 (32 GPU, 4 節點):

                    ┌─────────────┐
                    │  InfiniBand │
                    │   Switch    │
                    │   HDR 200G  │
                    └──────┬──────┘
           ┌──────────────┼──────────────┐
           │              │              │
     ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
     │  節點 0   │  │  節點 1   │  │  節點 2   │
     │ 8x H100   │  │ 8x H100   │  │ 8x H100   │
     │ NVSwitch  │  │ NVSwitch  │  │ NVSwitch  │
     └───────────┘  └───────────┘  └───────────┘

節點內: NVLink (900 GB/s)
節點間: InfiniBand HDR (200 Gbps per port)
```

### 路由優化

```bash
# 檢查網路拓撲
ibnetdiscover

# 檢查 RDMA 網路狀態
ibstat

# 效能測試
ib_send_bw -d mlx5_0 -s 4096 -n 1000
```

---

## 12. 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **連接建立失敗** | 防火牆阻擋 | 開放相關連接埠 |
| **頻寬低** | MTU 過小 | 設定 MTU 4096 |
| **延遲高** | 軟體堆疊問題 | 使用硬體 RDMA |
| **記憶體不足** | MR 過多 | 合併記憶體區域 |

### 診斷工具

```bash
# 檢查 RDMA 設備
ibv_devices

# 檢查網卡狀態
ip link show

# 測試 RDMA 連接
rping -s -a <server_ip> -c 100 &
rping -c -a <client_ip> -C 100

# 監控網路流量
perfquery
```

---

## 13. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **NVLink/NVSwitch** | 節點內 GPU 互連 |
| **InfiniBand** | RDMA 的主要實現 |
| **Tensor Parallelism** | 受益於 RDMA 的通訊 |
| **NCCL** | NVIDIA 通訊庫，支援 RDMA |

---

## 延伸閱讀

- [RDMA 基礎教學](https://www.mellanox.com/rdma)
- [NCCL RDMA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
- [InfiniBand Architecture](https://www.infinibandta.org/)
- [GPU Direct RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [RDMA Programming](https://www.mellanox.com/products/rdma-tools)