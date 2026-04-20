# RDMA

繞過 CPU 直接在兩台伺服器的記憶體之間傳輸資料的網路技術，讓 GPU 可以直接讀寫另一台伺服器的記憶體，實現超低延遲的高效能運算。

---

## 1. 什麼是？

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

## 延伸閱讀

- [RDMA 基礎教學](https://www.mellanox.com/rdma)
- [NCCL RDMA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
- [InfiniBand Architecture](https://www.infinibandta.org/)
- [GPU Direct RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)