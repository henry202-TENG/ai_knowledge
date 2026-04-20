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

## 延伸閱讀

- [NVLink/NVSwitch 白皮書](https://resources.nvidia.com/en-us-tensor-core)
- [Hopper Architecture](https://resources.nvidia.com/en-us-hopper-architecture)
- [DGX Systems](https://www.nvidia.com/en-us/data-center/dgx-systems/)