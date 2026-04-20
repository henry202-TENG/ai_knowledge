# Pipeline Parallelism (管線平行)

## 1. 什麼是？
Pipeline Parallelism（管線平行）是一種模型並行技術，將模型的不同層分配到不同的 GPU 上，讓資料像工廠管線一樣在各個 stage 間流動。

## 2. 為什麼重要？
- **支援超大模型**：單 GPU 放不下的模型
- **簡單直觀**：按層切分，邏輯清晰
- **節點擴展**：輕鬆擴展到多節點

## 3. 核心原理

### 基本概念
```
模型層: L1 → L2 → L3 → L4 → L5 → L6

Pipeline 配置 (4 GPU):
Stage 0: [L1, L2]
Stage 1: [L3, L4]
Stage 2: [L5]
Stage 3: [L6]

資料流動:
Input → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Output
```

### 問題：Pipeline Bubbles

#### 原始實現
```
時間 →

GPU 0: [F0][F1][F2][F3][F4][F5]
GPU 1:     [W0][W1][W2][W3][W4][W5]
GPU 2:         [W0][W1][W2][W3][W4][W5]
GPU 3:             [B0][B1][B2][B3][B4][B5]

[F: Forward, W: Wait, B: Backward]

問題：每個 GPU 有大量空閒時間（bubbles）
```

### 解決方案：Micro-batching

#### GPipe (Interleaved)
```
將 batch 切成多個 micro-batches:

Micro-batch 1 → GPU 0 → GPU 1 → GPU 2 → GPU 3
Micro-batch 2 → GPU 0 → GPU 1 → GPU 2 → GPU 3
Micro-batch 3 → GPU 0 → GPU 1 → GPU 2 → GPU 3
...

Bubble 時間大幅減少
```

#### PipeDream
```
雙向管線：
- 奇數 forward 階段
- 偶數 backward 階段

更高效的時間利用
```

### Pipeline schedule
| Schedule | 特色 | 優點 | 缺點 |
|----------|------|------|------|
| **Forward only** | 簡單 | 實現容易 | Memory 高 |
| **1F1B** | 交替 F/B | 記憶體低 | 控制複雜 |
| **Interleaved** | 多 stage F/B | Bubble 小 | 通訊多 |

### 通訊模式
```
P2P (Point-to-Point):
  Stage i → Stage i+1
  每個 stage 只與相鄰 stage 通訊

Barrier Synchronization:
  每個 micro-batch 結束同步
```

## 4. 知名實現

| 框架 | 實現 |
|------|------|
| **PyTorch DDP** | 原生 Pipeline 支援 |
| **Megatron-LM** | 優化的 PP 實現 |
| **DeepSpeed** | 3D Parallelism |
| **Fairscale** | PyTorch PP wrapper |

## 5. 與 Tensor Parallelism 比較

| 特性 | Pipeline | Tensor |
|------|---------|--------|
| 切分維度 | 層 | 權重矩陣 |
| 通訊對象 | 鄰近節點 | 所有節點 |
| 通訊量 | 較少 | 較多 |
| 同步顆粒度 | Micro-batch | 層輸出 |
| 適合拓撲 | 多節點集群 | 多 GPU 節點 |

## 6. 相關主題

| 技術 | 關係 |
|------|------|
| **Tensor Parallelism** | 常與 PP 結合 (3D Parallelism) |
| **GPipe** | 經典 PP 論文 |
| **1F1B Schedule** | 常用排程算法 |
| **Gradient Accumulation** | 增大 effective batch size |

## 7. 延伸閱讀
- [GPipe Paper](https://arxiv.org/abs/1811.06965)
- [PipeDream Paper](https://arxiv.org/abs/1907.13257)
- [Megatron-LM PP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)

---

*待補充...*