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

## 延伸閱讀

- [GPipe Paper](https://arxiv.org/abs/1811.06965)
- [PipeDream Paper](https://arxiv.org/abs/1907.13257)
- [Megatron-LM PP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)