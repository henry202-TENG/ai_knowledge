# Tensor Parallelism (張量平行)

## 1. 什麼是？
Tensor Parallelism（張量平行）是一種模型並行技術，將模型的核心矩陣運算（如線性層的權重矩陣）沿著特定維度切分到多個 GPU 上，實現單個運算的並行執行。

## 2. 為什麼重要？
- **支撐大模型**：數百億參數模型必須使用
- **高效率**：相比數據並行，通訊更少
- **記憶體分散**：每個 GPU 只需儲存部分權重

## 3. 核心原理

### 基本概念
```
假設有矩陣運算: Y = X × W

權重矩陣 W 形狀: [d_model, d_ff]
                   ↓
              沿著 d_ff 切分到 N 個 GPU

GPU 0: W[:, :d_ff/N] → Y0
GPU 1: W[:, d_ff/N:2*d_ff/N] → Y1
...
```

### 常見切分策略

#### 1. Column Parallel（行切分）
```
用於線性層的權重矩陣

W = [W₁ | W₂ | W₃] (按列切分)

輸入 X 同時廣播到所有 GPU
每個 GPU 計算部分輸出
最後 AllReduce 合併結果
```

#### 2. Row Parallel（行切分）
```
用於需要跨 GPU 聚合的層

W = [W₁]
    [W₂]
    [W₃] (按行切分)

每個 GPU 計算部分結果
需要通訊來聚合
```

### Attention 中的 Tensor Parallelism
```
Multi-Head Attention 切分:

Q, K, V 矩陣按 head 維度切分

Head 0-31: GPU 0
Head 32-63: GPU 1
...

每個 GPU 計算部分 head 的 Attention
最後合併
```

### 通訊模式
```
AllReduce: 將多個 GPU 的結果合併
  GPU 0: [a₁, a₂] ─┐
  GPU 1: [b₁, b₂] ──┼─→ [a₁+b₁, a₂+b₂]
  GPU 2: [c₁, c₂] ─┘

通訊量: O(d_model) per AllReduce
```

## 4. 知名實現

| 框架 | 實現 |
|------|------|
| **Megatron-LM** | 完整的 TP 實現 |
| **DeepSpeed** | ZeRO + TP |
| **PyTorch FSDP** | FSDP 的一部分 |
| **vLLM** | 推論時的 TP |

## 5. 與其他並行策略的比較

| 特性 | Tensor Parallelism | Pipeline Parallelism | Data Parallelism |
|------|-------------------|---------------------|-----------------|
| 切分方式 | 模型權重 | 模型層 | 數據 |
| 通訊量 | 中等 | 少 | 少 |
| 同步難度 | 高 | 中 | 低 |
| 適用場景 | 大模型層 | 多節點 | 多 GPU 同節點 |

## 6. 相關主題

| 技術 | 關係 |
|------|------|
| **Pipeline Parallelism** | 與 TP 常常結合使用 |
| **NVLink** | GPU 互連影響 TP 效率 |
| **AllReduce** | TP 的核心通訊操作 |
| **3D Parallelism** | TP + PP + DP 組合 |

## 7. 延伸閱讀
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [Tensor Parallelism 詳解](https://github.com/NVIDIA/Megatron-LM)
- [3D Parallelism](https://arxiv.org/abs/2205.05198)

---

*待補充...*