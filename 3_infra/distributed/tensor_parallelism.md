# Tensor Parallelism

將模型的核心矩陣運算（如線性層的權重矩陣）沿著特定維度切分到多個 GPU 上，實現單個運算的並行執行。

---

## 1. 什麼是？

### 簡單範例

```
矩陣運算: Y = X × W

W 形狀: [d_model, d_ff] = [4096, 16384]

4 GPU 切分:
GPU 0: W[:, 0:4096]    → Y[:, 0:4096]
GPU 1: W[:, 4096:8192] → Y[:, 4096:8192]
GPU 2: W[:, 8192:12288]→ Y[:, 8192:12288]
GPU 3: W[:, 12288:16384]→ Y[:, 12288:16384]

每個 GPU 計算部分輸出 → AllReduce 合併
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **支撐大模型** | 數百億參數模型必須使用 |
| **高效率** | 相比數據並行，通訊更少 |
| **記憶體分散** | 每個 GPU 只需儲存部分權重 |
| **細粒度並行** | 單層內並行 |

---

## 3. 核心原理

### 常見切分策略

#### Column Parallel

```
W = [W₁ | W₂ | W₃] (按列切分)

輸入 X 同時廣播到所有 GPU
每個 GPU 計算: Yᵢ = X × Wᵢ
結果: Y = [Y₁, Y₂, Y₃] → AllReduce 合併
```

#### Row Parallel

```
W = [W₁]
    [W₂] (按行切分)
    [W₃]

每個 GPU 計算: Yᵢ = X × Wᵢ
結果需要 AllReduce 聚合
```

### Attention 中的 Tensor Parallelism

```python
# Multi-Head Attention 切分

# Q, K, V 按 head 維度切分
num_heads = 32
num_gpus = 4
heads_per_gpu = num_heads // num_gpus

# GPU 0: heads 0-7
# GPU 1: heads 8-15
# GPU 2: heads 16-23
# GPU 3: heads 24-31

# 每個 GPU 計算部分 head 的 Attention
# 最後合併輸出
```

### 通訊模式

```python
# AllReduce: 將多個 GPU 的結果合併

# 例子: 4 GPU
GPU 0: [a₁, a₂] ─┐
GPU 1: [b₁, b₂] ──┼─→ [a₁+b₁, a₂+b₂]
GPU 2: [c₁, c₂] ─┘

# 通訊量: O(d_model) per AllReduce
# 對於大模型，這是主要瓶頸
```

### 3D Parallelism

```
Tensor Parallelism: 單層內並行 (GPU 數 1-8)
Pipeline Parallelism: 層級並行 (GPU 數 8-64)
Data Parallelism: 數據並行 (GPU 數 64+)

3D = TP × PP × DP
```

---

## 4. 知名實現

| 框架 | 說明 |
|------|------|
| **Megatron-LM** | 完整的 TP 實現，標準參照 |
| **DeepSpeed** | ZeRO + TP |
| **PyTorch FSDP** | FSDP 的一部分 |
| **vLLM** | 推論時的 TP |

---

## 5. 與其他並行策略的比較

| 特性 | Tensor Parallelism | Pipeline Parallelism | Data Parallelism |
|------|-------------------|---------------------|-----------------|
| 切分方式 | 模型權重 | 模型層 | 數據 |
| 通訊量 | 中等 | 少 | 少 |
| 同步難度 | 高 | 中 | 低 |
| 適用場景 | 大模型層 | 多節點 | 多 GPU 同節點 |
| GPU 數 | 1-8 | 8-64 | 64+ |

---

## 6. 挑戰與解決

### 挑戰 1：通訊瓶頸

**問題**：AllReduce 延遲高

**解決方案**：
- 使用 NVLink/NVSwitch
- 調整 TP 度
- Overlap 通訊與計算

### 挑戰 2：負載不均

**問題**：某些運算無法均分

**解決方案**：
- 仔細設計切分策略
- 使用虛擬管道

---

## 7. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Pipeline Parallelism** | 與 TP 常常結合使用 |
| **NVLink** | GPU 互連影響 TP 效率 |
| **AllReduce** | TP 的核心通訊操作 |
| **3D Parallelism** | TP + PP + DP 組合 |

---

## 延伸閱讀

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [Tensor Parallelism 詳解](https://github.com/NVIDIA/Megatron-LM)
- [3D Parallelism](https://arxiv.org/abs/2205.05198)