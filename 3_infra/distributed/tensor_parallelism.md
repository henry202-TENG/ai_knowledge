# Tensor Parallelism

將模型的核心矩陣運算（如線性層的權重矩陣）沿著特定維度切分到多個 GPU 上，實現單個運算的並行執行。

---

## 1. 什麼是？

### 深度定義

**Tensor Parallelism (TP)** 是模型並行的一種形式，與 Pipeline Parallelism 和 Data Parallelism 並列為三大並行策略。其核心特點是：

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Tensor Parallelism 定位                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  層級關係:                                                           │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    訓練 / 推論                                 │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
│                    ───────────────────────                          │
│                    │                                               │
│         ┌──────────┴──────────┐                                    │
│         ▼                     ▼                                    │
│  ┌─────────────┐       ┌─────────────┐                              │
│  │ Model       │       │ Data        │                              │
│  │ Parallelism │       │ Parallelism │                              │
│  └──────┬──────┘       └──────┬──────┘                              │
│         │                     │                                     │
│    ┌────┴────┐                │                                     │
│    ▼         ▼           ┌────┴────┐                                 │
│ ┌──────┐ ┌──────┐        ▼         ▼                                │
│ │Tensor│ │Pipe  │    ┌────────┐ ┌────────┐                          │
│ │Parallel│ │Parallel│    │DDP    │ │FSDP   │                          │
│ └──────┘ └──────┘    └────────┘ └────────┘                          │
│                                                                      │
│  Tensor Parallelism 特色:                                           │
│  - 單層內的矩陣運算並行                                              │
│  - 需要 GPU 間通訊 (AllReduce)                                      │
│  - 適合單節點多 GPU (1-8 卡)                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

### 深度挑戰分析

#### 挑戰 3：非線性層的同步問題

**問題描述**：
```
Transformer 中的非線性操作:
- LayerNorm
- Dropout
- Activation (GELU, ReLU)

這些操作需要在所有 GPU 上同步執行
```

**解決方案**：
```python
class SyncNonLinearLayers:
    """同步非線性層"""

    def __init__(self, tp_size):
        self.tp_size = tp_size

    def layer_norm(self, x):
        """同步 LayerNorm"""

        # 1. 先在本地計算
        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True)

        # 2. AllReduce 同步均值和方差
        global_mean = self._all_reduce_mean(local_mean)
        global_var = self._all_reduce_mean(local_var)

        # 3. 使用全局統計量歸一化
        normalized = (x - global_mean) / torch.sqrt(global_var + 1e-6)

        # 4. 學習的 scale 和 bias 也需要同步
        # (這些是 TP 獨立的，需要廣播)
        return self.gamma * normalized + self.beta

    def gelu(self, x):
        """同步 GELU - 可以並行"""
        # GELU 是元素級操作，不需要同步
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(2 / 3.14159265359) * (x + 0.044715 * x**3)
        ))
```

#### 挑戰 4：Embedding 層的切分

**問題描述**：
```
Embedding 層的處理:
- 輸入: token_ids [batch, seq]
- 輸出: hidden_states [batch, seq, d_model]

問題:
- 如果按詞彙表切分 (vocab_parallel):
  - 每個 GPU 只保存部分詞向量
  - 查詢時需要 AllReduce 收集完整 embedding
- 如果按隱藏維度切分:
  - 輸出會不完整
```

**解決方案**：
```python
class ParallelEmbedding(nn.Module):
    """並行 Embedding 層"""

    def __init__(self, vocab_size, d_model, tp_size):
        super().__init__()
        self.tp_size = tp_size

        # 按詞彙表切分
        self.vocab_per_gpu = vocab_size // tp_size
        self.embedding = nn.Embedding(
            self.vocab_per_gpu,
            d_model
        )

    def forward(self, token_ids):
        # 假設 token_ids 已經重新映射到本地範圍
        local_embeddings = self.embedding(token_ids)

        # AllGather 收集完整的 embedding
        # [batch, seq, d_model] × tp_size → [batch, seq, d_model × tp_size]
        all_embeddings = torch.cat(
            [local_embeddings]
            + [torch.zeros_like(local_embeddings) for _ in range(self.tp_size - 1)],
            dim=-1
        )

        dist.all_gather_into_tensor(all_embeddings, local_embeddings)

        return all_embeddings
```

#### 挑戰 5：輸出層的處理

**問題描述**：
```
輸出層 (LM Head):
- 輸入: [batch, seq, d_model]
- 輸出: [batch, seq, vocab_size]

如果使用 vocab_parallel:
- 每個 GPU 計算部分詞彙的 logits
- 需要 AllReduce 合併後才能計算 loss
```

**解決方案**：
```python
class ParallelLMHead(nn.Module):
    """並行 Language Modeling Head"""

    def __init__(self, d_model, vocab_size, tp_size):
        super().__init__()
        self.tp_size = tp_size

        # 按詞彙表切分
        self.vocab_per_gpu = vocab_size // tp_size
        self.linear = nn.Linear(
            d_model,
            self.vocab_per_gpu,
            bias=False
        )

    def forward(self, hidden_states):
        # 本地計算部分 logits
        local_logits = self.linear(hidden_states)

        # AllReduce 合併
        # 每個 GPU 有 vocab_per_gpu 個 logit
        # 合併後每個 GPU 有完整 vocab_size 個 logit
        dist.all_reduce(local_logits, op=dist.ReduceOp.SUM)

        return local_logits
```

#### 挑戰 6：梯度同步優化

**問題描述**：
```
TP 的梯度同步:
- 前向: 輸入廣播到所有 GPU
- 反向: 梯度需要 AllReduce

優化策略:
1. 梯度 AllReduce 與計算重疊
2. 混合精度訓練 (FP16/BF16)
3. 梯度壓縮
```

**實現**：
```python
class TensorParallelOptimizer(torch.optim.Optimizer):
    """TP 優化器 - 優化梯度同步"""

    def __init__(self, params, tp_size):
        super().__init__(params)
        self.tp_size = tp_size

    def step(self, closure=None):
        # 1. 梯度同步 (異步)
        for param in self.param_groups[0]['params']:
            if param.grad is not None:
                # 異步 AllReduce
                handle = dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )

                # 等待完成
                handle.wait()

                # 除以 TP size
                param.grad.div_(self.tp_size)

        # 2. 標準優化器 step
        return super().step(closure)
```

---

## 7. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Pipeline Parallelism** | 與 TP 常常結合使用 |
| **NVLink** | GPU 互連影響 TP 效率 |
| **AllReduce** | TP 的核心通訊操作 |
| **3D Parallelism** | TP + PP + DP 組合 |

---

## 8. 數學推導

### Column Parallel 數學

```
輸入 X ∈ ℝᵇⁿˣᵈ (batch, seq, d_model)
權重 W ∈ ℝᵈˣᵈff (d_model, d_ff)

Column Parallel 切分:
  W = [W₁, W₂, W₃, W₄], 每個 Wᵢ ∈ ℝᵈˣ(dff/4)
  
  Y = XW = X[W₁, W₂, W₃, W₄]
    = [XW₁, XW₂, XW₃, XW₄]
    = [Y₁, Y₂, Y₃, Y₄]

每個 GPU 計算:
  Yᵢ = X × Wᵢ

AllReduce:
  Y = AllReduce([Y₁, Y₂, Y₃, Y₄]) = Σᵢ Yᵢ
```

### Row Parallel 數學

```
Row Parallel 切分:
  W = [W₁]
      [W₂]
      [W₃]
      [W₄]  (dff/4, d_model)

每個 GPU 計算:
  Yᵢ = Xᵢ × Wᵢ  (只有部分輸入)

AllReduce 聚合:
  Y = Σᵢ Yᵢ
```

### 通訊量分析

```
假設:
  - 隱藏維度: d = 4096
  - FFN 維度: d_ff = 16384
  - 序列長度: s = 2048
  - GPU 數量: tp = 4

Column Parallel:
  - 輸入廣播: s × d = 2048 × 4096 = 8 MB
  - AllReduce: s × d = 8 MB

Row Parallel:
  - AllReduce: s × (dff/tp) = 2048 × 4096 = 8 MB

總通訊量 (每層):
  = 2 × 8 MB = 16 MB
  
相比 Data Parallel (all-reduce 全部梯度):
  = s × d × tp = 2048 × 4096 × 4 = 32 MB
```

---

## 9. 實現細節

### Megatron-LM 風格實現

```python
class ColumnParallelLinear(nn.Module):
    """Column Parallel Linear Layer"""

    def __init__(self, input_size, output_size, tp_size):
        super().__init__()
        self.tp_size = tp_size

        # 沿 output_size 維度切分
        self.weight = nn.Parameter(
            torch.randn(output_size // tp_size, input_size)
        )
        self.bias = nn.Parameter(
            torch.zeros(output_size // tp_size)
        )

    def forward(self, x):
        # x: [batch, seq, input_size]

        # 本地線性計算
        output = F.linear(x, self.weight, self.bias)
        # output: [batch, seq, output_size/tp_size]

        # AllReduce 聚合結果
        dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output


class RowParallelLinear(nn.Module):
    """Row Parallel Linear Layer"""

    def __init__(self, input_size, output_size, tp_size):
        super().__init__()
        self.tp_size = tp_size

        # 沿 input_size 維度切分
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size // tp_size)
        )

    def forward(self, x):
        # x 已經是分片的 [batch, seq, input_size/tp_size]

        # 本地線性計算
        output = F.linear(x, self.weight)
        # output: [batch, seq, output_size]

        # AllReduce 聚合
        dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output
```

### Attention 中的 TP

```python
class TensorParallelAttention(nn.Module):
    """帶 TP 的 Multi-Head Attention"""

    def __init__(self, d_model, num_heads, tp_size):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.tp_size = tp_size

        self.head_dim = d_model // num_heads

        # Q, K, V 使用 Column Parallel
        self.query = ColumnParallelLinear(d_model, d_model, tp_size)
        self.key = ColumnParallelLinear(d_model, d_model, tp_size)
        self.value = ColumnParallelLinear(d_model, d_model, tp_size)

        # Output 使用 Row Parallel
        self.dense = RowParallelLinear(d_model, d_model, tp_size)

    def forward(self, x, attention_mask=None):
        # 計算 Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 沿 head 維度分片
        # 每個 GPU 處理 num_heads / tp_size 個 heads
        q = q.view(-1, self.num_heads // self.tp_size, self.head_dim)
        k = k.view(-1, self.num_heads // self.tp_size, self.head_dim)
        v = v.view(-1, self.num_heads // self.tp_size, self.head_dim)

        # Attention 計算 (本地)
        attn_output = self._attention(q, k, v, attention_mask)

        # 輸出投影 (Row Parallel)
        output = self.dense(attn_output)

        return output
```

---

## 10. 通訊優化

### Overlap 通訊與計算

```python
class AsyncTensorParallel:
    """將通訊與計算重疊"""

    def __init__(self, tp_size):
        self.tp_size = tp_size
        self.stream = torch.cuda.Stream()

    def forward_with_overlap(self, x, layer):
        """
        重疊 AllReduce 與計算
        """

        # 啟動非同步計算
        with torch.cuda.stream(self.stream):
            output = layer(x)

        # 等待計算完成後進行通訊
        torch.cuda.synchronize()

        # 通訊 (另一個 stream)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output
```

### 批次通訊優化

```python
class BatchedAllreduce:
    """批量 AllReduce 減少通訊次數"""

    def __init__(self, tp_size):
        self.tp_size = tp_size
        self.pending_tensors = []
        self.batch_size = 4

    def add_and_reduce(self, tensor):
        """累加張量並批量 reduce"""

        self.pending_tensors.append(tensor)

        if len(self.pending_tensors) >= self.batch_size:
            return self._batch_reduce()

        return None

    def _batch_reduce(self):
        """批量減少"""

        # 堆疊張量
        stacked = torch.stack(self.pending_tensors)

        # 一次 AllReduce
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)

        # 分離
        results = list(stacked.unbind(0))
        self.pending_tensors.clear()

        return results
```

---

## 11. 3D Parallelism 整合

### TP + PP + DP 組合

```python
class ThreeDParallelModel(nn.Module):
    """3D Parallelism 模型"""

    def __init__(self, config, tp_size=2, pp_size=2, dp_size=4):
        super().__init__()
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size

        # 計算 ranks
        self.tp_rank = self._get_tensor_model_parallel_rank()
        self.pp_rank = self._get_pipeline_model_parallel_rank()
        self.dp_rank = self._get_data_parallel_rank()

        # 建立模型
        self.layers = self._build_layers()

    def _build_layers(self):
        """建立分層模型"""

        layers_per_stage = config.num_layers // self.pp_size

        layers = nn.ModuleList([
            self._build_transformer_layer(
                layer_idx=self.pp_rank * layers_per_stage + i
            )
            for i in range(layers_per_stage)
        ])

        return layers

    def forward(self, x):
        # Pipeline Stage 內的 Forward
        for layer in self.layers:
            x = layer(x)

            # TP 通信
            if self.tp_size > 1:
                dist.all_reduce(x, op=dist.ReduceOp.SUM)

        return x
```

### 記憶體分析

```
3D Parallelism 記憶體分佈:

假設:
  - 模型: LLaMA 70B
  - tp_size = 8
  - pp_size = 4  
  - dp_size = 4 (總 128 GPU)

每個 GPU:
  - 參數: 70B / 128 / 8 ≈ 70M 參數
  - 梯度: 70M × 4 bytes = 280 MB
  - Activations: 視序列長度而定
  
相比單 GPU:
  - 需要 512 GB → 只需 8 GB
```

---

## 12. 效能基準

### 通訊量比較

```
模型: LLaMA 70B, 序列長度 2048

| 方法      | 每層通訊量 | 總通訊量 |
|-----------|-----------|----------|
| Data Parallel | 32 MB    | 32 × 80 = 2.5 GB |
| Tensor Parallel| 16 MB   | 16 × 80 = 1.3 GB |
| Pipeline      | 8 MB     | 8 × 80 = 0.6 GB  |

最佳: 結合 PP + TP + DP
```

### 加速比分析

```
訓練時間比較 (64 A100, 175B 模型):

| 方法           | 時間 (小時) | 加速比 |
|----------------|-------------|--------|
| Data Parallel  | 40          | 1x     |
| TP=8           | 32          | 1.25x  |
| TP=8, PP=4     | 15          | 2.67x  |
| TP×PP×DP       | 8           | 5x     |
```

---

## 13. 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **負載不均** | 切分不均勻 | 調整 TP 大小或維度 |
| **通訊瓶頸** | AllReduce 太慢 | 使用 NVLink |
| **梯度不一致** | 同步問題 | 確保 AllReduce 完成 |
| **記憶體不足** | 參數太多 | 增大 TP 或使用 ZeRO |

### 調優參數

| 參數 | 建議值 | 說明 |
|------|--------|------|
| **tensor_model_parallel_size** | 4-8 | 通常 8 是上限 |
| **pipeline_model_parallel_size** | 2-8 | 視節點數量 |
| **context_parallel_size** | 1-8 | 序列並行 |
| **gradient_accumulation** | 視記憶體 | 增大 effective batch |

---

## 14. 相關主題

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
- [Megatron-LM v2](https://arxiv.org/abs/2205.05198)
- [ColossalAI TP](https://www.colossalai.org/)