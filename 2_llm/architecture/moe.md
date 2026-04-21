# MoE

稀疏化模型架構，透過動態激活不同的「專家」網路來處理不同的輸入，在保持強大能力的同時大幅降低計算成本。

---

## 1. 什麼是？

### 深度定義

**MoE (Mixture of Experts)** 是一種**條件計算 (Conditional Computation)** 的稀疏化架構：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MoE 核心設計思想                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  傳統 Dense 模型:                                                    │
│  - 輸入: x                                                          │
│  - 處理: 所有參數都參與計算                                            │
│  - 輸出: y = f(x; θ)                                                │
│  - 問題: 參數越多，計算量越大                                         │
│                                                                      │
│  MoE 稀疏化:                                                         │
│  - 輸入: x                                                          │
│  - 路由器: 決定哪些專家被激活                                          │
│  - 輸出: y = Σ g(x)ᵢ × Eᵢ(x)  (只選 top-k 個專家)                   │
│  - 優勢: 參數量不變，計算量大幅減少                                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  100B 參數的 MoE 模型 (8 專家，激活 2 個):                    │   │
│  │                                                              │   │
│  │  理論計算量: 100B / 8 × 2 = 25B FLOPs                        │   │
│  │  相比同參數 Dense 模型: 節省 4x 計算                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**關鍵創新**:
1. **稀疏激活**: 每個 token 只走少數專家
2. **可學習路由**: 路由器自動學習分發策略
3. **專家特殊化**: 不同專家學到不同能力

### 簡單範例

```
Dense 模型 (100B 參數):
  輸入 → [所有參數 100B] → 輸出
  每次推理: 100B 參數計算

MoE 模型 (100B 參數, 8 個專家):
  輸入 → [路由器] → 激活 2 個專家 → 輸出
  每次推理: 12.5B × 2 = 25B 參數計算

→ 4x 推理加速
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **大幅降低推理成本** | 只激活部分專家，而非整個模型 |
| **擴展模型參數** | 支援兆級參數模型（如 GPT-4 傳言使用 MoE） |
| **任務專家化** | 不同專家擅長不同類型的任務 |
| **訓練效率** | 稀疏更新，減少梯度同步 |

---

## 3. 核心原理

### 架構圖

```
         ┌─────────────────────────────────────────┐
         │           輸入 x                         │
         └─────────────────────────────────────────┘
                              │
                              ↓
         ┌─────────────────────────────────────────┐
         │    Router / Gate (可學習)                │
         │    output = softmax(W × x)              │
         └─────────────────────────────────────────┘
                              │
          ┌──────────┬──────────┼──────────┬──────────┐
          ↓          ↓          ↓          ↓          ↓
    ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
    │ Expert 1 ││ Expert 2 ││ Expert 3 ││ Expert 4 ││   ...   │
    │ (FFN)   ││ (FFN)   ││ (FFN)   ││ (FFN)   ││         │
    └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
          │          │          │          │          │
          └──────────┴──────────┼──────────┴──────────┘
                                 ↓
         ┌─────────────────────────────────────────┐
         │         加權輸出                          │
         │    y = Σ g(x)ᵢ × Eᵢ(x)                  │
         └─────────────────────────────────────────┘
```

### 關鍵組件

#### 1. Experts (專家網路)

```python
# 每個 Expert 是獨立的 FFN
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.gate(x)
```

#### 2. Router/Gate (路由器)

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # 計算每個專家的權重
        logits = self.gate(x)  # [batch, num_experts]

        # Top-K 選擇
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax (只在 top-k 上)
        weights = F.softmax(top_k_logits, dim=-1)

        return weights, indices
```

#### 3. Load Balancing (負載均衡)

```problem
問題：有時某些專家總是被選中，其他專家閒置

解決方案 1: 輔助 Loss
  auxiliary_loss = -Σ log(avg_load_i)

解決方案 2: Mixtral 的 z-loss
  z_loss = 0.001 × (logits.std())²

解決方案 3: Noisy Top-K Routing
  加入噪聲讓分配更均勻
```

```python
def load_balancing_loss(router_logits, expert_indices):
    # 計算每個 expert 被選中的次數
    num_experts = router_logits.shape[-1]
    expert_mask = F.one_hot(expert_indices, num_experts)
    expert_usage = expert_mask.sum(0) / expert_mask.sum()

    # 均勻分布的 Loss
    return -((expert_usage - 1/num_experts)**2).sum()
```

### MoE 公式

```
輸出 = Σᵢ g(x)ᵢ × Eᵢ(x)

其中:
- x = 輸入
- g(x) = 路由器輸出權重 (sparse, 只選 top-k)
- Eᵢ(x) = 第 i 個專家的輸出
```

---

## 4. 知名 MoE 模型

| 模型 | 專家數 | 激活數 | 總參數 | 有效參數 |
|------|:------:|:------:|:------:|:--------:|
| **Switch Transformers** | 128 | 1 | 1.6T | 12.5B |
| **Mixtral 8x7B** | 8 | 2 | 46.7B | 12B |
| **GShard** | 128 | 2 | 600B | 75B |
| **ST-MoE** | 64 | 2 | 269B | 40B |
| **DeepSeek-MoE** | 64 | 2 | 45B | 5B |

### 著名 MoE 架構比較

| 模型 | Router 策略 | Load Balancing | 特色 |
|------|-------------|----------------|------|
| **Switch** | Top-1 | Auxiliary Loss | 最早大規模 MoE |
| **Mixtral** | Top-2 | z-loss + Router Z-Loss | 開源效果好 |
| **GShard** | Top-2 | Auxiliary Loss | Google 訓練 |
| **ST-MoE** | Top-2 | 穩定訓練技巧 | 訓練穩定性好 |

---

## 5. 挑戰與解決方案

### 挑戰 1：訓練不穩定

**問題**：Router 可能倒塌到只選少數專家

**解決方案**：
- Load balancing loss
- Dropout on router
- Auxiliary z-loss (Mixtral)

### 挑戰 2：記憶體需求

**問題**：所有專家需要載入記憶體

**解決方案**：
- Expert Parallelism：將專家分布到不同 GPU
- 專家層級聯：不用時卸載
- MoE 與 Shared Expert：減少專家數

### 挑戰 3：通訊開銷

**問題**：跨 GPU 路由增加通訊

**解決方案**：
- 將相關專家放在同一節點
- 批次路由減少次數
- 結合 Pipeline Parallelism

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Sparse MoE** | MoE 的核心技術，稀疏激活 |
| **Token Routing** | 決定每個 token 去哪個專家 |
| **Expert Parallelism** | 分散式部署不同專家到不同 GPU |
| **KV Cache** | MoE 推理優化的重要技術 |
| **Transformer** | MoE 通常基於 Transformer 架構 |

---

## 7. 數學推導

### 門控機制數學

```
輸入 x ∈ ℝᵈ，N 個專家

Router 輸出:
  g(x) = softmax(W_g × x) ∈ ℝᴺ

選擇 Top-K:
  indices = top_k(g(x), k)
  g_k(x) = gather(g(x), indices)
  g_k(x) = g_k(x) / Σ g_k(x)  # 重新標準化

專家輸出:
  E_i(x) = f_i(x)  # 每個專家是獨立的 FFN

最終輸出:
  y = Σ_{i∈indices} g_k(x)_i × E_i(x)
```

### 負載均衡 Loss 推導

```python
def compute_load_balancing_loss(router_logits, expert_indices, num_experts):
    """
    目標: 均勻分配 load 到所有專家
    """

    # 計算每個 token 選擇的專家
    # shape: [batch, top_k]
    expert_mask = F.one_hot(expert_indices, num_experts)

    # 每個專家的使用頻率
    # shape: [num_experts]
    expert_fraction = expert_mask.sum(0) / expert_mask.sum()

    # 理想均勻分布
    ideal_fraction = 1.0 / num_experts

    # Loss = 分布的方差 (越小越好)
    lb_loss = -(
        expert_fraction * torch.log(expert_fraction + 1e-8)
    ).sum()

    return lb_loss * 0.01  # 權重係數
```

### 輔助 Loss 組合

```python
def total_loss(main_loss, router_logits, expert_indices, num_experts):
    """總 Loss = 主 Loss + 輔助 Loss"""

    # 1. 負載均衡 Loss
    lb_loss = compute_load_balancing_loss(
        router_logits, expert_indices, num_experts
    )

    # 2. Router z-loss (Mixtral)
    # 鼓勵 router logits 接近均勻分布
    z_loss = 0.001 * (router_logits ** 2).mean()

    # 3. 路由器 Dropout Loss
    # 防止過度擬合到特定專家
    router_dropout_loss = 0.1 * (
        router_logits.abs().mean() - 1.0
    ).abs()

    return main_loss + lb_loss + z_loss + router_dropout_loss
```

---

## 8. 路由策略詳解

### 專家選擇策略

| 策略 | 公式 | 優點 | 缺點 |
|------|------|------|------|
| **Top-K** | `top_k(softmax(Wx))` | 簡單有效 | 需要固定 K |
| **Noisy Top-K** | `top_k(softmax(Wx + ε))` | 更均衡 | 需調優噪聲 |
| **Hashing** | `h(x) mod N` | 無需訓練 | 確定性 |
| **Expert Choice** | 每個專家選 token | 負載均衡 | 通訊開銷 |

### Noisy Top-K 實作

```python
class NoisyTopKGate(nn.Module):
    def __init__(self, d_model, num_experts, top_k, noise_std=0.1):
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.noise_std = noise_std

    def forward(self, x):
        # 基礎 logits
        logits = self.gate(x)

        # 加入噪聲 (只在訓練時)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-K 選擇
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)

        # 遮罩其他專家
        full_logits = torch.full_like(logits, float('-inf'))
        full_logits.scatter_(2, indices, top_k_logits)

        # Softmax
        weights = F.softmax(full_logits, dim=-1)

        return weights, indices
```

### 專家特殊化分析

```python
def analyze_expert_specialization(model, test_data):
    """分析專家是否學到特殊化"""

    expert_usage = {i: 0 for i in range(model.num_experts)}
    expert_inputs = {i: [] for i in range(model.num_experts)}

    for batch in test_data:
        # 追蹤哪些 token 去哪個專家
        routing = model.get_routing_decisions(batch)

        for token_idx, expert_idx in enumerate(routing):
            expert_usage[expert_idx] += 1
            expert_inputs[expert_idx].append(batch[token_idx])

    # 計算專家之間的差異性
    specializations = []
    for expert_id, inputs in expert_inputs.items():
        embedding = compute_mean_embedding(inputs)
        specializations.append(embedding)

    # 專家之間的相似度
    similarity_matrix = cosine_similarity(specializations)
    # 低相似度 = 高特殊化

    return {
        "usage_distribution": expert_usage,
        "specialization_score": 1 - similarity_matrix.mean()
    }
```

---

## 9. 訓練穩定性

### Router Collapse 問題

```
問題描述:
  Router 學習到總是選擇同一個專家
  導致其他專家未被訓練

原因:
  1. 初始隨機導致某專家略強
  2. 正反饋放大差異
  3. 主 Loss 不鼓勵均勻分布
```

### 穩定訓練技巧

| 技巧 | 說明 | 效果 |
|------|------|------|
| **Router Dropout** | 訓練時隨機遮罩部分 logits | 防止過擬合 |
| **z-loss** | 懲罰大的 logits | 穩定 softmax |
| **epsilon** | 避免 log(0) | 數值穩定 |
| **Warm-up** | 前期固定 router | 防止初期 collapse |
| **Expert Capacity** | 限制每專家最大 token 數 | 防止負載過度不均 |

### Expert Capacity 機制

```python
class MoELayerWithCapacity(nn.Module):
    def __init__(self, num_experts, top_k, capacity_factor=1.5):
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity = capacity_factor

    def forward(self, x):
        # 標準 routing
        weights, indices = self.router(x)

        # 計算每個專家的 capacity
        tokens_per_expert = F.one_hot(
            indices.flatten(),
            self.num_experts
        ).sum(0)

        capacity = (
            x.size(0) * self.top_k / self.num_experts
        ) * self.capacity

        # 超過 capacity 的 token 被丟棄
        mask = tokens_per_expert > capacity
        weights = weights.masked_fill(mask.unsqueeze(1), 0)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

        return self.compute_experts(x, weights, indices)
```

---

## 10. 推理優化

### 推理延遲構成

```
MoE 推理延遲 = 計算延遲 + 通訊延遲

計算延遲:
  - 共享層 (Self-Attention): 30%
  - MoE FFN (激活的專家): 40%
  - 其他計算: 30%

通訊延遲 (跨 GPU):
  - Token dispatch: 10-20%
  - Expert 结果 gather: 10-20%
```

### 推理優化技術

```python
class MoEInferenceOptimizer:
    @staticmethod
    def prefetch_experts(model, batch_size):
        """預加載即將使用的專家"""

        # 根據歷史模式預測
        predicted_experts = model.predict_next_experts()

        # 預加載到 GPU
        for expert_id in predicted_experts:
            model.load_expert_to_gpu(expert_id)

    @staticmethod
    def cache_routing_decisions(model, input_ids):
        """快取 routing 決策"""

        # 相同輸入 → 相同路由
        cache_key = hash(input_ids)

        if cache_key in model.routing_cache:
            return model.routing_cache[cache_key]

        routing = model.compute_routing(input_ids)
        model.routing_cache[cache_key] = routing

        return routing
```

### 批次優化

```python
def batch_moe_inference(model, batched_inputs):
    """批次 MoE 推理優化"""

    # 1. 按專家分組
    expert_groups = group_by_expert(batched_inputs)

    # 2. 批量調用每個專家
    expert_outputs = {}
    for expert_id, tokens in expert_groups.items():
        expert_outputs[expert_id] = model.experts[expert_id](tokens)

    # 3. 合併結果
    output = reorder_outputs(expert_outputs, original_order)

    return output
```

---

## 11. 部署架構

### Expert Parallelism

```
8 GPU, 8 Experts:

GPU 0: Expert 0, Expert 4
GPU 1: Expert 1, Expert 5
GPU 2: Expert 2, Expert 6
GPU 3: Expert 3, Expert 7

Token flow:
  GPU 0 ──( Expert 0 output )──> 需要的地方
  GPU 1 ──( Expert 1 output )──> 需要的地方
  ...
```

### DeepSpeed MoE 支援

```python
import deepspeed

# DeepSpeed MoE 配置
ds_config = {
    "train_batch_size": 32,
    "moe": {
        "enabled": True,
        "num_experts": 8,
        "ep_size": 4,  # Expert Parallelism size
        "moe_param_group": True
    }
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)
```

---

## 12. 基準測試

### 效能比較

| 模型 | 參數量 | 推理速度 (tokens/s) | 記憶體 |
|------|--------|-------------------|--------|
| **LLaMA 70B Dense** | 70B | 100 (A100) | 140GB |
| **Mixtral 8x7B** | 46B | 250 (A100) | 95GB |
| **LLaMA 7B** | 7B | 500 (A100) | 14GB |

### FLOPs 比較

```
Dense Transformer (L 層, d_model):
  FLOPs = 4 × L × d_model² + L × n_heads × d_head²

MoE Transformer (L 層, N 個專家, 激活 K 個):
  FLOPs = 4 × L × d_model² + L × N/k × d_model × d_ff

範例 (LLaMA 70B vs Mixtral 8x7B):
  LLaMA 70B: ~1.4 TFLOPs/token
  Mixtral 8x7B: ~0.5 TFLOPs/token
  → 2.8x 計算節省
```

---

## 13. 相關主題

| 技術 | 關係 |
|------|------|
| **Sparse MoE** | MoE 的核心技術，稀疏激活 |
| **Token Routing** | 決定每個 token 去哪個專家 |
| **Expert Parallelism** | 分散式部署不同專家到不同 GPU |
| **KV Cache** | MoE 推理優化的重要技術 |
| **Transformer** | MoE 通常基於 Transformer 架構 |

---

## 延伸閱讀

- [Switch Transformers Paper](https://arxiv.org/abs/2101.03961)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [MoE 詳解](https://arxiv.org/abs/2208.02816)
- [ST-MoE Paper](https://arxiv.org/abs/2202.08906)
- [GShard Paper](https://arxiv.org/abs/2006.16668)
- [DeepSpeed MoE](https://www.microsoft.com/en-us/research/blog/deepspeed-speed-bert-training-up-to-3x-faster/)