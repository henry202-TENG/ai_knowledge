# MoE

稀疏化模型架構，透過動態激活不同的「專家」網路來處理不同的輸入，在保持強大能力的同時大幅降低計算成本。

---

## 1. 什麼是？

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

## 延伸閱讀

- [Switch Transformers Paper](https://arxiv.org/abs/2101.03961)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [MoE 詳解](https://arxiv.org/abs/2208.02816)
- [ST-MoE Paper](https://arxiv.org/abs/2202.08906)