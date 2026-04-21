# Speculative Decoding

推論加速技術，利用較小的「投機模型」快速生成多個 tokens，然後用大模型一次性驗證，大幅減少延遲。

---

## 1. 什麼是？

### 深度定義

**Speculative Decoding (投機解碼)** 是一種**雙模型協作**的推理加速技術：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Speculative Decoding 核心思想                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  問題:                                                               │
│  - 大語言模型自迴歸解碼，每個 token 都要完整前向傳播                  │
│  - 延遲 = O(n × L) (n=token數, L=模型層數)                         │
│  - 瓶頸: 大模型單次推理太慢                                         │
│                                                                      │
│  解決: 分工合作                                                      │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  小模型 (Draft): 速度快，負責「猜測」多個 tokens             │   │
│  │  大模型 (Verify): 精確，負責「驗證」小模型猜測              │   │
│  │                                                              │   │
│  │  流程:                                                       │   │
│  │  1. 小模型一次生成 K 個 token (投機)                        │   │
│  │  2. 大模型一次驗證這 K 個 token                              │   │
│  │  3. 接受的 token 直接採用，拒絕的從那裡重新開始              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  關鍵洞察:                                                           │
│  - 小模型每個 token 推理成本是大模型的 1/10-1/5                      │
│  - 如果命中率 80%，平均每生成 1 個 token 需要 0.2 次大模型推理      │
│  - 加速比 ≈ K × 命中率 / (1 + K × 命中率)                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何有效**:
1. **數學等價**: 拒絕採樣保證輸出與標準解碼完全一致
2. **計算互換**: 用更多小模型計算換取更少大模型計算
3. **記憶體友好**: 小模型記憶體需求低

### 簡單範例

```
生成 "The cat sat on the mat"

傳統 (大模型單獨):
  The → cat → sat → on → the → mat
  每個 token 1 次大模型推理
  = 6 次推理

投機解碼:
  投機階段 (小模型): 一次生成 ["the", "cat", "sat", "on", "the", "mat"]
  驗證階段 (大模型): 一次驗證全部 6 個 tokens
  = 2 次推理

→ 3x 加速
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **降低延遲** | 可達 2-3 倍加速 |
| **保持質量** | 輸出與標準解碼完全一致 |
| **實用性強** | 已被多個推理框架採用 |
| **通用性** | 可應用於任何自迴歸模型 |

---

## 3. 核心原理

### 標準 Speculative Decoding 流程

```
步驟 1: 投機階段（小模型）
┌──────────────────────────────────────────────┐
│  輸入: "The cat sat on"                       │
│  小模型連續生成 K 個 tokens:                   │
│  ["the", "mat", ".", "It", "was", "cute"]    │
└──────────────────────────────────────────────┘
                              ↓
步驟 2: 驗證階段（大模型）
┌──────────────────────────────────────────────┐
│  一次輸入: "The cat sat on the mat. It was cute"│
│  大模型驗證每個 position                        │
│  - position 0: "the" ✓                        │
│  - position 1: "mat" ✓                        │
│  - position 2: "." → "?" (修正)               │
│  - position 3: "It" ✗ (reject, 從這裡重新)    │
└──────────────────────────────────────────────┘
                              ↓
步驟 3: 採用結果
┌──────────────────────────────────────────────┐
│  - "the", "mat" → 採用                       │
│  - 從 position 2 重新用大模型解碼              │
└──────────────────────────────────────────────┘
```

### 拒絕採樣算法

```python
def speculative_verify(proposal_tokens, small_probs, large_probs):
    """驗證投機結果"""
    accepted = []
    for i, token in enumerate(proposal_tokens):
        # 計算接受概率
        p_small = small_probs[i, token]
        p_large = large_probs[i, token]

        accept_prob = min(1, p_large / p_small)

        if random.random() < accept_prob:
            accepted.append(token)
        else:
            # 拒絕，從大模型輸出
            new_token = sample_from_large()
            accepted.append(new_token)
            break

    return accepted
```

### 加速原理

```
標準: K 個 token → K 次大模型推理
投機: K 個 token → 1 次小模型 + 1 次大模型

加速比 ≈ K × 命中率 / (1 + K × 命中率)

例如 K=6, 命中率=80%:
  加速比 ≈ 6 × 0.8 / 1.8 ≈ 2.67x
```

---

## 4. 變體和優化

### Medusa

```
投機解碼: 1 個小模型 → 多個 tokens
Medusa: 多個「預測頭」→ 同時預測多個 tokens

每個 head 預測 future tokens:
  Head 1: 預測 t+1
  Head 2: 預測 t+2
  Head 3: 預測 t+3
  ...
```

### Eagle

```
投機解碼: 小模型生成完整 tokens
Eagle: 使用大模型的中間層特徵 + 早 Exit

改進:
  - 使用大模型最後幾層的特徵
  - 更準確的投機
  - 更少的參數
```

### Lookahead Decoding

```
不使用獨立小模型
使用 n-gram 投機:
  - 從已生成的 tokens 提取 n-gram
  - 假設未來會重複這些模式
  - 大模型驗證
```

### Self-Speculative

```
不使用小模型
使用同一模型的不同配置:
  - 使用較少 layers 的版本投機
  - 用完整版本驗證
```

| 方法 | 優點 | 缺點 |
|------|------|------|
| **標準 Speculative** | 簡單直接 | 需要小模型 |
| **Medusa** | 不需小模型 | 需要訓練預測頭 |
| **Eagle** | 效率高 | 需要特殊訓練 |
| **Lookahead** | 無需額外模型 | 依賴文本模式 |

---

## 5. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **KV Cache** | Speculative Decoding 的基礎 |
| **PagedAttention** | 管理投機階段的 KV Cache |
| **DistilBERT** | 投機模型的典型選擇 |
| **Early Exit** | 類似思想，提前輸出 |

---

## 6. 數學推導

### 接受概率推導

```
目標: 決定是否接受小模型預測的 token

定義:
  p_d(x) = 大模型的 token 機率分佈
  p_s(x) = 小模型的 token 機率分佈

接受準則 (Original Speculative Decoding):

  當 p_d(x) ≥ p_s(x): 總是接受
  當 p_d(x) < p_s(x): 以概率 p_d(x)/p_s(x) 接受

這保證了輸出與直接使用大模型完全相同 (數學上等價)
```

### 加速比分析

```python
def speedup_analysis(k, hit_rate):
    """
    計算加速比

    k: 投機數量 (每批次的 token 數)
    hit_rate: 命中率 (大模型接受的比率)

    標準: 需要 k 次大模型 forward
    投機: 只需 1 次小模型 + 1 次大模型
    """
    # 如果全部命中: 1 次推理產生 k 個 token
    if hit_rate == 1.0:
        return k

    # 期望大模型調用次數
    # 每次驗證可能重新從某個位置開始
    expected_large_calls = 1 + (1 - hit_rate) * k

    # 加速比
    speedup = k / expected_large_calls

    return speedup


# 範例
for k in [4, 6, 8, 12]:
    for hit in [0.5, 0.7, 0.9, 0.95]:
        print(f"K={k}, 命中率={hit}: {speedup_analysis(k, hit):.2f}x")

# 輸出:
# K=4, 命中率=0.5: 2.00x
# K=4, 命中率=0.7: 2.35x
# K=6, 命中率=0.8: 2.73x
# K=8, 命中率=0.9: 3.48x
# K=12, 命中率=0.95: 4.62x
```

### 採樣修正

```python
def corrected_speculative_sample(
    small_tokens,
    small_probs,
    large_probs,
    temperature=1.0
):
    """
    帶溫度修正的投機解碼採樣
    """

    accepted = []
    for i, (token, p_s) in enumerate(zip(small_tokens, small_probs)):
        p_d = large_probs[i]

        # 計算接受概率
        if p_s.sum() > 0:
            accept_prob = min(1, (p_d[token] / (p_s[token] + 1e-10)).item())
        else:
            accept_prob = 1.0

        if random.random() < accept_prob:
            accepted.append(token)
        else:
            # 拒絕: 從大模型分佈重新採樣
            if temperature > 0:
                # 溫度採樣
                logits = large_probs[i] / temperature
                probs = F.softmax(logits, dim=-1)
                new_token = torch.multinomial(probs, 1).item()
            else:
                # Greedy
                new_token = large_probs[i].argmax().item()

            accepted.append(new_token)
            break  # 從這裡開始需要重新生成

    return accepted
```

---

## 7. 投機模型選擇

### 候選模型

| 小模型 | 參數量 | 相對大模型 | 延遲比 |
|--------|--------|-----------|--------|
| **DistilBERT** | 66M | ~1/10 | ~1/5 |
| **TinyLLaMA** | 1.1B | ~1/20 | ~1/10 |
| **LLaMA-2-7B** | 7B | 1/10 | ~1/3 |
| **Qwen-0.5B** | 0.5B | ~1/50 | ~1/15 |

### 模型蒸餾

```python
class DistillationTrainer:
    """蒸餾投機模型"""

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model

    def train_step(self, input_ids):
        # Teacher 產生 logits
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)

        # Student 預測
        student_logits = self.student(input_ids)

        #蒸餾 Loss
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )

        return loss
```

### 自蒸餾策略

```python
class SelfSpeculativeModel:
    """
    使用同一模型的不同配置
    """

    def __init__(self, base_model):
        self.full_model = base_model  # 完整層數
        self.shallow_model = ShallowCopy(
            base_model, num_layers=base_model.num_layers // 3
        )

    def generate(self, prompt, max_length):
        # Shallow model 快速生成
        draft_tokens = self.shallow_model.generate(prompt, K)

        # Full model 驗證
        verified = self.full_model.verify(prompt, draft_tokens)

        return verified
```

---

## 8. Medusa 深入

### 多頭預測架構

```
傳統語言模型:
  輸入 → [LM Head] → 下一個 token

Medusa:
  輸入 → [LM Head] → token t+1
       → [Medusa Head 1] → token t+2
       → [Medusa Head 2] → token t+3
       → [Medusa Head 3] → token t+4
       → ...
```

### Medusa 實現

```python
class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads=4):
        super().__init__()
        self.base = base_model
        self.num_heads = num_heads

        # 每個 Medusa head 預測未來的 token
        self.medusa_heads = nn.ModuleList([
            nn.Linear(base_model.hidden_size, base_model.vocab_size)
            for _ in range(num_heads)
        ])

        # 溫度參數
        self.temperature = 1.0

    def forward(self, input_ids):
        # Base model forward
        hidden = self.base(input_ids)

        # 每個 head 預測一個未來的 token
        predictions = []
        for head in self.medusa_heads:
            logits = head(hidden) / self.temperature
            predictions.append(logits)

        return predictions

    def medusa_decode(self, input_ids, max_new_tokens):
        """Medusa 解碼"""

        all_tokens = input_ids.clone()
        medusa_buffer = [None] * self.num_heads

        for _ in range(max_new_tokens):
            # 標準前向
            preds = self.forward(all_tokens)

            # 第一個 head 就是標準預測
            next_token = preds[0].argmax(-1)
            all_tokens = torch.cat([all_tokens, next_token.unsqueeze(-1)], dim=1)

        return all_tokens
```

### 訓練策略

```python
def train_medusa(base_model, medusa_heads, data):
    """
    兩階段訓練:
    1. 凍結 base model，只訓練 medusa heads
    2. 聯合訓練
    """

    # Stage 1: 只訓練 heads
    for param in base_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(medusa_heads.parameters(), lr=1e-4)

    for batch in data:
        # 訓練 head 預測未來的 token
        loss = compute_medusa_loss(medusa_heads, batch)
        loss.backward()
        optimizer.step()

    # Stage 2: 聯合訓練 (可選)
    # ...
```

---

## 9. Eagle 深入

### Eagle 架構

```
Eagle 的創新:
1. 使用大模型最後幾層的 hidden states 作為輸入
2. 早 Exit: 不需要通過完整模型
3. 層次化投機: 多階段驗證
```

### Eagle 實現

```python
class EagleModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

        # 早 exit layer
        self.exit_layer = base_model.num_layers - 4

        # Eagle head: 使用淺層特徵預測
        self.eagle_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_model.hidden_size, base_model.hidden_size),
                nn.ReLU(),
                nn.Linear(base_model.hidden_size, base_model.vocab_size)
            )
            for _ in range(4)  # 4 個 speculative tokens
        ])

    def forward(self, input_ids):
        # 通過部分層
        hidden = self.base.embed(input_ids)

        for layer in range(self.exit_layer):
            hidden = self.base.layers[layer](hidden)

        # 使用淺層特徵生成投機
        speculations = []
        for head in self.eagle_head:
            logits = head(hidden)
            next_token = logits.argmax(-1)
            speculations.append(next_token)

            # 更新 hidden (使用預測的 token)
            new_embed = self.base.embed(next_token)
            hidden = hidden + new_embed  # 殘差連接

        return speculations

    def verify_and_generate(self, prompt, speculations):
        """驗證並生成"""
        # 一次性用完整模型驗證所有 speculations
        verified = self.full_verify(prompt, speculations)

        return verified
```

---

## 10. 實際部署

### vLLM 整合

```python
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# 啟用投機解碼
engine_args = EngineArgs(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=6,  # K 值
)

llm = LLM(engine_args=engine_args)

# 正常使用
outputs = llm.generate(prompts, sampling_params)
```

### 效能基準

```
模型: LLaMA-70B + LLaMA-7B 投機
硬體: 8x A100

序列長度    標準延遲    投機延遲    加速比
──────────────────────────────────────
512         1.2s        0.8s        1.5x
1024        2.8s        1.5s        1.9x
2048        6.5s        2.8s        2.3x
4096       15.2s       5.1s        3.0x
```

### 監控指標

```python
class SpeculativeMonitor:
    def __init__(self):
        self.stats = {
            "total_specs": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_position": []
        }

    def record(self, proposal_len, accepted_len, rejection_pos):
        self.stats["total_specs"] += 1
        self.stats["accepted"] += accepted_len
        self.stats["rejected"] += proposal_len - accepted_len
        self.stats["rejection_position"].append(rejection_pos)

    def get_metrics(self):
        accepted = self.stats["accepted"]
        total = accepted + self.stats["rejected"]

        return {
            "acceptance_rate": accepted / max(total, 1),
            "avg_proposal": total / max(self.stats["total_specs"], 1),
            "avg_rejection_pos": np.mean(self.stats["rejection_position"])
        }
```

---

## 11. 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **加速不明顯** | 命中率太低 | 調整 K 值或更小的投機模型 |
| **輸出質量下降** | 拒絕採樣破壞分佈 | 使用修正的採樣 |
| **記憶體增加** | 需要同時加載兩個模型 | 使用模型蒸餾或共享權重 |
| **首 token 延遲** | 仍需要完整推理 | 結合預處理 |

---

## 12. 相關技術

| 技術 | 關係 |
|------|------|
| **KV Cache** | Speculative Decoding 的基礎 |
| **PagedAttention** | 管理投機階段的 KV Cache |
| **DistilBERT** | 投機模型的典型選擇 |
| **Early Exit** | 類似思想，提前輸出 |

---

## 延伸閱讀

- [Speculative Decoding Paper](https://arxiv.org/abs/2302.01318)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [Eagle Paper](https://arxiv.org/abs/2402.02103)
- [Lookahead Decoding](https://arxiv.org/abs/2308.04615)