# RLHF

透過人類回饋來學習的訓練技術，結合強化學習與人類偏好來讓模型輸出更符合人類期望。

---

## 1. 什麼是？

### 簡單範例

```
用戶: "如何製作炸彈？"

預訓練模型:
  "Here's how to make a bomb: ..."  (可能有害)

RLHF 之後:
  "對不起，我無法幫助這個請求。"  (安全回覆)
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **對齊人類價值觀** | 讓模型回答更有幫助、更安全 |
| **超越簡單損失函數** | 無法用簡單標籤監督的任務 |
| **ChatGPT 成功的關鍵** | RLHF 是 ChatGPT 背後的核心技術 |
| **可擴展監督** | 人類反饋可重複使用 |

---

## 3. 核心原理

### RLHF 三階段流程

```
┌─────────────────────────────────────────────────────────────────┐
│  階段 1: 監督微調 (SFT)                                          │
│  原始模型 → [人類標註的問答對] → SFT 模型                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  階段 2: 訓練獎勵模型 (Reward Model)                            │
│  SFT 輸出 → [人類偏好比較] → 獎勵模型 (RM)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  階段 3: 近端策略優化 (PPO)                                      │
│  SFT 輸出 → [PPO + Reward] → 最終模型                           │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: SFT (Supervised Fine-Tuning)

```
訓練數據:
  Prompt: "如何修復這個 bug？"
  Response: "首先檢查錯誤日誌，然後..."

使用標準語言模型訓練:
  Loss = -Σ log P(token | context)
```

### Stage 2: Reward Model

```
訓練數據格式 (人類偏好):
  Prompt A: "如何學習編程？" → Response: "建議先學 Python"
  Prompt A: "如何學習編程？" → Response: "我不知道"

人類標註:
  Response 1 > Response 2 (第一個更好)

訓練目標: 讓 RM 預測人類偏好
  Loss = -log(σ(r₁ - r₂))

其中 r = RM(prompt, response)
```

### Stage 3: PPO 優化

```
PPO Loss = E[clip(π, π_ref) × A] - β × KL(π || π_ref) + γ × V

組件說明:
- clip(π, π_ref): 限制策略更新幅度
- A: 優勢函數 (來自 Reward Model)
- KL: 防止偏離 SFT 模型太遠
- V: 價值函數，估計預期回報
- β: KL 係數，通常 0.01-0.1
- γ: 價值損失權重
```

### KL Penalty 詳解

```
為什麼需要 KL Penalty？

沒有 KL:
  模型可能會偏離原本行為，變得無法預測

有 KL:
  Loss = Reward - β × KL(π_new || π_sft)

  效果:
  - 獎勵增加時允許更新
  - 獎勵減少時禁止過度偏離
  - 保持模型的基礎能力
```

### RLHF vs 傳統訓練

| 訓練方式 | 數據需求 | 學習目標 | 複雜度 |
|----------|----------|----------|--------|
| **Pretraining** | 大量未標註文本 | 預測下一個 token | 低 |
| **SFT** | 人類標註問答 | 模仿人類回覆 | 中 |
| **RLHF** | 人類偏好比較 | 優化整體質量 | 高 |

---

## 4. 實現細節

### PPO 訓練迴圈

```python
def ppo_training(model, ref_model, reward_model, prompts, data):
    for epoch in range(num_epochs):
        # 1. 使用當前策略生成回覆
        responses = model.generate(prompts)

        # 2. 計算獎勵
        rewards = reward_model(prompts, responses)

        # 3. 計算 KL Penalty
        kl = compute_kl(model, ref_model, prompts, responses)

        # 4. 組合最終獎勵
        final_rewards = rewards - β * kl

        # 5. PPO 更新
        for batch in data:
            # 計算優勢估計
            advantages = compute_gae(
                values, rewards, dones, gamma=0.95, lam=0.95
            )

            # PPO 損失
            ppo_loss = -min(
                ratio * advantages,
                clip(ratio, 1-ε, 1+ε) * advantages
            )

            # 反向傳播
            optimizer.step()
```

### 常見問題與解決

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **獎勵被 gaming** | 模型找到捷徑獲得高分 | 加入穩定項、變換獎勵信號 |
| **訓練不穩定** | PPO 更新幅度過大 | 減小 learning rate、調整 clip |
| **能力退化** | 過度優化獎勵 | 增加 KL penalty、混合 SFT 數據 |

---

## 5. 實作工具

| 框架 | 說明 |
|------|------|
| **TRL** (Transformers RL) | HuggingFace 開源，支援 DPO/PPO |
| **DeepSpeed-Chat** | Microsoft，支援 RLHF/DPO |
| **OpenChatKit** | Together AI |
| **Anthropic Constitutional AI** | Anthropic 的對齊方法 |

### TRL 使用範例

```python
from trl import SFTTrainer, PPOTrainer, RewardTrainer
from trl.core import set_seed

# Step 1: SFT
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    packing=True,
    max_seq_length=2048,
)
sft_trainer.train()

# Step 2: Reward Model
reward_trainer = RewardTrainer(
    model=model,
    train_dataset=reward_dataset,
)
reward_trainer.train()

# Step 3: PPO
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    train_dataset=prompts_dataset,
)
ppo_trainer.train()
```

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **DPO** | RLHF 的簡化替代方案，無需 PPO |
| **PPO** | RLHF 的核心強化學習算法 |
| **Reward Model** | RLHF 的第二階段 |
| **Constitutional AI** | Anthropic 的 AI 對齊方法 |
| **KTO** | Kahneman-Tversky Optimization |

---

## 7. 獎勵模型進階

### 獎勵模型架構

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # 添加獎勵頭
        self.reward_head = nn.Linear(
            base_model.config.hidden_size,
            1
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取最後隱藏層的 [EOS] token 表示
        last_hidden = outputs.last_hidden_state
        eos_idx = attention_mask.sum(-1) - 1

        # 收集每個序列結尾的 hidden states
        batch_indices = torch.arange(last_hidden.size(0))
        eos_hidden = last_hidden[batch_indices, eos_idx]

        # 計算獎勵
        reward = self.reward_head(eos_hidden)

        return reward
```

### 偏好數據構建

```python
def create_preference_dataset(responses, human_ratings):
    """
    將人類評分轉為偏好對

    ratings: dict, prompt_id -> [(response, score), ...]
    """
    dataset = []

    for prompt_id, response_scores in human_ratings.items():
        # 按分數排序
        sorted_responses = sorted(
            response_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # 創建偏好對
        for i in range(len(sorted_responses) - 1):
            for j in range(i + 1, len(sorted_responses)):
                if sorted_responses[i][1] > sorted_responses[j][1]:
                    dataset.append({
                        "prompt": prompt_id,
                        "chosen": sorted_responses[i][0],
                        "rejected": sorted_responses[j][0]
                    })

    return dataset
```

### 獎勵模型 Loss 推導

```
訓練目標: 最大化人類偏好的對數機率

給定偏好數據 (prompt, chosen, rejected):

Loss = -log σ(r_chosen - r_rejected)

其中:
  σ(x) = 1 / (1 + e^(-x))  (sigmoid)
  r = Reward Model 輸出的獎勵值

直觀理解:
  - 當 r_chosen > r_rejected 時，σ(positive) → 1，Loss → 0
  - 當 r_chosen < r_rejected 時，σ(negative) → 0，Loss → 大
```

---

## 8. PPO 數學推導

### 策略梯度

```
目標: 最大化期望獎勵

J(π) = E_{τ~π}[R(τ)]

其中 R(τ) 是軌跡 τ 的累積獎勵

梯度:
  ∇J = E_{τ~π}[∇ log π(τ) × R(τ)]

問題: 高方差估計

解決方案: 引入基線 b
  ∇J = E[∇ log π(a|s) × (Q(s,a) - b(s))]
```

### GAE (Generalized Advantage Estimation)

```python
def compute_gae(
    rewards,           # [T]
    values,           # [T]
    dones,            # [T]
    gamma=0.99,       # 折扣因子
    lam=0.95          # GAE 參數
):
    """
    GAE: 平衡 bias 和 variance
    lam = 0: Monte Carlo (低 variance, high bias)
    lam = 1: TD(0) (high variance, low bias)
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        # TD error
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE 累積
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages)
    returns = advantages + values

    return advantages, returns
```

### PPO Clip 機制

```python
def ppo_clip_loss(
    log_probs,      # 新策略的 log π(a|s)
    old_log_probs,  # 舊策略的 log π(a|s)
    advantages,
    clip_eps=0.2
):
    """
    PPO 的核心: 限制策略更新幅度

    目標: 最大化
      min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)

    其中 r(θ) = exp(log π_θ - log π_θ_old)
    """

    # 計算比率
    ratio = torch.exp(log_probs - old_log_probs)

    # 裁剪
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # 取較小值 (保守更新)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    return loss.mean()
```

---

## 9. 訓練穩定性

### 常見問題診斷

| 問題徵兆 | 可能原因 | 解決方案 |
|----------|----------|----------|
| Loss 爆炸 | learning rate 過高 | 降低 LR |
| 獎勵快速上升後下降 | reward gaming | 增加 KL  penalty |
| 生成重複內容 | 策略崩塌 | 增加 entropy bonus |
| 模型拒答過多 | 安全獎勵過強 | 調整 reward 權重 |
| 能力退化 | 過度 KL 約束 | 混合 SFT 數據 |

### 穩定訓練技巧

```python
class StablePPOTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # 1. Learning Rate Schedule
        self.lr_scheduler = warmup_cosine_lr(
            warmup_steps=100,
            total_steps=10000,
            min_lr_ratio=0.1
        )

        # 2. Gradient Clipping
        self.max_grad_norm = 1.0

        # 3. Early Stopping
        self.kl_threshold = 0.02

        # 4. Reward Normalization
        self.reward_rms = RunningMeanStd()

    def train_step(self, prompts):
        # 標準 PPO 步驟
        responses = self.generate(prompts)
        rewards = self.compute_rewards(prompts, responses)

        # 獎勵標準化
        rewards = self.reward_rms.normalize(rewards)

        # KL 檢查
        kl = self.compute_kl()
        if kl > self.kl_threshold:
            logger.warning(f"KL {kl:.3f} exceeds threshold")

        # 更新
        loss = self.compute_ppo_loss()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()
```

### Entropy Bonus

```python
def entropy_bonus(logits):
    """
    鼓勵探索，防止策略過於確定性

    H(π) = -Σ π(a|s) log π(a|s)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1)

    return entropy.mean() * 0.01  # 權重係數
```

---

## 10. 超參數調優

### 關鍵超參數

| 參數 | 典型值 | 影響 |
|------|--------|------|
| **PPO clip (ε)** | 0.1-0.3 | 太小:訓練慢，太大:不穩 |
| **KL penalty (β)** | 0.01-0.1 | 太小:偏離SFT，太大:訓練慢 |
| **Value loss weight** | 0.5-1.0 | 影響價值估計準確性 |
| **Entropy bonus** | 0.001-0.01 | 鼓勵多樣性 |
| **Batch size** | 64-512 | 影響訓練穩定性 |
| **Learning rate** | 1e-6 - 1e-5 | 需要精細調整 |
| **Epochs per batch** | 1-4 | 通常 1 最好 |

### KL Target 策略

```python
class AdaptiveKLPenalty:
    """自適應 KL  penalty"""

    def __init__(self, target_kl=0.01):
        self.target_kl = target_kl
        self.beta = 1.0

    def update(self, actual_kl):
        """根據實際 KL 動態調整 β"""

        if actual_kl < self.target_kl * 0.8:
            # KL 太低，減小 penalty
            self.beta *= 0.8
        elif actual_kl > self.target_kl * 1.2:
            # KL 太高，增加 penalty
            self.beta *= 1.2

        return self.beta
```

---

## 11. 獎勵塑形

### Reward Hacking 問題

```
問題: 模型找到「作弊」方式獲得高分，
      而不是真正完成任務

範例:
  任務: "寫一個詩"
  獎勵: 生成文字的長度
  後果: 模型生成超長文字，質量很低
```

### 獎勵塑形技術

```python
def shape_reward(
    raw_reward,
    prompt,
    response,
    metrics
):
    """
    獎勵塑形: 將原始獎勵轉為更豐富的信號
    """

    reward = raw_reward

    # 1. 格式獎勵
    if "```" in response and "```" in response[::-1]:
        reward += 0.1  # 有程式碼塊

    # 2. 長度懲罰
    if len(response) < 50:
        reward -= 0.1  # 太短

    if len(response) > 5000:
        reward -= 0.1  # 太長

    # 3. 重複懲罰
    unique_ratio = len(set(response)) / len(response)
    if unique_ratio < 0.5:
        reward -= 0.2

    # 4. 毒性檢測
    toxicity = metrics.get("toxicity", 0)
    reward -= toxicity * 0.5

    return reward
```

### Constitutional AI 思路

```
Anthropic 的 Constitutional AI:

1. 讓 LLM 生成行為準則 (Constitution)
2. 用這些準則評估回覆
3. 透過 RLAIF (RL from AI Feedback) 訓練

優勢:
- 減少人類標註需求
- 可擴展
- 可解釋
```

---

## 12. 實踐考量

### 訓練資源需求

| 模型大小 | GPU 數 | 訓練時間 |
|----------|--------|----------|
| 7B | 8 A100 | ~1 天 |
| 13B | 16 A100 | ~2 天 |
| 70B | 128 A100 | ~1 週 |

### 數據量級

| 階段 | 數據量 | 品質要求 |
|------|--------|----------|
| SFT | 10K-100K | 高品質 |
| Reward Model | 100K-1M | 多樣性 |
| PPO | 10K-100K | 變化性 |

---

## 13. 相關主題

| 技術 | 關係 |
|------|------|
| **DPO** | RLHF 的簡化替代方案，無需 PPO |
| **PPO** | RLHF 的核心強化學習算法 |
| **Reward Model** | RLHF 的第二階段 |
| **Constitutional AI** | Anthropic 的 AI 對齊方法 |
| **KTO** | Kahneman-Tversky Optimization |

---

## 延伸閱讀

- [InstructGPT Paper (OpenAI)](https://arxiv.org/abs/2203.02155)
- [RLHF from Human Feedback](https://arxiv.org/abs/2206.07682)
- [HuggingFace TRL](https://huggingface.co/docs/trl/index)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DeepMind RLHF](https://arxiv.org/abs/2204.05862)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)