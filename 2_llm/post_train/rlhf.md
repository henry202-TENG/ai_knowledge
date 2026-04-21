# RLHF

透過人類回饋來學習的訓練技術，結合強化學習與人類偏好來讓模型輸出更符合人類期望。

---

## 1. 什麼是？

### 深度定義

**RLHF (Reinforcement Learning from Human Feedback)** 是一種將強化學習與人類偏好結合的訓練範式，其核心目標是解決**規範對齊問題 (Alignment Problem)**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RLHF 解決的核心問題                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  預訓練目標: 預測下一個 token                                         │
│  ↓                                                                    │
│  問題: 模型學會了語言模式，但不知道「什麼是好的回覆」                    │
│                                                                      │
│  RLHF 解決方案:                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 階段 1: SFT (行為克隆)                                         │   │
│  │   - 學習人類標註的高品質回覆                                   │   │
│  │   - 建立基礎對話能力                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 階段 2: Reward Model (偏好學習)                               │   │
│  │   - 學習人類偏好的隱含獎勵函數                                 │   │
│  │   - 將主觀判斷轉為可優化的信號                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 階段 3: PPO 優化 (策略學習)                                   │   │
│  │   - 在獎勵信號指導下進一步提升                                 │   │
│  │   - 保持與 SFT 模型的相似性 (KL penalty)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

### 深度價值分析

#### 價值 1：解決「主觀質量」的優化問題

**傳統監督學習的局限**：
```
任務: 判斷回覆質量
標籤: "好" 或 "不好"

問題:
1. 標籤不一致 - 不同標註者對「好」的理解不同
2. 難以量化 - 什麼構成「好」很難用簡單規則定義
3. 梯度失效 - 二元分類 loss 難以捕捉細緻的質量差異
```

**RLHF 的解決方案**：
```
相對偏好學習:
  - 不要求絕對標籤
  - 只比較「A 比 B 更好」
  - 更穩定、更一致性

優勢:
  - 減少標註主觀性影響
  - 捕捉細緻質量差異
  - 可累積人類判斷
```

#### 價值 2：可擴展的對齊

```
┌──────────────────────────────────────────────────────────────┐
│                    對齊成本 vs 模型能力                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  傳統方法:                                                    │
│    對齊成本 ∝ 模型能力                                       │
│    → 更強的模型需要更多人類標註                               │
│                                                               │
│  RLHF + RLAIF:                                               │
│    ┌────────────────────────────┐                           │
│    │  人類反饋 → 訓練 RM       │                           │
│    │         ↓                 │                           │
│    │  RM 可以評估百萬樣本      │                           │
│    │         ↓                 │                           │
│    │  PPO 學習人類偏好        │                           │
│    └────────────────────────────┘                           │
│                                                               │
│    結果: 對齊成本 amortized over 大量推理                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

#### 價值 3：處理「隱式目標」

**無法直接監督的任務**：
| 任務 | 傳統監督問題 | RLHF 解決 |
|------|-------------|-----------|
| 幫助性 | 難以定義「幫助」 | 人類偏好判斷 |
| 安全性 | 規則難以窮舉 | 偏好學習 |
| 有趣性 | 主觀性強 | 相對比較 |
| 毒性 | 複雜語境判斷 | 多維度偏好 |

---

### 挑戰與解決方案

#### 挑戰 1：Reward Model 偏差

**問題描述**：
```
RM 學習到的是「人類偏好的代理」而非真正的偏好
可能出現:
- 偏好長回覆 (因為人類傾向選擇較長的)
- 偏好特定格式 (因為訓練數據有偏)
- 對抗樣本脆弱 (被簡單技巧欺騙)
```

**解決方案**：
```python
class DebiasedRewardModel:
    """去偏 Reward Model"""

    def __init__(self, base_model):
        self.model = base_model
        self.length_bias = None
        self.format_bias = None

    def compute_reward(self, prompt, response):
        base_reward = self.model(prompt, response)

        # 去除長度偏差
        length_correction = self._remove_length_bias(response, base_reward)

        # 去除格式偏差
        format_correction = self._remove_format_bias(response, length_correction)

        return format_correction

    def _remove_length_bias(self, response, reward):
        """回歸分析估計長度貢獻，去除它"""
        predicted_length_effect = self.length_bias.predict(len(response))
        return reward - predicted_length_effect
```

#### 挑戰 2：PPO 訓練不穩定

**問題描述**：
```
- Policy 更新幅度過大
- Reward 波動劇烈
- KL 散度失控
```

**解決方案**：
```python
class StablePPOTrainer:
    """穩定 PPO 訓練器"""

    def __init__(self, model, config):
        # 1. 保守的 clip 範圍
        self.clip_eps = 0.1  # 較小的 clip 範圍

        # 2. Adaptive KL Target
        self.kl_controller = AdaptiveKLController(
            target_kl=0.01,
            horizon=100
        )

        # 3. 值函數預期歸一化
        self.value_normalizer = RunningMeanStd()

        # 4. 梯度裁剪
        self.max_grad_norm = 0.5

    def compute_advantages(self, rewards, values, dones):
        """使用 GAE 計算優勢，更穩定的估計"""

        advantages = []
        gae = 0
        gamma = 0.99
        lam = 0.95

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages)
```

#### 挑戰 3：獎勵被 Gaming

**問題描述**：
```
模型發現一些「作弊」模式:
- 生成很長但空洞的回覆
- 過度使用某些詞彙
- 操縱 RM 的弱點
```

**解決方案**：
```python
class MultiSignalRewardShaping:
    """多信號獎勵塑形"""

    def compute_reward(self, prompt, response, model_output):
        rewards = {}

        # 1. 基礎獎勵 (RM)
        rewards["rm"] = self.reward_model(prompt, response)

        # 2. 長度約束
        length = len(response)
        rewards["length"] = -0.001 * abs(length - self.target_length)

        # 3. 重複懲罰
        unique_ratio = len(set(response)) / len(response)
        rewards["repeat"] = -0.5 * (1 - unique_ratio)

        # 4. 毒性檢測
        rewards["safety"] = -1.0 if self.toxicity_detector(response) else 0

        # 5. 格式獎勵
        rewards["format"] = 0.1 if self.has_proper_format(response) else 0

        # 6. 信息量估計
        rewards["informative"] = self.estimate_informativeness(response)

        # 加權組合
        total_reward = sum(
            w * rewards[k]
            for k, w in self.weights.items()
        )

        return total_reward, rewards  # 返回分解以便分析
```

---

## 補充：PPO 數學推導進階

### 策略梯度深入理解

```
┌─────────────────────────────────────────────────────────────────────┐
│                      策略梯度推導                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  目標: 最大化期望累積獎勵                                            │
│                                                                     │
│  J(π) = E_τ~π [R(τ)]                                               │
│       = E_{s,a~π}[Σ γ^t r(s_t, a_t)]                                │
│                                                                     │
│  對數技巧:                                                          │
│    ∇_θ J = ∇_θ E_τ~πθ [R(τ)]                                        │
│          = E_τ~πθ [∇_θ log π_θ(τ) × R(τ)]                          │
│                                                                     │
│  展開:                                                              │
│    log π_θ(τ) = log π_θ(a_0|s_0) + log π_θ(s_1|s_0,a_0) + ...      │
│                                                                     │
│    ∇_θ log π_θ(τ) = Σ ∇_θ log π_θ(a_t|s_t)                         │
│                                                                     │
│  直覺解釋:                                                          │
│    - 增加「高獎勵」軌跡的機率                                        │
│    - 減少「低獎勵」軌跡的機率                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### PPO 目標函數深度解析

```python
def ppo_objective(
    π_theta,      # 新策略
    π_theta_old,  # 舊策略
    advantages,   # 優勢函數 A(s,a)
    clip_eps=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01
):
    """
    PPO 目標函數

    L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

    其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    """

    # 1. 比率 r(θ)
    log_probs = π_theta.get_log_prob(actions)
    old_log_probs = π_theta_old.get_log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)

    # 2. Clip 目標
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # 3. 取較小值 (保守更新)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # 4. 值函數損失
    values = π_theta.get_values(states)
    value_loss = F.mse_loss(values, returns)

    # 5. 熵獎勵 (鼓勵探索)
    entropy = π_theta.entropy()

    # 6. 總損失
    total_loss = (
        policy_loss.mean()
        + value_coeff * value_loss
        - entropy_coeff * entropy
    )

    return total_loss
```

### KL Penalty 的物理意義

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KL Penalty 的物理意義                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  KL(π_new || π_ref) = Σ π_new(a|s) log(π_new(a|s) / π_ref(a|s))   │
│                                                                      │
│  直覺:                                                               │
│    - 測量兩個分佈的「距離」                                          │
│    - KL = 0: 完全相同                                               │
│    - KL → ∞: 完全不同                                               │
│                                                                      │
│  在 RLHF 中的作用:                                                   │
│                                                                      │
│    Loss = Reward - β × KL(π || π_SFT)                              │
│                                                                      │
│    這意味著:                                                         │
│    ┌────────────────────────────────────────────────────────────┐  │
│    │  要獲得更高的獎勵，必須:                                    │  │
│    │    1. 提高 Reward (提高質量)                               │  │
│    │    2. 同時控制 KL (不要偏離 SFT 太遠)                       │  │
│    └────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  β 的選擇:                                                           │
│    - β 太大: 訓練停滞，模型無法學習新行為                             │
│    - β 太小: 模型偏離 SFT，可能喪失能力                              │
│    - 典型值: 0.01 - 0.1                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

## 14. RLHF 進階主題

### 14.1 離線策略評估 (Offline Policy Evaluation)

**問題**：如何評估未實際執行的策略？

```python
class OfflinePolicyEvaluator:
    """離線策略評估"""

    def evaluate(self, new_policy, historical_data):
        """
        使用重要性採樣估計新策略的性能
        """

        # 1. 計算重要性權重
        importance_weights = []

        for trajectory in historical_data:
            weight = 1.0
            for step in trajectory:
                # w = π_new(a|s) / π_old(a|s)
                w = new_policy.prob(step.action, step.state) / step.old_prob
                weight *= w

            importance_weights.append(weight)

        # 2. 加權估計
        weighted_rewards = [
            w * trajectory.total_reward
            for w, trajectory in zip(importance_weights, historical_data)
        ]

        # 3. 歸一化
        sum_weights = sum(importance_weights)
        estimated_value = sum(weighted_rewards) / sum_weights

        # 4. 置信區間
        variance = self._compute_variance(importance_weights, weighted_rewards)

        return {
            "value": estimated_value,
            "variance": variance,
            "ci_95": (estimated_value - 1.96 * variance**0.5,
                      estimated_value + 1.96 * variance**0.5)
        }
```

### 14.2 獎勵模型的魯棒性

**挑戰**：RM 容易被對抗樣本欺騙

```python
class RobustRewardModel:
    """魯棒 Reward Model - 抵抗對抗攻擊"""

    def __init__(self, base_model):
        self.model = base_model
        self.adversarial_trainer = AdversarialTrainer()

    def compute_reward_robust(self, prompt, response):
        # 1. 基礎獎勵
        base_reward = self.model(prompt, response)

        # 2. 對抗檢測
        if self._is_adversarial(response):
            # 如果檢測到對抗模式，降低獎勵
            base_reward *= 0.5

        # 3. 多視角評估
        multi_view_reward = self._multi_view_evaluate(prompt, response)

        # 4. 融合
        return 0.7 * base_reward + 0.3 * multi_view_reward

    def _is_adversarial(self, response):
        """檢測對抗模式"""
        # 常見對抗模式
        adversarial_patterns = [
            "非常抱歉",
            "我無法",  # 過度道歉/拒絕
            "作為 AI", # 過度聲明
        ]

        count = sum(1 for p in adversarial_patterns if p in response)
        return count > 2
```

### 14.3 PPO 的實際超參數調優指南

```python
"""
PPO 超參數調優矩陣

| 參數           | 任務類型     | 建議起始值 | 調整策略           |
|----------------|-------------|-----------|------------------|
| clip_eps       | 一般任務    | 0.2       | 不穩定時降低       |
| clip_eps       | 複雜推理    | 0.1       | 需要保守更新       |
| gamma          | 短回覆     | 0.9       | -                 |
| gamma          | 長回覆     | 0.99      | -                 |
| lam            | -          | 0.95      | 偏向 TD 時降低      |
| lr             | 7B 模型    | 1e-5      | 觀察 KL 調整       |
| lr             | 70B 模型   | 5e-6      | 需要更保守         |
| batch_size     | GPU 記憶體  | 動態      | 增大直到 OOM       |
| ppo_epochs     | -          | 4         | 數據少時增大       |
| kl_target      | -          | 0.01      | 根據實際 KL 調整   |
"""
```

### 14.4 RLHF 與其他對齊方法的比較

```
┌─────────────────────────────────────────────────────────────────────┐
│                    對齊方法全面比較                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  方法        數據需求    計算成本    對齊質量    可擴展性            │
│  ──────      ──────     ──────     ──────     ──────             │
│  SFT         高         低         中         低                    │
│  RLHF        高         高         高         中                   │
│  DPO         中         中         高         高                   │
│  KTO         中         中         高         高                   │
│  RLAIF       低         中         中         高                   │
│  Con AI      低         中         高         高                   │
│                                                                      │
│  適用場景:                                                           │
│  - SFT: 基礎能力建立                                                │
│  - RLHF: 最高質量對齊，資源充足                                      │
│  - DPO: 資源有限但需高質量                                          │
│  - KTO: 偏好數據有偏差                                              │
│  - RLAIF: 缺乏人類偏好數據                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 延伸閱讀

- [InstructGPT Paper (OpenAI)](https://arxiv.org/abs/2203.02155)
- [RLHF from Human Feedback](https://arxiv.org/abs/2206.07682)
- [HuggingFace TRL](https://huggingface.co/docs/trl/index)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DeepMind RLHF](https://arxiv.org/abs/2204.05862)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)