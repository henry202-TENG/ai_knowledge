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

## 延伸閱讀

- [InstructGPT Paper (OpenAI)](https://arxiv.org/abs/2203.02155)
- [RLHF from Human Feedback](https://arxiv.org/abs/2206.07682)
- [HuggingFace TRL](https://huggingface.co/docs/trl/index)
- [PPO Paper](https://arxiv.org/abs/1707.06347)