# DPO

簡化的對齊訓練方法，直接從人類偏好數據中學習，無需訓練獎勵模型和強化學習，透過簡單的分類損失達到 RLHF 的效果。

---

## 1. 什麼是？

### 簡單範例

```
訓練數據:
  Prompt: "如何修復這個 bug?"
  Chosen: "首先檢查錯誤日誌，然後..."
  Rejected: "我不知道，隨便試試"

DPO 訓練:
  輸入: (Prompt, Chosen, Rejected)
  輸出: 更新後的 LLM

結果:
  模型學會生成類似 "Chosen" 的回覆
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **更簡單** | 移除複雜的 RL 流程 |
| **更穩定** | 沒有強化學習的不穩定性 |
| **更高效** | 訓練速度更快，計算資源更少 |
| **效果相當** | 實驗顯示 DPO 效果與 RLHF 相近 |

---

## 3. 核心原理

### RLHF vs DPO 流程比較

```
RLHF:
  模型 → 輸出 → RM 評分 → PPO 優化 → 模型
              ↓
         需要複雜的 RL 流程

DPO:
  模型 → 直接優化偏好損失 → 模型
              ↓
         只需要分類損失
```

### DPO Loss 函數

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO Loss

    目標: 讓 policy 傾向 chosen，遠離 rejected
    """

    # 計算 log ratio
    chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratio = policy_rejected_logps - reference_rejected_logps

    # DPO Loss
    losses = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio))

    return losses.mean()
```

### 數學推導

```
DPO 證明了以下等價關係:

PPO 優化目標 ⟺ 直接優化偏好損失

數學上:
max_θ E[(x,y_w,y_l) ~ D] [log(σ(r_θ(x, y_w) - r_θ(x, y_l)))]

其中:
- θ = 模型參數
- x = 輸入 prompt
- y_w = 偏好的回覆 (winner)
- y_l = 不偏好的回覆 (loser)
- σ = sigmoid 函數
```

### 為什麼有效？

```
直觀理解:

如果 model(chosen) > model(rejected):
  - σ(r_chosen - r_rejected) → 1
  - Loss → 0 (正確，不需要更新)

如果 model(chosen) < model(rejected):
  - σ(r_chosen - r_rejected) → 0
  - Loss → 大 (需要更新，讓 chosen 機率提高)
```

### DPO 的优势

| 特性 | RLHF | DPO |
|------|------|-----|
| 訓練階段 | 3 階段 | 1 階段 |
| 需要 RM | 是 | 否 |
| 超參數 | 多 (PPO) | 少 |
| 訓練穩定性 | 較不穩 | 穩定 |
| 計算資源 | 高 | 中等 |
| 實現複雜度 | 高 | 低 |

---

## 4. 實現細節

### 數據格式

```python
# 標準 DPO 數據格式
{
    "prompt": "如何修復這個 bug?",
    "chosen": "首先檢查錯誤日誌，然後嘗試以下方法...",
    "rejected": "我不知道，隨便試試"
}
```

### 訓練技巧

```python
# 1. KL Penalty: 防止過度偏離 reference 模型
def compute_loss(policy_logps, ref_logps, beta=0.1):
    return -F.logsigmoid(beta * (policy_logps - ref_logps))

# 2. β 參數影響
# β 越小: 對偏好越敏感，訓練越快
# β 越大: 訓練越保守，類似 RLHF 的 KL penalty

# 3. Data Mixing: 混合 SFT 數據保持能力
# 通常 10-20% SFT 數據 + 80-90% DPO 數據
```

### β 參數選擇

| β 值 | 效果 | 適用場景 |
|------|------|----------|
| 0.01 | 非常保守 | 初始階段 |
| 0.1 | 標準 | 一般場景 |
| 0.3 | 激進 | 快速對齊 |

---

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **RLHF** | DPO 要簡化的方法 |
| **PPO** | RLHF 使用的強化學習算法 |
| **ORPO** | 另一種偏好優化方法 |
| **KTO** | Kahneman-Tversky Optimization |

---

## 6. 數學推導

### 從 PPO 到 DPO

```
DPO 的核心洞見: 直接優化偏好資料，等價於 RLHF

回顧 RLHF (PPO) 目標:
  max_θ E[log π_θ(a|q) - β × KL(π_θ || π_ref)]

DPO 證明這可以轉化為:
  max_θ E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]

其中 r_θ(x, y) 是 LLM 生成的 reward signal
```

### DPO Loss 推導

```python
def dpo_loss(
    policy_chosen_logps,      # policy 在 chosen 上的 log probs
    policy_rejected_logps,    # policy 在 rejected 上的 log probs
    reference_chosen_logps,    # reference 在 chosen 上的 log probs
    reference_rejected_logps, # reference 在 rejected 上的 log probs
    beta=0.1
):
    """
    DPO Loss 推導
    """

    # 計算 log ratio (相對於 reference)
    chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratio = policy_rejected_logps - reference_rejected_logps

    # 計算差異
    log_diff = chosen_log_ratio - rejected_log_ratio

    # DPO Loss
    losses = -F.logsigmoid(beta * log_diff)

    # 返回 mean loss
    return losses.mean()


def reward_function(log_probs, attention_mask):
    """
    從 log probs 計算 reward

    Reward = sum of token log probs (加權)
    """
    # 對每個位置加權
    weights = attention_mask.float()
    rewards = (log_probs * weights).sum(-1) / weights.sum(-1)

    return rewards
```

### 為什麼 DPO 有效

```
直覺解釋:

我們希望:
  P(chosen) > P(rejected)

也就是:
  log P(chosen) > log P(rejected)
  log P(chosen) - log P(rejected) > 0

定義:
  Δ = log π_θ(chosen) - log π_θ(rejected)

使用 sigmoid:
  σ(Δ) = P(chosen) / (P(chosen) + P(rejected))

最大化 σ(Δ) ⟺ 最大化 Δ

DPO Loss: -log σ(β × (r_chosen - r_rejected))

這鼓勵模型增加 chosen 的 reward，降低 rejected 的 reward
```

---

## 7. 數據構建

### 偏好數據格式

```python
# 標準 DPO 數據集格式
{
    "prompt": "用戶問題或指令",
    "chosen": "人類偏好的回覆",
    "rejected": "人類不偏好的回覆",
    # 可選欄位
    "chosen_reward": 5.0,    # 獎勵模型分數
    "rejected_reward": 2.0,
    "metadata": {
        "source": "anthropic_hh",
        "model": "gpt-4"
    }
}
```

### 數據品質過濾

```python
class DPODataFilter:
    """DPO 數據過濾"""

    @staticmethod
    def filter_by_reward_margin(dataset, min_margin=1.0):
        """只保留有足夠差距的偏好對"""

        filtered = []
        for item in dataset:
            margin = item["chosen_reward"] - item["rejected_reward"]
            if margin >= min_margin:
                filtered.append(item)

        return filtered

    @staticmethod
    def filter_by_length_ratio(dataset, max_ratio=2.0):
        """過濾長度差異過大的偏好對"""

        filtered = []
        for item in dataset:
            chosen_len = len(item["chosen"])
            rejected_len = len(item["rejected"])

            ratio = max(chosen_len, rejected_len) / max(min(chosen_len, rejected_len), 1)

            if ratio <= max_ratio:
                filtered.append(item)

        return filtered

    @staticmethod
    def remove_duplicates(dataset):
        """移除重複樣本"""
        seen = set()
        unique = []

        for item in dataset:
            key = (item["prompt"], item["chosen"], item["rejected"])
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique
```

### 數據增強

```python
class DPODataAugmentation:
    """DPO 數據增強"""

    @staticmethod
    def add_system_prompt(dataset, system_prompt):
        """添加系統提示"""

        for item in dataset:
            item["prompt"] = f"{system_prompt}\n\n{item['prompt']}"

        return dataset

    @staticmethod
    def generate_rejected_responses(llm, prompts, num_samples=3):
        """使用 LLM 生成 rejected _response"""

        rejected_responses = []

        for prompt in prompts:
            # 多次採樣得到多個回覆
            samples = llm.generate(prompt, num_samples=num_samples, temperature=0.8)

            # 選擇一個較差的作為 rejected
            # 可以根據長度、風格或額外的評估模型
            rejected = select_worst_response(samples)
            rejected_responses.append(rejected)

        return rejected_responses
```

---

## 8. 訓練細節

### 超參數選擇

| 參數 | 典型值 | 影響 |
|------|--------|------|
| **β (beta)** | 0.1-0.5 | 越大越保守 |
| **learning_rate** | 1e-6 - 1e-5 | 需要精細調整 |
| **batch_size** | 8-32 | 取決於 GPU 記憶體 |
| **gradient_accumulation** | 1-8 | 增大 effective batch |
| **epochs** | 1-3 | DPO 容易過擬合 |
| **max_seq_length** | 512-2048 | 取決於 GPU 記憶體 |

### 訓練監控

```python
class DPO TrainingMonitor:
    def __init__(self):
        self.metrics = {
            "chosen_logps": [],
            "rejected_logps": [],
            "log_ratio": [],
            "loss": []
        }

    def record_batch(self, batch):
        self.metrics["chosen_logps"].append(batch["chosen_logps"].mean().item())
        self.metrics["rejected_logps"].append(batch["rejected_logps"].mean().item())
        self.metrics["log_ratio"].append(batch["log_ratio"].mean().item())
        self.metrics["loss"].append(batch["loss"].item())

    def get_summary(self):
        return {
            "avg_chosen_logps": np.mean(self.metrics["chosen_logps"][-100:]),
            "avg_rejected_logps": np.mean(self.metrics["rejected_logps"][-100:]),
            "avg_log_ratio": np.mean(self.metrics["log_ratio"][-100:]),
            "convergence": self._check_convergence()
        }

    def _check_convergence(self):
        """檢查是否收斂"""
        recent_losses = self.metrics["loss"][-50:]
        if len(recent_losses) < 50:
            return "training"

        # 檢查損失是否穩定
        std = np.std(recent_losses)
        if std < 0.01:
            return "converged"

        # 檢查趨勢
        first_half = np.mean(recent_losses[:25])
        second_half = np.mean(recent_losses[25:])

        if second_half < first_half * 0.9:
            return "improving"

        return "stable"
```

### 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **Loss 不下降** | β 過大或過小 | 調整 β 或 learning rate |
| **模型只生成短回覆** | 偏好數據偏向短回答 | 過濾或增強長回覆 |
| **過擬合** | epochs 太多 | 減少 epochs，增加數據 |
| **不穩定** | 梯度爆炸 | 降低 learning rate |

---

## 9. Reference Model 管理

### Reference Model 更新

```python
class DPOWithMovingReference:
    """使用移動平均更新 reference model"""

    def __init__(self, model, tau=0.995):
        self.policy = model
        self.reference = copy.deepcopy(model)
        self.tau = tau  # 移動平均係數

    def update_reference(self):
        """更新 reference model"""

        for policy_param, ref_param in zip(
            self.policy.parameters(),
            self.reference.parameters()
        ):
            # Polyak averaging
            ref_param.data.copy_(
                self.tau * ref_param + (1 - self.tau) * policy_param
            )
```

### Reference-free DPO

```python
class ReferenceFreeDPO:
    """
    無需 reference model 的 DPO 變體
    適合無法加載 reference 的場景
    """

    def __init__(self, model):
        self.model = model

    def reference_free_loss(self, chosen_logps, rejected_logps, beta=0.1):
        """
        直接使用 policy 的 log probs
        """

        # 計算差異
        log_ratio = chosen_logps - rejected_logps

        # Loss
        losses = -F.logsigmoid(beta * log_ratio)

        return losses.mean()
```

---

## 10. 與 RLHF 比較

### 訓練效率

```
訓練時間比較 (同一硬件):

RLHF:
  - SFT: 1 小時
  - Reward Model: 2 小時
  - PPO: 8-12 小時
  - 總計: ~12-15 小時

DPO:
  - SFT: 1 小時
  - DPO: 2-3 小時
  - 總計: ~3-4 小時

→ 3-4x 加速
```

### 記憶體比較

```
GPU 記憶體需求:

RLHF:
  - Policy Model: 1x
  - Reference Model: 1x
  - Reward Model: 1x
  - Value Network: 1x
  - 總計: ~4x

DPO:
  - Policy Model: 1x
  - Reference Model: 1x (可選)
  - 總計: ~1-2x
```

### 質量比較

| 指標 | RLHF | DPO |
|------|------|-----|
| **對齊品質** | 優秀 | 相近 |
| **訓練時間** | 長 | 短 |
| **實現複雜度** | 高 | 低 |
| **超參數敏感度** | 高 | 低 |
| **大規模訓練** | 成熟 | 發展中 |

---

## 11. 實作工具

### TRL DPO Trainer

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    beta=0.1,                    # DPO beta
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=512,
    max_prompt_length=256,
    num_train_epochs=3,
)

trainer.train()
```

### 自定義 DPO Trainer

```python
import torch
import torch.nn.functional as F
from transformers import Trainer

class CustomDPOTrainer(Trainer):
    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        """自定義 DPO Loss"""

        # 取得 prompt 和回覆
        prompt_ids = inputs["input_ids"]
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]

        # 計算 policy 的 log probs
        chosen_logps = self._get_log_probs(model, chosen_ids)
        rejected_logps = self._get_log_probs(model, rejected_ids)

        # 計算 reference 的 log probs (如果可用)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen_logps = self._get_log_probs(self.ref_model, chosen_ids)
                ref_rejected_logps = self._get_log_probs(self.ref_model, rejected_ids)
        else:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)

        # DPO Loss
        chosen_log_ratio = chosen_logps - ref_chosen_logps
        rejected_log_ratio = rejected_logps - ref_rejected_logps

        loss = -F.logsigmoid(
            self.beta * (chosen_log_ratio - rejected_log_ratio)
        ).mean()

        return loss
```

---

## 12. 相關主題

| 技術 | 關係 |
|------|------|
| **RLHF** | DPO 要簡化的方法 |
| **PPO** | RLHF 使用的強化學習算法 |
| **ORPO** | 另一種偏好優化方法 |
| **KTO** | Kahneman-Tversky Optimization |
| **SimPO** | 簡化的 DPO 變體 |

---

## 延伸閱讀

- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [HuggingFace DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [DPO vs RLHF 比較](https://arxiv.org/abs/2402.13228)
- [TRL DPO 實作](https://github.com/huggingface/trl)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [KTO Paper](https://arxiv.org/abs/2402.01306)