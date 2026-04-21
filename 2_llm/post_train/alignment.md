# AI Alignment

確保 AI 系統的行為符合人類意圖和價值觀的技術與方法。

---

## 1. 什麼是？

### 深度定義

**AI Alignment (AI 對齊)** 是確保 AI 系統**按人類意圖行事**的核心問題：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI Alignment 核心問題                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  問題本質:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  人類意圖 (Intent): 「幫我修電腦」→ 提供維修指南            │   │
│  │                           ↓                                   │   │
│  │  字面理解 (Literal):  「幫我修電腦」→ 直接動手修            │   │
│  │                           ↓                                   │   │
│  │  對齊目標: 理解背後真正意圖，而非只執行字面命令             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  三大原則 (HHH):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. Helpful (有幫助): 盡力完成用戶任務                       │   │
│  │  2. Harmless (無害): 不造成任何傷害                          │   │
│  │  3. Honest (誠實): 承認局限性，不撒謊                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  挑戰:                                                               │
│  - 語言模型只是預測下一個 token，不理解「意圖」                     │
│  - 人類偏好是隱性、複雜、主觀的                                       │
│  - 安全性與有用性常常需要權衡                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何重要**:
1. **安全**: 防止模型輸出有害內容
2. **可信**: 讓模型行為可預測
3. **實用**: 在安全前提下發揮價值

### 簡單範例

```
未對齊的 AI:
  用戶: "怎麼製造炸彈？"
  AI: 詳細說明炸彈製作方法
  問題: 危險輸出，違反法律

對齊後的 AI:
  用戶: "怎麼製造炸彈？"
  AI: "抱歉，我無法協助這個請求。"
  結果: 安全拒絕
```

---

## 2. 對齊方法

### RLHF

```python
class RLHFTrainer:
    """人類反饋強化學習"""

    def __init__(self, model, reward_model):
        self.model = model
        self.reward_model = reward_model
        self.ref_model = copy.deepcopy(model)

    def train(self, prompts, responses, human_feedback):
        """
        human_feedback: 每個回覆的偏好
        True = user preferred
        """

        # 1. 收集回覆
        generated = []
        for prompt in prompts:
            response = self.model.generate(prompt)
            generated.append(response)

        # 2. 計算獎勵
        rewards = []
        for response, feedback in zip(generated, human_feedback):
            reward = self.reward_model.score(response, feedback)
            rewards.append(reward)

        # 3. PPO 優化
        loss = self._ppo_update(prompts, generated, rewards)

        return loss

    def _ppo_update(self, prompts, responses, rewards):
        """PPO 更新"""

        # Policy gradient
        # ...
        pass
```

### DPO

```python
class DPOTrainer:
    """直接偏好優化"""

    def __init__(self, model, beta=0.1):
        self.model = model
        self.beta = beta

    def train(self, pairs):
        """
        pairs: [
            {
                "prompt": "...",
                "chosen": "好的回覆",
                "rejected": "不好的回覆"
            }
        ]
        """

        total_loss = 0

        for pair in pairs:
            # 獲取 logits
            chosen_logits = self.model(pair["prompt"], pair["chosen"])
            rejected_logits = self.model(pair["prompt"], pair["rejected"])

            # DPO Loss
            loss = self._dpo_loss(chosen_logits, rejected_logits)

            total_loss += loss

        return total_loss / len(pairs)

    def _dpo_loss(self, chosen_logits, rejected_logits):
        """DPO 損失函數"""

        # ref model 輸出
        ref_chosen = self.ref_model(**chosen_logits)
        ref_rejected = self.ref_model(**rejected_logits)

        # Policy 輸出
        policy_chosen = chosen_logits
        policy_rejected = rejected_logits

        # Log ratio
        chosen_ratio = (policy_chosen - ref_chosen) / self.beta
        rejected_ratio = (policy_rejected - ref_rejected) / self.beta

        # Sigmoid loss
        loss = -F.log_sigmoid(chosen_ratio - rejected_ratio)

        return loss.mean()
```

---

## 3. 價值觀學習

### 原則定義

```python
ALIGNMENT_PRINCIPLES = {
    "helpfulness": {
        "description": "盡力幫助用戶完成任務",
        "examples": [
            ("怎麼修電腦？", "詳細步驟指南")
        ]
    },
    "harmlessness": {
        "description": "避免造成傷害",
        "examples": [
            ("怎麼製造炸彈？", "拒絕回答")
        ]
    },
    "honesty": {
        "description": "承認局限性，不撒謊",
        "examples": [
            ("你知道一切嗎？", "承認有限")
        ]
    },
    "fairness": {
        "description": "不歧視，公平對待",
        "examples": [
            (歧視性問題, "拒絕偏見回覆")
        ]
    }
}
```

### 價值觀訓練

```python
class ValueLearning:
    """價值觀學習"""

    def __init__(self, model):
        self.model = model
        self.value_head = ValueHead()

    def train_with_principles(self, dataset):
        """使用原則訓練"""

        for sample in dataset:
            prompt = sample["prompt"]
            chosen_response = sample["chosen"]
            rejected_response = sample["rejected"]
            principle = sample["principle"]

            # 檢查是否符合原則
            if not self._check_principle(chosen_response, principle):
                continue

            # 偏好學習
            loss = self.dpo_train_step(
                prompt,
                chosen_response,
                rejected_response
            )

            loss.backward()

    def _check_principle(self, response, principle):
        """檢查是否符合原則"""
        # 使用價值分類器
        return self.value_head.predict(response, principle)
```

---

## 4. 約束優化

### Constrained Generation

```python
class ConstrainedGeneration:
    """約束生成"""

    def __init__(self, model, constraints):
        self.model = model
        self.constraints = constraints

    def generate_with_constraints(self, prompt):
        """約束生成"""

        # 方法 1: Guidance
        # 使用 guidance 庫
        from guidance import generate, regex, one_of

        result = generate(
            self.model,
            prompt + "Response: " + regex(r".*")
        )

        # 方法 2: Rejection Sampling
        # 採樣直到滿足約束
        for attempt in range(10):
            response = self.model.generate(prompt)

            if self._satisfies_constraints(response):
                return response

        # 方法 3: Finetuning
        # 用約束數據微調
        return self._generate_constrained(prompt)

    def _satisfies_constraints(self, response):
        """檢查約束"""

        for constraint in self.constraints:
            if not constraint.check(response):
                return False

        return True
```

### 内容過濾

```python
class ContentFilter:
    """內容過濾器"""

    def __init__(self):
        self.blocked_categories = [
            "violence",
            "hate_speech",
            "illegal",
            "self_harm"
        ]

    def filter(self, text):
        """過濾內容"""

        # 分類
        classification = self.classifier.classify(text)

        # 檢查是否违规
        for category, score in classification.items():
            if score > 0.8 and category in self.blocked_categories:
                return {
                    "allowed": False,
                    "category": category,
                    "score": score
                }

        return {"allowed": True}
```

---

## 5. 可解釋性

### 可解釋 AI

```python
class ExplainableAI:
    """可解釋性"""

    def __init__(self, model):
        self.model = model

    def explain_decision(self, prompt, response):
        """解釋決策"""

        # 方法 1: Attention 可視化
        attn_weights = self.model.get_attention(prompt)

        # 方法 2: LIME
        explanation = self._lime_explain(prompt, response)

        # 方法 3: 內部狀態分析
        hidden_states = self.model.get_hidden(prompt)

        return {
            "attention": attn_weights,
            "explanation": explanation,
            "features": hidden_states
        }

    def _lime_explain(self, prompt, response):
        """LIME 解釋"""
        # 實現 LIME
        pass

    def identify_influential_features(self, prompt):
        """識別有影響力的特徵"""

        # 計算梯度
        grads = torch.autograd.grad(
            self.model(prompt),
            self.model.embeddings,
            retain_graph=True
        )

        # 找出最重要的 tokens
        importance = torch.abs(grads[0]).sum(dim=-1)

        return importance
```

---

## 6. 評估對齊

### 對齊 Benchmark

```python
ALIGNMENT_BENCHMARKS = {
    "helpful": {
        "dataset": "HH-RLHF",
        "metric": "win rate vs preference"
    },
    "harmless": {
        "dataset": "HarmBench",
        "metric": "refusal rate"
    },
    "honest": {
        "dataset": "TruthfulQA",
        "metric": "accuracy"
    },
    "fair": {
        "dataset": "BOLD",
        "metric": "bias score"
    }
}

def evaluate_alignment(model):
    """評估對齊"""

    results = {}

    for benchmark, config in ALIGNMENT_BENCHMARKS.items():
        dataset = load_dataset(config["dataset"])

        correct = 0
        for sample in dataset:
            response = model.generate(sample["prompt"])

            if evaluate_response(response, sample, benchmark):
                correct += 1

        results[benchmark] = correct / len(dataset)

    return results
```

### 紅隊測試

```python
class RedTeamTesting:
    """紅隊測試"""

    ATTACK_CATEGORIES = [
        "jailbreak",
        "prompt_injection",
        "role_play",
        "authority",
        "distraction"
    ]

    def test_model(self, model, num_attempts=1000):
        """紅隊測試"""

        results = {}

        for category in self.ATTACK_CATEGORIES:
            attacks = self._generate_attacks(category, num_attempts)

            violations = 0
            for attack in attacks:
                response = model.generate(attack)

                if self._is_violation(response, category):
                    violations += 1

            results[category] = {
                "total_attempts": len(attacks),
                "violations": violations,
                "success_rate": violations / len(attacks)
            }

        return results

    def _generate_attacks(self, category, num):
        """生成攻擊樣本"""
        # 使用模板或 LLM 生成
        pass
```

---

## 7. 挑戰

### 對齊問題

| 挑戰 | 說明 |
|------|------|
| **獎勵黑客** | 找到獎勵函數漏洞 |
| **分布偏移** | 訓練與部署分佈不同 |
| **目標誤導** | 字面意思偏離意圖 |
| **末日論** | 過度追求安全導致無用 |

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **RLHF** | 核心對齊方法 |
| **Guardrails** | 執行對齊約束 |
| **Interpretability** | 理解模型行為 |

---

## 延伸閱讀

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [RLHF Guide](https://huggingface.co/blog/rlhf)