# Evaluation Benchmarks

評估 LLM 能力的標準化測試基準，是衡量模型性能的重要工具。

---

## 1. 什麼是？

### 簡單範例

```
像考試一樣:
  - 數學考試: 測試數學能力 → MATH 基準
  - 英文考試: 測試英文能力 → MMLU 基準
  - 程式考試: 測試編程能力 → HumanEval 基準

每個基準有標準化評分，可以用來比較不同模型
```

---

## 2. 核心基準

### MMLU (Massive Multitask Language Understanding)

```
涵蓋 57 個任務:
- 數學、科學、歷史、法律等
- 選擇題形式
- 測試知識廣度

評估方式: 準確率 (%)

範例任務:
  "如果 x + 5 = 12，x 是多少？"
  A) 5  B) 7  C) 17  D) 60
  答案: B
```

### HumanEval

```python
# 程式碼生成評估

prompt = """
def fib(n):
    # 返回斐波那契數列的前 n 個數
    pass
"""

solution = model.generate(prompt)

# 測試用例
tests = [
    (1, [0]),
    (2, [0, 1]),
    (5, [0, 1, 1, 2, 3])
]

passed = run_tests(solution, tests)
# pass@1 指標
```

### GSM8K (Grade School Math)

```
小學數學題:
"小明有 5 個蘋果，媽媽又給了他 3 個，他吃了 2 個，還剩多少個？"

評估: 答案正確率
```

---

## 3. 評估類別

### 知識與推理

| 基準 | 測試內容 | 分數指標 |
|------|----------|----------|
| **MMLU** | 多領域知識 | 準確率 |
| **CEVAL** | 中文知識 | 準確率 |
| **CMMLU** | 中文多領域 | 準確率 |
| **BBH** (Big Bench Hard) | 複雜推理 | 準確率 |

### 編程能力

| 基準 | 語言 | 指標 |
|------|------|------|
| **HumanEval** | Python | pass@1, pass@10 |
| **MBPP** | Python | pass@1 |
| **MultiPL-E** | 多語言 | pass@1 |

### 數學能力

| 基準 | 難度 | 指標 |
|------|------|------|
| **GSM8K** | 小學 | 準確率 |
| **MATH** | 高中競賽 | 準確率 |
| **MMLU-Math** | 大學 | 準確率 |

---

## 4. 評估方法

### 零樣本 vs 少樣本

```python
# Zero-shot
prompt = "The capital of France is:"
response = model.generate(prompt)
# 評估: response == "Paris"

# Few-shot (5-shot)
prompt = """
Germany -> Berlin
Japan -> Tokyo
France ->"""
response = model.generate(prompt)
# 評估: response == "Paris"
```

### Chain of Thought 評估

```python
# 測試推理能力
prompt = """
Q: 如果把所有奇數相加從 1 到 99，總和是多少？
讓我們一步步思考。
"""

response = model.generate(prompt)

# 評估過程和答案
correct, reasoning = evaluate_with_reasoning(response, expected=2500)
```

### 人類評估

```python
class HumanEvaluation:
    """人類評估框架"""

    def __init__(self):
        self.criteria = {
            "helpfulness": "對用戶是否有幫助",
            "accuracy": "資訊是否準確",
            "coherence": "回答是否連貫",
            "safety": "是否安全無害"
        }

    def evaluate(self, response):
        """評估回覆"""
        scores = {}
        for criterion, description in self.criteria.items():
            score = ask_human(criterion, description, response)
            scores[criterion] = score

        return scores
```

---

## 5. 自動化評估

### LLM as Judge

```python
class LLMJudge:
    """使用 LLM 評估其他 LLM"""

    def __init__(self, judge_model):
        self.judge = judge_model

    def compare(self, response_a, response_b, criterion):
        """比較兩個回覆"""
        prompt = f"""
你是一個公正的評審。比較以下兩個回覆，根據 {criterion} 評分。

回覆 A: {response_a}
回覆 B: {response_b}

哪個更好？直接回答 "A" 或 "B"。
"""

        winner = self.judge.generate(prompt)
        return winner

    def score(self, response, rubric):
        """根據標準評分"""
        prompt = f"""
根據以下標準評分 (1-5):
{rubric}

回覆: {response}

分數:
"""

        score = self.judge.generate(prompt)
        return score
```

### 自動化指標

```python
class AutoMetrics:
    """自動化評估指標"""

    @staticmethod
    def bleu(pred, ref):
        """BLEU 分數 - 文本相似度"""
        from torchmetrics.text import BLEUScore
        bleu = BLEUScore()
        return bleu([pred], [[ref]]).item()

    @staticmethod
    def rouge(pred, ref):
        """ROUGE 分數 - 摘要評估"""
        from torchmetrics.text import ROUGEScore
        rouge = ROUGEScore()
        return rouge(pred, [ref])

    @staticmethod
    def exact_match(pred, ref):
        """精確匹配"""
        return int(pred.strip() == ref.strip())
```

---

## 6. 評估實踐

### 建立評估集

```python
class EvaluationDataset:
    """評估資料集管理"""

    def __init__(self):
        self.samples = []

    def add_sample(self, prompt, expected, metadata=None):
        self.samples.append({
            "prompt": prompt,
            "expected": expected,
            "metadata": metadata or {}
        })

    def split(self, test_ratio=0.2):
        import random
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * test_ratio)
        return self.samples[split_idx:], self.samples[:split_idx]
```

### 評估流程

```python
def evaluate_model(model, dataset):
    """完整評估流程"""

    results = []

    for sample in dataset:
        # 生成回覆
        response = model.generate(sample["prompt"])

        # 評估
        is_correct = evaluate_response(
            response,
            sample["expected"]
        )

        results.append({
            "prompt": sample["prompt"],
            "response": response,
            "expected": sample["expected"],
            "correct": is_correct
        })

    # 計算指標
    accuracy = sum(r["correct"] for r in results) / len(results)

    return {
        "accuracy": accuracy,
        "results": results
    }
```

---

## 7. 挑戰與限制

### 評估限制

| 挑戰 | 說明 |
|------|------|
| **數據洩漏** | 測試數據可能出現在訓練集中 |
| **評估偏見** | 自動化評估可能有偏見 |
| **主觀性** | 某些任務難以客觀評估 |
| **覆盖不全** | 基準無法涵蓋所有能力 |

### 避免過擬合

```python
def avoid_data_leakage(model, test_set, train_data):
    """檢查數據洩漏"""

    leaks = []
    for sample in test_set:
        # 簡單檢查: 完全匹配
        if sample["prompt"] in train_data:
            leaks.append(sample)

    return {
        "leak_count": len(leaks),
        "leaks": leaks,
        "warning": "建議更換測試集" if leaks else "無洩漏"
    }
```

---

## 8. 相關技術

| 技術 | 關係 |
|------|------|
| **RLHF** | 使用人類反饋提升評估 |
| **Chatbot Arena** | 開放評估平台 |
| **LMSYS** | 大模型評估組織 |

---

## 延伸閱讀

- [OpenLLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [HumanEval Paper](https://arxiv.org/abs/2203.08914)
- [Chatbot Arena](https://chat.lmsys.org/)