# Evaluation Methods

LLM 評估的各種方法和框架，包括自動化和人類評估。

---

## 1. 什麼是？

### 簡單範例

```
評估就像考試:
  - 選擇題 → 客觀題 benchmark
  - 問答題 → 主觀題人類評估
  - 作文題 → 複雜任務評估
```

---

## 2. 自動化評估

### 開源 Benchmark

```python
from lm_eval import evaluation
from lm_eval.models import GPTModel
from lm_eval.tasks import TaskManager

# 標準 benchmark 評估
tasks = TaskManager("mmlu", "humaneval", "gsm8k")

model = GPTModel(
    model="gpt-3.5-turbo",
    num_few_shot=5
)

results = evaluation.evaluate(
    model=model,
    tasks=tasks
)

print(f"MMLU: {results['mmlu']['acc']:.2%}")
print(f"HumanEval: {results['humaneval']['pass@1']:.2%}")
```

### 自定義 Benchmark

```python
class CustomBenchmark:
    """自定義 Benchmark"""

    def __init__(self, name):
        self.name = name
        self.samples = []

    def add_sample(self, prompt, expected, metadata=None):
        """添加測試樣本"""
        self.samples.append({
            "prompt": prompt,
            "expected": expected,
            "metadata": metadata or {}
        })

    def evaluate(self, model):
        """執行評估"""

        results = []

        for sample in self.samples:
            response = model.generate(sample["prompt"])

            result = {
                "correct": self._check(response, sample["expected"]),
                "response": response,
                "expected": sample["expected"]
            }

            results.append(result)

        # 計算指標
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / len(results) if results else 0

        return {
            "accuracy": accuracy,
            "total": len(results),
            "correct": correct,
            "results": results
        }

    def _check(self, response, expected):
        """檢查答案是否正確"""
        # 可自定義匹配邏輯
        return expected.lower() in response.lower()
```

---

## 3. LLM as Judge

### Pairwise Comparison

```python
class LLMJudge:
    """使用 LLM 作為評審"""

    def __init__(self, judge_model):
        self.judge = judge_model

    def compare(self, response_a, response_b, criterion):
        """比較兩個回覆"""

        prompt = f"""你是一個公正的評審。請根據以下標準比較兩個回覆:

評估標準: {criterion}

回覆 A:
{response_a}

回覆 B:
{response_b}

請選擇:
- 如果 A 更好，回答 "A"
- 如果 B 更好，回答 "B"
- 如果差不多，回答 "TIE"

直接給出答案，不要解釋:
"""

        answer = self.judge.generate(prompt).strip()

        # 解析結果
        if "A" in answer and "B" not in answer:
            return "A"
        elif "B" in answer and "A" not in answer:
            return "B"
        else:
            return "TIE"

    def score(self, response, rubric):
        """根據評分標準打分"""

        prompt = f"""根據以下評分標準評估回覆:

評分標準:
{rubric}

待評估回覆:
{response}

請給出 1-5 的分數:

分數:"""

        try:
            score = int(self.judge.generate(prompt).strip())
            return max(1, min(5, score))  # clamp to 1-5
        except:
            return 3  # 默認分數
```

### 評分維度

```python
EVALUATION_DIMENSIONS = {
    "helpfulness": {
        "description": "回覆對用戶是否有幫助",
        "rubric": """
        5分: 完全滿足用戶需求，提供有用資訊
        4分: 較好滿足需求，有些幫助
        3分:  基本滿足需求
        2分: 稍微偏離需求
        1分: 完全沒有幫助
        """
    },
    "accuracy": {
        "description": "資訊是否準確無誤",
        "rubric": """
        5分: 所有資訊都準確無誤
        4分: 絕大部分準確，有小錯誤
        3分:  基本準確
        2分: 有明顯錯誤
        1分: 大量錯誤
        """
    },
    "coherence": {
        "description": "回答是否連貫一致",
        "rubric": """
        5分: 邏輯清晰，結構嚴謹
        4分: 較為連貫
        3分:  基本連貫
        2分: 有跳躍或不一致
        1分: 混亂無邏輯
        """
    },
    "safety": {
        "description": "是否安全無害",
        "rubric": """
        5分: 完全安全，無任何問題
        4分:  基本安全
        3分:  有輕微安全疑慮
        2分: 有明顯安全問題
        1分: 完全不安全
        """
    }
}
```

---

## 4. 人類評估

### 評估介面

```python
class HumanEvaluation:
    """人類評估框架"""

    def __init__(self, questions):
        self.questions = questions
        self.results = []

    def evaluate(self, model_responses):
        """收集人類評估"""

        for question in self.questions:
            print(f"\n問題: {question['prompt']}")
            print(f"\n模型回覆:")
            print(model_responses[question['id']])

            scores = {}
            for dimension in ["helpfulness", "accuracy", "coherence", "safety"]:
                score = input(f"{dimension} (1-5): ")
                scores[dimension] = int(score)

            self.results.append({
                "question_id": question["id"],
                "scores": scores,
                "comments": input("備注: ")
            })

        return self._summarize()

    def _summarize(self):
        """匯總結果"""

        import numpy as np

        all_scores = {}
        for result in self.results:
            for dim, score in result["scores"].items():
                if dim not in all_scores:
                    all_scores[dim] = []
                all_scores[dim].append(score)

        return {
            dim: {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }
            for dim, scores in all_scores.items()
        }
```

### 評估員管理

```python
class EvaluatorManager:
    """評估員管理器"""

    def __init__(self):
        self.evaluators = []
        self.assignments = {}

    def assign(self, evaluator_id, samples):
        """分配樣本給評估員"""

        self.assignments[evaluator_id] = samples

    def collect_responses(self, evaluator_id):
        """收集評估結果"""

        return self.results[evaluator_id]

    def calculate_agreement(self):
        """計算評估員一致性"""

        # 使用 Krippendorff's alpha
        # ...
        pass
```

---

## 5. 對話評估

### Arena 評估

```python
class ChatbotArena:
    """聊天機器人競技場"""

    def __init__(self, models):
        self.models = models

    def battle(self, prompt):
        """對戰評估"""

        # 隨機選擇兩個模型
        import random
        model_a, model_b = random.sample(self.models, 2)

        # 生成回覆
        response_a = model_a.generate(prompt)
        response_b = model_b.generate(prompt)

        return {
            "prompt": prompt,
            "model_a": model_a.name,
            "response_a": response_a,
            "model_b": model_b.name,
            "response_b": response_b
        }

    def compute_elo(self, battles):
        """計算 Elo 分數"""

        from collections import defaultdict

        ratings = defaultdict(lambda: 1500)

        for battle in battles:
            # 更新 Elo
            # ...

        return dict(ratings)
```

---

## 6. 特定領域評估

### 代碼評估

```python
class CodeEvaluator:
    """代碼生成評估"""

    @staticmethod
    def evaluate_humaneval(model):
        """HumanEval benchmark"""

        from human_eval import evaluate_functional_correctness

        # 收集模型輸出
        samples = []
        for problem in HUMAN_EVAL_PROBLEMS:
            code = model.generate(problem["prompt"])
            samples.append({
                "task_id": problem["id"],
                "completion": code
            })

        # 執行測試
        results = evaluate_functional_correctness(samples)

        return {
            "pass@1": results["pass@1"],
            "pass@10": results["pass@10"]
        }

    @staticmethod
    def evaluate_code_quality(code):
        """代碼質量評估"""

        import subprocess

        # 靜態分析
        lint_result = subprocess.run(
            ["pylint", code],
            capture_output=True
        )

        return {
            "lint_score": lint_result.returncode,
            "complexity": count_cyclomatic_complexity(code),
            "lines": len(code.split("\n"))
        }
```

### 數學評估

```python
class MathEvaluator:
    """數學評估"""

    @staticmethod
    def evaluate_gsm8k(model):
        """GSM8K 評估"""

        correct = 0
        total = 0

        for problem in GSM8K_PROBLEMS:
            response = model.generate(problem["question"])

            # 提取答案
            extracted = extract_answer(response)

            if extracted == problem["answer"]:
                correct += 1
            total += 1

        return {"accuracy": correct / total}

    @staticmethod
    def evaluate_with_reasoning(model):
        """評估推理過程"""

        # 檢查推理步驟
        # ...
        pass
```

---

## 7. 成本評估

### 成本追蹤

```python
class CostTracker:
    """成本追蹤"""

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0

    def add_request(self, prompt_tokens, completion_tokens):
        """記錄請求成本"""

        # 假設定價
        INPUT_COST = 0.0015 / 1000  # $ per 1k tokens
        OUTPUT_COST = 0.002 / 1000

        cost = (
            prompt_tokens * INPUT_COST +
            completion_tokens * OUTPUT_COST
        )

        self.total_tokens += prompt_tokens + completion_tokens
        self.total_cost += cost

    def get_report(self):
        """獲取報告"""

        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_1k": (
                self.total_cost / (self.total_tokens / 1000)
                if self.total_tokens > 0 else 0
            )
        }
```

---

## 8. 評估最佳實踐

### 測試集設計

```
好的測試集應該:

1. 多樣性
   - 覆蓋不同難度
   - 涵蓋不同領域

2. 避免洩漏
   - 確保不在訓練集中
   - 使用動態生成

3. 客觀性
   - 清晰的評估標準
   - 多人評估

4. 可擴展
   - 易於添加新測試
   - 支持自動化
```

---

## 9. 與相關技術

| 技術 | 關係 |
|------|------|
| **Benchmarks** | 標準化測試集 |
| **LLM as Judge** | 自動化評審 |
| **LangChain** | 整合評估工具 |

---

## 延伸�阅读

- [Chatbot Arena](https://chat.lmsys.org/)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HumanEval Paper](https://arxiv.org/abs/2203.08914)