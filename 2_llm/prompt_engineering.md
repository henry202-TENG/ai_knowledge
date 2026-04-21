# Prompt Engineering

優化輸入提示以獲得更好輸出的技術，是目前最有效的 LLM 調優方法之一。

---

## 1. 什麼是？

### 深度定義

**Prompt Engineering** 本質上是一種與 LLM 溝通的藝術和科學，其核心原理基於：

```
┌─────────────────────────────────────────────────────────────────────┐
│                Prompt Engineering 基礎原理                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LLM 本質: 下一個 token 預測器                                        │
│                                                                      │
│  Prompt = 條件機率分佈的「控制訊號」                                   │
│                                                                      │
│  P(next_token | prompt, history)                                     │
│                                                                      │
│  好的 Prompt 的作用:                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. 縮小答案空間 - 明確輸出期望                                  │  │
│  │  2. 激發相關知識 - 觸發模型內部相關表示                           │  │
│  │  3. 建立推理框架 - 引導思考過程                                 │  │
│  │  4. 控制風格/語氣 - 調整輸出特徵                                │  │
│  │  5. 提供執行策略 - 給出具體行動指引                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 簡單範例

```
基礎 Prompt:
  "把這段文字翻譯成英文"

優化後 Prompt:
  "你是一個專業的學術翻譯員，擅長翻譯科技論文。請將以下中文段落翻譯成正式的學術英文，保留專業術語的原文。翻譯時請注意：1) 語意準確 2) 術語統一 3) 語法正確"
```

---

## 2. 核心原則

### CLEAR 原則

| 原則 | 說明 | 範例 |
|------|------|------|
| **C**ontext | 提供充足背景 | "假設你是資深工程師..." |
| **L**imit | 明確限制範圍 | "只考慮開源方案..." |
| **E**xample | 給出範例 | "例如:..." |
| **A**ction | 清楚動作 | "請分析並給出建議" |
| **R**ole | 設定角色 | "你是一個..." |

### BUILD 原則

```
B - Background (背景): 任務背景
U - Urgency (緊急程度): 時間限制
I - Details (細節): 具體要求
L - Deadline (期限): 完成時間
D - Delivery (交付): 輸出格式
```

---

## 3. 核心技巧

### Zero-shot vs Few-shot

```python
# Zero-shot (無範例)
prompt = "把以下情緒分類: '今天工作好累'"
# 輸出: 可能準確

# Few-shot (有範例)
prompt = """
把以下情緒分類:
'今天工作好累' -> 負面
'中了彩票!' -> 正面
'天氣不錯' -> 中性
'收到禮物好開心' ->
"""
# 輸出: 正面 (更精確)
```

### Chain of Thought (CoT)

```python
# 標準 Prompt
"計算 23 * 47 = ?"

# CoT Prompt
"讓我們一步步思考:
23 * 47 = 23 * (50 - 3)
= 23 * 50 - 23 * 3
= 1150 - 69
= 1081"

# 結果: LLM 學會展示推理過程
```

### 深度 CoT 技術

#### 1. Self-Consistency (自我一致性)

```python
"""
Self-Consistency 核心思想:
讓 LLM 生成多條推理路徑，選擇最一致的答案

Step 1: 使用 CoT 生成多個解答路徑
Step 2: 選擇出現頻率最高的答案
"""

self_consistency_prompt = """
讓我們用多種方法解答這個問題，然後比較結果：

問題: {question}

方法 1: [第一種解法]
...
答案: [答案 1]

方法 2: [第二種解法]
...
答案: [答案 2]

方法 3: [第三種解法]
...
答案: [答案 3]

最終答案: [選擇最一致的答案]
"""
```

#### 2. Tree of Thoughts (ToT)

```python
"""
Tree of Thoughts:
擴展 CoT 到樹狀結構，支援探索和回溯
"""

tot_prompt = """
你是一個解決問題的決策樹。請探索多種可能的思考路徑：

問題: {question}

路徑 A:
- 初始假設: ...
- 推理步驟 1: ...
- 推理步驟 2: ...
- 結論: ...

路徑 B:
- 初始假設: ...
- 推理步驟 1: ...
- 結論: ...

路徑 C:
- 初始假設: ...
- 推理步驟 1: ...
- 結論: ...

評估各路徑，選擇最佳解決方案:
"""
```

#### 3. CoT + Few-shot 組合

```python
"""
CoT + Few-shot 最強组合:
"""

cot_fewshot_prompt = """
請一步步思考並回答以下問題。

範例 1:
問題: 5 個蘋果，每個 3 元，總共多少元？
思考: 5 × 3 = 15
答案: 15 元

範例 2:
問題: 一輛車每小時行駛 60 公里，3 小時行駛多少公里？
思考: 60 × 3 = 180
答案: 180 公里

現在請回答:
問題: {question}
"""
```

### ReAct (Reasoning + Acting)

```python
reAct_prompt = """
你是一個 AI 助手。請按照以下格式回答:

Thought: [你的思考過程]
Action: [要執行的行動]
Observation: [行動結果]

範例:
Thought: 我需要計算這個數學問題
Action: 使用計算器 23*47
Observation: 1081

現在開始:
問題: 如果有 100 元，每天花費 5 元，幾天後會用完?
"""
```

---

## 4. 進階技巧

### Role Playing

```python
role_prompt = """
你是一個資深軟體架構師，有 20 年設計大型系統的經驗。

你的風格:
- 重視可擴展性
- 傾向微服務架構
- 強調監控和 Observability

請評論以下系統設計:
[系統描述]
"""
```

### Style Guidance

```python
style_prompt = """
用以下風格撰寫文章:
- 語氣: 專業但親和
- 段落長度: 2-3 句話
- 避免技術術語，或首次使用時解釋
- 使用主動語態

主題: [你的主題]
"""
```

### Structure Output

```python
structure_prompt = """
請按照以下 JSON 格式輸出:

{
  "summary": "100 字內的摘要",
  "key_points": ["要點1", "要點2", "要點3"],
  "action_items": [
    {"task": "任務描述", "owner": "負責人", "deadline": "日期"}
  ]
}

內容: [你的內容]
"""
```

---

## 4.5 進階 Prompt 模式

### 4.5.1 System Prompt 工程

```python
"""
System Prompt 是最強大的 Prompt 類型之一
"""

# ❌ 弱 System Prompt
system_prompt = "你是一個AI助手"

# ✅ 強 System Prompt
system_prompt = """
你是一個專業的 {role}。

專業領域: {domain}
經驗水平: {experience_level}

行為準則:
1. {guideline_1}
2. {guideline_2}
3. {guideline_3}

輸出約束:
- 輸出語言: {language}
- 格式要求: {format}
- 長度限制: {length_limit}

範例輸出:
{examples}
"""
```

### 4.5.2 Prompt 注入攻擊與防禦

```python
class PromptSecurity:
    """Prompt 安全性處理"""

    @staticmethod
    def detect_injection(user_input: str) -> bool:
        """檢測 Prompt 注入"""

        dangerous_patterns = [
            "忽略之前的指示",
            "ignore previous",
            "disregard",
            "新的指令是",
            "系統 prompt",
            "你現在是",
            "forget everything"
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in user_input.lower():
                return True

        return False

    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """清理用戶輸入"""

        # 移除可能的注入嘗試
        sanitized = user_input

        # 移除系統提示詞
        system_keywords = ["system:", "prompt:", "instructions:"]
        for keyword in system_keywords:
            if keyword in sanitized.lower():
                # 截斷之後的內容
                sanitized = sanitized.lower().split(keyword)[0]

        return sanitized.strip()

    @staticmethod
    def build_secure_prompt(
        system_prompt: str,
        user_input: str,
        injection_detection: bool = True
    ) -> dict:
        """構建安全的 Prompt"""

        if injection_detection and PromptSecurity.detect_injection(user_input):
            user_input = "[過濾的輸入]"

        return {
            "system": system_prompt,
            "user": user_input
        }
```

### 4.5.3 Prompt 版本管理

```python
class PromptVersionManager:
    """Prompt 版本管理"""

    def __init__(self):
        self.versions = {}
        self.metrics = {}

    def register_version(
        self,
        name: str,
        prompt: str,
        test_results: dict
    ):
        """註冊新版本"""

        self.versions[name] = {
            "prompt": prompt,
            "test_results": test_results,
            "timestamp": datetime.now(),
            "performance_score": self._calculate_score(test_results)
        }

    def select_best_version(self, metric: str = "accuracy") -> str:
        """選擇最佳版本"""

        best_version = max(
            self.versions.items(),
            key=lambda x: x[1]["test_results"].get(metric, 0)
        )

        return best_version[0]

    def ab_test(
        self,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5
    ) -> dict:
        """A/B 測試"""

        return {
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "metrics_to_track": ["latency", "accuracy", "user_satisfaction"]
        }
```

---

## 5. 最佳實踐

### Prompt 模板

```python
class PromptTemplate:
    """可重複使用的 Prompt 模板"""

    template = """你是 {role}。

背景: {background}

任務: {task}

要求:
{requirements}

輸出格式:
{output_format}

{examples}"""

    def __init__(self, **kwargs):
        self.variables = kwargs

    def render(self, **kwargs):
        """填充變數並渲染"""
        merged = {**self.variables, **kwargs}
        return self.template.format(**merged)


# 使用範例
template = PromptTemplate(
    role="數據分析師",
    background="幫助電商分析銷售數據",
    task="分析上月的銷售趨勢",
    requirements="1. 使用圖表說明\n2. 找出成長和下降的產品類別",
    output_format="以 Markdown 格式輸出",
    examples="範例: 2024年1月銷售額..."
)

prompt = template.render(month="2024年3月")
```

### Iteration 框架

```
1. 測試基礎 Prompt
2. 分析輸出問題
3. 調整一個元素
4. 測試並比較
5. 重複直到滿意
```

### A/B 測試

```python
def evaluate_prompt(prompt, test_cases):
    """評估不同 Prompt 的效果"""

    results = []
    for case in test_cases:
        response = llm.generate(prompt.format(**case))
        score = evaluate(response, case["expected"])
        results.append({
            "prompt": prompt,
            "case": case,
            "response": response,
            "score": score
        })

    return results
```

---

## 6. 常見錯誤

| 錯誤 | 修正 |
|------|------|
| 過於簡潔 | 添加詳細說明和範例 |
| 模糊不清 | 使用具體明確的詞彙 |
| 矛盾要求 | 檢查並移除衝突的指示 |
| 忽略格式 | 明確指定輸出格式 |
| 一次改太多 | 每次只改一個元素 |

---

## 7. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Function Calling** | Prompt 設計影響 function 調用準確性 |
| **ReAct** | Prompt 技術的一種應用 |
| **Few-shot Learning** | Prompt 工程的核心技術 |
| **RAG** | Prompt 需要結合檢索內容 |

---

## 延伸閱讀

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/prompt-engineering)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)