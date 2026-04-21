# Prompt Engineering

優化輸入提示以獲得更好輸出的技術，是目前最有效的 LLM 調優方法之一。

---

## 1. 什麼是？

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