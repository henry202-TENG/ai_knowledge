# ReAct

結合推理（Reasoning）與行動（Acting）的 AI Agent 框架，讓模型在解決問題時能夠交替進行思考與執行動作。

---

## 1. 什麼是？

### 簡單範例

```
用戶: "今天台北的天氣如何？"

ReAct 流程:

Step 1:
  Thought: 我需要查詢台北的天氣資訊
  Action: get_weather(city="台北")
  Observation: {"temp": 28, "condition": "晴朗"}

Step 2:
  Thought: 現在我可以回答用戶了
  Action: 生成最終答案
  Observation: 台北今天天氣晴朗，氣溫 28 度
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **解決幻覺問題** | 透過與環境互動獲取真實資訊，而非胡亂猜測 |
| **增強推理能力** | 讓模型能夠進行多步驟推理 |
| **實現真正的主動性** | AI 能夠主動查詢資訊而非被動回應 |
| **可追溯** | 每個 Thought/Action/Observation 可追蹤除錯 |

### 傳統 LLM 的限制

```
輸入: "今天的氣溫是多少？"
輸出: "今天氣溫 25 度" ← 可能幻覺

問題：LLM 不知道實時資訊，會胡亂猜測
```

---

## 3. 核心原理

### 公式化

```
ReAct Prompt = Thought + Action + Observation 循環
直到達成目標或達到最大迭代次數
```

### ReAct 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│  用戶輸入: "台北的天氣如何？還有明天會下雨嗎？"                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Thought                                                │
│  我需要先查詢台北今天的天氣                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Action: get_weather(city="台北")                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Observation: {"temp": 28, "condition": "晴朗"}                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Thought                                                │
│  現在查詢明天台北的天氣                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Action: get_weather(city="台北", tomorrow=true)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Observation: {"temp": 25, "condition": "小雨"}                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 生成最終答案                                            │
│  台北今天天氣晴朗，氣溫 28 度；明天會下雨，氣溫 25 度               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 實現方案

### 經典 ReAct Prompt 模板

```python
ReAct_SYSTEM_PROMPT = """請根據以下格式回答：

Thought: [你的思考過程，描述你為什麼要做這個動作]
Action: [要執行的行動，如調用工具或 API]
Observation: [行動的結果]

重複這個過程直到任務完成。
"""

# 完整的對話範例
messages = [
    {"role": "user", "content": "台北天氣如何？"},
    {"role": "assistant", "content": "Thought: 我需要查詢台北的天氣\nAction: get_weather(city=\"台北\")"},
    {"role": "system", "content": "Observation: 台北今天晴朗，28度"},
    {"role": "assistant", "content": "Thought: 我已經獲得天氣資訊\nAction: 生成最終回覆"},
]
```

### ReAct + Function Calling 結合

```python
from openai import OpenAI

def react_with_function_calling(query):
    client = OpenAI()

    messages = [{"role": "user", "content": query}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,  # 定義好的 functions
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # 檢查是否需要調用 function
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                # 執行 function
                result = execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )

                # 加入觀察結果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            # 沒有 function call，直接回覆
            return msg.content
```

### 停止條件設計

```python
def should_stop(response, max_iterations=10):
    """決定是否停止 ReAct 迴圈"""

    # 條件 1: 到達最大迭代次數
    if response.iteration >= max_iterations:
        return True

    # 條件 2: 已達成任務
    if response.is_complete:
        return True

    # 條件 3: 連續無進展
    if response.no_progress_count >= 3:
        return True

    # 條件 4: 遇到明確的失敗標記
    if response.contains("無法完成"):
        return True

    return False
```

### 錯誤恢復機制

```python
def handle_action_error(error, attempt=0):
    """處理 Action 執行錯誤"""

    if attempt >= 3:
        return "抱歉，無法完成此任務"

    error_types = {
        "timeout": "請求逾時，讓我重試一次",
        "not_found": "找不到資源，請確認輸入是否正確",
        "permission": "沒有權限訪問，讓我嘗試其他方式",
        "rate_limit": "請求太頻繁，讓我稍後重試"
    }

    return error_types.get(error.type, "發生錯誤，讓我重試")
```

---

## 5. ReAct 變體

| 變體 | 說明 | 特色 |
|------|------|------|
| **ReAct (經典)** | Thought → Action → Observation 循環 | 最通用 |
| **Act-Then-Think** | 先執行多個 Action，再統一思考 | 減少思考次數 |
| **Think-Actor** | 先思考完整計畫，再執行 | 更謹慎 |
| **Self-Ask** | 模型自己問自己問題 | 適合複雜推理 |
| **Chain of Thought** | 只有 Thought，無 Action | 簡單推理 |

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Function Calling** | ReAct 的工具調用實現方式 |
| **Tree of Thoughts** | ReAct 的進階版本，多路徑推理 |
| **Chain of Thought** | ReAct 的前身，純推理無行動 |
| **Agent** | ReAct 是 Agent 的核心機制之一 |

---

## 延伸閱讀

- [ReAct Paper (arxiv)](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [ReAct vs Chain of Thought](https://arxiv.org/abs/2210.03629)