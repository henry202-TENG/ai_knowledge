# Function Calling

讓 LLM 調用外部工具和函數的技術，是構建 AI Agent 的關鍵能力。

---

## 1. 什麼是？

### 簡單範例

```
沒有 Function Calling:
  用戶: 今天天氣如何？
  AI: 我不知道 (因為訓練數據有截止日期)

有 Function Calling:
  用戶: 今天天氣如何？
  AI → 調用天氣 API → 獲取數據 → 回答用戶
```

---

## 2. 核心原理

### 函數調用流程

```python
class FunctionCaller:
    """函數調用器"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools  # 可用工具列表

    def generate_with_functions(self, prompt):
        """帶函數調用的生成"""

        # 1. 構建包含工具描述的 prompt
        messages = self._build_messages(prompt)

        # 2. LLM 決定是否調用函數
        response = self.llm.chat(
            messages=messages,
            tools=self.tools  # 傳遞工具定義
        )

        # 3. 檢查是否需要調用函數
        if response.tool_calls:
            # 執行函數
            results = self._execute_tools(response.tool_calls)

            # 4. 將結果加入對話
            messages.extend(response.tool_calls)
            messages.append({
                "role": "tool",
                "content": results
            })

            # 5. 再次生成最終回答
            final_response = self.llm.chat(messages=messages)

            return final_response

        return response
```

### 工具定義

```python
# OpenAI function calling 格式
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取指定城市的天氣資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名稱，如台北、東京"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "溫度單位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "搜尋最新新聞",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜尋關鍵詞"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多結果數",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

---

## 3. 調用模式

### 單次調用

```python
# 簡單的單次函數調用
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "台北現在幾度？"}],
    tools=tools
)

# 檢查是否有函數調用
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]

    # 執行函數
    result = execute_function(
        tool_call.function.name,
        json.loads(tool_call.function.arguments)
    )

    print(f"天氣: {result}")
```

### 多次調用

```python
class MultiFunctionCaller:
    """多次函數調用"""

    MAX_ITERATIONS = 5

    def generate(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.MAX_ITERATIONS):
            # 生成回覆（可能帶函數調用）
            response = self.llm.chat(messages, tools=self.tools)

            if not response.tool_calls:
                # 沒有更多調用，返回結果
                return response.content

            # 執行所有函數調用
            for tool_call in response.tool_calls:
                result = self._execute(tool_call)

                # 添加結果到對話
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        # 達到最大迭代次數
        return "處理時間過長"
```

### 並行調用

```python
# 一次調用多個函數
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "比較台北和東京的天氣，還有股價"
    }],
    tools=tools  # 天氣 API + 股票 API
)

# GPT-4 可能同時調用多個函數
if response.choices[0].message.tool_calls:
    # 並行執行
    results = []

    for tool_call in response.choices[0].message.tool_calls:
        result = execute_function(tool_call.function)
        results.append(result)

    # 整合結果
    combined = self._combine_results(results)
```

---

## 4. 實作範例

### 天氣查詢

```python
import json
import requests

def get_weather(city: str, unit: str = "celsius") -> dict:
    """天氣查詢函數"""

    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.weather.com/v3/wx/conditions/current"

    response = requests.get(url, params={
        "city": city,
        "unit": unit,
        "apiKey": api_key
    })

    return response.json()

# 註冊工具
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "獲取城市天氣資訊",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    }
}

# 使用
result = get_weather("台北", "celsius")
print(f"台北溫度: {result['temperature']} 度")
```

### 資料庫查詢

```python
import sqlite3

def query_database(query: str, limit: int = 10) -> list:
    """執行 SQL 查詢"""

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    cursor.execute(query)
    results = cursor.fetchall()

    conn.close()

    return results[:limit]

# 工具定義
db_tool = {
    "function": {
        "name": "query_database",
        "description": "查詢資料庫",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        }
    }
}
```

---

## 5. 錯誤處理

### 調用失敗

```python
def safe_execute(tool_call):
    """安全執行函數"""

    try:
        # 執行函數
        result = execute_function(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )

        return {"success": True, "result": result}

    except Exception as e:
        # 返回錯誤資訊
        return {
            "success": False,
            "error": str(e),
            "message": "函數執行失敗，請重試或調整參數"
        }
```

### 重試邏輯

```python
class RetryFunctionCaller:
    """帶重試的函數調用"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def call_with_retry(self, tool_call):
        """重試調用"""

        for attempt in range(self.max_retries):
            try:
                result = execute_function(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                return result

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise

                # 指數退避
                time.sleep(2 ** attempt)

        return None
```

---

## 6. 最佳實踐

### 工具設計

```python
# 好的工具設計
GOOD_TOOL_EXAMPLE = {
    "name": "calculate",
    "description": "執行數學計算",  # 清楚描述用途
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要計算的數學表達式，如 2+2*3"
            }
        },
        "required": ["expression"]
    }
}

# 不好的工具設計
BAD_TOOL_EXAMPLE = {
    "name": "calc",  # 名稱不明確
    "description": "計算",  # 描述太簡短
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "string"}  # 參數說明不足
        }
    }
}
```

### 調用策略

```
1. 明確意圖
   - 工具名稱要清楚表達功能
   - 參數描述要完整

2. 適當顆粒度
   - 不要過於細碎
   - 也不要過於複雜

3. 錯誤處理
   - 必須有錯誤處理
   - 提供有用的錯誤訊息

4. 安全考慮
   - 驗證參數
   - 限制訪問權限
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **ReAct** | 推理 + 行動框架 |
| **LangChain** | 提供工具封裝 |
| **MCP** | 標準化工具調用 |

---

## 延伸閱讀

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/tool-use)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)