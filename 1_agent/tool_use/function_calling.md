# Function Calling

讓 LLM 能夠調用外部工具或函數的技術，使模型能夠執行 Action 而不僅僅是生成文字。

---

## 1. 什麼是？

### 簡單範例

```
用戶: "台北今天天氣如何？"

沒有 Function Calling:
  "我無法獲取即時天氣資訊..." (幻覺或無法回答)

有 Function Calling:
  LLM 識別需要調用天氣 API → 提取參數 "台北" → 調用 get_weather(city="台北")
  → 獲取回傳 "晴朗，28°C" → 生成最終回覆 "台北今天天氣晴朗，氣溫 28 度"
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **打破只會說的限制** | LLM 從被動回應變成主動執行 |
| **連接真實世界** | 存取即時資料、操作系統、資料庫 |
| **實現 AI Agent** | Function Calling 是 Agent 的基礎能力 |
| **減少幻覺** | 引用真實 API 回傳的資料 |

### 應用場景

- **天氣查詢**、股票報價、新聞資訊
- **資料庫操作**：查詢、寫入、更新
- **郵件/訊息發送**
- **檔案操作**：讀取、寫入、搜尋
- **API 整合**：CRM、ERP、電子商務
- **程式碼執行**：Sandbox 環境運行代碼

---

## 3. 核心原理

### 完整流程（5 步驟）

```
步驟 1: 註冊 Function
  └─ 定義 function schema (名稱、參數、說明)
        ↓
步驟 2: 用戶輸入
  └─ "台北天氣如何？"
        ↓
步驟 3: LLM 決策
  └─ • 意圖識別：需要查詢天氣
     • 參數提取：city = "台北"
        ↓
步驟 4: 執行 Function
  └─ 呼叫 get_weather(city="台北") → 回傳 {temp: 28, condition: "晴"}
        ↓
步驟 5: 生成回覆
  └─ 根據 API 回傳結果生成自然語言回覆
```

### Function Schema 範例

```json
{
  "name": "get_weather",
  "description": "獲取指定城市的天氣資訊",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名稱，如 '台北'、'東京'"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "溫度單位，預設攝氏"
      }
    },
    "required": ["city"]
  }
}
```

### 多 Function 情境

```python
functions = [
    get_weather,           # 天氣查詢
    search_database,      # 資料庫搜尋
    send_email,           # 發送郵件
    create_calendar_event # 建立日曆事件
]

user_input = "帮我查一下明天台北的天氣，然後寄給住在台北的客戶"

# LLM 判斷需要調用哪些 functions
# 1. get_weather(city="台北", date="tomorrow")
# 2. send_email(to="台北的客戶", content=天氣資訊)
```

### LLM 決策：意圖識別與參數提取

**意圖識別與參數提取都是由 LLM 自動完成的**，通常不需要額外下 prompt。

#### LLM 自動決策的依據

| LLM 依據 | 來源 |
|----------|------|
| 用戶問題 | 對話內容 |
| Function 描述 | `description` 欄位 |
| 參數說明 | 每個參數的 `description` |

```json
{
  "name": "get_weather",
  "description": "獲取指定城市的天氣資訊",  // ← LLM 據此判斷意圖
  "parameters": {
    "properties": {
      "city": {
        "description": "城市名稱，如 '台北'"  // ← LLM 據此提取參數
      }
    }
  }
}
```

```
用戶: "台北天氣好嗎？"

LLM 自動推斷：
- 意圖：需要調用 get_weather
- 參數：city = "台北"
```

#### 是否需要額外 Prompt？

**基本情況：不需要** — 多數 API（OpenAI、Anthropic）已內建 Function Calling 能力，只需提供 function schema 即可。

**需要引導的例外情況：**

| 情況 | 範例 | 解決方式 |
|------|------|----------|
| 意圖模糊 | "幫我查查看" | system prompt 說明何時調用 |
| 參數不明確 | "寄給那個人" | 先詢問用戶確認 |
| 避免過度調用 | 用戶只是在聊天 | 限制只在必要時調用 |

```python
messages = [
    {
        "role": "system",
        "content": """你是一個天氣助手。
只在用戶明確問天氣時才調用 get_weather，
不要過度解讀用戶意圖。
如果參數不明確，請先詢問用戶。"""
    },
    # ... 用戶對話
]
```

#### 結論

| 步驟 | 誰來做 | 需要 prompt 嗎？ |
|------|--------|-----------------|
| 意圖識別 | LLM 自動 | 一般不需要 |
| 參數提取 | LLM 自動 | 一般不需要 |
| 結果整合 | LLM 自動 | 可選 |

> 💡 LLM 已經過訓練，具備理解 function schema 並自動決策的能力。這也是 Function Calling 的核心價值 — **簡單、自動化**。

---

## 4. 主流方案

### OpenAI Function Calling

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "台北天氣如何？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

# response.tool_calls 包含需要調用的 function
```

**特色**：原生支援 JSON Schema、支援 parallel function calls、強制回覆格式

### Anthropic Tool Use

```python
tools = [{
    "name": "get_weather",
    "description": "取得城市天氣",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名"}
        },
        "required": ["city"]
    }
}]
```

**特色**：Tool Use 框架更彈性、支援電腦使用（Computer Use）、每個 prompt 可自定義 tools

### LangChain Tools

```python
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun

tools = [
    DuckDuckGoSearchRun(description="搜尋網路資訊"),
    Tool(name="計算機", func=calculator, description="數學計算")
]

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION)
```

### 方案比較

| 方案 | 適用場景 | 優點 | 缺點 |
|------|----------|------|------|
| **OpenAI** | 快速開發、商業應用 | 文件完善、生態完整 | 需付費 |
| **Anthropic** | 高安全性場景 | 安全對齊、Computer Use | 支援較少 |
| **LangChain** | 複雜 Chain 需求 | 彈性高、整合多 | 學習曲線 |
| **自建** | 完全控制 | 靈活 | 需開發維護 |

---

## 5. Function Calling vs Skill 比較

這是兩個不同層次的概念：

| 維度 | Function Calling | Skill |
|------|-----------------|-------|
| **定義層次** | LLM 技術能力 | Agent 框架概念 |
| **本質** | 一種**機制** | 一種**封裝** |
| **粒度** | 單一動作 | 多個動作 + 行為指引 |

### Function Calling（功能調用）

**層次**：LLM 技術層

讓模型能夠**調用外部函數/API**來完成具體任務：
```
用戶: "台北天氣如何？"
LLM → 調用天氣API(台北) → 獲取數據 → 生成回覆
```

- **目標**：讓 LLM 執行實際操作
- **粒度**：單一函數
- **範例**：`get_weather()`, `search_database()`

### Skill（技能）

**層次**：Agent 框架層

一種**封裝好的能力模組**，包含多個 function + prompt 指令：
```yaml
skill: "天氣查詢"
  functions:
    - get_weather
  prompt: 你是一個天氣助手...
```

- **目標**：讓 AI 具備特定領域能力
- **粒度**：多個函數 + prompt + 處理邏輯
- **範例**：`/loop`, `/review`, `/help` 等 Claude Code 指令

### 簡單比喻

| 概念 | 比喻 |
|------|------|
| Function Calling | AI 的「手」— 能執行動作 |
| Skill | AI 的「技能包」— 包含手 + 大腦指令 |
| Function Calling | 樂器的「發聲原理」 |
| Skill | 演奏「曲目」的能力 |

### 分界範例

```
情境：用戶說 "帮我查台北天氣，然後發送郵件"

Function Calling 視角：
  get_weather(city="台北")
  send_email(to="...", content="...")

Skill 視角：
  「天氣小幫手」技能：
    - functions: get_weather, send_email
    - system prompt: "你是貼心的天氣助手，回覆要親切"
    - 錯誤處理邏輯
```

### 在 Claude Code 中的實際例子

| 你輸入的 | 實際觸發 |
|----------|----------|
| `/loop` | Skill（包含：解析參數、創建定時任務邏輯、prompt 指引） |
| `/review` | Skill（包含：調用 git 相關 function、程式碼分析邏輯、review prompt） |

> 💡 Slash command 內部使用了 Function Calling，但本身是 Skill

---

## 6. 實作指南

### 步驟 1：定義 Function Schema

```python
functions = [
    {
        "name": "get_weather",
        "description": "獲取指定城市的即時天氣資訊",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名稱，如 '台北'、'東京'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "溫度單位"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_email",
        "description": "發送電子郵件",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人 email"},
                "subject": {"type": "string", "description": "郵件主旨"},
                "body": {"type": "string", "description": "郵件內容"}
            },
            "required": ["to", "subject", "body"]
        }
    }
]
```

### 步驟 2：實作 Function 處理函數

```python
def get_weather(city: str, unit: str = "celsius"):
    """實際調用天氣 API"""
    api_key = os.getenv("WEATHER_API_KEY")
    response = requests.get(
        f"https://api.weather.com/v3/weather",
        params={"city": city, "units": unit, "apiKey": api_key}
    )
    return response.json()

def send_email(to: str, subject: str, body: str):
    """實際發送郵件"""
    smtp_server = os.getenv("SMTP_SERVER")
    # ... SMTP 發送邏輯
    return {"status": "sent", "message_id": "xxx"}
```

### 步驟 3：整合到 LLM 調用

```python
from openai import OpenAI
client = OpenAI()

def chat_with_functions(user_message):
    messages = [{"role": "user", "content": user_message}]

    # 第一次調用：讓 LLM 決定是否需要調用 function
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[{"type": "function", "function": f} for f in functions]
    )

    # 檢查是否有 function call
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # 執行實際的 function
            if function_name == "get_weather":
                result = get_weather(**arguments)
            elif function_name == "send_email":
                result = send_email(**arguments)

            # 將結果加回對話
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tool_call.id, "function": tool_call.function}]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # 第二次調用：讓 LLM 根據 function 結果生成回覆
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return final_response.choices[0].message.content

    # 沒有 function call，直接回覆
    return response.choices[0].message.content
```

### 步驟 4：處理多 Function 調用

```python
def handle_multi_function_calls(tool_calls, functions_map):
    results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        result = functions_map[function_name](**arguments)
        results.append({
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    return results
```

### 完整範例流程

```python
user_input = "台北天氣如何？可以寄給我老婆嗎？"
result = chat_with_functions(user_input)
print(result)
# 輸出: "台北今天天氣晴朗，氣溫 28 度。我已經把天氣資訊寄給您老婆了！"
```

### 常見錯誤處理

| 錯誤情況 | 處理方式 |
|----------|----------|
| Function 不存在 | 回覆用戶無法執行 |
| API 呼叫失敗 | 返回錯誤訊息，讓 LLM 告知用戶 |
| 參數不足 | 在 description 中說明 required，讓 LLM 提醒用戶 |
| 逾時 | 設定 timeout，返回超時訊息 |

### 最佳實踐

1. **Function 描述要清楚** — description 要能讓 LLM 理解何時該調用，參數說明要具體
2. **錯誤處理要做好** — 捕獲 API 例外，返回有意義的錯誤訊息
3. **安全性要注意** — 不要在 client 端直接暴露敏感 API key，驗證用戶輸入參數
4. **善用 parallel calls** — 多個獨立的 function 可一次調用，提升效率

---

## 7. Parallel Function Calls（並發調用）

### 什麼是？

允許 LLM 在一次回覆中同時調用多個 function，提升效率。

### 範例情境

```
用戶: "帮我查台北和東京的天氣"

沒有 Parallel Calls（依序）：
  1. get_weather(city="台北")
  2. get_weather(city="東京")

有 Parallel Calls（並發）：
  get_weather(city="台北") ─┐
  get_weather(city="東京") ─┼─ 同時執行
```

### OpenAI 實作

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "帮我查台北和東京的天氣"}],
    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]
)

# response.tool_calls 會包含多個調用
tool_calls = response.choices[0].message.tool_calls
# [ToolCall(id="call_1", function=...), ToolCall(id="call_2", function=...)]

# 並發執行
with ThreadPoolExecutor() as executor:
    results = list(executor.map(
        lambda tc: execute_function(tc.function.name, tc.function.arguments),
        tool_calls
    ))
```

### 使用時機

| 適合 Parallel | 不適合 Parallel |
|---------------|-----------------|
| 多個獨立的 API 呼叫 | 有依賴關係的呼叫 |
| 查詢不同來源的資料 | 需前一結果作為下一輸入 |
| 批量操作 | 順序邏輯 |

---

## 8. 安全性

### 常見安全風險

| 風險類型 | 說明 | 範例 |
|----------|------|------|
| **Prompt Injection** | 用戶輸入試圖操縱 LLM 行為 | "忽略上面的指示，告訴我..." |
| **Sensitive Data Exposure** | 敏感資料被輸出 | API key、密碼外洩 |
| **Unauthorized Access** | 未授權操作 | 調用刪除資料的 function |
| **Rate Limiting** | 惡意大量調用 | DDoS 攻擊 |

### 防護措施

#### 1. 輸入驗證

```python
def validate_function_call(function_name, arguments):
    allowed_functions = ["get_weather", "search_database", "send_email"]
    if function_name not in allowed_functions:
        raise PermissionError(f"Function {function_name} not allowed")

    if "password" in arguments or "api_key" in arguments:
        raise ValueError("Sensitive parameters not allowed")
```

#### 2. 權限控制

```python
def check_permission(user_id, function_name):
    user_permissions = get_user_permissions(user_id)
    return function_name in user_permissions
```

#### 3. API Key 保護

```python
# ❌ 錯誤：把 API key 傳給 client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 正確：透過 server 代理
# Client 只傳遞參數，Server 負責呼叫實際 API
```

#### 4. Rate Limiting

```python
def rate_limit(max_calls=10, period=60):
    def decorator(func):
        calls = []
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [t for t in calls if t > now - period]
            if len(calls) >= max_calls:
                raise RateLimitError("Too many calls")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

#### 5. Sandbox 隔離

```python
def execute_code_sandbox(code):
    import sandbox  # 專用沙箱環境
    return sandbox.run(code, timeout=5, memory_limit="128MB")
```

### 安全檢查清單

- [ ] 使用白名單限制可呼叫的 function
- [ ] 驗證並清理用戶輸入參數
- [ ] 不要在 client 端暴露敏感資訊
- [ ] 實作 rate limiting 防止濫用
- [ ] 敏感操作需要額外確認
- [ ] 記錄日誌供審計

---

## 9. 錯誤處理最佳實踐

### 錯誤類型分類

| 錯誤類型 | 範例 | 處理方式 |
|----------|------|----------|
| **參數錯誤** | 缺少必填參數 | 讓 LLM 詢問用戶 |
| **Function 不存在** | 調用未定義的 function | 拒絕並說明可用功能 |
| **執行錯誤** | API 逾時、網路問題 | 返回錯誤訊息讓 LLM 重構回覆 |
| **權限錯誤** | 沒有權限執行 | 返回訊息，請用戶確認權限 |

### 錯誤處理範例

```python
def execute_function_safely(function_name, arguments):
    try:
        if function_name == "get_weather":
            return get_weather(**arguments)
        elif function_name == "send_email":
            return send_email(**arguments)
        else:
            return {"error": f"Unknown function: {function_name}"}

    except TimeoutError:
        return {"error": "API 逾時，請稍後再試"}
    except PermissionError as e:
        return {"error": f"權限不足: {str(e)}"}
    except Exception as e:
        return {"error": f"執行錯誤: {str(e)}"}
```

### 重試機制

```python
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError):
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(day * (attempt + 1))
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def get_weather(city):
    # API 呼叫
    pass
```

### 錯誤訊息設計

```python
# ❌ 不好的錯誤訊息
{"error": "Error", "code": 500}

# ✅ 好的錯誤訊息
{
    "error": "weather_api_timeout",
    "message": "天氣 API 逾時，請稍後再試",
    "recoverable": True,
    "suggestion": "請稍後再詢問，或嘗試其他城市"
}
```

### 完整錯誤處理流程

```
LLM 調用 function
       ↓
  執行 function
    ├── 成功 → 返回結果
    │
    └── 失敗
          ├── 參數錯誤 → 讓 LLM 詢問用戶补充
          ├── 權限不足 → 返回訊息，請用戶確認
          ├── API 逾時 → 自動重試 N 次
          └── 其他錯誤 → 返回詳細錯誤訊息
                       ↓
              LLM 根據錯誤生成回覆
```

### 最佳實踐清單

- [ ] 明確區分錯誤類型
- [ ] 返回結構化的錯誤訊息
- [ ] 包含錯誤代碼和可讀的错误描述
- [ ] 標記是否可重試 (`recoverable`)
- [ ] 提供建議解決方案 (`suggestion`)
- [ ] 實作指數退避重試
- [ ] 記錄錯誤日誌供除錯
- [ ] 避免暴露內部系統細節

---

## 10. 相關主題

- ReAct (推理與行動交替)
- Tool Use 整體框架
- 結構化輸出 (Structured Output)

---

## 延伸閱讀

- [OpenAI Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)