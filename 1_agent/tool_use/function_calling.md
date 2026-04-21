# Function Calling

讓 LLM 能夠調用外部工具或函數的技術，使模型能夠執行 Action 而不僅僅是生成文字。

---

## 1. 什麼是？

### 深度定義

**Function Calling** 是 LLM 與外部系統交互的核心能力，本質上是將 LLM 的**推理能力**與**執行能力**分離的設計模式：

```
┌────────────────────────────────────────────────────────────────────┐
│                        Function Calling 架構                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────┐    意圖識別    ┌──────────────┐    參數提取          │
│   │ 用戶輸入  │ ───────────▶ │     LLM      │ ──────────────▶      │
│   └──────────┘               │  (決策引擎)  │                      │
│                               └──────────────┘                      │
│                                        │                            │
│                                        ▼                            │
│                               ┌──────────────┐    執行調用            │
│                               │ Tool Runtime │ ──────────────▶      │
│                               └──────────────┘      外部 API        │
│                                        │                            │
│                                        ▼                            │
│                               ┌──────────────┐    生成回覆            │
│                               │     LLM      │ ◀─────────────       │
│                               │  (生成引擎)  │    API 回傳結果        │
│                               └──────────────┘                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 技術定位

| 層次 | 技術 | 說明 |
|------|------|------|
| **應用層** | Agent Framework | LangChain, AutoGen |
| **協議層** | MCP, A2A | 標準化工具調用 |
| **能力層** | Function Calling | LLM 內建能力 |
| **執行層** | Tool Runtime | 實際函數執行 |

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

### 深度場景分析

#### 場景 1：企業知識庫問答系統

**挑戰**：
- 文件版本管理困難
- 跨系統知識整合
- 準確性要求極高

**Function Calling 解決方案**：
```
┌─────────────────────────────────────────────────────────┐
│  用戶: "去年Q3的營收報告數據是多少？"                        │
│                    │                                      │
│                    ▼                                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Step 1: 意圖識別                                     │ │
│  │   - 需要查詢營收數據                                 │ │
│  │   - 需要特定時間範圍 (去年Q3)                        │ │
│  └────────────────────────────────────────────────────┘ │
│                    │                                      │
│                    ▼                                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Step 2: 工具調用序列                                  │ │
│  │   1. search_documents(query="Q3 營收報告 2024")   │ │
│  │   2. get_financial_data(quarter="Q3", year=2024)  │ │
│  │   3. compare_with_previous(data, period="Q2")     │ │
│  └────────────────────────────────────────────────────┘ │
│                    │                                      │
│                    ▼                                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Step 3: 結果生成                                     │ │
│  │   "根據2024年Q3財務報告，營收為..."                 │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### 場景 2：多租戶 SaaS 系統

**挑戰**：
- 租戶隔離
- 權限控制
- 資源限制

**解決方案架構**：
```python
class TenantAwareFunctionRegistry:
    """租戶感知的工具註冊表"""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.allowed_tools = self._load_tenant_permissions()

    def execute(self, tool_name: str, params: dict):
        # 1. 權限檢查
        if not self._check_permission(tool_name):
            raise PermissionError(f"租戶 {self.tenant_id} 無權調用 {tool_name}")

        # 2. 隔離執行
        with TenantContext(self.tenant_id):
            result = self._execute_tool(tool_name, params)

        # 3. 審計日誌
        self._log_access(tool_name, params, result)

        return result
```

#### 場景 3：即時金融交易系統

**挑戰**：
- 延遲要求極高 (<100ms)
- 數據一致性
- 風險控制

**優化策略**：
```python
class LowLatencyFunctionExecutor:
    """低延遲函數執行器"""

    def __init__(self):
        # 預熱連接池
        self._connection_pool = self._init_pool()
        # 預編譯常用查詢
        self._prepared_statements = {}
        # 本地緩存
        self._local_cache = LRUCache(maxsize=10000)

    def execute_with_latency_budget(
        self,
        func: Callable,
        params: dict,
        budget_ms: float = 100
    ) -> Result:
        """在延遲預算內執行"""

        start = time.perf_counter()

        # 1. 快取查詢
        cache_key = self._make_key(func, params)
        if cached := self._local_cache.get(cache_key):
            return cached

        # 2. 執行（超時則回退）
        try:
            result = self._execute_with_timeout(func, params, budget_ms)
        except TimeoutError:
            # 回退到備用方案
            result = self._fallback_execute(func, params)

        # 3. 異步更新緩存
        self._async_cache_update(cache_key, result)

        return result
```

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

### 深度原理：LLM 如何決定調用函數

#### 決策流程深度解析

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM Function Calling 決策流程                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 輸入處理                                                          │
│     ┌──────────────┐                                                │
│     │ 用戶查詢 +    │                                                │
│     │ Tool Schema  │                                                │
│     └──────┬───────┘                                                │
│            │                                                         │
│            ▼                                                         │
│  2. 語意匹配                                                          │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │  LLM 內部:                                                 │   │
│     │  - 理解用戶意圖                                            │   │
│     │  - 比對 Tool Description                                   │   │
│     │  - 評估每個工具的適用性                                      │   │
│     │                                                            │   │
│     │  關鍵問題: "這個查詢是否需要外部信息/操作？"                   │   │
│     └──────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            ▼                                                         │
│  3. 參數提取                                                          │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │  - 從用戶輸入提取必要參數                                    │   │
│     │  - 類型轉換 (字串 → 數字/日期)                              │   │
│     │  - 預設值處理                                               │   │
│     │  - 驗證必需參數是否存在                                      │   │
│     └──────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            ▼                                                         │
│  4. 輸出生成                                                          │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │  tool_calls 格式:                                           │   │
│     │  {                                                          │   │
│     │    "name": "get_weather",                                   │   │
│     │    "arguments": {"city": "台北", "unit": "celsius"}        │   │
│     │  }                                                          │   │
│     └──────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 內部機制：LLM 如何「理解」Tool Schema

```python
# Tool Schema 的關鍵設計原則

schema = {
    "name": "get_weather",
    "description": "獲取指定城市的即時天氣資訊",  # ← LLM 據此判斷「何時」調用
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名稱，如 '台北'、'東京'"  # ← LLM 據此提取參數
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],  # ← 限制取值範圍
                "default": "celsius"  # ← 處理可選參數
            }
        },
        "required": ["city"]  # ← LLM 確保必需參數存在
    }
}

"""
LLM 內部處理解釋:

1. 語意理解階段:
   - "天氣" → 天氣相關工具
   - "台北" → 匹配 city 參數
   - "如何" → 需要獲取資訊 (非執行操作)

2. 參數匹配階段:
   - "台北" → 映射到 city 參數 (string type match)
   - 缺少 unit → 使用預設值 "celsius"

3. 驗證階段:
   - 檢查 required: ["city"] → city 已提供 ✓
   - 檢查 type: string → city 是字串 ✓
"""
```

#### 決策邊界與極限情況

| 情況 | LLM 行為 | 開發者應對 |
|------|----------|-----------|
| **參數不全** | 拒絕調用，詢問用戶 | 在 schema 中標記 required |
| **意圖模糊** | 不調用，直接回答或詢問 | 改善 description |
| **多工具適用** | 選擇最相關的一個 | 優化 description 區分度 |
| **無需工具** | 正常生成回覆 | 這是正確行為 |

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

### 深度挑戰與解決方案

#### 挑戰 1：Tool Schema 設計不當導致誤調用

**問題場景**：
```json
// ❌ 設計不良的 schema
{
  "name": "search",
  "description": "搜尋資訊",
  "parameters": {...}
}
```
後果：LLM 過度傾向調用，任何問題都會嘗試 search

**解決方案**：
```json
// ✅ 設計良好的 schema
{
  "name": "search_knowledge_base",
  "description": "只在以下情況調用：
    1. 用戶明確要求查詢內部文件/知識庫
    2. 用戶詢問需要準確事實的問題
    不要對一般性問題或主觀意見調用",
  "parameters": {...}
}
```

#### 挑戰 2：多層次依賴調用

**問題場景**：
```
用戶: "幫我查詢最新一季蘋果公司的營收，然後發送郵件給 CFO"
```
需要：查詢 → 處理結果 → 發送郵件（依賴關係）

**解決方案**：
```python
class DependencyAwareExecutor:
    """依賴感知執行器"""

    def execute_plan(self, tool_calls: list) -> list:
        """分析並執行有依賴的調用計劃"""

        # 1. 構建依賴圖
        graph = self._build_dependency_graph(tool_calls)

        # 2. 拓撲排序
        execution_order = self._topological_sort(graph)

        # 3. 按序執行
        results = {}
        for tool_call in execution_order:
            # 替換依賴參數
            resolved_params = self._resolve_dependencies(
                tool_call.arguments,
                results
            )

            # 執行
            result = self._execute(tool_call.name, resolved_params)
            results[tool_call.id] = result

        return list(results.values())
```

#### 挑戰 3：Tool 返回結果過大

**問題場景**：
```python
def get_all_company_docs():
    """獲取所有公司文件"""
    return fetch_thousands_of_documents()  # 返回 100MB+
```
後果：Context 溢出或處理緩慢

**解決方案**：
```python
class ResultSizeController:
    """結果大小控制器"""

    MAX_RESULT_SIZE = 50 * 1024  # 50KB
    MAX_ITEMS = 100

    def truncate_result(self, result: dict, tool_name: str) -> dict:
        """智慧截斷結果"""

        # 1. 根據工具類型設定限制
        limits = {
            "get_documents": {"max_items": 10, "max_size": "10KB"},
            "search_database": {"max_items": 50, "max_size": "30KB"},
            "list_files": {"max_items": 100, "max_size": "20KB"}
        }
        limit = limits.get(tool_name, {"max_items": 10, "max_size": "10KB"})

        # 2. 截斷過長結果
        if isinstance(result, list) and len(result) > limit["max_items"]:
            truncated = result[:limit["max_items"]]
            return {
                "data": truncated,
                "_truncated": True,
                "_total_count": len(result),
                "_message": f"僅顯示前 {limit['max_items']} 項，共 {len(result)} 項"
            }

        # 3. 按大小截斷
        result_str = json.dumps(result)
        if len(result_str) > self._parse_size(limit["max_size"]):
            return {
                "data": result_str[:self._parse_size(limit["max_size"])] + "...",
                "_truncated": True,
                "_message": "結果已被截斷"
            }

        return result
```

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

---

## 12. 進階主題：MCP 協議

### 什麼是 MCP？

MCP (Model Context Protocol) 是一個開放標準，讓 AI 模型能夠標準化地與外部系統和工具互動。

### MCP vs 傳統 Function Calling

| 特性 | 傳統 Function Calling | MCP |
|------|----------------------|-----|
| 標準化 | 每個平台不同 | 統一協議 |
| 雙向溝通 | 單向 function 調用 | 雙向對話 |
| 發現機制 | 靜態定義 | 動態發現 |
| 狀態管理 | 無 | 有狀態會話 |

### MCP 架構

```
┌─────────────┐      MCP      ┌─────────────┐
│   LLM/Agent │ ◄─────────────► │  MCP Server │
└─────────────┘               └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             ┌──────────┐     ┌──────────┐     ┌──────────┐
             │  File    │     │   API    │     │ Database │
             │  System  │     │  Gateway │     │  Server  │
             └──────────┘     └──────────┘     └──────────┘
```

### MCP Tools 定義

```json
{
  "name": "mcp-server-filesystem",
  "version": "1.0.0",
  "capabilities": {
    "tools": {
      "read_file": {
        "description": "Read contents of a file",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": {"type": "string"}
          }
        }
      },
      "write_file": {
        "description": "Write content to a file",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
          }
        }
      }
    }
  }
}
```

### MCP 資源 (Resources)

```json
{
  "resources": {
    "uri": "file:///home/user/docs",
    "name": "User Documents",
    "mimeType": "application/directory"
  }
}
```

### MCP Prompt 模板

```json
{
  "prompts": {
    "analyze_code": {
      "description": "Analyze code for issues",
      "arguments": [
        {"name": "language", "required": false},
        {"name": "path", "required": true}
      ]
    }
  }
}
```

---

## 13. A2A 協議 (Agent-to-Agent)

### 什麼是 A2A？

A2A 是一個讓不同 AI Agent 能夠相互溝通和協作的協議。

### A2A vs MCP

| 特性 | MCP | A2A |
|------|-----|-----|
| 層次 | Agent ↔ 工具 | Agent ↔ Agent |
| 用途 | 單一 Agent 擴展能力 | 多 Agent 協作 |
| 通訊 | 結構化 JSON | 訊息對話 |

### A2A 訊息格式

```json
{
  "jsonrpc": "2.0",
  "id": "msg-001",
  "method": "tasks/send",
  "params": {
    "task": {
      "id": "task-123",
      "message": {
        "role": "user",
        "parts": [
          {"type": "text", "text": "請幫我分析這個問題"}
        ]
      },
      "history": []
    }
  }
}
```

### A2A 能力協商

```json
{
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransition": ["input", "working", "output"]
  },
  "skills": [
    {
      "id": "code-review",
      "name": "Code Review",
      "description": "擅長程式碼審查"
    }
  ]
}
```

---

### 使用時機

| 適合 Parallel | 不適合 Parallel |
|---------------|-----------------|
| 多個獨立的 API 呼叫 | 有依賴關係的呼叫 |
| 查詢不同來源的資料 | 需前一結果作為下一輸入 |
| 批量操作 | 順序邏輯 |

### Parallel 深度優化

#### 1. 智慧批次分組

```python
class SmartBatchGrouper:
    """智慧批次分組 - 自動識別可並發的調用"""

    def group_tool_calls(self, tool_calls: list) -> list[list]:
        """將 tool calls 分組為可並發執行的批次"""

        groups = []
        current_group = []

        for call in tool_calls:
            # 檢查是否可加入當前組
            if self._can_parallel(current_group, call):
                current_group.append(call)
            else:
                # 提交當前組，開始新組
                if current_group:
                    groups.append(current_group)
                current_group = [call]

        # 提交最後一組
        if current_group:
            groups.append(current_group)

        return groups

    def _can_parallel(self, current_group: list, new_call: dict) -> bool:
        """判斷是否可以並發執行"""

        if not current_group:
            return True

        # 1. 檢查是否有依賴關係
        for existing in current_group:
            if self._has_dependency(existing, new_call):
                return False

        # 2. 檢查資源限制
        total_cost = sum(self._estimate_cost(c) for c in current_group)
        if total_cost + self._estimate_cost(new_call) > self.max_cost:
            return False

        return True
```

#### 2. 失敗容忍機制

```python
class FaultTolerantParallelExecutor:
    """容錯並發執行器"""

    def execute_parallel(self, tool_calls: list, max_concurrent: int = 5):
        """並發執行，容忍部分失敗"""

        results = []
        failed_calls = []

        # 使用信號量限制並發數
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_error_handling(call):
            try:
                async with semaphore:
                    result = await self._execute_async(call)
                    return {"success": True, "call": call, "result": result}
            except Exception as e:
                return {"success": False, "call": call, "error": str(e)}

        # 執行所有調用
        task_results = await asyncio.gather(
            *[execute_with_error_handling(tc) for tc in tool_calls],
            return_exceptions=True
        )

        # 分離成功和失敗
        for r in task_results:
            if isinstance(r, dict) and r.get("success"):
                results.append(r["result"])
            else:
                failed_calls.append(r.get("call") if isinstance(r, dict) else r)

        # 重試失敗的調用（串行）
        for call in failed_calls:
            try:
                result = await self._execute_async(call)
                results.append(result)
            except:
                results.append({"error": f"調用失敗: {call}"})

        return results
```

---

## 14. 進階實作：Tool Runtime

### 動態 Tool 註冊

```python
class DynamicToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, func: callable, schema: dict):
        """動態註冊 Tool"""
        self.tools[name] = {
            "func": func,
            "schema": schema,
            "usage_count": 0
        }

    def unregister(self, name: str):
        """動態移除 Tool"""
        if name in self.tools:
            del self.tools[name]

    def get_tools(self):
        """取得所有可用 Tools"""
        return [
            {"name": k, "schema": v["schema"]}
            for k, v in self.tools.items()
        ]
```

### Tool 版本管理

```python
class VersionedTool:
    def __init__(self, name: str):
        self.name = name
        self.versions = {}  # version -> (func, schema)

    def add_version(self, version: str, func, schema):
        self.versions[version] = (func, schema)

    def call(self, version: str, *args, **kwargs):
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        func, _ = self.versions[version]
        return func(*args, **kwargs)
```

---

## 15. 進階安全：Tool 審計

### 呼叫日誌

```python
import logging
from datetime import datetime

class ToolAuditLogger:
    def __init__(self, log_file: str):
        self.logger = logging.getLogger("tool_audit")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        self.logger.addHandler(handler)

    def log_call(self, tool_name: str, args: dict, user: str, result: dict):
        """記錄 Tool 呼叫"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "args": args,
            "user": user,
            "success": "error" not in result
        }
        self.logger.info(json.dumps(entry))
```

### 異常行為偵測

```python
class ToolAnomalyDetector:
    def __init__(self, threshold: int = 100):
        self.call_history = {}  # tool_name -> [timestamps]
        self.threshold = threshold

    def detect(self, tool_name: str) -> bool:
        """偵測異常呼叫"""
        now = time.time()
        # 記錄這次呼叫
        if tool_name not in self.call_history:
            self.call_history[tool_name] = []
        self.call_history[tool_name].append(now)

        # 清除 1 分鐘前的記錄
        self.call_history[tool_name] = [
            t for t in self.call_history[tool_name]
            if now - t < 60
        ]

        # 檢查是否超過閾值
        return len(self.call_history[tool_name]) > self.threshold
```

---

## 16. Tool 效能優化

### 批量 Tool 呼叫

```python
async def batch_tool_calls(tools: list, max_concurrent: int = 5):
    """批量執行 Tool 呼叫"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def call_with_limit(tool):
        async with semaphore:
            return await tool()

    results = await asyncio.gather(*[call_with_limit(t) for t in tools])
    return results
```

### Tool 結果快取

```python
from functools import lru_cache

class CachedTool:
    def __init__(self, func: callable, ttl: int = 300):
        self.func = func
        self.ttl = ttl
        self.cache = {}

    def _make_key(self, args, kwargs):
        return hash((str(args), str(sorted(kwargs.items()))))

    def __call__(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        now = time.time()

        if key in self.cache:
            result, timestamp = self.cache[key]
            if now - timestamp < self.ttl:
                return result

        result = self.func(*args, **kwargs)
        self.cache[key] = (result, now)
        return result
```