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

## 7. 數學形式化

### ReAct 形式化定義

```
Given: 任務查詢 Q，工具集 T = {t₁, t₂, ..., tₙ}

ReAct 目標: 找到序列 A = [a₁, a₂, ..., aₖ] 使得:
  f(Q, execute(A)) = 最優答案

其中每個 action aᵢ:
  aᵢ = (thoughtᵢ, action_typeᵢ, action_inputᵢ)
  
觀察結果:
  observationᵢ = execute(action_inputᵢ)
```

### 推理狀態機

```
┌──────────┐     tool_call      ┌──────────┐
│  THINK   │ ─────────────────► │  ACTION  │
└──────────┘                    └──────────┘
     ▲                                │
     │                                ▼
     │                         ┌──────────┐
     │                         │ OBSERVE  │
     └──────────────────────── └──────────┘
              observation
```

### 收斂條件

| 條件 | 數學表達 | 說明 |
|------|----------|------|
| 任務完成 | `∃i: is_complete(thoughtᵢ)` | 找到答案 |
| 最大迭代 | `i ≥ max_iterations` | 避免無限循環 |
| 無進展 | `progress_count ≥ 3` | 連續無新資訊 |
| 失敗標記 | `contains(thoughtᵢ, "FAIL")` | 明確失敗 |

---

## 8. ReAct vs ReWOO

### ReWOO (Reasoning Without Observation Overhead)

ReWOO 是一種更高效的變體，分離推理與觀察：

```
ReAct:
  Thought → Action → Observation → Thought → Action → Observation...
  (每次都需要 Observation 才能下一步)

ReWOO:
  Plan: [Action₁, Action₂, Action₃]
  ────────────────────────────────
  Execution: 一次執行所有 Action
  ────────────────────────────────
  Reasoning: 根據結果統一推理
```

### 效能比較

| 指標 | ReAct | ReWOO |
|------|-------|-------|
| LLM 調用次數 | N | 2-3 |
| Token 消耗 | 高 | 低 |
| 依賴處理 | 強 | 弱 |
| 複雜任務 | 適合 | 不適合 |

### ReWOO 實作

```python
def rewoo_plan_and_execute(query, tools):
    """
    ReWOO: 先規劃再執行
    """
    client = OpenAI()

    # Step 1: 產生計劃
    plan_prompt = f"""Given: {query}
可用工具: {list(tools.keys())}

產生一個行動計劃，列出需要執行的工具順序。
只輸出行動，不要執行。"""

    plan_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": plan_prompt}]
    )

    # Step 2: 解析計劃
    actions = parse_plan(plan_response.content)

    # Step 3: 執行所有 action
    observations = []
    for action in actions:
        result = execute_tool(action, tools)
        observations.append(result)

    # Step 4: 根據觀察生成答案
    final_prompt = f"""查詢: {query}
觀察結果: {observations}

請根據觀察結果回答。"""

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return final_response.content
```

---

## 9. 生產環境實踐

### Token 消耗優化

```python
class ReActOptimizer:
    def __init__(self, max_tokens_per_iteration=500):
        self.max_tokens = max_tokens_per_iteration

    def should_truncate_history(self, messages):
        """檢查是否需要截斷歷史"""
        total_tokens = estimate_tokens(messages)
        if total_tokens > self.max_tokens * 10:
            # 保留關鍵觀察，壓縮歷史
            return self._compress_history(messages)
        return False

    def _compress_history(self, messages):
        """壓縮對話歷史"""
        # 只保留最後 N 輪的完整記錄
        # 更早的回合壓縮成摘要
        recent = messages[-6:]  # 最近 3 回合
        summary = self._summarize(messages[:-6])
        return [summary] + recent
```

### 成本追蹤

```python
class CostTracker:
    def __init__(self):
        self.costs = []
        self.action_counts = {}

    def track(self, action_type, tokens_used, latency_ms):
        self.costs.append({
            "action": action_type,
            "tokens": tokens_used,
            "latency": latency_ms,
            "timestamp": datetime.now()
        })

    def get_summary(self):
        return {
            "total_tokens": sum(c["tokens"] for c in self.costs),
            "total_cost": sum(c["tokens"] for c in self.costs) * 0.001,  # 假設 $0.001/token
            "avg_latency": np.mean([c["latency"] for c in self.costs]),
            "action_distribution": Counter(c["action"] for c in self.costs)
        }
```

### 監控指標

| 指標 | 目標 | 告警閾值 |
|------|------|----------|
| 成功率 | > 95% | < 90% |
| 平均迭代次數 | < 5 | > 10 |
| Token 消耗 | < 2000 | > 5000 |
| 延遲 (P99) | < 30s | > 60s |

---

## 10. 調試與監控

### 追蹤格式

```python
import json
from datetime import datetime

class ReActTracer:
    def __init__(self):
        self.traces = []

    def trace_step(self, step_num, thought, action, observation):
        entry = {
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "action": action,
            "observation": observation,
            "thought_token_count": len(thought.split()),
            "observation_size": len(str(observation))
        }
        self.traces.append(entry)

    def export(self):
        return json.dumps(self.traces, indent=2, ensure_ascii=False)

    def get_bottlenecks(self):
        """分析效能瓶頸"""
        if not self.traces:
            return []

        times = [(t["step"], t.get("duration_ms", 0)) for t in self.traces]
        return sorted(times, key=lambda x: x[1], reverse=True)[:3]
```

### 常見問題診斷

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| 無限循環 | 停止條件不當 | 加入 max_iterations |
| 錯誤工具 | 工具描述不清 | 改進 description |
| 幻覺參數 | 參數提取失敗 | 加入 validate 步驟 |
| 觀察忽略 | 未正確使用結果 | 加入 reflection 步驟 |

### 診斷 Prompt

```python
DIAGNOSTIC_PROMPT = """診斷以下 ReAct 軌跡：

{trace}

請分析：
1. 是否有思維跳躍或邏輯錯誤？
2. 是否有工具調用不當？
3. 是否正確使用了觀察結果？
4. 是否有改進空間？

輸出具體建議。"""
```

---

## 11. 變體比較詳解

### 各變體適用場景

| 變體 | 適用場景 | 不適用場景 |
|------|----------|------------|
| **ReAct** | 需要真實世界資訊的任務 | 純數學推理 |
| **Chain of Thought** | 數學、邏輯推理 | 需要外部資料 |
| **Self-Ask** | 複雜多步問題 | 簡單查詢 |
| **Act-Then-Think** | 批量操作為主的任務 | 順序依賴強的任務 |
| **ReWOO** | 獨立工具調用 | 有依賴關係的任務 |

### 混合策略

```python
def hybrid_reasoning(query, tools):
    """根據查詢類型自動選擇策略"""

    # 策略選擇邏輯
    if needs_external_data(query):
        return react_reasoning(query, tools)
    elif is_mathematical(query):
        return cot_reasoning(query)
    elif is_multi_step_complex(query):
        return self_ask_reasoning(query, tools)
    else:
        return simple_reasoning(query)
```

---

## 12. 相關主題

- Function Calling - 工具調用技術
- Tree of Thoughts - 多路徑推理
- Chain of Thought - 純推理無行動
- Agent - ReAct 是核心機制

---

## 延伸閱讀

- [ReAct Paper (arxiv)](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [ReAct vs Chain of Thought](https://arxiv.org/abs/2210.03629)
- [ReWOO Paper](https://arxiv.org/abs/2305.18323)
- [Reflexion Agent](https://arxiv.org/abs/2303.11366)