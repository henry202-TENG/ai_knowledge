# AutoGen

微軟開源的多代理協作框架，支援構建複雜的 AI Agent 系統。

---

## 1. 什麼是？

### 簡單範例

```
傳統 Agent: 單一 Agent 處理所有任務
  User → Agent → Response

AutoGen: 多個專業 Agent 協作
  User → Planner Agent → Coder Agent → Tester Agent → Response
              ↓
         Reviewer Agent ←
```

---

## 2. 核心架構

### 基礎組件

```python
from autogen import ConversableAgent, AssistantAgent

# 定義 Agent
assistant = ConversableAgent(
    name="assistant",
    llm_config={
        "model": "gpt-4",
        "api_key": "your-key"
    },
    system_message="你是一個專業的 AI 助手"
)

user = ConversableAgent(
    name="user",
    llm_config=False  # 用戶代理
)

# 開始對話
result = user.initiate_chat(
    assistant,
    message="幫我寫一個排序算法"
)
```

### Agent 類型

```python
# 1. AssistantAgent - AI 代理
coder = AssistantAgent(
    name="coder",
    llm_config={"model": "gpt-4"},
    system_message="你是一個專業的程式設計師"
)

# 2. UserProxyAgent - 用戶代理
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

# 3. GroupChat - 群組聊天
group_chat = GroupChat(
    agents=[coder, reviewer, tester],
    messages=[],
    max_round=10
)
```

---

## 3. 多代理協作

### 序列執行

```python
# 序列執行: Planner → Coder → Reviewer
planner = AssistantAgent(
    name="planner",
    system_message="你負責制定計劃"
)

coder = AssistantAgent(
    name="coder",
    system_message="你負責實現代碼"
)

reviewer = AssistantAgent(
    name="reviewer",
    system_message="你負責審查代碼"
)

# 序列執行
planner.initiate_chat(
    coder,
    message="實現一個快速排序算法"
)

coder.initiate_chat(
    reviewer,
    message="審查剛才實現的代碼"
)
```

### 群組聊天

```python
from autogen import GroupChat, GroupChatManager

# 定義參與者
agents = [
    AssistantAgent(name="pm", system_message="產品經理"),
    AssistantAgent(name="dev", system_message="開發工程師"),
    AssistantAgent(name="qa", system_message="測試工程師")
]

# 創建群組
group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=5,
    speaker_selection_method="round_robin"
)

# 創建管理器
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "gpt-4"}
)

# 啟動群組討論
user_proxy.initiate_chat(
    manager,
    message="我們需要開發一個電商網站"
)
```

### 自定義 Speaker 選擇

```python
def custom_speaker_selection(last_speaker, agents, messages):
    """自定義發言人選擇邏輯"""

    last_message = messages[-1]["content"]

    # 根據最後一條消息決定下一個發言人
    if "代碼" in last_message or "實現" in last_message:
        # 程式碼相關 → 交給測試
        return agents[2]  # tester
    elif "bug" in last_message or "錯誤" in last_message:
        # 測試結果 → 交給開發
        return agents[1]  # dev
    else:
        # 默認輪詢
        current_idx = agents.index(last_speaker)
        return agents[(current_idx + 1) % len(agents)]

group_chat = GroupChat(
    agents=agents,
    speaker_selection_method=custom_speaker_selection
)
```

---

## 4. 工具整合

### 定義工具

```python
from autogen import tool

@tool
def search_codebase(query: str) -> str:
    """搜索代碼庫中的相關代碼"""
    # 實現搜索邏輯
    results = search(query)
    return format_results(results)

@tool
def execute_code(code: str) -> str:
    """執行 Python 代碼並返回結果"""
    result = exec(code)
    return str(result)

@tool
def read_file(path: str) -> str:
    """讀取文件內容"""
    with open(path) as f:
        return f.read()

# 註冊工具
coder = AssistantAgent(
    name="coder",
    tool_names=["search_codebase", "execute_code", "read_file"]
)
```

### 工具選擇策略

```python
# 自定義工具選擇
def select_tool(last_message, available_tools):
    """根據上下文選擇合適的工具"""

    if "搜索" in last_message or "找" in last_message:
        return "search_codebase"
    elif "執行" in last_message or "運行" in last_message:
        return "execute_code"
    elif "讀取" in last_message or "查看" in last_message:
        return "read_file"

    return None
```

---

## 5. 工作流程模式

### 雙方協作

```python
# Developer - Reviewer 模式
developer = AssistantAgent(
    name="developer",
    system_message="""你是專業開發者。
    1. 根據需求實現代碼
    2. 確保代碼可運行
    3. 寫好測試"""
)

reviewer = AssistantAgent(
    name="reviewer",
    system_message="""你是代碼審查者。
    1. 檢查代碼質量
    2. 找出潛在 bug
    3. 提出改進建議"""
)

# 迴圈直到滿意
user_proxy = UserProxyAgent(human_input_mode="TERMINATE")

chat_result = user_proxy.initiate_chat(
    developer,
    message="實現一個 LRU Cache",
    max_consecutive_auto_reply=5
)
```

### 多階段開發

```python
# 軟體開發生命週期
agents = {
    "pm": AssistantAgent(name="pm", system_message="產品經理"),
    "arch": AssistantAgent(name="architect", system_message="系統架構師"),
    "dev": AssistantAgent(name="developer", system_message="開發工程師"),
    "tester": AssistantAgent(name="tester", system_message="測試工程師"),
    "reviewer": AssistantAgent(name="reviewer", system_message="技術評審")
}

# 階段 1: PM 產生需求
agents["pm"].initiate_chat(
    agents["arch"],
    message="設計一個電商系統"
)

# 階段 2: 架構師設計
agents["arch"].initiate_chat(
    agents["dev"],
    message="基於以下架構實現"
)

# 階段 3: 開發實現...
```

---

## 6. 高級功能

### 對話記憶

```python
from autogen import ChatCompletion

# 配置對話記憶
assistant = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    system_message="你記住之前的對話",
    chat_messages={
        "user": [...],  # 用戶歷史
        "assistant": [...]  # 助手歷史
    }
)
```

### 執行控制

```python
# 控制執行流程
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 完全自動
    max_consecutive_auto_reply=3,  # 最多自動回覆次數
    is_termination_msg=lambda x: "完成" in x.get("content", ""),
)
```

### 錯誤處理

```python
@assistant.register_for_execution()
def safe_execute(code: str):
    """安全的代碼執行"""
    try:
        result = execute(code)
        return result
    except Exception as e:
        return f"錯誤: {str(e)}"
```

---

## 7. 應用場景

### 程式碼生成

```python
# 完整的代碼生成工作流
task_manager = ConversableAgent(name="manager")

dev_agent = AssistantAgent(
    name="dev",
    system_message="""你是一個 Python 專家。
    1. 實現高質量的代碼
    2. 包含錯誤處理
    3. 寫好文檔字符串"""
)

test_agent = AssistantAgent(
    name="tester",
    system_message="""你為代碼編寫測試。
    1. 覆蓋主要場景
    2. 包含邊界情況"""
)

# 執行
task_manager.initiate_chat(
    dev_agent,
    message="實現一個 HTTP 客戶端"
)
```

### 數據分析

```python
analyst = AssistantAgent(
    name="analyst",
    system_message="""你擅長數據分析。
    使用 Python 進行數據處理和可視化"""
)

visualizer = AssistantAgent(
    name="visualizer",
    system_message="""你創建數據可視化。
    使用 matplotlib/seaborn 創建圖表"""
)

# 分析任務
user_proxy.initiate_chat(
    analyst,
    message="分析銷售數據，找出趨勢"
)
```

---

## 8. 最佳實踐

### Agent 設計原則

```
1. 單一職責 - 每個 Agent 專注特定任務
2. 清晰角色 - 明確每個 Agent 的職責
3. 適當數量 - 避免過多 Agent 導致複雜
4. 有限上下文 - 避免上下文溢出
5. 明確終止 - 設定終止條件
```

### 調試技巧

```python
# 啟用詳細日誌
import autogen

autogen.ChatCompletion.start_logging()

# 檢查消息歷史
for msg in chat_result.chat_history:
    print(f"{msg['name']}: {msg['content'][:100]}")
```

---

## 9. 與相關技術

| 技術 | 關係 |
|------|------|
| **LangChain** | 類似的 Agent 框架 |
| **Function Calling** | AutoGen 支持工具調用 |
| **ReAct** | AutoGen 內置支持 |

---

## 延伸閱讀

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AutoGen GitHub](https://github.com/microsoft/autogen)
- [AutoGen Examples](https://github.com/microsoft/autogen/tree/main/notebook)