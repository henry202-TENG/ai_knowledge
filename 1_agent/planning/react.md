# ReAct (Reasoning + Acting)

## 1. 什麼是？
ReAct 是一種結合推理（Reasoning）與行動（Acting）的 AI  Agent 框架，讓模型在解決問題時能夠交替進行思考與執行動作。

## 2. 為什麼重要？
- **解決 LLM 幻覺問題**：透過與環境互動獲取真實資訊
- **增強推理能力**：讓模型能夠進行多步驟推理
- **實現真正的主動性**：AI 能夠主動查詢資訊而非被動回應

## 3. 核心原理

### 傳統 LLM 的限制
```
輸入: "今天的氣溫是多少？"
輸出: "今天氣溫 25 度" ← 可能幻覺

LLM 不知道實時資訊，會胡亂猜測
```

### ReAct 流程
```
用戶: "今天台北的天氣如何？"

思考 (Thought): "我需要查詢台北的天氣資訊"
行動 (Action): 調用天氣 API
觀察 (Observation): "台北今天晴朗，28 度"

思考 (Thought): "現在我可以回答用戶了"
行動 (Action): 輸出最終答案
```

### 公式化
```
ReAct Prompt = Thought + Action + Observation 循環
直到達成目標或達到最大迭代次數
```

## 4. 實現方案

### 經典 ReAct Prompt 模板
```
請根據以下格式回答：
Thought: [你的思考過程]
Action: [要執行的行動，如調用工具]
Observation: [行動的結果]
...
```

### 與 Function Calling 結合
- ReAct 提供推理框架
- Function Calling 提供具體工具調用能力
- 兩者結合實現完整的 Agent 行為

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **Function Calling** | ReAct 的工具調用實現方式 |
| **Tree of Thoughts** | ReAct 的進階版本，多路徑推理 |
| **Chain of Thought** | ReAct 的前身，純推理無行動 |
| **Agent** | ReAct 是 Agent 的核心機制之一 |

## 6. 延伸閱讀
- [ReAct Paper (arxiv)](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [ReAct vs Chain of Thought](https://arxiv.org/abs/2210.03629)

---

*待補充...*