# LangChain

開源的 LLM 應用開發框架，提供組件化和鏈式調用的方式構建 AI 應用。

---

## 1. 什麼是？

### 深度定義

**LangChain** 是一個**模組化的 AI 應用開發框架**，其核心設計理念：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LangChain 架構設計                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  目標: 讓 AI 應用開發像搭積木一樣簡單                                  │
│                                                                      │
│  核心抽象:                                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Component (組件): 可重用的功能單元                              │  │
│  │  - Model: LLM 接口                                             │  │
│  │  - Prompt: 提示詞管理                                          │  │
│  │  - Tool: 外部工具                                              │  │
│  │  - Memory: 記憶系統                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Chain (鏈): 組件的有序組合                                     │  │
│  │  - LLMChain: 簡單的 prompt → LLM → output                     │  │
│  │  - RouterChain: 動態路由                                       │  │
│  │  - SequentialChain: 順序執行                                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Agent (代理): 動態決策 + 工具使用                              │  │
│  │  - ReAct Agent: 推理 + 行動                                     │  │
│  │  - Tool Agent: 工具調用                                         │  │
│  │  - Conversational Agent: 對話記憶                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 簡單範例

```
傳統開發:
  1. 調用 OpenAI API
  2. 處理回應
  3. 顯示結果

LangChain:
  LLM → Tools → Agents → Memory → Chains
  → 可組合的 AI 應用構建塊
```

---

## 2. 核心概念

### LCEL (LangChain Expression Language)

```python
from langchain_core.runnables import RunnableSequence

# 鏈式調用
chain = (
    prompt_template
    | llm
    | output_parser
)

# 等同於:
# prompt -> llm -> parser
result = chain.invoke({"topic": "AI"})
```

### Components

| 組件 | 說明 |
|------|------|
| **Models** | LLM 接口 (OpenAI, Anthropic, 本地模型) |
| **Prompts** | Prompt 模板管理 |
| **Chains** | 組件組合 |
| **Agents** | 動態決策和工具使用 |
| **Memory** | 對話歷史狀態 |
| **Tools** | 外部功能整合 |

---

## 3. 核心組件

### Model I/O

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# 同步調用
response = llm.invoke("Hello!")
print(response.content)

# 串流調用
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

# 簡單模板
prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

# 多元角色模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "Hello {name}, how are you?"),
    ("ai", "Greetings! I'm doing well.")
])
```

### Chains

```python
from langchain.chains import LLMChain

# 簡單 Chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key="joke"
)

result = chain.invoke({"topic": "cats"})
print(result["joke"])

# Sequential Chain
from langchain.chains import SequentialChain

chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="summary")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="analysis")

overall = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["document"],
    output_variables=["summary", "analysis"]
)
```

---

## 4. Agents

### ReAct Agent

```python
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# 定義 Tools
tools = [
    Tool(
        name="Search",
        func=DuckDuckGoSearchRun().run,
        description="搜尋最新資訊"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="數學計算"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 使用
result = agent.run("台北的人口是多少？")
```

### Conversational Agent

```python
from langchain.agents.conversational import ConversationalChatAgent

# 帶 Memory 的 Agent
agent = ConversationalChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    system_message="你是個有用的助手"
)

# 對話
response = agent.invoke("幫我搜尋今天的天氣")
```

---

## 5. Memory

### ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 添加訊息
memory.chat_memory.add_user_message("你好")
memory.chat_memory.add_ai_message("你好！有什麼可以幫你？")

# 檢索
history = memory.load_memory_variables({})
print(history["chat_history"])
```

### ConversationSummaryMemory

```python
from langchain.memory import ConversationSummaryMemory

# 自動摘要長對話
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)
```

### Entity Memory

```python
from langchain.memory import EntityMemory

# 記住實體信息
memory = EntityMemory()

# 儲存實體
memory.save_context(
    {"input": "我住在台北"},
    {"output": "很高興認識你，台北市民！"}
)

# 檢索
entities = memory.load_memory_variables({})
```

---

## 6. Tools

### 自定義 Tool

```python
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 使用現有工具
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

# 自定義工具
def get_weather(city: str) -> str:
    """獲取城市天氣"""
    return f"{city} 現在天氣晴朗，25 度"

weather_tool = Tool(
    name="天氣查詢",
    func=get_weather,
    description="用於獲取特定城市的天氣資訊"
)
```

### Tool 組合

```python
from langchain.agents import AgentExecutor

# 組合多個 Tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)
```

---

## 7. RAG 整合

### 完整 RAG Chain

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa import RetrievalQA

# 1. Load
loader = TextLoader("document.txt")
documents = loader.load()

# 2. Split
splitter = CharacterTextSplitter(chunk_size=1000)
docs = splitter.split_documents(documents)

# 3. Embed & Store
vectorstore = Chroma.from_documents(
    docs,
    OpenAIEmbeddings()
)

# 4. Retriever
retriever = vectorstore.as_retriever()

# 5. QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 使用
result = qa.invoke("這篇文章的主題是什麼？")
```

---

## 8. 部署

### LangServe

```python
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI()

add_routes(
    app,
    chain,
    path="/chain"
)

# 啟動服務
# uvicorn app:app --reload
```

### LangSmith 監控

```python
import os
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# 自動追蹤所有 LangChain 調用
# 可在 LangSmith Dashboard 查看
```

---

## 9. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Function Calling** | LangChain 內建支援 |
| **RAG** | LangChain 核心應用場景 |
| **Agents** | LangChain Agent 系統 |
| **VectorDB** | LangChain 整合多種向量資料庫 |

---

## 延伸閱讀

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangSmith](https://smith.langchain.com/)