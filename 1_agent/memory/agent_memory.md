# Agent Memory Systems

Agent 記憶系統的設計與實現，支援長期知識保持和上下文管理。

---

## 1. 什麼是？

### 簡單範例

```
無記憶 Agent:
  每次對話都是新的開始
  "你好" → "你好" → "你是誰" → "不知道"

有記憶 Agent:
  記住對話歷史
  "你好" → 記住 → "我叫 AI" → 回憶起來 → "記得你叫 AI"
```

---

## 2. 記憶類型

### 短期記憶

```python
class ShortTermMemory:
    """短期記憶 - 當前對話"""

    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.buffer = []

    def add(self, role, content):
        """添加訊息"""

        self.buffer.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

        # 超過上限，移除最早的
        if len(self.buffer) > self.max_turns:
            self.buffer.pop(0)

    def get_context(self):
        """獲取上下文"""

        return self.buffer[-self.max_turns:]

    def clear(self):
        """清除記憶"""
        self.buffer = []
```

### 長期記憶

```python
class LongTermMemory:
    """長期記憶 - 持久化存儲"""

    def __init__(self, vectorstore):
        self.store = vectorstore

    def add(self, content, metadata=None):
        """存儲記憶"""

        embedding = self.embedder.embed(content)

        self.store.add(
            vectors=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )

    def search(self, query, top_k=5):
        """檢索記憶"""

        query_embedding = self.embedder.embed(query)

        results = self.store.similarity_search(
            query_embedding,
            k=top_k
        )

        return results
```

### 工作記憶

```python
class WorkingMemory:
    """工作記憶 - 當前任務相關"""

    def __init__(self):
        self.current_task = None
        self.relevant_info = []
        self.intermediate_results = []

    def set_task(self, task):
        """設置當前任務"""
        self.current_task = task
        self.relevant_info = []
        self.intermediate_results = []

    def add_relevant(self, info):
        """添加相關資訊"""
        self.relevant_info.append(info)

    def add_result(self, step, result):
        """添加中間結果"""
        self.intermediate_results.append({
            "step": step,
            "result": result,
            "timestamp": time.time()
        })

    def get_summary(self):
        """獲取任務摘要"""
        return {
            "task": self.current_task,
            "steps": len(self.intermediate_results),
            "info_count": len(self.relevant_info)
        }
```

---

## 3. 記憶架構

### 分層記憶

```python
class LayeredMemory:
    """分層記憶架構"""

    def __init__(self):
        # 傳感記憶 - 原始輸入
        self.sensory = []

        # 工作記憶 - 短期
        self.working = WorkingMemory()

        # 情節記憶 - 事件序列
        self.episodic = EpisodicMemory()

        # 語義記憶 - 知識
        self.semantic = SemanticMemory()

    def process(self, input_text):
        """處理輸入"""

        # 1. 感官記憶
        self.sensory.append(input_text)

        # 2. 檢索相關記憶
        relevant = self.episodic.retrieve(input_text)

        # 3. 更新工作記憶
        self.working.add_relevant(relevant)

        # 4. 決策輸出
        return self._generate_response()

    def store_episode(self, episode):
        """存儲情節"""

        self.episodic.add(episode)
```

### 總結壓縮

```python
class MemorySummarizer:
    """記憶總結壓縮"""

    def __init__(self, llm):
        self.llm = llm

    def summarize_old_memories(self, memories, max_tokens=500):
        """總結舊記憶"""

        if len(memories) <= 5:
            return memories

        # 構建總結 prompt
        prompt = f"""總結以下對話的要點，保留重要資訊:

對話:
{self._format_memories(memories)}

總結:"""

        summary = self.llm.generate(prompt)

        # 保留最近的和總結
        recent = memories[-3:]

        return [{
            "type": "summary",
            "content": summary,
            "source_count": len(memories)
        }] + recent

    def _format_memories(self, memories):
        """格式化記憶"""
        return "\n".join(
            f"- {m['role']}: {m['content'][:100]}..."
            for m in memories
        )
```

---

## 4. 檢索策略

### 主動回憶

```python
class ActiveRecall:
    """主動回憶"""

    def __init__(self, memory_system):
        self.memory = memory_system

    def retrieve_context(self, query):
        """根據查詢檢索上下文"""

        # 1. 精確匹配
        exact = self._exact_match(query)

        # 2. 語意搜索
        semantic = self.memory.long_term.search(query)

        # 3. 時間衰減
        time_weighted = self._apply_time_decay(semantic)

        # 4. 重要性加權
        importance_weighted = self._apply_importance(time_weighted)

        return importance_weighted

    def _apply_time_decay(self, memories):
        """時間衰減"""

        current_time = time.time()
        decay = 0.95  # 每小時

        for mem in memories:
            age_hours = (current_time - mem["timestamp"]) / 3600
            mem["weight"] *= (decay ** age_hours)

        return memories
```

### 重要性評估

```python
class ImportanceScorer:
    """重要性評分"""

    @staticmethod
    def score(memory):
        """評估記憶重要性"""

        factors = {
            "recency": ScoreCalculator.recency(memory),
            "relevance": ScoreCalculator.relevance(memory),
            "emotion": ScoreCalculator.emotion(memory),
            "frequency": ScoreCalculator.frequency(memory)
        }

        weights = {
            "recency": 0.2,
            "relevance": 0.4,
            "emotion": 0.2,
            "frequency": 0.2
        }

        return sum(
            factors[k] * weights[k]
            for k in factors
        )

    @staticmethod
    def should_store(memory, threshold=0.5):
        """判斷是否應該存儲"""
        return ImportanceScorer.score(memory) > threshold
```

---

## 5. 實作框架

### LangChain Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 緩衝記憶
buffer = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 向量記憶
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings()
)

vector_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(k=3),
    memory_key="context",
    return_messages=True
)

# 組合使用
from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=[buffer, vector_memory]
)
```

### 自定義記憶

```python
class CustomAgentMemory:
    """自定義 Agent 記憶"""

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = VectorStoreRetrieverMemory()
        self.working = WorkingMemory()

    def store_interaction(self, user_input, response):
        """存儲交互"""

        # 短期記憶
        self.short_term.add("user", user_input)
        self.short_term.add("assistant", response)

        # 長期記憶 (選擇性)
        if self._is_important(response):
            self.long_term.save_context(
                {"input": user_input},
                {"output": response}
            )

    def retrieve(self, query):
        """檢索記憶"""

        # 工作記憶
        working = self.working.get_summary()

        # 長期記憶
        long_term = self.long_term.search(query)

        # 短期記憶
        short_term = self.short_term.get_context()

        return {
            "working": working,
            "long_term": long_term,
            "short_term": short_term
        }
```

---

## 6. 遗忘機制

### 優先級淘汰

```python
class MemoryEviction:
    """記憶淘汰策略"""

    def __init__(self, max_size=1000):
        self.max_size = max_size

    def evict_if_needed(self, memory_system):
        """需要時淘汰"""

        size = memory_system.size()

        if size > self.max_size:
            # 計算優先級
            priorities = []

            for mem in memory_system.all():
                priority = ImportanceScorer.score(mem)
                priorities.append((mem, priority))

            # 淘汰最低優先級
            priorities.sort(key=lambda x: x[1])

            to_evict = size - self.max_size

            for mem, _ in priorities[:to_evict]:
                memory_system.remove(mem)
```

### 時間基礎遗忘

```python
class TimeBasedDecay:
    """時間衰減遗忘"""

    def __init__(self):
        self.halflife = 24 * 3600  # 24 小時

    def decay(self, memory):
        """計算衰減"""

        age = time.time() - memory["timestamp"]

        remaining = 0.5 ** (age / self.halflife)

        if remaining < 0.1:
            return "forget"

        return remaining
```

---

## 7. 監控與分析

### 記憶指標

```python
class MemoryMetrics:
    """記憶指標"""

    def __init__(self):
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def record_store(self):
        self.stats["total_stored"] += 1

    def record_retrieve(self, found):
        self.stats["total_retrieved"] += 1

        if found:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1

    def get_hit_rate(self):
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        return self.stats["cache_hits"] / total if total > 0 else 0
```

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **VectorDB** | 長期記憶存儲 |
| **LangChain** | 提供記憶組件 |
| **RAG** | 記憶檢索 |

---

## 延伸閱讀

- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [Agent Memory Survey](https://arxiv.org/abs/2309.07867)
- [MemGPT](https://github.com/facebookresearch/MemGPT)