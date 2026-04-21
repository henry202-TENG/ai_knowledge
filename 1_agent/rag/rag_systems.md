# RAG Systems

檢索增強生成系統的完整實作，結合外部知識庫提升 LLM 能力。

---

## 1. 什麼是？

### 簡單範例

```
沒有 RAG:
  用戶: 2024 年蘋果發布會說了什麼？
  AI: (不知道，因為知識截止)

有 RAG:
  用戶: 2024 年蘋果發布會說了什麼？
  AI → 檢索最新資訊 → 生成回答
```

---

## 2. 核心架構

### 完整流程

```python
class RAGSystem:
    """RAG 系統"""

    def __init__(self, llm, retriever, chunker):
        self.llm = llm
        self.retriever = retriever
        self.chunker = chunker

    def answer(self, question):
        """回答問題"""

        # 1. 檢索
        relevant_docs = self.retriever.retrieve(question)

        # 2. 處理上下文
        context = self._build_context(relevant_docs)

        # 3. 生成回答
        prompt = f"""基於以下上下文回答問題:

上下文:
{context}

問題: {question}

回答:"""

        answer = self.llm.generate(prompt)

        # 4. 添加引用
        citations = self._extract_citations(
            relevant_docs,
            answer
        )

        return {
            "answer": answer,
            "citations": citations,
            "sources": relevant_docs
        }
```

### 索引構建

```python
class RAGIndexer:
    """RAG 索引器"""

    def __init__(self, vectorstore, embedder):
        self.store = vectorstore
        self.embedder = embedder

    def index_documents(self, documents):
        """索引文檔"""

        for doc in documents:
            # 分塊
            chunks = self.chunker.chunk(doc)

            # 嵌入
            embeddings = self.embedder.embed(chunks)

            # 存儲
            self.store.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{
                    "source": doc["source"],
                    "title": doc["title"]
                }] * len(chunks)
            )

        print(f"已索引 {len(documents)} 個文檔")
```

---

## 3. 檢索優化

### 混合檢索

```python
class HybridRetriever:
    """混合檢索"""

    def __init__(self):
        self.semantic = SemanticRetriever()
        self.keyword = BM25Retriever()

    def retrieve(self, query, top_k=5):
        """混合檢索"""

        # 語意檢索
        semantic_results = self.semantic.retrieve(query)

        # 關鍵詞檢索
        keyword_results = self.keyword.retrieve(query)

        # RRF 融合
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            top_k
        )

        return combined

    def _reciprocal_rank_fusion(self, results_a, results_b, k=60):
        """RRF 融合"""

        scores = {}

        # 處理結果 A
        for rank, doc in enumerate(results_a):
            score = 1.0 / (k + rank)
            scores[doc.id] = scores.get(doc.id, 0) + score

        # 處理結果 B
        for rank, doc in enumerate(results_b):
            score = 1.0 / (k + rank)
            scores[doc.id] = scores.get(doc.id, 0) + score

        # 排序
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc_id, _ in sorted_docs[:top_k]]
```

### 查詢擴展

```python
class QueryExpansion:
    """查詢擴展"""

    def __init__(self, llm):
        self.llm = llm

    def expand(self, query):
        """擴展查詢"""

        # 生成相關術語
        prompt = f"""為以下查詢生成相關術語:

原始查詢: {query}

生成 3-5 個相關的檢索術語:"""

        terms = self.llm.generate(prompt)

        # 組合
        expanded = [query] + terms.split("\n")

        return expanded

    def expand_with_synonyms(self, query):
        """同義詞擴展"""

        synonyms = {
            "電腦": ["電腦", "計算機", "computer"],
            "網路": ["網路", "互聯網", "internet"],
            "軟體": ["軟體", "software", "程式"]
        }

        expanded = [query]

        for word, syns in synonyms.items():
            if word in query:
                expanded.extend(syns)

        return expanded
```

---

## 4. 生成優化

### 上下文選擇

```python
class ContextSelector:
    """上下文選擇"""

    def __init__(self, llm):
        self.llm = llm

    def select(self, question, retrieved_docs, max_tokens=4000):
        """選擇相關上下文"""

        selected = []
        current_tokens = 0

        # 按相關性排序
        sorted_docs = sorted(
            retrieved_docs,
            key=lambda x: x["score"],
            reverse=True
        )

        for doc in sorted_docs:
            doc_tokens = self._count_tokens(doc["content"])

            if current_tokens + doc_tokens > max_tokens:
                break

            selected.append(doc)
            current_tokens += doc_tokens

        return selected

    def _count_tokens(self, text):
        """計算 tokens"""
        return len(text.split()) * 1.3  # 估算
```

### Prompt 優化

```python
class RAGPromptOptimizer:
    """RAG Prompt 優化器"""

    @staticmethod
    def basic_template():
        """基礎模板"""

        return """基於以下上下文回答問題。如果無法從上下文找到答案，請如實說明。

上下文:
{context}

問題: {question}

回答:"""

    @staticmethod
    def with_citation_template():
        """引用模板"""

        return """根據提供的上下文回答問題。必須引用來源。

上下文:
{context}

問題: {question}

要求:
1. 只使用提供的上下文
2. 標註來源 [來源]
3. 如無法確定，請說「根據現有資訊無法確定」

回答:"""

    @staticmethod
    def chain_of_verification():
        """驗證鏈"""

        return """基於上下文回答問題，並驗證答案的準確性。

步驟:
1. 從上下文找到相關資訊
2. 基於資訊生成回答
3. 驗證回答是否與上下文一致

上下文:
{context}

問題: {question}

回答:"""
```

---

## 5. 評估優化

### RAG 評估指標

```python
class RAGEvaluator:
    """RAG 評估"""

    def evaluate(self, question, answer, retrieved_docs, ground_truth):
        """評估 RAG"""

        return {
            "retrieval_precision": self._precision(
                retrieved_docs,
                ground_truth
            ),
            "retrieval_recall": self._recall(
                retrieved_docs,
                ground_truth
            ),
            "answer_accuracy": self._accuracy(
                answer,
                ground_truth
            ),
            "context_relevance": self._context_relevance(
                question,
                retrieved_docs
            )
        }

    def _precision(self, retrieved, relevant):
        """檢索精確度"""
        retrieved_ids = set(d["id"] for d in retrieved)
        relevant_ids = set(r["id"] for r in relevant)

        return len(retrieved_ids & relevant_ids) / max(len(retrieved_ids), 1)

    def _recall(self, retrieved, relevant):
        """檢索召回率"""
        retrieved_ids = set(d["id"] for d in retrieved)
        relevant_ids = set(r["id"] for r in relevant)

        return len(retrieved_ids & relevant_ids) / max(len(relevant_ids), 1)
```

---

## 6. 持續優化

### 反饋循環

```python
class RAGFeedbackLoop:
    """RAG 反饋循環"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.feedback_data = []

    def record_interaction(self, question, answer, feedback):
        """記錄交互"""

        self.feedback_data.append({
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "timestamp": time.time()
        })

    def identify_gaps(self):
        """識別差距"""

        # 找出低質量案例
        low_quality = [
            f for f in self.feedback_data
            if f["feedback"] == "unsatisfactory"
        ]

        # 分析原因
        gaps = {
            "retrieval": 0,
            "generation": 0,
            "context": 0
        }

        for item in low_quality:
            reason = self._analyze_failure(item)
            gaps[reason] += 1

        return gaps

    def optimize(self):
        """優化 RAG"""
        # 根據反饋優化
        # - 調整 chunk size
        # - 調整 top_k
        # - 改進 prompt
        pass
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **VectorDB** | 存儲向量 |
| **Advanced RAG** | 高級 RAG 技術 |
| **LangChain** | RAG 框架 |

---

## 延伸閱讀

- [RAG Survey](https://arxiv.org/abs/2312.10997)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering)
- [LlamaIndex](https://www.llamaindex.ai/)