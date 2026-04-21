# Advanced RAG

高級檢索增強生成技術，整合多種檢索策略和優化方法以提升問答質量。

---

## 1. 什麼是？

### 簡單範例

```
基礎 RAG:
  用戶問題 → 簡單關鍵詞檢索 → 回覆

高級 RAG:
  用戶問題 → 意圖分析 → 多路檢索 → 重排 → 生成回覆
             ↓
           查詢擴展/壓縮 → 更精確的檢索
```

---

## 2. 查詢處理

### 查詢擴展

```python
class QueryExpander:
    """查詢擴展 - 增加檢索關鍵詞"""

    def __init__(self, llm):
        self.llm = llm

    def expand(self, query):
        """生成擴展查詢"""

        prompt = f"""
        為以下查詢生成 3 個相關的檢索關鍵詞/短語:

        原始查詢: {query}

        要求:
        - 包含同義詞
        - 包含上位詞/下位詞
        - 保留原始查詢

        輸出格式:
        1. [關鍵詞1]
        2. [關鍵詞2]
        3. [關鍵詞3]
        """

        expansion = self.llm.generate(prompt)
        keywords = self._parse(expansion)

        # 原始 + 擴展
        return [query] + keywords

    def _parse(self, response):
        """解析關鍵詞"""
        lines = response.strip().split("\n")
        return [line.split(".")[1].strip() for line in lines]
```

### 查詢壓縮

```python
class QueryCompressor:
    """查詢壓縮 - 提取核心意圖"""

    def compress(self, query):
        """壓縮成長度較短的查詢"""

        # 去除冗餘資訊
        stop_words = [
            "請問", "能否麻煩", "不好意思",
            "可以幫我", "謝謝", "感謝"
        ]

        compressed = query
        for word in stop_words:
            compressed = compressed.replace(word, "")

        return compressed.strip()

    def extract_core_intent(self, query):
        """使用 LLM 提取核心意圖"""

        prompt = f"""
        從以下查詢中提取核心資訊需求:

        查詢: {query}

        輸出:
        - 核心主題:
        - 所需資訊類型:
        - 約束條件:
        """

        return self.llm.generate(prompt)
```

### 子查詢分解

```python
class QueryDecomposer:
    """查詢分解 - 拆分成多個子查詢"""

    def decompose(self, query):
        """將複雜問題分解"""

        prompt = f"""
        將以下問題分解成簡單的子問題:

        問題: {query}

        子問題應該:
        1. 每個子問題可以獨立檢索答案
        2. 按邏輯順序排列
        3. 涵蓋原始問題的所有面向
        """

        sub_queries = self.llm.generate(prompt)
        return self._parse(sub_queries)

    def parallel_retrieve(self, query, retriever):
        """並行檢索子查詢"""

        sub_queries = self.decompose(query)
        results = []

        for sq in sub_queries:
            docs = retriever.retrieve(sq)
            results.extend(docs)

        return self._merge_results(results)
```

---

## 3. 檢索策略

### 混合檢索

```python
class HybridRetriever:
    """混合檢索 - 結合多個檢索器"""

    def __init__(self):
        # 語意檢索
        self.semantic_retriever = SemanticRetriever()

        # 關鍵詞檢索
        self.bm25_retriever = BM25Retriever()

        # 權重
        self.weights = {"semantic": 0.7, "bm25": 0.3}

    def retrieve(self, query, top_k=10):
        # 兩種檢索並行
        semantic_results = self.semantic_retriever.retrieve(query)
        bm25_results = self.bm25_retriever.retrieve(query)

        # 合併結果
        merged = self._merge(
            semantic_results,
            bm25_results,
            top_k
        )

        return merged

    def _merge(self, results_a, results_b, top_k):
        """RRF (Reciprocal Rank Fusion) 合併"""

        scores = {}

        for docs, weight in [
            (results_a, self.weights["semantic"]),
            (results_b, self.weights["bm25"])
        ]:
            for rank, doc in enumerate(docs):
                doc_id = doc["id"]
                rrf_score = weight / (rank + 60)  # RRF formula
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # 按分數排序
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc_id, _ in sorted_docs[:top_k]]
```

### 上下文檢索

```python
class ContextualRetriever:
    """上下文感知檢索"""

    def __init__(self, llm):
        self.llm = llm

    def retrieve_with_context(self, query, history):
        """利用對話歷史增強檢索"""

        # 提取歷史關鍵資訊
        context = self._extract_context(history)

        # 構建增強查詢
        enhanced_query = f"""
        對話上下文:
        {context}

        當前問題: {query}

        考慮上下文後的查詢:
        """

        enhanced = self.llm.generate(enhanced_query)

        # 使用增強查詢檢索
        return self.base_retriever.retrieve(enhanced)

    def _extract_context(self, history):
        """提取關鍵上下文"""
        recent = history[-3:]  # 最近 3 輪

        entities = []
        for msg in recent:
            entities.extend(self._extract_entities(msg))

        return ", ".join(set(entities))
```

### 探索性檢索

```python
class ExploratoryRetriever:
    """探索性檢索 - 迭代檢索"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def retrieve_iterative(self, query, max_iterations=3):
        """迭代檢索直到收集足夠資訊"""

        all_docs = []
        current_query = query

        for i in range(max_iterations):
            # 檢索
            docs = self.retriever.retrieve(current_query)
            all_docs.extend(docs)

            # 檢查是否足夠
            if self._is_sufficient(all_docs):
                break

            # 生成下一輪查詢
            current_query = self._generate_followup(
                query,
                all_docs
            )

        return all_docs

    def _is_sufficient(self, docs):
        """判斷是否收集足夠"""
        # 簡單判斷: 超過 5 個文檔
        return len(docs) >= 5

    def _generate_followup(self, original, collected):
        """生成後續查詢"""
        # LLM 分析已收集的文檔，生成下一輪查詢
        pass
```

---

## 4. 重排與過濾

### Cross-Encoder 重排

```python
class CrossEncoderReranker:
    """Cross-Encoder 重排"""

    def __init__(self, model_name="cross-encoder/ms-marco"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=3):
        """重排候選文檔"""

        # 構建查詢-文檔對
        pairs = [
            (query, doc["content"])
            for doc in candidates
        ]

        # Cross-Encoder 評分
        scores = self.model.predict(pairs)

        # 按分數排序
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]
```

### 語義過濾

```python
class SemanticFilter:
    """語義過濾 - 去除不相關文檔"""

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.embedder = SentenceTransformer()

    def filter(self, query, documents):
        """過濾不相關文檔"""

        query_emb = self.embedder.encode(query)

        filtered = []
        for doc in documents:
            doc_emb = self.embedder.encode(doc["content"])

            similarity = self._cosine(query_emb, doc_emb)

            if similarity >= self.threshold:
                filtered.append({
                    **doc,
                    "relevance_score": similarity
                })

        return filtered

    def _cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 5. 生成增強

### 提示構建

```python
class RAGPromptBuilder:
    """RAG 提示構建器"""

    def build_prompt(self, query, retrieved_docs):
        """構建 RAG 提示"""

        # 格式化文檔
        context = self._format_context(retrieved_docs)

        prompt = f"""使用以下參考資料回答問題。如果資料不足，請明確說明。

參考資料:
{context}

問題: {query}

要求:
1. 只使用提供的資料回答
2. 標註來源
3. 如果不確定，請說「根據現有資料無法確定」
"""

        return prompt

    def _format_context(self, docs):
        """格式化檢索到的文檔"""

        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(
                f"[{i+1}] {doc['title']}\n"
                f"來源: {doc.get('source', '未知')}\n"
                f"內容: {doc['content'][:500]}..."
            )

        return "\n\n".join(formatted)
```

### 多來源融合

```python
class MultiSourceFusion:
    """多來源融合生成"""

    def __init__(self, llm):
        self.llm = llm

    def generate_with_sources(self, query, source_groups):
        """
        source_groups: {
            "wiki": [...],
            "docs": [...],
            "news": [...]
        }
        """

        # 分別從每個來源生成
        responses = {}

        for source_name, docs in source_groups.items():
            if not docs:
                continue

            prompt = self.build_prompt(query, docs)
            responses[source_name] = self.llm.generate(prompt)

        # 融合多個回覆
        final = self.fuse_responses(responses, query)

        return final

    def fuse_responses(self, responses, query):
        """融合多個回覆"""

        prompt = f"""
        根據以下不同來源的回覆，給出最終答案:

        {chr(10).join(f"來源 {i+1}: {r}" for i, r in enumerate(responses.values()))}

        問題: {query}

        原則:
        - 選擇最可靠的來源
        - 如有矛盾，說明差異
        - 綜合各來源的最佳資訊
        """

        return self.llm.generate(prompt)
```

---

## 6. 評估

### RAG 評估指標

```python
class RAGEvaluator:
    """RAG 評估"""

    METRICS = {
        "context_precision": "檢索品質",
        "context_recall": "召回率",
        "faithfulness": "事實一致性",
        "answer_relevance": "答案相關性"
    }

    def evaluate(self, rag_pipeline, eval_dataset):
        """評估 RAG 系統"""

        results = {
            metric: []
            for metric in self.METRICS
        }

        for sample in eval_dataset:
            # 生成
            response, contexts = rag_pipeline.generate(
                sample["query"]
            )

            # 評估每個指標
            results["context_precision"].append(
                self._eval_precision(contexts, sample["relevant_docs"])
            )
            results["faithfulness"].append(
                self._eval_faithfulness(response, contexts)
            )
            results["answer_relevance"].append(
                self._eval_relevance(response, sample["query"])
            )

        return {
            name: sum(vals) / len(vals)
            for name, vals in results.items()
        }

    def _eval_faithfulness(self, response, contexts):
        """評估事實一致性"""
        # 使用 LLM 判斷
        prompt = f"""
        判斷以下回覆是否與參考資料一致:

        回覆: {response}

        參考資料: {contexts}

        輸出: 一致的數量 / 總聲明數量
        """

        result = self.llm.generate(prompt)
        return self._parse_fraction(result)
```

---

## 7. 優化技巧

### 緩存策略

```python
class RAGCache:
    """RAG 緩存"""

    def __init__(self):
        self.query_cache = {}
        self.embedding_cache = {}

    def get_cached_results(self, query, top_k=5):
        """檢索緩存的結果"""

        # 精確匹配
        if query in self.query_cache:
            return self.query_cache[query]

        # 近似匹配 - 找到相似的查詢
        similar = self._find_similar(query)
        if similar:
            return similar

        return None

    def cache_results(self, query, docs):
        """緩存檢索結果"""
        self.query_cache[query] = docs

        # LRU 淘汰
        if len(self.query_cache) > 10000:
            self._evict_oldest()
```

### 監控儀表板

```python
class RAGMonitor:
    """RAG 監控"""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "avg_retrieved_docs": 0,
            "cache_hit_rate": 0,
            "no_results_rate": 0
        }

    def record(self, query, retrieval_time, num_docs):
        """記錄指標"""

        self.metrics["total_queries"] += 1

        # 移動平均
        n = self.metrics["total_queries"]
        self.metrics["avg_retrieval_time"] = (
            self.metrics["avg_retrieval_time"] * (n-1) + retrieval_time
        ) / n
        self.metrics["avg_retrieved_docs"] = (
            self.metrics["avg_retrieved_docs"] * (n-1) + num_docs
        ) / n

    def get_dashboard(self):
        """獲取監控儀表板"""
        return {
            **self.metrics,
            "health_score": self._calculate_health()
        }
```

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **GraphRAG** | RAG + 知識圖譜增強 |
| **VectorDB** | RAG 的核心檢索引擎 |
| **LangChain** | 提供 RAG 組件 |
| **Fine-tuning** | 可用於優化 RAG 生成 |

---

## 延伸閱讀

- [RAG Survey](https://arxiv.org/abs/2312.10997)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex](https://www.llamaindex.ai/)