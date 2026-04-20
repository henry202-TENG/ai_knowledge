# GraphRAG

結合知識圖譜（Knowledge Graph）與檢索增強生成（RAG），利用圖結構的關係資訊增強檢索品質和答案準確性。

---

## 1. 什麼是？

### 簡單範例

```
用戶: "張三的配偶的公司的 CEO 是誰？"

傳統 RAG:
  檢索 "張三的配偶" → 找到 "張三的配偶是李四"
  → 但無法串聯 "李四的公司" → "CEO"
  → 輸出: "我不知道" ✗

GraphRAG:
  知識圖譜:
    [張三] ──(配偶)──▶ [李四] ──(CEO)──▶ [王五]

  檢索流程:
    1. 找到 "張三" 節點
    2. 沿著 "配偶" 邊找到 "李四"
    3. 沿著 "CEO" 邊找到 "王五"
    4. 輸出: "是王五" ✓
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **理解關係** | 能夠回答「A 與 B 的關係是什麼」類型的問題 |
| **減少幻覺** | 基於結構化知識提供答案，減少 LLM 幻想 |
| **多跳推理** | 能夠透過關係鏈進行推理 (A→B→C→...) |
| **Context 濃縮** | 用更少的 token 包含更多結構化資訊 |
| **可解釋性** | 答案可回溯到知識圖譜中的具體節點 |

---

## 3. 核心原理

### GraphRAG 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│  步驟 1: 知識抽取 (Knowledge Extraction)                        │
│  文檔 → LLM → 實體 + 關係 + 屬性 → 圖結構                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  步驟 2: 圖索引 (Graph Indexing)                                 │
│  存入圖資料庫 (Neo4j, NebulaGraph)                              │
│  + 建立向量索引 (可選)                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  步驟 3: 檢索 (Retrieval)                                        │
│  查詢 → 擴展相關子圖 → 向量搜尋補充                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  步驟 4: 生成 (Generation)                                       │
│  上下文 + 圖譜資訊 → LLM → 生成答案                               │
└─────────────────────────────────────────────────────────────────┘
```

### 知識抽取 Prompt 設計

```python
EXTRACTION_PROMPT = """
從以下文本中抽取實體和關係。

文本: {document}

請以以下 JSON 格式輸出:
{{
    "entities": [
        {{"name": "實體名", "type": "實體類型", "attributes": {{}}}}
    ],
    "relations": [
        {{"source": "來源實體", "target": "目標實體", "type": "關係類型"}}
    ]
}}

規則:
1. 實體名稱使用原文
2. 關係類型使用通用詞 (如 "僱用"、"擁有"、"位於")
3. 只抽取重要實體和關係
"""
```

### 圖向量化策略

| 方法 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| **Graph Embedding** | 將圖結構編碼為向量 (DeepWalk, Node2Vec) | 統一表示 | 丟失語義 |
| **Hybrid Search** | 圖檢索 + 向量檢索混合 | 互補 | 複雜 |
| **Text2Cypher** | 將自然語言轉為圖查詢 Cypher | 精確 | 需要訓練 |
| **實體鏈接** | 將檢索到的實體擴展到相關實體 | 擴展上下文 | 可能過多 |

### 混合檢索策略

```python
def hybrid_graph_rag(query, graph_db, vector_index):
    # 1. 向量檢索
    vector_results = vector_index.search(query, top_k=10)

    # 2. 圖檢索 - 找到相關實體
    entities = extract_entities(query)
    graph_results = []
    for entity in entities:
        # 擴展 2 跳鄰居
        subgraph = graph_db.get_subgraph(entity, hops=2)
        graph_results.append(subgraph)

    # 3. 合併結果
    combined_context = merge(
        vector_results,
        graph_results,
        rerank=True
    )

    # 4. 生成答案
    return llm.generate(query, combined_context)
```

---

## 4. 實作細節

### 圖資料庫 Schema 設計

```cypher
// 節點定義
CREATE (p:Person {name: '張三', age: 35})
CREATE (c:Company {name: 'ABC公司', industry: '科技'})
CREATE (e:Employee {name: '李四', title: 'CEO'})

// 關係定義
CREATE (p)-[:配偶]->(e)
CREATE (e)-[:CEO_OF]->(c)

// 索引
CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.name)
CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name)
```

### 常見問題與解決

| 問題 | 原因 | 解決方案 |
|------|------|----------|
| **實體遺漏** | Prompt 不夠清楚 | 改進抽取 Prompt，加入範例 |
| **關係雜訊** | 抽取過多無關關係 | 設定關係置信度閾值 |
| **擴展爆炸** | 圖擴展太大 | 限制跳數和節點數 |
| **語義鴻溝** | 圖結構 vs 語義檢索 | 使用 Hybrid Search |

---

## 5. 主流方案

### 圖資料庫

| 資料庫 | 特色 | 適用場景 |
|--------|------|----------|
| **Neo4j** | 最流行、Cypher 語法 | 通用場景 |
| **NebulaGraph** | 國產、高效能 | 大規模圖譜 |
| **ArangoDB** | 多模型 (圖+文檔) | 靈活架構 |
| **TuGraph** | 螞蟻開源 | 超大規模 |

### GraphRAG 框架

| 框架 | 說明 |
|------|------|
| **Microsoft GraphRAG** | 開源完整方案，包含抽取+索引+檢索+生成 |
| **LangChain Graph Chains** | 整合 LangChain 生態 |
| **Neo4j + LangChain** | 常用組合 |
| **GraphRAG SDK** | 微軟提供的 Python SDK |

### Microsoft GraphRAG 流程

```python
from graphrag import GraphRAG

# 1. 初始化
graphrag = GraphRAG(
    input_dir="./data",
    output_dir="./output",
    graph_db="neo4j"
)

# 2. 建立知識圖譜 (自動抽取)
graphrag.build()

# 3. 檢索和生成
response = graphrag.query("張三的配偶的公司的 CEO 是誰？")
```

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **VectorDB** | GraphRAG 常常結合向量檢索 (Hybrid Search) |
| **RAG** | GraphRAG 是 RAG 的進階版本 |
| **Knowledge Graph** | GraphRAG 的核心技術基礎 |
| **Entity Extraction** | 從文本抽取實體的技術 |
| **Chunking** | 圖譜建構前的文本分塊 |

---

## 7. 圖嵌入算法

### 常見圖嵌入方法

| 方法 | 原理 | 維度 | 適用場景 |
|------|------|------|----------|
| **DeepWalk** | 隨機遊走 + Word2Vec | 64-256 | 節點分類 |
| **Node2Vec** | 控制隨機遊走策略 | 64-256 | 連結預測 |
| **GraphSAGE** | 鄰居聚合 | 64-256 | 大規模圖 |
| **TransE** | 翻譬模型 | 64-256 | 關係推理 |
| **KEPLER** | 聯合表示 | 768 | 語義 + 結構 |

### Node2Vec 實作

```python
import networkx as nx
from node2vec import Node2Vec

def generate_graph_embeddings(graph, dimensions=128):
    """生成圖嵌入"""

    # 隨機遊走參數
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=30,
        num_walks=200,
        workers=4,
        p=1.0,  # Return parameter
        q=1.0   # In-out parameter
    )

    # 訓練 Word2Vec
    model = node2vec.fit(window=10, min_count=1)

    # 取得節點嵌入
    embeddings = {
        node: model.wv[node]
        for node in graph.nodes()
    }

    return embeddings
```

### 知識圖譜表示學習

```
TransE 原理:

h + r ≈ t

範例:
  張三 + 配偶 ≈ 李四
  李四 + CEO_of ≈ ABC公司

訓練目標: 讓正確的三元組距離近，錯誤的三元組距離遠
```

---

## 8. 知識圖譜建構流程

### 完整 Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  文本預處理   │ -> │  實體抽取     │ -> │  關係抽取    │
│  - 分塊       │    │  - NER       │    │  - 關係分類   │
│  - 清理       │    │  - 實體鏈接   │    │  - 屬性提取   │
└──────────────┘    └──────────────┘    └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  圖譜存儲     │ <- │  圖譜融合     │ <- │  實體消歧     │
│  - 圖資料庫   │    │  - 合併冗餘   │    │  - 統一引用   │
│  - 向量索引   │    │  - 衝突解決   │    │  - 對齊現有   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 實體消歧策略

```python
class EntityDisambiguator:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.entity_cache = {}

    def disambiguate(self, entity_mentions):
        """消歧實體"""

        disambiguated = []

        for mention in entity_mentions:
            # 向量相似度匹配
            candidates = self.vector_db.search(
                mention.text,
                top_k=5
            )

            # 候選排序
            best = self._rank_candidates(mention, candidates)

            # 消歧決策
            if best and best.score > 0.85:
                disambiguated.append(best.entity)
            else:
                # 建立新實體
                new_entity = self._create_new_entity(mention)
                disambiguated.append(new_entity)

        return disambiguated

    def _rank_candidates(self, mention, candidates):
        """候選排序"""
        scores = []
        for candidate in candidates:
            # 合併多個信號
            vector_score = candidate.similarity
            context_score = self._context_similarity(mention, candidate)
            link_score = candidate.prior_probability

            final_score = (
                0.5 * vector_score +
                0.3 * context_score +
                0.2 * link_score
            )
            scores.append((candidate, final_score))

        return max(scores, key=lambda x: x[1]) if scores else None
```

---

## 9. 多跳查詢處理

### 查詢分解

```python
class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, query):
        """將複雜查詢分解為簡單子查詢"""

        prompt = f"""
查詢: {query}

將這個查詢分解為多個簡單的圖譜查詢。

範例:
輸入: "張三的配偶的公司的 CEO 是誰？"
輸出:
  1. 找張三的配偶
  2. 找該公司
  3. 找該公司的 CEO

請輸出分解後的子查詢（每行一個）:
"""

        response = self.llm.generate(prompt)
        sub_queries = [line.strip() for line in response.split('\n')
                      if line.strip()]

        return sub_queries
```

### 遞迴檢索

```python
def multi_hop_retrieval(query, graph_db, max_hops=3):
    """遞迴多跳檢索"""

    # 解析查詢獲取起始實體
    start_entities = extract_main_entities(query)

    results = []
    current_entities = start_entities

    for hop in range(max_hops):
        # 擴展鄰居
        expanded = graph_db.expand_nodes(
            current_entities,
            relation_types=get_relevant_relations(query)
        )

        # 評估相關性
        relevant = filter_by_relevance(expanded, query)

        results.extend(relevant)

        # 準備下一跳
        current_entities = [r["target"] for r in relevant]

        if not current_entities:
            break

    return results
```

---

## 10. 圖譜質量管理

### 品質評估指標

```python
class GraphQualityMetrics:
    @staticmethod
    def completeness(kg):
        """完整性: 實體覆蓋率"""
        total_mentions = kg.get_total_entity_mentions()
        linked_entities = kg.get_linked_entities()
        return linked_entities / total_mentions if total_mentions > 0 else 0

    @staticmethod
    def accuracy(kg):
        """準確性: 正確關係比例"""
        # 可通過抽樣人工評估或自動化驗證
        return kg.get_verified_relations() / kg.get_total_relations()

    @staticmethod
    def consistency(kg):
        """一致性: 語義衝突檢測"""
        conflicts = kg.detect_conflicts()
        return 1 - (conflicts / kg.get_total_relations())

    @staticmethod
    def freshness(kg):
        """時效性: 最近更新時間"""
        last_update = kg.get_last_update_time()
        days_old = (datetime.now() - last_update).days
        return max(0, 1 - days_old / 30)  # 30 天內為新鮮
```

### 圖譜監控儀表板

| 指標 | 目標 | 告警閾值 |
|------|------|----------|
| 實體數 | 持續增長 | 下降 |
| 平均入度 | > 2 | < 1 |
| 孤立節點比例 | < 10% | > 20% |
| 查詢延遲 P99 | < 500ms | > 1000ms |
| 圖譜完整性 | > 90% | < 80% |

---

## 11. 擴展策略

### 增量更新

```python
class IncrementalGraphUpdate:
    def __init__(self, graph_db):
        self.db = graph_db
        self.change_log = []

    def update_from_documents(self, new_documents):
        """增量更新圖譜"""

        for doc in new_documents:
            # 抽取新實體關係
            new_er = extract_entities_relations(doc)

            # 與現有圖譜比對
            merged = self._merge_with_existing(new_er)

            # 計算差異
            changes = self._compute_diff(merged)

            if changes:
                # 應用更新
                self.db.apply_changes(changes)
                self.change_log.append(changes)

    def _compute_diff(self, merged):
        """計算差異"""
        added = merged["new"]
        updated = merged["updated"]
        unchanged = merged["unchanged"]

        return {
            "added_entities": added["entities"],
            "added_relations": added["relations"],
            "updated_entities": updated,
            "conflict_resolutions": self._resolve_conflicts(merged)
        }
```

### 圖譜版本控制

```python
class GraphVersioning:
    def __init__(self, graph_db):
        self.db = graph_db

    def create_snapshot(self, label):
        """創建圖譜快照"""
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 導出完整圖譜
        export_data = {
            "nodes": self.db.export_nodes(),
            "edges": self.db.export_edges(),
            "metadata": {"label": label, "timestamp": datetime.now()}
        }

        # 存儲到備份
        self._store_snapshot(snapshot_id, export_data)

        return snapshot_id

    def restore(self, snapshot_id):
        """恢復到指定版本"""
        snapshot = self._load_snapshot(snapshot_id)
        self.db.restore(snapshot)
```

---

## 12. 成本與效能

### 架構選擇

| 方案 | 規模 | 成本 | 延遲 |
|------|------|------|------|
| **Neo4j 單機** | < 1000 萬節點 | 低 | < 100ms |
| **Neo4j 集群** | < 1 億節點 | 中 | < 200ms |
| **NebulaGraph** | 10 億+ 節點 | 中高 | < 100ms |
| **TuGraph** | 超大規模 | 高 | < 50ms |

### 查詢優化

```python
class GraphQueryOptimizer:
    def __init__(self, graph_db):
        self.db = graph_db

    def optimize_query(self, cypher):
        """優化 Cypher 查詢"""

        # 分析查詢計劃
        plan = self.db.explain(cypher)

        # 優化建議
        suggestions = []

        if "SCAN" in plan:
            suggestions.append("添加索引")

        if plan["rows"] > 10000:
            suggestions.append("限制結果數量")

        if "Cartesian" in plan:
            suggestions.append("添加關係類型過濾")

        return suggestions
```

---

## 13. 相關主題

| 技術 | 關係 |
|------|------|
| **VectorDB** | GraphRAG 常常結合向量檢索 (Hybrid Search) |
| **RAG** | GraphRAG 是 RAG 的進階版本 |
| **Knowledge Graph** | GraphRAG 的核心技術基礎 |
| **Entity Extraction** | 從文本抽取實體的技術 |
| **Chunking** | 圖譜建構前的文本分塊 |

---

## 延伸閱讀

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Neo4j GraphRAG Guide](https://neo4j.com/product/graphrag/)
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [Knowledge Graph for RAG](https://arxiv.org/abs/2304.08466)
- [Node2Vec Paper](https://arxiv.org/abs/1607.00653)
- [TransE Paper](https://arxiv.org/abs/1301.3781)