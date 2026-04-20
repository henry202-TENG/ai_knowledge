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

## 延伸閱讀

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Neo4j GraphRAG Guide](https://neo4j.com/product/graphrag/)
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [Knowledge Graph for RAG](https://arxiv.org/abs/2304.08466)