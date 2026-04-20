# GraphRAG (Graph + RAG)

## 1. 什麼是？
GraphRAG 結合知識圖譜（Knowledge Graph）與檢索增強生成（RAG），利用圖結構的關係資訊增強檢索品質和答案準確性。

## 2. 為什麼重要？
- **理解關係**：能夠回答「A 與 B 的關係是什麼」
- **減少幻覺**：基於結構化知識提供答案
- **多跳推理**：能夠透過關係鏈進行推理
- **Context 濃縮**：用更少的 token 包含更多資訊

## 3. 核心原理

### 傳統 RAG 的問題
```
查詢: "張三的配偶的公司的 CEO 是誰？"

傳統 RAG:
- 可能只檢索到 "張三的配偶是李四"
- 但無法串聯 "李四的公司" → "CEO"
- 需要多跳推理，傳統 RAG 難以勝任
```

### GraphRAG 解決方案
```
知識圖譜:
  [張三] ──(配偶)──▶ [李四] ──(CEO)──▶ [王五]

GraphRAG 能夠:
1. 檢索 "張三" 節點
2. 透過 "配偶" 邊找到 "李四"
3. 透過 "CEO" 邊找到 "王五"
4. 輸出: "是王五"
```

### GraphRAG 流程
```
1. 知識抽取 (Knowledge Extraction)
   文檔 → LLM → 實體 + 關係 → 圖結構

2. 圖索引 (Graph Indexing)
   存入圖資料庫 (如 Neo4j)

3. 檢索 (Retrieval)
   查詢 → 擴展相關子圖 → 向量搜尋補充

4. 生成 (Generation)
   上下文 → LLM → 生成答案
```

### 圖向量化策略
| 方法 | 說明 |
|------|------|
| **Graph Embedding** | 將圖結構編碼為向量 |
| **Hybrid Search** | 圖檢索 + 向量檢索混合 |
| **Text2Cypher** | 將自然語言轉為圖查詢 |

## 4. 主流方案

### 圖資料庫
- **Neo4j** - 最流行的圖資料庫
- **NebulaGraph** - 國產開源圖資料庫
- **ArangoDB** - 多模型資料庫

### GraphRAG 框架
- **Microsoft GraphRAG** - 開源完整方案
- **LangChain Graph Chains** - 整合 LangChain
- **Neo4j + LangChain** - 常用組合

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **VectorDB** | GraphRAG 常常結合向量檢索 |
| **RAG** | GraphRAG 是 RAG 的進階版本 |
| **Knowledge Graph** | GraphRAG 的核心技術 |
| **Entity Extraction** | 從文本抽取實體的技術 |

## 6. 延伸閱讀
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Neo4j GraphRAG Guide](https://neo4j.com/product/graphrag/)
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)

---

*待補充...*