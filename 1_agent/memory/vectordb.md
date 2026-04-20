# Vector Database

專門用於存儲和檢索高維向量（embeddings）的資料庫，能夠實現語義相似性檢索，是 RAG 系統的核心組件。

---

## 1. 什麼是？

### 簡單範例

```
用戶: "狗狗的照片"

關鍵字檢索:
  查詢 "狗狗" → 只返回包含 "狗狗" 的文檔

向量檢索:
  查詢 "狗狗" → 返回 "寵物"、"犬類"、"小明養的動物"、"寵物狗"...
  （語義相近的結果，即使沒有關鍵字 "狗狗"）
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **語義檢索** | 找出「意思相近」而非「關鍵字匹配」的結果 |
| **RAG 基礎設施** | 讓 LLM 能夠訪問外部知識 |
| **長記憶體** | 讓 AI 記住對話歷史和知識 |
| **可擴展性** | 支援十億級向量檢索 |

---

## 3. 核心原理

### Embedding 流程

```
文字輸入: "今天天氣很好"

        ↓ Embedding Model
        ↓ (e.g., text-embedding-3-small)

向量輸出: [0.12, -0.34, 0.56, ..., 0.89]  (1536 維)
         ↓
      存入向量資料庫
```

### 相似度計算

| 方法 | 公式 | 適用場景 | 計算成本 |
|------|------|----------|----------|
| **Cosine Similarity** | cos(θ) = A·B / (\|A\|\|B\|) | 最常用，方向比大小重要 | O(d) |
| **Euclidean Distance** | √(Σ(Aᵢ-Bᵢ)²) | 距離敏感 | O(d) |
| **Dot Product** | A·B | 當向量已標準化 | O(d) |

> d = 向量維度

### 索引技術 (近似最近鄰 ANN)

| 技術 | 原理 | 優點 | 缺點 | 適用場景 |
|------|------|------|------|----------|
| **HNSW** | 分層圖結構 | 高效率、高品質 | 記憶體消耗大 | 線上服務 |
| **IVF** | 倒排索引 | 支援大規模數據 | 精度可能下降 | 離線批次 |
| **PQ** | 產品量化 | 壓縮率高 | 精度損失 | 記憶體受限 |
| **LSH** | 局部敏感雜湊 | 精確搜索 | 記憶體高 | 近似相等 |

### HNSW 參數調優

| 參數 | 說明 | 建議值 |
|------|------|--------|
| **M** (每節點連接數) | 控制圖的密度 | 16-64 |
| **efConstruction** | 建立時的搜索寬度 | 100-500 |
| **efSearch** | 查詢時的搜索寬度 | 50-200 |

```python
# Milvus HNSW 配置
index_params = {
    "index_type": "HNSW",
    "M": 32,
    "efConstruction": 200,
    "efSearch": 128,
    "metric_type": "COSINE"
}
```

### 向量量化技術

| 方法 | 說明 | 壓縮比 | 精度損失 |
|------|------|--------|----------|
| **PQ (Product Quantization)** | 將向量分塊分別量化 | 4-32x | 中等 |
| **SQ (Scalar Quantization)** | 將 float32 轉為 int8 | 4x | 低 |
| **Binary Quantization** | 轉為 0/1 | 32x | 高 |
| **BVQ (Binary Vector Quantization)** | 優化的二值化 | 16-32x | 中等 |

---

## 4. 搜尋參數調優

### 混合檢索 (Hybrid Search)

```python
# 向量搜尋 + 關鍵字篩選
search_params = {
    "vector": {
        "top_k": 10,
        "ef": 128
    },
    "filter": {
        "category": "tech",
        "year": {"$gte": 2020}
    },
    "rerank": True  # 使用 reranker 重新排序
```

### Reranking

```python
# 兩階段檢索
# 階段 1: ANN 快速檢索 (top_k=100)
candidates = vector_db.search(query, top_k=100)

# 階段 2: Cross-encoder rerank (top_k=10)
from rerankers import Reranker
reranker = Reranker("cross-encoder")
results = reranker.rank(query, candidates, top_k=10)
```

### 過濾器與向量混合

```python
# 過濾條件 + 向量搜尋
results = collection.search(
    query_vector=query_embedding,
    filter="category == 'AI' and year >= 2023",
    limit=10
)
```

---

## 5. 主流方案

### 開源向量資料庫

| 資料庫 | 語言 | 特色 | 適用場景 |
|--------|------|------|----------|
| **Milvus** | Go | 功能完整，支援多種索引 | 企業級 |
| **Qdrant** | Rust | 高性能，延遲低 | 低延遲服務 |
| **Chroma** | Python | 輕量，易用 | 中小規模 |
| **Weaviate** | Go | 支援 GraphQL + 混合搜尋 | 靈活架構 |

### 雲端服務

| 服務 | 供應商 | 特色 |
|------|--------|------|
| **Pinecone** | AWS/Azure | 最流行的托管服務 |
| **Azure AI Search** | Microsoft | 企業級搜尋 |
| **Amazon OpenSearch** | AWS | 向量搜尋 |
| **Vertex AI Vector Search** | Google Cloud | 大規模 |

### 輕量方案

| 方案 | 特色 | 限制 |
|------|------|------|
| **FAISS** | Facebook 實現，記憶體型 | 單機 |
| **Annoy** | Spotify 實現，磁碟型 | 僅支援 Euclidean/Cosine |
| **ScaNN** | Google 實現，高效能 | 需要 TensorFlow |

---

## 6. 與相關技術的關係

| 技術 | 關係 |
|------|------|
| **Embedding** | 向量資料庫的基礎，文字→向量 |
| **RAG** | 依賴向量資料庫實現檢索 |
| **GraphRAG** | 結合知識圖譜增強檢索 |
| **Chunking** | 將長文本分塊以便檢索 |
| **Reranking** | 兩階段檢索提升精度 |

---

## 延伸閱讀

- [Pinecone Vector DB Guide](https://www.pinecone.io/learn/)
- [Milvus Documentation](https://milvus.io/docs)
- [Vector Database Comparison](https://arxiv.org/abs/2202.09583)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)