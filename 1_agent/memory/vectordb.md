# Vector Database (向量資料庫)

## 1. 什麼是？
向量資料庫是一種專門用於存儲和檢索高維向量（embeddings）的資料庫，能夠實現語義相似性檢索，是 RAG 系統的核心組件。

## 2. 為什麼重要？
- **語義檢索**：找出「意思相近」而非「關鍵字匹配」的結果
- **RAG 基礎設施**：讓 LLM 能夠訪問外部知識
- **長記憶體**：讓 AI 記住對話歷史和知識

## 3. 核心原理

### 傳統關鍵字檢索 vs 向量檢索
```
關鍵字檢索:
  查詢 "狗狗" → 只返回包含 "狗狗" 的文檔

向量檢索:
  查詢 "狗狗" → 返回 "寵物"、"犬類"、"小明養的動物"...
  （語義相近的結果）
```

### Embedding 流程
```
文字輸入: "今天天氣很好"

        ↓ Embedding Model
        ↓ (e.g., text-embedding-3-small)

向量輸出: [0.12, -0.34, 0.56, ..., 0.89]
         ↓
      存儲到向量資料庫
```

### 相似度計算
| 方法 | 公式 | 適用場景 |
|------|------|----------|
| **Cosine Similarity** | cos(θ) = A·B / (\|A\|\|B\|) | 最常用，方向比大小重要 |
| **Euclidean Distance** | √(Σ(Aᵢ-Bᵢ)²) | 距離敏感 |
| **Dot Product** | A·B | 當向量已標準化 |

### 索引技術 (近似最近鄰 ANN)
| 技術 | 原理 | 優點 | 缺點 |
|------|------|------|------|
| **HNSW** | 分層圖結構 | 高效率、高品質 | 記憶體消耗大 |
| **IVF** | 倒排索引 | 支援大規模數據 | 精度可能下降 |
| **Pinecone** | 雲端托管 | 易用、 scalable | 需付費 |

## 4. 主流方案

### 開源向量資料庫
- **Milvus** - 功能完整，支援多種索引
- **Qdrant** - Rust 實現，高性能
- **Chroma** - 輕量，適合中小規模
- **Weaviate** - 支援 GraphQL

### 雲端服務
- **Pinecone** - 最流行的托管服務
- **Azure AI Search** - 微軟雲端
- **Amazon OpenSearch Service** - AWS
- **Vertex AI Vector Search** - Google Cloud

### 輕量方案
- **FAISS** - Facebook 實現，記憶體型
- **Annoy** - Spotify 實現，磁碟型

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **Embedding** | 向量資料庫的基礎，文字→向量 |
| **RAG** | 依賴向量資料庫實現檢索 |
| **GraphRAG** | 結合知識圖譜增強檢索 |
| **Chunking** | 將長文本分塊以便檢索 |

## 6. 延伸閱讀
- [Pinecone Vector DB Guide](https://www.pinecone.io/learn/)
- [Milvus Documentation](https://milvus.io/docs)
- [Vector Database Comparison](https://arxiv.org/abs/2202.09583)

---

*待補充...*