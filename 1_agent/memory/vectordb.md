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

## 7. 數學原理

### 向量空間模型

```
給定查詢向量 q 和文檔向量 d，相似度計算：

1. Cosine Similarity (餘弦相似度):
   sim(q, d) = (q · d) / (||q|| × ||d||)
   
   - 範圍: [-1, 1]
   - 忽略向量大小，只考慮方向
   - 最適用於非標準化 embeddings

2. Dot Product (點積):
   sim(q, d) = q · d = Σ(qᵢ × dᵢ)
   
   - 範圍: [-∞, ∞]
   - 當向量已 L2 標準化時，等價於 Cosine
   - 計算最快

3. Euclidean Distance (歐氏距離):
   dist(q, d) = √(Σ(qᵢ - dᵢ)²)
   
   - 範圍: [0, ∞]
   - 轉為相似度: 1 / (1 + dist)
```

### ANN 複雜度分析

| 算法 | 構建複雜度 | 查詢複雜度 | 空間複雜度 |
|------|------------|------------|------------|
| **Brute Force** | O(1) | O(n) | O(n) |
| **HNSW** | O(n log n) | O(log n) | O(n log n) |
| **IVF** | O(n log n) | O(log n + k/n₁) | O(n) |
| **PQ** | O(n × d / m) | O(d/m + k × m) | O(n × d / (m × b)) |

```
n = 總向量數
d = 向量維度
k = 返回數量
n₁ = 每個 cluster 的平均向量數
m = PQ 子向量數
b = 碼本大小
```

### 精確度 vs 速度權衡

```
recall@k vs 查詢時間關係:

Brute Force:  ████████████████████████████████████████ 100% recall, 最慢
HNSW (ef=128): ████████████████████████▌ 95% recall, 中等
HNSW (ef=32):  ████████████████░░░░░░░░░ 80% recall, 快
IVF (nlist=100):███████████████░░░░░░░░░░░ 70% recall, 快

選擇建議:
- 對話系統: recall > 90%, 延遲 < 100ms
- 推薦系統: recall > 80%, 延遲 < 50ms
- 離線分析: recall > 95%, 可接受較慢
```

---

## 8. 進階索引策略

### 分層索引架構

```python
class HierarchicalVectorIndex:
    """分層索引: 粗召回 → 精排序"""

    def __init__(self, coarse_index, fine_index, reranker):
        self.coarse = coarse_index    # HNSW 快速召回
        self.fine = fine_index        # IVF 精確篩選
        self.reranker = reranker      # Cross-encoder 排序

    def search(self, query_vector, top_k=10):
        # Stage 1: 粗召回 (1000 candidates)
        coarse_results = self.coarse.search(
            query_vector,
            top_k=1000
        )

        # Stage 2: 精篩選 (100 candidates)
        fine_results = self.fine.search(
            query_vector,
            filter={"id IN": coarse_results.ids},
            top_k=100
        )

        # Stage 3: Rerank (10 final)
        final_results = self.reranker.rank(
            query_vector,
            fine_results,
            top_k=top_k
        )

        return final_results
```

### 動態索引更新

```python
class DynamicVectorIndex:
    """支援即時更新的索引"""

    def __init__(self):
        self.index = hnswlib.Index(space='cosine', dim=1536)
        self.pending_updates = []
        self.batch_size = 100
        self.flush_interval = 300  # 5 分鐘

    def add_vector(self, vector_id, embedding, metadata):
        """添加向量"""
        self.pending_updates.append({
            "id": vector_id,
            "vector": embedding,
            "metadata": metadata
        })

        if len(self.pending_updates) >= self.batch_size:
            self._flush_updates()

    def _flush_updates(self):
        """批量更新索引"""
        if not self.pending_updates:
            return

        ids = [u["id"] for u in self.pending_updates]
        vectors = [u["vector"] for u in self.pending_updates]

        self.index.add_items(vectors, ids)
        self.pending_updates.clear()
```

---

## 9. 多租戶架構

### 隔離策略

```python
class MultiTenantVectorDB:
    def __init__(self):
        self.namespaces = {}  # namespace -> collection

    def create_tenant(self, tenant_id):
        """為每個租戶創建獨立命名空間"""
        self.namespaces[tenant_id] = {
            "collection": Collection(name=f"tenant_{tenant_id}"),
            "metadata": {"created_at": datetime.now()},
            "quota": {"max_vectors": 1_000_000, "used": 0}
        }

    def search_with_tenant(self, tenant_id, query, **kwargs):
        """隔離搜索"""
        if tenant_id not in self.namespaces:
            raise TenantNotFoundError(tenant_id)

        collection = self.namespaces[tenant_id]["collection"]
        return collection.search(query, **kwargs)

    def enforce_quota(self, tenant_id):
        """配額管理"""
        quota = self.namespaces[tenant_id]["quota"]
        if quota["used"] >= quota["max_vectors"]:
            raise QuotaExceededError(tenant_id)
```

### 資源共享策略

| 策略 | 優點 | 缺點 |
|------|------|------|
| **獨立 Collection** | 完全隔離 | 資源浪費 |
| **Shared Collection + Filter** | 資源效率高 | 查詢開銷增加 |
| **Shared Collection + Namespace** | 平衡 | 需要隔離元資料 |

---

## 10. 安全與隱私

### 訪問控制

```python
class VectorDBAccessControl:
    def __init__(self):
        self.permissions = {}  # user_id -> {collection: [read, write]}

    def check_permission(self, user_id, collection, operation):
        """權限檢查"""
        user_perms = self.permissions.get(user_id, {})
        col_perms = user_perms.get(collection, [])

        if operation not in col_perms:
            raise PermissionDeniedError(
                f"User {user_id} lacks {operation} permission"
            )

    def grant_access(self, user_id, collection, operations):
        """授權"""
        if user_id not in self.permissions:
            self.permissions[user_id] = {}
        self.permissions[user_id][collection] = operations
```

### 資料加密

| 層級 | 方法 | 應用 |
|------|------|------|
| **傳輸加密** | TLS 1.3 | 客戶端與服務器之間 |
| **儲存加密** | AES-256 | 磁碟上的向量資料 |
| **查詢加密** | Homomorphic Encryption | 密文檢索 (實驗性) |
| **脫敏處理** | Differential Privacy | 查詢日誌分析 |

### 合規考慮

```
GDPR 合規要點:
- 資料刪除權: 支援完全刪除用戶資料
- 資料遷移: 匯出向量和元資料
- 存取記錄: 審計日誌保留
- 處理記錄: 資料處理活動記錄
```

---

## 11. 效能監控

### 關鍵指標

```python
class VectorDBMonitor:
    def __init__(self):
        self.metrics = {
            "queries": [],
            "latencies": [],
            "errors": []
        }

    def record_query(self, query_id, latency_ms, result_count, error=None):
        self.metrics["queries"].append({
            "id": query_id,
            "latency_ms": latency_ms,
            "results": result_count,
            "error": error,
            "timestamp": time.time()
        })

    def get_stats(self):
        recent = self.metrics["queries"][-1000:]

        return {
            "qps": len(recent) / 300,  # 最近 5 分鐘
            "p50_latency": np.percentile([q["latency_ms"] for q in recent], 50),
            "p99_latency": np.percentile([q["latency_ms"] for q in recent], 99),
            "error_rate": sum(1 for q in recent if q["error"]) / len(recent),
            "avg_results": np.mean([q["results"] for q in recent])
        }
```

### 告警規則

| 指標 | 告警閾值 | 嚴重程度 |
|------|----------|----------|
| P99 延遲 | > 500ms | Warning |
| P99 延遲 | > 2000ms | Critical |
| 錯誤率 | > 1% | Warning |
| 錯誤率 | > 5% | Critical |
| 磁碟使用 | > 80% | Warning |
| 記憶體使用 | > 90% | Critical |

---

## 12. 成本優化

### 存儲優化策略

```python
class VectorDBCostOptimizer:
    def __init__(self, db):
        self.db = db

    def apply_quantization(self, collection, method="sq"):
        """自動量化壓縮"""

        if method == "sq":
            # Scalar Quantization: float32 → int8
            self.db.alter_collection(
                collection,
                params={"quantization": "scalar", "type": "int8"})
        elif method == "pq":
            # Product Quantization
            self.db.alter_collection(
                collection,
                params={"quantization": "product", "sub_vectors": 8})

    def tiered_storage(self, collection):
        """分層存儲: 熱數據 vs 冷數據"""

        # 最近 30 天: SSD 存儲
        self.db.set_tier(collection, "hot", {
            "storage": "ssd",
            "retention": "30d"
        })

        # 較舊數據: 歸檔存儲
        self.db.set_tier(collection, "archive", {
            "storage": "s3",
            "retention": "1y"
        })
```

### 成本計算

| 方案 | 100 萬向量/月 | 1 億向量/月 |
|------|---------------|-------------|
| **Pinecone (p1)** | ~$70 | ~$700 |
| **Milvus (自託管)** | ~$50 | ~$500 |
| **Qdrant (雲端)** | ~$60 | ~$600 |
| **Chroma (本地)** | ~$10* | ~$100* |

*不含硬體成本

---

## 13. 相關主題

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
- [Facebook FAISS](https://github.com/facebookresearch/faiss)
- [Annoy Documentation](https://github.com/spotify/annoy)