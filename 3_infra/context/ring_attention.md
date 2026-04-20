# Ring Attention

## 1. 什麼是？
Ring Attention（環形注意力）是一種分散式長上下文注意力機制，將 KV Cache 分布到多個 GPU 裝置上，實現百萬級 token 上下文。

## 2. 為什麼重要？
- **突破單 GPU 記憶體限制**：標準 Attention 需要 O(n²) 記憶體
- **支援超長上下文**：百萬級 token 上下文
- **線性通訊成本**：相比傳統方法大幅降低節點間通訊

## 3. 核心原理

### 傳統問題
```
單 GPU 處理長序列：
- 序列長度 n
- KV Cache 記憶體：O(n)
- Attention 計算：O(n²)
- 受限於 GPU 記憶體大小
```

### Ring Attention 解決方案
```
多 GPU 環形分布：

GPU 0: [tokens 0-1000]    GPU 1: [tokens 1001-2000]
GPU 2: [tokens 2001-3000] GPU 3: [tokens 3001-4000]

環形通訊：
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
  ↑                                ↓
  └────────────────────────────────┘
```

### 關鍵技術
1. **分塊計算**：將 attention 分成多個 chunk
2. **流水線傳遞**：KV Cache 環形傳遞
3. **Overlap**：計算與通訊重疊

## 4. 相關主題

| 技術 | 關係 |
|------|------|
| **KV Cache** | Ring Attention 的基礎 |
| **Pipeline Parallelism** | 可與 Ring Attention 結合 |
| **RoPE** | 需要位置編碼外推配合長上下文 |

## 5. 延伸閱讀
- [Ring Attention Paper](https://arxiv.org/abs/2310.01889)
- [Long Context Training](https://arxiv.org/abs/2402.17762)

---

*待補充...*