# Tree of Thoughts (ToT)

## 1. 什麼是？
Tree of Thoughts 是一種讓 AI 能夠探索多条推理路徑的框架，類似人類的「三思而後行」，在做出決定前評估多種可能性。

## 2. 為什麼重要？
- **解決複雜問題**：對於需要規劃的任務，單一路徑可能失敗
- **避免局部最優**：探索多種方案找到最佳解
- **增強創造力**：能夠產生更多樣化的解決方案

## 3. 核心原理

### 與 ReAct 的比較
```
ReAct: 單一路徑 chain
  A → B → C → D → 答案

ToT:  多路徑 tree
        答案
       / | \
     B1  B2  B3
    / \  |  / \
   A1  A2 A3 A4 A5
```

### ToT 四個步驟
1. **Thought Generation** (生成思考)
   - 從當前狀態生成多個可能的思考

2. **State Evaluation** (狀態評估)
   - 評估每個狀態的可行性

3. **Move Generation** (移動生成)
   - 選擇要擴展的分支

4. **Search Algorithm** (搜索算法)
   - BFS (廣度優先) 或 DFS (深度優先)

### 範例：24 點遊戲
```
輸入: [5, 5, 5, 1] 使用 +, -, *, / 計算 24

分支 1:
  5 + 5 = 10 → 10 + 5 = 15 → 15 + 1 = 16 ✗

分支 2:
  5 × 5 = 25 → 25 - 1 = 24 ✓  找到解答！

分支 3:
  5 - 1 = 4 → 5 × 4 = 20 → 20 + 5 = 25 ✗
```

## 4. 實現方案

### BFS ToT
- 廣度優先探索
- 保留每層所有有希望的節點
- 適合答案需要多樣性的場景

### DFS ToT
- 深度優先探索
- 深入單一路徑直到找到解答
- 適合空間大但解法明確的問題

### Best-first Search
- 根據啟發式函數排序
- 優先探索最有希望的分支

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **ReAct** | ToT 的基礎，單路徑版本 |
| **Chain of Thought** | ToT 的簡化版，鏈狀思維 |
| **Self-Correction** | ToT 中的狀態評估可用於自我修正 |
| **Planning** | ToT 是規劃問題的核心技術 |

## 6. 延伸閱讀
- [Tree of Thoughts Paper (arxiv)](https://arxiv.org/abs/2305.08291)
- [LangChain ToT](https://python.langchain.com/docs/modules/agents/agent_types/tot)
- [Google ToT Implementation](https://github.com/google-research/google-research/tree/master/tot)

---

*待補充...*