# Tree of Thoughts (ToT)

讓 AI 能夠探索多条推理路徑的框架，類似人類的「三思而後行」，在做出決定前評估多種可能性。

---

## 1. 什麼是？

### 簡單範例

```
用戶: "如何用 [5, 5, 5, 1] 計算 24？"

ReAct (單路徑):
  5 + 5 = 10 → 10 + 5 = 15 → 15 + 1 = 16 ✗  (失敗)

ToT (多路徑):
       嘗試 1: (5+5) × 5 - 1 = 24 ✓  找到解答！
       嘗試 2: 5 × 5 - 5 + 1 = 21 ✗
       嘗試 3: (5-1) × 5 + 5 = 25 ✗
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **解決複雜問題** | 對於需要規劃的任務，單一路徑可能失敗 |
| **避免局部最優** | 探索多種方案找到最佳解 |
| **增強創造力** | 能夠產生更多樣化的解決方案 |
| **錯誤恢復** | 某路徑失敗可嘗試其他路徑 |

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

---

## 3. 核心原理

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

### 狀態評估方法

#### 1. LLM 投票

```python
def evaluate_state(state):
    """讓 LLM 評估狀態"""
    prompt = f"""
    評估以下狀態是否有希望找到解答：
    {state}

    回覆格式：
    - 分數 (1-10): _
    - 理由: _
    """
    return llm.generate(prompt)
```

#### 2. 啟發式函數

```python
def heuristic(state):
    """傳統啟發式評估"""
    if is_goal_state(state):
        return 10
    if is_dead_end(state):
        return 0
    return estimate_distance_to_goal(state)
```

#### 3. 外部工具驗證

```python
def evaluate_state(state):
    """用外部工具驗證"""
    if can_derive_to_int(state, target):
        return 10  # 確定可達
    return partial_match_score(state, target)
```

---

## 4. 實際應用場景

### 數學證明

```
問題: 證明勾股定理

ToT 探索:
  - 路徑 1: 幾何證明 (面積切割)
  - 路徑 2: 代數證明 (向量內積)
  - 路徑 3: 三角學證明
```

### 程式碼 Debugging

```
問題: 修復 Bug

ToT 探索:
  - 假設 1: 變數命名錯誤
  - 假設 2: 邏輯判斷錯誤
  - 假設 3: 邊界條件未處理
  - 假設 4: 異步問題
```

### 策略遊戲

```
Chess / Go:

ToT 探索:
  - 評估每步棋的所有可能
  - 模擬多步後的局勢
  - 選擇最優策略
```

### 創意寫作

```
問題: 寫一個故事開頭

ToT 探索:
  - 風格 1: 懸疑開頭
  - 風格 2: 溫馨開頭
  - 風格 3: 動作開頭

選擇最佳或混合使用
```

---

## 5. 實現方案

### BFS ToT (廣度優先)

- 廣度優先探索
- 保留每層所有有希望的節點
- 適合答案需要多樣性的場景

```python
def bfs_tot(initial_state, max_depth=3):
    # Level 0
    frontier = [initial_state]

    for depth in range(max_depth):
        next_frontier = []

        # 擴展所有節點
        for state in frontier:
            # 生成多個思考
            thoughts = generate_thoughts(state)

            # 評估每個思考
            for thought in thoughts:
                if evaluate(thought) > threshold:
                    next_frontier.append(thought)

        frontier = next_frontier

    return best_state(frontier)
```

### DFS ToT (深度優先)

- 深度優先探索
- 深入單一路徑直到找到解答
- 適合空間大但解法明確的問題

```python
def dfs_tot(state, depth=0):
    if is_goal(state):
        return state

    if depth > max_depth:
        return None

    # 生成思考
    thoughts = generate_thoughts(state)

    # 選擇最好的幾個
    thoughts = select_top_k(thoughts, k=2)

    for thought in thoughts:
        result = dfs_tot(thought, depth + 1)
        if result:
            return result

    return None
```

### Best-first Search

- 根據啟發式函數排序
- 優先探索最有希望的分支

```python
def best_first_tot(initial_state):
    # Priority queue: (score, state)
    pq = [(score(initial_state), initial_state)]

    while pq:
        _, state = heappop(pq)

        if is_goal(state):
            return state

        # 擴展狀態
        thoughts = generate_thoughts(state)
        for thought in thoughts:
            s = score(thought)
            heappush(pq, (s, thought))

    return None
```

### 搜尋策略比較

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **BFS** | 找到最短路徑 | 記憶體消耗大 | 答案需多樣性 |
| **DFS** | 記憶體效率高 | 可能錯過最優解 | 解法明確 |
| **Best-first** | 智慧選擇 | 需要好的啟發式 | 大搜索空間 |

---

## 6. 相關主題

| 技術 | 關係 |
|------|------|
| **ReAct** | ToT 的基礎，單路徑版本 |
| **Chain of Thought** | ToT 的簡化版，鏈狀思維 |
| **Self-Correction** | ToT 中的狀態評估可用於自我修正 |
| **Planning** | ToT 是規劃問題的核心技術 |

---

## 7. Function Calling vs ToT

| 維度 | Function Calling | ToT |
|------|-----------------|-----|
| **目的** | 調用外部工具 | 探索推理路徑 |
| **輸出** | 執行結果 | 最佳解答 |
| **複雜度** | 較低 | 較高 |
| **搜索** | 無 | 有 (BFS/DFS/Best-first) |

> 💡 Function Calling 是 ToT 的「工具」，可以讓 ToT 調用外部工具來驗證狀態

---

## 延伸閱讀

- [Tree of Thoughts Paper (arxiv)](https://arxiv.org/abs/2305.08291)
- [LangChain ToT](https://python.langchain.com/docs/modules/agents/agent_types/tot)
- [Google ToT Implementation](https://github.com/google-research/google-research/tree/master/tot)