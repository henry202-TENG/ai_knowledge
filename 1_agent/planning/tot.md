# Tree of Thoughts (ToT)

讓 AI 能夠探索多条推理路徑的框架，類似人類的「三思而後行」，在做出決定前評估多種可能性。

---

## 1. 什麼是？

### 深度定義

**Tree of Thoughts (ToT)** 是一種**結構化推理框架**，將單一路徑擴展為多路徑搜索樹：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ToT 推理架構                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  傳統 Chain vs ToT:                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Chain of Thought (鏈狀):                                   │   │
│  │                                                              │   │
│  │    輸入 ──▶ [思考1] ──▶ [思考2] ──▶ [思考3] ──▶ 輸出        │   │
│  │                  ↓                                           │   │
│  │              失敗? → 無法恢復                                │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Tree of Thoughts (樹狀):                                   │   │
│  │                                                              │   │
│  │                      ┌─ [思考1.1] ──▶ 輸出1 ✓              │   │
│  │                     /                                        │   │
│  │    輸入 ──▶ [思考1] ┤                                        │   │
│  │                     \                                        │   │
│  │                      ├─ [思考1.2] ──▶ 輸出2                  │   │
│  │                       \                                      │   │
│  │                        └─ [思考1.3] ──▶ 輸出3                 │   │
│  │                                                              │   │
│  │    優勢: 探索多條路徑，失敗可回溯                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ToT 核心概念:                                                        │
│  - Thought (思考): 推理过程中的中间状态                              │
│  - State (狀態): 當前搜索樹中的節點                                   │
│  - Evaluation (評估): 判斷狀態是否有希望                             │
│  - Search (搜索): BFS/DFS/MCTS 等策略                               │
│                                                                      │
│  與人類思維對比:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  直覺反應: 快速、單路徑 (≈ Chain of Thought)                 │   │
│  │  深思熟慮: 評估多種可能、權衡利弊 (≈ ToT)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何重要**:
1. **避免局部最優**: 多路徑搜索找到全局最優解
2. **錯誤恢復**: 路徑失敗可嘗試其他路徑
3. **創造性解決**: 探索非常規解決方案
4. **可解释性**: 清晰的推理路徑追踪

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

## 8. 數學形式化

### ToT 形式化定義

```
ToT 可定義為五元組 (S, A, G, E, P):

- S: 狀態空間
  S = {s₀, s₁, s₂, ...}  // 所有可能的推理狀態

- A: 動作空間
  A(s) = {a₁, a₂, ..., aₖ}  // 從狀態 s 可執行的動作

- G: 目標狀態
  G ⊆ S  // 達成任務的狀態集合

- E: 評估函數
  E: S × policy → [0, 1]  // 評估狀態的價值

- P: 搜索策略
  P ∈ {BFS, DFS, Best-first, MCTS}
```

### 搜索複雜度

| 參數 | 表達式 | 說明 |
|------|--------|------|
| 分支因子 | `b = avg(|A(s)|)` | 每個狀態的平均動作數 |
| 深度 | `d` | 搜索深度 |
| 總節點數 | `O(b^d)` | 搜索樹大小 |
| BFS 記憶體 | `O(b^d)` | 需要儲存所有節點 |
| DFS 記憶體 | `O(d)` | 只需要棧 |

### 剪枝策略

```python
class ToTPruner:
    """ToT 剪枝策略"""

    def __init__(self, max_width=5, min_score_threshold=0.3):
        self.max_width = max_width
        self.min_score = min_score_threshold

    def prune(self, states_with_scores):
        """剪枝邏輯"""

        # 1. 過濾低分狀態
        filtered = [s for s, score in states_with_scores
                   if score >= self.min_score]

        # 2. 限制寬度
        if len(filtered) > self.max_width:
            # 保留 top-k
            filtered = sorted(filtered,
                           key=lambda x: x[1],
                           reverse=True)[:self.max_width]

        # 3. 去重
        return self._deduplicate(filtered)

    def _deduplicate(self, states):
        """去除重複狀態"""
        seen = set()
        unique = []
        for state in states:
            state_key = self._hash_state(state)
            if state_key not in seen:
                seen.add(state_key)
                unique.append(state)
        return unique
```

---

## 9. MCTS 變體

### Monte Carlo Tree Search

MCTS 結合隨機模擬與樹搜索，是 ToT 的進階版本：

```
四個階段:
1. Selection: 選擇最有希望的節點
2. Expansion: 擴展節點
3. Simulation: 隨機模擬到目標
4. Backpropagation: 更新節點分數
```

### MCTS ToT 實作

```python
class MCTSToT:
    def __init__(self, exploration_constant=1.41):
        self.C = exploration_constant  # UCB 探索常數
        self.tree = {}  # state -> Node

    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visit_count = 0
            self.total_score = 0

        def ucb_score(self):
            if self.visit_count == 0:
                return float('inf')
            exploitation = self.total_score / self.visit_count
            exploration = self.C * np.sqrt(
                np.log(self.parent.visit_count) / self.visit_count
            )
            return exploitation + exploration

    def select(self, node):
        """Selection: 選擇 UCB 分數最高的子節點"""
        while node.children:
            node = max(node.children, key=lambda x: x.ucb_score())
        return node

    def expand(self, node, thought):
        """Expansion: 添加新節點"""
        new_node = self.Node(thought, node)
        node.children.append(new_node)
        return new_node

    def simulate(self, state):
        """Simulation: 隨機模擬到目標"""
        # 隨機生成後續步驟
        # 返回模擬結果
        return self._random_rollout(state)

    def backpropagate(self, node, score):
        """Backpropagation: 更新路徑上所有節點"""
        while node:
            node.visit_count += 1
            node.total_score += score
            node = node.parent

    def search(self, initial_state, iterations=1000):
        """完整 MCTS 搜索"""
        root = self.Node(initial_state)
        self.tree[initial_state] = root

        for _ in range(iterations):
            # 1. Selection
            node = self.select(root)

            # 2. Expansion
            if not is_goal(node.state):
                thoughts = generate_thoughts(node.state)
                for thought in thoughts:
                    self.expand(node, thought)
                node = node.children[0]  # 選擇第一個子節點

            # 3. Simulation
            score = self.simulate(node.state)

            # 4. Backpropagation
            self.backpropagate(node, score)

        return self.get_best_path(root)
```

### MCTS vs 傳統 ToT

| 特性 | 傳統 ToT | MCTS ToT |
|------|----------|----------|
| 搜索策略 | 確定性 | 隨機性 |
| 探索/利用 | 需手動平衡 | 自動平衡 |
| 計算需求 | 高 | 中等 |
| 實現難度 | 低 | 高 |

---

## 10. ToT + Function Calling

### 整合架構

```
ToT 搜索 ─────────────────────────────────────────┐
                                                │
  每個狀態 ──► 評估 (E)                           │
               │                                  │
               ▼                                  │
         需要外部驗證？ ──► Function Calling      │
                              │                  │
                              ▼                  │
                        API 回傳結果 ──────────────┘
```

### 實作範例

```python
class ToTWithFunctionCalling:
    def __init__(self, tools):
        self.tools = tools

    def evaluate_state_with_tools(self, state):
        """使用 Function Calling 評估狀態"""

        # 根據狀態類型選擇工具
        tool_choice = self._select_evaluation_tool(state)

        if tool_choice:
            # 調用工具驗證
            result = self._call_tool(tool_choice, state)

            # 根據結果計算分數
            return self._compute_score(result)
        else:
            # 使用 LLM 評估
            return self._llm_evaluate(state)

    def _select_evaluation_tool(self, state):
        """根據狀態選擇評估工具"""

        if state.type == "code":
            return "execute_code"
        elif state.type == "math":
            return "calculate"
        elif state.type == "search":
            return "web_search"
        return None

    def _call_tool(self, tool_name, state):
        """調用工具"""
        if tool_name == "execute_code":
            return self.tools["execute_code"](state.content)
        elif tool_name == "calculate":
            return self.tools["calculate"](state.expression)
        # ...
```

### 工具選擇策略

| 狀態類型 | 評估工具 | 優勢 |
|----------|----------|------|
| 數學計算 | Python interpreter | 精確驗證 |
| 程式碼 | Code compiler | 語法錯誤檢測 |
| 事實查詢 | Web search | 即時資訊 |
| 天氣/股票 | Weather/Stock API | 準確數據 |

---

## 11. 效能優化

### Token 優化策略

```python
class ToTTokenOptimizer:
    def __init__(self, max_tokens_per_thought=200):
        self.max_tokens = max_tokens_per_thought

    def compress_thought(self, thought):
        """壓縮思考內容"""

        # 移除冗餘詞彙
        compressed = self._remove_redundancy(thought)

        # 截斷過長內容
        if len(compressed) > self.max_tokens:
            compressed = compressed[:self.max_tokens] + "..."

        return compressed

    def _remove_redundancy(self, text):
        """移除冗餘"""
        # 移除重複詞彙
        words = text.split()
        seen = set()
        result = []
        for word in words:
            if word.lower() not in seen or len(word) > 3:
                result.append(word)
                seen.add(word.lower())
        return " ".join(result)
```

### 批量評估

```python
async def batch_evaluate(states, evaluator, max_concurrent=10):
    """批量評估多個狀態"""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_with_limit(state):
        async with semaphore:
            return await evaluator(state)

    tasks = [eval_with_limit(s) for s in states]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 過濾錯誤
    return [r for r in results if not isinstance(r, Exception)]
```

### 緩存策略

```python
from functools import lru_cache

class CachedToT:
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl

    def get_cached_evaluation(self, state_hash):
        """緩存評估結果"""
        if state_hash in self.cache:
            result, timestamp = self.cache[state_hash]
            if time.time() - timestamp < self.ttl:
                return result
        return None
```

---

## 12. 監控與診斷

### 追蹤指標

```python
class ToTMonitor:
    def __init__(self):
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "total_nodes": [],
            "path_lengths": [],
            "evaluation_scores": []
        }

    def record_search(self, nodes, path_length, success):
        self.stats["total_searches"] += 1
        if success:
            self.stats["successful_searches"] += 1
        self.stats["total_nodes"].append(nodes)
        self.stats["path_lengths"].append(path_length)

    def get_metrics(self):
        return {
            "success_rate": self.stats["successful_searches"] /
                          self.stats["total_searches"],
            "avg_nodes": np.mean(self.stats["total_nodes"]),
            "avg_path_length": np.mean(self.stats["path_lengths"]),
            "p95_path_length": np.percentile(self.stats["path_lengths"], 95)
        }
```

### 診斷問題

| 問題 | 徵兆 | 解決方案 |
|------|------|----------|
| 搜索空間爆炸 | 節點數指數增長 | 增加剪枝 |
| 局部最優 | 分數停滯不前 | 增加探索 |
| 評估過慢 | 單次評估 > 10s | 批量評估 + 緩存 |
| 記憶體不足 | OOM 錯誤 | 限制節點數 + 定期清理 |

---

## 13. 相關主題

| 技術 | 關係 |
|------|------|
| **ReAct** | ToT 的基礎，單路徑版本 |
| **Chain of Thought** | ToT 的簡化版，鏈狀思維 |
| **MCTS** | ToT 的進階搜索算法 |
| **Planning** | ToT 是規劃問題的核心技術 |

---

## 延伸閱讀

- [Tree of Thoughts Paper (arxiv)](https://arxiv.org/abs/2305.08291)
- [LangChain ToT](https://python.langchain.com/docs/modules/agents/agent_types/tot)
- [Google ToT Implementation](https://github.com/google-research/google-research/tree/master/tot)
- [MCTS Survey](https://arxiv.org/abs/1909.13328)
- [Self-Consistency in ToT](https://arxiv.org/abs/2203.11171)