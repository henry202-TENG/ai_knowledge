# Agent Planning

AI Agent 的規劃與推理能力，支援複雜任務的分解和執行。

---

## 1. 什麼是？

### 簡單範例

```
沒有規劃:
  用戶: 幫我寫一個網站
  AI: (不知道從哪裡開始)

有規劃:
  用戶: 幫我寫一個網站
  AI: 
    步驟 1: 需求分析
    步驟 2: 設計架構
    步驟 3: 實作前端
    步驟 4: 實作後端
    步驟 5: 測試上線
```

---

## 2. 規劃方法

### Chain of Thought

```python
class ChainOfThoughtAgent:
    """思維鏈 Agent"""

    def __init__(self, llm):
        self.llm = llm

    def solve(self, problem):
        """解決問題"""

        # 逐步思考
        prompt = f"""讓我們一步步思考這個問題: {problem}

步驟 1: 理解問題
[分析問題的各個方面]

步驟 2: 制定計劃
[列出解決步驟]

步驟 3: 執行
[逐步實現]

步驟 4: 驗證
[檢查結果]

答案:"""

        return self.llm.generate(prompt)
```

### Tree of Thoughts

```python
class TreeOfThoughtsAgent:
    """思維樹 Agent"""

    def __init__(self, llm, num_thoughts=3, depth=3):
        self.llm = llm
        self.num_thoughts = num_thoughts
        self.depth = depth

    def solve(self, problem):
        """樹狀搜索解決"""

        # 根節點
        root = {
            "thought": problem,
            "children": []
        }

        # BFS 擴展
        queue = [root]
        best_solution = None

        for _ in range(self.depth):
            level_nodes = []

            for node in queue:
                # 生成多個思考分支
                thoughts = self._generate_thoughts(
                    node["thought"]
                )

                for thought in thoughts:
                    child = {
                        "thought": thought,
                        "children": [],
                        "score": self._evaluate(thought)
                    }
                    node["children"].append(child)
                    level_nodes.append(child)

            # 評估並選擇 top-k
            level_nodes.sort(key=lambda x: x["score"], reverse=True)
            queue = level_nodes[:self.num_thoughts]

            # 更新最佳解
            if queue[0]["score"] > (best_solution or {}).get("score", 0):
                best_solution = queue[0]

        return best_solution["thought"]

    def _generate_thoughts(self, current):
        """生成思考分支"""

        prompt = f"基於當前思考 '{current}'，產生出不同的解決方向:"

        response = self.llm.generate(prompt)
        thoughts = response.split("\n")

        return thoughts[:self.num_thoughts]

    def _evaluate(self, thought):
        """評估思考"""
        # 使用 LLM 評估品質
        pass
```

### ReAct

```python
class ReActAgent:
    """ReAct Agent - 推理 + 行動"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def solve(self, problem):
        """解決問題"""

        # 初始化
        observation = ""
        thought = "我需要解決這個問題"

        for step in range(10):
            # 1. 推理
            reasoning = self._reason(problem, thought, observation)

            # 2. 行動
            action = self._select_action(reasoning)

            # 3. 執行
            if action["type"] == "tool":
                observation = self._execute_tool(
                    action["name"],
                    action["args"]
                )
            else:
                # 完成
                return action["result"]

            # 更新
            thought = reasoning

        return "無法完成"

    def _reason(self, problem, thought, observation):
        """推理步驟"""

        prompt = f"""問題: {problem}
當前思考: {thought}
觀察結果: {observation}

讓我推理下一步應該做什麼:"""

        return self.llm.generate(prompt)
```

---

## 3. 任務分解

### 自動分解

```python
class TaskDecomposer:
    """任務分解器"""

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, task):
        """分解複雜任務"""

        prompt = f"""將以下任務分解成可執行的子任務:

任務: {task}

輸出格式:
1. [子任務 1]
2. [子任務 2]
3. [子任務 3]
..."""

        response = self.llm.generate(prompt)

        # 解析
        subtasks = []
        for line in response.split("\n"):
            if line.strip():
                subtasks.append({
                    "description": line.strip(),
                    "status": "pending"
                })

        return subtasks

    def decompose_recursive(self, task, max_depth=3):
        """遞歸分解"""

        if max_depth == 0:
            return [task]

        subtasks = self.decompose(task)

        if len(subtasks) == 1:
            return subtasks

        # 遞歸分解每個子任務
        result = []
        for subtask in subtasks:
            result.extend(
                self.decompose_recursive(
                    subtask["description"],
                    max_depth - 1
                )
            )

        return result
```

### 依賴管理

```python
class DependencyManager:
    """依賴管理"""

    def __init__(self):
        self.graph = {}

    def add_task(self, task_id, task, depends_on=None):
        """添加任務及依賴"""

        self.graph[task_id] = {
            "task": task,
            "depends_on": depends_on or [],
            "status": "pending"
        }

    def get_execution_order(self):
        """獲取執行順序"""

        # 拓撲排序
        in_degree = {
            task_id: len(info["depends_on"])
            for task_id, info in self.graph.items()
        }

        queue = [
            task_id for task_id, degree in in_degree.items()
            if degree == 0
        ]

        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)

            # 更新依賴
            for task_id, info in self.graph.items():
                if current in info["depends_on"]:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)

        return order
```

---

## 4. 反思機制

### 自我反思

```python
class ReflectiveAgent:
    """反思 Agent"""

    def __init__(self, llm):
        self.llm = llm

    def solve_with_reflection(self, problem):
        """帶反思的解決"""

        # 初始嘗試
        solution = self._attempt_solve(problem)

        # 反思
        for _ in range(3):
            feedback = self._reflect(solution, problem)

            if feedback["improvement_needed"]:
                # 改進
                solution = self._improve(
                    solution,
                    feedback["suggestions"]
                )
            else:
                break

        return solution

    def _reflect(self, solution, problem):
        """反思解決方案"""

        prompt = f"""評估這個解決方案:

問題: {problem}
解決方案: {solution}

評估:
1. 解決方案是否正確？
2. 有哪些可以改進的地方？
3. 需要重試嗎？

輸出:
- 改進需求: 是/否
- 建議:
"""

        response = self.llm.generate(prompt)

        return self._parse_reflection(response)
```

---

## 5. 計劃執行

### 執行引擎

```python
class PlanExecutor:
    """計劃執行器"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    async def execute(self, plan):
        """執行計劃"""

        results = []

        for step in plan["steps"]:
            # 執行每個步驟
            result = await self._execute_step(step)

            results.append({
                "step": step["id"],
                "result": result,
                "status": "success" if result else "failed"
            })

            # 檢查依賴
            if not result and step.get("critical"):
                return {
                    "status": "failed",
                    "failed_step": step["id"],
                    "results": results
                }

        return {
            "status": "success",
            "results": results
        }

    async def _execute_step(self, step):
        """執行單個步驟"""

        if step["type"] == "llm":
            return self.llm.generate(step["prompt"])

        elif step["type"] == "tool":
            return self._execute_tool(
                step["tool"],
                step["args"]
            )
```

---

## 6. 監控與調整

### 動態調整

```python
class AdaptivePlanner:
    """自適應規劃"""

    def __init__(self, llm):
        self.llm = llm

    def plan_with_monitoring(self, task):
        """監控規劃執行"""

        plan = self.create_initial_plan(task)
        execution_log = []

        for step_idx in range(len(plan["steps"])):
            step = plan["steps"][step_idx]

            # 執行
            result = self._execute_step(step)

            # 記錄
            execution_log.append({
                "step": step,
                "result": result
            })

            # 評估
            if not self._is_successful(result, step):
                # 重新規劃
                plan = self._replan(
                    task,
                    execution_log
                )

                # 重置執行
                execution_log = []
                break

        return {
            "plan": plan,
            "execution_log": execution_log
        }
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **ReAct** | 推理+行動 |
| **Tool Use** | 執行計劃 |
| **Memory** | 記憶上下文 |

---

## 延伸閱讀

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [ToT Paper](https://arxiv.org/abs/2305.10601)
- [Plan+Execute](https://arxiv.org/abs/2308.10186)