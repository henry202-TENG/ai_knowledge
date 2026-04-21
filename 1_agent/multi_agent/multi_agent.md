# Multi-Agent Systems

多代理系統的設計與實現，實現複雜任務的協作處理。

---

## 1. 什麼是？

### 簡單範例

```
單一 Agent:
  User → Agent → 處理所有事情
  缺點: 能力有限，容易犯錯

多 Agent 協作:
  User → [Planner] → [Coder] → [Reviewer] → User
           ↓
        [Tester]
  優勢: 專業分工、錯誤檢查、質量提升
```

---

## 2. 架構模式

### 層次式架構

```python
class HierarchicalAgents:
    """層次式多 Agent"""

    def __init__(self):
        # 高層規劃
        self.planner = PlannerAgent()

        # 中層協調
        self.coordinator = CoordinatorAgent()

        # 基層執行
        self.workers = {
            "coder": WorkerAgent("coder"),
            "researcher": WorkerAgent("researcher"),
            "writer": WorkerAgent("writer")
        }

    def process(self, task):
        """層次處理"""

        # 1. 規劃
        plan = self.planner.create_plan(task)

        # 2. 協調
        subtasks = self.coordinator.decompose(plan)

        # 3. 執行
        results = {}
        for subtask in subtasks:
            worker = self.workers[subtask["type"]]
            results[subtask["id"]] = worker.execute(subtask)

        # 4. 整合
        return self.coordinator.integrate(results)
```

### 網狀架構

```python
class MeshAgents:
    """網狀對等架構"""

    def __init__(self):
        self.agents = {}
        self.message_queue = {}

    def register(self, agent_id, agent):
        """註冊 Agent"""
        self.agents[agent_id] = agent
        self.message_queue[agent_id] = []

    def broadcast(self, sender, message):
        """廣播訊息"""

        for agent_id in self.agents:
            if agent_id != sender:
                self.message_queue[agent_id].append({
                    "from": sender,
                    "content": message,
                    "timestamp": time.time()
                })

    def process_messages(self, agent_id):
        """處理訊息"""

        messages = self.message_queue[agent_id]
        self.message_queue[agent_id] = []

        return messages
```

---

## 3. 通信機制

### 消息傳遞

```python
class MessagePassing:
    """消息傳遞系統"""

    def __init__(self):
        self.channels = {}

    def create_channel(self, channel_id, participants):
        """創建通道"""
        self.channels[channel_id] = {
            "participants": participants,
            "messages": []
        }

    def send(self, channel_id, sender, content):
        """發送訊息"""

        message = {
            "sender": sender,
            "content": content,
            "timestamp": time.time()
        }

        self.channels[channel_id]["messages"].append(message)

    def receive(self, channel_id, receiver):
        """接收訊息"""

        messages = self.channels[channel_id]["messages"]

        # 過濾給自己的訊息
        relevant = [
            m for m in messages
            if receiver in self.channels[channel_id]["participants"]
        ]

        return relevant
```

### 共享狀態

```python
class SharedState:
    """共享狀態管理"""

    def __init__(self):
        self.state = {}
        self.locks = {}

    def read(self, key):
        """讀取"""
        return self.state.get(key)

    def write(self, key, value):
        """寫入 (簡單實現)"""

        if key not in self.locks:
            self.locks[key] = asyncio.Lock()

        # 使用鎖
        async with self.locks[key]:
            self.state[key] = value

    def update(self, key, updater):
        """原子更新"""

        async with self.locks[key]:
            old_value = self.state.get(key)
            new_value = updater(old_value)
            self.state[key] = new_value

        return new_value
```

---

## 4. 協調模式

### 協作式

```python
class CollaborativeCoordinator:
    """協作式協調"""

    def __init__(self, agents):
        self.agents = agents

    async def solve_collaboratively(self, problem):
        """協作解決問題"""

        # 初始提案
        proposals = await self._gather_proposals(problem)

        # 迭代討論
        for round in range(5):
            # 評估提案
            evaluations = await self._evaluate(proposals)

            # 改進提案
            proposals = await self._improve(
                proposals,
                evaluations
            )

        # 選擇最佳方案
        best = self._select_best(proposals)

        return best

    async def _gather_proposals(self, problem):
        """收集提案"""

        tasks = [
            agent.propose(problem)
            for agent in self.agents
        ]

        return await asyncio.gather(*tasks)
```

### 竞争式

```python
class CompetitiveCoordinator:
    """競爭式協調"""

    def __init__(self, agents):
        self.agents = agents

    async def solve_competitive(self, problem):
        """競爭解決"""

        # 同時解決
        tasks = [
            agent.solve(problem)
            for agent in self.agents
        ]

        results = await asyncio.gather(*tasks)

        # 評估結果
        evaluations = await self._evaluate_all(results)

        # 選擇最佳
        best = max(evaluations, key=lambda x: x["score"])

        return best["result"]
```

### 主持人模式

```python
class ModeratorCoordinator:
    """主持人協調"""

    def __init__(self, moderator, participants):
        self.moderator = moderator
        self.participants = participants

    async def discuss(self, topic):
        """討論流程"""

        # 主持人開題
        opening = await self.moderator.introduce(topic)

        # 參與者發言
        for _ in range(3):  # 3 輪
            for participant in self.participants:
                response = await participant.respond(opening)

                # 主持人總結
                summary = await self.moderator.summarize(response)

                opening = summary

        # 達成結論
        conclusion = await self.moderator.conclude()

        return conclusion
```

---

## 5. 衝突解決

### 投票機制

```python
class VotingResolver:
    """投票解決衝突"""

    def __init__(self, agents):
        self.agents = agents

    async def resolve_via_vote(self, options):
        """投票解決"""

        # 每個 agent 投票
        votes = {}
        for agent in self.agents:
            vote = await agent.vote(options)
            votes[agent.id] = vote

        # 計數
        counts = {}
        for vote in votes.values():
            counts[vote] = counts.get(vote, 0) + 1

        # 最多的獲勝
        winner = max(counts.items(), key=lambda x: x[1])

        return winner[0]
```

### 協商機制

```python
class NegotiationResolver:
    """協商解決衝突"""

    async def negotiate(self, agents, issue):
        """協商過程"""

        positions = {}
        for agent in agents:
            positions[agent.id] = await agent.get_position(issue)

        # 找到共同點
        common = self._find_common(positions)

        if common:
            return common

        # 讓步
        concessions = await self._make_concessions(positions)

        return concessions

    async def _make_concessions(self, positions):
        """讓步協商"""

        # 逐步讓步直到達成一致
        pass
```

---

## 6. 決策制定

### 共識形成

```python
class ConsensusBuilder:
    """共識構建"""

    def __init__(self, agents):
        self.agents = agents

    async def build_consensus(self, proposal):
        """構建共識"""

        rounds = 0
        max_rounds = 5

        while rounds < max_rounds:
            # 收集意見
            opinions = await self._collect_opinions(proposal)

            # 檢查是否達成共識
            if self._is_consensus(opinions):
                return proposal

            # 修改提案
            proposal = await self._revise(proposal, opinions)

            rounds += 1

        # 返回最終提案 (可能不是全票通過)
        return proposal

    def _is_consensus(self, opinions):
        """檢查共識"""
        # 至少 80% 同意
        agreement = sum(1 for o in opinions if o["agree"])
        return agreement / len(opinions) >= 0.8
```

---

## 7. 監控與調試

### 追蹤系統

```python
class MultiAgentTracer:
    """多 Agent 追蹤"""

    def __init__(self):
        self.traces = []

    def trace_message(self, from_agent, to_agent, content):
        """追蹤訊息"""

        self.traces.append({
            "from": from_agent,
            "to": to_agent,
            "content": content[:100],  # 截斷
            "timestamp": time.time()
        })

    def get_trace(self, conversation_id):
        """獲取追蹤"""

        return [
            t for t in self.traces
            if t.get("conversation_id") == conversation_id
        ]
```

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **AutoGen** | 多 Agent 框架 |
| **LangChain** | Agent 協調 |
| **Function Calling** | 工具使用 |

---

## 延伸閱讀

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Multi-Agent Survey](https://arxiv.org/abs/2308.11432)
- [CAMEL](https://github.com/camel-ai/camel)