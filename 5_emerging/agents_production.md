# AI Agents in Production

將 AI Agent 部署到生產環境的最佳實踐與工程考量。

---

## 1. 什麼是？

### 深度定義

**AI Agents in Production** 是將 AI Agent 系統**部署到生產環境**的工程實踐：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    生產級 Agent 架構                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  生產環境 vs 開發環境:                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  開發環境:                                                    │   │
│  │    - 單一用戶                                                │   │
│  │    - 調試模式                                                │   │
│  │    - 錯誤可接受                                              │   │
│  │    - 性能不關鍵                                              │   │
│  │                                                              │   │
│  │  生產環境:                                                    │   │
│  │    - 大量並發用戶                                            │   │
│  │    - 7x24 可用                                               │   │
│  │    - 零錯誤目標                                               │   │
│  │    - 延遲敏感                                                │   │
│  │    - 成本敏感                                                 │   │
│  │    - 安全合規                                                 │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  生產級組件:                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  1. 可靠性                                                    │   │
│  │     - 錯誤處理: 重試、降級、超時                            │   │
│  │     - 健康檢查: 定期檢查組件狀態                            │   │
│  │     - 故障轉移: 自動切換到備用系統                          │   │
│  │                                                              │   │
│  │  2. 安全性                                                    │   │
│  │     - 輸入驗證: 長度、內容、格式                            │   │
│  │     - 工具權限: RBAC 權限控制                                │   │
│  │     - 審計日誌: 完整的操作記錄                               │   │
│  │     - 數據隔離: 用戶數據分離                                 │   │
│  │                                                              │   │
│  │  3. 性能                                                      │   │
│  │     - 請求緩存: 減少重複調用                                 │   │
│  │     - 請求合併: 批量處理                                    │   │
│  │     - 限流: 防止濫用                                         │   │
│  │     - - 排隊: 負載管理                                        │   │
│  │                                                              │   │
│  │  4. 可觀測性                                                  │   │
│  │     - 分散式追蹤: 追蹤每個步驟                               │   │
│  │     - 結構化日誌: 便於分析                                   │   │
│  │     - 指標: 性能、錯誤、成本                                 │   │
│  │     - 告警: 及時發現問題                                     │   │
│  │                                                              │   │
│  │  5. 監控                                                      │   │
│  │     - 儀表板: 實時狀態                                       │   │
│  │     - SLO: 服務水平目標                                      │   │
│  │     - 報告: 定期總結                                         │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  核心挑戰:                                                           │
│  1. 非確定性: Agent 行為不可完全預測                                │
│  2. 長期運行: 多步驟任務可能跨越很長時間                            │
│  3. 外部依賴: 工具、API 可能失敗                                    │
│  4. 成本控制: 大量 LLM 調用成本高昂                                │
│  5. 安全性: Agent 可能有意外行為                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何重要**:
1. **穩定服務**: 確保 Agent 7x24 可靠運行
2. **成本控制**: 優化資源使用和 LLM 調用
3. **安全合規**: 滿足企業安全要求
4. **持續監控**: 及時發現和解決問題

---

## 2. 架構設計

### Agent 系統架構

```python
class ProductionAgent:
    """生產級 Agent"""

    def __init__(self):
        # 核心組件
        self.llm = self._init_llm()
        self.tools = self._init_tools()
        self.memory = self._init_memory()

        # 基礎設施
        self.cache = self._init_cache()
        self.monitor = self._init_monitor()
        self.rate_limiter = self._init_rate_limiter()

    def process(self, request):
        """處理請求"""

        # 1. 驗證
        self._validate_request(request)

        # 2. 限流
        if not self.rate_limiter.allow(request.user_id):
            raise RateLimitError()

        # 3. 緩存檢查
        cached = self.cache.get(request)
        if cached:
            return cached

        # 4. Agent 執行
        response = self._execute(request)

        # 5. 緩存
        self.cache.set(request, response)

        # 6. 監控
        self.monitor.record(request, response)

        return response
```

---

## 2. 可靠性設計

### 錯誤處理

```python
class AgentErrorHandler:
    """Agent 錯誤處理"""

    ERROR_STRATEGIES = {
        "llm_timeout": "retry_with_backoff",
        "invalid_tool_args": "validate_and_retry",
        "tool_execution_failed": "try_alternative_tool",
        "max_iterations": "return_partial_result",
        "rate_limit": "queue_and_wait"
    }

    def handle(self, error, context):
        """處理錯誤"""

        error_type = self._classify_error(error)
        strategy = self.ERROR_STRATEGIES.get(error_type, "return_error")

        if strategy == "retry_with_backoff":
            return self._retry_with_backoff(context)
        elif strategy == "return_partial_result":
            return self._return_partial(context)

    def _retry_with_backoff(self, context):
        """指數退避重試"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                return context["agent"].execute(context["task"])
            except Exception as e:
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        # 超過重試次數
        return self._fallback_response(context)
```

### 健康檢查

```python
class AgentHealthCheck:
    """Agent 健康檢查"""

    def __init__(self, agent):
        self.agent = agent

    async def health_check(self):
        """執行健康檢查"""

        checks = {
            "llm": await self._check_llm(),
            "tools": self._check_tools(),
            "memory": await self._check_memory(),
            "cache": self._check_cache()
        }

        # 計算健康分數
        health_score = sum(
            1 for result in checks.values()
            if result["status"] == "healthy"
        ) / len(checks)

        return {
            "healthy": health_score > 0.8,
            "score": health_score,
            "checks": checks
        }

    async def _check_llm(self):
        """檢查 LLM"""

        try:
            response = await self.agent.llm.agenerate(
                ["health check"]
            )

            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
```

---

## 3. 安全性

### 輸入驗證

```python
class AgentInputValidator:
    """Agent 輸入驗證"""

    def __init__(self):
        self.validators = [
            self._validate_length,
            self._validate_content,
            self._validate_rate,
        ]

    def validate(self, input_data):
        """驗證輸入"""

        for validator in self.validators:
            result = validator(input_data)
            if not result["valid"]:
                return result

        return {"valid": True}

    def _validate_length(self, data):
        """長度驗證"""

        max_length = 10000  # characters

        if len(data.get("prompt", "")) > max_length:
            return {
                "valid": False,
                "error": "Input exceeds maximum length"
            }

        return {"valid": True}

    def _validate_content(self, data):
        """內容驗證"""

        # 檢查敏感詞
        sensitive_patterns = [
            "password",
            "secret",
            "api_key"
        ]

        prompt = data.get("prompt", "").lower()

        for pattern in sensitive_patterns:
            if pattern in prompt:
                return {
                    "valid": False,
                    "error": "Contains sensitive content"
                }

        return {"valid": True}
```

### 工具權限

```python
class ToolPermissionManager:
    """工具權限管理"""

    def __init__(self):
        # 用戶角色權限
        self.role_permissions = {
            "admin": ["read", "write", "execute", "delete"],
            "user": ["read", "execute"],
            "guest": ["read"]
        }

    def check_permission(self, user_role, tool_name):
        """檢查權限"""

        allowed_tools = self.role_permissions.get(user_role, [])

        if tool_name not in allowed_tools:
            raise PermissionDenied(
                f"User role '{user_role}' cannot access tool '{tool_name}'"
            )

        return True
```

---

## 4. 可觀測性

### 追蹤

```python
import opentelemetry as otel
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class AgentTracer:
    """Agent 追蹤"""

    def __init__(self):
        self.tracer = tracer

    def trace_execution(self, agent, task):
        """追蹤執行"""

        with self.tracer.start_as_current_span(
            "agent_execution",
            attributes={
                "task.type": task.get("type"),
                "task.id": task.get("id")
            }
        ) as span:
            # 追蹤每個步驟
            span.set_attribute("llm.start_time", time.time())

            result = agent.execute(task)

            span.set_attribute("llm.end_time", time.time())
            span.set_attribute("result.status", result["status"])

            return result
```

### 日誌

```python
import structlog

logger = structlog.get_logger()

class AgentLogger:
    """Agent 日誌"""

    def log_request(self, request):
        """記錄請求"""

        logger.info(
            "agent_request",
            user_id=request.user_id,
            task_type=request.type,
            timestamp=time.time()
        )

    def log_execution(self, agent_name, step, result):
        """記錄執行"""

        logger.info(
            "agent_step",
            agent=agent_name,
            step=step,
            success=result.get("success", False),
            duration=result.get("duration", 0)
        )

    def log_error(self, error, context):
        """記錄錯誤"""

        logger.error(
            "agent_error",
            error_type=type(error).__name__,
            message=str(error),
            context=context
        )
```

---

## 5. 性能優化

### 請求優化

```python
class RequestOptimizer:
    """請求優化"""

    def __init__(self):
        self.cache = {}

    def optimize_request(self, request):
        """優化請求"""

        # 合併相似請求
        similar = self._find_similar_request(request)

        if similar and similar["result"] is not None:
            return {
                "cached": True,
                "result": similar["result"]
            }

        # 優化 prompt
        optimized = self._optimize_prompt(request)

        return {"cached": False, "prompt": optimized}

    def _optimize_prompt(self, request):
        """優化 prompt"""

        prompt = request["prompt"]

        # 移除冗餘
        prompt = self._remove_redundancy(prompt)

        # 壓縮
        if len(prompt) > 4000:
            prompt = self._compress(prompt)

        return prompt
```

---

## 6. 監控儀表板

### 關鍵指標

```python
DASHBOARD_METRICS = {
    "performance": {
        "requests_per_second": "每秒請求數",
        "latency_p50": "延遲 P50",
        "latency_p95": "延遲 P95",
        "latency_p99": "延遲 P99"
    },
    "reliability": {
        "success_rate": "成功率",
        "error_rate": "錯誤率",
        "retry_rate": "重試率",
        "timeout_rate": "超時率"
    },
    "usage": {
        "total_requests": "總請求數",
        "total_tokens": "總 tokens",
        "unique_users": "唯一用戶"
    },
    "cost": {
        "cost_per_request": "每次請求成本",
        "cost_per_user": "每用戶成本"
    }
}
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **LangChain** | Agent 框架 |
| **Observability** | 可觀測性 |
| **Kubernetes** | 部署 |

---

## 延伸閱讀

- [LLM Observability](https://www.honeycomb.io/)
- [Agent Best Practices](https://www.anthropic.com/)
- [Production ML](https://mlops.org/)