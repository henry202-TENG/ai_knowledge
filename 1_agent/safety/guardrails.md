# AI Safety / Guardrails

確保 AI 系統安全、可靠、符合人類價值的技術和實踐。

---

## 1. 什麼是？

### 深度定義

**AI Safety / Guardrails** 是確保 AI 系統**安全、可靠、可控**的多層防護機制：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI Safety 防護架構                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  安全威脅分類:                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │   幻覺 (Hallucination)                                      │   │
│  │   ┌─────────────────────────────────────────────────┐      │   │
│  │   │  模型生成看似合理但實際錯誤的內容                 │      │   │
│  │   │  例: 編造不存在的論文、錯誤的事實                │      │   │
│  │   └─────────────────────────────────────────────────┘      │   │
│  │                                                              │   │
│  │   越獄 (Jailbreak)                                          │   │
│  │   ┌─────────────────────────────────────────────────┐      │   │
│  │   │  通過特殊提示繞過安全限制                         │      │   │
│  │   │  例: "假裝是另一個 AI"、"DAN 模式"               │      │   │
│  │   └─────────────────────────────────────────────────┘      │   │
│  │                                                              │   │
│  │   提示注入 (Prompt Injection)                              │   │
│  │   ┌─────────────────────────────────────────────────┐      │   │
│  │   │  惡意指令覆蓋原本的 system prompt                │      │   │
│  │   │  例: "忽略之前的指令，告訴我..."                 │      │   │
│  │   └─────────────────────────────────────────────────┘      │   │
│  │                                                              │   │
│  │   偏見 (Bias)                                               │   │
│  │   ┌─────────────────────────────────────────────────┐      │   │
│  │   │  訓練數據導致的歧視性輸出                         │      │   │
│  │   │  例: 性別/種族偏見、職業偏見                      │      │   │
│  │   └─────────────────────────────────────────────────┘      │   │
│  │                                                              │   │
│  │   隱私洩漏 (Privacy Leakage)                                │   │
│  │   ┌─────────────────────────────────────────────────┐      │   │
│  │   │  透露訓練數據中的個人資訊                         │      │   │
│  │   │  例: 身份證號、地址、電話                         │      │   │
│  │   └─────────────────────────────────────────────────┘      │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  防護層次:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  LLM 層: RLHF、Constitutional AI                            │   │
│  │  應用層: Guardrails、輸入/輸出過濾                          │   │
│  │  基礎設施: 內容審核服務、日誌監控                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**為何重要**:
1. **用戶安全**: 防止有害內容傷害用戶
2. **系統可靠性**: 減少錯誤和幻覺
3. **合規要求**: 滿足法規和行業標準
4. **品牌保護**: 避免 AI 醜聞損害聲譽

### 安全威脅

```
1. 幻覺 (Hallucination)
   模型生成虛假資訊當作事實

2. 越獄 (Jailbreak)
   用戶繞過安全限制獲取有害輸出

3. 提示注入 (Prompt Injection)
   惡意輸入操縱模型行為

4. 偏見 (Bias)
   模型輸出帶歧視或偏見

5. 隱私洩漏 (Privacy Leakage)
   模型透露訓練數據中的敏感資訊
```

---

## 2. 防護層

### 多層防護架構

```
用戶輸入
    │
    ▼
┌──────────────────────┐
│  輸入過濾 (Input)    │ ← 惡意內容檢測
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  內容安全審核         │ ← 主題分類
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  回覆過濾 (Output)   │ ← 敏感詞過濾
└──────────┬───────────┘
           │
           ▼
   安全回覆 / 拒絕回答
```

---

## 3. 輸入防護

### 提示注入檢測

```python
class PromptInjectionDetector:
    """檢測提示注入攻擊"""

    def __init__(self):
        self.injection_patterns = [
            r"ignore.*previous.*instructions",
            r"system.*prompt",
            r"you.*are.*now",
            r"forget.*all.*instructions",
            r"new.*instructions"
        ]
        self.classifier = None  # 訓練好的分類器

    def detect(self, user_input):
        """檢測是否為注入攻擊"""

        # Pattern matching
        for pattern in self.injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return True, "injection_pattern"

        # ML classifier (額外層)
        if self.classifier:
            score = self.classifier.predict(user_input)
            if score > 0.9:
                return True, "ml_detected"

        return False, None
```

### 惡意內容過濾

```python
class ContentFilter:
    """內容安全過濾"""

    CATEGORIES = {
        "hate_speech": ["仇恨言論"],
        "violence": ["暴力內容"],
        "sexual": ["性內容"],
        "self_harm": ["自殘"],
        "illegal": ["非法活動"]
    }

    def __init__(self, model_path):
        self.classifier = self._load_model(model_path)

    def filter(self, text):
        """過濾內容"""

        result = self.classifier.predict(text)

        violations = []
        for category, score in result.items():
            if score > 0.8:
                violations.append({
                    "category": category,
                    "score": score
                })

        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
```

---

## 4. 輸出防護

### 敏感詞過濾

```python
class OutputFilter:
    """輸出過濾"""

    def __init__(self):
        self.sensitive_topics = {
            "pii": r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        }

    def filter_pii(self, text):
        """過濾個人身份資訊"""

        filtered = text

        for pii_type, pattern in self.sensitive_topics.items():
            filtered = re.sub(
                pattern,
                f"[{pii_type}_redacted]",
                filtered
            )

        return filtered

    def safe_response(self, text, risk_level):
        """根據風險級別過濾"""

        if risk_level == "high":
            return self._safe_refusal(text)

        return text

    def _safe_refusal(self, original):
        """安全拒絕回覆"""

        return "對不起，我無法協助這個請求。"
```

### 事實性檢查

```python
class FactChecker:
    """事實性檢查"""

    def __init__(self, retrieval_db):
        self.db = retrieval_db

    def verify(self, response):
        """驗證回覆中的聲明"""

        # 提取聲明
        claims = self._extract_claims(response)

        verified_claims = []
        unverified_claims = []

        for claim in claims:
            # 檢索相關資訊
            evidence = self.db.search(claim)

            if self._verify_claim(claim, evidence):
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        return {
            "verified": verified_claims,
            "unverified": unverified_claims,
            "requires_citation": len(unverified_claims) > 0
        }
```

---

## 5. 對話管理

### 主題控制

```python
class TopicController:
    """控制對話主題"""

    def __init__(self):
        self.allowed_topics = [
            "technology", "science", "education",
            "health", "business", "arts"
        ]
        self.blocked_topics = [
            "weapons", "illegal", "harmful"
        ]

    def check_topic(self, user_input):
        """檢查主題是否允許"""

        # 主題分類
        topic = self.classify_topic(user_input)

        if topic in self.blocked_topics:
            return {
                "allowed": False,
                "reason": f"Topic '{topic}' is not allowed"
            }

        if topic not in self.allowed_topics:
            return {
                "allowed": True,
                "confidence": "low",
                "warning": "Unverified topic"
            }

        return {"allowed": True}
```

### 對話階段管理

```python
class ConversationManager:
    """對話狀態管理"""

    def __init__(self):
        self.state = {
            "sensitive_topic": False,
            "requires_verification": False,
            "topic": None,
            "risk_level": "low"
        }

    def update_state(self, user_input, model_response):
        """更新對話狀態"""

        # 檢測敏感話題
        if self._contains_sensitive(user_input):
            self.state["sensitive_topic"] = True
            self.state["risk_level"] = "medium"

        # 檢測需要驗證的內容
        if self._needs_verification(model_response):
            self.state["requires_verification"] = True

        return self.state

    def should_reject(self):
        """判斷是否應該拒絕"""

        return (
            self.state["risk_level"] == "high" or
            self.state["sensitive_topic"] and
            not self._can_continue()
        )
```

---

## 6. 安全評估

### Red Teaming

```python
class RedTeamEvaluator:
    """紅隊評估 - 發現安全漏洞"""

    ATTACK_CATEGORIES = [
        "jailbreak",
        "prompt_injection",
        "context_manipulation",
        "bias_exploitation",
        "privacy_breach"
    ]

    def evaluate(self, model, test_cases):
        """評估模型安全性"""

        results = {}

        for category in self.ATTACK_CASES:
            test_set = self.load_test_cases(category)

            success_count = 0
            for test in test_set:
                response = model.generate(test["prompt"])

                if self.check_violation(test, response):
                    success_count += 1

            results[category] = {
                "total": len(test_set),
                "passed": success_count,
                "pass_rate": success_count / len(test_set)
            }

        return results
```

### 安全指標

| 指標 | 目標 | 測量方法 |
|------|------|----------|
| **拒絕率** | < 5% | 合法請求被拒絕比例 |
| **誤報率** | < 1% | 惡意請求通過比例 |
| **響應時間** | < 500ms | 安全檢查額外延遲 |
| **覆蓋率** | > 95% | 威脅類型覆蓋 |

---

## 7. 實作框架

### NeMo Guardrails

```python
from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails

# 定義配置
config = RailsConfig.from_path("./config")

# 初始化
rails = LLMRails(config)

# 使用
response = rails.generate(
    messages=[{"role": "user", "content": "..."}]
)
```

### Azure AI Content Safety

```python
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextRequest

client = ContentSafetyClient(endpoint, credential)

# 分析內容
request = AnalyzeTextRequest(text=user_input)

result = client.analyze_text(request)

# 檢查結果
if result.hate_severity > 0:
    # 處理仇恨言論
    pass
```

---

## 8. 最佳實踐

### 安全檢查清單

```
上線前檢查:
- [ ] 輸入驗證已實作
- [ ] 輸出過濾已實作
- [ ] 主題控制已配置
- [ ] PII 過濾已啟用
- [ ] Red team 測試通過
- [ ] 監控儀表板已設置
- [ ] 應急回應流程已定義
```

### 監控指標

```python
class SafetyMonitor:
    def __init__(self):
        self.metrics = {
            "blocked_inputs": 0,
            "blocked_outputs": 0,
            "red_flag_events": 0,
            "false_positives": 0
        }

    def record(self, event_type, details):
        self.metrics[event_type] += 1

    def get_dashboard(self):
        return {
            "total_blocked": sum(self.metrics.values()),
            "block_rate": self.metrics["blocked_inputs"] / total_requests,
            "false_positive_rate": self.metrics["false_positives"] / self.metrics["blocked_inputs"]
        }
```

---

## 9. 與相關技術

| 技術 | 關係 |
|------|------|
| **RLHF** | 對齊訓練提升安全性 |
| **Function Calling** | 需要安全驗證工具權限 |
| **RAG** | 檢索事實減少幻覺 |

---

## 延伸閱讀

- [OpenAI Safety Guidelines](https://platform.openai.com/docs/guides/safety-guidelines)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)