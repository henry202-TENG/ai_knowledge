# Chatbot Development

聊天機器人的開發從基礎架構到生產部署的完整指南。

---

## 1. 架構設計

### 系統架構

```python
class ChatbotArchitecture:
    """聊天機器人架構"""

    def __init__(self):
        # 1. 輸入處理
        self.input_processor = InputProcessor()

        # 2. 意圖識別
        self.intent_classifier = IntentClassifier()

        # 3. 對話管理
        self.dialogue_manager = DialogueManager()

        # 4. 回覆生成
        self.response_generator = ResponseGenerator()

        # 5. 輸出處理
        self.output_processor = OutputProcessor()

    def process_message(self, user_message, user_id):
        """處理消息"""

        # 1. 輸入驗證和清理
        cleaned = self.input_processor.process(user_message)

        # 2. 獲取對話歷史
        history = self.dialogue_manager.get_history(user_id)

        # 3. 意圖識別
        intent = self.intent_classifier.predict(cleaned, history)

        # 4. 獲取相關資訊
        context = self._gather_context(intent, user_id)

        # 5. 生成回覆
        response = self.response_generator.generate(
            message=cleaned,
            intent=intent,
            context=context,
            history=history
        )

        # 6. 後處理
        formatted = self.output_processor.process(response)

        # 7. 保存對話歷史
        self.dialogue_manager.add_message(user_id, "user", cleaned)
        self.dialogue_manager.add_message(user_id, "assistant", formatted)

        return formatted
```

### 對話狀態

```python
class DialogueState:
    """對話狀態管理"""

    def __init__(self):
        self.states = {}

    def get_state(self, user_id):
        """獲取狀態"""

        if user_id not in self.states:
            self.states[user_id] = {
                "current_intent": None,
                "collected_info": {},
                "conversation_stage": "greeting",
                "last_topic": None,
                "context": {}
            }

        return self.states[user_id]

    def update(self, user_id, **kwargs):
        """更新狀態"""

        state = self.get_state(user_id)
        state.update(kwargs)

        self.states[user_id] = state

    def reset(self, user_id):
        """重置狀態"""

        if user_id in self.states:
            del self.states[user_id]
```

---

## 2. 對話流程

### 意圖識別

```python
class IntentClassifier:
    """意圖分類器"""

    INTENTS = [
        "greeting",
        "goodbye",
        "help",
        "information",
        "complaint",
        "order",
        "payment",
        "refund"
    ]

    def __init__(self, model):
        self.model = model

    def predict(self, message, history):
        """預測意圖"""

        # 使用 LLM 進行零樣本分類
        prompt = f"""Classify the user message into one of these intents:
{', '.join(self.INTENTS)}

Message: {message}
Context: {history[-3:] if history else 'No history'}

Intent:"""

        intent = self.model.generate(prompt).strip().lower()

        # 匹配已知意圖
        for known_intent in self.INTENTS:
            if known_intent in intent:
                return known_intent

        return "unknown"
```

### 槽位填充

```python
class SlotFilling:
    """槽位填充"""

    REQUIRED_SLOTS = {
        "order": ["item", "quantity", "address"],
        "payment": ["amount", "method"],
        "refund": ["order_id", "reason"]
    }

    def __init__(self):
        self.entity_extractor = EntityExtractor()

    def extract(self, message, intent):
        """提取槽位"""

        if intent not in self.REQUIRED_SLOTS:
            return {}

        slots = {}

        # 提取實體
        entities = self.entity_extractor.extract(message)

        # 填充槽位
        for slot in self.REQUIRED_SLOTS[intent]:
            if slot in entities:
                slots[slot] = entities[slot]

        return slots

    def get_missing_slots(self, intent, collected):
        """獲取缺失槽位"""

        required = self.REQUIRED_SLOTS.get(intent, [])
        missing = [s for s in required if s not in collected]

        return missing

    def generate_prompt_for_slot(self, slot):
        """生成槽位請求 prompt"""

        prompts = {
            "item": "請告訴我您想訂購什麼商品？",
            "quantity": "請告訴我數量？",
            "address": "請告訴我送貨地址？",
            "amount": "請告訴我金額？",
            "order_id": "請告訴我訂單編號？",
            "reason": "請告訴我退款原因？"
        }

        return prompts.get(slot, f"請提供 {slot}")
```

---

## 3. 多輪對話

### 對話管理器

```python
class DialogueManager:
    """對話管理器"""

    def __init__(self):
        self.conversations = {}
        self.max_history = 10

    def get_history(self, user_id, limit=None):
        """獲取歷史"""

        if user_id not in self.conversations:
            return []

        history = self.conversations[user_id]

        if limit:
            return history[-limit:]

        return history

    def add_message(self, user_id, role, content):
        """添加消息"""

        if user_id not in self.conversations:
            self.conversations[user_id] = []

        self.conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

        # 限制歷史長度
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]

    def clear_history(self, user_id):
        """清除歷史"""

        if user_id in self.conversations:
            del self.conversations[user_id]
```

### 對話流程控制

```python
class ConversationFlow:
    """對話流程控制"""

    FLOWS = {
        "order": {
            "start": "greeting",
            "steps": [
                {"slot": "item", "prompt": "請問您想訂什麼？"},
                {"slot": "quantity", "prompt": "請問數量是多少？"},
                {"slot": "address", "prompt": "請問送貨地址？"},
                {"slot": "confirm", "prompt": "確認訂單嗎？"}
            ],
            "end": "order_confirmed"
        }
    }

    def __init__(self):
        self.current_flows = {}

    def start_flow(self, user_id, flow_name):
        """開始流程"""

        flow = self.FLOWS[flow_name]
        self.current_flows[user_id] = {
            "flow_name": flow_name,
            "current_step": 0,
            "collected": {}
        }

        return flow["steps"][0]["prompt"]

    def next_step(self, user_id, user_input):
        """下一步"""

        if user_id not in self.current_flows:
            return None

        flow_state = self.current_flows[user_id]
        flow = self.FLOWS[flow_state["flow_name"]]
        current_step = flow_state["current_step"]

        # 收集槽位
        slot = flow["steps"][current_step]["slot"]

        if slot != "confirm":
            # 提取並保存
            flow_state["collected"][slot] = user_input

        # 前進一步
        flow_state["current_step"] += 1

        # 檢查是否完成
        if flow_state["current_step"] >= len(flow["steps"]):
            return {"complete": True, "data": flow_state["collected"]}

        # 返回下一個 prompt
        next_prompt = flow["steps"][flow_state["current_step"]]["prompt"]

        return {"complete": False, "prompt": next_prompt}
```

---

## 4. 個性化

### 用戶画像

```python
class UserProfile:
    """用戶画像"""

    def __init__(self):
        self.profiles = {}

    def get_profile(self, user_id):
        """獲取用戶画像"""

        return self.profiles.get(user_id, {
            "name": None,
            "preferences": {},
            "conversation_style": "neutral",
            "language": "zh-TW",
            "last_interaction": None
        })

    def update(self, user_id, **kwargs):
        """更新画像"""

        if user_id not in self.profiles:
            self.profiles[user_id] = self.get_profile(user_id)

        self.profiles[user_id].update(kwargs)
        self.profiles[user_id]["last_interaction"] = time.time()

    def get_style_prompt(self, user_id):
        """獲取風格 prompt"""

        profile = self.get_profile(user_id)
        style = profile.get("conversation_style", "neutral")

        styles = {
            "formal": "使用正式、專業的語言。",
            "casual": "使用輕鬆、親切的語言。",
            "friendly": "使用友好、熱情的語言。",
            "neutral": "使用中性、客觀的語言。"
        }

        return styles.get(style, styles["neutral"])
```

---

## 5. 安全與合規

### 內容過濾

```python
class ContentFilter:
    """內容過濾"""

    def __init__(self):
        self.sensitivity = "medium"

    def filter_input(self, message):
        """過濾輸入"""

        # 檢查敏感詞
        sensitive = self._check_sensitive(message)

        if sensitive:
            return {
                "allowed": False,
                "reason": "contains_sensitive_content",
                "suggestion": "請更換表達方式"
            }

        return {"allowed": True}

    def filter_output(self, message):
        """過濾輸出"""

        # 檢查輸出
        if self._is_harmful(message):
            return {
                "allowed": False,
                "response": "抱歉，我無法提供這個資訊。"
            }

        return {"allowed": True, "message": message}
```

---

## 6. 監控與分析

### 對話指標

```python
class ChatbotMetrics:
    """聊天機器人指標"""

    def __init__(self):
        self.metrics = {
            "total_conversations": 0,
            "total_messages": 0,
            "successful_intents": 0,
            "fallback_count": 0,
            "avg_turns_per_conversation": 0,
            "user_satisfaction": []
        }

    def record(self, user_id, intent_detected, success, feedback=None):
        """記錄指標"""

        self.metrics["total_messages"] += 1

        if success:
            self.metrics["successful_intents"] += 1
        else:
            self.metrics["fallback_count"] += 1

        if feedback:
            self.metrics["user_satisfaction"].append(feedback)

    def get_dashboard(self):
        """獲取儀表板"""

        total = self.metrics["total_messages"]

        return {
            "total_messages": total,
            "success_rate": self.metrics["successful_intents"] / max(total, 1),
            "fallback_rate": self.metrics["fallback_count"] / max(total, 1),
            "avg_satisfaction": sum(self.metrics["user_satisfaction"]) / max(len(self.metrics["user_satisfaction"]), 1)
        }
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **RAG** | 知識增強 |
| **Memory** | 對話記憶 |
| **Function Calling** | 工具使用 |

---

## 延伸閱讀

- [Dialogue Systems](https://arxiv.org/abs/2109.06142)
- [Conversational AI](https://developer.amazon.com/en-US/docs/alexa/alexa-skills-kit/)
- [Chatbot Design](https://chatbotsmagazine.com/)