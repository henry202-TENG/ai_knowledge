# Generation Strategies

LLM 生成策略，包括解碼方法、温度控制、核采样等技術優化輸出質量。

---

## 1. 什麼是？

### 簡單範例

```
貪心解碼:
  每步選擇最高概率的詞
  優點: 確定性
  缺點: 容易重複

隨機解碼:
  根據概率分布隨機選擇
  優點: 多樣性
  缺點: 不穩定

Top-K:
  只從最高 K 個詞中隨機選擇
  平衡: 質量 vs 多樣性

Top-P (Nucleus):
  從累積概率達到 P 的詞中選擇
  自適應: 根據分布動態調整
```

---

## 2. 解碼方法

### Greedy Search

```python
def greedy_decode(model, input_ids, max_length):
    """貪心解碼"""

    for _ in range(max_length):
        # 獲取 logits
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]

        # 選擇最高概率
        next_token = torch.argmax(next_token_logits, dim=-1)

        # 追加
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        # 結束標記
        if next_token == model.config.eos_token_id:
            break

    return input_ids
```

### Beam Search

```python
def beam_search(model, input_ids, num_beams=5, max_length=50):
    """束搜索"""

    # 初始化 beam
    beams = [(input_ids, 0.0)]  # (序列, 分數)

    for step in range(max_length):
        all_candidates = []

        for beam in beams:
            current_ids, score = beam

            # 獲取 logits
            outputs = model(current_ids)
            log_probs = F.log_softmax(outputs.logits[:, -1, :], dim=-1)

            # Top-K beams
            topk_log_probs, topk_indices = torch.topk(
                log_probs,
                num_beams
            )

            # 擴展每個 beam
            for i in range(num_beams):
                token = topk_indices[0, i]
                new_score = score + topk_log_probs[0, i].item()

                new_ids = torch.cat([current_ids, token.unsqueeze(0).unsqueeze(0)], dim=-1)

                all_candidates.append((new_ids, new_score))

        # 選擇 top beams
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:num_beams]

    return beams[0][0]
```

### Sampling

```python
def multinomial_sample(model, input_ids, temperature=1.0, top_k=None, top_p=None):
    """多項式採樣"""

    # 獲取 logits
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :] / temperature

    # Top-K 過濾
    if top_k is not None:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
        logits[indices_to_remove] = float('-inf')

    # Top-P (Nucleus) 過濾
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累積概率超過 top_p 的
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # 採樣
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

---

## 3. 溫度控制

### 溫度原理

```
T = 1.0: 標準 softmax
  - 概率分布不變

T > 1.0: 更平滑
  - 低概率詞也有機會被選中
  - 輸出更多樣，但可能不太連貫
  - 適合創意寫作

T < 1.0: 更尖銳
  - 高概率詞更容易被選中
  - 輸出更確定，但可能過於保守
  - 適合翻譯、問答
```

### 動態溫度

```python
class DynamicTemperature:
    """動態溫度"""

    def __init__(self, base_temp=1.0):
        self.base_temp = base_temp
        self.phase = "creative"  # creative, balanced, precise

    def get_temperature(self, position, total_length):
        """根據位置動態調整"""

        # 開頭需要創意
        if position < total_length * 0.1:
            return 1.2  # 高溫

        # 中間平衡
        elif position < total_length * 0.5:
            return 1.0  # 標準

        # 結尾精確
        else:
            return 0.7  # 低溫

    def adjust_by_confidence(self, logits):
        """根據置信度調整"""

        # 計算置信度
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max().item()

        # 置信度高 → 降低溫度
        if confidence > 0.9:
            return 0.5
        # 置信度低 → 提高溫度
        elif confidence < 0.3:
            return 1.5

        return 1.0
```

---

## 4. Top-K 與 Top-P

### 參數選擇

```python
# 典型配置

# 創意任務 (寫作、詩歌)
GENERATION_CONFIG = {
    "temperature": 1.2,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.2
}

# 平衡任務 (對話)
BALANCED_CONFIG = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1
}

# 精確任務 (翻譯、代碼)
PRECISE_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0
}
```

### 對比效果

```python
def compare_strategies(prompt):
    """比較不同策略"""

    strategies = {
        "greedy": {"temperature": 0},
        "top_k_10": {"temperature": 1.0, "top_k": 10},
        "top_k_50": {"temperature": 1.0, "top_k": 50},
        "top_p_0.7": {"temperature": 1.0, "top_p": 0.7},
        "top_p_0.95": {"temperature": 1.0, "top_p": 0.95},
    }

    results = {}
    for name, config in strategies.items():
        output = model.generate(prompt, **config)
        results[name] = output

    return results

# 觀察結果:
# greedy: 確定性、重複
# top_k_10: 較少多樣性
# top_k_50: 較多樣性
# top_p_0.7: 控制嚴格
# top_p_0.95: 自適應、多樣
```

---

## 5. 重複控制

### 重複懲罰

```python
def generate_with_repetition_penalty(
    model,
    input_ids,
    penalty=1.2
):
    """帶重複懲罰的生成"""

    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]

    # 懲罰已出現的 token
    for token_id in set(input_ids[0].tolist()):
        logits[0, token_id] /= penalty

    # 貪心選擇
    next_token = torch.argmax(logits, dim=-1)

    return next_token


class RepetitionPenalty:
    """重複懲罰調度"""

    def __init__(self, base_penalty=1.2, decay=0.98):
        self.penalty = base_penalty
        self.decay = decay
        self.seen_tokens = set()

    def step(self, token):
        """每步更新"""

        if token in self.seen_tokens:
            self.penalty *= self.decay

        self.seen_tokens.add(token)

        return self.penalty
```

---

## 6. 最佳實踐

### 生成策略選擇

```
1. 任務類型
   - 創意寫作 → 高 temperature + top_p
   - 問答 → 中等 temperature
   - 代碼生成 → 低 temperature

2. 應用場景
   - 探索 → 高多樣性
   - 產品發布 → 低變異性

3. 用戶偏好
   - 年輕用戶 → 創意
   - 專業用戶 → 精確
```

### 調優流程

```python
class GenerationOptimizer:
    """生成策略優化"""

    def tune(self, test_cases, metrics):
        """調優參數"""

        best_config = None
        best_score = 0

        # Grid search
        for temp in [0.5, 0.7, 1.0, 1.2]:
            for top_p in [0.8, 0.9, 0.95]:
                config = {"temperature": temp, "top_p": top_p}

                score = self.evaluate(config, test_cases, metrics)

                if score > best_score:
                    best_score = score
                    best_config = config

        return best_config

    def evaluate(self, config, test_cases, metrics):
        """評估配置"""
        # 實現評估邏輯
        pass
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **Speculative Decoding** | 加速生成 |
| **KV Cache** | 記憶優化 |
| **Quantization** | 效率優化 |

---

## 延伸閱讀

- [Nucleus Sampling](https://arxiv.org/abs/1904.09751)
- [Beam Search](https://arxiv.org/abs/1911.06562)
- [Generation Best Practices](https://platform.openai.com/docs/guides/)