# LLM Troubleshooting

LLM 應用開發中的常見問題診斷與解決方案。

---

## 1. 生成問題

### 模型不生成

```python
class GenerationIssues:
    """生成問題診斷"""

    @staticmethod
    def diagnose_no_output(model, prompt):
        """診斷無輸出"""

        # 1. 檢查 tokenization
        tokens = model.tokenizer(prompt, return_tensors="pt")
        if len(tokens.input_ids[0]) == 0:
            return "Empty input after tokenization"

        # 2. 檢查模型輸入
        try:
            with torch.no_grad():
                outputs = model.model(tokens.input_ids)
        except Exception as e:
            return f"Model forward error: {e}"

        # 3. 檢查 logits
        logits = outputs.logits
        if logits.max() == float('-inf'):
            return "All logits are -inf (numerical issue)"

        # 4. 檢查 sampling
        if model.generation_config.temperature == 0:
            # Greedy - 檢查是否有 valid tokens
            pass

        return "No issue found"

    @staticmethod
    def diagnose_repetitive(model, prompt):
        """診斷重複輸出"""

        # 計算 n-gram 重複率
        output = model.generate(prompt, max_tokens=200)
        text = model.tokenizer.decode(output)

        ngrams = {}
        for i in range(len(text) - 3):
            ngram = text[i:i+4]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1

        repeated = {k: v for k, v in ngrams.items() if v > 3}

        if repeated:
            return {
                "issue": "repetition",
                "repeated_ngrams": repeated,
                "suggestion": "Increase temperature or add repetition penalty"
            }

        return "No repetition detected"
```

### 輸出質量差

```python
class QualityIssues:
    """質量問題診斷"""

    @staticmethod
    def diagnose_quality(prompt, response):
        """診斷輸出質量"""

        issues = []

        # 1. 長度檢查
        if len(response) < 10:
            issues.append("Too short output")

        # 2. 重複檢查
        words = response.split()
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append(f"High repetition (unique ratio: {unique_ratio:.2f})")

        # 3. 主題相關性
        # 使用 embedding 相似度
        prompt_emb = get_embedding(prompt)
        response_emb = get_embedding(response[:500])
        similarity = cosine(prompt_emb, response_emb)

        if similarity < 0.3:
            issues.append(f"Low relevance (similarity: {similarity:.2f})")

        # 4. 完整性
        if response.endswith("..."):
            issues.append("Truncated output")

        return issues
```

---

## 2. 性能問題

### 延遲過高

```python
class LatencyIssues:
    """延遲問題診斷"""

    @staticmethod
    def diagnose_high_latency(model, prompt):
        """診斷高延遲"""

        # 分步測量
        steps = {}

        # 1. Tokenization
        start = time.time()
        tokens = model.tokenizer(prompt)
        steps["tokenization_ms"] = (time.time() - start) * 1000

        # 2. Forward pass
        start = time.time()
        with torch.no_grad():
            outputs = model.model(
                torch.tensor([tokens.input_ids])
            )
        steps["forward_ms"] = (time.time() - start) * 1000

        # 3. Sampling
        start = time.time()
        next_token = torch.argmax(outputs.logits[0, -1])
        steps["sampling_ms"] = (time.time() - start) * 1000

        # 識別瓶頸
        bottleneck = max(steps, key=steps.get)

        return {
            "steps": steps,
            "bottleneck": bottleneck,
            "suggestions": {
                "tokenization": "Use faster tokenizer (HuggingFace tokenizers)",
                "forward": "Enable KV cache, use quantization, enable Flash Attention",
                "sampling": "Use greedy or reduce vocab size"
            }
        }
```

### GPU OOM

```python
class OOMIssues:
    """OOM 問題診斷"""

    @staticmethod
    def diagnose_oom():
        """診斷 OOM"""

        import torch

        if not torch.cuda.is_available():
            return "CUDA not available"

        # 當前內存狀態
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        return {
            "current_allocated_gb": allocated,
            "current_reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "suggestions": [
                "Reduce batch size",
                "Use gradient checkpointing",
                "Use mixed precision training",
                "Enable CPU offloading",
                "Use model quantization (int8/int4)"
            ]
        }

    @staticmethod
    def prevent_oom():
        """防止 OOM"""

        # 1. 清理緩存
        torch.cuda.empty_cache()

        # 2. 使用 gradient checkpointing
        # model.gradient_checkpointing_enable()

        # 3. 減少 batch size
        # batch_size = 1

        # 4. 使用混合精度
        # with torch.cuda.amp.autocast():
```

---

## 3. 部署問題

### API 錯誤

```python
class DeploymentIssues:
    """部署問題診斷"""

    @staticmethod
    def diagnose_api_error(error):
        """診斷 API 錯誤"""

        error_types = {
            "429": {
                "name": "Rate Limit",
                "solution": "Implement exponential backoff, reduce request rate"
            },
            "500": {
                "name": "Server Error",
                "solution": "Retry with backoff, check server status"
            },
            "401": {
                "name": "Authentication Error",
                "solution": "Verify API key, check permissions"
            },
            "403": {
                "name": "Forbidden",
                "solution": "Check API key permissions"
            },
            "404": {
                "name": "Not Found",
                "solution": "Verify model name, check endpoint"
            }
        }

        error_code = str(error).split()[0] if error else "unknown"

        return error_types.get(error_code, {
            "name": "Unknown",
            "solution": str(error)
        })

    @staticmethod
    def diagnose_timeout():
        """診斷超時"""

        return {
            "causes": [
                "Model too large",
                "GPU not fast enough",
                "Network latency",
                "Server overload"
            ],
            "solutions": [
                "Increase timeout",
                "Use faster model",
                "Optimize prompt length",
                "Add caching"
            ]
        }
```

---

## 4. 數據問題

### 訓練數據問題

```python
class DataIssues:
    """數據問題診斷"""

    @staticmethod
    def analyze_dataset(dataset):
        """分析數據集"""

        stats = {
            "total_samples": len(dataset),
            "avg_length": 0,
            "length_distribution": {},
            "language_distribution": {},
            "quality_issues": []
        }

        lengths = []
        for sample in dataset[:1000]:  # 抽樣
            text = sample.get("text", "")
            lengths.append(len(text))

            # 檢查質量
            if len(text) < 10:
                stats["quality_issues"].append("Too short")
            if text.strip() != text:
                stats["quality_issues"].append("Leading/trailing whitespace")

        import numpy as np
        stats["avg_length"] = np.mean(lengths)
        stats["length_std"] = np.std(lengths)

        return stats
```

---

## 5. 調試工具

### 調試輔助

```python
class LLMDebugger:
    """LLM 調試器"""

    def __init__(self, model):
        self.model = model

    def debug_generation(self, prompt, max_tokens=50):
        """調試生成過程"""

        # 1. Tokenize
        tokens = self.model.tokenizer(prompt, return_tensors="pt")
        print(f"Input tokens: {tokens.input_ids.shape}")

        # 2. Forward
        with torch.no_grad():
            outputs = self.model(**tokens)

        # 3. 分析 logits
        logits = outputs.logits[0, -1]
        top_tokens = torch.topk(logits, 5)

        print("\nTop 5 tokens:")
        for i, (token_id, logit) in enumerate(zip(top_tokens.indices, top_tokens.values)):
            token = self.model.tokenizer.decode(token_id)
            prob = torch.softmax(logits, dim=0)[token_id].item()
            print(f"  {i+1}. '{token}': logit={logit:.2f}, prob={prob:.3f}")

        # 4. 生成
        generated = self.model.generate(
            tokens.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        print(f"\nGenerated tokens: {generated.shape}")
        print(f"Output: {self.model.tokenizer.decode(generated[0])}")

    def debug_attention(self, prompt, layer_idx=0):
        """調試注意力"""

        # 啟用梯度追蹤
        tokens = self.model.tokenizer(prompt, return_tensors="pt")
        tokens = {k: v.requires_grad_(True) for k, v in tokens.items()}

        outputs = self.model(**tokens, output_attentions=True)

        # 獲取注意力權重
        attention = outputs.attentions[layer_idx][0]

        print(f"Attention shape: {attention.shape}")
        print(f"Attention head: {attention[0].sum()}")  # 第一個頭

        return attention
```

---

## 6. 常見錯誤速查

```
Quick Fixes:

1. "CUDA out of memory"
   → Reduce batch size, enable gradient checkpointing

2. "IndexError: list index out of range"
   → Check empty input, validate prompt

3. "ValueError: too many values to unpack"
   → Check API response format

4. "ConnectionError"
   → Check network, API endpoint

5. "KeyError: 'choices'"
   → Check API response structure

6. "RuntimeError: CUDA error"
   → Clear cache, restart runtime

7. "AttributeError: NoneType"
   → Check None values before processing
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **Profiling** | 性能分析 |
| **Optimization** | 性能優化 |
| **Monitoring** | 持續監控 |

---

## 延伸閱讀

- [HuggingFace Debugging](https://huggingface.co/docs/transformers/en/debugging)
- [PyTorch Troubleshooting](https://pytorch.org/docs/stable/notes/)
- [OpenAI Errors](https://platform.openai.com/docs/guides/error-codes)