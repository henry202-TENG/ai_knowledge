# Fine-Tuning

對預訓練模型進行微調，使其適應特定任務或領域的技術。

---

## 1. 什麼是？

### 簡單範例

```
預訓練模型: 像一個會說多語言的翻譯員
  - 學會了語言的普遍規律
  - 但不擅長特定專業術語

Fine-tuning: 讓翻譯員專精某個領域
  - 醫學翻譯員 → 學習醫學術語
  - 法律翻譯員 → 學習法律用語
```

---

## 2. Fine-Tuning 方法

### Full Fine-Tuning

```python
class FullFineTuner:
    """全參數微調 - 更新所有層"""

    def __init__(self, model, learning_rate=2e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

    def train_step(self, batch):
        # 所有參數都可訓練
        for param in self.model.parameters():
            param.requires_grad = True

        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

# 缺點: 需要大量 GPU 記憶體
# 70B 模型需要 ~140GB GPU 記憶體
```

### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """LoRA 層 - 注入低秩矩陣"""

    def __init__(self, d_model, d_rank=16, scaling=1.0):
        super().__init__()
        self.d_model = d_model
        self.scaling = scaling

        # A: 降維矩陣 (d_model x r)
        self.lora_A = nn.Parameter(
            torch.zeros(d_model, d_rank)
        )
        # B: 升維矩陣 (r x d_model)
        self.lora_B = nn.Parameter(
            torch.zeros(d_model, d_rank)
        )

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始輸出
        # output = W @ x

        # LoRA 輸出
        # output = W @ x + (B @ A) @ x * scaling
        lora_output = (
            x @ self.lora_A @ self.lora_B.T
        ) * self.scaling

        return lora_output

    def merge_weights(self):
        # 合併 LoRA 權重到主模型
        # W_new = W + B @ A * scaling
        pass

# 只需要訓練 ~1% 參數
# 70B 模型 LoRA 只需 ~1-2GB GPU 記憶體
```

### QLoRA

```python
class QLoRALayer:
    """QLoRA - 量化 + LoRA"""

    def __init__(self, model, quantize_bits=4):
        self.quantize_bits = quantize_bits

        # 1. 量化模型到 4-bit
        self.quantized_weights = self.quantize(
            model.weight,
            bits=quantize_bits
        )

        # 2. 添加 LoRA 適配器
        self.lora_A = nn.Parameter(
            torch.randn(model.out_features, model.rank)
        )
        self.lora_B = nn.Parameter(
            torch.randn(model.rank, model.in_features)
        )

    def quantize(self, weights, bits):
        """量化權重"""
        # NF4 量化
        # ...

    def forward(self, x):
        # 解量化進行前向傳播
        dequant_w = self.dequantize(self.quantized_weights)

        # 加上 LoRA
        lora_contrib = x @ self.lora_A.T @ self.lora_B.T

        return dequant_w @ x + lora_contrib
```

### Adapter

```python
class AdapterLayer(nn.Module):
    """Adapter 模塊 - 插入到每層 Transformer"""

    def __init__(self, d_model, bottleneck=64):
        super().__init__()

        # Down-project
        self.down = nn.Linear(d_model, bottleneck)
        # Non-linear
        self.activation = nn.ReLU()
        # Up-project
        self.up = nn.Linear(bottleneck, d_model)

        # 初始化 small
        self.down.weight.data.zero_()
        self.up.weight.data.zero_()

    def forward(self, x):
        h = self.activation(self.down(x))
        return self.up(h) + x  # 殘差連接
```

---

## 3. 訓練策略

### 逐步微調

```python
class ProgressiveFineTuning:
    """逐步解凍 - 從最外層開始"""

    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers

    def train_step(self, batch, current_layer):
        # 冻结前面的層
        for i in range(current_layer):
            self.freeze_layer(i)

        # 解凍當前層
        self.unfreeze_layer(current_layer)

        # 训练
        outputs = self.model(**batch)
        return outputs.loss

    def schedule(self):
        # 從最後一層開始，逐步向前
        # Layer 11 → 10 → 9 → ... → 0
        pass
```

### 多任務微調

```python
class MultiTaskFineTuner:
    """多任務微調"""

    TASKS = ["classification", "summarization", "qa"]

    def __init__(self, model):
        self.model = model
        self.task_heads = nn.ModuleDict({
            task: TaskHead(model.config)
            for task in self.TASKS
        })

    def train_step(self, batch, task):
        # 共享 backbone
        encoder_output = self.model.backbone(
            batch["input_ids"]
        )

        # 任務特定 head
        logits = self.task_heads[task](
            encoder_output.hidden_states[-1]
        )

        loss = F.cross_entropy(
            logits,
            batch["labels"]
        )

        return loss
```

### 領域自適應微調

```python
class DomainAdaptiveFT:
    """領域自適應預訓練 (DAP)"""

    def __init__(self, base_model, domain_corpus):
        self.model = base_model
        self.corpus = domain_corpus

    def pretrain_domain(self):
        """第一階段: 領域連續預訓練"""

        for epoch in range(5):
            for batch in self.corpus.dataloader():
                # Next token prediction
                outputs = self.model(
                    input_ids=batch,
                    labels=batch
                )
                loss.backward()

    def finetune_task(self, task_data):
        """第二階段: 任務微調"""
        # Standard fine-tuning...
        pass
```

---

## 4. 數據準備

### 數據格式化

```python
class FineTuningDataset:
    """Fine-tuning 數據集"""

    FORMATS = {
        "instruction": {
            "prompt": "Instruction: {instruction}\nInput: {input}",
            "response": "{output}"
        },
        "chat": {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    }

    def format_sample(self, sample, format_type="instruction"):
        template = self.FORMATS[format_type]

        if format_type == "instruction":
            return (
                template["prompt"].format(**sample) +
                "\nOutput: " +
                template["response"].format(**sample)
            )
```

### 數據增強

```python
class DataAugmentation:
    """數據增強策略"""

    @staticmethod
    def back_translation(sample):
        """回譯增強"""
        # 中文 → 英文 → 中文
        english = translate(sample, to="en")
        chinese = translate(english, to="zh")
        return chinese

    @staticmethod
    def paraphrase(sample):
        """改寫增強"""
        prompt = f"用不同的方式表達: {sample}"
        return llm.generate(prompt)

    @staticmethod
    def noise_injection(sample):
        """噪音注入"""
        words = sample.split()
        # 隨機替換 5% 的詞
        for i in range(len(words)):
            if random.random() < 0.05:
                words[i] = random.choice(words)
        return " ".join(words)
```

---

## 5. 訓練配置

### 超參數

```python
TRAINING_CONFIG = {
    # 學習率
    "learning_rate": 2e-5,      # LoRA 常用 1e-4 ~ 2e-4
    "warmup_steps": 100,

    # Batch size
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,

    # 優化器
    "optimizer": "adamw_torch",
    "weight_decay": 0.01,
    "beta": (0.9, 0.999),

    # 正則化
    "max_grad_norm": 1.0,
    "dropout": 0.1,

    # LoRA 特定
    "lora_r": 16,               # 秩
    "lora_alpha": 32,           # 縮放因子
    "lora_dropout": 0.05,
    "target_modules": [        # 目標層
        "q_proj", "k_proj",
        "v_proj", "o_proj"
    ]
}
```

### LoRA 排名選擇

```
| 模型大小 | LoRA r | 參數% | 效果 |
|----------|--------|-------|------|
| 7B       | 8      | 0.5%  | 基礎 |
| 7B       | 16     | 1%    | 標準 |
| 7B       | 32     | 2%    | 較好 |
| 70B      | 64     | 0.5%  | 標準 |
| 70B      | 128    | 1%    | 較好 |

經驗法則: r = min(8, d_model / 16)
```

---

## 6. 評估

### 過擬合檢測

```python
def detect_overfitting(train_losses, eval_losses):
    """檢測過擬合"""

    # 如果 eval loss 開始上升
    if len(eval_losses) > 3:
        recent = eval_losses[-3:]
        if all(recent[i] >= recent[i-1] for i in range(1, 3)):
            return {
                "overfitting": True,
                "action": "early_stop"
            }

    return {"overfitting": False}

# 正常曲線:
# Train: ████████████
# Eval:  ████████▓▓░░  ← 開始上升 = 過擬合
```

### 基準測試

```python
def evaluate_finetuned(model, benchmarks):
    """評估微調後的模型"""

    results = {}
    for name, benchmark in benchmarks.items():
        if name == "mmlu":
            results[name] = benchmark.evaluate(
                model,
                num_few_shot=5
            )
        elif name == "humaneval":
            results[name] = benchmark.evaluate(
                model,
                pass_at_k=[1, 10]
            )

    return results
```

---

## 7. 挑戰與解決

| 挑戰 | 解決方案 |
|------|----------|
| **灾难性遗忘** | LoRA + 原始數據 replay |
| **領域偏移** | 逐步解凍，從顶层开始 |
| **過擬合** | 早停、正則化、數據增強 |
| **GPU 不够** | LoRA/QLoRA/Gradient checkpointing |
| **評估困難** | 使用多個 benchmark 组合 |

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **RLHF** | Fine-tuning 後使用 RLHF 對齊 |
| **Distillation** | 蒸餾可以減少 Fine-tuning 成本 |
| **Prompt Tuning** | 更輕量的 adaptation 方法 |

---

## 延伸閱讀

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/en/training)