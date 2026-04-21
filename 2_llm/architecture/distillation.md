# Model Distillation

將大模型的知識遷移到小模型的技術，實現模型壓縮和效率提升。

---

## 1. 什麼是？

### 簡單範例

```
老師-學生比喻:

老師 (大模型, 70B 參數):
  - 知識淵博，能力強
  - 但推理慢，資源消耗大

學生 (小模型, 7B 參數):
  - 體積小，速度快
  - 但能力有限

蒸餾: 讓小模型學習大模型的「思考方式」
  → 保留 80% 能力，但快 10 倍
```

---

## 2. 核心原理

### 蒸餾損失函數

```python
def distillation_loss(
    student_logits,    # 學生模型輸出
    teacher_logits,   # 老師模型輸出
    ground_truth,     # 真實標籤
    temperature=2.0, # 溫度參數
    alpha=0.5         # 蒸餾權重
):
    """
    蒸餾 Loss = α × KL(teacher, student) + (1-α) × CE(student, labels)
    """

    # Softmax with temperature
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL Divergence (蒸餾 Loss)
    kl_loss = F.kl_div(
        soft_student,
        soft_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)

    # Standard Cross-Entropy (任務 Loss)
    ce_loss = F.cross_entropy(student_logits, ground_truth)

    # 組合
    return alpha * kl_loss + (1 - alpha) * ce_loss
```

### 溫度參數

```
T = 1: 標準 softmax
T > 1: 輸出更平滑，機率分佈更均勻
T < 1: 輸出更尖銳，接近 argmax

蒸餾時常用 T = 2-10:
  - 老師輸出更平滑，學生更容易學習
  - 保留更多「暗知識」
```

---

## 3. 蒸餾方法

### Response Distillation

```python
class ResponseDistillation:
    """Response-level 蒸餾"""

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model

    def train_step(self, inputs):
        # 老師生成
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        # 學生生成
        student_output = self.student(inputs)

        # Response 蒸餾 Loss
        loss = F.kl_div(
            F.log_softmax(student_output / self.temperature),
            F.softmax(teacher_output / self.temperature),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss
```

### Feature Distillation

```python
class FeatureDistillation:
    """Feature-level 蒸餾 - 匹配中間層表示"""

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model

    def feature_loss(self, student_hidden, teacher_hidden):
        """中間層特徵蒸餾"""

        # MSE Loss on hidden states
        loss = F.mse_loss(student_hidden, teacher_hidden)

        # 或者使用 Cosine Loss
        loss = 1 - F.cosine_similarity(
            student_hidden,
            teacher_hidden
        ).mean()

        return loss
```

### Attention Distillation

```python
class AttentionDistillation:
    """Attention 蒸餾"""

    def __init__(self):
        self.temperature = 2.0

    def attention_loss(self, student_attn, teacher_attn):
        """
        蒸餾 attention 權重
        """

        # 使用 KL Divergence
        student_attn = F.log_softmax(
            student_attn / self.temperature, dim=-1
        )
        teacher_attn = F.softmax(
            teacher_attn / self.temperature, dim=-1
        )

        return F.kl_div(
            student_attn,
            teacher_attn,
            reduction='batchmean'
        ) * (self.temperature ** 2)
```

---

## 4. 實作流程

### 完整蒸餾 Pipeline

```python
class DistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature=4.0,
        alpha=0.7
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

        # 冻结老師
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_epoch(self, dataloader):
        """一個 epoch 的蒸餾訓練"""

        total_loss = 0

        for batch in dataloader:
            # Forward - Teacher (frozen)
            with torch.no_grad():
                teacher_output = self.teacher(batch["input"])

            # Forward - Student
            student_output = self.student(batch["input"])

            # 計算 Loss
            loss = self.compute_distillation_loss(
                student_output,
                teacher_output,
                batch["labels"]
            )

            # Backward - Student
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        labels
    ):
        # Soft target loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard target loss
        ce_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss
```

---

## 5. 蒸餾策略

### 逐步蒸餾

```
蒸餾路徑:
70B → 13B → 7B → 3B

每次蒸餾:
- 減少約 50% 參數
- 保留 80-90% 能力

優勢:
- 每步難度較小
- 可在中間大小評估
```

### 多老師蒸餾

```python
class MultiTeacherDistillation:
    """多老師蒸餾"""

    def __init__(self, teachers, student):
        self.teachers = teachers
        self.student = student

    def train_step(self, inputs):
        # 收集多個老師的輸出
        teacher_outputs = []
        for teacher in self.teachers:
            with torch.no_grad():
                output = teacher(inputs)
            teacher_outputs.append(output)

        # 平均作為最終目標
        avg_teacher = torch.mean(
            torch.stack(teacher_outputs),
            dim=0
        )

        # 蒸餾到學生
        return self.distill(avg_teacher, self.student(inputs))
```

### 任務特定蒸餾

```python
class TaskSpecificDistillation:
    """針對特定任務蒸餾"""

    TASKS = {
        "code": ["HumanEval", "MBPP"],
        "math": ["GSM8K", "MATH"],
        "dialogue": ["PersonaChat"]
    }

    def distill_for_task(self, task_name, models):
        """針對任務選擇性蒸餾"""

        teacher = models[task_name]["teacher"]
        student = models[task_name]["student"]

        # 使用該任務的數據
        task_data = self.load_task_data(task_name)

        # 蒸餾
        for batch in task_data:
            loss = self.compute_loss(
                teacher, student, batch
            )
            loss.backward()
```

---

## 6. 評估蒸餾效果

### 壓縮率 vs 能力

```python
def evaluate_distillation(
    original_model,
    distilled_model,
    benchmark
):
    """評估蒸餾效果"""

    # 參數量
    original_params = sum(p.numel() for p in original_model.parameters())
    distilled_params = sum(p.numel() for p in distilled_model.parameters())

    compression_ratio = original_params / distilled_params

    # 速度
    original_time = benchmark.speed(original_model)
    distilled_time = benchmark.speed(distilled_model)

    speedup = original_time / distilled_time

    # 準確率
    original_acc = benchmark.evaluate(original_model)
    distilled_acc = benchmark.evaluate(distilled_model)

    return {
        "compression_ratio": compression_ratio,
        "speedup": speedup,
        "original_accuracy": original_acc,
        "distilled_accuracy": distilled_acc,
        "accuracy_retention": distilled_acc / original_acc
    }
```

### 典型結果

```
蒸餾 70B → 7B:
  - 壓縮比: 10x
  - 速度提升: 8-10x
  - 能力保留: 80-90%

蒸餾 13B → 7B:
  - 壓縮比: 2x
  - 速度提升: 1.5-2x
  - 能力保留: 90-95%
```

---

## 7. 挑戰與解決

| 挑戰 | 解決方案 |
|------|----------|
| **能力差距大** | 逐步蒸餾，多階段 |
| **過擬合** | 增加正則化，Early stopping |
| **任務偏移** | 使用任務特定數據 |
| **記憶體** | Gradient checkpointing |

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **量化** | 與蒸餾互補的壓縮技術 |
| **Pruning** | 另一種模型壓縮方法 |
| **Speculative Decoding** | 使用蒸餾的小模型加速推斷 |

---

## 延伸閱讀

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [MiniLM](https://arxiv.org/abs/2002.10957)
- [Multi-task Learning](https://arxiv.org/abs/1706.02677)