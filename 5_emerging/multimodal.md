# Multimodal Models

多模態大型語言模型，處理文本、圖像、音頻、視頻等多種輸入輸出。

---

## 1. 什麼是？

### 簡單範例

```
文本模型:
  輸入: "描述這張圖"
  輸出: "一隻貓坐在窗台上"

多模態模型:
  輸入: 圖像 + 文字問題
  輸出: 文字回答 / 生成圖像 / 語音回覆
```

---

## 2. 架構類型

### 早期融合

```
早期融合 (Early Fusion):
  
  圖像 → CNN/ViT → 特徵向量 ─┐
                                   ├→ Transformer → 輸出
  文本 → Embedding ─────────────┘

  特點: 在輸入層融合
  優點: 更好的表示學習
  缺點: 需要重新訓練
```

### 晚期融合

```
晚期融合 (Late Fusion):

  圖像 → 圖像模型 ───────┐
                          ├→ 融合層 → 輸出
  文本 → 語言模型 ───────┘

  特點: 分別處理後融合
  優點: 可利用現有模型
  缺點: 交互有限
```

---

## 3. 視覺語言模型

### LLaVA

```python
class LLaVA:
    """LLaVA 視覺語言模型"""

    def __init__(self, vision_encoder, llm, projector):
        self.vision = vision_encoder  # CLIP ViT
        self.llm = llm               # Vicuna
        self.projector = projector   # 投影層

    def forward(self, image, text):
        # 圖像編碼
        image_features = self.vision.encode(image)

        # 投影到語言空間
        projected_features = self.projector(image_features)

        # 與文本拼接
        combined = self._combine(projected_features, text)

        # LLM 生成
        return self.llm.generate(combined)
```

### BLIP-2

```python
class BLIP2:
    """BLIP-2 視覺語言模型"""

    def __init__(self):
        self.vision = "CLIP ViT-L/14"
        self.q_former = QFormer()
        self.llm = "FlanT5-XXL"
```

---

## 4. 訓練方法

### 對齊訓練

```python
class MultimodalAlignment:
    """多模態對齊訓練"""

    def train(self, image_features, text_features):
        """對齊圖像和文本特徵"""

        # 對比學習 Loss
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = image_features @ text_features.T / 0.1

        labels = torch.arange(len(logits))
        loss = F.cross_entropy(logits, labels)

        return loss
```

### 指令微調

```python
class MultimodalInstructionTuning:
    """多模態指令微調"""

    def __init__(self, model):
        self.model = model

    def finetune(self, dataset):
        """指令微調"""

        for sample in dataset:
            # 多模態輸入
            inputs = {
                "image": sample["image"],
                "text": sample["instruction"]
            }

            # 目標輸出
            targets = sample["response"]

            # 計算 Loss
            outputs = self.model(**inputs)
            loss = F.cross_entropy(
                outputs.logits,
                targets
            )

            loss.backward()
            self.optimizer.step()
```

---

## 5. 生成任務

### 圖像生成

```python
class TextToImage:
    """文本到圖像生成"""

    def __init__(self, model):
        self.model = model  # Stable Diffusion, DALL-E

    def generate(self, prompt, num_images=1):
        """生成圖像"""

        images = self.model.generate(
            prompt,
            num_images=num_images,
            guidance_scale=7.5,
            num_inference_steps=50
        )

        return images
```

### 圖像理解

```python
class ImageUnderstanding:
    """圖像理解"""

    def __init__(self, model):
        self.model = model

    def describe(self, image):
        """描述圖像"""

        return self.model.generate(
            image,
            task="caption"
        )

    def answer_visual_qa(self, image, question):
        """視覺問答"""

        return self.model.generate(
            image,
            question,
            task="vqa"
        )
```

---

## 6. 音視頻理解

### 音頻理解

```python
class AudioUnderstanding:
    """音頻理解"""

    def __init__(self, model):
        self.model = model
        self.audio_encoder = AudioEncoder()

    def transcribe(self, audio):
        """語音轉文字"""

        features = self.audio_encoder.encode(audio)
        text = self.model.generate(features, task="asr")

        return text

    def describe_audio(self, audio):
        """描述音頻內容"""

        features = self.audio_encoder.encode(audio)
        description = self.model.generate(features, task="caption")

        return description
```

### 視頻理解

```python
class VideoUnderstanding:
    """視頻理解"""

    def __init__(self, model):
        self.model = model

    def summarize_video(self, video_frames):
        """視頻摘要"""

        # 採樣關鍵幀
        key_frames = self._sample_frames(video_frames)

        # 處理每幀
        frame_features = [
            self.model.encode(frame)
            for frame in key_frames
        ]

        # 時序建模
        video_features = self._temporal_aggregate(frame_features)

        # 生成描述
        return self.model.generate(video_features, task="caption")
```

---

## 7. 評估基準

### 視覺語言基準

```python
VISUAL_LM_BENCHMARKS = {
    "MME": {
        "description": "多任務視覺語言理解",
        "tasks": ["recognition", "perception", "reasoning"]
    },
    "MMBench": {
        "description": "多模態基準",
        "tasks": ["attribute", "object", "scene"]
    },
    "SEED-Bench": {
        "description": "視覺語言理解",
        "tasks": ["image", "video"]
    },
    "OCRVQA": {
        "description": "OCR 問答",
        "tasks": ["text_recognition", "text_understanding"]
    }
}
```

---

## 8. 應用場景

### 常見應用

```
1. 智能助手
   - 看圖說話
   - 圖像問答
   - 文生圖

2. 醫療
   - 醫學影像分析
   - 病歷解讀

3. 教育
   - 作業輔導
   - 圖書館推薦

4. 無障礙
   - 視覺描述
   - 語音描述
```

---

## 9. 與相關技術

| 技術 | 關係 |
|------|------|
| **CLIP** | 視覺語言對齊 |
| **Diffusion Models** | 圖像生成 |
| **Whisper** | 音頻識別 |

---

## 延伸閱讀

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [Multimodal Survey](https://arxiv.org/abs/2306.13549)