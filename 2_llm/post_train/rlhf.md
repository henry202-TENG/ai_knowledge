# RLHF (Reinforcement Learning from Human Feedback)

## 1. 什麼是？
RLHF 是一種讓 AI 模型透過人類回饋來學習的訓練技術，結合強化學習與人類偏好來讓模型輸出更符合人類期望。

## 2. 為什麼重要？
- **對齊人類價值觀**：讓模型回答更有幫助、更安全
- **超越簡單損失函數**：無法用簡單標籤監督的任務
- **ChatGPT成功的關鍵**：RLHF 是 ChatGPT 背後的核心技術

## 3. 核心原理

### RLHF 三階段流程
```
階段 1: 監督微調 (SFT)
  原始模型 → [人類標註的問答對] → SFT 模型

階段 2: 訓練獎勵模型 (RM)
  SFT 輸出 → [人類偏好比較] → 獎勵模型 (Reward Model)

階段 3: 近端策略優化 (PPO)
  SFT 輸出 → [PPO + Reward] → 最終模型
```

### 階段詳解

#### Stage 1: SFT (Supervised Fine-Tuning)
```
輸入: "如何製作炸彈？"
人類回覆: "對不起，我無法幫助這個請求。"

使用這些人類回覆數據微調模型
```

#### Stage 2: Reward Model
```
訓練數據格式:
Prompt A > Prompt B > Prompt C (人類偏好排序)

範例:
[好回覆] > [一般回覆] > [壞回覆]

訓練目標: 讓 RM 預測人類偏好
Loss = -log(σ(r₁ - r₂))
```

#### Stage 3: PPO 優化
```
使用強化學習優化:

Loss = Reward - β × KL(新模型 || SFT模型) + γ × 穩定項

- Reward: RM 給出的分數
- KL: 防止偏離 SFT 太遠
- 穩定項: 防止獎勵被 gaming
```

### RLHF vs 傳統訓練
| 訓練方式 | 數據需求 | 學習目標 |
|----------|----------|----------|
| **Pretraining** | 大量未標註文本 | 預測下一個 token |
| **SFT** | 人類標註問答 | 模仿人類回覆 |
| **RLHF** | 人類偏好比較 | 優化整體質量 |

## 4. 實作工具

- **TRL** (Transformers Reinforcement Learning)
- **DeepSpeed-Chat**
- **OpenChatKit**
- **Anthropic 的 RLHF**

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **DPO** | RLHF 的簡化替代方案 |
| **PPO** | RLHF 的核心強化學習算法 |
| **Reward Model** | RLHF 的第二階段 |
| **Constitutional AI** | Anthropic 的 AI 對齊方法 |

## 6. 延伸閱讀
- [InstructGPT Paper (OpenAI)](https://arxiv.org/abs/2203.02155)
- [RLHF from Human Feedback](https://arxiv.org/abs/2206.07682)
- [HuggingFace TRL](https://huggingface.co/docs/trl/index)

---

*待補充...*