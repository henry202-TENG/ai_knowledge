# DPO (Direct Preference Optimization)

## 1. 什麼是？
DPO 是一種簡化的對齊訓練方法，直接從人類偏好數據中學習，無需訓練獎勵模型和強化學習，透過簡單的分類損失達到 RLHF 的效果。

## 2. 為什麼重要？
- **更簡單**：移除複雜的 RL 流程
- **更穩定**：沒有強化學習的不穩定性
- **更高效**：訓練速度更快，計算資源更少
- **效果相當**：實驗顯示 DPO 效果與 RLHF 相近

## 3. 核心原理

### RLHF vs DPO 流程比較
```
RLHF:
  模型 → 輸出 → RM 評分 → PPO 優化 → 模型
         ↓
      需要複雜的 RL 流程

DPO:
  模型 → 直接優化偏好損失 → 模型
         ↓
      只需要分類損失
```

### DPO Loss 函數
```
給定 prompt x，兩個回覆 y₁, y₂（y₁ 被人類偏好）

DPO Loss = -log(σ(r(y₁) - r(y₂)))

其中 r(y) = 模型對回覆 y 的「好壞」評分

直觀理解:
- 如果 y₁ 比 y₂ 好 (σ(r₁ - r₂) 接近 1)，Loss 接近 0
- 如果模型預測錯誤，Loss 大
```

### DPO 的理論基礎
```
DPO 證明了以下等價關係:

PPO 優化目標 ⟺ 直接優化偏好損失

數學上:
max_θ E[(x,y_w,y_l) ~ D] [log(σ(r_θ(x, y_w) - r_θ(x, y_l)))]

其中:
- θ = 模型參數
- x = 輸入 prompt
- y_w = 偏好的回覆 (winner)
- y_l = 不偏好的回覆 (loser)
```

### DPO 的优势
| 特性 | RLHF | DPO |
|------|------|-----|
| 訓練階段 | 3 階段 | 1 階段 |
| 需要 RM | 是 | 否 |
| 超參數 | 多 (PPO) | 少 |
| 訓練穩定性 | 較不穩 | 穩定 |
| 計算資源 | 高 | 中等 |

## 4. 實現細節

### 數據格式
```
{
  "prompt": "如何修復這個 bug?",
  "chosen": "首先檢查錯誤日誌，然後...",
  "rejected": "我不知道，隨便試試"
}
```

### 訓練技巧
- **KL penalty**: 防止過度偏離 reference 模型
- **β 參數**: 控制對偏好數據的敏感度
- **Data mixing**: 混合 SFT 數據保持能力

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **RLHF** | DPO 要簡化的方法 |
| **PPO** | RLHF 使用的強化學習算法 |
| **ORPO** | 另一種偏好優化方法 |
| **KTO** | Kahneman-Tversky Optimization |

## 6. 延伸閱讀
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [HuggingFace DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [DPO vs RLHF 比較](https://arxiv.org/abs/2402.13228)

---

*待補充...*