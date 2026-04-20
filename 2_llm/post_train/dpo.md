# DPO

簡化的對齊訓練方法，直接從人類偏好數據中學習，無需訓練獎勵模型和強化學習，透過簡單的分類損失達到 RLHF 的效果。

---

## 1. 什麼是？

### 簡單範例

```
訓練數據:
  Prompt: "如何修復這個 bug?"
  Chosen: "首先檢查錯誤日誌，然後..."
  Rejected: "我不知道，隨便試試"

DPO 訓練:
  輸入: (Prompt, Chosen, Rejected)
  輸出: 更新後的 LLM

結果:
  模型學會生成類似 "Chosen" 的回覆
```

---

## 2. 為什麼重要？

### 核心價值

| 價值 | 說明 |
|------|------|
| **更簡單** | 移除複雜的 RL 流程 |
| **更穩定** | 沒有強化學習的不穩定性 |
| **更高效** | 訓練速度更快，計算資源更少 |
| **效果相當** | 實驗顯示 DPO 效果與 RLHF 相近 |

---

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

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO Loss

    目標: 讓 policy 傾向 chosen，遠離 rejected
    """

    # 計算 log ratio
    chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratio = policy_rejected_logps - reference_rejected_logps

    # DPO Loss
    losses = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio))

    return losses.mean()
```

### 數學推導

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
- σ = sigmoid 函數
```

### 為什麼有效？

```
直觀理解:

如果 model(chosen) > model(rejected):
  - σ(r_chosen - r_rejected) → 1
  - Loss → 0 (正確，不需要更新)

如果 model(chosen) < model(rejected):
  - σ(r_chosen - r_rejected) → 0
  - Loss → 大 (需要更新，讓 chosen 機率提高)
```

### DPO 的优势

| 特性 | RLHF | DPO |
|------|------|-----|
| 訓練階段 | 3 階段 | 1 階段 |
| 需要 RM | 是 | 否 |
| 超參數 | 多 (PPO) | 少 |
| 訓練穩定性 | 較不穩 | 穩定 |
| 計算資源 | 高 | 中等 |
| 實現複雜度 | 高 | 低 |

---

## 4. 實現細節

### 數據格式

```python
# 標準 DPO 數據格式
{
    "prompt": "如何修復這個 bug?",
    "chosen": "首先檢查錯誤日誌，然後嘗試以下方法...",
    "rejected": "我不知道，隨便試試"
}
```

### 訓練技巧

```python
# 1. KL Penalty: 防止過度偏離 reference 模型
def compute_loss(policy_logps, ref_logps, beta=0.1):
    return -F.logsigmoid(beta * (policy_logps - ref_logps))

# 2. β 參數影響
# β 越小: 對偏好越敏感，訓練越快
# β 越大: 訓練越保守，類似 RLHF 的 KL penalty

# 3. Data Mixing: 混合 SFT 數據保持能力
# 通常 10-20% SFT 數據 + 80-90% DPO 數據
```

### β 參數選擇

| β 值 | 效果 | 適用場景 |
|------|------|----------|
| 0.01 | 非常保守 | 初始階段 |
| 0.1 | 標準 | 一般場景 |
| 0.3 | 激進 | 快速對齊 |

---

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **RLHF** | DPO 要簡化的方法 |
| **PPO** | RLHF 使用的強化學習算法 |
| **ORPO** | 另一種偏好優化方法 |
| **KTO** | Kahneman-Tversky Optimization |

---

## 延伸閱讀

- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [HuggingFace DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [DPO vs RLHF 比較](https://arxiv.org/abs/2402.13228)
- [TRL DPO 實作](https://github.com/huggingface/trl)