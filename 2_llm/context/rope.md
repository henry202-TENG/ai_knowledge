# RoPE (Rotary Position Embedding)

## 1. 什麼是？
RoPE（旋轉位置編碼）是一種位置編碼方法，透過旋轉矩陣將位置資訊編碼到每個 token 的表示中，能夠自然地處理相對位置關係。

## 2. 為什麼重要？
- **相對位置編碼**：直接建模 token 之間的相對距離
- **可擴展到更長上下文**：是長上下文模型的關鍵技術
- **無需額外學習**：旋轉矩陣是確定的，不需要訓練
- **主流採用**：LLaMA, Falcon 等模型都使用 RoPE

## 3. 核心原理

### 傳統位置編碼問題
```
Absolute Position Embedding:
  每個位置有一個固定的向量
  
問題：
- 只知道絕對位置
- 難以推廣到未見過的長度

Relative Position Embedding:
  計算 token 之間的相對距離
  
問題：
- 需要修改 Attention 計算
- 實現複雜
```

### RoPE 核心思想
```
將位置編碼為「旋轉矩陣」:

RoPE(x_m, m) = [cos(mθ)  -sin(mθ)] × [x_m]
               [sin(mθ)   cos(mθ)]   [x_m]

解釋：
- 每個位置 m 對應一個旋轉角度 mθ
- 兩個 token 的 Attention 分數與相對位置相關
```

### 2D RoPE（簡化說明）
```
對於每對 token (位置 i, j):

Attention(Q_i, K_j) ∝ cos((i-j)θ)

優點：
- 只需要相對位置 (i-j)
- 自然處理任意長度
```

### 多維度 RoPE (LLaMA 採用)
```
將維度分成多組，每組使用不同的頻率:

θ_i = θ_base^(-2i/d), i = 0, 1, ..., d/2-1

低頻維度 → 編碼遠距離位置
高頻維度 → 編碼近距離位置
```

### RoPE vs 其他位置編碼
| 特性 | Sinusoidal | RoPE | ALiBi |
|------|------------|------|-------|
| 絕對/相對 | 絕對 | 相對 | 相對 |
| 可學習 | 可選 | 否 | 否 |
| 長上下文 | 一般 | 較好 | 較好 |
| 實現複雜度 | 低 | 中 | 中 |
| 主流採用 | BERT | LLaMA | BLOOM |

## 4. 位置編碼外推

### 問題
```
訓練長度: 2048
測試長度: 4096

直接使用會導致性能下降
```

### 解決方案
| 方法 | 說明 |
|------|------|
| **位置線性插值** | 將位置除以縮放因子 |
| **NTK-aware Scaling** | 改變 RoPE 頻率 |
| **YaRN** | 動態調整頻率 |
| **Hotpot** | 混合多個 RoPE |

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **Ring Attention** | 需要 RoPE 處理超長上下文 |
| **Context Window** | RoPE 決定了上下文長度 |
| **ALiBi** | 另一種位置編碼 |
| **Flash Attention** | 與 RoPE 兼容 |

## 6. 延伸閱讀
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [LLaMA RoPE 實現](https://github.com/meta-llama/llama)
- [位置編碼外推綜述](https://arxiv.org/abs/2309.00071)

---

*待補充...*