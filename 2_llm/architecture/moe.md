# MoE (Mixture of Experts)

## 1. 什麼是？
Mixture of Experts（混合專家模型）是一種稀疏化模型架構，通過動態激活不同的「專家」網路來處理不同的輸入，在保持強大能力的同時大幅降低計算成本。

## 2. 為什麼重要？
- **大幅降低推理成本**：只激活部分專家，而非整個模型
- **擴展模型參數**：支援兆級參數模型（如 GPT-4 傳言使用 MoE）
- **任務專家化**：不同專家擅長不同類型的任務

## 3. 核心原理

### 傳統 Dense 模型 vs MoE
```
Dense 模型:
  輸入 → [所有參數] → 輸出
  (每次推理使用 100% 參數)

MoE 模型:
  輸入 → [路由器] → 激活 2-3 個專家 → 輸出
  (每次推理使用 5-10% 參數)
```

### 關鍵組件

#### 1. Experts (專家網路)
```
每個 Expert 是一個獨立的神經網路
通常是 FFN (Feed-Forward Network)

Expert 1: 擅長數學推理
Expert 2: 擅長創意寫作
Expert 3: 擅長程式碼
Expert N: ...
```

#### 2. Router/Gate (路由器)
```
決定輸入應該交給哪個專家處理

Softmax-based Router:
  output = softmax(W × input)
  每個 Expert 獲得一個權重

Top-K Routing:
  選擇權重最高的 K 個專家
  通常 K = 2~8
```

#### 3. Load Balancing (負載均衡)
```
問題：有時某些專家總是被選中
解決：加入輔助 loss 強制均勻分配

 Auxiliary Loss = -Σ log(avg_load_i)
```

### MoE 公式
```
輸出 = Σ(g(x)ᵢ × Eᵢ(x))

其中:
- x = 輸入
- g(x) = 路由器輸出（權重）
- Eᵢ(x) = 第 i 個專家的輸出
```

## 4. 知名 MoE 模型

| 模型 | 專家數 | 激活數 | 總參數 |
|------|:------:|:------:|:------:|
| **Switch Transformers** | 128 | 1 | 1.6T |
| **Mixtral 8x7B** | 8 | 2 | 46.7B |
| **GShard** | 128 | 2 | 600B |
| **ST-MoE** | 64 | 2 | 269B |

## 5. 相關主題

| 技術 | 關係 |
|------|------|
| **Sparse MoE** | MoE 的核心技術，稀疏激活 |
| **Token Routing** | 決定每個 token 去哪個專家 |
| **Expert Parallelism** | 分散式部署不同專家到不同 GPU |
| **KV Cache** | MoE 推理優化的重要技術 |

## 6. 延伸閱讀
- [Switch Transformers Paper](https://arxiv.org/abs/2101.03961)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [MoE 詳解](https://arxiv.org/abs/2208.02816)

---

*待補充...*