# AI 知識學習筆記

> 由淺入深的 AI/LLM 學習路徑

---

## 📁 資料夾結構

```
ai_knowledge/
├── 1_agent/                    # 🤖 應用層 (最易入門)
│   ├── planning/               # 任務規劃
│   │   ├── react.md            # ReAct 推理
│   │   └── tot.md             # Tree of Thoughts
│   ├── tool_use/               # 工具調用
│   │   └── function_calling.md
│   └── memory/                 # 記憶系統
│       ├── vectordb.md        # 向量資料庫
│       └── graphrag.md        # 知識圖譜 RAG
│
├── 2_llm/                      # 🧠 模型層 (核心知識)
│   ├── architecture/           # 模型架構
│   │   ├── moe.md             # Mixture of Experts
│   │   └── mamba_ssm.md       # 狀態空間模型
│   ├── post_train/             # 後訓練對齊
│   │   ├── rlhf.md            # RLHF
│   │   └── dpo.md             # Direct Preference Optimization
│   ├── decode_opt/             # 解碼優化
│   │   ├── kv_cache.md        # KV Cache ⭐ 重要
│   │   └── speculative_decoding.md
│   └── context/                # 長上下文
│       ├── rope.md            # 旋轉位置編碼
│       └── ring_attention.md
│
├── 3_infra/                    # ⚙️ 基礎設施 (最難)
│   ├── distributed/            # 分散式推論
│   │   ├── tensor_parallelism.md
│   │   └── pipeline_parallelism.md
│   ├── context/                # 超長文本
│   │   ├── ring_attention.md
│   │   └── rope.md
│   └── hardware/               # 硬體互連
│       ├── nvlink_nvswitch.md
│       └── rdma.md
│
└── README.md                   # 知識圖譜
```

---

## 📖 學習路徑（由淺入深）

### 第一階段：🤖 應用層 (⭐-⭐⭐)
| 順序 | 主題 | 資料夾 | 難度 |
|:---:|------|--------|:---:|
| 1 | Function Calling | `1_agent/tool_use` | ⭐ |
| 2 | VectorDB 語義檢索 | `1_agent/memory` | ⭐ |
| 3 | ReAct 推理 | `1_agent/planning` | ⭐ |
| 4 | GraphRAG | `1_agent/memory` | ⭐⭐ |
| 5 | Tree of Thoughts | `1_agent/planning` | ⭐⭐ |

### 第二階段：🧠 LLM 核心 (⭐⭐-⭐⭐⭐)
| 順序 | 主題 | 資料夾 | 難度 |
|:---:|------|--------|:---:|
| 6 | RLHF 對齊 | `2_llm/post_train` | ⭐⭐ |
| 7 | DPO 偏好最佳化 | `2_llm/post_train` | ⭐⭐ |
| 8 | **KV Cache** | `2_llm/decode_opt` | ⭐⭐ |
| 9 | Speculative Decoding | `2_llm/decode_opt` | ⭐⭐⭐ |
| 10 | RoPE 位置編碼 | `2_llm/context` | ⭐⭐⭐ |
| 11 | Mamba / SSM | `2_llm/architecture` | ⭐⭐⭐ |
| 12 | MoE 混合專家 | `2_llm/architecture` | ⭐⭐⭐ |

### 第三階段：⚙️ 基礎設施 (⭐⭐⭐-⭐⭐⭐⭐⭐)
| 順序 | 主題 | 資料夾 | 難度 |
|:---:|------|--------|:---:|
| 13 | Ring Attention | `3_infra/context` | ⭐⭐⭐⭐ |
| 14 | Pipeline Parallelism | `3_infra/distributed` | ⭐⭐⭐⭐ |
| 15 | Tensor Parallelism | `3_infra/distributed` | ⭐⭐⭐⭐⭐ |
| 16 | NVLink/NVSwitch | `3_infra/hardware` | ⭐⭐⭐⭐⭐ |
| 17 | RDMA | `3_infra/hardware` | ⭐⭐⭐⭐⭐⭐ |

---

## 🔑 核心知識關聯圖

```
Function Calling
      │
      ▼
VectorDB ──────▶ ReAct ──────▶ ToT
      │              │
      └──────────────┴──▶ GraphRAG
                            │
                            ▼
                     RLHF ──▶ DPO
                            │
                            ▼
                     KV Cache ◀── Speculative Decoding
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
         Ring Attn       RoPE            MoE
            │               │               │
            └───────────────┴───────────────┘
                            │
                            ▼
              Pipeline Parallelism
                            │
                            ▼
              Tensor Parallelism
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
        NVLink/NVSwitch                   RDMA
```

---

## 📝 筆記格式建議

每個主題筆記建議包含：

```markdown
# [主題名稱]

## 1. 什麼是？
- 一句话概述

## 2. 為什麼重要？
- 解決什麼問題
- 應用場景

## 3. 核心原理
- 關鍵概念
- 數學/算法基礎

## 4. 實現細節
- 主流方案
- 程式碼範例

## 5. 相關主題
- 與哪些知識點關聯

## 6. 延伸閱讀
- 論文、部落格、影片
```

---

## 🚀 快速開始

1. **初學者**：從 `1_agent/tool_use/function_calling.md` 開始
2. **进阶者**：直接进入 `2_llm/decode_opt/kv_cache.md`
3. **專家**：挑戰 `3_infra/hardware/rdma.md`

---

*持續更新中...*