# AI 知識學習筆記

> 由淺入深的 AI/LLM 學習路徑 | 深度技術細節全面強化

---

## 📁 資料夾結構

```
ai_knowledge/
├── 1_agent/                    # 🤖 應用層 (最易入門)
│   ├── planning/               # 任務規劃
│   │   ├── react.md            # ReAct 推理
│   │   ├── tot.md             # Tree of Thoughts
│   │   └── agent_planning.md  # Agent 規劃系統
│   ├── tool_use/               # 工具調用
│   │   ├── function_calling.md
│   │   └── mcp.md             # Model Context Protocol
│   ├── tools/                  # 工具開發
│   │   └── function_calling.md
│   ├── memory/                 # 記憶系統
│   │   ├── vectordb.md        # 向量資料庫
│   │   ├── graphrag.md        # 知識圖譜 RAG
│   │   └── agent_memory.md    # Agent 記憶系統
│   ├── framework/              # 開發框架
│   │   ├── langchain.md       # LangChain 框架
│   │   └── autoagent.md       # AutoGen 多代理框架
│   ├── rag/                    # 檢索增強生成
│   │   ├── advanced_rag.md    # 高級 RAG 技術
│   │   ├── rag_systems.md    # RAG 系統實作
│   │   └── data_processing.md # RAG 數據處理
│   ├── multi_agent/            # 多代理系統
│   │   └── multi_agent.md
│   ├── safety/                 # 安全防護
│   │   ├── guardrails.md      # AI 安全護欄
│   │   └── security.md        # AI 系統安全
│   ├── applications/           # 應用開發
│   │   └── chatbot.md        # 聊天機器人開發
│   └── debugging/             # 故障排除
│       └── troubleshooting.md # LLM 問題診斷
│
├── 2_llm/                      # 🧠 模型層 (核心知識)
│   ├── architecture/           # 模型架構
│   │   ├── moe.md             # Mixture of Experts
│   │   ├── mamba_ssm.md      # 狀態空間模型
│   │   ├── transformer_evolution.md  # Transformer 演進
│   │   ├── distillation.md   # 模型蒸餾
│   │   └── efficient_models.md # 高效模型設計
│   ├── post_train/             # 後訓練對齊
│   │   ├── rlhf.md            # RLHF 對齊訓練
│   │   ├── dpo.md             # Direct Preference Optimization
│   │   ├── fine_tuning.md     # Fine-tuning 技術
│   │   └── alignment.md       # AI 對齊技術
│   ├── decode_opt/             # 解碼優化
│   │   ├── kv_cache.md        # KV Cache ⭐ 重要
│   │   ├── speculative_decoding.md  # 投機解碼
│   │   └── generation_strategies.md  # 生成策略
│   ├── context/                # 長上下文
│   │   ├── rope.md            # 旋轉位置編碼
│   │   ├── long_context.md    # 長上下文處理
│   │   └── ring_attention.md  # 環形注意力
│   └── prompt_engineering.md  # 提示工程
│
├── 3_infra/                    # ⚙️ 基礎設施 (最難)
│   ├── distributed/            # 分散式訓練與推論
│   │   ├── tensor_parallelism.md
│   │   ├── pipeline_parallelism.md
│   │   └── training.md         # 分散式訓練
│   ├── hardware/               # 硬體互連
│   │   ├── nvlink_nvswitch.md
│   │   └── rdma.md
│   ├── serving/                # 模型服務部署
│   │   └── model_serving.md
│   ├── platform/               # ML 平台
│   │   └── kubernetes_ml.md   # Kubernetes for ML
│   └── performance/             # 性能優化
│       ├── optimization.md    # 性能優化
│       └── profiling.md       # 性能分析
│
├── 4_eval/                     # 📊 評估與測試
│   ├── benchmarks.md           # 評估基準
│   ├── evals.md               # 評估方法
│   ├── quality.md             # 質量保證
│   └── testing.md             # 測試策略
│
├── 5_emerging/                 # 🚀 新興技術
│   ├── multimodal.md          # 多模態模型
│   ├── emerging_architectures.md  # 新興架構
│   ├── agents_production.md   # 生產級 Agent
│   └── emerging_tech.md        # 新興技術趨勢
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
| 3 | LangChain 框架 | `1_agent/framework` | ⭐ |
| 4 | ReAct 推理 | `1_agent/planning` | ⭐ |
| 5 | Prompt Engineering | `2_llm` | ⭐ |
| 6 | GraphRAG | `1_agent/memory` | ⭐⭐ |
| 7 | Tree of Thoughts | `1_agent/planning` | ⭐⭐ |
| 8 | MCP 協議 | `1_agent/tool_use` | ⭐⭐ |
| 9 | AutoGen 多代理 | `1_agent/framework` | ⭐⭐ |

### 第二階段：🧠 LLM 核心 (⭐⭐-⭐⭐⭐)
| 順序 | 主題 | 資料夾 | 難度 |
|:---:|------|--------|:---:|
| 10 | RLHF 對齊 | `2_llm/post_train` | ⭐⭐ |
| 11 | DPO 偏好最佳化 | `2_llm/post_train` | ⭐⭐ |
| 12 | **KV Cache** | `2_llm/decode_opt` | ⭐⭐ |
| 13 | Fine-tuning | `2_llm/post_train` | ⭐⭐ |
| 14 | AI Alignment | `2_llm/post_train` | ⭐⭐ |
| 15 | Speculative Decoding | `2_llm/decode_opt` | ⭐⭐⭐ |
| 16 | RoPE 位置編碼 | `2_llm/context` | ⭐⭐⭐ |
| 17 | Generation Strategies | `2_llm/decode_opt` | ⭐⭐⭐ |
| 18 | Mamba / SSM | `2_llm/architecture` | ⭐⭐⭐ |
| 19 | MoE 混合專家 | `2_llm/architecture` | ⭐⭐⭐ |
| 20 | Model Distillation | `2_llm/architecture` | ⭐⭐⭐ |
| 21 | Transformer Evolution | `2_llm/architecture` | ⭐⭐⭐ |
| 22 | Long Context | `2_llm/context` | ⭐⭐⭐ |
| 23 | Efficient Models | `2_llm/architecture` | ⭐⭐⭐ |

### 第三階段：⚙️ 基礎設施 (⭐⭐⭐-⭐⭐⭐⭐⭐)
| 順序 | 主題 | 資料夾 | 難度 |
|:---:|------|--------|:---:|
| 24 | Ring Attention | `2_llm/context` | ⭐⭐⭐⭐ |
| 25 | Pipeline Parallelism | `3_infra/distributed` | ⭐⭐⭐⭐ |
| 26 | Distributed Training | `3_infra/distributed` | ⭐⭐⭐⭐ |
| 27 | Tensor Parallelism | `3_infra/distributed` | ⭐⭐⭐⭐⭐ |
| 28 | Model Serving | `3_infra/serving` | ⭐⭐⭐⭐ |
| 29 | Kubernetes for ML | `3_infra/platform` | ⭐⭐⭐⭐ |
| 30 | Performance Optimization | `3_infra/performance` | ⭐⭐⭐⭐ |
| 31 | NVLink/NVSwitch | `3_infra/hardware` | ⭐⭐⭐⭐⭐ |
| 32 | RDMA | `3_infra/hardware` | ⭐⭐⭐⭐⭐⭐ |

### 第四階段：📊 評估與新興技術
| 順序 | 主題 | 資料夾 | 難度 |
|:---:|------|--------|:---:|
| 33 | Evaluation Benchmarks | `4_eval` | ⭐⭐ |
| 34 | LLM Evals | `4_eval` | ⭐⭐ |
| 35 | Quality Assurance | `4_eval` | ⭐⭐⭐ |
| 36 | Testing Strategies | `4_eval` | ⭐⭐ |
| 37 | Multimodal Models | `5_emerging` | ⭐⭐⭐ |
| 38 | Emerging Architectures | `5_emerging` | ⭐⭐⭐⭐ |
| 39 | Agents in Production | `5_emerging` | ⭐⭐⭐⭐ |
| 40 | Emerging Tech Trends | `5_emerging` | ⭐⭐⭐ |

---

## 🔑 核心知識關聯圖

```
🤖 應用層
    │
    ├── Function Calling ──────▶ MCP
    │         │
    ├── VectorDB ──────────────▶ ReAct ──────▶ ToT
    │         │                       │
    │         └───────────────────────┼───────────────▶ Agent Planning
    │                                   │
    ├── LangChain ───────────────────────▶ AutoGen ──▶ Multi-Agent
    │         │
    └── GraphRAG ────────────────────────▶ Advanced RAG
                                              │
                                              ▼
🧠 LLM 核心
    │
    ├── Prompt Engineering ──▶ RLHF ──▶ DPO ──▶ Alignment
    │         │
    ├── Fine-tuning ──────────▶ Distillation ──▶ Efficient Models
    │         │
    ├── KV Cache ◀───────────────◀ Speculative Decoding
    │         │                           │
    │         └──▶ Generation Strategies ┘
    │                   │
    ├── RoPE ───────────┼──────────▶ Long Context
    │                   │
    ├── MoE ────────────┼──────────▶ Mamba/SSM
    │                   │
    └── Transformer Evolution
                      │
                      ▼
⚙️ 基礎設施
    │
    ├── Ring Attention ──────▶ Pipeline Parallelism
    │         │                        │
    └─────────┼────────────────────────┘
              ▼
    Tensor Parallelism ──────────▶ Distributed Training
              │
              ├──▶ NVLink/NVSwitch
              │
              └──▶ RDMA

📊 評估與質量
    │
    ├── Benchmarks ──▶ Evals
    │         │
    └── Quality ──▶ Testing

🚀 新興技術
    │
    ├── Multimodal ──▶ Emerging Architectures
    │         │
    └── Agents Production ──▶ Emerging Tech
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

## 📊 知識庫統計

- **總文件數**: 60+ MD 文件
- **1_agent**: 20+ 主題
- **2_llm**: 14+ 主題
- **3_infra**: 10+ 主題
- **4_eval**: 4 主題
- **5_emerging**: 4 主題

---

*持續更新中...*