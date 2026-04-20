graph LR
    %% 核心節點
    Agent[🤖 AI Agent 應用與認知層]
    LLM[🧠 LLM 模型與演算法層]
    Infra[⚙️ AI Infra 底層與分散式架構]

    %% 關聯線
    Agent -- "依賴推理與規劃" --> LLM
    LLM -- "分散式部署於" --> Infra

    %% 🤖 Agent 深度展開
    Agent --> Planning[任務規劃 Planning]
    Planning --> ReAct[ReAct 思考與行動交替]
    Planning --> ToT[Tree of Thoughts 思維樹]
    Agent --> Tool[工具調用 Tool Use]
    Tool --> FC[Function Calling 結構化輸出]
    Agent --> Memory[長期/短期記憶]
    Memory --> VectorDB[向量資料庫語義檢索]
    Memory --> GraphRAG[GraphRAG 知識圖譜增強]

    %% 🧠 LLM 深度展開
    LLM --> Arch[進階模型架構]
    Arch --> MoE[MoE 混合專家模型]
    Arch --> Mamba[Mamba / SSM 非注意力架構]
    LLM --> PostTrain[對齊與後訓練]
    PostTrain --> RLHF[人類回饋強化學習]
    PostTrain --> DPO[直接偏好最佳化]
    LLM --> DecodeOpt[解碼加速技術]
    DecodeOpt --> SpecDecode[Speculative Decoding 投機解碼]

    %% ⚙️ Infra 深度展開
    Infra --> Dist[分散式推論 Distributed Inference]
    Dist --> TP[張量平行 Tensor Parallelism]
    Dist --> PP[管線平行 Pipeline Parallelism]
    Infra --> Context[超長文本支援]
    Context --> RingAttn[Ring Attention 環形注意力]
    Context --> RoPE[旋轉位置編碼外推]
    Infra --> HW[硬體互連]
    HW --> NVLink[NVLink / NVSwitch]
    HW --> RDMA[RDMA 節點間通訊]

    %% 跨層次影響
    SpecDecode -. "減少" .-> DecodeOpt
    MoE -. "降低推論算力" .-> Arch
    GraphRAG -. "提升準確率" .-> Planning
