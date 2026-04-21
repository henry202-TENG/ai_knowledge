# Emerging LLM Architectures

新興的大型語言模型架構，包括 SSM、線性注意力、專家混合等最新技術。

---

## 1. 狀態空間模型 (SSM)

### Mamba 架構

```python
class MambaBlock:
    """Mamba 狀態空間塊"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # 投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 卷積
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            d_conv,
            padding=d_conv - 1
        )

        # SSM 參數
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 輸出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        """前向傳播"""

        # 投影
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 卷積
        x_conv = self.conv1d(x_inner.transpose(1, 2))
        x_conv = x_conv[:, :, x.size(1)]
        x_conv = x_conv.transpose(1, 2)

        # SSM
        ssm_output = self._ssm_block(x_conv)

        # 門控
        output = F.silu(z) * ssm_output

        return self.out_proj(output)

    def _ssm_block(self, x):
        """SSM 計算"""
        # 簡化的 SSM
        A = -torch.exp(self.A_log.float())

        # 離散化
        dt = F.softplus(self.dt_proj(x))

        # 狀態更新
        # y = C * (A * x + B * u)
        # ...
        pass
```

---

## 2. 線性注意力

### Linear Transformer

```python
class LinearAttention(nn.Module):
    """線性注意力機制"""

    def __init__(self, d_model, dim_head=64):
        super().__init__()
        self.d_model = d_model
        self.dim_head = dim_head
        self.num_heads = d_model // dim_head

        # 線性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 特徵映射
        self.feature_map = nn.Linear(dim_head, dim_head)

    def forward(self, x):
        """線性注意力前向"""

        B, L, _ = x.shape

        # 投影
        q = self.q_proj(x).view(B, L, self.num_heads, self.dim_head)
        k = self.k_proj(x).view(B, L, self.num_heads, self.dim_head)
        v = self.v_proj(x).view(B, L, self.num_heads, self.dim_head)

        # 特徵映射 (使用隨機特徵)
        q = self._apply_feature_map(q)
        k = self._apply_feature_map(k)

        # 線性注意力的核心
        kv = torch.einsum("nld,nlm->nlm", k, v)
        qkv = torch.einsum("nld,nlm->nlm", q, kv)

        return qkv

    def _apply_feature_map(self, x):
        """隨機特徵映射"""
        # 使用 elu+1 作為特徵映射
        return F.elu(x) + 1
```

---

## 3. 混合架構

### Hybrid Mamba-Transformer

```python
class HybridMambaTransformer:
    """混合 Mamba-Transformer"""

    def __init__(self, config):
        self.layers = nn.ModuleList([])

        for i in range(config.num_layers):
            if i % 2 == 0:
                # Mamba 層
                self.layers.append(MambaBlock(config.d_model))
            else:
                # Transformer 層
                self.layers.append(TransformerBlock(config))

    def forward(self, x):
        """混合前向"""

        for layer in self.layers:
            x = layer(x)

        return x
```

---

## 4. 稀疏 MoE

### MoE 變體

```python
class SparseMoE:
    """稀疏 MoE"""

    def __init__(self, d_model, num_experts, k=2):
        self.num_experts = num_experts
        self.k = k  # 激活專家數

        # 專家
        self.experts = nn.ModuleList([
            MLP(d_model)
            for _ in range(num_experts)
        ])

        # 路由器
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """稀疏 MoE 前向"""

        # 路由器 logits
        router_logits = self.router(x)

        # 選擇 top-k 專家
        top_k_logits, top_k_indices = torch.topk(
            router_logits,
            self.k,
            dim=-1
        )

        # 應用 softmax
        weights = F.softmax(top_k_logits, dim=-1)

        # 收集專家輸出
        output = torch.zeros_like(x)

        for i in range(self.k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = weights[:, :, i]

            # 選擇的專家輸出
            expert_output = self.experts[expert_idx](x)

            # 加權
            output += expert_output * expert_weight.unsqueeze(-1)

        return output
```

---

## 5. 持久化記憶

### 外部記憶

```python
class PersistentMemory(nn.Module):
    """持久化記憶"""

    def __init__(self, d_model, memory_size=1000):
        super().__init__()
        self.memory_size = memory_size

        # 記憶矩陣
        self.memory = nn.Parameter(
            torch.randn(memory_size, d_model)
        )

        # 讀寫頭
        self.read_head = AttentionHead(d_model, d_model)
        self.write_head = AttentionHead(d_model, d_model)

    def forward(self, x):
        """讀寫記憶"""

        # 讀取
        read_output = self.read_head(x, self.memory)

        # 寫入
        self._write(x)

        # 結合
        return x + read_output

    def _write(self, x):
        """寫入記憶"""
        # 可學習的寫入策略
        pass
```

---

## 6. 分塊遞歸

### Block Recurrence

```python
class BlockRecurrence:
    """分塊遞歸"""

    def __init__(self, block_size=512):
        self.block_size = block_size

    def forward(self, x):
        """分塊處理"""

        B, L, D = x.shape

        # 分塊
        num_blocks = (L + self.block_size - 1) // self.block_size

        outputs = []
        hidden_state = None

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, L)

            block = x[:, start:end, :]

            # 處理當前塊
            output, hidden_state = self._process_block(
                block,
                hidden_state
            )

            outputs.append(output)

        return torch.cat(outputs, dim=1)

    def _process_block(self, block, hidden):
        """處理單個塊"""
        # 可選的跨塊依賴
        pass
```

---

## 7. 架構比較

| 架構 | 優勢 | 劣勢 |
|------|------|------|
| **Transformer** | 通用、成熟 | O(n²) 複雜度 |
| **Mamba/SSM** | O(n) 線性 | 較新、訓練不穩 |
| **Linear Attn** | O(n) 內存 | 精度損失 |
| **MoE** | 可擴展 | 負載均衡 |

---

## 8. 與相關技術

| 技術 | 關係 |
|------|------|
| **Mamba SSM** | 狀態空間模型 |
| **Ring Attention** | O(1) 注意力 |
| **MoE** | 稀疏專家 |

---

## 延伸閱讀

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Linear Attention](https://arxiv.org/abs/2006.16236)
- [State Space Models](https://arxiv.org/abs/2111.00396)