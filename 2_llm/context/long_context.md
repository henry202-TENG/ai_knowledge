# Long Context Models

處理超長上下文（100K+ tokens）的 LLM 技術與最佳實踐。

---

## 1. 什麼是？

### 深度定義

**長上下文模型**的出現是為了解決 LLM 的「上下文瓶頸」：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    上下文長度的演進                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  時代 1: 2K-4K tokens                                               │
│  ├── GPT-2: 1K                                                      │
│  ├── GPT-3: 4K                                                      │
│  └── 限制: 只能處理短文本                                           │
│                                                                      │
│  時代 2: 8K-32K tokens                                              │
│  ├── GPT-3.5 Turbo: 16K                                            │
│  ├── GPT-4: 32K                                                     │
│  └── 突破: 完整論文、代碼文件                                       │
│                                                                      │
│  時代 3: 100K-1M+ tokens                                            │
│  ├── Claude 2.1: 200K                                              │
│  ├── GPT-4 Turbo: 128K                                             │
│  ├── Gemini 1.5: 1M                                                │
│  └── 突破: 完整書籍、代碼庫、對話歷史                                │
│                                                                      │
│  技術驅動:                                                           │
│  - 注意力機制優化 (Sparse, Sliding Window)                          │
│  - 位置編碼擴展 (RoPE, YARN)                                        │
│  - 記憶體優化 (KV Cache 量化、分頁)                                  │
│  - 分散式計算 (Ring Attention)                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 簡單範例

```
短上下文模型 (4K):
  只能處理 1 篇短論文
  超過後截斷 → 遺漏資訊

長上下文模型 (100K+):
  可處理整本書籍
  多篇論文、完整代碼庫
  全程對話記錄
```

---

## 2. 核心挑戰

### 挑戰分析

```
挑戰:
1. 注意力計算 O(n²) - 序列越長越慢
2. GPU 記憶體爆炸 - KV Cache 太大
3. 位置編碼失效 - 超過訓練長度
4. 精度下降 - Softmax 溢出
```

### 深度挑戰分析

#### 挑戰 1：注意力計算複雜度

```
┌─────────────────────────────────────────────────────────────────────┐
│                    注意力複雜度分析                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  標準 Self-Attention:                                                │
│                                                                      │
│  時間複雜度: O(n² × d)                                              │
│  空間複雜度: O(n²)  (Attention Matrix)                              │
│                                                                      │
│  n = 序列長度，d = 隱藏維度                                          │
│                                                                      │
│  實際數據:                                                           │
│  ┌─────────────┬──────────────┬──────────────┐                     │
│  │ 序列長度    │ 計算量      │ 記憶體 (FP16) │                     │
│  ├─────────────┼──────────────┼──────────────┤                     │
│  │ 4K          │ 16M ops     │ 64 MB        │                     │
│  │ 32K         │ 1B ops     │ 512 MB       │                     │
│  │ 128K        │ 16B ops    │ 8 GB         │                     │
│  │ 1M          │ 1T ops     │ 64 GB        │                     │
│  └─────────────┴──────────────┴──────────────┘                     │
│                                                                      │
│  這解釋了為什麼需要稀疏注意力和 Flash Attention！                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 挑戰 2：位置編碼失效

```python
class PositionEncodingAnalysis:
    """位置編碼失效分析"""

    @staticmethod
    def analyze_rope_failure(seq_len, base_freq=10000):
        """
        RoPE 在超長序列失效的原因
        """

        # 位置編碼公式: θ_i = base^(-2i/d)
        freqs = 1.0 / (base_freq ** (np.arange(0, 128, 2) / 128))

        # 問題: 位置差距過大時，相對位置信息丟失
        for pos in [1000, 10000, 100000, 1000000]:
            # 相鄰位置的角度差
            adjacent_diff = freqs[0]  # cos(pos+1) - cos(pos) ≈ freq

            # 遠距離位置的角度差
            distant_diff = freqs[0] * pos  # cos(pos+M) - cos(pos)

            # 問題: 當 distant_diff >> adjacent_diff 時，
            # cos 函數會繞過多個週期，導致週期性混淆
            print(f"Position {pos}: adjacent={adjacent_diff:.6f}, distant={distant_diff:.6f}")

        return "位置編碼在長序列下失去區分度"
```

#### 挑戰 3：精度下降

```python
class SoftmaxStabilityAnalysis:
    """Softmax 精度問題分析"""

    @staticmethod
    def analyze_softmax_overflow(q, k):
        """
        當序列太長時，QK^T 可能非常大
        導致 softmax 計算不穩定
        """

        # 計算 QK^T
        scores = torch.matmul(q, k.transpose(-2, -1))

        # 問題: 如果 scores 的範圍太大
        print(f"Score range: {scores.min():.2f} to {scores.max():.2f}")

        # Softmax 公式: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
        # 問題: 沒有减去 max 值時，可能導致 exp 溢出

        # 解決: 必須使用数值稳定的 softmax
        def stable_softmax(x):
            # 减去最大值避免溢出
            x_max = x.max(dim=-1, keepdim=True)[0]
            exp_x = torch.exp(x - x_max)
            return exp_x / exp_x.sum(dim=-1, keepdim=True)

        return stable_softmax(scores)
```

---

## 3. 技術方案

### 稀疏注意力

```python
class SparseAttention:
    """稀疏注意力 - 只計算局部和全局"""

    def __init__(self, window_size=512, global_size=64):
        self.window = window_size
        self.global_size = global_size

    def compute_mask(self, seq_len):
        """計算注意力遮罩"""

        # 局部窗口
        mask = torch.ones(seq_len, seq_len)

        for i in range(seq_len):
            # 窗口範圍內
            start = max(0, i - self.window)
            end = min(seq_len, i + self.window + 1)

            # 遮罩外部
            mask[i, :start] = 0
            mask[i, end:] = 0

        return mask

    def forward(self, x):
        """稀疏注意力計算"""

        mask = self.compute_mask(x.shape[1])

        # 標準 attention
        attn = self._scaled_dot_product(x, x)

        # 應用遮罩
        attn = attn.masked_fill(mask == 0, float('-inf'))

        return F.softmax(attn, dim=-1)
```

### 滑動窗口

```python
class SlidingWindowAttention:
    """滑動窗口注意力"""

    def __init__(self, window_size=4096):
        self.window = window_size

    def forward(self, q, k, v):
        """滑動窗口計算"""

        seq_len = q.shape[1]

        # 只看 window 範圍內
        outputs = []

        for i in range(seq_len):
            start = max(0, i - self.window + 1)
            end = i + 1

            q_i = q[:, i:i+1]
            k_j = k[:, start:end]
            v_j = v[:, start:end]

            attn = torch.matmul(q_i, k_j.transpose(-2, -1))
            attn = attn / (q.shape[-1] ** 0.5)

            out = torch.matmul(F.softmax(attn, dim=-1), v_j)
            outputs.append(out)

        return torch.cat(outputs, dim=1)
```

### Streaming LLM

```python
class StreamingLLM:
    """Streaming LLM - 流式處理長文本"""

    def __init__(self, model, window_size=4096):
        self.model = model
        self.window = window_size
        self.kv_cache = None

    def generate_streaming(self, prompt):
        """流式生成"""

        # 初始化
        input_ids = self.tokenizer.encode(prompt)

        for token in self._generate(input_ids):
            yield token

            # 滑動窗口: 只保留最近的
            self.kv_cache = self._slide_window(self.kv_cache)

    def _slide_window(self, cache):
        """滑動窗口管理"""

        if cache is None:
            return cache

        # 保持 window 大小
        for layer_cache in cache:
            layer_cache = layer_cache[:, :, -self.window:, :]

        return cache
```

### YARN 位置編碼

```python
class YARNPosition:
    """YARN - 高效的位置編碼擴展"""

    def __init__(self, base=10000, scale=512):
        self.base = base
        self.scale = scale

    def get_position_ids(self, seq_len):
        """生成位置 ID"""

        # 使用非線性位置編碼
        positions = torch.arange(seq_len)

        # 頻率遞減
        freqs = 1.0 / (
            self.base ** (torch.arange(0, 64, 2) / 64)
        )

        # 擴展到目標長度
        scaled_positions = positions / seq_len * self.scale

        # 計算角度
        angles = scaled_positions.unsqueeze(1) * freqs

        return torch.cat([angles.sin(), angles.cos()], dim=-1)
```

---

## 4. 部署優化

### 量化技術

```python
class LongContextQuantization:
    """長上下文量化"""

    @staticmethod
    def kv_cache_quant(kv_cache, bits=8):
        """KV Cache 量化"""

        # 動態量化
        for layer_idx, layer_cache in enumerate(kv_cache):
            # 按 channel 分組量化
            k_cache = layer_cache[0]  # keys
            v_cache = layer_cache[1]  # values

            # 量化
            k_quant = torch.quantize_per_channel(
                k_cache,
                torch.tensor([1.0] * k_cache.shape[0]),
                torch.tensor(0),
                dtype=torch.qint8
            )

            # 存儲量化後的 cache
            layer_cache[0] = k_quant

        return kv_cache

    @staticmethod
    def flash_attention_quant(quantization_type="fp8"):
        """Flash Attention 量化"""
        # 使用 NVIDIA FP8 Flash Attention
        pass
```

### 分塊處理

```python
class ChunkedInference:
    """分塊推理"""

    def __init__(self, model, chunk_size=8192, overlap=512):
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_long_document(self, document):
        """處理長文檔"""

        tokens = self.tokenizer.encode(document)

        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(chunk)

        # 分塊處理
        outputs = []
        for chunk in chunks:
            output = self.model.generate(chunk)
            outputs.append(output)

        # 合併結果
        return self._merge_outputs(outputs)

    def _merge_outputs(self, outputs):
        """合併輸出"""
        # 去重、重疊處理
        merged = []

        for i, output in enumerate(outputs):
            start = 0 if i == 0 else self.overlap
            merged.extend(output[start:])

        return merged
```

---

## 5. 評估基準

### 長上下文 Benchmark

```python
LONG_CONTEXT_BENCHMARKS = {
    "passkey": {
        "description": "在長文本中插入金鑰，測試檢索",
        "lengths": ["4K", "8K", "16K", "32K", "100K", "200K"],
        "metric": "準確率"
    },
    "needle": {
        "description": "多根針測試 (Needle in Haystack)",
        "num_needles": [1, 5, 10, 20, 50],
        "metric": "檢索準確率"
    },
    "book_sum": {
        "description": "書籍摘要任務",
        "length": "500K+ tokens",
        "metric": "ROUGE"
    },
    "code_nav": {
        "description": "代碼庫導航",
        "length": "完整代碼庫",
        "metric": "任務完成率"
    }
}
```

### 測試範例

```python
def test_needle_in_haystack(model, num_needles=10):
    """大海撈針測試"""

    # 生成 haystack
    haystack = "This is a test document. " * 10000

    # 插入 needle
    needles = []
    for i in range(num_needles):
        needle = f"The special token {i} is at position {i * 1000}"
        needles.append(needle)

        # 插入隨機位置
        pos = random.randint(0, len(haystack))
        haystack = haystack[:pos] + needle + haystack[pos:]

    # 測試檢索
    correct = 0

    for needle in needles:
        result = model.query(
            f"Find the token in: {needle}",
            context=haystack
        )

        if str(result) in needle:
            correct += 1

    return correct / num_needles
```

---

## 6. 應用場景

### 常見應用

```
1. 完整代碼庫問答
   - 整個 GitHub 倉庫
   - 跨文件上下文理解

2. 處理長文檔
   - 學位論文、法律合同
   - 書籍、劇本

3. 多輪對話歷史
   - 長期項目對話
   - 客服對話

4. 多文檔分析
   - 對比多篇論文
   - 跨文檔總結
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **Ring Attention** | 長上下文並行 |
| **RoPE** | 位置編碼擴展 |
| **Flash Attention** | 高效注意力計算 |

---

## 延伸閱讀

- [Long Context Survey](https://arxiv.org/abs/2309.16039)
- [Streaming LLM](https://arxiv.org/abs/2309.17453)
- [YARN Paper](https://arxiv.org/abs/2309.00071)