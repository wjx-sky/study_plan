# Day 2 学习总结

> 日期：2026 年 5 月 12-13 日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 完整 Transformer Block 的模块组成（Pre-Norm 架构）

```
            ┌─────────────┐
            │     x       │  (B, T, n_embd)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │   LayerNorm  │  ← ln1
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ Multi-Head  │  ← causal mask 阻止未来信息流入
            │  Attention  │     每个头独立算 Q·K，mask 在 softmax 前加 -inf
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  残差连接    │  x = x + Attn(LN(x))
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │   LayerNorm  │  ← ln2
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  FeedForward│  ← Linear(4×) → ReLU → Linear(→原维度)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  残差连接    │  x = sa_out + FFN(LN(sa_out))
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │    输出      │  (B, T, n_embd)
            └─────────────┘
```

### 1.2 完整 GPT 模型的数据流

```
idx (B, T) 整数序列
     │
     ├──→ token_embedding_table    → (B, T, C)
     ├──→ position_embedding_table → (B, T, C)
     │         ↓ 相加
     │    x = tok_emb + pos_emb
     │         ↓
     │    × N 个 Block（MHA + FFN + 残差 + LayerNorm）
     │         ↓
     │    LayerNorm_final
     │         ↓
     │    lm_head (Linear → vocab_size)
     │         ↓
     │    logits (B, T, vocab_size)
     │         ↓
     │    loss = cross_entropy(logits.view(B*T, V), target.view(B*T))
```

### 1.3 今日深入讨论的概念汇总

| 概念 | 核心理解 |
|------|---------|
| **Causal Mask 为什么生效** | Q·K 只是算了个分数，信息并未流动；真正的信息搬运发生在 `weights @ V`。Mask 把 future token 的权重压为 0，V 的信息零流入 |
| **推理时 mask 的必要性** | 无 mask → 中间位置的表示被未来 token 污染 → 通过残差和深层注意力的间接传播 → 最后一层看到"脏"表示 |
| **proj (W^O)** | 把拼接后的多头信息线性混合，让各头的不同视角在当前层立即融合，不需要等下一层 |
| **Pre-Norm vs Post-Norm** | Post-Norm 梯度每层都要穿过 Norm 被压缩；Pre-Norm 残差路径上无 Norm，梯度 `1+∂SubLayer` 无损回传 |
| **RMSNorm** | 去掉均值中心化，只做缩放，省计算且效果持平。LLaMA/Mistral 等全在用 |
| **GQA 的广播不占显存** | `expand` 是逻辑视图不分配新内存；`repeat` 是物理拷贝会分配。GQA 必须用 expand |
| **repeat vs repeat_interleave** | `repeat([1,2], 2)` = `[1,2,1,2]`，`repeat_interleave([1,2], 2)` = `[1,1,2,2]`；GQA 需要后者 |
| **nn.Embedding 初始化** | 默认 N(0,1)，GPT 常用 `normal_(0, 0.02)` 小方差防止训练初期 loss 过高 |
| **view 与梯度** | 不打断梯度，只改 shape 不改数据，反向时 reshape 回原形状即可 |
| **self(x) = forward** | nn.Module 的 `__call__` 调用 `forward`，但中间会触发 hooks |
| **with 不是作用域** | `with` 是资源管理（调用 `__enter__`/`__exit__`），不创建新作用域，块内变量外部可用 |
| **LayerNorm 的 γ/β** | 归一化先强制分布统一，γ/β 在稳定基础上做逐维度精细调节——尺度控制权从前面层收归中央 |
| **BatchNorm 推理** | 推理时用训练累积的 running_mean/std，不依赖当前 batch。但 NLP 不用 BN，因变长序列和 padding 污染统计 |
| **RoPE** | 通过旋转矩阵作用于 Q 和 K，使 `Q_m·K_n` 自然成为 `(m-n)` 的函数，融入相对位置信息。不是加在 embedding 上，是旋转 Q/K 向量 |
| **GQA KV cache 节省** | n_kv=2 服务 n_q=8，省 6 个 KV 头：`6 × T × d_head × 2(K+V) × 2 bytes`，LLaMA 7B 32 层共省 ~400MB |

---

## 二、代码产出

### 已完成的模块（transformer_demo.py）

| 模块 | 状态 | 说明 |
|------|------|------|
| 全局配置 | ✅ | n_embed=128, block_size=64, n_head=4, n_layer=4, batch_size=16 |
| 数据加载 + vocab | ✅ | 字符级 tokenization, encode/decode, train/val split |
| get_batch | ✅ | 随机采样 (x, y) 对，y 为 x 右移一位 |
| Head | ✅ | 单头因果注意力，tril mask + masked_fill |
| MultiHeadAttention | ✅ | 多头拼接 + proj 线性融合 + dropout |
| FeedForward | ✅ | 4× 膨胀 + ReLU + Dropout |
| Block | ✅ | Pre-Norm，残差连接正确 |
| BigramLanguageModel | ✅ | token_emb + pos_emb + N blocks + LN + lm_head |
| 训练循环 | ✅ | AdamW, cross_entropy, max_iter=50 |
| generate | ✅ | 自回归生成，每步截断 block_size，取最后一个位置 softmax 后采样 |

### 遗留 Bug

全部已修复。✅

### 从昨天到今天修复的 Bug（11 处已修）

- `__int__` → `__init__`（两处）
- `dim=1` → `dim=-1`
- `trill` → `tril`
- `for _ in num_heads` → `range(num_heads)`
- 多余 `F.softmax` 删除，补上 `self.proj` + `self.dropout`
- `super().__init__(n_embed)` → 正确构造
- `nn.ReLU(out)` → `F.relu(out)`
- `norm.LayerNorm` → `nn.LayerNorm`
- `n_embed / n_head` → `n_embed // n_head`
- Block FFN 残差 `+ x` → `+ sa_out`
- Head 缩放因子 `n_embed**-0.5` → `head_size**-0.5`（补上 `self.head_size`）

---

## 三、原始 Transformer vs 现代 LLaMA 架构对比

| 组件 | 原始 Transformer | LLaMA-style |
|------|-----------------|-------------|
| Normalization | Post-Norm | Pre-Norm + RMSNorm |
| Position Encoding | Sinusoidal / Learned | RoPE |
| Attention | MHA | GQA |
| Activation | ReLU | SwiGLU |
| FFN | 2-layer MLP (4×) | SwiGLU FFN (~2.7×) |
| LayerNorm bias | 有 | 无（bias=False） |

---

## 四、尚未解决的问题（明天重点）

1. RoPE 的完整代码实现（概念理解了，编码未做）
2. GQA 的完整代码实现（概念理解了，编码未做）
3. LeetCode ×3 题：Container With Most Water, 3Sum, 自选一题
4. 训练结果评估（当前 loss 下降情况）

---

## 五、明日预告

- Day 3：ViT 论文 + CLIP 论文
- 理解 image → patch → token 的流程
- 理解对比学习在 CLIP 中的作用

---

## 六、额外记录

- 代码所有 bug 已修复，完整 GPT（token_emb + pos_emb + N blocks + LN + lm_head）可正常运行
- 讨论中发现 Mask 机制理解的关键：Q·K 只算分数不搬运信息，×V 才是信息流动
- Pre-Norm 优于 Post-Norm 的原因：残差路径上那条 `+ 1` 的直通梯度是最本质的

---

## 七、面试题汇总

### Q1: 手写 Multi-Head Attention 的 PyTorch 实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.query = nn.Linear(n_embed, n_embed)
        self.key   = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.proj  = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, h, T, d_k)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)                  # (B, h, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v                                                                # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)                        # (B, T, C)
        return self.proj(out)
```

**关键点**：Q/K/V 投影用 `n_embed → n_embed`，reshape 分出多头维度（`transpose(1,2)`），缩放用 `head_size**-0.5`，mask 加 -inf，concat 通过 `view(B,T,C)` 实现，最后 proj 融合。

---

### Q2: 为什么 Transformer 可以并行训练而 RNN 不能？

**RNN** 的时间步有严格依赖：`h_t = f(x_t, h_{t-1})`，必须按顺序计算——算 `h_5` 之前必须有 `h_4`。反向传播时 BPTT 也必须串行展开，O(T) 个顺序操作。

**Transformer** 每个位置的输出只依赖于所有输入的加权和：`z_i = Σ_j α_{ij} v_j`。所有 α 通过 `Q@K^T` 一次性算出，不需要等上一个位置的结果。前向和反向都可以在 O(1) 个顺序操作内完成（假设无限并行）。

**代价**：Transformer 计算复杂度 O(T²)，RNN 是 O(T)。但 GPU 的并行能力让 O(T²) 的 Transformer 实际跑得更快。

---

### Q3: softmax 前为什么要除以 √d_k？写成 d_k 行不行？

除以 √d_k 是因为点积 `q·k` 的方差 = d_k，标准差 = √d_k。除以 √d_k 将方差控制在 1，防止 softmax 饱和。

**写成 d_k 不行**——分母太大，点积被过度压缩，方差 = 1/d_k，softmax 输出趋于均匀分布，注意力失去聚焦能力。

**一句话**：√d_k 是为了方差归 1；d_k 会让方差变成 1/d_k 太小；不除会让方差 = d_k 太大。

---

### Q4: Pre-Norm 比 Post-Norm 好在哪里？

**梯度角度**：
```
Post-Norm: out = Norm(x + SubLayer(x))
Pre-Norm:  out = x + SubLayer(Norm(x))
```
Post-Norm 里梯度回传要穿过 Norm，每层被压缩一次，深层梯度消失。
Pre-Norm 残差直通路上的梯度系数恒为 1，`∂out/∂x = 1 + ∂SubLayer/∂x`，无损回传。

**训练角度**：Post-Norm 需要 warmup 逐步增加学习率防止前期梯度不稳定；Pre-Norm 不需要 warmup，堆 100+ 层也能稳定训练。

---

### Q5: GQA 是怎么减少 KV cache 的？具体算一下省了多少

8 个 Q 头、2 个 KV 头的情况：KV 只算 2 个头，Q 算 8 个头，计算注意力分数时 K/V 通过 `expand`（不复制数据）广播到 8 个头。

**节省量（单层）**：`(8-2) × 2(K+V) × T × d_head × 2 bytes(fp16)`

LLaMA 7B（32 层，T=4096，d_head=128）：约为 400 MB。这对边缘设备（手机、笔记本）部署是决定性的。

---

### Q6: MHA、MQA、GQA 的区别？各有什么取舍？

| 类型 | KV 头数 | Q 头数 | 显存 | 效果 |
|------|---------|--------|------|------|
| MHA | = Q 头数 | h | 最大 | 最好 |
| GQA | < Q 头数 | h | 中等 | 接近 MHA |
| MQA | 1 | h | 最小 | 略差 |

- **MHA**：每个 Q 有独立 K/V，效果最好但推理内存大
- **MQA**：所有 Q 共享一个 K/V，省内存最多但多头信息融合受限
- **GQA**：折中方案（如 8 Q 头配 2 KV 头），兼顾效果与效率

---

### Q7: LayerNorm 和 BatchNorm 的区别？为什么 NLP 用 LayerNorm？

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 沿 batch 维度（每个特征独立） | 沿 feature 维度（每个样本独立） |
| 依赖 batch | 是（训练时用 batch 统计，推理用 running 统计） | 否（每个样本内部独立归一化） |
| 序列长度敏感性 | 敏感（变长 + padding 污染统计） | 不敏感 |
| 适用场景 | CV（固定尺寸图像） | NLP（变长序列）、小 batch |

NLP 用 LayerNorm 的核心原因：变长序列中 padding 位置会污染 BatchNorm 的 batch 统计，且 LayerNorm 每个 token 独立归一化，天然适配序列建模。

---

### Q8: RMSNorm 跟 LayerNorm 有什么区别？为什么现代 LLM 都在用？

LayerNorm：`(x - μ) / σ * γ + β`（去均值 + 除标准差）
RMSNorm：`x / rms(x) * γ`（只除均方根，不去均值）

**好处**：少算一次均值，快 10%~15%；实验效果持平甚至略好。Transformer 中 ReLU/SiLU 天然输出有偏置，强制去均值反而和激活函数对着干。LLaMA、Mistral、Qwen 全部用 RMSNorm。

---

### Q9: RoPE 和 Sinusoidal 位置编码的本质区别是什么？

**Sinusoidal / Learned**：位置向量**加在** token embedding 上 —— 绝对位置。`x = tok_emb + pos_enc`，然后 `Q·K` 里混杂了内容相似度和位置相似度。

**RoPE**：位置信息**乘入** Q 和 K（通过旋转）—— 相对位置。`Q_m · K_n` 自然只依赖于位置差 `(m-n)`，不依赖于绝对位置 m 或 n。

RoPE 的优势：天然的外推能力（训练 2k 推理 4k），相对位置建模更干净，不需要额外加法。

---

### Q10: Transformer 的 FFN 为什么中间维度膨胀 4 倍？

不是理论必然，是经验选择。原始论文 d_model=512, d_ff=2048 → 4×。瓶颈结构先升维（更高维的非线性空间做变换），再降维（压缩回原维），增强表达力同时控制参数量。现代变体：SwiGLU 时三个权重矩阵，整体倍率需打折（LLaMA 用 ~2.7× 保持总参数量不变）。
