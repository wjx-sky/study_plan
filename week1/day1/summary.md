# Day 1 学习总结

> 日期：2026 年 5 月 11 日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 关键概念理解

（用自己的话写出今天最重要的 3 个概念）

1. Scaled Dot-Product Attention 的完整计算流程与每一步的术语
2. Q/K/V 的物理含义——键值检索模型
3. Multi-Head Attention 的设计动机——解决单头"有效分辨率降低"

### 1.2 公式推导

（默写 Scaled Dot-Product Attention 公式，不查资料）

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

---

## 二、上午讨论精要：Attention 机制深度解析

### 2.1 背景：为什么需要 Transformer

**RNN 的并行化瓶颈**：
- RNN/LSTM/GRU 处理序列是逐步的：h_t = f(x_t, h_{t-1})，必须等前一步算完
- 顺序操作数 O(T)，无法并行
- 远距离依赖时信息需穿过 O(T) 个时间步，梯度消失/爆炸

**ByteNet / ConvS2S 的尝试——膨胀卷积（Dilated Convolution）**：

膨胀卷积 ≠ 普通卷积。普通卷积 dilation=1，连续取 kernel_size 个位置。膨胀卷积 dilation=d，采样点之间有 d-1 个位置被跳过：

```
dilation=1: ☆☆☆                感受野=3，采样3个
dilation=2: ☆·☆·☆              感受野=5，采样3个
dilation=4: ☆···☆···☆          感受野=9，采样3个
```

ByteNet 堆叠多层膨胀卷积，dilation 逐层翻倍（1, 2, 4, 8...），log(T) 层覆盖整个序列。

**但远距离依赖问题仍未解决**：
- 两个远距离位置之间的信息传递仍需穿过 O(log T) 层
- 路径长度仍随距离增长，梯度衰减和信息损失依然存在

**Transformer 的解决方案**：
- Self-Attention 让任意两个位置在同一层直接计算相似度

| 架构 | 路径长度（位置 i → j） |
|------|----------------------|
| RNN | O(\|i-j\|) |
| ByteNet | O(log\|i-j\|) |
| **Transformer** | **O(1)** |

代价：计算复杂度从 O(T) 升到 O(T²)，但换来任意位置间恒定路径长度。

---

### 2.2 Scaled Dot-Product Attention 完整计算流程

输入：X ∈ R^(T × d_model)

| 步骤 | 数学表达 | 官方术语 | Shape |
|------|---------|---------|-------|
| 投影 | Q = X @ W_Q | Query | (T, d_k) |
| 投影 | K = X @ W_K | Key | (T, d_k) |
| 投影 | V = X @ W_V | Value | (T, d_v) |
| ① | Q @ K^T | **Attention Scores**（注意力分数） | (T, T) |
| ② | Q @ K^T / √d_k | **Scaled Attention Scores** | (T, T) |
| ③ | softmax(...) | **Attention Weights**（注意力权重） | (T, T) |
| ④ | weights @ V | **Context Vector / Attention Output** | (T, d_v) |

关键点：
- 最终输出 Z = weights @ V，shape 为 (T, d_v)，不是 (T, T)
- Multi-Head 时 d_k = d_model / h，例如 d_model=512, h=8 → d_k=64
- Causal Mask 加在第 ② 步之后、softmax 之前（scaled scores + mask → softmax）
- 加 mask 而非乘 mask：加 -inf 后 exp(-inf) = 0，权重归零；乘 0 后 softmax 仍可能非零

---

### 2.3 Q/K/V 的物理含义——键值检索模型

Attention 机制来源于数据库的键值检索（Key-Value Lookup）：

```
Q (Query)：  "位置 i 想要找什么"         —— 查询意图
K (Key)：    "位置 j 能提供什么类型的信息"  —— 索引/标签（和 Q 在同一检索空间）
V (Value)：  "位置 j 的实际语义内容"      —— 正文内容（和 Q 不在同一空间）
```

**Q/K 管 "Where to look"（寻址），V 管 "What to read"（取值）。**

**为什么不能只用 Q 和 V（去掉 K）？**
- Q 和 V 不在同一个语义空间。Q 是"我想找什么"（查询语义），V 是"我是什么内容"（内容语义）
- K 作为中间层，和 Q 在检索空间对齐（W_Q、W_K 训练后使 Q·K 有意义），和 V 独立
- 强行 Q·V 会让"查询意图"和"内容表达"强制对齐，丧失灵活性

**为什么必须乘以 V（weights 不能直接当输出）？**
- weights 是 (T, T) 的概率分布，只告诉你"看哪个位置"，不携带语义信息
- 必须用 V 从目标位置取回实际内容，才能将"关注分配"转化为"语义聚合"
- shape 对齐（weights @ V 产出 (T, d_v) 和输入同形）是工程副产品，不是 V 的设计原因

---

### 2.4 为什么 softmax 前要除以 √d_k

**问题根源**：
- 假设 q 和 k 的每个分量独立同分布 ~ N(0, 1)
- 点积 q·k = Σ q_i k_i，各项独立
- E[q_i·k_i] = 0，Var(q_i·k_i) = 1
- 所以 Var(q·k) = Σ Var(q_i·k_i) = d_k，标准差 = √d_k

**不除的后果**：
- d_k 越大，点积幅值越大（d_k=64 时 std≈8，d_k=1024 时 std≈32）
- 大输入使 softmax 输出趋近 one-hot：`softmax([10, 30, 15]) ≈ [0, 1, 0]`
- Softmax 梯度公式：∂s_i/∂x_i = s_i(1-s_i)，当 s_i≈0 或 s_i≈1 时梯度→0
- → 梯度消失，模型学不动

**除以 √d_k 的作用**：
- Var(q·k / √d_k) = d_k / d_k = 1
- 不管 d_k 多大，点积标准差始终 ≈ 1，softmax 工作在非饱和区

**为什么是 √d_k 不是 d_k？**
- 方差是 d_k，需要标准化的是标准差（√d_k），不是方差

**标准面试回答**："q 和 k 的各分量独立同分布，点积的方差是 d_k，除以 √d_k 把方差控制为 1，防止 softmax 进入饱和区导致梯度消失。"

---

### 2.5 "有效分辨率降低"与 Multi-Head 如何解决

**什么是"有效分辨率降低"**：

单头 Attention 只有一个注意分布。当权重分散均匀时：
- 位置 i 的输出 z_i = Σ_j α_{i,j} v_j ≈ 全局 V 的均值
- 位置 k 的输出 z_k = Σ_j α_{k,j} v_j ≈ 同一个全局均值
- → 不同位置的 z 向量趋于相似，cos_sim(z_i, z_k) → 1
- → 位置间的区分度被 average 操作抹掉 = "有效分辨率降低"

**Multi-Head 如何解决**：
- 不让一个头干所有事，让 8 个头各自在低维子空间（d_k=64）专精一种模式
- 低维度优势：维度越低，向量间不易出现方向坍塌，注意力分布更易集中 → 更锐利
- 8 个锐利视角 concat 起来 → 全局有广度，局部有精度

```
单头 (d_model=512, d_k=512):  一个模糊的全局平均
多头 (d_model=512, h=8, d_k=64):  8 个锐利视角拼接，各管一摊
```

---

### 2.6 Mask 机制

**Causal Mask（自回归 mask）**：训练 GPT 时防止位置 i 看到位置 i+1 及之后的 token。

```python
# T=4 的 mask 矩阵（0=可见，-inf=不可见）
[[  0, -inf, -inf, -inf],
 [  0,   0, -inf, -inf],
 [  0,   0,   0, -inf],
 [  0,   0,   0,   0]]
```

加在 scaled attention scores 之后、softmax 之前：
```
scores = Q @ K^T / √d_k
scores = scores + mask      # 加 -inf 到不可见位置
weights = softmax(scores)   # exp(-inf) = 0，权重严格为 0
```

**Padding Mask**：变长序列中 pad 位置也设为 -inf，防止被关注。

---

## 三、代码产出

### 今天写完的代码模块

- [ ] SingleHeadAttention
- [ ] MultiHeadAttention（如果完成）
- [ ] 其他：

### SingleHeadAttention 核心代码

```python
class SingleHeadAttention(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size))
                                     .masked_fill(torch.tril(torch.ones(block_size, block_size)) == 0, float('-inf')))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # scaled scores
        wei = wei + self.mask[:T, :T]   # causal mask
        wei = F.softmax(wei, dim=-1)    # attention weights
        out = wei @ v                    # context vector
        return out
```

### 遇到的 Bug 与解决

```
```

---

## 四、尚未解决的问题（明天重点）

1.
2.
3.

---

## 五、论文阅读笔记

### Attention Is All You Need

**核心贡献**（一句话）：
完全基于 Attention 机制构建序列转换模型，消除了 RNN 的循环结构，将任意两位置的通信路径缩短到 O(1)。

**关键公式**：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**架构核心组件**：
- Multi-Head Self-Attention（核心创新）
- Position-wise Feed-Forward Network
- Layer Normalization（Post-Norm）
- Residual Connections
- Positional Encoding（Sinusoidal）

**重要设计决策**：
- 为什么是 Scaled Dot-Product Attention：简单、可并行、路径恒定
- 为什么是 Multi-Head：单头有效分辨率低，多头分而治之
- 为什么除以 √d_k：防止点积方差随 d_k 线性增长导致 softmax 饱和

---

## 六、明日预告

- Day 2：完成 Karpathy GPT 视频后半部分 + 手写 Multi-Head Attention + FFN + LayerNorm
- 学习 Pre-Norm vs Post-Norm、GQA/MQA、RoPE 位置编码

---

## 七、额外记录（想法 / 疑问 / 灵感）

