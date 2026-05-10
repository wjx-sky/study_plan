# Day 1：Transformer 原理（上）—— Attention 机制与论文精读

## 今日目标

深入理解 Transformer 的核心组件：Scaled Dot-Product Attention 和 Multi-Head Attention 的数学原理与直觉。

---

## 上午（09:00 - 12:00）：论文精读

### 1. 精读 Attention Is All You Need（2.5h）

**论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**重点章节**：
- Section 1: Introduction — 理解为什么需要 Transformer（RNN 的并行化瓶颈）
- Section 2: Background — 了解 self-attention 的动机
- Section 3.1: Scaled Dot-Product Attention — **公式 (1) 必须完全理解**
- Section 3.2: Multi-Head Attention — **公式 (2)，面试高频考点**
- Section 5: Training — 了解训练配置

**精读要求**：
- [ ] 能徒手写出 Attention 的计算公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- [ ] 理解 Q, K, V 各自的物理含义
- [ ] 理解为什么除以 √d_k（方差稳定）
- [ ] 理解 Multi-Head 为什么 work（不同子空间捕捉不同关系）

### 2. 论文笔记输出（0.5h）

用自己的话写出：
- Self-Attention vs Cross-Attention 的区别
- Multi-Head Attention 的计算流程（画图）

---

## 下午（14:00 - 17:00）：视频学习 + 代码

### 3. Karpathy "Let's build GPT from scratch"（前半部分）

**视频**：[YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)（总长约 4h，今天看前 2h）

**跟写要点**：
- [ ] 理解 token embedding + positional embedding 的生成
- [ ] 理解一个 batch 的数据 shape 变化：(B, T) → (B, T, C)
- [ ] 手写单头 self-attention

**代码产出目标**（保存在 `day1/code/` 目录下）：

```python
# 今天完成的核心代码片段
class SingleHeadAttention(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril mask for autoregressive

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out
```

---

## 晚间（19:00 - 21:00）：笔记整理

- [ ] 整理今日学习笔记（Notion / Markdown）
- [ ] 把 Attention 公式推导流程默写一遍
- [ ] 列出今天没完全理解的 3 个点，作为明天重点关注

---

## 今日 Checklist

- [ ] 能默写 Scaled Dot-Product Attention 公式
- [ ] 能解释 Q, K, V 的物理含义
- [ ] 能解释 Multi-Head 的设计动机
- [ ] 跑通单头 self-attention 的 PyTorch 代码
- [ ] 理解 softmax 前为什么要除以 √d_k

---

## 推荐补充阅读

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 可视化理解
- Lil'Log [The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-transformer/) — 如时间不够可仅浏览前两节
