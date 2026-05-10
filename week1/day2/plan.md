# Day 2：Transformer 原理（下）—— 完整实现与进阶概念

## 今日目标

手写完整的 Transformer Block（Multi-Head Attention + FFN + LayerNorm），并理解现代 LLM 中的改进版本（Pre-Norm, RoPE, GQA）。

---

## 上午（09:00 - 12:00）：完成视频 + 代码

### 1. Karpathy GPT 视频后半部分（2h）

继续跟写：
- [ ] Multi-Head Attention 的拼接逻辑
- [ ] Feed-Forward Network（两层 MLP + GELU）
- [ ] LayerNorm 的实现与位置
- [ ] 残差连接
- [ ] 完整的 Transformer Block 堆叠

### 2. LeetCode（11:30 - 12:00）

- [ ] 数组/双指针 ×2 题
- 推荐今日题目：Container With Most Water (LC 11), 3Sum (LC 15)

### 3. The Annotated Transformer 对照阅读（1h）

访问：https://nlp.seas.harvard.edu/annotated-transformer/

- [ ] 逐行对照你写的代码和 annotated version 的差异
- [ ] 重点关注 mask 的处理（padding mask / causal mask）

---

## 下午（14:00 - 17:00）：手写 + 进阶概念

### 3. 手写完整 Transformer Block（2h）

**不查资料，从零写出以下组件**：

```python
class MultiHeadAttention(nn.Module):   # 多头的拼接与投影
class FeedForward(nn.Module):          # 两层 MLP，中间维度 4×
class TransformerBlock(nn.Module):     # MHA + FFN + LayerNorm + Residual
```

验证正确性：输入一个 (B, T, C) 的随机张量，确保 shape 不变。

### 4. 进阶概念理解（1h）

- [ ] **Pre-Norm vs Post-Norm**：LayerNorm 放在 MHA/FFN 之前还是之后？现代 Transformer 为什么选 Pre-Norm？
- [ ] **GQA (Grouped Query Attention)**：LLaMA 怎么减少 KV cache？和 MQA 的区别？
- [ ] **RoPE (Rotary Position Embedding)**：相对位置编码，为什么比绝对位置好？推导 RoPE 的旋转矩阵形式。

---

## 晚间（19:00 - 21:00）：复习 + 输出 + LeetCode

- [ ] 画出完整 Transformer Block 的结构图（标注每个操作的 shape 变化）
- [ ] 回答面试题：**"为什么 Transformer 可以并行训练但 RNN 不行？"**
- [ ] 阅读 [LLaMA 2 架构图](https://github.com/meta-llama/llama)，对比与原始 Transformer 的区别
- [ ] LeetCode ×1-2 题（继续数组/双指针专题）

---

## 今日 Checklist

- [ ] 能徒手写出 Multi-Head Attention 的 PyTorch 实现
- [ ] 能解释 Pre-Norm 比 Post-Norm 好在哪（梯度流）
- [ ] 能解释 GQA 为什么减少了 KV cache
- [ ] 理解 RoPE 的直觉（向量旋转实现相对位置）
- [ ] 完整 Transformer Block 的 forward 不发生 shape 错误
