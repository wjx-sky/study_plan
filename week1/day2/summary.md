# Day 2 学习总结

> 日期：____年____月____日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 完整 Transformer Block 的模块组成

（用自己的话描述，画图）

### 1.2 三大进阶概念

| 概念 | 一句话解释 | 解决了什么问题 |
|------|-----------|---------------|
| Pre-Norm | | |
| GQA | | |
| RoPE | | |

---

## 二、代码产出

### 今天写完的代码模块

- [ ] MultiHeadAttention
- [ ] FeedForward
- [ ] TransformerBlock
- [ ] RoPE（如果完成）
- [ ] GQA（如果完成）

### 遇到的 Bug 与解决

```
// 特别记录 shape mismatch 相关 bug（Transformer 代码最常见的坑）
```

---

## 三、原始 Transformer vs 现代 LLaMA 架构对比

| 组件 | 原始 Transformer | LLaMA-style |
|------|-----------------|-------------|
| Normalization | Post-Norm | Pre-Norm |
| Position Encoding | Sinusoidal / Learned | RoPE |
| Attention | MHA | GQA |
| Activation | ReLU | SwiGLU |
| FFN | 2-layer MLP | SwiGLU FFN |

---

## 四、尚未解决的问题

1. 
2. 

---

## 五、明日预告

- Day 3：ViT 论文 + CLIP 论文
- 理解 image → patch → token 的流程
- 理解对比学习（contrastive learning）在 CLIP 中的作用

---

## 六、额外记录
