# Day 3 学习总结

> 日期：____年____月____日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 ViT 的核心设计

**Image → Token 流程**（写出 shape 变化）：

```
输入图像: (B, 3, 224, 224)
→ Patch Extraction: (B, 196, 768)   # 14×14 patches
→ + [CLS] Token:    (B, 197, 768)
→ + Position Embed: (B, 197, 768)
→ Transformer Encoder ×12: (B, 197, 768)
→ [CLS] output: (B, 768) → classification head
```

### 1.2 CLIP 双塔架构

**画图**：

```
Image ──→ Image Encoder ──→ I₁  ──┐
                                   ├──→ cosine sim → contrastive loss
Text  ──→ Text Encoder  ──→ T₁  ──┘
```

### 1.3 Contrastive Loss 推导

（不查资料写出公式，并解释每一步）

---

## 二、代码产出

### 今天跑通的代码

- [ ] ViT 推理（打印每层 shape）
- [ ] CLIP 推理（zero-shot 分类一个例子）

### 关键代码片段
```python
# 记录 CLIP forward 的核心逻辑
```

---

## 三、核心面试自测

**Q：CLIP 怎么做到 zero-shot 分类？**

> 你的回答：

**Q：为什么 ViT 比 CNN 更适合做多模态？**

> 你的回答：

**Q：Contrastive loss 中温度参数 τ 的作用？设太大或太小会怎样？**

> 你的回答：

---

## 四、尚未解决的问题

1. 
2. 

---

## 五、明日预告

- Day 4：SigLIP 论文 + CLIP 代码深挖
- 理解 SigLIP 对 CLIP 的改进
- 跑通 OpenCLIP，对比不同视觉 backbone

---

## 六、额外记录
