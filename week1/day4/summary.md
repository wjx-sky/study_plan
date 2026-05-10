# Day 4 学习总结

> 日期：____年____月____日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 CLIP vs SigLIP 损失函数对比

| | CLIP | SigLIP |
|------|------|--------|
| Loss 类型 | InfoNCE (softmax) | Sigmoid + BCE |
| Batch 依赖 | 需要大 batch | 对小 batch 友好 |
| 计算方式 | batch 内归一化 | per-pair 独立 |
| 梯度特点 | | |

### 1.2 三大 Vision Encoder 实测对比

#### 同一张测试图片，三种 encoder 的特征分析：

| Encoder | 特征维度 | 语义能力 | 空间结构 | 适用场景 |
|---------|---------|---------|---------|---------|
| CLIP ViT-B | | | | |
| SigLIP ViT-B | | | | |
| DinoV2 ViT-B | | | | |

### 1.3 为什么 OpenVLA 用 SigLIP + DinoV2？

> 你的回答：

---

## 二、代码产出

- [ ] CLIP encoder 推理
- [ ] SigLIP encoder 推理
- [ ] DinoV2 encoder 推理
- [ ] 三种特征的 cosine similarity 分析

### 核心代码记录
```python
# 记录双 encoder 融合的实现方式
```

---

## 三、面试自测

**Q：CLIP 和 SigLIP 的区别？什么时候用哪个？**

> 你的回答：

**Q：DinoV2 是自监督训练的，为什么和 SigLIP 互补？**

> 你的回答：

---

## 四、尚未解决的问题

1. 
2. 

---

## 五、明日预告

- Day 5：LLaVA 1.5 论文精读
- 理解 visual encoder → projector → LLM 的串联范式
- 这是 VLA 架构的直接前置知识

---

## 六、额外记录
