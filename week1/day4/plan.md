# Day 4：SigLIP + Vision Encoder 对比实践

## 今日目标

理解 SigLIP 对 CLIP 的改进，动手对比不同的 vision encoder，为后续理解 OpenVLA 为什么选 SigLIP+DinoV2 打基础。

---

## 上午（09:00 - 12:00）：SigLIP 论文精读

### 1. SigLIP 论文

**论文**：[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

**核心改进**：
- [ ] CLIP 的 softmax 需要全局归一化 → SigLIP 用 sigmoid + binary cross-entropy，每个 (image, text) pair 独立处理
- [ ] 损失函数对比：
  - CLIP: softmax over batch → 需要足够大的 batch（通常 32K+）
  - SigLIP: sigmoid per pair → 对小 batch 更友好
- [ ] 理解 label smoothing 在 SigLIP 中的作用
- [ ] **为什么 OpenVLA 选 SigLIP？** → 小 batch fine-tune 更稳定，且无需大 batch 的负样本对比

### 2. 损失函数对比推导（必做）

- [ ] 写出 CLIP InfoNCE Loss 的公式
- [ ] 写出 SigLIP Sigmoid BCE Loss 的公式
- [ ] 分析两者的梯度差异：谁更适合 fine-tune 场景？

---

## 下午（14:00 - 17:00）：Vision Encoder 选型实践

### 3. 对比三种 Vision Encoder

用 `transformers` 和 `timm` 加载并对比：

| Encoder | 模型卡 | 输出特征维度 | 特点 |
|---------|--------|-------------|------|
| CLIP ViT-B | `openai/clip-vit-base-patch32` | 512 | 对齐图文，语义强 |
| SigLIP ViT-B | `google/siglip-base-patch16-224` | 768 | 改进损失，语义强 |
| DinoV2 ViT-B | `facebook/dinov2-base` | 768 | 自监督，结构/纹理强 |

- [ ] 对同一张图片，提取三种 encoder 的特征
- [ ] 计算三种特征的 cosine similarity（理解互补性）
- [ ] **关键理解**：OpenVLA 为什么用 SigLIP + DinoV2 双 encoder？
  - SigLIP 捕获语义（语言对齐）
  - DinoV2 捕获空间结构（抓取/操作需要）

### 4. OpenCLIP 快速上手（1h）

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
```

- [ ] 了解 OpenCLIP 支持的模型列表
- [ ] 对比 OpenAI CLIP vs OpenCLIP（LAION 数据训练的版本）

---

## 晚间（19:00 - 21:00）：复习 + 面试准备

- [ ] 写一篇简短的技术对比笔记：CLIP vs SigLIP vs DinoV2
- [ ] **面试题准备**：
  - "CLIP 的局限性是什么？SigLIP 怎么改进的？"
  - "为什么 VLA 需要好的 visual encoder？只用 CLIP 够吗？"
  - "multi-resolution / dynamic resolution 在 VLM 中怎么实现？"

---

## 今日 Checklist

- [ ] 理解 SigLIP 的 sigmoid loss 与 CLIP 的 softmax loss 的数学差异
- [ ] 明白为什么 SigLIP 更适合小 batch 训练
- [ ] 实际跑过 3 种 vision encoder 的推理对比
- [ ] 理解 SigLIP + DinoV2 互补的原因
- [ ] 了解 OpenCLIP 生态
