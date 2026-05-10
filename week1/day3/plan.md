# Day 3：Vision Transformer (ViT) + CLIP 对比学习

## 今日目标

理解如何把 Transformer 应用于视觉，以及 CLIP 如何完成图像-文本的对齐——这是所有 VLM 和 VLA 的根基。

---

## 上午（09:00 - 12:00）：ViT 精读

### 1. ViT 论文精读

**论文**：[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

**重点理解**：
- [ ] 图像 → Patch → Token 的流水线：
  ```
  Image (224×224×3) → 196 Patches (14×14) → 196 Tokens → + [CLS] Token
  ```
- [ ] Patch Embedding 的实现：Conv2d 怎么等价于 patch extraction + projection
- [ ] Position Embedding 在 ViT 中的作用（没有它，ViT 会变成"词袋模型"）
- [ ] [CLS] Token 的设计与 BERT 的关系
- [ ] 为什么 ViT 需要大规模预训练（ImageNet-21K / JFT-300M）

**输出**：
- [ ] 画出 ViT 完整前向流程图
- [ ] 理解表 1（ViT variants: Base / Large / Huge 的参数配置）

### 2. ViT 代码速览

用 `timm` / `transformers` 加载 ViT，打印各模块的 shape：
```python
from transformers import ViTModel, ViTImageProcessor
# 追踪 image → pixel_values → patch_embeddings → encoder → output 的 shape 变化
```

---

## 下午（14:00 - 17:00）：CLIP 精读

### 3. CLIP 论文精读

**论文**：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

**重点理解**：
- [ ] 双塔架构：Image Encoder（ViT/ResNet） + Text Encoder（Transformer）→ 共享 embedding space
- [ ] **Contrastive Loss（InfoNCE）**：面试必考！
  ```
  L = -1/2 [log(exp(sim(i,t)/τ) / Σ exp(sim(i,t')/τ)) +
            log(exp(sim(t,i)/τ) / Σ exp(sim(t,i')/τ))]
  ```
  其中 sim(i,t) = image_emb · text_emb（cosine similarity）
- [ ] 为什么用 contrastive loss 而不是 cross-entropy？（对比 vs 分类的本质区别）
- [ ] Temperature parameter τ 的作用（控制分布的 sharpness）
- [ ] Zero-shot transfer 的能力来源
- [ ] Data scale：400M image-text pairs 从互联网爬取

### 4. CLIP 代码实战（1h）

```python
import torch
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# 重点理解：
# - model.vision_model 的结构
# - model.text_model 的结构
# - model.visual_projection / model.text_projection 的对齐作用
# - forward 输出 logits_per_image = image_emb @ text_emb.T
```

---

## 晚间（19:00 - 21:00）：对比学习深入

- [ ] 理解 contrastive loss 的 batch 内负样本机制
- [ ] 阅读李沐 [CLIP 论文逐段精读](https://www.bilibili.com/video/BV1SL4y1s7LQ/)（1h）
- [ ] **面试模拟**：口述 "CLIP 是怎么训练的？zero-shot 分类是怎么做的？"

---

## 今日 Checklist

- [ ] 理解 Patch Embedding 的实现细节（Conv2d → reshape）
- [ ] 能解释 ViT 为什么需要大规模预训练
- [ ] 能徒手写出 CLIP 的 contrastive loss 公式
- [ ] 理解 temperature parameter τ 的作用
- [ ] 跑通 CLIP 推理代码
