# Day 4：ViT/CLIP 代码补完 + SigLIP + Vision Encoder 对比实践

## 今日目标

补齐 Day 3 代码部分 + 理解 SigLIP 对 CLIP 的改进 + 动手对比不同 vision encoder。

---

## 上午（09:00 - 11:00）：补 Day 3 代码部分（ViT + CLIP）

### 1. ViT 代码 walkthrough（1h）

```python
from transformers import ViTModel, ViTImageProcessor
# 追踪 image → pixel_values → patch_embeddings → encoder → output 的 shape 变化
```

- [x] 打印 ViT 各阶段 shape：patch embedding、各层 MHA 输出、最终 [CLS]
- [x] 验证 Patch Embedding 等价于 Conv2d
- [x] 尝试 2D 插值位置编码（224×224 → 384×384）

### 2. CLIP 论文速读 + 代码实战（1h）

**论文**：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

重点理解：
- [x] 双塔架构：Image Encoder + Text Encoder → 共享 embedding space
- [x] **Contrastive Loss（InfoNCE）公式推导**（不查资料写出）
- [x] Temperature parameter τ 的作用
- [x] Zero-shot transfer 原理

```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# 跑通一次 zero-shot 分类：给定图片 + 候选文本标签，输出匹配分数
```

---

## 上午（11:00 - 12:00）：SigLIP 论文精读（Day 4 原有）

### 1. SigLIP 论文

**论文**：[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

**核心改进**：
- [x] CLIP 的 softmax 需要全局归一化 → SigLIP 用 sigmoid + binary cross-entropy，每个 (image, text) pair 独立处理
- [x] 损失函数对比：
  - CLIP: softmax over batch → 需要足够大的 batch（通常 32K+）
  - SigLIP: sigmoid per pair → 对小 batch 更友好
- [x] 理解 label smoothing 在 SigLIP 中的作用
- [x] **为什么 OpenVLA 选 SigLIP？** → 小 batch fine-tune 更稳定，且无需大 batch 的负样本对比

### 2. LeetCode（上午弹性时间）

- [ ] 先补 Day 3 未做：Longest Substring Without Repeating Characters (LC 3), Minimum Window Substring (LC 76)
- [ ] 有时间再加 Day 4 推荐：Group Anagrams (LC 49), Longest Consecutive Sequence (LC 128)

---

### 3. 损失函数对比推导（必做）

- [x] 写出 CLIP InfoNCE Loss 的公式
- [x] 写出 SigLIP Sigmoid BCE Loss 的公式
- [ ] 分析两者的梯度差异：谁更适合 fine-tune 场景？

---

## 下午（14:00 - 17:00）：Vision Encoder 选型实践

### 4. 对比三种 Vision Encoder

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

### 5. OpenCLIP 快速上手（1h）

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
```

- [ ] 了解 OpenCLIP 支持的模型列表
- [ ] 对比 OpenAI CLIP vs OpenCLIP（LAION 数据训练的版本）

---

## 晚间（19:00 - 21:00）：复习 + 面试准备 + LeetCode

- [ ] 写一篇简短的技术对比笔记：CLIP vs SigLIP vs DinoV2
- [ ] **面试题准备**：
  - "CLIP 的局限性是什么？SigLIP 怎么改进的？"
  - "为什么 VLA 需要好的 visual encoder？只用 CLIP 够吗？"
  - "multi-resolution / dynamic resolution 在 VLM 中怎么实现？"
- [ ] LeetCode ×1-2 题（继续哈希表专题）

---

## 今日 Checklist

- [x] （补 Day 3）ViT 代码 walkthrough：打印各阶段 shape，验证 Conv2d 等价性
- [x] （补 Day 3）CLIP 代码实战：跑通一次 zero-shot 分类
- [x] （补 Day 3）能徒手写出 CLIP contrastive loss 公式
- [x] 理解 SigLIP 的 sigmoid loss 与 CLIP 的 softmax loss 的数学差异
- [x] 明白为什么 SigLIP 更适合小 batch 训练
- [ ] 实际跑过 3 种 vision encoder 的推理对比
- [ ] 理解 SigLIP + DinoV2 互补的原因
- [ ] LeetCode 补完 Day 3 滑动窗口 ×2 题
