# Day 3 学习总结

> 日期：2026 年 5 月 14 日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 ViT 论文理解脑图

```
视觉模型
├── CNN（强归纳偏置）
│   ├── 空间结构
│   │   ├── 2D neighborhood（邻域结构）
│   │   ├── locality（局部性）
│   │   └── translation equivariance（平移等变）
│   │
│   ├── 实现方式
│   │   ├── 卷积核（局部感受野）
│   │   ├── 权重共享
│   │   └── feature map 保留 H×W
│   │
│   ├── inductive bias 特点
│   │   ├── 强先验（hard-coded）
│   │   ├── 小数据友好
│   │   └── 训练更稳定
│   │
│   ├── normalization
│   │   ├── BatchNorm（依赖 batch）
│   │   └── 更适合 CNN feature map 结构
│   │
│   └── 缺点：全局建模能力弱，结构限制较强
│
├── ViT（弱归纳偏置）
│   ├── image → patch → token
│   │   └── Patch Embedding = Conv2d(kernel=patch_size, stride=patch_size, out_ch=D)
│   │       输入 (C, H, W) → 输出 (N, D)，其中 N=HW/P²，D=C·P²
│   │
│   ├── 空间结构处理
│   │   ├── patch extraction（保留局部性）
│   │   ├── patch flatten（丢失内部 2D 结构）
│   │   ├── position embedding（可学习绝对位置编码，2D 插值适配高分辨率）
│   │   └── interpolation（微调时高分辨率适配）
│   │
│   ├── 2D inductive bias 来源（很少）
│   │   ├── patch 切分（局部区域）
│   │   └── position embedding 2D 插值
│   │
│   ├── attention 全局建模，无邻域结构，需要数据学习空间关系
│   │
│   ├── normalization
│   │   ├── LayerNorm（token 级，不依赖 batch）
│   │   └── 为什么 ViT 不能用 BN：patch 间特征分布差异大，强制共享统计量不合理
│   │
│   ├── training
│   │   ├── pretrain（JFT-300M / ImageNet-21K 大规模数据）
│   │   ├── fine-tune（小任务）
│   │   └── higher resolution fine-tune
│   │
│   ├── few-shot evaluation
│   │   ├── freeze backbone + linear probe / ridge regression
│   │   ├── ridge regression 有闭式解：(X^T X + λI)^{-1} X^T y
│   │   └── softmax + CE 无闭式解，必须梯度下降
│   │
│   └── 优缺点
│       ├── 优点：全局建模强，可扩展性强
│       └── 缺点：大量数据依赖，小数据不如 CNN
│
├── 关键对比总结
│   ├── CNN：结构先验强（hard inductive bias），小数据友好
│   ├── ViT：结构先验弱（data-driven），大数据上限更高
│   ├── locality：CNN 天然局部 vs ViT attention 全局
│   └── 表达能力：ViT 更自由，上限更高
│
└── 核心概念补充
    ├── Inductive Bias：模型先验假设，决定了对数据的依赖程度
    ├── Transfer Learning：pretrain → downstream，backbone 复用，少样本适配
    ├── few-shot accuracy：frozen feature + linear classifier → 衡量表示能力
    └── closed-form solution：仅限线性回归 + 平方损失，分类任务不可解
```

### 1.2 ViT 前向数据流

```
输入 Image:     (B, 3, 224, 224)
       │
Patch Embedding: Conv2d(kernel=16, stride=16, out_ch=768)
       │
Patch Tokens:   (B, 196, 768)       # 196 = 14×14 patches
       │
[CLS] Token:    (B, 1, 768)         # prepend
       │
Concat:         (B, 197, 768)       # [CLS] + 196 patches
       │
+ Pos Embed:    (B, 197, 768)       # 可学习绝对位置编码
       │
Transformer ×12: (B, 197, 768)      # MHA + FFN + LayerNorm + 残差
       │
取 [CLS] 输出:   (B, 768)            # 或 GAP over all patches
       │
MLP Head:       (B, num_classes)
```

### 1.3 论文关键实验结论

| 预训练数据 | 数据量 | ViT vs ResNet |
|-----------|--------|--------------|
| ImageNet | 1.2M | ViT 弱于 ResNet |
| ImageNet-21K | 14M | 基本持平 |
| JFT-300M | 300M | ViT 显著超越 |

> ViT 需要大规模数据的原因：弱归纳偏置意味着空间关系、局部性等全部要从数据中学习，数据少时学不到这些，CNN 的强先验占优。

---

## 二、ViT 面试题自测

### Q1: Patch Embedding 怎么用 Conv2d 实现？

```python
# 等价实现
proj = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
x = proj(image)              # (B, 768, 14, 14)
x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
```

本质：每个 16×16×3 的 patch 拉平成 768 维向量，再通过 Linear 投影到 D。Conv2d 一步完成。

### Q2: [CLS] token 有什么用？有哪些替代方案？

**[CLS]**：一个可学习的全局聚合 token，Transformer 各层对它做 attention，最终用它的输出做分类。

**替代方案**：GAP（Global Average Pooling）——对所有 patch token 取平均，效果接近，不需要额外 token。

### Q3: ViT 微调时输入更高分辨率图像，位置编码怎么处理？

训练时 224×224 → 14×14=196 个 patch → 196 个位置编码（可学习）。
推理时 384×384 → 24×24=576 个 patch → 需要 576 个位置编码。

**解决方案**：对原始位置编码做 2D 插值。将 (196, D) reshape 成 (14, 14, D)，用双线性/三次插值放大到 (24, 24, D)，再 flatten 回 (576, D)。

### Q4: 为什么 ViT 用 LayerNorm 而不用 BatchNorm？

1. **Patch 间特征分布差异大**：不同位置（天空、物体、边缘）的 patch 统计量不同，强制共享 BN 的均值和方差不合理
2. **Token 级独立性**：LayerNorm 每个 token 独立归一化，天然适配 Transformer 的序列化结构
3. **train/eval 不一致**：BN 在训练和推理时用不同统计量，对 ViT 的变长 patch 序列不友好

### Q5: few-shot evaluation 中 ridge regression 的闭式解是什么？

```
W = (X^T X + λI)^{-1} X^T y
```

其中 X 是 frozen features，y 是 one-hot labels。这只能在线性回归（平方损失）下求解；分类任务（softmax + CE）无闭式解，需要迭代优化。这也是为什么 few-shot 评估用线性 probe 而不是微调整个分类头——前者衡量 representation quality。

---

## 三、代码产出

| 模块 | 状态 | 说明 |
|------|------|------|
| ViT 论文精读 | ✅ | 脑图 + 数据流图已完成 |
| ViT 代码 walkthrough | ❌ | 明天上午补 |
| CLIP 论文精读 | ❌ | 未开始 |
| CLIP 代码实战 | ❌ | 未开始 |
| Contrastive Loss 推导 | ❌ | 未开始 |
| LeetCode | ⏳ | 待进行 |

---

## 四、尚未解决的问题

1. ViT 代码 walkthrough（明天上午补）
2. CLIP 论文 + 对比学习 loss 推导
3. CLIP 代码推理实战
4. LeetCode 滑动窗口 ×2 题（LC 3, LC 76）

---

## 五、明日预告（Day 4）

**原有内容不变，上午追加 Day 3 未完成项**：

- 上午：ViT/CLIP 代码快速 walkthrough（补 Day 3）
- 下午：SigLIP 论文 + Vision Encoder 对比实践（Day 4 原有）
- 晚间：CLIP vs SigLIP loss 对比推导 + 面试题准备 + LeetCode

---

## 六、额外记录

- ViT 论文核心洞察：CNN 强先验→小数据好，ViT 弱先验→大数据上限高
- Patch 间统计分布差异大是对 ViT 结构理解的很好切入点，解释了 LN 的选择
- ViT 数据流的每个 shape 变化已在脑图中标注
