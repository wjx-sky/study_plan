# Day 4 学习总结

> 日期：2026 年 5 月 19 日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 CLIP vs SigLIP 损失函数对比

| | CLIP | SigLIP |
|------|------|--------|
| Loss 类型 | InfoNCE (softmax) | Sigmoid + BCE |
| Batch 依赖 | 需要大 batch（32K+） | 小 batch 友好 |
| 计算方式 | batch 内全局归一化 | per-pair 独立 |
| 正负样本关系 | 耦合（分母共享） | 解耦（独立打分） |
| 梯度稳定性 | batch 小梯度消失 | 与 batch 大小无关 |

### 1.2 Loss 公式

```python
# CLIP InfoNCE（对称版）
logits = I @ T.T * exp(t)         # (N, N)
L_i2t = CE(logits, labels=range(N), dim=1)  # image → text
L_t2i = CE(logits, labels=range(N), dim=0)  # text → image
L = (L_i2t + L_t2i) / 2

# CLIP 本质问题：softmax 分母 Σ_j exp(S_ij) 将所有负样本耦合在一起
# 每个 p_i = exp(S_ii)/Σexp(S_ij) 依赖于整个 batch

# SigLIP sigmoid BCE
logits = I @ T.T * exp(t) + bias   # (N, N)
labels = 2 * eye(N) - 1            # 对角线 = 1（正），其余 = -1（负）
L = -log(sigmoid(labels * logits)).mean()

# SigLIP 核心：每个 (i,j) pair 独立计算 BCE，batch 内互不干扰
```

### 1.3 梯度差异深度分析（Day 4 核心）

**场景**：batch N=3，图文对 (I₁,T₁), (I₂,T₂), (I₃,T₃)

**CLIP 梯度（image → text 方向）**：

```
设 p₁ⱼ = softmax(S₁/τ)ⱼ

∂L/∂S₁₁ = (p₁₁ - 1) / τ    ← 正样本，梯度取决于 p₁₁
∂L/∂S₁₂ = p₁₂ / τ           ← 负样本
∂L/∂S₁₃ = p₁₃ / τ           ← 负样本
```

因为 `p₁₁ + p₁₂ + p₁₃ = 1`，正样本梯度完全被负样本参与度牵制：
- N=8 时，分母只有 8 项，几个 epoch 后 `p₁₁→1`，其余全逼近 0 → **梯度消失**，模型轻松"蒙对"，学到的东西不深
- N=32768 时，32767 个负样本里总有语义相近的假阴性，`p₁₁` 到不了 1 → 梯度持续有信号 → 逼着模型学精细语义

**SigLIP 梯度**：

```
∂L/∂S₁₁ = 1 - σ(S₁₁)       ← 只取决于 S₁₁ 自己
∂L/∂S₁₂ = -σ(S₁₂)           ← 只取决于 S₁₂ 自己
∂L/∂S₁₃ = -σ(S₁₃)           ← 只取决于 S₁₃ 自己
```

三个梯度**完全独立**。N=4 还是 N=32768，梯度一模一样。

**梯度公式对比速查**：

```
CLIP:   ∂L/∂S正 = p正 - 1   ← p正 被分母里所有负样本稀释
                             负样本越多 → p正 越小 → 梯度越大
                             → 强依赖大 N

SigLIP: ∂L/∂S正 = 1 - σ(S正) ← 只取决于正样本自己
                             跟 N 完全无关
```

**为什么 fine-tune 选 SigLIP**：

机器人场景 20 个训练样本，batch=8。CLIP 几个 epoch 后 p→1，梯度消失，学不动了。SigLIP 每个 pair 照常产出梯度，大小 batch 通吃。

> 核心直觉：**CLIP = 批内竞争，战场越大越激烈学得越好；SigLIP = 一对一独立打分，战场规模与 batch 无关。**

### 1.4 Label Smoothing in SigLIP

CLIP 负样本标签 = 硬 0（完全不相关），但有些负样本语义上其实相关，只是没配对（假阴性）。

SigLIP 引入 label smoothing：把负样本标签从 -1 改成 -0.9 之类的小负值 → "我不确定你们完全不相关，只是不太可能"。缓解假阴性对训练的干扰。

### 1.5 三大 Vision Encoder 定位

| Encoder | 训练方式 | 擅长 | 适用 |
|---------|---------|------|------|
| CLIP ViT | 图文对比（InfoNCE） | 语义理解、图文对齐 | zero-shot 分类、检索 |
| SigLIP ViT | 图文对比（BCE） | 同上 + 小 batch 稳定 | fine-tune、机器人 |
| DinoV2 ViT | **纯图像自监督**（无文本） | 空间结构、纹理、边界 | 需要几何理解的场景 |

DinoV2 不看文字，只靠"图片里相邻 patch 长得像"自监督学习。特征不与语言对齐，但**几何结构清晰**——知道物体在哪、边界在哪。

### 1.6 OpenVLA 为什么用 SigLIP + DinoV2 双 encoder

- **SigLIP** 提供语言对齐的语义信息——"这是什么"
- **DinoV2** 提供空间几何信息——"在哪儿、什么形状"

机器人抓取需要同时知道两者。只用 SigLIP：知道是杯子但不知道精确位置 → 抓不住。只用 DinoV2：知道物体边界但不知道叫啥名字 → 听不懂语言指令。

---

## 二、代码产出

| 模块 | 状态 |
|------|------|
| CLIP encoder 推理 | ❌ |
| SigLIP encoder 推理 | ❌ |
| DinoV2 encoder 推理 | ❌ |
| 三种特征 cosine similarity | ❌ |

> 代码部分全面放弃，仅保留论文理解。

---

## 三、面试自测

**Q：CLIP 和 SigLIP 的区别？什么时候用哪个？**

CLIP: InfoNCE（softmax），batch 内全局归一化，需要大 batch（32K+）提供充足负样本。零样本分类 / 检索场景。
SigLIP: Sigmoid + BCE，per-pair 独立，大小 batch 通用。Fine-tune 小数据 + 需稳定梯度时用。
核心差异可视化：`CLIP 梯度 ∂L/∂S正 = p正-1`（含 N-1 负竞争项），`SigLIP 梯度 ∂L/∂S正 = 1-σ(S正)`（只看自己）。

**Q：DinoV2 为什么和 SigLIP 互补？**

SigLIP 对齐语言→语义强；DinoV2 自监督→几何结构强。机器人需要"知道这是什么 + 知道它在哪里"。SigLIP 做前一半，DinoV2 做后一半，concat 后喂给 LLM。

**Q：为什么小 batch 下 CLIP 几个 epoch 后 p→1？**

N=8 时 softmax 只有 8 项，模型快速学会让正样本分数高于 7 个负样本，p→1，梯度消失。这不是学好了——负样本太少区分度低，模型根本没学到深层语义。大 batch 时 3 万负样本里总有语义相近的假阴性，p 永远到不了 1，梯度持久。

---

## 四、尚未解决的问题

1. LeetCode（Day 3 + Day 4 题目均未推进）
2. Vision encoder 代码实战（放弃）
3. OpenCLIP 上手（放弃）

---

## 五、明日预告

- Day 5：LLaVA 1.5 论文精读
- 理解 visual encoder → projector → LLM 的串联范式
- 这是 VLA 架构的直接前置知识
- Day 3/4 论文学习完成，ViT/CLIP/SigLIP 基础已就位

---

## 六、额外记录

- CLIP → SigLIP 的演进本质：从"全局竞争"到"独立打分"的一步解耦
- 梯度公式对比是今天核心收获：`p-1` vs `1-σ`
- DinoV2 没读论文，仅建立了概念认知
