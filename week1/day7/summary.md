# Day 7 学习总结 + 第一周复盘

> 日期：____年____月____日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 LLM 训练三阶段

| 阶段 | 方法 | 数据 | 在 VLA 中 |
|------|------|------|-----------|
| Pre-training | 自回归语言建模 | 万亿 token 文本 | = VLM 预训练 |
| SFT | 指令微调 | 指令-回答 pair | **VLA 的核心训练** |
| RLHF/DPO | 偏好对齐 | 人类偏好 | 前沿探索中 |

### 1.2 Action Tokenization

**离散化方案**：
```
连续 action 空间 [-1, 1] → 均匀分 256 个 bin
每个 bin → 一个特殊 token (如 <act_0_128>)
VLA 输出这些 token → 解码回连续值
```

---

## 二、第一周知识图谱

（把今天下午画的完整知识图谱贴在这里，或描述关键连接）

```
Transformer → ViT → CLIP/SigLIP → VLM (LLaVA/Qwen2-VL)
                                    ↓
                               LLaMA/SFT → VLA Token → Action
```

---

## 三、第一周完整自检

### 概念理解（能不看资料讲出来才算通过）

- [ ] Multi-Head Self-Attention 的公式和物理含义
- [ ] Pre-Norm vs Post-Norm 及其对训练稳定性的影响
- [ ] GQA 如何减少 KV cache
- [ ] RoPE 为什么比绝对位置编码好
- [ ] CLIP 的 contrastive loss 推导
- [ ] SigLIP 的 sigmoid loss 及其优势
- [ ] ViT 的图像 tokenization 流程（patch embedding）
- [ ] LLaVA 的 visual encoder → projector → LLM 架构
- [ ] 动态分辨率解决了什么问题
- [ ] LLM 训练三阶段：Pre-training → SFT → RLHF
- [ ] Action tokenization 基本思路
- [ ] VLM 和 VLA 的架构关系

### 代码能力

- [ ] 能手写 MultiHeadAttention（PyTorch）
- [ ] 跑通过 CLIP / SigLIP / DinoV2 特征提取
- [ ] 跑通过 LLaVA 推理并追踪过 shape 变化
- [ ] 跑通过 Qwen2-VL 推理

### 未解决/不牢固的问题

1. 
2. 
3. 

---

## 四、第二周准备

- [ ] RT-2 论文已下载
- [ ] OpenVLA 环境已配好
- [ ] 对 VLA 的理解："VLM backbone + action tokenization + SFT"

---

## 五、额外记录（一周总感想 / 学习调整）

