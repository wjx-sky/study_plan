# Day 7：LLM 基础认知 + 第一周收官

## 今日目标

补齐 LLM 训练管线的基本认知——不需要深入 NLP 细节，但必须理解 VLA 训练中复用的 SFT 范式。同时完成第一周的知识整理与查漏补缺。

---

## 上午（09:00 - 12:00）：LLM 训练管线

### 1. LLM 训练三阶段（1.5h）

**推荐阅读**：[A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — 仅读 Section 4（Training）、Section 5（Adaptation）

**理解三个阶段**：

| 阶段 | 数据 | 目标 | 在 VLA 中的对应 |
|------|------|------|----------------|
| **Pre-training** | 万亿 token 级文本 | 学习语言统计规律 | VLM 的预训练（已经帮你做了） |
| **SFT** | 指令-回答 pair | 学会遵循指令格式 | **VLA 的核心训练阶段**（教模型输出 action） |
| **RLHF/DPO** | 偏好比较数据 | 对齐人类偏好 | 目前 VLA 中较少使用（前沿方向） |

**关键理解**：
- [ ] VLA 本质上是把 action 当作一种新的输出模态，通过 SFT 教 VLM 生成
- [ ] 所以 VLA 训练 ≈ VLM 的 SFT 阶段，只是输出从 text 变成了 action tokens
- [ ] RLHF 在 VLA 中的对应是 RL fine-tuning（用真实交互奖励信号），但目前还在早期

### 2. LeetCode（11:30 - 12:00）

- [ ] 树/递归 ×2 题
- 推荐今日题目：Maximum Depth of Binary Tree (LC 104), Invert Binary Tree (LC 226)

---

### 3. Tokenizer 与 Chat Template（1h）

- [ ] 理解 BPE tokenizer 的原理（subword tokenization）
- [ ] 理解 Chat Template 的作用：
  ```
  <|system|> You are a helpful robot assistant. </s>
  <|user|> Pick up the red block. </s>
  <|assistant|> [action_tokens] </s>
  ```
- [ ] **VLA 相关性**：action token 怎么加入 tokenizer 的 vocab？
  - 方法 1：扩词表（新增 action tokens → resize embedding）
  - 方法 2：复用已有 tokens（用特殊 token 表示 action 值）
  - OpenVLA 用方法 1：定义 `256 * action_dim` 个 action bins 作为特殊 token

### 4. LLaMA 架构速览（0.5h）

- [ ] 阅读 [LLaMA 论文](https://arxiv.org/abs/2302.13971) 的关键改进：
  - Pre-Norm（RMSNorm）
  - SwiGLU activation
  - RoPE
- [ ] 对比 Day 2 手写的原始 Transformer 和 LLaMA 的差异

---

## 下午（14:00 - 17:00）：第一周知识图谱

### 5. 绘制 VLA 前置知识图谱（2h）

用一张大图整理本周学到的所有概念之间的关系：

```
                    ┌─────────────────────┐
                    │    Transformer       │
                    │  (Day 1-2)           │
                    │  MHA, FFN, Norm      │
                    └──────┬──────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                 ▼
   ┌──────────┐    ┌──────────────┐   ┌──────────┐
   │   ViT    │    │    CLIP      │   │  LLaMA   │
   │ (Day 3)  │    │  (Day 3)     │   │ (Day 7)  │
   │ Patch    │    │  对比学习     │   │  RoPE,   │
   │ Embed    │    │  双塔架构     │   │  GQA     │
   └────┬─────┘    └──────┬───────┘   └────┬─────┘
        │                 │                 │
        └────────┬────────┘                 │
                 ▼                          │
         ┌──────────────┐                   │
         │ SigLIP/DinoV2│                   │
         │   (Day 4)    │                   │
         └──────┬───────┘                   │
                │                           │
                └──────┬────────────────────┘
                       ▼
              ┌────────────────┐
              │    VLM         │
              │ (Day 5-6)      │
              │ LLaVA/Qwen2-VL │
              │ Projector + LLM│
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │     VLA        │
              │  (Week 2 →)    │
              │ VLM + Action   │
              └────────────────┘
```

### 6. 查漏补缺（1h）

- [ ] 重读第一周每天的 summary.md，看"尚未解决的问题"是否已解决
- [ ] 快速过一遍 README 中第一周的 Checklist，确认无遗漏
- [ ] 列出 3 个你仍然模糊的概念，去查资料/看视频解决

---

## 晚间（19:00 - 21:00）：第一周总结 + LeetCode

### 7. 写"第一周学习总结"（2h）

用一篇完整的技术文档总结本周学习，建议包含：

- [ ] **Transformer 核心组件**（1 页）：公式 + 架构图 + 关键 shape 变化
- [ ] **视觉编码器对比**（1 页）：ViT / CLIP / SigLIP / DinoV2 的表格对比
- [ ] **VLM 架构范式**（1 页）：LLaVA 全流程 + 与其他 VLM 的差异
- [ ] **VLA 前置知识自检**：以下概念是否都能解释清楚？
  - Multi-Head Self-Attention
  - Pre-Norm vs Post-Norm
  - CLIP contrastive loss vs SigLIP sigmoid loss
  - Vision tokenization (image → patches → tokens)
  - VLM projector 的作用
  - LLM 训练三阶段
  - Action tokenization 的基本思路
- [ ] LeetCode ×1-2 题（树/递归继续）

### 8. 二周预习

- [ ] 下载 [RT-2 论文](https://arxiv.org/abs/2307.15818)，快速浏览 Abstract + Figure 1-3
- [ ] Clone [OpenVLA 仓库](https://github.com/openvla/openvla)，确认环境能跑通

---

## 今日 Checklist

- [ ] 理解 LLM 训练三阶段及其在 VLA 中的对应
- [ ] 理解 action token 如何加入 tokenizer vocab
- [ ] 完成第一周知识图谱
- [ ] 所有 Day 1-6 "尚未解决的问题"都已解决或记录在案
- [ ] 完成第一周技术总结文档
- [ ] 预习 RT-2 摘要至少读完
