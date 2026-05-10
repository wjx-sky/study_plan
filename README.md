# 具身智能 VLA + 世界模型 学习路线图

## 个人背景

- **编程**：Python / C++ 熟练，PyTorch 熟练
- **ML/DL**：熟悉 MLP / CNN / RNN / GRU / LSTM 的原理与实现
- **RL/IL**：了解 ACT、Diffusion Policy、A2C 等算法
- **项目经历**：RC 竞赛（轨迹预测方向），独立设计并实现 GRU + 物理模型 hybrid 方案预测排球飞行轨迹，通过 ROS2 部署到真实机器人系统
  - GRU 预测物理参数 → 非线性拟合反推初速度与初始位置
  - 独立设计完整 pipeline（建模 → 训练 → ROS2 通信 → 真机部署）
  - 理解 learning + model-based optimization hybrid 范式

## 你目前的知识盲区（基于你的背景推断）

| 已掌握 | 缺失/薄弱 |
|--------|-----------|
| CNN, RNN, LSTM, GRU（扎实） | **Transformer**（VLA 的绝对基础） |
| ACT, Diffusion Policy 基本了解 | **VLM 基础**（CLIP, SigLIP, LLaVA, Qwen-VL） |
| A2C, 基本 RL/IL | **LLM 基础**（预训练/SFT/RLHF 管线认知即可） |
| PyTorch 熟练 | **VLA 架构全貌**（RT-2, OpenVLA, Octo, π0 等） |
| ROS2 部署 + 真机经验 | **多模态对齐与融合**（visual tokenization, action tokenization） |
| 时序建模 + 物理先验注入 | **具身数据生态**（Open X-Embodiment, 数据混合策略） |
| learning + model-based hybrid | **世界模型**（视频生成 + 具身规划） |

---

## 总览：6 周路线

```
Week 1   ████████  Transformer + VLM 基础补盲
Week 2   ████████  VLA 核心论文精读（RT-2, OpenVLA, Octo）
Week 3   ████████  VLA 代码深挖 + Action Head 范式
Week 4   ████████  前沿 VLA + 动手复现
Week 5-6  ████████  世界模型入门
```

---

## 第一周：Transformer + VLM 基础补盲

> 目标：彻底理解 Transformer 和多模态 VLM 的原理，能手写核心模块。

### Day 1-2：Transformer 原理与代码

- [ ] **论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 精读 Section 3（Attention）, Section 3.2.2（Multi-Head）
- [ ] **必看**：Andrej Karpathy [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)（4h 视频，跟写一遍）
- [ ] **必看**：[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — 对着 PyTorch 代码逐行理解
- [ ] **手写**：从零实现 Multi-Head Self-Attention + Positional Encoding + FFN + LayerNorm（纯 PyTorch，不调库）
- [ ] **理解**：Pre-Norm vs Post-Norm、GQA/MQA（LLaMA 用的 grouped-query attention）、RoPE 位置编码

### Day 3-4：Vision Transformer (ViT) + 多模态对齐

- [ ] **论文**：[ViT](https://arxiv.org/abs/2010.11929) — 理解 image → patch → token 的 pipeline
- [ ] **论文**：[CLIP](https://arxiv.org/abs/2103.00020) — 对比学习，image-text 对齐的核心范式
- [ ] **论文**：[SigLIP](https://arxiv.org/abs/2303.15343) — CLIP 改进版，很多 VLA 在用
- [ ] **代码**：跑通 `open_clip` 或 `transformers.CLIPModel`，理解 vision encoder + text encoder 双塔结构
- [ ] **核心概念**：contrastive loss, temperature parameter, projection head

### Day 5-6：VLM 架构（LLaVA / Qwen-VL / InternVL）

- [ ] **论文**：[LLaVA 1.5](https://arxiv.org/abs/2310.03744) — 理解 visual encoder → projector → LLM 的串联范式
- [ ] **论文**：[Qwen2-VL](https://arxiv.org/abs/2409.12191) — 动态分辨率 + 原生视频理解，当前 VLA 最常用的 VLM backbone 之一
- [ ] **代码**：跑通 LLaVA 推理，理解 `image_features → MLP projector → token_embeds` 的数据流
- [ ] **关键理解**：visual token 和 text token 如何在 LLM embedding space 中对齐
- [ ] **了解**：InternVL / Phi-Vision / Molmo 的架构差异（cross-attention vs self-attention 融合）

### Day 7：LLM 基础认知（够用即可）

- [ ] 理解 Pre-training → SFT → RLHF/DPO 的管线
- [ ] 推荐快速阅读：LLaMA 系列博客（The LLaMA Ecosystem），重点看 LLaMA 3/3.1 的架构
- [ ] 理解 tokenizer（BPE）、chat template、system prompt 在 VLA 中怎么用
- [ ] **VLA 相关性**：VLA 本质上是 LLM/VLM 的延续——你把 action token 当成一种新的"语言"，SFT 的方式训练模型预测 action

---

## 第二周：VLA 核心论文精读

> 目标：熟悉 VLA 的发展脉络和核心范式，能用白板画出主流架构。

### Day 8-9：VLA 奠基论文

- [ ] **[RT-1](https://arxiv.org/abs/2212.06817)**（Google, 2022）：Robotics Transformer
  - TokenLearner + FiLM EfficientNet + Transformer
  - 理解 "把图像历史和语言指令 token 化 → Transformer → action" 的早期范式
  - 关注：ImageNet 预训练 → 机器人微调的迁移策略
- [ ] **[RT-2](https://arxiv.org/abs/2307.15818)**（Google DeepMind, 2023）：Vision-Language-Action Models
  - **VLA 的里程碑论文，面试必问**
  - 核心思路：VLM（PaLI-X / PaLM-E）的 output token → co-fine-tune 为 action token
  - 理解：web-scale VLM knowledge → robot action 的迁移
  - 关键：action tokenization（离散化 action space → token）、chain-of-thought 推理
- [ ] **[PaLM-E](https://arxiv.org/abs/2303.03378)**（Google, 2023）：Embodied Multimodal Language Model
  - 直接把 sensor observations（images, state estimation）注入 LLM
  - 理解：embodied 场景中多模态输入如何拼接

### Day 10-11：开源 VLA 标杆

- [ ] **[OpenVLA](https://arxiv.org/abs/2406.09246)**（Stanford, 2024）：An Open-Source Vision-Language-Action Model
  - **当前最重要的开源 VLA，面试重点**
  - Backbone：Prismatic VLM（SigLIP + DinoV2 双 vision encoder + LLaMA/Phi）
  - 核心贡献：在 Open X-Embodiment 数据集上训练 7B VLA 并开源
  - 重点理解：visual encoder 选择（DinoV2 + SigLIP 为什么比单一 encoder 好）、freeze vs unfreeze 策略
  - 代码：[openvla](https://github.com/openvla/openvla)
- [ ] **[Octo](https://arxiv.org/abs/2405.12213)**（UC Berkeley, 2024）：An Open-Source Generalist Robot Policy
  - 基于 Transformer 的 diffusion policy
  - 支持多机器人、多传感器、多任务的通用策略
  - 理解：observation tokenization + action chunking + diffusion head
  - 代码：[octo-models](https://github.com/octo-models/octo)

### Day 12-13：π0 与 工业级 VLA

- [ ] **[π0](https://arxiv.org/abs/2410.24164)**（Physical Intelligence, 2024）：A Vision-Language-Action Flow Model
  - 当前最强的通用机器人基础模型
  - 理解 flow matching 替代 diffusion 做 action generation
  - 理解 VLM（PaliGemma）+ Action Expert（transformer + flow matching）的混合架构
  - 关注：跨 embodiment 泛化能力
- [ ] **[RDT-1B](https://arxiv.org/abs/2410.07864)**（清华, 2024）：1B 参数的 diffusion-based VLA
  - 理解在 diffusion policy 框架下 scale up 的思路
  - 与 π0 的异同对比

### Day 14：数据生态与方法论

- [ ] **[Open X-Embodiment](https://arxiv.org/abs/2310.08864)** — 理解当前最大的机器人数据集集合
  - 数据混合策略（sample weighting, domain rebalancing）
  - 跨 embodiment 泛化的关键挑战
- [ ] **[EmbodiedChain](https://arxiv.org/pdf/2312.04923)** / 了解具身数据 scaling law 的讨论
- [ ] 快速概览：[RT-X](https://arxiv.org/abs/2310.08864) 中 cross-embodiment 泛化的实验结果

---

## 第三周：VLA 代码深挖 + Visual-Motor Policy 范式

> 目标：能读懂并修改主流 VLA 代码库，理解各模块的实现细节。

### Day 15-16：OpenVLA 代码精读

- [ ] Clone 并跑通 OpenVLA 推理（用已有 checkpoint 控制仿真/真实机器人）
- [ ] 逐文件理解：
  - `modeling_prismatic.py` — VLM backbone（visual encoder → projector → LLM）
  - `modeling_vla.py` — 如何在 VLM 上加 action head
  - Action tokenization：discretized action bins → token → decode 回连续 action
- [ ] 理解训练流程：frozen VLM + LoRA fine-tune action tokens 的策略
- [ ] **必做练习**：在自己的数据上 fine-tune 一个 OpenVLA（用 SimplerEnv 等仿真环境）

### Day 17-18：Action Head 三大范式对比

| 范式 | 代表工作 | 核心思路 | 优缺点 |
|------|---------|---------|--------|
| **Discrete Token** | RT-2, OpenVLA | action 离散化为 token，VLM 直接生成 | 复用 VLM 能力，但精度受离散化限制 |
| **Diffusion Head** | Diffusion Policy, Octo, RDT-1B | 去噪过程生成连续 action | 连续高精度，但推理慢 |
| **Flow Matching** | π0 | 用 flow matching 替代 diffusion | 更快推理，保持精度 |

- [ ] **手写练习**：实现三种 action head 的原型（输入 visual features，输出 action）
  - Discrete: action bins → cross-entropy loss
  - Diffusion: DDPM action head，conditioned on visual features
  - Flow Matching: 简化版 rectified flow，理解与 diffusion 的数学关系

### Day 19-20：Visual-Motor Policy 专项

- [ ] **论文精读**：[Diffusion Policy](https://arxiv.org/abs/2303.04137)（你已经了解，但需要深入实现细节）
  - Action chunking 为什么 work
  - Temporal ensemble 的数学
  - CNN-based vs Transformer-based diffusion head
- [ ] **[ACT](https://arxiv.org/abs/2304.13705)** — 你已了解，回顾重点：
  - Action chunking + CVAE 的架构设计
  - 与 diffusion policy 的对比：什么时候选 ACT 什么时候选 diffusion
- [ ] **[3D Diffusion Policy](https://arxiv.org/abs/2403.05644)** — 3D 表示的 diffusion policy
- [ ] **关键理解**：VLM 提供 semantic grounding → visual-motor policy 负责精细操作。面试常问：VLM part 和 motor policy part 各负责什么，为什么不端到端训一个 big model？

### Day 21：训练策略与工程

- [ ] Co-training / co-fine-tuning 策略：web data + robot data 混合训练
- [ ] LoRA / QLoRA 在 VLA 中的应用
- [ ] 数据集平衡：不同 robot / task / scene 的采样权重
- [ ] Evaluation protocol：CALVIN, LIBERO, SimplerEnv, BridgeData
- [ ] 了解 [SimplerEnv](https://github.com/simpler-env/SimplerEnv) 和 [ManiSkill](https://github.com/haosulab/ManiSkill)

---

## 第四周：前沿 VLA + 动手项目

> 目标：补齐最新进展，完成一个 mini VLA 项目，达到面试 ready。

### Day 22-23：2024-2025 前沿 VLA 跟进

- [ ] **[Gemini Robotics](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)**（Google, 2025.03）
  - Gemini 2.0 + robotics，spatial reasoning + embodied reasoning
- [ ] **[Helix](https://www.figure.ai/news/helix)**（Figure, 2025.02）
  - 人形机器人的 VLA 系统，7B + 7B 双模型架构（System 1 + System 2）
  - 理解快慢双系统的设计哲学
- [ ] **[Groot N1](https://developer.nvidia.com/groot)**（NVIDIA, 2025.03）
  - 双系统架构：VLM for reasoning + diffusion policy for action
- [ ] **快速浏览**：VLA 相关综述 [Toward Generalist Robot Policies](https://arxiv.org/abs/2412.01306)

### Day 24-25：Mini VLA 项目（必做）

动手实现一个 mini VLA pipeline，最直接的方式是用 OpenVLA + 仿真环境：

```
任务：给定语言指令，控制机械臂完成 pick-and-place
Pipeline：
  1. 摄像头采集 RGB 图像
  2. VLM（冻结的 Prismatic/SigLIP+LLaMA）提取 visual features
  3. LoRA-finetuned action head 输出 7-DoF action
  4. 在 SimplerEnv 中执行并评估
```

- [ ] 在 SimplerEnv 的 Google Robot 环境中跑通 OpenVLA 推理
- [ ] 在 BridgeData V2 的一个子集上 LoRA fine-tune OpenVLA
- [ ] 记录训练曲线，理解 loss 收敛行为
- [ ] 录制视频展示效果

### Day 26-27：面试专题准备

#### 必须能口头回答的核心问题：

1. **RT-2 怎么把 VLM 变成 VLA？** Action tokenization + co-fine-tuning 的具体做法
2. **OpenVLA 的架构设计选择？** 为什么用双 vision encoder（SigLIP + DinoV2）+ LLaMA？prismatic VLM 的 projector 怎么设计？
3. **Discrete action vs continuous action（diffusion/flow）各自的优劣？** 什么时候用什么？
4. **VLA 的泛化能力从哪来？** Internet-scale pre-training 的作用 vs robot data fine-tuning
5. **Action chunking 为什么有效？** Temporal consistency + 减少 compounding error
6. **VLA 与 traditional IL（ACT, Diffusion Policy）的本质区别？** VLM 提供 semantic grounding + language grounding
7. **如何评估一个 VLA？** 用什么 benchmark，什么 metric
8. **数据混合策略？** Open X-Embodiment 怎么处理不同 robot/domain 的数据不平衡
9. **π0 的 flow matching 与 diffusion 的区别？** 为什么更快？
10. **VLA 的推理延迟怎么优化？** Quantization, speculative decoding, action chunk 预测

### Day 28：复盘 + 查漏补缺

- [ ] 用费曼学习法：找一面白板，画出 RT-2 / OpenVLA / π0 的架构图并讲解
- [ ] 整理自己的 notes，形成 VLA 知识体系图谱
- [ ] 浏览 [VLA 领域 GitHub Awesome List](https://github.com/JackyLiang9060/awesome-vision-language-action)，确认无重大遗漏

---

## 第五-六周：世界模型入门

> 目标：理解世界模型的核心范式及其与 VLA 的关系。这 15 天的目标是"建立认知框架 + 能跑通关键代码"，而非面面俱到。

### Day 29-30：世界模型概念与奠基

- [ ] **[World Models](https://arxiv.org/abs/1803.10122)**（Ha & Schmidhuber, 2018）— 世界模型的奠基作
  - VAE 编码观测 → RNN 建模 dynamics → controller 输出 action
  - 虽然是 2018 年的老工作，但概念框架至今通用
- [ ] **论文**：[DreamerV3](https://arxiv.org/abs/2301.04104)（DeepMind, 2023）
  - 当前 model-based RL 的 SOTA 世界模型
  - RSSM（Recurrent State Space Model）架构
  - 理解：world model 如何在 latent space 中做 rollout
- [ ] **区分概念**：
  - Model-based RL 中的 World Model（Dreamer 系列）
  - 视频生成式 World Model（Sora, Genie 系列）
  - 两者的联系与区别

### Day 31-33：视频生成式世界模型

- [ ] **[Genie](https://arxiv.org/abs/2402.14775)**（Google DeepMind, 2024）：Generative Interactive Environments
  - 从无标注视频中学习可交互的世界模型
  - latent action space + video tokenizer + dynamics model
- [ ] **[Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)**（Google DeepMind, 2024.12）
  - Scale up 到 3D 场景生成
- [ ] **[UniSim](https://arxiv.org/abs/2312.03271)**（Google DeepMind / UC Berkeley, 2023）
  - Universal simulator via video generation
  - 用世界模型做具身任务的 planning
- [ ] **理解核心组件**：
  - Video tokenizer（VQ-VAE / VQ-GAN / MAGVIT）
  - Autoregressive / diffusion dynamics model
  - Action-conditioned video prediction

### Day 34-36：视频生成基础 + Sora

- [ ] **[Sora](https://openai.com/index/sora/)** + 技术报告阅读
  - patch-based video representation
  - diffusion transformer（DiT）做 video generation
  - 理解为什么视频生成能力 ≈ 物理世界理解能力
- [ ] **[DiT](https://arxiv.org/abs/2212.09748)**（Meta, 2022）：Diffusion Transformer
  - Sora 和很多世界模型的 backbone
  - 代码：[DiT 官方实现](https://github.com/facebookresearch/DiT)
- [ ] **[Video Diffusion Models](https://arxiv.org/abs/2204.03458)**（Google, 2022）— 视频扩散的奠基作
- [ ] **跑通代码**：DiT 训练 + 推理（用较小规模的数据）

### Day 37-38：具身世界模型 + 与 VLA 的关系

- [ ] **[UniPi](https://arxiv.org/abs/2312.03272)**（Google, 2023）
  - Universal policy via text-to-video generation
  - VLM/VLA → 生成 plan video → 提取 action
  - 这是 VLA 和世界模型的交汇点
- [ ] **[SuSIE](https://arxiv.org/abs/2310.09122)**（UC Berkeley, 2023）
  - 用图像编辑做 subgoal generation
- [ ] **关键理解**：世界模型在具身智能中的作用
  - 提供 "what if" 的模拟能力
  - 辅助 long-horizon planning
  - 弥补真实数据不足（synthetic data generation）
- [ ] **思考题**（面试常考）：
  - VLA + 世界模型如何结合？
  - 世界模型用于 data augmentation 的具体方案？
  - 为什么现在的世界模型还不能替代真实交互？

### Day 39-42：动手项目 + 面试准备

- [ ] 跑通 DreamerV3 在 DMControl 上的训练
- [ ] 阅读 [IRIS](https://arxiv.org/abs/2210.14496) — 离散 token 世界模型 + 规划
- [ ] 理解 [Cosmos](https://developer.nvidia.com/cosmos)（NVIDIA 世界模型平台）的 API 和用例
- [ ] **世界模型面试核心问题**：
  1. DreamerV3 的 RSSM 怎么 work？Deterministic + stochastic state 的设计
  2. 视频生成模型作为世界模型的局限性？
  3. 如何评估世界模型的质量？
  4. VLA 与 world model 的分工：VLA 做 reactive control，world model 做 planning
- [ ] **展望阅读**：读 2-3 篇 2025 年最新的具身世界模型论文（arXiv 搜索 embodied world model 2025）

---

## 学习习惯与方法论

### 每日节奏

```
09:00-11:00  论文精读（做笔记，写 summary）
11:00-11:30  关键论文的代码阅读
11:30-12:00  LeetCode ×1-2 题（每天固定，不跳过）
14:00-17:00  写代码 / 跑实验
19:00-20:30  整理笔记 + 输出（博客/Notion）
20:30-21:00  LeetCode ×1-2 题
```

### LeetCode 路线（2 个月，目标 150 题）

与大厂实习面试直接相关，每天 1 小时（早上 30min + 晚上 30min），日均 2-4 题。

**刷题顺序**（按面试出现频率从高到低）：

| 阶段 | 专题 | 题量 | 时间 |
|------|------|------|------|
| 第 1-2 周 | 数组/字符串/双指针/滑动窗口 | ~40 题 | 基础必刷 |
| 第 3-4 周 | 哈希表/栈/队列/链表 | ~35 题 | 高频 |
| 第 5-6 周 | 树/递归/DFS/BFS | ~35 题 | 高频 |
| 第 7-8 周 | DP/贪心/回溯/图 | ~40 题 | 拉开差距 |

**刷题原则**：
- Easy 5 min 内出思路 → 跳过，只刷 Medium + 少量 Hard
- 每道题先自己想 10 min，没思路直接看题解，理解后手写一遍
- 同一题 3 天后重做，做不出来说明没真懂
- 优先刷 [LeetCode Hot 100](https://leetcode.cn/problem-list/2cktkvj/) 和 [剑指 Offer](https://leetcode.cn/problem-list/xb9nqhhg/)

### 论文阅读方法（三遍法）

1. **第一遍（30 min）**：Abstract → 所有 Figure/Table → Conclusion。判断论文重要性和核心贡献。
2. **第二遍（1-2 h）**：完整通读，理解 method 的技术细节，标注不懂的部分。
3. **第三遍（1 h）**：用自己的话写出 method 的 pipeline，画出架构图，列出 3 个 takeaway。

### 代码学习方法

- 不要只读，一定要跑起来。优先从 `demo.py` / `eval.py` 入口
- 用 debugger 打断点，追踪一个 batch 的 tensor shape 变化
- 至少修改一个模块（换 backbone / 换 action head / 换数据集）并观察结果

### 输出驱动

- 每周末写一篇本周学习总结（中英文皆可）
- 把笔记整理成面试 Q&A 格式
- 可以考虑录一个 10 分钟的 paper walkthrough 视频（加深记忆）

---

## 推荐资源汇总

### 课程

| 课程 | 链接 | 用途 |
|------|------|------|
| CS25: Transformers United (V3) | [Stanford CS25](https://web.stanford.edu/class/cs25/) | Transformer 全方位理解 |
| CS285: Deep RL | [UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/) | RL 基础（你已经了解，可快进到 model-based 部分） |
| CS224N: NLP with Deep Learning | [Stanford](https://web.stanford.edu/class/cs224n/) | Transformer + LLM 基础 |
| Robot Learning (CS294-277) | [UC Berkeley FA24](https://robots-learning.com/) | 具身智能专项课程 |

### 代码库（必 Clone）

| 仓库 | 用途 |
|------|------|
| [OpenVLA](https://github.com/openvla/openvla) | VLA 入门首选框架 |
| [Octo](https://github.com/octo-models/octo) | Transformer + Diffusion VLA |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Action head 核心实现 |
| [SimplerEnv](https://github.com/simpler-env/SimplerEnv) | VLA 评估仿真环境 |
| [DiT](https://github.com/facebookresearch/DiT) | Diffusion Transformer |
| [DreamerV3](https://github.com/danijar/dreamerv3) | 世界模型代码 |

### 必读论文速查

```
VLA:
  RT-1 (2022)          → Brohan et al.        → Robotics Transformer 开篇
  RT-2 (2023)          → Brohan et al.        → VLA 里程碑 ⭐
  PaLM-E (2023)        → Driess et al.        → 多模态具身 LLM
  Octo (2024)          → Octo Model Team      → 开源通用策略 ⭐
  OpenVLA (2024)       → Kim et al.           → 开源 VLA 标杆 ⭐
  π0 (2024)            → Black et al.         → Flow matching VLA ⭐
  RDT-1B (2024)        → Liu et al.           → 大规模 diffusion VLA
  Open X-Embodiment    → Padalkar et al.      → 数据生态 ⭐

世界模型:
  World Models (2018)  → Ha & Schmidhuber    → 奠基
  DreamerV3 (2023)     → Hafner et al.        → Model-based RL ⭐
  Genie (2024)         → Bruce et al.         → 视频生成世界模型 ⭐
  UniSim (2023)        → Yang et al.          → 视频生成做规划
  DiT (2023)           → Peebles & Xie        → Diffusion backbone
  UniPi (2023)         → Du et al.            → VLA + 世界模型 交汇 ⭐
```

---

## 面试 Checklist（最终自检）

### VLA 部分

- [ ] 能画出 RT-2 的完整架构图，解释 action tokenization
- [ ] 能对比 OpenVLA、Octo、π0 三者的架构差异
- [ ] 能解释 discrete action token vs diffusion action head vs flow matching 的数学与工程 trade-off
- [ ] 能讲清楚 VLM backbone 的选择（为什么用 SigLIP+DinoV2？为什么不用纯 CLIP？）
- [ ] 能回答 "VLA 的泛化为什么好" 的底层原因
- [ ] 动手 fine-tune 过至少一个 VLA 并调过参
- [ ] 了解数据混合策略和 evaluation protocol

### 世界模型部分

- [ ] 能区分 model-based RL 世界模型 vs 视频生成式世界模型
- [ ] 能画出 DreamerV3 RSSM 的结构
- [ ] 能解释世界模型如何辅助 VLA 做 long-horizon planning
- [ ] 能说出当前世界模型的 3 个主要局限
- [ ] 跑通过至少一个世界模型代码（DreamerV3 或 DiT）
