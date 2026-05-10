# Day 5：LLaVA 1.5 —— VLM 架构范式

## 今日目标

彻底理解 LLaVA 的架构："visual encoder → MLP projector → LLM"，这是大多数 VLA（包括 OpenVLA）的 backbone 形态。

---

## 上午（09:00 - 12:00）：LLaVA 论文精读

### 1. LLaVA 1.5 论文精读（2h）

**论文**：[Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)

**重点理解**：
- [ ] 核心架构：`Vision Encoder (CLIP) → MLP Projector (2-layer) → LLM (Vicuna/LLaMA)`
- [ ] Visual token 和 text token 如何拼接：
  ```
  [<image> tokens (576)] + [system prompt] + [user text] → LLM → response
  ```
- [ ] Projector 的三种设计（LLaVA 1.0 → 1.5 的改进）：
  - 1-layer linear（太简单）
  - 2-layer MLP + GELU（LLaVA 1.5 的选择）
  - Q-Former (BLIP-2 的方式，复杂但更强)
- [ ] 训练两阶段策略：
  - Stage 1: Pre-training for Feature Alignment（只训 projector）
  - Stage 2: Visual Instruction Tuning（训 projector + LLM）
- [ ] Data：CC3M 595K（Stage 1），LLaVA-Instruct-665K（Stage 2）

### 2. LeetCode（11:30 - 12:00）

- [ ] 链表专题 ×2 题
- 推荐今日题目：Reverse Linked List (LC 206), Merge Two Sorted Lists (LC 21)

---

### 3. 画出 LLaVA 前向图（1h）

必须能徒手画出：
```
┌─────────────┐    ┌──────────────┐    ┌───────────┐
│ Image       │ → │ CLIP ViT-L   │ → │ 576 tokens│
│ (336×336)   │    │ (24 layers)  │    │ (1024-d)  │
└─────────────┘    └──────────────┘    └─────┬─────┘
                                             │
                                    ┌────────▼────────┐
                                    │ MLP Projector   │
                                    │ (1024 → 4096 →  │
                                    │  4096, GELU)    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                  Text Tokens → ──→ │  LLM (Vicuna)   │ → Response
                                    └─────────────────┘
```

---

## 下午（14:00 - 17:00）：代码深挖

### 4. LLaVA 代码精读（2h）

**仓库**：[LLaVA](https://github.com/haotian-liu/LLaVA) 或 transformers 中的 `LlavaForConditionalGeneration`

**逐模块理解**：

```python
from transformers import LlavaForConditionalGeneration

# 关键模块：
# model.vision_tower        — CLIP ViT
# model.multi_modal_projector — MLP (2-layer)
# model.language_model      — LLaMA/Vicuna

# 追踪一个 batch 的 shape 变化：
# image: (B, 3, 336, 336)
# → vision_features: (B, 576, 1024)    # CLIP penultimate layer
# → projected: (B, 576, 4096)          # projector 对齐到 LLM hidden_dim
# → inputs_embeds: (B, 576 + L, 4096)  # visual tokens + text tokens
# → outputs: (B, 576 + L, 4096)        # 经过 LLM
```

### 5. 动手跑 LLaVA 推理（1h）

- [ ] 用 transformers 跑 LLaVA 1.5 对一个场景图 + 问题的推理
- [ ] 修改 projector 的结构（加一层 → 3 层 MLP），观察输出是否变化
- [ ] 打印 visual token 在前向传播中的 attention pattern（观察 LLM 如何"看"图像）

---

## 晚间（19:00 - 21:00）：VLM 全景 + LeetCode

- [ ] 快速浏览 BLIP-2 的 Q-Former 设计（与 LLaVA projector 的对比）
- [ ] 理解 VLM 的 evaluation benchmark：MMBench, MME, SEED-Bench, MMMU
- [ ] **思考题**：如果把 LLaVA 的 LLM 输出从 text 改成 action token，是不是就是一个简单的 VLA？
- [ ] LeetCode ×1-2 题（继续链表专题，快慢指针）

---

## 今日 Checklist

- [ ] 能徒手画出 LLaVA 的架构图
- [ ] 理解 projector 的作用（视觉空间 → LLM 空间的维度和语义对齐）
- [ ] 理解两阶段训练策略的动机
- [ ] 跑通 LLaVA 推理，追踪过 shape 变化
- [ ] 能回答："LLaVA 和 VLA 的架构关系是什么？"
