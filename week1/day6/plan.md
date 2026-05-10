# Day 6：VLM 全景对比 —— Qwen2-VL / InternVL / Phi-Vision

## 今日目标

建立 VLM 选型视野。VLA 的 backbone 不只有 LLaVA，理解不同 VLM 的架构差异，为第二周理解各 VLA 为什么选某个 backbone 做准备。

---

## 上午（09:00 - 12:00）：Qwen2-VL 精读

### 1. Qwen2-VL 论文

**论文**：[Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)

**重点理解**：
- [ ] **动态分辨率（Dynamic Resolution）**：
  - 传统 VLM 把图片 resize 到固定尺寸 → 小物体丢失
  - Qwen2-VL 把高分辨率图切成多个子图 + 全局缩略图 → 保留细节
- [ ] **Naive Dynamic Resolution vs Qwen2-VL 的改进**
- [ ] Native Video Understanding：视频帧如何输入 VLM
- [ ] **为什么 Qwen2-VL 在 VLA 中受欢迎？**
  - 开源、性能强、动态分辨率适合机器人场景（需要看清操作物体）
  - 支持多分辨率输入 → 适合操作任务中的细小物体识别

### 2. Qwen2-VL 代码速览

用 `transformers` 加载：
```python
from transformers import Qwen2VLForConditionalGeneration

# 关注：
# - visual 模块如何处理多分辨率输入
# - 与 LLaVA 架构的差异
```

---

## 下午（14:00 - 17:00）：VLM 架构对比

### 3. 三/四种 VLM 架构横向对比

| 特性 | LLaVA 1.5 | Qwen2-VL | InternVL2 | Phi-Vision |
|------|-----------|----------|-----------|------------|
| Vision Encoder | CLIP ViT-L | ViT-bigG | InternViT | SigLIP |
| Projector | 2-layer MLP | MLP | MLP + Pixel Shuffle | Cross-Attn |
| LLM | Vicuna | Qwen2 | InternLM2 | Phi-3 |
| 动态分辨率 | ✗ | ✓ | ✓ | ✓ |
| 开源程度 | 完全开源 | 完全开源 | 完全开源 | 权重开源 |

- [ ] 理解 "cross-attention vs self-attention" 两种融合范式：
  - Self-attention fusion（LLaVA 方式）：visual token 和 text token 一起做 self-attention
  - Cross-attention fusion（Flamingo/Phi-Vision 方式）：text token 通过 cross-attn 查询 visual feature

### 4. 动手代码练习

- [ ] 跑通 Qwen2-VL 推理
- [ ] 测试同一张图在 Qwen2-VL 和 LLaVA 上的效果差异
- [ ] **关键练习**：用一张机器人操作场景图（从 BridgeData 或网上找），测试两个 VLM 的描述能力

---

## 晚间（19:00 - 21:00）：总结 + 预备 VLA

- [ ] 写一篇笔记：**"VLA 对 VLM backbone 的需求分析"**
  - 需要强 spatial reasoning（知道物体在哪）
  - 需要好的 fine-grained 识别（操作对象可能很小）
  - 需要高效的推理速度（机器人需要实时）
  - 哪些 VLM 特性最匹配这些需求？

- [ ] **预习**：快速浏览 [OpenVLA 论文](https://arxiv.org/abs/2406.09246) 的 Section 2-3（明天正式学 VLA 时你会感谢今晚的预习）

---

## 今日 Checklist

- [ ] 理解动态分辨率解决了什么问题
- [ ] 能对比 LLaVA vs Qwen2-VL 的架构差异
- [ ] 理解 self-attention vs cross-attention 两种融合范式
- [ ] 跑通过至少两个不同的 VLM 的推理
- [ ] 完成"VLA 对 VLM backbone 的需求分析"笔记
