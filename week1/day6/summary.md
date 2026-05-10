# Day 6 学习总结

> 日期：____年____月____日
> 学习时长：____小时

---

## 一、今日核心收获

### 1.1 VLM 架构全景对比

（画出四种 VLM 的简化架构图，标注关键差异）

### 1.2 两种融合范式

| | Self-Attention Fusion | Cross-Attention Fusion |
|------|------|------|
| 代表 | LLaVA, Qwen2-VL | Flamingo, Phi-Vision |
| 机制 | visual+text 一起 self-attn | text 通过 cross-attn 查 visual |
| 优点 | 简单、高效 | 可插入任意 LLM |
| 缺点 | 需要维度对齐 | 额外参数、稍慢 |

### 1.3 VLA 对 VLM backbone 的需求

1. 
2. 
3. 
4. 

---

## 二、代码产出

- [ ] Qwen2-VL 推理跑通
- [ ] LLaVA vs Qwen2-VL 对比实验

### 核心发现
```
// 用机器人场景图测试两个 VLM 的对比结果
```

---

## 三、面试自测

**Q：为什么 VLA 需要好的视觉 backbone？CLIP 够用吗？**

> 你的回答：

**Q：动态分辨率在机器人场景中为什么重要？**

> 你的回答：

---

## 四、尚未解决的问题

1. 
2. 

---

## 五、明日预告

- Day 7：LLM 基础认知（Pre-training → SFT → RLHF 管线）
- 理解 tokenizer、chat template
- 第一周收官：整理知识图谱 + 查漏补缺

---

## 六、额外记录
