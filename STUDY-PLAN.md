# 学习计划（合并版）

> **内容来自 A 计划**（`深度学习_12周90天详细学习计划.pdf`）—— 12 周深度学习专线，产出清单明确
> **节奏与机制来自 B 计划**（`deep_learning_90_day_plan.pdf`）—— 抗中断、可持续
>
> 合并日期：2026-09-15

---

## 一、每日节奏（45 / 60 / 15）

| 时段 | 时长 | 做什么 |
|---|---|---|
| **概念** | ~45 min | 读指定章节 / 看指定视频。**边读边用自己的话复述**——不允许只看不写 |
| **编码** | ~60 min | 动手实现。写「从零实现」的不调库；写「用 PyTorch」的不重复造轮子 |
| **检验与思考** | ~15 min | 合上资料答当天的思考题。**答不出的标红**，进当周补课清单 |

合计约 **2 小时**。

---

## 二、每周结构

```
Day 1 – 6   内容日（按上面的 45/60/15）
Day 7       复盘 + 弹性缓冲日
```

**Day 7 做什么**——两种状态，自己选：

- **有欠账**：补前面六天没完成的，**补齐为止**
- **没欠账**：默写一张本周知识地图（A4 纸）／把工具函数抽进 `utils.py`／整理 repo 和周报

> 每周第 7 天**本来就是留给你的缓冲**，不是"落后了才用"。

---

## 三、容错规则（卡住了怎么办）

### 三条底线

1. **允许某天只做 30 分钟，但不允许连续两天空缺。**
2. **跟不上就顺延，不要跳过。** 这份计划的依赖是线性的——第 25 天的手写反向传播依赖第 5 天的数值梯度检验。
3. **不要跳过手写实现。** 手写反向传播、手写 Transformer 是最有价值、也最容易被跳过的部分。

### 三类卡壳，三种处理

| 卡在哪 | 怎么办 |
|---|---|
| **数学推导** | **先接受结论继续往下走**，在当周 Day 7 回头补。不要在一个公式上耗掉三天 |
| **代码报错** | 设 **25 分钟上限**。超时就把最小复现代码拿去搜/问 AI，但**必须理解修复原因**再继续 |
| **算力不够** | 90% 的任务 CPU 或 Colab 免费 GPU 可完成。不够就改精读源码 + 小规模玩具实验 |

---

## 四、复习机制

1. **每周 Day 7 复盘**——不要因为"今天想学新东西"而跳过
2. **第 21 天、第 87–88 天系统自测**
3. **思考题答不出 → 标红 → 当周补课清单**
4. 刻意设计：**前面学的东西在后面被复用**（低秩近似 → LoRA；数值梯度 → 手写反向传播）

---

## 五、12 周产出清单

> 目录名照 A 计划原样保留，每个模块建成仓库里的一个目录。

| 周 | 主题 | 核心产出 |
|---|---|---|
| **1** | 数学重构 + 机器学习训练范式 | `01-linear-regression`（纯 NumPy + loss curve）<br>`02-softmax-classifier`（+ accuracy）<br>`week01.md` |
| **2** | MLP + Backprop + Mini Autograd | `03-micrograd`（Value / backward / topo sort）<br>`04-mlp-from-autograd`<br>`week02.md` |
| **3** | PyTorch 核心工作流 | `05-pytorch-basics`<br>`06-fashion-mnist`（训练/验证/保存/加载）<br>可复用的 `train_one_epoch` / `evaluate` 模板 |
| **4** | 优化、正则化、CNN 与 ResNet | `07-cnn-cifar10`（SimpleCNN baseline）<br>`08-resnet`（ResidualBlock）<br>一张实验对比表 |
| **5** | 序列建模、Embedding、RNN | `09-rnn-lm`（char tokenizer + RNN LM）<br>可生成文本的 `generate()`<br>`week05.md` |
| **6** | Attention：Q/K/V 到 Multi-Head | `10-attention`（`numpy_attention.py` + `self_attention.py` + `mha.py`）<br>一页 QKV 手算 + shape/causal mask 测试 |
| **7** | Transformer 架构 | `11-transformer`（位置编码 / LayerNorm demo / TransformerBlock）<br>一张 shape flow 图 + 论文一读笔记 |
| **8** | MiniGPT From Scratch | `12-minigpt`（端到端可训练）<br>`generate.py` + README |
| **9** | Tokenizer、HuggingFace、预训练模型 | `13-bpe-tokenizer`<br>`14-hf-finetuning`<br>手写 loop vs Trainer 对照笔记 |
| **10** | Fine-tuning、SFT、LoRA | `15-lora`（LoRALinear from scratch + PEFT）<br>训练效率对比表<br>`week10.md` |
| **11** | Capstone：独立实验与分析 | `final-project`（升级版 MiniGPT）<br>`experiments.csv` + 4 张图 + 2–3 页分析 |
| **12** | 论文阅读、工程重构、复盘 | 论文笔记 3 份（ResNet / Transformer / LoRA）<br>最终 repo + 30 分钟讲解提纲 |

---

## 六、可选加餐（来自 B 计划，不占主线）

主线跑完之后，或者某周有余力时可以插入：

- **经典机器学习**：KNN / 决策树 / 集成 / SVM / 聚类
- **生成模型**：VAE / GAN / 扩散模型
- **工程优化**：混合精度、量化

> B 计划把这三块各占了一周。放进主线会摊薄 Transformer/LLM 那条线，所以列为可选。

---

## 七、仓库结构

```
deep-learning-journey/
├── README.md              总索引 + 当前进度快照
├── STUDY-PLAN.md          本文件
├── utils.py               自己积累的工具函数库（Day 7 起逐步抽取）
├── week01/ ... week12/    按周组织
│   └── 01-linear-regression/ ...
├── minigpt/               第 8 周
├── final-project/         第 11 周
├── notes/                 每日笔记与思考题答案
└── papers/                论文精读笔记
```

---

## 八、打卡

> **每完成一天，勾掉一个。连续的方格比任何计划表都更能推着你往前走。**

### Week 1 · 数学重构 + 机器学习训练范式

- [x] Day 1 · 线性代数：shape 推导 + broadcasting
- [ ] Day 2 · 微积分：梯度与链式法则
- [ ] Day 3 · 概率、MLE、Entropy、Cross Entropy
- [ ] Day 4 · Linear Regression from scratch
- [ ] Day 5 · Logistic Regression
- [ ] Day 6 · Softmax Multiclass Classifier
- [ ] Day 7 · 复盘 + 弹性缓冲

### Week 2 · MLP + Backpropagation + Mini Autograd

- [ ] Day 8 · 从线性模型到 MLP
- [ ] Day 9 · Activation functions
- [ ] Day 10 · 手推反向传播
- [ ] Day 11 · 手写 autograd 引擎
- [ ] Day 12 · 用 autograd 搭 MLP
- [ ] Day 13 · 梯度问题排查
- [ ] Day 14 · 复盘 + 弹性缓冲

（后续周次按同样格式往下补）
