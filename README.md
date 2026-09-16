# deep-learning-journey

12 周 / 90 天深度学习主线：**数学 → 训练范式 → Backprop → MLP → PyTorch → CNN → Attention → Transformer → GPT → Fine-tuning → LoRA**

> 📋 **完整计划见 [STUDY-PLAN.md](STUDY-PLAN.md)** —— 内容取自 A 计划，节奏与容错机制取自 B 计划（2026-09-15 合并）
> 执行原则：**理解 → 手推 → 手写 → 实验 → Debug → 总结**
> 完成标准：不是"视频看完"，而是"**能从空文件实现 / 能解释 / 能做实验**"

---

## 📍 当前进度快照

> **每次学习结束后更新这里。** 下次开终端 `cat README.md` 一眼就知道从哪继续。
> 最后更新：2026-09-16

| | |
|---|---|
| **进行到** | Week 1 · **Day 3**（概率 / MLE / Cross Entropy） |
| **Day 1** | ✅ 完成（09-15）· quiz 25/31 → 速查表 19/20 |
| **Day 2** | ✅ 完成（09-16）· 验收 A 组 5/5，三组梯度测试通过 |
| **下次第一步** | 在 `losses.py` 实现三个函数，跑 `test_losses.py` |
| **计划版本** | 合并版（A 的内容 + B 的节奏），2026-09-15 切换 |

**Day 3 的文件**（`week01/day03_losses/`）—— 待完成：

| 文件 | 用途 |
|---|---|
| `losses.py` | **要你填的**：`stable_softmax` / `cross_entropy` / `cross_entropy_grad` |
| `test_losses.py` | 10 项测试（形状、和为1、数值稳定、手算、梯度检验）|
| `notes.md` | 概念链条 + 6 道检验题 |

**已完成的日子**：

- Day 1 · `week01/day01_shapes/` —— shape 语义、广播、31 题 + 速查表
- Day 2 · `week01/day02_calculus/` —— 链式法则、计算图、梯度验证器

---

## 环境

用 [uv](https://docs.astral.sh/uv/) 管理，Python 3.12 + numpy + matplotlib。

```bash
# 激活虚拟环境
source .venv/bin/activate

# 或者不激活，直接跑
uv run python <script.py>
```

已装：`numpy 2.5.3` · `matplotlib 3.11.2`

---

## 目录结构

```
deep-learning-journey/
├── README.md              本文件（进度追踪 + 当前快照）
├── STUDY-PLAN.md          完整计划（合并版）
├── requirements.txt
├── .venv/
└── week01/
    ├── day01_shapes/           ✅ 已完成
    │   ├── numpy_semantics.md      NumPy 语义速览
    │   ├── numpy_demo.py           配套可运行演示
    │   ├── shape_quiz.py           31 题交互答题
    │   ├── shape_drills.py         31 题填 guess 版
    │   └── tensor_shape_notes.md   速查表 + 错题本
    ├── day02_calculus/         ✅ 已完成
    │   ├── grad_check.py           梯度验证器
    │   └── derivative_notes.md     手推模板 + 验收答案
    └── day03_losses/           ← 进行中
        ├── losses.py               【要你填】softmax / CE / 梯度
        ├── test_losses.py          10 项测试
        └── notes.md                概念链条 + 检验题
```

> 后续每周的新目录按 `STUDY-PLAN.md` 第五节的产出清单建（如 `01-linear-regression/`、`03-micrograd/`）。

---

## 每日 2 小时模板（45 / 60 / 15）

| 时段 | 时长 | 做什么 |
|---|---|---|
| **概念** | ~45 min | 读指定章节 / 看指定视频。**边读边用自己的话复述**——不允许只看不写 |
| **编码** | ~60 min | 动手实现。写「从零实现」的不调库；写「用 PyTorch」的不重复造轮子 |
| **检验与思考** | ~15 min | 合上资料答当天的思考题。**答不出的标红**，进当周补课清单 |

**每周第 7 天是复盘 + 弹性缓冲日**：有欠账就补，没欠账就默写知识地图 / 抽工具函数 / 整理 repo。

> 三条底线：允许某天只做 30 分钟，**不允许连续两天空缺**；跟不上**顺延不跳过**；**不要跳过手写实现**。

---

## 进度追踪

### Week 1｜数学重构 + 机器学习训练范式
> 目标：把"数据 - 模型 - 损失 - 梯度 - 更新"这条链打通

预计投入 15–17 h

- [x] **Day 1** 线性代数：shape 推导 + broadcasting ✅
      → `shape_drills.py`（31 题，填 guess 版）
      → `shape_quiz.py`（31 题，终端交互答题版）
      → `numpy_semantics.md`（NumPy 语义速览）
      → `numpy_demo.py`（配套可运行演示）
      → `tensor_shape_notes.md`（速查表）
- [x] **Day 2** 微积分：偏导 / 梯度 / 链式法则 ✅ → `derivative_notes.md`
- [ ] **Day 3** 概率 / MLE / Entropy / Cross Entropy → `losses.py`  ← 进行中
- [ ] **Day 4** Linear Regression from scratch → `01-linear-regression/` + loss curve
- [ ] **Day 5** Logistic Regression → `logistic_regression_numpy.py`
- [ ] **Day 6** Softmax Multiclass → `02-softmax-classifier/`
- [ ] **Day 7** 复盘 + 小测 → `week01.md`

**硬性产出**
- [ ] `01-linear-regression`：纯 NumPy 训练，包含 loss curve
- [ ] `02-softmax-classifier`：多分类训练脚本 + accuracy
- [ ] `week01.md`：用自己的语言解释一次完整训练循环

<details>
<summary>Week 2–12 总览（点击展开）</summary>

| 周 | 主题 | 核心产出 |
|---|---|---|
| Week 2 | MLP + Backpropagation + Mini Autograd | `03-micrograd` |
| Week 3 | PyTorch 核心工作流 | `05-pytorch-basics` |
| Week 4 | 优化、正则化、CNN 与 ResNet | `07-cnn-cifar10` |
| Week 5 | 序列建模、Embedding、RNN 与语言模型 | `09-rnn-lm` |
| Week 6 | Attention：Q/K/V 到 Multi-Head | `10-attention` |
| Week 7 | Transformer Architecture | `11-transformer` |
| Week 8 | MiniGPT From Scratch | `12-minigpt` |
| Week 9 | Tokenizer、Hugging Face 与预训练模型 | `13-bpe-tokenizer` |
| Week 10 | LLM Fine-tuning、SFT、LoRA 与训练工程 | `15-lora` |
| Week 11 | Capstone：独立实验与模型分析 | `final-project` |
| Week 12 | 论文阅读、工程重构与最终复盘 | 论文笔记 3 份 |
| Day 85–90 | 查漏补缺 + 最终模拟面试 | 闭卷重构 + 项目审计 |

</details>

---

## 主线资料（一周只用一个主线，其余卡住时再查）

| 编号 | 资料 | 周次 |
|---|---|---|
| R1 | Dive into Deep Learning — MLP / Backprop | Week 1–5 |
| R2 | PyTorch — Learn the Basics | Week 3 / 10 / 12 |
| R3 | Stanford CS231n | Week 1 / 4 / 11 |
| R4 | Karpathy — Neural Networks: Zero to Hero | Week 2 / 5 / 8 |
| R5 | Dive into Deep Learning — Attention & Transformers | Week 6–7 |
| R6 | 3Blue1Brown — Attention in Transformers | Week 6 |
| R7 | Stanford CS336 — Language Modeling from Scratch | Week 8–12 |
| R8 | Hugging Face — LLM Course | Week 9–10 |
| R9 | Hugging Face PEFT — LoRA | Week 10 |
| P1 | ResNet Paper | Week 4 / 12 |
| P2 | Attention Is All You Need | Week 7 / 12 |
| P3 | LoRA Paper | Week 10 / 12 |

---

## 常见误区（计划里点名的）

- ❌ 不要重学完整高数 / 线代教材 —— 只补当天真正用得到的
- ❌ 不要直接调用 sklearn 的训练器
- ❌ 不要用"视频看完"当完成标准
- ✅ 每天至少一半时间留给代码
- ✅ 任何模型先做 **tiny-overfit test**（极小数据都过拟合不了，别急着扩大训练）
- ✅ 保存失败实验 —— 失败结果也是判断力的一部分
