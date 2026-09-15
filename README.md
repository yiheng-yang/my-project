# deep-learning-journey

12 周 / 90 天深度学习主线：**数学 → 训练范式 → Backprop → MLP → PyTorch → CNN → Attention → Transformer → GPT → Fine-tuning → LoRA**

> 执行原则：**理解 → 手推 → 手写 → 实验 → Debug → 总结**
> 学习比例：~30% 理论 + 60% Coding/实验 + 10% 复盘
> 完成标准：不是"视频看完"，而是"**能从空文件实现 / 能解释 / 能做实验**"

---

## 📍 当前进度快照

> **每次学习结束后更新这里。** 下次开终端 `cat README.md` 一眼就知道从哪继续。
> 最后更新：2026-09-15

| | |
|---|---|
| **进行到** | Week 1 · **Day 2**（微积分 / 链式法则） |
| **Day 1** | ✅ **完成**（2026-09-15） |
| **Day 1 成绩** | quiz 25/31 → 速查表 19/20 |
| **下次第一步** | 手推 `y = (wx+b)²` 对 `w`、`b`、`x` 的导数 |
| **产出** | Week 1 进度表里 Day 1 已勾选 |

**Day 1 用到的文件**（都在 `week01/day01_shapes/`）：

| 文件 | 用途 |
|---|---|
| `numpy_semantics.md` | 原理速览（axis / 广播 / 1-D 提升 / reshape / keepdims / norm） |
| `numpy_demo.py` | 配套可运行演示，边读边跑 |
| `shape_quiz.py` | **31 题交互答题**（推荐） |
| `shape_drills.py` | 31 题填 `guess=` 版 / `--reveal` 看答案 |
| `tensor_shape_notes.md` | 速查表 + 错题本（**待填**） |

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
├── README.md              ← 本文件（进度追踪）
├── requirements.txt
├── .venv/
└── week01/
    └── day01_shapes/
        ├── shape_drills.py         ← 31 道 shape 判断题
        └── tensor_shape_notes.md   ← Tensor shape 速查表（长期复用）
```

---

## 每日 2.5 小时模板

| 时间 | 任务 |
|---|---|
| 0–15 min | 闭卷回忆昨天 + 看昨天 TODO |
| 15–60 min | 教材 / 视频 / 推导 |
| 60–125 min | Coding / from scratch 实现 |
| 125–145 min | 实验 / Debug / shape check |
| 145–150 min | 当天笔记：**一句结论 + 一个疑问 + 明日第一步** |

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
- [ ] **Day 2** 微积分：偏导 / 梯度 / 链式法则 → `derivative_notes.md`
- [ ] **Day 3** 概率 / MLE / Entropy / Cross Entropy → `losses.py`
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
