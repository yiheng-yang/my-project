# 从数学到深度学习：术语对照表

> **用法**：数学你懂，但 DL 里换了个说法。这份文件做两件事——
> ① 解释**为什么**要换说法；② 给**中英对照**，帮你建立英文概念。
>
> 会随学习进度增补。

---

## 一、为什么数学概念在 DL 里要换个说法

不是 DL 故意造词。是**问题变了**，名字就跟着变。五个真实原因：

### 1. 数学说「是什么」，DL 说「怎么算」

数学里 `f(g(x))` 是一个复合函数，一句话说完了。
DL 里不行——参数有几亿个，得**拆成能按顺序执行的节点**，还要知道每个节点先算谁。

> 数学：chain rule
> DL：**forward pass → backward pass**（多了"顺序"和"方向"）

**数学里没有"方向"的概念，DL 有。** 因为 DL 要真的按顺序跑一遍。

### 2. 从「一个运算」到「一个可堆叠的模块」

数学里 `Wx + b` 就是一个仿射变换，算完就完了。
DL 里它叫 **layer（层）**，因为它**带参数、能堆叠、能复用**。

> 数学：矩阵乘法 / 仿射变换
> DL：**linear layer / dense / fully-connected**
>
> 同一件事，名字变了——因为**角色**变了（从"一个运算"变成了"一个组件"）。

### 3. 数学不管内存，DL 必须管

数学里你可以写"对全部数据求梯度"。DL 里数据太大，必须切片。

> 所以出现了这些**数学里根本不存在的词**：
> **batch**（一批）、**epoch**（轮）、**gradient accumulation**（梯度累积）、
> **in-place**（原地操作）、**broadcasting**（广播）

这些是**工程约束**的产物，不是数学概念。

### 4. 数学里参数是固定的，DL 里参数一直在变

数学写 `f(x) = wx + b` 时，`w` 和 `b` 是常数。
DL 里它们是**要被优化的对象**——所以要专门起名叫 **parameter（参数）**，
和 **input（输入）** 区分开。

> 还要区分 **training（训练期）** 和 **inference（推理期）**，
> 因为同一个网络在两个阶段行为可能不同（比如 BatchNorm、Dropout）。

### 5. 有些词纯粹是历史积累

`forward` / `backward` 这套说法来自控制论和信号处理的传统。
`tensor`（张量）是从物理和微分几何借来的词，但在 DL 里**只是指"n 维数组"**，
和数学里的张量（有坐标变换律的那个）不是一回事。

> ⚠️ **这条最要小心**：DL 借用的词，含义常常比数学里窄得多。
> `tensor` 就是典型——DL 的 tensor ≈ n 维数组。

---

## 二、一个例子的完整旅程：矩阵乘法

同一件事，在不同层次有不同名字：

| 层次 | 叫什么 | 为什么这么叫 |
|---|---|---|
| 数学 | 矩阵乘法 / 线性变换 | 描述"是什么" |
| NumPy | `np.matmul` / `@` | 描述"怎么调用" |
| PyTorch | `torch.matmul` / `@` | 同上，还能上 GPU、能求导 |
| 神经网络层 | **linear layer** | 强调它是"一层"，带参数、可堆叠 |
| Transformer | `W_q` / `W_k` / `W_v` **投影**（projection）| 强调它把输入"投影"到另一个空间 |

**全是同一个 `@`。** 名字变，是因为你**看它的角度**变了。

这就是为什么你会觉得"数学我懂，但 DL 的说法不熟"——
你不是不懂数学，是**还没建立"角色"这一层认知**。

---

## 三、词汇对照表

### 3.1 基础对象

| 中文 | English | 备注 |
|---|---|---|
| 标量 | **scalar** | 0 维 |
| 向量 | **vector** | 1 维 |
| 矩阵 | **matrix** | 2 维 |
| 张量 | **tensor** | DL 里 ≈ n 维数组 |
| 阶 / 秩 | **rank** / **order** | 几个维度 |
| 形状 | **shape** | |
| 维度 / 轴 | **dimension** / **axis** / **dim** | PyTorch 用 `dim` |
| 样本 | **sample** / **instance** / **example** | 一行数据 |
| 特征 | **feature** | 一列数据 |
| 批次 | **batch** | 一次喂进去的一小组样本 |
| 批次大小 | **batch size** | |

### 3.2 基本运算

| 中文 | English | 备注 |
|---|---|---|
| 点积 / 内积 | **dot product** / **inner product** | |
| 矩阵乘法 | **matrix multiplication** / **matmul** | |
| 逐元素运算 | **element-wise** | `*` 不是 `@` |
| 广播 | **broadcasting** | 纯工程概念，数学里没有 |
| 转置 | **transpose** | |
| 重塑 | **reshape** | |
| 范数 | **norm** | |
| 归约 | **reduction** | sum/mean/max 这类"压维度"的操作 |
| 拼接 | **concatenate** / **cat** | |

### 3.3 导数与梯度

| 中文 | English | 备注 |
|---|---|---|
| 导数 | **derivative** | 单变量 |
| 偏导数 | **partial derivative** | 多变量，其余当常数 |
| 梯度 | **gradient** | 所有偏导组成的向量 |
| 链式法则 | **chain rule** | |
| 局部导数 | **local derivative** | 这个节点自己的导数 |
| 上游梯度 | **upstream gradient** | 从后面传回来的 |
| 前向传播 | **forward pass** / **forward** | 输入 → 输出 |
| 反向传播 | **backward pass** / **backprop** | loss → 梯度 |
| 计算图 | **computational graph** | |
| 自动微分 | **automatic differentiation** / **autograd** | |
| 雅可比矩阵 | **Jacobian** | 多元函数的"导数矩阵" |
| 梯度下降 | **gradient descent** | |
| 随机梯度下降 | **SGD** (stochastic gradient descent) | |
| 学习率 | **learning rate** / **lr** | |
| 数值梯度 | **numerical gradient** | 用差分算的 |
| 有限差分 | **finite difference** | |
| 梯度检验 | **gradient check** | |
| 梯度消失 / 爆炸 | **vanishing / exploding gradient** | |

### 3.4 网络结构

| 中文 | English | 备注 |
|---|---|---|
| 层 | **layer** | |
| 线性层 / 全连接层 | **linear layer** / **dense** / **fully-connected (FC)** | 同一件事，三个名字 |
| 仿射变换 | **affine transformation** | 数学叫法：`Wx + b` |
| 激活函数 | **activation function** | |
| 神经元 / 单元 | **neuron** / **unit** | |
| 权重 | **weight** | 通常记作 `W` |
| 偏置 | **bias** | 通常记作 `b` |
| 参数 | **parameter** | 要被训练的东西（含 weight 和 bias）|
| 超参数 | **hyperparameter** | 你手动设的（学习率、层数…）|
| 隐藏层 | **hidden layer** | |
| 深度 / 宽度 | **depth** / **width** | 层数 / 每层宽度 |
| 多层感知机 | **MLP** (multi-layer perceptron) | |
| 表示 / 表征 | **representation** | 中间层的输出 |
| 嵌入 | **embedding** | 把离散对象映射成向量 |

### 3.5 激活函数

| 中文 | English | 备注 |
|---|---|---|
| 修正线性单元 | **ReLU** (rectified linear unit) | `max(0, x)` |
| S 型函数 | **sigmoid** | |
| 双曲正切 | **tanh** | |
| 归一化指数 | **softmax** | 输出概率分布 |
| — | **GELU** / **SiLU** / **Swish** | 现代 Transformer 常用 |

### 3.6 训练过程

| 中文 | English | 备注 |
|---|---|---|
| 损失函数 | **loss function** | |
| 损失 | **loss** / **cost** / **objective** | PyTorch 里叫 **criterion** |
| 训练 | **training** / **fit** | |
| 推理 | **inference** / **evaluation** | 不更新参数 |
| 轮次 | **epoch** | 完整过一遍数据集 |
| 迭代 / 步 | **iteration** / **step** | 一次参数更新 |
| 优化器 | **optimizer** | |
| 动量 | **momentum** | |
| 收敛 | **convergence** | |
| 过拟合 / 欠拟合 | **overfitting** / **underfitting** | |
| 正则化 | **regularization** | |
| 权重衰减 | **weight decay** | |
| 学习率调度 | **learning rate schedule** | |
| 早停 | **early stopping** | |
| 泛化 | **generalization** | |
| 验证集 / 测试集 | **validation set** / **test set** | |

---

## 四、几组特别容易混的

### `loss` / `cost` / `objective` / `criterion`

**基本上是同一件事**，只是来源不同：

- **loss** —— 最常用，指单个样本或一批的误差
- **cost** —— 常指整个数据集的平均损失（老教材用得多）
- **objective** —— "目标函数"，可以是 loss，也可以在优化里指要最大化的东西
- **criterion** —— **PyTorch 的 API 名**：`nn.CrossEntropyLoss()` 是 criterion

### `parameter` vs `hyperparameter`

| | 谁定 | 例子 |
|---|---|---|
| **parameter** | 训练学出来的 | `W`、`b` |
| **hyperparameter** | 你手动设的 | 学习率、层数、batch size |

### `logits`（这个没人解释过，但天天用）

**logits = 还没经过 softmax 的原始分数。**

```
logits  →  softmax  →  probabilities
[2, 1, 0.1]           [0.66, 0.24, 0.10]
```

PyTorch 的 `CrossEntropyLoss` **内部自带 softmax**，所以你要传 **logits**，
不是概率——这是新手最常踩的坑之一。

### `forward` vs `inference`

- **forward** —— 泛指"算一遍前向"，训练和推理都会做
- **inference** —— 特指"训练完之后拿来用"，不计算梯度、不更新参数

### DL 的 `tensor` ≠ 数学的 `tensor`

| | 含义 |
|---|---|
| 数学 / 物理 | 有坐标变换律的多维数组（应力张量、度规张量…）|
| **深度学习** | **就是 n 维数组，没别的** |

别被这个词吓到。PyTorch 的 `tensor` 和 NumPy 的 `array` 基本是一回事
（多了 GPU 支持和自动求导）。

---

## 五、怎么用这份表

1. **看到陌生英文词，先回来查**——大概率是数学里你认识的东西
2. **卡住时问自己**：这是数学概念，还是工程约束的产物？
   - 数学概念 → 你有基础，只是名字变了
   - 工程产物（batch / broadcasting / epoch）→ 数学里没有，得新学
3. **建立英文概念**：以后看到 `upstream gradient` 要能**直接反应**，
   不要在脑子里先翻译成中文
