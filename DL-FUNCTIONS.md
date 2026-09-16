# 数学里少见、深度学习里常用的函数

> 配套 **[GLOSSARY.md](GLOSSARY.md)**（术语对照）。这份专注**函数和公式**。
> 标注说明：🟢 数学里有（只是换名了） · 🟡 数学里有但 DL 里用法完全不同 · 🔴 数学里基本没有，DL 独有

---

## 速查表（先看这张）

| 函数 | 英文 | 公式 | 输出范围 | 类型 |
|---|---|---|---|---|
| ReLU | rectified linear unit | `max(0, x)` | `[0, ∞)` | 🔴 |
| Sigmoid | sigmoid / logistic | `1/(1+e^(−x))` | `(0, 1)` | 🟢 |
| Tanh | hyperbolic tangent | `(eˣ−e^(−x))/(eˣ+e^(−x))` | `(−1, 1)` | 🟢 |
| Softmax | softmax | `e^(xᵢ) / Σe^(xⱼ)` | `(0,1)`，和为 1 | 🟡 |
| GELU | Gaussian error linear unit | `x·Φ(x)` | `(−0.17, ∞)` | 🔴 |
| SiLU / Swish | sigmoid linear unit | `x·σ(x)` | `(−0.28, ∞)` | 🔴 |
| Softplus | softplus | `ln(1+eˣ)` | `(0, ∞)` | 🔴 |
| LogSumExp | log-sum-exp | `ln Σe^(xᵢ)` | `[max(x), max(x)+ln n]` | 🔴 |
| LayerNorm | layer normalization | `(x−μ)/√(σ²+ε)·γ+β` | — | 🔴 |
| Cosine sim | cosine similarity | `a·b / (‖a‖‖b‖)` | `[−1, 1]` | 🟢 |

---

## 一、激活函数

### 🟢 Sigmoid —— 数学里叫「logistic function」

```
σ(x) = 1 / (1 + e^(−x))

σ'(x) = σ(x)·(1 − σ(x))        ← 导数最大只有 0.25
```

**数学里**：逻辑斯蒂函数，用来描述增长曲线。
**DL 里**：把任意实数压成 `(0,1)`，当概率用。

⚠️ **为什么现在很少用**：每经过一层 sigmoid，梯度最多只有 0.25 倍，几层下来梯度就消失了。

### 🟢 Tanh —— 数学里有，双曲正切

```
tanh(x) = (eˣ − e^(−x)) / (eˣ + e^(−x))

tanh'(x) = 1 − tanh²(x)        ← 导数最大 1，比 sigmoid 好
```

比 sigmoid 好一点，因为输出**关于 0 对称**（均值是 0）。

### 🔴 ReLU —— DL 最常用的激活函数

```
ReLU(x) = max(0, x)

ReLU'(x) = 1  (x > 0)
           0  (x < 0)          ← x=0 处不可导（工程上取 0 或 1）
```

**数学里几乎没有**——因为它在 0 处不可导，数学家不太喜欢。
**但 DL 爱死它了**，原因就一个：

> **导数要么是 0 要么是 1，不衰减。** 所以深层网络也能把梯度传下去。

**代价**：`x < 0` 时梯度是 0，这个神经元就"死"了（dead ReLU）——你 Day 2 的 `grad_check.py` 里 h<0 那组看到的就是这个。

### 🔴 ReLU 的变体（都是为了让"死区"不那么死）

```
LeakyReLU(x) = max(αx, x)           α 通常 0.01，负半轴给个小斜率
ELU(x)       = x           (x > 0)
               α(eˣ − 1)   (x ≤ 0)
```

### 🔴 SiLU / Swish —— 现代网络常用

```
SiLU(x) = x · σ(x)
```

**平滑版 ReLU**。形状像 ReLU，但处处可导，且在负半轴有个小"坑"。

### 🔴 GELU —— Transformer 默认用这个

```
GELU(x) = x · Φ(x)        Φ 是标准正态分布的累积分布函数

近似： 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**直觉**：`Φ(x)` 是"x 在正态分布里有多靠前"。
x 很大 → Φ≈1 → 通过；x 是负数 → Φ≈0 → 挡住。**但边缘是平滑过渡的**。

GPT / BERT 用 GELU。你要手写 MiniGPT 时会遇到。

### 🔴 Softplus

```
Softplus(x) = ln(1 + eˣ)
```

**Softplus 的导数正好是 sigmoid。** 可以理解成 "ReLU 的平滑版"，但实际很少直接用。

---

## 二、把分数变成概率 —— DL 最独特的一类

> 这一整类在数学里几乎没有对应，是 DL 自己发展出来的。

### 🟡 Softmax —— DL 最有代表性的函数

```
softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ)
```

**它的作用**：把一个任意实数向量，变成**概率分布**。

```
[2.0, 1.0, 0.1]  →  [0.659, 0.242, 0.099]    每个都 > 0，和为 1
```

**为什么数学里少**：这是信息论/统计力学里的 Boltzmann 分布，但"用它处理分类网络的输出"是 DL 的做法。

**三个性质**（都很关键）：
1. 输出恒为正（因为有 exp）
2. 和为 1（归一化）
3. **保持大小顺序**（最大的还是最大）→ 所以只用 `argmax` 时可以跳过 softmax

### 🔴 LogSumExp —— 纯数值技巧，但无处不在

```
LSE(x) = ln( Σ e^(xᵢ) )

关键不等式： max(x) ≤ LSE(x) ≤ max(x) + ln(n)
```

**它解决的问题**：`Σ e^(xᵢ)` 容易**溢出**（x 大一点就 inf）。
取个 log 就能避免。

**它的作用**：softmax 的"稳定版"核心。

### 🔴 LogSoftmax

```
log_softmax(x)ᵢ = xᵢ − LSE(x)
```

**为什么要它**：计算交叉熵时，如果先 softmax 再取 log，会有精度损失。
直接算 `log_softmax` 更稳。

> PyTorch 的 `nn.CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`，**合在一起了**。
> 所以你传 logits 进去，不要自己先 softmax。

### 🔴 归一化三兄弟 —— 数学里完全没有，DL 独有

| | 在哪个维度上归一化 | 用在哪 |
|---|---|---|
| **BatchNorm** | 跨 **batch** 维 | CNN 常用 |
| **LayerNorm** | 跨 **特征** 维（每条样本自己） | Transformer 标配 |
| **RMSNorm** | 同 LayerNorm，但不减均值 | LLaMA 等新模型 |

```
LayerNorm(x) = (x − μ) / √(σ² + ε) · γ + β
                   ↑         ↑        ↑   ↑
                减均值    除标准差   可学习的缩放/平移

RMSNorm(x)   = x / √(mean(x²) + ε) · γ      ← 省掉减均值
```

**`+ ε` 是干嘛的**：防止除以 0。**这个细节数学里不会教，工程里必须写。**

---

## 三、损失函数里常用的

### 🟡 Cross Entropy —— 从信息论搬来的，但用法变了

```
CE(p, q) = −Σ pᵢ · ln(qᵢ)
```

**在 DL 里**：`p` 是真实标签（one-hot，只有一个 1），所以式子**塌缩**成：

```
CE = −ln(正确类别的预测概率)
```

**这就是为什么分类用 CE**：它只关心"正确答案那一个位置上的概率有多大"。

> 信息论里的交叉熵是衡量两个分布的差异。DL 把它当成损失函数用。

### 🟡 KL Divergence —— 数学里叫「相对熵」

```
KL(p ‖ q) = Σ pᵢ · ln(pᵢ / qᵢ)

关系：  KL(p‖q) = CE(p,q) − H(p)
                  ↑         ↑
                交叉熵    真实分布的熵（常数）
```

**所以最小化 CE ≡ 最小化 KL。** 这也是为什么分类用 CE 就够了。

### 🟢 MSE（均方误差）

```
MSE = (1/n) Σ (yᵢ − ŷᵢ)²
```

数学里就是"平均平方误差"。DL 里作为**回归任务**的标准损失。

### 🔴 Huber Loss —— DL 特有的折中

```
Huber(x) = 0.5x²           (|x| ≤ δ)      ← 小误差时像 MSE
           δ(|x| − 0.5δ)   (|x| > δ)      ← 大误差时像 MAE
```

**为什么需要它**：MSE 对**离群点**太敏感（平方放大了），MAE 又处处不可导。
Huber 两头兼顾。**纯粹是工程折中，数学里不会讲。**

---

## 四、数值稳定技巧 —— 数学里完全不存在

> 这一类**不是数学概念，是"浮点数会溢出"这个现实逼出来的**。

### 🔴 max-shift（softmax 减最大值）

```python
# ❌ 会溢出
exp(x) / exp(x).sum()

# ✅ 稳定版
x = x - x.max()               # 减去最大值
exp(x) / exp(x).sum()
```

**为什么可以**：softmax 对**平移不变**——所有输入加同一个常数，输出不变。

**为什么必须**：`exp(1000)` 直接变 `inf`。你 Day 3 会亲手实现这个。

### 🔴 epsilon（加个小量防除零）

```python
x / (norm + 1e-8)             # 不加 eps，norm=0 时就炸
log(p + 1e-8)                 # p=0 时 log(0) = -inf
```

**所有归一化和 log 的地方都要写这个。** 数学里不需要，因为数学里没有"浮点精度"。

### 🔴 clamp / clip（截断）

```python
np.clip(x, 0, 1)              # 夹到 [0, 1]
torch.clamp(x, min=0)         # 等价于 ReLU

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**梯度裁剪**是训练 RNN / Transformer 的标配——防止偶尔出现的巨大梯度把参数带飞。

### 🔴 top-k / argmax

```python
x.argmax()                    # 最大值的下标
x.topk(5)                     # 最大的 5 个值 + 下标
```

**注意**：`argmax` 不可导。所以训练时用 softmax，推理时才用 argmax 取答案。

---

## 五、张量专门操作 —— 数学里没有对应

### 🔴 one-hot 编码

```python
# 类别 2 → [0, 0, 1, 0, 0]
np.eye(5)[2]
```

**为什么需要**：分类的标签是"第几类"，但损失函数需要"概率分布"的形式。

### 🔴 gather / scatter

```python
x.gather(1, index)            # 按 index 取元素
x.scatter_(1, index, value)   # 按 index 写入
```

**用在哪**：Embedding 查表、按标签取正确类别的概率。

### 🔴 masked_fill —— Transformer 的核心操作

```python
scores.masked_fill(mask == 0, -inf)   # 把不该看的位置设成 -inf
```

**作用**：softmax 之前把"不允许看"的位置设成 `-inf`，
**softmax 之后这些位置就变成 0 概率**（因为 `e^(−inf) = 0`）。

> 这就是 **causal mask** 的实现方式——Week 6/7 你会亲手写。

### 🟢 cosine similarity

```
cos(a, b) = a·b / (‖a‖·‖b‖)
```

数学里就是"夹角的余弦"。DL 里用于**向量检索**、**对比学习**、**Embedding 相似度**。

> 实现时的坑：分母要加 `eps`，否则零向量会炸。

---

## 六、用这张表当"翻译器"

遇到不懂的函数，先问两个问题：

```
1. 它是数学概念换了个名，还是 DL 独有？
     ├─ 数学换名（sigmoid、tanh、KL）-> 你懂数学就够了
     └─ DL 独有（ReLU、softmax、LayerNorm、logsumexp）
        -> 得新学，但通常理解"它解决什么问题"就够了

2. 它是在解决「数学问题」还是「工程问题」？
     ├─ 数学问题：表达能力、概率意义
     └─ 工程问题：溢出、除零、梯度消失、不可导
        -> 这类（eps、max-shift、clamp、Huber）占了很大比例
```

**第二问尤其重要。** DL 里大量"奇怪的函数"，其实都是在补浮点数和可导性的坑。
