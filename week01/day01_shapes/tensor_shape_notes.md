# Tensor Shape 速查表

> Day 1 产出 · Week 1｜数学重构 + 机器学习训练范式
> 这份表后面 CNN / Attention / Transformer 都会反复回来查，别写完就丢。

---

## 0. 一句话

**在深度学习里，shape 就是语义。**
写任何一行代码，你都要能说出**每一个维度代表什么**。
如果说不出来，那这行代码你还没真正写明白。

---

## 1. 张量的阶（rank）

| 阶 | 名字 | 例 | 典型含义 |
|---|---|---|---|
| 0 | scalar | `()` | 一个 loss 值 |
| 1 | vector | `(D,)` | 一个样本的特征 / 一个词向量 |
| 2 | matrix | `(B, D)` | 一批样本 |
| 3 | 3-D tensor | `(B, T, D)` | 一批序列 |
| 4 | 4-D tensor | `(B, C, H, W)` | 一批图像 |

---

## 2. 常见维度的字母含义（先背这个）

| 字母 | 含义 | 出现场景 |
|---|---|---|
| `B` | batch size，一批多少个样本 | 几乎所有地方 |
| `D` | feature dim / model dim，特征或模型维度 | MLP、Embedding |
| `H` | hidden dim，隐藏层宽度 | MLP；**注意** 在图像里 `H` 是 height，别混 |
| `T` | time / sequence length，序列长度 | RNN、Transformer |
| `C` | channel，通道数 | 图像 `(B,C,H,W)` |
| `H,W` | height, width | 图像空间尺寸 |
| `N,L` | 有时表示 `num_heads` / `num_layers` | Transformer |

**三个最常出现的组合，务必记住：**

```
MLP / 全连接：   (B, D)
图像 CNN：       (B, C, H, W)      ← PyTorch 是 NCHW，不是 NHWC
序列 / Attention：(B, T, D)
```

---

## 3. 矩阵乘法：内维相消

唯一规则：

```
(..., m, k) @ (..., k, n)  ->  (..., m, n)
              ↑ 这两个必须相等，然后消失
                ↑前面的 batch 维度靠 broadcasting 对齐
```

**助记：内维必须相等，然后消失；外维留下。**

```python
(4, 3) @ (3, 5)   -> (4, 5)     ✅ 内维 3 == 3
(4, 3) @ (3, 5, 2)-> 报错        ❌ 2-D 和 3-D 按 batch 对齐后内维不等
(4, 6, 3) @ (3, 5)-> (4, 6, 5)  ✅ batch 维自动广播（这叫 batched matmul）
```

### ⚠️ 一维向量的坑（B2 / B3 这两题）

NumPy 对 1-D 操作数的处理是**不对称**的：

- **1-D 在左边** → 当作行向量 `(1, k)`，算完**去掉前面**那个 1
- **1-D 在右边** → 当作列向量 `(k, 1)`，算完**去掉后面**那个 1

```python
b = (5,)          # 一维
W = (3, 5)

b    @ W.T   # (1,5) @ (5,3) -> (1,3) -> 去掉 1 -> (3,)   ✅
W.T  @ b     # (5,3) @ (5,1) -> 内维 3 vs 5 -> 报错       ❌
```

记忆法：**右边的 1-D 被当成「一根竖着的列」，左边的 1-D 被当成「一根横着的行」。**
这也是为什么大多数代码里你都该显式写 `(B, D)`、不要依赖 1-D 自动推断。

---

## 4. Broadcasting：三条规则

**对齐方式：从最右边开始，逐维比较。**

对每一维：

1. 相等 → 通过
2. 其中一个是 `1` → 拉伸到另一个的长度
3. 都不是 `1` 且不相等 → **报错**

```python
x   (4, 3)
b   (5,)      →  x + b     ❌  最右维 3 vs 5，且都不是 1
b[:3] (3,)    →  x + b[:3] ✅  (4,3)+(3,) → (4,3)

img (4, 2, 32, 32)  mean(axis=(2,3))            (4, 2)
img - mean(axis=(2,3))                          ❌  (4,2,32,32) - (4,2)
img - mean(axis=(2,3), keepdims=True)           ✅  (4,2,1,1) 广播成 (4,2,32,32)
```

### 为什么几乎人人都踩过 `keepdims` 的坑

`sum/mean(axis=k)` 默认会**把第 k 维整个吃掉**（降维）。
要"减掉均值但不改变形状"，就必须 `keepdims=True`，让它变成 `1` 而不是消失 ——
`1` 可以被广播，**消失的维度不能**。

```python
np.linalg.norm(x, axis=1)               # (4, 3) -> (4,)      ❌ 维度没了
np.linalg.norm(x, axis=1, keepdims=True)# (4, 3) -> (4, 1)    ✅ 可以广播

x / np.linalg.norm(x, axis=1)              # ❌ (4,3)/(4,) -> 3 vs 4 报错
x / np.linalg.norm(x, axis=1, keepdims=True)  # ✅ 每行 L2 归一化
```

**口诀：要做减法/除法归一化，先 keepdims。**

### 一个立刻能用的副产品：成对差值

```python
x[:, None, :]        # (4, 1, 3)
x[:, :, None]        # (4, 3, 1)
x[:, None, :] - x[:, :, None]   # (4, 3, 3)  ← 第 i 行 j 列 = x_i - x_j
```

这就是 Attention 里算「每对 token 之间关系」的形状来源。
`None`（等价于 `np.newaxis`）就是**手动插入一个长度为 1 的轴**，用来指挥广播往哪走。

---

## 5. reshape vs transpose

| | 做什么 | 数据顺序 |
|---|---|---|
| `reshape` | 按内存顺序**重新切分** | 不变 |
| `transpose` | **换轴顺序** | 变（需要跨步读取） |

```python
seq (4, 6, 3)
seq.reshape(6, 4, 3)      # ❌ 不是转置！只是把前两维揉在一起重切
seq.transpose(1, 0, 2)    # ✅ (6, 4, 3) 才是交换前两维
```

**关键区分：要「换维度顺序」用 transpose，要「合并/拆开维度」用 reshape。**
`reshape(-1)` 拍平成一维；`reshape(B, -1)` 中 `-1` 表示"这一维你自己算"。

---

## 6. Norm

| 写法 | 含义 |
|---|---|
| `np.linalg.norm(x)` | Frobenius 范数，整个张量 → 标量 `()` |
| `np.linalg.norm(x, axis=1)` | 沿第 1 维求 L2 → `(4,)` |
| `np.linalg.norm(x, axis=1, keepdims=True)` | `(4, 1)`，可广播 |

L2 norm = √(各元素平方和)。深度学习中它出现在：权重衰减（weight decay）、梯度裁剪、Embedding 归一化、LayerNorm 的内部。

---

## 7. 速查表（Day 1 填空版）

> **这是 Day 1 的硬性产出**，计划里点名要"整理一页速查表，之后 CNN / Attention 都复用"。

### 怎么填

**设定**（和 `shape_drills.py` 完全一致）：

```python
B, D, H, C, T = 4, 3, 5, 2, 6

x   = (4, 3)              # 一批特征向量
W   = (3, 5)              # 全连接层权重
b   = (5,)                # 偏置
img = (4, 2, 32, 32)      # 一批图像
seq = (4, 6, 3)           # 一批序列
```

**答题格式**（照抄这个写法）：

| 情况 | 写什么 | 例 |
|---|---|---|
| 有 shape | 元组 | `(4, 3)` |
| 标量 | 空元组 | `()` |
| 会报错 | 大写 ERROR | `ERROR` |

**规则：先自己想，再对答案。** 不许运行一遍再把结果抄进来——那样这页表就白做了。

### 对答案的两种方式

```bash
# 方式一：交互答题，一题一题判分（推荐）
uv run python week01/day01_shapes/shape_quiz.py

# 方式二：直接看全部答案（只在填完之后用）
uv run python week01/day01_shapes/shape_drills.py --reveal
```

### 每题怎么想（5 步）

```
1. 把所有操作数的 shape 写出来          ← 不许跳
2. 有 @ 吗？        → 标出内维，内维必须相等然后消失
3. 有广播吗？       → 从最右对齐，在纸上画出来
4. 有 axis 吗？     → 那维消失（除非 keepdims=True）
5. 有 reshape/transpose 吗？ → 先算新 shape
```

> 详细原理见同目录 `numpy_semantics.md`；常见陷阱见那篇的 §8.6。

---

### 基础
| 操作 | 输出 shape |
|---|---|
| `x + x` | `(4, 3)` |
| `x.sum()` | `()` |
| `x.sum(axis=0)` | `(3,)` |
| `x.sum(axis=1)` | `(4,)` |
| `x.sum(axis=1, keepdims=True)` | `(4, 1)` |
| `x.T` | `(3, 4)` |

### 矩阵乘法
| 操作 | 输出 shape |
|---|---|
| `x @ W` | `(4, 5)` |
| `W.T @ x.T` | `(5, 4)` |
| `b @ W.T` | `(3,)` |
| `W.T @ b` | `ERROR` |
| `b @ W` | `ERROR` |
| `seq @ W` | `(4, 6, 5)` |
| `seq @ x` | `ERROR` |

### Broadcasting
| 操作 | 输出 shape |
|---|---|
| `x + b` | `ERROR` |
| `x + b[:D]` | `(4, 3)` |
| `seq * b` | `ERROR` |
| `seq * b[:D]` | `(4, 6, 3)` |
| `img.mean(axis=(2,3))` | `(4, 2)` |
| `img.mean(axis=(2,3), keepdims=True)` | `(4, 2, 1, 1)` |
| `img - img.mean(axis=(2,3), keepdims=True)` | `(4, 2, 32, 32)` |
| `x[:, None, :] - x[:, :, None]` | `(4, 3, 3)` |

### reshape / transpose
| 操作 | 输出 shape |
|---|---|
| `seq.transpose(0, 2, 1)` | `(4, 3, 6)` |
| `seq.reshape(B*T, D)` | `(24, 3)` |
| `seq.reshape(B, T*D)` | `(4, 18)` |
| `seq.reshape(-1)` | `(72,)` |
| `seq @ seq.transpose(0, 2, 1)` | `(4, 6, 6)` |

### Norm
| 操作 | 输出 shape |
|---|---|
| `np.linalg.norm(x)` | `()` |
| `np.linalg.norm(x, axis=1)` | `(4,)` |
| `np.linalg.norm(x, axis=1, keepdims=True)` | `(4, 1)` |
| `x / np.linalg.norm(x, axis=1, keepdims=True)` | `(4, 3)` |
| `x / np.linalg.norm(x, axis=1)` | `ERROR` |

> 一共 31 行，对应 `shape_drills.py` 的全部 31 题。
> 前四个表（20 行）是你自己填的，答对 19 道 —— 唯一漏的是 `b @ W.T`（详见 §8 第 4 条）。
> 后面补的 11 行是本表原来漏掉的。

---

## 8. 我的易错点

> 1–3 条是你自己写的，4–6 条是从 quiz 错题补上的。

**1. 广播机制要从右往左对齐。**
这条是你真踩过的坑——`x / norm(x, axis=1)` 就是因为它报的错。

**2. 一开始不清楚 `mean` 是干什么的。**
`mean` 就是求平均值。
补一句区分：`axis` 不是函数，是**参数**，告诉 `mean` 往哪个方向求平均。
`mean` 是动作，`axis` 是方向——两者不是并列关系。

**3. 三维向量的乘法 = 保留第一维、后两维做矩阵乘法。**
`(4,6,3) @ (3,5) → (4,6,5)`，中间那维 3 相消。这就是 batched matmul。

**4. 1-D 参与 matmul 是「补 1 → 运算 → 删 1」两段式，我只记住了中间那步。**
`b @ W.T`：b 补成 `(1,5)` → `(1,5)@(5,3)` → `(1,3)` → **删掉前导 1** → `(3,)`
我写成 `(1,3)`，忘了最后要删。速查表 20 道里唯一错的就是这道。

**5. `(n,)` 对齐到最后一维，不是第 0 维。**
`x / norm(x, axis=1)` 我以为能跑，实际报错——
因为 `(4,)` 里的 4 落在**最后一维**，和 x 的 3 撞上，不是落在 batch 维。
（这就是 `keepdims` 存在的全部理由。）

**6. 广播不减少维数。**
`seq * b[:D]` 我猜 `(4,6)`，实际 `(4,6,3)`。
看到 `+ - * /`，维数只会变多不会变少；只有 `sum / mean / norm` 才会让维度消失。

---

## 9. 一句话结论 + 一个疑问 + 明日第一步

- **今天的一句话结论**：初步搞懂了 NumPy 的形状机制，能自己判断常见矩阵运算的输出 shape。
  最大的收获是分清了「数学上的矩阵运算」和「这个库怎么表达它」是两回事。

- **今天没搞懂的疑问**：为什么 `.T` 对 1-D 数组什么都不做？
  `(3,).T` 还是 `(3,)` —— 这是刻意的设计，还是历史原因？

- **明天第一步**：读 R1 的 MLP / Backprop 那节，然后手推 `y = (wx+b)²` 对 `w`、`b`、`x` 的导数。

---

## 打卡

- [x] 理论完成（复习 vector / matrix / transpose / dot product / matmul / norm）
- [x] 代码完成（31 题全部作答）
- [x] 实验完成（答错的题搞清楚原因）
- [x] 当日笔记完成（第 7 节填空 + 第 8/9 节）

**完成标准：看到一段矩阵运算，不运行代码就能判断输出 shape；能解释 broadcasting 何时合法。**
