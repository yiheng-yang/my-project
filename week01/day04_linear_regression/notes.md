# Day 4 · Linear Regression from scratch

> Week 1 · Day 4 —— **第一次完整训练一个模型**

---

## 一、训练循环（这是 Week 1 要打通的整条链）

```
for epoch in range(epochs):

    ① forward     y_hat = X @ w + b          算预测
    ② loss        L = mean((y_hat - y)²)     错得多离谱
    ③ gradients   dw, db = ...               往哪调   ← 链式法则
    ④ update      w -= lr * dw               调一步   ← 优化器
                  b -= lr * db
```

**四步循环，跑几百次，loss 就下去了。**

前三天学的零件，今天全用上了：

| 步骤 | 用到 |
|---|---|
| ① forward | Day 1 的形状语义（`(N,D) @ (D,) -> (N,)`）|
| ② loss | MSE（其实是高斯假设下的 MLE）|
| ③ gradients | Day 2 的链式法则 |
| ④ update | 今天新学的（其实就一行）|

---

## 二、手推 MSE 的梯度（今天最关键的一步）

```
L = (1/N) Σᵢ (y_hatᵢ − yᵢ)²          其中 y_hat = X·w + b

设 r = y_hat − y        残差，shape (N,)

dL/dy_hat = (2/N) · r

dy_hat/dw = X      ->   dL/dw = (2/N) · Xᵀ @ r
dy_hat/db = 1      ->   dL/db = (2/N) · r.sum()
```

**形状自检**：
```
Xᵀ @ r :  (D,N) @ (N,)  ->  (D,)    ← 和 w 同形 ✓
r.sum():  标量                      ← 和 b 同形 ✓
```

> 形状对不上，说明推错了。**这是最实用的自检方法**——不用看公式，看 shape。

---

## 三、实验结论（5 行，跑完脚本后填）

跑这个命令，它会试三个学习率并画出曲线：

```bash
uv run python week01/day04_linear_regression/linear_regression_numpy.py
```

然后看 `loss_curve.png`，回答：

**1. 学习率太小（0.001）会怎样？**


**2. 学习率合适（0.1）会怎样？**


**3. 学习率太大（1.0）呢？曲线最后是收敛还是爆炸？**


**4. 三个学习率里，哪个恢复出来的 w、b 最接近真实值？**


**5. 为什么梯度下降要往【负】梯度走？**
> 提示：梯度指向哪个方向？


---

## 四、检验与思考（15 min）

**1. 为什么 MSE 的梯度里有 (2/N)？这个 2 是从哪来的？**


**2. 如果把 learning rate 调大 10 倍，会发生什么？为什么？**
> 提示：想想梯度下降是"走一步"，步子太大会怎样


**3. 为什么从 w=0, b=0 开始？换成别的初值结果会不一样吗？**
> 提示：线性回归的 loss 曲面是什么形状？


**4. 训练 loss 降到 0.25 就不动了，这是不是训练失败？**
> 提示：数据里有 noise=0.5。想想 0.5² 等于多少


**5. （选做）为什么说"最小化 MSE"就是"在高斯噪声假设下做 MLE"？**


---

## 打卡

- [ ] 概念完成（训练循环四步 + MSE 梯度手推）
- [ ] 编码完成（`linear_regression_numpy.py` 四个函数）
- [ ] 检验完成（`test_training.py` 全部通过）
- [ ] 实验完成（三个学习率的曲线 + 5 行结论）
- [ ] 思考题落笔
