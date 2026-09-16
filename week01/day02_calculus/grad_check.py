"""
Day 2 | 梯度验证器
=====================================================================
手推完导数之后，用【数值梯度】验证你推得对不对。

原理：不用管链式法则，直接给参数加一个极小的扰动，
      看函数值变了多少 —— 这就是导数的定义本身。

              f(x + ε) - f(x - ε)
      f'(x) ≈  -------------------
                     2ε

      这叫「中心差分」，误差比单边差分小得多。

用法
----
    uv run python week01/day02_calculus/grad_check.py

    # 想不出来的时候（建议先自己推 20 分钟再看）
    uv run python week01/day02_calculus/grad_check.py --answers

你要做的
--------
把两个 `analytic_grad` 里 TODO 的部分换成你手推的结果，然后跑。
数值梯度和解析梯度对上了，说明你推对了。
"""

import sys
import numpy as np

SHOW_ANSWERS = "--answers" in sys.argv


# =====================================================================
# 工具函数
# =====================================================================
def numerical_grad(f, p, eps=1e-6):
    """中心差分：数值估计梯度（不需要任何推导）"""
    p = np.asarray(p, dtype=float)
    g = np.zeros_like(p)
    for i in range(p.size):
        pp = p.copy(); pp[i] += eps
        pm = p.copy(); pm[i] -= eps
        g[i] = (f(pp) - f(pm)) / (2 * eps)
    return g


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def relu(t):
    return np.maximum(0.0, t)


def check(title, f, analytic_grad, p0, names, answer_grad=None):
    """对照数值梯度和你的解析梯度"""
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print(f"{'=' * 66}")

    ana = np.asarray(analytic_grad(np.asarray(p0, dtype=float)), dtype=float)

    # 防护：还没填就不显示数值梯度，避免直接照抄
    if np.allclose(ana, 0):
        print(f"\n  测试点：  " + "   ".join(f"{n}={v:g}" for n, v in zip(names, p0)))
        print(f"\n  看起来你还没填（结果全是 0）。")
        print(f"  先去纸上手推，把 grad1 / grad2 里的 TODO 换掉，再跑一次。")
        return None

    num = numerical_grad(f, p0)

    print(f"\n  测试点：  " + "   ".join(f"{n}={v:g}" for n, v in zip(names, p0)))
    print(f"\n  {'参数':<6} {'数值梯度':>14} {'你的推导':>14}   结果")
    print(f"  {'-' * 54}")

    all_ok = True
    for i, n in enumerate(names):
        ok = np.isclose(num[i], ana[i], rtol=1e-4, atol=1e-6)
        all_ok &= ok
        mark = "✓" if ok else "✗ 对不上"
        print(f"  {n:<6} {num[i]:>14.6f} {ana[i]:>14.6f}   {mark}")

    if all_ok:
        print(f"\n  全部通过 —— 你推对了。")
    else:
        print(f"\n  有对不上的。先别看答案，回头检查哪一步的链式法则漏了。")
        if SHOW_ANSWERS and answer_grad is not None:
            ans = np.asarray(answer_grad(np.asarray(p0, dtype=float)), dtype=float)
            print(f"\n  参考答案：")
            for i, n in enumerate(names):
                print(f"    d?/d{n} = {ans[i]:.6f}")
    return all_ok


# =====================================================================
# 练习 1：  y = (w·x + b)²
# ---------------------------------------------------------------------
#   对 w、b、x 三个变量分别求偏导。
#
#   提示：设 u = w·x + b，则 y = u²。
#        先求 dy/du，再乘上 du/d? —— 这就是链式法则。
# =====================================================================
def f1(p):
    w, b, x = p
    return (w * x + b) ** 2


def grad1(p):
    """
    在这里填你手推的结果。
    p = [w, b, x]，返回 [dy/dw, dy/db, dy/dx]
    """
    w, b, x = p

    # 你手推的结果： dw = 2(wx+b)·x   db = 2(wx+b)   dx = 2(wx+b)·w
    # 换成 Python 写法（不能省乘号，变量名不能连写）：
    u  = w * x + b
    dw = 2 * u * x
    db = 2 * u
    dx = 2 * u * w

    return np.array([dw, db, dx])


def grad1_answer(p):
    w, b, x = p
    u = w * x + b
    return np.array([2 * u * x, 2 * u * 1.0, 2 * u * w])


# =====================================================================
# 练习 2：  z = sigmoid( w2 · relu(w1·x + b1) + b2 )
# ---------------------------------------------------------------------
#   对 w1、b1、w2、b2、x 五个变量分别求偏导。
#
#   这是计划里点名的那个两层复合函数。建议按这个顺序推：
#
#       h  = w1·x + b1           一层的线性部分
#       a  = relu(h)             一层的激活
#       u  = w2·a + b2           二层的线性部分
#       z  = sigmoid(u)          二层的激活
#
#   然后从后往前：
#       dz/du  ->  dz/da 和 dz/dw2、dz/db2
#       dz/dh  ->  再往前传到 w1、b1、x
#
#   关键：relu 的导数在 h<0 时是 0，h>0 时是 1。
# =====================================================================
def f2(p):
    w1, b1, w2, b2, x = p
    h = w1 * x + b1
    a = relu(h)
    u = w2 * a + b2
    return sigmoid(u)


def grad2(p):
    """
    在这里填你手推的结果。
    p = [w1, b1, w2, b2, x]，返回 [dz/dw1, dz/db1, dz/dw2, dz/db2, dz/dx]
    """
    w1, b1, w2, b2, x = p

    # ---- forward：先把中间量算出来 ----
    h = w1 * x + b1          # 一层的线性部分
    a = relu(h)              # 一层的激活
    u = w2 * a + b2          # 二层的线性部分
    z = sigmoid(u)           # 二层的激活

    # ---- backward：从 z 开始，一步步往前推 ----
    # 每一步都是「局部导数 × 上游梯度」
    dz_du = z * (1 - z)      # Sigmoid 的导数 σ'(u) = σ(u)·(1−σ(u))

    dz_dw2 = dz_du * a       # u = w2·a + b2 -> 对 w2 的局部导数是 a
    dz_db2 = dz_du * 1.0     #                   对 b2 的局部导数是 1

    dz_da = dz_du * w2       #                   对 a  的局部导数是 w2
    dz_dh = dz_da * (h > 0)  # ReLU 的导数：h>0 时是 1，否则 0

    dz_dw1 = dz_dh * x       # h = w1·x + b1 -> 对 w1 的局部导数是 x
    dz_db1 = dz_dh * 1.0     #                   对 b1 的局部导数是 1
    dz_dx = dz_dh * w1       #                   对 x  的局部导数是 w1

    return np.array([dz_dw1, dz_db1, dz_dw2, dz_db2, dz_dx])


def grad2_answer(p):
    w1, b1, w2, b2, x = p
    h = w1 * x + b1
    a = relu(h)
    u = w2 * a + b2
    z = sigmoid(u)

    dz_du = z * (1 - z)          # sigmoid 的导数
    dz_dw2 = dz_du * a
    dz_db2 = dz_du
    dz_dh = dz_du * w2 * (h > 0)   # 乘上 relu 的导数
    dz_dw1 = dz_dh * x
    dz_db1 = dz_dh
    dz_dx = dz_dh * w1
    return np.array([dz_dw1, dz_db1, dz_dw2, dz_db2, dz_dx])


# =====================================================================
def main():
    print("\n" + "=" * 66)
    print("  Day 2 · 梯度验证器")
    print("=" * 66)
    print("""
  原理：数值梯度不需要任何推导，直接按定义算。
        把你手推的结果填进 grad1 / grad2，跑一下就知道对不对。
""")

    r1 = check(
        "练习1  y = (w·x + b)²",
        f1, grad1, [2.0, 1.0, 3.0], ["w", "b", "x"], grad1_answer,
    )

    # 两个测试点：一个 h>0（relu 通着），一个 h<0（relu 关着）
    r2a = check(
        "练习2  z = sigmoid(w2·relu(w1·x + b1) + b2)   [h > 0，relu 通]",
        f2, grad2, [0.5, 1.0, 2.0, 0.5, 1.0],
        ["w1", "b1", "w2", "b2", "x"], grad2_answer,
    )
    r2b = check(
        "练习2  同一个式子   [h < 0，relu 关]",
        f2, grad2, [-0.5, -1.0, 2.0, 0.5, 1.0],
        ["w1", "b1", "w2", "b2", "x"], grad2_answer,
    )

    def status(r):
        return "还没填" if r is None else ("通过" if r else "未通过")

    print(f"\n{'=' * 66}")
    print(f"  汇总：练习1 {status(r1)} | 练习2(relu通) {status(r2a)} | 练习2(relu关) {status(r2b)}")
    print(f"{'=' * 66}")
    print("""
  注意练习 2 的两个测试点 —— 对比一下 h>0 和 h<0 的结果：

    h > 0：五个梯度都有值，梯度能一路传到最前面。
    h < 0：relu 关掉了，【经过 relu 的那条路】全断——
           w1、b1、x 的梯度都变成 0；
           但 b2 那条路绕开了 relu，所以它还有梯度。

  这就是 dead ReLU 现象的数学来源：
  一旦某个神经元的输入长期为负，它的上游权重就再也收不到梯度、不再更新。

  提示：如果你 h<0 那组对不上，八成是 relu 的导数忘了写。
""")
    if not SHOW_ANSWERS:
        print("  （卡住了可以加 --answers 看参考答案，但建议先自己推 20 分钟）\n")


if __name__ == "__main__":
    main()
