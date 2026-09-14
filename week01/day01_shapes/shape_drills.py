"""
Day 1 | Tensor Shape Drills
===========================
Week 1 · Day 1 —— 线性代数只补深度学习真正会用到的部分

目标：看到一段矩阵运算时，先不运行代码就能判断输出 shape。

用法
----
    # 1. 先读题，把猜的 shape 填进每题最后的 guess=
    # 2. 一组填完就运行一次，看对错
    uv run python shape_drills.py

    # 3. 全部答完后，看答案复盘
    uv run python shape_drills.py --reveal

纪律（重要）
------------
不允许"先运行、再把结果抄进 guess"。猜错的题比猜对的题值钱 ——
猜错的位置就是你要写进 tensor_shape_notes.md 的易错点。

填法示例：  guess=(4, 3)     # 元组
            guess=()         # 标量
            guess="ERROR"    # 你判断这行会报错
"""

import sys
import numpy as np

REVEAL = "--reveal" in sys.argv

# --------------------------------------------------------------------------
# 固定随机种子：保证每次跑出来的数据一样，方便你对比
# --------------------------------------------------------------------------
rng = np.random.default_rng(0)

B, D, H, C, T = 4, 3, 5, 2, 6          # batch / feature / hidden / channel / time

x   = rng.normal(size=(B, D))          # (4, 3)  一批特征向量
W   = rng.normal(size=(D, H))          # (3, 5)  一个全连接层的权重
b   = rng.normal(size=(H,))            # (5,)    偏置
img = rng.normal(size=(B, C, 32, 32))  # (4, 2, 32, 32)  一批图像 NCHW
seq = rng.normal(size=(B, T, D))       # (4, 6, 3)  一批序列 B,T,D

# --------------------------------------------------------------------------
# 打分器
# --------------------------------------------------------------------------
_results = []


def drill(label, code, fn, guess=None):
    """跑一道 shape 题，和你的猜测对比。"""
    try:
        out = fn()
        actual = getattr(out, "shape", None)
        if actual is None:
            actual = np.shape(out)
        actual = tuple(actual)
    except Exception as e:
        actual = "ERROR"

    if guess is None:
        _results.append((label, None, actual, False))
        shown = "?" * 8 if not REVEAL else str(actual)
        print(f"  {label:<5} {code:<44} guess=________   actual={shown}")
        return

    ok = (guess == actual)
    _results.append((label, guess, actual, ok))
    mark = "\033[32m OK \033[0m" if ok else "\033[31m XX \033[0m"
    print(f"  {label:<5} {code:<44} guess={str(guess):<11} actual={actual}  {mark}")


def section(title):
    print(f"\n\033[1m{title}\033[0m")


# ==========================================================================
# A. 基础：形状、求和轴、转置
#    记住：sum(axis=k) 会「吃掉」第 k 维，除非 keepdims=True
# ==========================================================================
section("A. 基础 —— 求和轴 / 转置")

drill("A1", "x + x",                              lambda: x + x,              guess=None)
drill("A2", "x.sum()",                            lambda: x.sum(),            guess=None)
drill("A3", "x.sum(axis=0)",                      lambda: x.sum(axis=0),      guess=None)
drill("A4", "x.sum(axis=1)",                      lambda: x.sum(axis=1),      guess=None)
drill("A5", "x.sum(axis=1, keepdims=True)",       lambda: x.sum(axis=1, keepdims=True), guess=None)
drill("A6", "x.T",                                lambda: x.T,                guess=None)
drill("A7", "x @ W",                              lambda: x @ W,              guess=None)


# ==========================================================================
# B. 矩阵乘法的「内维相消」
#    规则： (..., m, k) @ (..., k, n) -> (..., m, n)
#    —— 中间那个 k 必须相等，然后消失；batch 维度靠广播对齐
# ==========================================================================
section("B. 矩阵乘法 —— 内维必须相等，然后消失")

drill("B1", "W.T @ x.T",                          lambda: W.T @ x.T,          guess=None)
drill("B2", "W.T @ b",                            lambda: W.T @ b,            guess=None)
drill("B3", "b @ W.T",                            lambda: b @ W.T,            guess=None)
drill("B4", "b @ W",                              lambda: b @ W,              guess=None)
drill("B5", "seq @ W",                            lambda: seq @ W,            guess=None)
drill("B6", "seq @ x",                            lambda: seq @ x,            guess=None)


# ==========================================================================
# C. Broadcasting
#    三条规则，从最右一维开始逐维比较：
#      1) 相等            -> 通过
#      2) 其中一个是 1     -> 拉伸到另一个的长度
#      3) 都不是 1 且不等  -> 报错
# ==========================================================================
section("C. Broadcasting —— 从最右对齐，逐维比较")

drill("C1", "x + b",                              lambda: x + b,              guess=None)
drill("C2", "x + b[:D]",                          lambda: x + b[:D],          guess=None)
drill("C3", "seq * b",                            lambda: seq * b,            guess=None)
drill("C4", "seq * b[:D]",                        lambda: seq * b[:D],        guess=None)
drill("C5", "img.mean(axis=(2, 3))",              lambda: img.mean(axis=(2, 3)),            guess=None)
drill("C6", "img.mean(axis=(2,3), keepdims=True)",lambda: img.mean(axis=(2, 3), keepdims=True), guess=None)
drill("C7", "img - img.mean(axis=(2,3), keepdims=True)",
                                                  lambda: img - img.mean(axis=(2, 3), keepdims=True), guess=None)
drill("C8", "x[:, None, :] - x[:, :, None]",      lambda: x[:, None, :] - x[:, :, None],    guess=None)


# ==========================================================================
# D. reshape / transpose
#    reshape = 按内存顺序重新切分（不改数据顺序）
#    transpose = 换轴顺序（改数据顺序）
# ==========================================================================
section("D. reshape / transpose")

drill("D1", "seq.transpose(0, 2, 1)",             lambda: seq.transpose(0, 2, 1),           guess=None)
drill("D2", "seq.reshape(B * T, D)",              lambda: seq.reshape(B * T, D),            guess=None)
drill("D3", "seq.reshape(B, T * D)",              lambda: seq.reshape(B, T * D),            guess=None)
drill("D4", "seq.reshape(-1)",                    lambda: seq.reshape(-1),                  guess=None)
drill("D5", "seq @ seq.transpose(0, 2, 1)",       lambda: seq @ seq.transpose(0, 2, 1),     guess=None)


# ==========================================================================
# E. Norm —— 以及一个非常常见的 broadcasting 陷阱
# ==========================================================================
section("E. Norm / 归一化")

drill("E1", "np.linalg.norm(x)",                  lambda: np.linalg.norm(x),                guess=None)
drill("E2", "np.linalg.norm(x, axis=1)",          lambda: np.linalg.norm(x, axis=1),        guess=None)
drill("E3", "np.linalg.norm(x, axis=1, keepdims=True)",
                                                  lambda: np.linalg.norm(x, axis=1, keepdims=True), guess=None)
drill("E4", "x / np.linalg.norm(x, axis=1, keepdims=True)",
                                                  lambda: x / np.linalg.norm(x, axis=1, keepdims=True), guess=None)
drill("E5", "x / np.linalg.norm(x, axis=1)",      lambda: x / np.linalg.norm(x, axis=1),    guess=None)


# ==========================================================================
# 结算
# ==========================================================================
answered = [r for r in _results if r[1] is not None]
correct = [r for r in answered if r[3]]
print(f"\n{'=' * 68}")
print(f"  作答 {len(answered)} / {len(_results)}    正确 {len(correct)} / {len(answered)}")

if len(answered) < len(_results):
    missing = ", ".join(r[0] for r in _results if r[1] is None)
    print(f"  还没作答：{missing}")
    print("  -> 填完 guess= 再运行一次；看答案加 --reveal")

wrong = [r for r in answered if not r[3]]
if wrong:
    print("\n  猜错的题（这才是今天真正学到东西的地方，写进 notes）：")
    for label, guess, actual, _ in wrong:
        print(f"    {label}: 你猜 {guess}  ->  实际 {actual}")
print(f"{'=' * 68}")
