"""
NumPy 语义速览 —— 配套可运行演示
对应文档：numpy_semantics.md

跑法：
    cd ~/Desktop/my-project/deep-learning-journey
    uv run python week01/day01_shapes/numpy_demo.py

建议：一边读 numpy_semantics.md，一边对应着看输出。
"""

import numpy as np

np.set_printoptions(precision=1, suppress=True, linewidth=100)


def title(n, text):
    print(f"\n\n{'=' * 70}")
    print(f"  {n}. {text}")
    print('=' * 70)


def show(label, obj):
    """打印一个数组的 shape 和值。"""
    arr = np.asarray(obj)
    print(f"\n  {label}")
    print(f"    shape = {arr.shape}")
    if arr.size <= 24:
        print(f"    value = {arr.tolist()}" if arr.ndim <= 2 else f"    value = {arr.tolist()}")


# ==========================================================================
title(0, "数学 vs NumPy —— * 和 @ 是两个不同的东西")

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("\n  A =", A.tolist(), " B =", B.tolist())
print("\n  A * B  (逐元素乘，Hadamard 积):")
print("   ", (A * B).tolist())
print("\n  A @ B  (矩阵乘法):")
print("   ", (A @ B).tolist())
print("\n  >>> 看清楚了：结果完全不一样。数学里你不区分，NumPy 里必须区分。")


# ==========================================================================
title(1, "axis=k 的意思是「第 k 维要消失」")

x = np.arange(1, 13).reshape(4, 3)          # (B=4, D=3)
print(f"\n  x.shape = {x.shape}   # (B, D) = (batch, feature)")
print(f"  x =\n{x}")

print(f"\n  x.sum(axis=0).shape = {x.sum(axis=0).shape}     ← 第 0 维没了（跨样本相加）")
print(f"    {x.sum(axis=0).tolist()}")

print(f"\n  x.sum(axis=1).shape = {x.sum(axis=1).shape}     ← 第 1 维没了（每个样本内部求和）")
print(f"    {x.sum(axis=1).tolist()}")

print(f"\n  x.sum().shape = {x.sum().shape}                ← 全压掉，标量")
print(f"    {x.sum()}")

# axis 传元组
img = np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2)   # 小号 (B,C,H,W)
print(f"\n  img.shape = {img.shape}   # (B, C, H, W)")
print(f"  img.mean(axis=(2, 3)).shape = {img.mean(axis=(2, 3)).shape}    ← 一次压掉两维")


# ==========================================================================
title(2, "Broadcasting —— 从最右对齐，逐维比较")

print("""
  规则：
    ① 相等          -> 通过
    ② 其中一个是 1   -> 拉伸到另一个的长度
    ③ 都不等且都不是 1 -> 报错
  缺的维度当作 1。""")

a = np.arange(12).reshape(4, 3)
c = np.array([10, 20, 30])

print(f"\n  例1: a.shape={a.shape}, c.shape={c.shape}")
print("""
        纸上这样写：
            a:  4   3
            c:      3
                 ✓   ✓     -> (4, 3)
""")
r = a + c
print(f"  a + c  ->  shape = {r.shape}   ✓")

print(f"\n  例2: a.shape={a.shape}, d.shape={(4,)}")
d = np.array([10, 20, 30, 40])
print("""
        纸上这样写：
            a:  4   3
            d:      4
                 ✗        <- 3 vs 4，不等且都不是 1
""")
try:
    a + d
    print("  没报错？（不该发生）")
except ValueError as e:
    print(f"  a + d  ->  ❌ ValueError: {e}")

print("""
  >>> 注意：d 的长度是 4，在你脑子里像"batch 维"，
      但 NumPy 从【右边】对齐，把它当成了最后一维。
      语义和机制不一致 —— 这就是 keepdims 存在的理由（见第 5 节）。""")

# None 插轴
print(f"\n  用 None 手动插轴：")
print(f"    a[:, None, :].shape  = {a[:, None, :].shape}")
print(f"    a[:, :, None].shape  = {a[:, :, None].shape}")
print(f"    a[:, None, :] - a[:, :, None]  ->  { (a[:, None, :] - a[:, :, None]).shape}")
print("""
    结果 [i, j, k] = a[i,k] - a[i,j]  —— 每对元素之间的关系。
    这就是 Attention 里 QK^T 生成 (B, T, T) 分数的思路雏形。""")


# ==========================================================================
title(3, "1-D 数组在 @ 里的提升规则（不对称！）")

b = np.arange(5, dtype=float)          # (5,)
W = np.zeros((3, 5))                    # (3, 5)

print(f"\n  b.shape   = {b.shape}")
print(f"  W.shape   = {W.shape}")
print(f"  W.T.shape = {W.T.shape}")

print("""
  规则：
    1-D 在【右】边 -> 当成列向量 (k,1)，算完去掉【后面】的 1
    1-D 在【左】边 -> 当成行向量 (1,k)，算完去掉【前面】的 1
""")

print(f"  b @ W.T   (b 在左，当 (1,5)):")
try:
    print(f"    (1,5) @ (5,3) -> (1,3) -> 去掉前导 1 -> shape = {(b @ W.T).shape}   ✓")
except ValueError as e:
    print(f"    ❌ {e}")

print(f"\n  W.T @ b   (b 在右，当 (5,1)):")
try:
    print(f"    shape = {(W.T @ b).shape}")
except ValueError as e:
    print(f"    ❌ ValueError: {e}")
    print(f"       <<< 内维 3 vs 5，对不上")

print(f"\n  b @ W     (b 在左，当 (1,5)):")
try:
    print(f"    shape = {(b @ W).shape}")
except ValueError as e:
    print(f"    ❌ ValueError: {e}")
    print(f"       <<< 内维 5 vs 3，对不上")

print("""
  >>> 同一个 b，站左边能跑，站右边就炸 —— 这就是不对称。
      工程结论：DL 代码里永远显式写 (B, D)，别依赖 1-D 自动推断。""")


# ==========================================================================
title(4, ".T / reshape / transpose")

x = np.arange(6).reshape(2, 3)
print(f"\n  x.shape = {x.shape}")
print(f"  x =\n{x}")

print(f"\n  x.reshape(3, 2)  ->  {x.reshape(3, 2).shape}   # 按内存顺序重新切")
print(f"{x.reshape(3, 2)}")

print(f"\n  x.T              ->  {x.T.shape}   # 真的转置")
print(f"{x.T}")

print("""
  >>> 两个结果完全不一样！
      reshape   = 换一种"切法"，数据在内存里的顺序没变
      transpose = 真的换轴，数据顺序变了""")

# 3-D
seq = np.arange(4 * 6 * 3).reshape(4, 6, 3)
print(f"\n  3-D 时更明显： seq.shape = {seq.shape}   # (B, T, D)")

r1 = seq.reshape(6, 4, 3)
r2 = seq.transpose(1, 0, 2)
print(f"    seq.reshape(6, 4, 3).shape    = {r1.shape}")
print(f"    seq.transpose(1, 0, 2).shape  = {r2.shape}")
print(f"    shape 一样，但内容一样吗？ {np.array_equal(r1, r2)}")

print(f"\n    对比不同位置的元素：")
print(f"      位置 [0,0]   reshape -> {r1[0, 0].tolist()}    transpose -> {r2[0, 0].tolist()}   <- 恰好相同！")
print(f"      位置 [1,0]   reshape -> {r1[1, 0].tolist()}   transpose -> {r2[1, 0].tolist()}")
print("""
  >>> 这才是最阴险的地方：
      shape 一模一样，连 [0,0] 位置的元素都碰巧相同，
      但 [1,0] 已经完全不是一回事了。

      这就是 Attention 里最容易埋 bug 的坑 ——
      【形状对上了，不代表数据对上了。】

      口诀：换维度顺序 -> transpose ； 合并/拆开维度 -> reshape""")

# .T 对 1-D 无效
v = np.arange(3)
print(f"\n  坑：.T 对 1-D 无效")
print(f"    v.shape    = {v.shape}")
print(f"    v.T.shape  = {v.T.shape}    <<< 什么也没做！")


# ==========================================================================
title(5, "keepdims —— 为什么必须有它")

x = np.arange(1, 13).reshape(4, 3).astype(float)
print(f"\n  x.shape = {x.shape}")

m_bad = x.mean(axis=1)
print(f"\n  m = x.mean(axis=1)                  -> {m_bad.shape}   # 第 1 维被删了")
try:
    x - m_bad
    print("  没报错？（不该发生）")
except ValueError as e:
    print(f"  x - m  ->  ❌ ValueError: {e}")
    print(f"     (4,3) - (4,) -> 从右对齐 3 vs 4 -> 对不上")

m_ok = x.mean(axis=1, keepdims=True)
print(f"\n  m = x.mean(axis=1, keepdims=True)   -> {m_ok.shape}   # 第 1 维变成 1")
print(f"  x - m  ->  {(x - m_ok).shape}   ✓")
print(f"{x - m_ok}")

print("""
  >>> 为什么 (4,1) 就行？
      (4,1) 最后一维是 1 -> 可以被拉伸成 3 -> 通过
      (4,)  最后一维是 4 -> 和 3 冲突       -> 报错

      口诀：1 可以被广播，消失的维度不能。
           归约之后还要参与运算 -> 加 keepdims=True""")


# ==========================================================================
title(6, "np.linalg.norm 的 axis")

x = np.arange(1, 13).reshape(4, 3).astype(float)
print(f"\n  x.shape = {x.shape}")

print(f"\n  np.linalg.norm(x).shape                     = {np.linalg.norm(x).shape}    # 整个拉平求 -> 标量")
print(f"    {np.linalg.norm(x):.3f}")

n1 = np.linalg.norm(x, axis=1)
print(f"\n  np.linalg.norm(x, axis=1).shape             = {n1.shape}    # 每行一个长度")
print(f"    {n1.round(3).tolist()}")

n2 = np.linalg.norm(x, axis=1, keepdims=True)
print(f"\n  np.linalg.norm(x, axis=1, keepdims=True)    = {n2.shape}")

print(f"\n  归一化：")
try:
    x / n1
    print("  没报错？（不该发生）")
except ValueError as e:
    print(f"    x / norm(x, axis=1)                  ->  ❌ {e}")

print(f"    x / norm(x, axis=1, keepdims=True)   ->  {(x / n2).shape}   ✓")

print("""
  >>> 就是第 5 节那个坑，换个函数再来一次。

  另外注意：x[:, None, :] 那边用的是 None 插轴，norm 用的是 keepdims，
  两者效果一样，都是"造一根长度为 1 的轴"。""")


# ==========================================================================
title(7, "收尾：回来做 shape_drills.py")

print("""
  这 7 节覆盖了 shape_drills.py 的全部 31 题：

      A 组（求和轴/转置）   -> 第 1、4 节
      B 组（矩阵乘法）       -> 第 3 节     尤其 B2/B3/B4
      C 组（广播）           -> 第 2、5 节
      D 组（reshape/transpose）-> 第 4 节
      E 组（norm）           -> 第 6 节

  现在去跑：

      uv run python week01/day01_shapes/shape_drills.py

  每题在纸上走这 5 步：
      1. 写出所有操作数的 shape
      2. 有 @ 吗？        -> 标内维
      3. 有广播吗？       -> 从右对齐，画出来
      4. 有 axis 吗？     -> 那维消失（除非 keepdims）
      5. 有 reshape/transpose 吗？ -> 先算新 shape
""")
