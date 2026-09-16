"""
Day 3 | losses.py 的测试
=====================================================================
跑法：
    cd ~/Desktop/my-project/deep-learning-journey
    uv run python week01/day03_losses/test_losses.py

三组测试，写好一个跑一次：
    T1-T4   stable_softmax
    T5-T6   cross_entropy
    T7      cross_entropy_grad（梯度检验）

还没实现的那组会友好跳过，不会报错 —— 你可以随时跑来检查进度。
"""

import numpy as np
import losses as L

OK, BAD = "✓", "✗"
_results = []


# ---------------------------------------------------------------------
# 工具：数值梯度（就是 Day 2 你用的那个思路，这里扩展成支持矩阵）
# ---------------------------------------------------------------------
def numerical_grad(f, x, eps=1e-6):
    """中心差分。x 可以是任意形状的数组，f 返回标量。"""
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps; fp = f(x)
        x[idx] = old - eps; fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def check(name, cond, detail=""):
    _results.append(bool(cond))
    mark = OK if cond else BAD
    print(f"  [{mark}] {name}")
    if detail:
        print(f"        {detail}")
    return cond


# =====================================================================
# T1-T4 · stable_softmax
# =====================================================================
def section_softmax(rng):
    print("\n--- T1-T4 · stable_softmax ---\n")
    logits = rng.normal(size=(4, 5))

    # T1 形状 + 和为 1
    p = L.stable_softmax(logits)
    check("T1a 形状不变", p.shape == logits.shape,
          f"输入 {logits.shape} -> 输出 {p.shape}")
    check("T1b 每行和为 1", np.allclose(p.sum(axis=-1), 1.0),
          f"各行的和 = {p.sum(axis=-1).round(6)}")
    check("T1c 全部为正", np.all(p > 0),
          f"最小值 = {p.min():.3e}")

    # T2 平移不变
    p2 = L.stable_softmax(logits + 100.0)
    check("T2 平移不变（加 100 结果一样）", np.allclose(p, p2, atol=1e-12),
          "softmax(x) == softmax(x + 100)")

    # T3 数值稳定性
    big = np.array([[1000.0, 1001.0, 1002.0]])
    try:
        pb = L.stable_softmax(big)
        ok = np.all(np.isfinite(pb)) and np.allclose(pb.sum(), 1.0)
        check("T3 大数值不溢出", ok,
              f"logits=[1000,1001,1002] -> {pb.round(6)}")
    except Exception as e:
        check("T3 大数值不溢出", False, f"抛异常了：{e}")
        pb = None

    # T4 对照朴素实现
    if pb is not None:
        print("\n        （对照）朴素实现 exp(x)/Σexp(x) 遇到大数值会怎样：")
        with np.errstate(over="ignore", invalid="ignore"):
            naive = np.exp(big) / np.exp(big).sum()
        print(f"        朴素版 -> {naive}   ← inf/nan")
        print(f"        你的版 -> {pb}   ← 正常")


# =====================================================================
# T5-T6 · cross_entropy
# =====================================================================
def section_ce(rng):
    print("\n--- T5-T6 · cross_entropy ---\n")

    # T5 手算验证：logits 全 0 -> softmax = 1/3 -> CE = -log(1/3) = log(3)
    lg1, lb1 = np.zeros((1, 3)), np.array([0])
    expected = np.log(3)
    got = L.cross_entropy(lg1, lb1)
    check("T5 手算验证：logits 全 0 -> log(3)",
          np.isclose(got, expected, atol=1e-9),
          f"期望 {expected:.6f}，得到 {got:.6f}")

    # T5b 极准的预测 -> loss 接近 0
    got2 = L.cross_entropy(np.array([[10.0, 0.0, 0.0]]), lb1)
    check("T5b 预测极准时 loss 接近 0", got2 < 0.001, f"得到 {got2:.3e}")

    # T6 与参考一致
    logits_b = rng.normal(size=(8, 5))
    labels_b = rng.integers(0, 5, size=8)
    mine = L.cross_entropy(logits_b, labels_b)
    ref = L._ref_cross_entropy(logits_b, labels_b)
    check("T6 与参考实现一致", np.isclose(mine, ref, atol=1e-10),
          f"你的 {mine:.8f} vs 参考 {ref:.8f}")

    # T6b 大 logits 也不炸
    try:
        v = L.cross_entropy(np.array([[1000.0, 1001.0], [0.0, 0.0]]), np.array([0, 1]))
        check("T6b 大 logits 下 loss 有限", np.isfinite(v), f"得到 {v:.6f}")
    except Exception as e:
        check("T6b 大 logits 下 loss 有限", False, f"抛异常了：{e}")


# =====================================================================
# T7 · 梯度检验
# =====================================================================
def section_grad(rng):
    print("\n--- T7 · 梯度检验（用数值梯度验证你的解析梯度）---\n")

    logits_g = rng.normal(size=(4, 3))
    labels_g = rng.integers(0, 3, size=4)

    num = numerical_grad(lambda x: L.cross_entropy(x, labels_g), logits_g)
    ana = L.cross_entropy_grad(logits_g, labels_g)

    close = np.allclose(num, ana, rtol=1e-5, atol=1e-7)
    check("T7 梯度与数值梯度一致", close)
    if not close:
        print(f"\n        数值: {num[0].round(6)}")
        print(f"        你的: {ana[0].round(6)}")
        return

    print(f"\n        ∂loss/∂logits 形状 = {ana.shape}")
    print(f"        观察这行梯度：{ana[0].round(6)}")
    print(f"          • 正确答案是类别 {labels_g[0]}，它是这一行里【最负】的那个 "
          f"（{'✓' if np.argmin(ana[0]) == labels_g[0] else '✗'}）")
    print(f"          • 其他位置都是正的")
    print(f"          • 整行加起来 ≈ {ana[0].sum():.3e}（应该接近 0）")
    print(f"\n        为什么会这样：梯度 = softmax − one_hot")
    print(f"        正确类别位置上减了 1，所以变负；其他位置原样保留，所以是正。")
    print(f"        含义：正确类别的分数该【往上推】，其他类别该【往下压】。")


# =====================================================================
def main():
    print("\n" + "=" * 68)
    print("  Day 3 · losses.py 测试")
    print("=" * 68)

    rng = np.random.default_rng(0)
    dummy = np.array([[1.0, 2.0, 3.0]])
    lb = np.array([0])

    # 每组：标题 / 探针（检查实现了没）/ 正文
    sections = [
        ("T1-T4 · stable_softmax",
         lambda: L.stable_softmax(dummy),
         lambda: section_softmax(rng)),
        ("T5-T6 · cross_entropy",
         lambda: L.cross_entropy(dummy, lb),
         lambda: section_ce(rng)),
        ("T7 · 梯度检验",
         lambda: L.cross_entropy_grad(dummy, lb),
         lambda: section_grad(rng)),
    ]

    n_done = 0
    for title, probe, run in sections:
        try:
            probe()
        except NotImplementedError:
            print(f"\n--- {title} ---\n")
            print(f"  ⏳ 还没实现 —— 跳过")
            print(f"     （填完 losses.py 里对应的 TODO，再跑一次）")
            continue
        n_done += 1
        run()

    # ---------------- 结算 ----------------
    print("\n" + "=" * 68)
    if _results:
        n_ok = sum(_results)
        print(f"  已跑部分：{n_ok} / {len(_results)} 通过")
    print(f"  实现进度：{n_done} / 3 个函数")
    print("=" * 68)

    if n_done < 3:
        todo = ["stable_softmax", "cross_entropy", "cross_entropy_grad"][n_done:]
        print(f"\n  还差：{', '.join(todo)}")
        print(f"  一次写一个，写完就重跑这条命令。\n")
    elif all(_results):
        print("""
  三个函数全部写好，10 项测试全过。

  留意 T7 那个结果 —— ∂loss/∂logits = softmax − one_hot，
  简单到不像话，但这是真的。Day 4 训练线性回归和
  Day 6 训练 softmax 分类器时，用的就是这个梯度。
""")
    else:
        print("""
  有测试没过。按组看：
    T1-T4 没过 -> softmax。检查有没有减最大值、keepdims 加了吗
    T5-T6 没过 -> cross_entropy。检查 logsumexp 的 max-shift 写对没
    T7 没过    -> 梯度。检查是不是忘了除以 N（loss 是平均）
""")
    print()


if __name__ == "__main__":
    main()
