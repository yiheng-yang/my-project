"""
Day 4 | linear_regression_numpy.py 的测试
=====================================================================
跑法：
    cd ~/Desktop/my-project/deep-learning-journey
    uv run python week01/day04_linear_regression/test_training.py

没实现的函数会友好跳过，可以随时跑来检查进度。

测试组：
    T1      forward
    T2      mse_loss
    T3      compute_gradients（数值梯度检验 ← 最关键）
    T4      sgd_step
    T5      端到端训练（loss 下降 + 参数恢复）
    T6      tiny-overfit（极小数据应该能过拟合到接近 0）
"""

import numpy as np
import linear_regression_numpy as LR

OK, BAD = "✓", "✗"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{OK if cond else BAD}] {name}")
    if detail:
        print(f"        {detail}")
    return cond


# ---------------------------------------------------------------------
def loss_at(X, y, w, b):
    return LR.mse_loss(LR.forward(X, w, b), y)


def numerical_grads(X, y, w, b, eps=1e-6):
    """中心差分：数值估计 dw、db（不需要任何推导）"""
    dw = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        dw[i] = (loss_at(X, y, wp, b) - loss_at(X, y, wm, b)) / (2 * eps)
    db = (loss_at(X, y, w, b + eps) - loss_at(X, y, w, b - eps)) / (2 * eps)
    return dw, db


# =====================================================================
def main():
    print("\n" + "=" * 70)
    print("  Day 4 · 测试")
    print("=" * 70)

    rng = np.random.default_rng(0)
    X, y, w_true, b_true = LR.make_data(n=50, d=3, seed=1)
    w = rng.normal(size=3)
    b = 0.7

    n_done = 0

    # ---------------------------------------------------------------
    # T1 forward
    # ---------------------------------------------------------------
    try:
        out = LR.forward(X, w, b)
    except NotImplementedError:
        print("\n--- T1 · forward ---\n  ⏳ 还没实现 —— 跳过")
        out = None
    else:
        n_done += 1
        print("\n--- T1 · forward ---\n")
        check("T1a 输出形状是 (N,)", out.shape == (X.shape[0],),
              f"{X.shape} @ {w.shape} -> {out.shape}")
        ref = X @ w + b
        check("T1b 值正确（X@w + b）", np.allclose(out, ref),
              f"最大误差 {np.abs(out - ref).max():.2e}")
        check("T1c b 广播正确（每个样本都加了）",
              np.isclose(out[0] - (X[0] @ w), b),
              f"out[0] - X[0]@w = {out[0] - (X[0] @ w):.4f}，b = {b:.4f}")

    # ---------------------------------------------------------------
    # T2 mse_loss
    # ---------------------------------------------------------------
    try:
        v = LR.mse_loss(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    except NotImplementedError:
        print("\n--- T2 · mse_loss ---\n  ⏳ 还没实现 —— 跳过")
    else:
        n_done += 1
        print("\n--- T2 · mse_loss ---\n")
        check("T2a 返回标量", np.isscalar(v) or np.ndim(v) == 0, f"得到 {v}")
        check("T2b 手算验证: [1,2] vs [1,4] -> 2.0", np.isclose(v, 2.0),
              f"期望 2.0，得到 {v}")
        check("T2c 完全预测正确时 loss = 0",
              np.isclose(LR.mse_loss(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 0.0))

    # ---------------------------------------------------------------
    # T3 梯度检验  ★
    # ---------------------------------------------------------------
    try:
        if out is None:
            raise NotImplementedError
        dw, db = LR.compute_gradients(X, y, out)
    except NotImplementedError:
        print("\n--- T3 · compute_gradients ---\n  ⏳ 还没实现 —— 跳过")
    else:
        n_done += 1
        print("\n--- T3 · compute_gradients（数值梯度检验）---\n")
        num_dw, num_db = numerical_grads(X, y, w, b)
        check("T3a dw 形状是 (D,)", dw.shape == w.shape, f"得到 {dw.shape}")
        check("T3b dw 与数值梯度一致",
              np.allclose(dw, num_dw, rtol=1e-5, atol=1e-7),
              f"\n            解析: {dw.round(6)}\n            数值: {num_dw.round(6)}")
        check("T3c db 与数值梯度一致",
              np.isclose(db, num_db, rtol=1e-5, atol=1e-7),
              f"解析 {db:.8f} vs 数值 {num_db:.8f}")

    # ---------------------------------------------------------------
    # T4 sgd_step
    # ---------------------------------------------------------------
    try:
        w2, b2 = LR.sgd_step(w, b, np.ones_like(w), 1.0, lr=0.1)
    except NotImplementedError:
        print("\n--- T4 · sgd_step ---\n  ⏳ 还没实现 —— 跳过")
    else:
        n_done += 1
        print("\n--- T4 · sgd_step ---\n")
        check("T4a 往负梯度方向走", np.allclose(w2, w - 0.1 * 1.0),
              f"w 减去了 lr*dw")
        check("T4b 返回两个值 (w, b)", b2 == b - 0.1 * 1.0,
              f"b = {b2:.4f}")

    # ---------------------------------------------------------------
    if n_done < 4:
        print(f"\n  实现进度：{n_done} / 4 个函数（T1-T4 需要前四个都实现）")
        print("  先把 4 个 TODO 填完，再跑一次。\n")
        return

    # ---------------------------------------------------------------
    # T5 端到端训练
    # ---------------------------------------------------------------
    print("\n--- T5 · 端到端训练（lr=0.1, 300 epochs）---\n")
    w_fit, b_fit, hist = LR.train(X, y, lr=0.1, epochs=300)
    check("T5a loss 明显下降（降到初始的 5% 以下）", hist[-1] < hist[0] * 0.05,
          f"{hist[0]:.4f} -> {hist[-1]:.6f}   （噪声 0.5，不可约误差 ≈ 0.25）")
    check("T5b 后期仍在下降（无发散）", hist[-1] < hist[len(hist) // 2],
          f"后半程 {hist[len(hist)//2]:.6f} -> {hist[-1]:.6f}")
    check("T5c w 接近真实值", np.allclose(w_fit, w_true, atol=0.3),
          f"拟合 {w_fit.round(3)} vs 真实 {w_true}")
    check("T5d b 接近真实值", abs(b_fit - b_true) < 0.3,
          f"拟合 {b_fit:.3f} vs 真实 {b_true}")

    # ---------------------------------------------------------------
    # T6 tiny-overfit
    # ---------------------------------------------------------------
    #   模型有 4 个参数（3 个 w + 1 个 b）。
    #   样本数 <= 4 才可能【精确】拟合；样本更多就是超定方程组，
    #   最优解是"最小二乘残差"，不可能到 0。
    #   所以这里取 4 个样本 —— 恰好可以精确解出来。
    # ---------------------------------------------------------------
    print("\n--- T6 · tiny-overfit（4 样本 / 4 参数，应该几乎完美拟合）---\n")
    Xs, ys = X[:4], y[:4]
    w_s, b_s, hist_s = LR.train(Xs, ys, lr=0.05, epochs=10000)
    check("T6 极小数据上 loss 接近 0", hist_s[-1] < 1e-3,
          f"loss = {hist_s[-1]:.3e}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  已跑部分：{sum(_results)} / {len(_results)} 通过")
    print(f"  实现进度：{n_done} / 4 个函数")
    print("=" * 70)

    if all(_results):
        print("""
  全过了。你的线性回归能跑起来了。

  现在去跑真正的实验：
      uv run python week01/day04_linear_regression/linear_regression_numpy.py

  它会试三个学习率、画出 loss 曲线、存成 loss_curve.png。
  看完曲线回答脚本最后那 5 个问题。
""")
    else:
        print("""
  有测试没过。按组看：
    T1  -> forward。检查 (N,D)@(D,) 的形状，b 能不能广播上去
    T2  -> mse_loss。检查是不是先算残差再平方求平均
    T3  -> 梯度。跟手推的公式对一遍，尤其是 (2/N) 那个系数
    T5/T6 -> 训练不收敛。八成是梯度符号或缩放错了
""")
    print()


if __name__ == "__main__":
    main()
