"""
Day 4 | Linear Regression from scratch（纯 NumPy）
=====================================================================
第一次完整训练一个模型。

把前三天学的零件装起来：

    数据 → 模型(forward) → 损失(MSE) → 梯度(链式法则) → 更新(SGD)
                                          ↑
                              这一步你会手推，然后数值检验

你要填 4 个函数（标了 TODO）：
    forward            前向：算预测
    mse_loss           损失：算错得多离谱
    compute_gradients  梯度：算往哪调   ← 今天最核心
    sgd_step           更新：真的调一步

跑法：
    cd ~/Desktop/my-project/deep-learning-journey
    uv run python week01/day04_linear_regression/test_training.py   # 先验证
    uv run python week01/day04_linear_regression/linear_regression_numpy.py  # 再训练+画图
"""

import numpy as np


# =====================================================================
# 1. 造数据（已给，不用改）
# ---------------------------------------------------------------------
#   真实关系： y = X·w_true + b_true + 噪声
#   训练完看模型能不能把 w_true、b_true 找回来
# =====================================================================
def make_data(n=200, d=3, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5])[:d]
    b_true = 1.5
    y = X @ w_true + b_true + rng.normal(scale=noise, size=n)
    return X, y, w_true, b_true


# =====================================================================
# 2. forward —— 前向：算预测   ← TODO ①
# ---------------------------------------------------------------------
#   y_hat = X·w + b
#
#   X: (N, D)    w: (D,)    b: 标量
#   返回: (N,)
#
#   提示：用 @ 就行。想想 Day 1 学的：(N,D) @ (D,) -> (N,)
#         加上 b 的时候注意 broadcasting
# =====================================================================
def forward(X, w, b):
    # ↓↓↓ TODO ① ↓↓↓
    raise NotImplementedError("还没实现 forward")
    # ↑↑↑ TODO ① ↑↑↑


# =====================================================================
# 3. mse_loss —— 损失：均方误差   ← TODO ②
# ---------------------------------------------------------------------
#   MSE = (1/N) Σ (y_hat - y)²
#
#   返回一个标量（float）
#
#   提示：先算残差 r = y_hat - y，再 r**2 求平均
# =====================================================================
def mse_loss(y_pred, y):
    # ↓↓↓ TODO ② ↓↓↓
    raise NotImplementedError("还没实现 mse_loss")
    # ↑↑↑ TODO ② ↑↑↑


# =====================================================================
# 4. compute_gradients —— 梯度   ← TODO ③  ★ 今天最核心
# ---------------------------------------------------------------------
#   手推：
#
#       L = (1/N) Σ (y_hat_i - y_i)²        其中 y_hat = X·w + b
#
#       设 r = y_hat - y        残差，shape (N,)
#
#       dL/dy_hat = (2/N) · r
#       dy_hat/dw = X           ->  dL/dw = (2/N) · X.T @ r
#       dy_hat/db = 1           ->  dL/db = (2/N) · r.sum()
#
#   注意 X.T @ r 的形状：(D,N) @ (N,) -> (D,)   正好和 w 同形 ✓
#
#   参数：X (N,D)、y (N,)、y_pred (N,)
#   返回：(dw, db)，dw 形状 (D,)，db 是标量
# =====================================================================
def compute_gradients(X, y, y_pred):
    # ↓↓↓ TODO ③ ↓↓↓
    raise NotImplementedError("还没实现 compute_gradients")
    # ↑↑↑ TODO ③ ↑↑↑


# =====================================================================
# 5. sgd_step —— 沿负梯度走一步   ← TODO ④
# ---------------------------------------------------------------------
#   w := w - lr · dw
#   b := b - lr · db
#
#   返回更新后的 (w, b)
#
#   为什么要减？梯度指向【上升】最快的方向，所以下降要反着走。
#   lr 是学习率，决定这一步走多大。
# =====================================================================
def sgd_step(w, b, dw, db, lr):
    # ↓↓↓ TODO ④ ↓↓↓
    raise NotImplementedError("还没实现 sgd_step")
    # ↑↑↑ TODO ④ ↑↑↑


# =====================================================================
# 6. 训练循环（已给，看懂结构就行）
# ---------------------------------------------------------------------
#   这就是 Week 1 要打通的那条链，每个 epoch 跑一遍：
#
#       forward -> loss -> gradients -> update
# =====================================================================
def train(X, y, lr=0.1, epochs=200, record_every=1):
    N, D = X.shape
    w = np.zeros(D)          # 从全 0 开始（什么都不知道）
    b = 0.0
    history = []

    for epoch in range(epochs):
        y_pred = forward(X, w, b)              # ① 前向：算预测
        loss = mse_loss(y_pred, y)             # ② 损失：错得多离谱
        if epoch % record_every == 0:
            history.append(loss)

        dw, db = compute_gradients(X, y, y_pred)   # ③ 梯度：往哪调
        w, b = sgd_step(w, b, dw, db, lr)          # ④ 更新：调一步

    history.append(mse_loss(forward(X, w, b), y))
    return w, b, history


# =====================================================================
# 7. 画 loss 曲线（matplotlib 样板，不用改）
# =====================================================================
def plot_curves(results, save_path="loss_curve.png"):
    """
    results: [(label, history), ...]
    """
    import matplotlib
    matplotlib.use("Agg")          # 无窗口环境也能存图
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, hist in results:
        ax.plot(hist, label=label, linewidth=2)

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Linear Regression · loss curves under different learning rates")
    ax.set_yscale("log")           # 对数轴，慢的和快的都看得见
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"\n  曲线已保存 -> {save_path}")


# =====================================================================
# 8. 实验：三个学习率  ← 你来跑，然后写结论
# =====================================================================
if __name__ == "__main__":
    import pathlib

    X, y, w_true, b_true = make_data()

    print("=" * 66)
    print("  Day 4 · Linear Regression from scratch")
    print("=" * 66)
    print(f"\n  数据: X {X.shape}, y {y.shape}")
    print(f"  真实参数  w = {w_true},  b = {b_true}")
    print(f"  （训练完看能不能找回来）\n")

    # ---------------------------------------------------------------
    # 实验：至少 3 个学习率
    #   0.001（太小）/ 0.1（合适）/ 1.0（太大，会发散）
    #   跑完对比曲线，回答脚本最后那 5 个问题。
    #
    #   💡 想自己找发散阈值？在 0.9 ~ 1.0 之间试几个值。
    #      阈值不是固定的 —— 换一批数据它就变了（想想为什么）
    # ---------------------------------------------------------------
    LEARNING_RATES = [0.001, 0.1, 1.0]      # ← 你可以改
    EPOCHS = 200

    results = []
    for lr in LEARNING_RATES:
        w, b, hist = train(X, y, lr=lr, epochs=EPOCHS)
        results.append((f"lr = {lr}", hist))
        print(f"  lr={lr:<6}  最终 loss = {hist[-1]:>12.4f}   "
              f"w = {np.round(w, 3)}   b = {b:.3f}")

    print(f"\n  真实值        {'':<13} w = {w_true}   b = {b_true}")

    plot_curves(results, save_path=str(pathlib.Path(__file__).parent / "loss_curve.png"))

    print("""
  ──────────────────────────────────────────────────────────────
  写 5 行实验结论（填在下面，或者写进 notes.md）：

    1. 学习率太小会怎样？

    2. 学习率合适会怎样？

    3. 学习率太大呢？看看 loss 曲线最后是收敛还是爆炸

    4. 三个学习率里，哪个恢复出来的 w、b 最接近真实值？

    5. 为什么梯度下降要往【负】梯度走？

  ──────────────────────────────────────────────────────────────
""")
