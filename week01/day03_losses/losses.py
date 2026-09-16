"""
Day 3 | 损失函数：softmax + cross entropy
=====================================================================
今天要写三个函数，它们是后面所有分类任务的基石。

    stable_softmax       把任意实数向量变成概率分布
    cross_entropy        从 logits 直接算交叉熵
    cross_entropy_grad   交叉熵对 logits 的梯度

三个函数之间的关系：

    logits ──softmax──> 概率
       │
       └──cross_entropy──> loss（一个标量）      ← 训练要压小的那个
       │
       └──cross_entropy_grad──> 梯度              ← 反向传播的起点

写好之后跑：
    uv run python week01/day03_losses/test_losses.py
"""

import numpy as np


# =====================================================================
# ① stable_softmax —— 把分数变成概率
# ---------------------------------------------------------------------
#   数学定义：
#
#                       exp(x_i)
#       softmax(x)_i = ─────────────────
#                       Σ_j exp(x_j)
#
#   但直接这么写会溢出：exp(1000) = inf
#
#   稳定版技巧：先减去最大值（softmax 对平移不变，结果完全一样）
#
#       x = x - max(x)
#       再套上面的公式
#
#   参数：
#       logits : (N, C)  —— N 个样本，每个 C 个类别的原始分数
#       axis   : 沿哪一维做归一化。默认 -1（最后一维，也就是类别维）
#                想想 Day 1 学的 axis 语义：这一维会被"压掉"
#
#   返回：
#       (N, C) 的概率，每行的和必须是 1
# =====================================================================
def stable_softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=float)

    # ↓↓↓ TODO: 实现稳定版 softmax ↓↓↓
    #
    # 提示三步：
    #   1. 减去最大值（要用 keepdims=True，想想为什么）
    #   2. 取 exp
    #   3. 除以和（同样要 keepdims=True）
    #
    raise NotImplementedError("还没实现 stable_softmax")
    # ↑↑↑ TODO ↑↑↑


# =====================================================================
# ② cross_entropy —— 从 logits 直接算交叉熵
# ---------------------------------------------------------------------
#   它要算的是：
#
#       loss = 平均( −log( 模型给【正确类别】的预测概率 ) )
#
#   ⚠️ 关键：参数叫 logits，不是概率。
#      PyTorch 的 nn.CrossEntropyLoss 也是这样 —— 传 logits 进去。
#
#   为什么不先 softmax 再 log？
#       softmax 之后再 log，数值精度会丢。
#       正确做法是用 log-sum-exp 直接算：
#
#       log(正确类别的概率) = logit[正确类别] − logsumexp(logits)
#
#       logsumexp(x) = log(Σ exp(x_j)) = max(x) + log(Σ exp(x − max(x)))
#
#   参数：
#       logits : (N, C)
#       labels : (N,)   每个样本的正确类别下标（整数）
#
#   返回：
#       一个标量 —— 这一批的平均 loss
#
#   提示：用 np.arange(N) 配合 labels 取到"每个样本正确类别"的 logit
# =====================================================================
def cross_entropy(logits, labels):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)

    # ↓↓↓ TODO: 实现 ↓↓↓
    #
    # 步骤提示：
    #   1. 算 logsumexp（用 max-shift 保持稳定）
    #   2. 取出每个样本正确类别的 logit
    #   3. loss = logsumexp − 正确类别的 logit，再取平均
    #
    raise NotImplementedError("还没实现 cross_entropy")
    # ↑↑↑ TODO ↑↑↑


# =====================================================================
# ③ cross_entropy_grad —— 交叉熵对 logits 的梯度
# ---------------------------------------------------------------------
#   这是 DL 里最优雅的结果之一：
#
#       ∂loss/∂logits = softmax(logits) − one_hot(labels)
#
#   就是"预测减真实"，没有指数也没有除法。
#
#   ⚠️ 注意 loss 是【平均】，所以梯度要除以 N
#
#   参数：
#       logits : (N, C)
#       labels : (N,)
#
#   返回：
#       (N, C) 的梯度
# =====================================================================
def cross_entropy_grad(logits, labels):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    N = logits.shape[0]

    # ↓↓↓ TODO: 实现 ↓↓↓
    #
    # 步骤提示：
    #   1. probs = stable_softmax(logits)
    #   2. 在【正确类别】的位置上减 1
    #   3. 除以 N（因为 loss 是平均）
    #
    raise NotImplementedError("还没实现 cross_entropy_grad")
    # ↑↑↑ TODO ↑↑↑


# =====================================================================
# 参考实现（测试用，别提前看）
# =====================================================================
def _ref_softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=float)
    x = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _ref_cross_entropy(logits, labels):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    m = logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(logits - m).sum(axis=-1)) + m.squeeze(-1)
    correct = logits[np.arange(len(labels)), labels]
    return float((lse - correct).mean())


def _ref_cross_entropy_grad(logits, labels):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    N = logits.shape[0]
    p = _ref_softmax(logits)
    p[np.arange(N), labels] -= 1.0
    return p / N
