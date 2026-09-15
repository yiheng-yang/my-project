"""
Day 1 | Tensor Shape Quiz —— 交互式答题版
=====================================================================
31 道 shape 判断题，一题一题来，输入答案即时判分。
不用改文件、不用记答案位置。

跑法：
    cd ~/Desktop/my-project/deep-learning-journey
    uv run python week01/day01_shapes/shape_quiz.py

输入格式（随便哪种都行）：
    (4, 3)      4,3      4 3          -> 元组
    ()          scalar   标量          -> 标量
    err         error    x            -> 你认为这行会报错
    （直接回车）= 跳过；  q = 退出

纪律：先自己想，别急着按回车看答案。
      猜错的地方才是今天真正学到的东西。
"""

import sys
import numpy as np

# =====================================================================
# 固定数据（和 shape_drills.py 完全一致）
# =====================================================================
rng = np.random.default_rng(0)

B, D, H, C, T = 4, 3, 5, 2, 6

x   = rng.normal(size=(B, D))          # 一批特征向量
W   = rng.normal(size=(D, H))          # 全连接层权重
b   = rng.normal(size=(H,))            # 偏置
img = rng.normal(size=(B, C, 32, 32))  # 一批图像
seq = rng.normal(size=(B, T, D))       # 一批序列


def actual_of(fn):
    """跑一遍，拿到真实 shape（或 'ERROR'）。"""
    try:
        out = fn()
        shp = getattr(out, "shape", None)
        return tuple(shp) if shp is not None else tuple(np.shape(out))
    except Exception:
        return "ERROR"


# =====================================================================
# 题库
# =====================================================================
QUESTIONS = [
    # ---- A 组 ----
    ("A", "x + x",                              lambda: x + x,
     "逐元素相加，两个 shape 一样 -> 结果 shape 不变"),
    ("A", "x.sum()",                            lambda: x.sum(),
     "不给 axis -> 全部压掉 -> 标量 ()"),
    ("A", "x.sum(axis=0)",                      lambda: x.sum(axis=0),
     "axis=0 压掉第 0 维（batch 维）"),
    ("A", "x.sum(axis=1)",                      lambda: x.sum(axis=1),
     "axis=1 压掉第 1 维（feature 维）"),
    ("A", "x.sum(axis=1, keepdims=True)",       lambda: x.sum(axis=1, keepdims=True),
     "keepdims=True -> 被压的那维【变成 1】，而不是消失"),
    ("A", "x.T",                                lambda: x.T,
     "2-D 转置：行列互换"),
    ("A", "x @ W",                              lambda: x @ W,
     "(4,3) @ (3,5)：内维 3 相消，留下外维"),

    # ---- B 组 ----
    ("B", "W.T @ x.T",                          lambda: W.T @ x.T,
     "W.T=(5,3)，x.T=(3,4)：内维 3 相消"),
    ("B", "W.T @ b",                            lambda: W.T @ b,
     "b 在【右】边 -> 当列向量 (5,1)；(5,3)@(5,1) 内维 3 vs 5"),
    ("B", "b @ W.T",                            lambda: b @ W.T,
     "b 在【左】边 -> 当行向量 (1,5)；(1,5)@(5,3)->(1,3)->去掉前导 1"),
    ("B", "b @ W",                              lambda: b @ W,
     "b 当 (1,5)，W 是 (3,5)：内维 5 vs 3"),
    ("B", "seq @ W",                            lambda: seq @ W,
     "batched matmul：(4,6,3)@(3,5)，内维 3 相消，batch 维保留"),
    ("B", "seq @ x",                            lambda: seq @ x,
     "batch 对齐后是 (6,3)@(4,3)：内维 3 vs 4"),

    # ---- C 组 ----
    ("C", "x + b",                              lambda: x + b,
     "从最右对齐：(4,3) vs (5,) -> 3 vs 5，都不等且都不是 1"),
    ("C", "x + b[:D]",                          lambda: x + b[:D],
     "从最右对齐：(4,3) vs (3,) -> 最右维 3==3，前者缺的一维当 1"),
    ("C", "seq * b",                            lambda: seq * b,
     "从最右对齐：(4,6,3) vs (5,) -> 最右维 3 vs 5"),
    ("C", "seq * b[:D]",                        lambda: seq * b[:D],
     "从最右对齐：(4,6,3) vs (3,) -> 最右维 3==3"),
    ("C", "img.mean(axis=(2, 3))",              lambda: img.mean(axis=(2, 3)),
     "axis=(2,3) 一次压掉第 2、3 两维"),
    ("C", "img.mean(axis=(2,3), keepdims=True)",lambda: img.mean(axis=(2, 3), keepdims=True),
     "两维都【变成 1】，轴数保住"),
    ("C", "img - img.mean(axis=(2,3), keepdims=True)",
                                                lambda: img - img.mean(axis=(2, 3), keepdims=True),
     "(4,2,1,1) 广播回 (4,2,32,32) —— 这就是逐通道归一化"),
    ("C", "x[:, None, :] - x[:, :, None]",      lambda: x[:, None, :] - x[:, :, None],
     "(4,1,3) 和 (4,3,1) 广播 -> 结果 [i,j,k] = x[i,k] - x[i,j]"),

    # ---- D 组 ----
    ("D", "seq.transpose(0, 2, 1)",             lambda: seq.transpose(0, 2, 1),
     "交换第 1、2 维"),
    ("D", "seq.reshape(B * T, D)",              lambda: seq.reshape(B * T, D),
     "合并前两维：4*6=24"),
    ("D", "seq.reshape(B, T * D)",              lambda: seq.reshape(B, T * D),
     "合并后两维：6*3=18"),
    ("D", "seq.reshape(-1)",                    lambda: seq.reshape(-1),
     "全拍平：4*6*3"),
    ("D", "seq @ seq.transpose(0, 2, 1)",       lambda: seq @ seq.transpose(0, 2, 1),
     "(4,6,3)@(4,3,6)：内维 3 相消 -> 剩下 (B, T, T)。这就是 Attention Score 的形状"),

    # ---- E 组 ----
    ("E", "np.linalg.norm(x)",                  lambda: np.linalg.norm(x),
     "不给 axis -> 整个数组拉平求 -> 标量"),
    ("E", "np.linalg.norm(x, axis=1)",          lambda: np.linalg.norm(x, axis=1),
     "沿第 1 维求，那一维消失"),
    ("E", "np.linalg.norm(x, axis=1, keepdims=True)",
                                                lambda: np.linalg.norm(x, axis=1, keepdims=True),
     "那一维变成 1，可广播"),
    ("E", "x / np.linalg.norm(x, axis=1, keepdims=True)",
                                                lambda: x / np.linalg.norm(x, axis=1, keepdims=True),
     "(4,3) / (4,1) -> 每行 L2 归一化"),
    ("E", "x / np.linalg.norm(x, axis=1)",      lambda: x / np.linalg.norm(x, axis=1),
     "(4,3) / (4,) -> 从最右对齐 3 vs 4 —— 差一个 keepdims 就炸"),
]


# =====================================================================
# 输入解析
# =====================================================================
def parse(s):
    """把用户输入解析成元组 或 'ERROR'。解析不了返回 None。"""
    t = s.strip().lower().replace(" ", "")
    if t in ("err", "error", "e", "x", "报错", "错"):
        return "ERROR"
    if t in ("", "()", "scalar", "标量"):
        return ()
    t = t.strip("()")
    if not t:
        return ()
    parts = [p for p in t.split(",") if p]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def main():
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"{BOLD}  Day 1 · Tensor Shape Quiz —— 共 {len(QUESTIONS)} 题{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}")

    print(f"\n{BOLD}已知变量的 shape：{RESET}")
    print(f"  B, D, H, C, T = {B}, {D}, {H}, {C}, {T}")
    print(f"  x   = ({B}, {D})              一批特征向量")
    print(f"  W   = ({D}, {H})              全连接层权重")
    print(f"  b   = ({H},)                偏置")
    print(f"  img = ({B}, {C}, 32, 32)          一批图像")
    print(f"  seq = ({B}, {T}, {D})              一批序列")

    print(f"\n{DIM}输入：(4,3) 或 4,3 或 () 或 err（报错）| 回车=跳过 | q=退出{RESET}")

    right, wrong = [], []

    for i, (grp, code, fn, why) in enumerate(QUESTIONS, 1):
        print(f"\n{DIM}{'-' * 66}{RESET}")
        print(f"{BOLD}[{i:>2}/{len(QUESTIONS)}]  {grp} 组   {code}{RESET}")

        try:
            raw = input(f"     你的答案 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n（已中断）")
            break

        if raw.lower() in ("q", "quit", "exit"):
            print("\n（退出）")
            break

        act = actual_of(fn)

        if raw == "":
            print(f"     {DIM}跳过。答案是 {act}{RESET}")
            continue

        ans = parse(raw)
        if ans is None:
            print(f"     {RED}看不懂这个输入，跳过。答案是 {act}{RESET}")
            continue

        if ans == act:
            right.append(code)
            print(f"     {GREEN}✓ 对{RESET}")
        else:
            wrong.append((code, ans, act, why))
            print(f"     {RED}✗ 错了{RESET}  {DIM}你猜 {ans}，实际是 {act}{RESET}")

        print(f"     {DIM}{why}{RESET}")

    # ---------------- 结算 ----------------
    total = len(right) + len(wrong)
    print(f"\n\n{BOLD}{'=' * 66}{RESET}")
    print(f"{BOLD}  答对 {len(right)} / {total}{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}")

    if wrong:
        print(f"\n{BOLD}猜错的题（这才是今天真正的收获，抄进 tensor_shape_notes.md 第 8 节）：{RESET}")
        for code, ans, act, why in wrong:
            print(f"\n  {RED}✗{RESET} {BOLD}{code}{RESET}")
            print(f"      你猜 {ans}  ->  实际 {act}")
            print(f"      {DIM}{why}{RESET}")
        print(f"\n{DIM}提示：归纳一下你错在哪一类（广播？1-D 提升？keepdims？reshape？）{RESET}")
    else:
        print(f"\n{GREEN}全对！{RESET}{DIM}那可以去看 tensor_shape_notes.md 补速查表了。{RESET}")

    if total < len(QUESTIONS):
        print(f"\n{DIM}还有 {len(QUESTIONS) - total} 题没答，重跑一次补齐。{RESET}")

    print(f"\n{DIM}完整答案：uv run python week01/day01_shapes/shape_drills.py --reveal{RESET}\n")


if __name__ == "__main__":
    main()
