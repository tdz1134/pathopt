"""G² 连续三次贝塞尔拐角平滑（Yang & Sukkarieh 对称构造简化版）。

思路与样条类算法完全不同：直线段大部分保留，仅在每个拐角处
用两段三次贝塞尔"削角"，满足：
  - 接入/接出直线处曲率为 0（G2 与直线拼接）
  - 两段贝塞尔在中点处位置、切向、曲率连续（G2）
  - 削角距离 d 隐式控制最大曲率上界

对称构造（γ 为转角，u' = normalize(t2 - t1) 为外角平分线）:
  P0 = W2 - d·t1,  P1 = W2 - 0.5d·t1,  P2 = W2 - 0.25d·t1,  P3 = W2 + m·u'
  Q0 = P3,  Q1 = W2 + 0.25d·t2,  Q2 = W2 + 0.5d·t2,  Q3 = W2 + d·t2
其中 m = 0.25·d·sin(γ/2)。P0/P1/P2 共线保证 κ(P0)=0；
关于 u' 轴的镜像对称保证 Q 段与 P 段在 P3 处 G2 连续。
"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import resample_by_spacing


def _cubic_bezier(P0, P1, P2, P3, t):
    s = 1.0 - t
    return (s ** 3)[:, None] * P0 + (3 * s ** 2 * t)[:, None] * P1 \
        + (3 * s * t ** 2)[:, None] * P2 + (t ** 3)[:, None] * P3


@registry.algorithm("bezier_corner_g2")
def bezier_corner_g2(pts: np.ndarray, start_tangent: float,
                     spacing: float = 0.1) -> np.ndarray:
    """G² 连续贝塞尔拐角平滑：直线 + 拐角处双三次贝塞尔削角。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return resample_by_spacing(pts, spacing) if n >= 2 else pts.copy()
    sp = max(spacing, 1e-6)

    # 各拐角单位切向与削角距离初值
    L = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.zeros(n)
    t_hat = np.zeros((n, 2))
    for i in range(1, n - 1):
        n_in, n_out = L[i - 1], L[i]
        if n_in < 1e-9 or n_out < 1e-9:
            continue
        t_hat[i] = (pts[i + 1] - pts[i]) / n_out
        d[i] = 0.4 * min(n_in, n_out)

    # 防止相邻拐角的削角区间重叠：同一段上 d_i + d_{i+1} <= 0.9·L
    for _ in range(3):
        for i in range(n - 1):
            s = d[i] + d[i + 1]
            if s > 0.9 * L[i] and s > 1e-12:
                scale = 0.9 * L[i] / s
                d[i] *= scale
                d[i + 1] *= scale

    dense = [pts[0:1]]
    cursor = pts[0].copy()  # 当前直线段起点
    for i in range(1, n - 1):
        if d[i] < 1e-9:
            continue
        t1 = (pts[i] - pts[i - 1]) / max(L[i - 1], 1e-12)
        t2 = t_hat[i]
        diff = t2 - t1
        n_diff = np.linalg.norm(diff)
        summ = t1 + t2
        n_sum = np.linalg.norm(summ)
        if n_diff < 1e-9 or n_sum < 1e-9:
            # 共线（无需削角）或 180° 回头（无法用双贝塞尔平滑）
            continue
        u_axis = diff / n_diff              # 外角平分线
        sin_half = n_diff / 2.0             # sin(γ/2)
        m = 0.25 * d[i] * sin_half
        W2 = pts[i]
        A = W2 - d[i] * t1
        B = W2 + d[i] * t2
        M = W2 + m * u_axis

        # 直线段：cursor → A
        straight_len = np.linalg.norm(A - cursor)
        if straight_len > 1e-12:
            k = max(1, int(np.ceil(straight_len / sp)))
            tt = np.arange(1, k + 1, dtype=np.float64) / k
            dense.append(cursor + tt[:, None] * (A - cursor))

        # 两段贝塞尔削角
        P0, P1, P2, P3 = A, W2 - 0.5 * d[i] * t1, W2 - 0.25 * d[i] * t1, M
        Q0, Q1, Q2, Q3 = M, W2 + 0.25 * d[i] * t2, W2 + 0.5 * d[i] * t2, B
        k1 = max(1, int(np.ceil(d[i] / sp)))
        tt = np.arange(1, k1 + 1, dtype=np.float64) / k1
        dense.append(_cubic_bezier(P0, P1, P2, P3, tt))
        tt = np.arange(1, k1 + 1, dtype=np.float64) / k1
        dense.append(_cubic_bezier(Q0, Q1, Q2, Q3, tt))
        cursor = B

    # 末段直线：cursor → pts[-1]
    tail_len = np.linalg.norm(pts[-1] - cursor)
    if tail_len > 1e-12:
        k = max(1, int(np.ceil(tail_len / sp)))
        tt = np.arange(1, k + 1, dtype=np.float64) / k
        dense.append(cursor + tt[:, None] * (pts[-1] - cursor))
    else:
        dense.append(pts[-1:])

    dense = np.vstack(dense)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense
