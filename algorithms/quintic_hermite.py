"""五次 Hermite 插值 —— 匹配位置、切向、曲率（C2 过点）。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import chord_param


def _quintic_hermite_basis(u):
    """五次 Hermite 基函数 (h00, h10, h20, h01, h11, h21)。"""
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    h00 = 1 - 10 * u3 + 15 * u4 - 6 * u5
    h10 = u - 6 * u3 + 8 * u4 - 3 * u5
    h20 = u2 / 2 - 3 * u3 / 2 + 3 * u4 / 2 - u5 / 2
    h01 = 10 * u3 - 15 * u4 + 6 * u5
    h11 = -4 * u3 + 7 * u4 - 3 * u5
    h21 = u3 / 2 - u4 + u5 / 2
    return h00, h10, h20, h01, h11, h21


@registry.algorithm("quintic_hermite")
def quintic_hermite(pts: np.ndarray, start_tangent: float,
                    spacing: float = 0.1) -> np.ndarray:
    """五次 Hermite 插值：除位置、切向外还匹配节点加速度（曲率），C2 过点。

    切向取弦长参数下的中心差分，起点用 start_tangent 单位向量；
    加速度取切向的中心差分。与 clothoid 对比 G2 的两种实现路线。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    s = chord_param(pts)
    if s[-1] < 1e-12:
        return pts.copy()
    sp = max(spacing, 1e-6)

    # 一阶导（单位弧长切向）
    V = np.zeros((n, 2))
    V[0] = np.array([np.cos(start_tangent), np.sin(start_tangent)])
    for i in range(1, n - 1):
        V[i] = (pts[i + 1] - pts[i - 1]) / max(s[i + 1] - s[i - 1], 1e-12)
    V[n - 1] = (pts[n - 1] - pts[n - 2]) / max(s[n - 1] - s[n - 2], 1e-12)

    # 二阶导（加速度）
    A = np.zeros((n, 2))
    A[0] = (V[1] - V[0]) / max(s[1] - s[0], 1e-12)
    for i in range(1, n - 1):
        A[i] = (V[i + 1] - V[i - 1]) / max(s[i + 1] - s[i - 1], 1e-12)
    A[n - 1] = (V[n - 1] - V[n - 2]) / max(s[n - 1] - s[n - 2], 1e-12)

    dense = [pts[0].copy()]
    for i in range(n - 1):
        ds = s[i + 1] - s[i]
        if ds < 1e-12:
            continue
        k = max(1, int(np.ceil(ds / sp)))
        u = np.arange(1, k + 1, dtype=np.float64) / k
        h00, h10, h20, h01, h11, h21 = _quintic_hermite_basis(u)
        seg = (h00[:, None] * pts[i]
               + h10[:, None] * (V[i] * ds)
               + h20[:, None] * (A[i] * ds * ds)
               + h01[:, None] * pts[i + 1]
               + h11[:, None] * (V[i + 1] * ds)
               + h21[:, None] * (A[i + 1] * ds * ds))
        dense.append(seg)

    dense = np.vstack(dense)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense
