"""拉普拉斯平滑 —— 迭代 p_i ← (1-λ)p_i + λ·(邻域均值)，控制点固定。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import chord_param, resample_by_spacing


@registry.algorithm("laplacian_smooth")
def laplacian_smooth(pts: np.ndarray, start_tangent: float,
                     spacing: float = 0.1) -> np.ndarray:
    """拉普拉斯平滑 baseline：每步向邻接点平均收缩，实现极简。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()
    P = resample_by_spacing(pts, max(spacing, 1e-3), min_samples=40)
    M = len(P)
    if M < 4:
        return P

    s = chord_param(pts)
    sd = chord_param(P)
    fixed = np.zeros(M, dtype=bool)
    fixed[np.unique(np.clip(np.searchsorted(sd, s), 0, M - 1))] = True

    lam = 0.5
    for _ in range(100):
        Pnew = P.copy()
        Pnew[1:-1] = (1 - lam) * P[1:-1] + lam * 0.5 * (P[:-2] + P[2:])
        Pnew[fixed] = P[fixed]
        P = Pnew

    P[0] = pts[0]
    P[-1] = pts[-1]
    return P
