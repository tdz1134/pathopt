"""曲线缩短流 —— 沿法向按曲率演化 ∂P/∂t = κ·n，控制点固定。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import chord_param, resample_by_spacing


@registry.algorithm("curve_shortening")
def curve_shortening(pts: np.ndarray, start_tangent: float,
                     spacing: float = 0.1) -> np.ndarray:
    """几何曲线缩短流：每步将内部点沿局部法向移动 λ·(邻域中点-自身)。

    与拉普拉斯平滑的区别：只沿法向演化（切向分量仅重参数化），
    是纯粹的几何流，会真实缩短弧长并把尖角磨圆。
    """
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

    lam = 0.25
    for _ in range(200):
        # 拉普拉斯 ≈ κ·n（弧长参数下）
        lap = np.zeros_like(P)
        lap[1:-1] = 0.5 * (P[:-2] + P[2:]) - P[1:-1]
        # 局部切向与法向
        tan = np.zeros_like(P)
        tan[1:-1] = P[2:] - P[:-2]
        norm = np.linalg.norm(tan, axis=1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        tangent = tan / norm
        # 只保留法向分量
        normal_disp = lap - np.sum(lap * tangent, axis=1, keepdims=True) * tangent
        Pnew = P + lam * normal_disp
        Pnew[fixed] = P[fixed]
        P = Pnew

    P[0] = pts[0]
    P[-1] = pts[-1]
    return P
