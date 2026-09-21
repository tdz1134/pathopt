"""Reinsch 平滑样条 —— 最小化 ∫f″² Subject to 残差约束（UnivariateSpline）。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

from registry import registry
from ._common import chord_param


@registry.algorithm("reinsch_smooth_spline")
def reinsch_smooth_spline(pts: np.ndarray, start_tangent: float,
                          spacing: float = 0.1) -> np.ndarray:
    """经典 Reinsch 平滑样条：在允许残差 s 下最小化二阶导能量。

    分别对 x(t)、y(t) 拟合，是平滑样条的理论标杆（EBSA/Reinsch 算法），
    s>0 时不严格过点，展示拟合-光顺权衡。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    t = chord_param(pts)
    if t[-1] < 1e-12:
        return pts.copy()

    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    t_u, pts_u = t[keep], pts[keep]
    m = len(pts_u)
    if m < 2:
        return pts.copy()

    k = min(3, m - 1)
    # 允许的残差总量：与弦长尺度、点数挂钩（>0 才产生平滑）
    s_tol = 1e-2 * (t_u[-1] ** 2) / max(m, 1)

    def fit(coord):
        try:
            sp = UnivariateSpline(t_u, coord, k=k, s=s_tol)
        except Exception:
            sp = UnivariateSpline(t_u, coord, k=k, s=0.0)
        return sp

    sx = fit(pts_u[:, 0])
    sy = fit(pts_u[:, 1])
    ts = np.linspace(t_u[0], t_u[-1], 300)
    xy = np.column_stack([sx(ts), sy(ts)])
    xy[0] = pts[0]
    xy[-1] = pts[-1]
    return xy
