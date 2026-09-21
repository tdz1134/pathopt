"""B 样条最小二乘逼近（splprep 带平滑因子）—— 展示平滑度与偏离度的权衡。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splprep, splev

from registry import registry
from ._common import chord_param


@registry.algorithm("bspline_approx")
def bspline_approx(pts: np.ndarray, start_tangent: float,
                   spacing: float = 0.1) -> np.ndarray:
    """参数化 B 样条最小二乘逼近：不强制过点，用平滑因子 s 控制拟合松紧。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    t = chord_param(pts)
    if t[-1] < 1e-12:
        return pts.copy()
    if n < 4:
        # splprep k=3 需要至少 4 个点，退化为插值
        k = n - 1
    else:
        k = 3

    # 去掉重复点（splrep 要求参数严格递增）
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    pts_u, t_u = pts[keep], t[keep]
    if len(pts_u) < k + 1:
        return pts.copy()

    # 平滑因子与点数、弦长尺度挂钩
    s = 1e-3 * len(pts_u) * (t_u[-1] ** 2)
    try:
        tck, _ = splprep([pts_u[:, 0], pts_u[:, 1]], k=k, s=s)
    except Exception:
        tck, _ = splprep([pts_u[:, 0], pts_u[:, 1]], k=k, s=0.0)
    uv = np.linspace(0.0, 1.0, 300)
    x, y = splev(uv, tck)
    return np.column_stack([x, y])
