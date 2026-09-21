"""均匀三次 B 样条（控制点 = 数据点，光滑逼近，不严格过点）。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from registry import registry


@registry.algorithm("bspline_uniform_cubic")
def bspline_uniform_cubic(pts: np.ndarray, start_tangent: float,
                          spacing: float = 0.1) -> np.ndarray:
    """均匀 clamped B 样条：数据点直接作为控制点，曲线首末过点、中间逼近。

    节点向量 t = [0]^(k+1) + 1..(n-k-1) + [n-k]^(k+1)，k = min(3, n-1)。
    n < 4 时自动降阶（n=2 退化为直线，n=3 为二阶）。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()

    k = min(3, n - 1)
    # clamped 均匀节点向量：首末各 k+1 个重复节点，中间均匀内插
    internal = np.arange(1, n - k, dtype=np.float64)      # n-k-1 个内部节点
    t = np.concatenate([np.zeros(k + 1), internal, np.full(k + 1, float(n - k))])
    spline = BSpline(t, pts, k, extrapolate=False)
    u = np.linspace(0.0, float(n - k), 300)
    xy = np.asarray(spline(u))
    xy[0] = pts[0]
    xy[-1] = pts[-1]
    return xy
