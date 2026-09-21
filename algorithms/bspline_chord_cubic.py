"""非均匀（弦长）三次 B 样条插值 —— 反解控制点，严格过点。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from registry import registry
from ._common import chord_param


@registry.algorithm("bspline_chord_cubic")
def bspline_chord_cubic(pts: np.ndarray, start_tangent: float,
                        spacing: float = 0.1) -> np.ndarray:
    """弦长参数化三次 B 样条插值：求解控制点使曲线严格经过所有数据点。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    t = chord_param(pts)
    if t[-1] < 1e-12:
        return pts.copy()
    k = min(3, n - 1)
    spline = make_interp_spline(t, pts, k=k)
    ts = np.linspace(t[0], t[-1], 300)
    xy = np.asarray(spline(ts))
    xy[0] = pts[0]
    xy[-1] = pts[-1]
    return xy
