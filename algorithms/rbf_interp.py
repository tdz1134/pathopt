"""径向基函数（RBF）插值 —— thin_plate_spline 核，对噪声点天然光滑。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from registry import registry
from ._common import chord_param


@registry.algorithm("rbf_interp")
def rbf_interp(pts: np.ndarray, start_tangent: float,
               spacing: float = 0.1) -> np.ndarray:
    """弦长参数化 RBF 插值 x(t)、y(t)，薄板样条核。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    t = chord_param(pts)
    if t[-1] < 1e-12:
        return pts.copy()

    # 去掉重复参数点（RBF 要求节点互异）
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    t_u, pts_u = t[keep], pts[keep]
    if len(pts_u) < 2:
        return pts.copy()

    rbf = RBFInterpolator(t_u[:, None], pts_u, kernel='thin_plate_spline',
                          smoothing=1e-4)
    ts = np.linspace(t_u[0], t_u[-1], 300)
    xy = rbf(ts[:, None])
    xy[0] = pts[0]
    xy[-1] = pts[-1]
    return np.asarray(xy, dtype=np.float64)
