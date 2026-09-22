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
    # 首末点重复加权：最小二乘中等效权重 ×(reps+1)，把路径端点锚住
    reps = 5
    rng = t_u[-1] - t_u[0]
    head_t = t_u[0] - rng * np.linspace(5e-4, 1e-4, reps)
    tail_t = t_u[-1] + rng * np.linspace(1e-4, 5e-4, reps)
    data_t = np.concatenate([head_t, t_u, tail_t])
    data = np.vstack([np.tile(pts_u[0], (reps, 1)), pts_u,
                      np.tile(pts_u[-1], (reps, 1))])
    try:
        tck, _ = splprep([data[:, 0], data[:, 1]], u=data_t, k=k, s=s)
    except Exception:
        tck, _ = splprep([data[:, 0], data[:, 1]], u=data_t, k=k, s=0.0)
    uv = np.linspace(0.0, 1.0, 300)
    x, y = splev(uv, tck)
    out = np.column_stack([x, y])
    # 末点硬性锚定（消除残余的微小参数偏移）
    out[0] = pts_u[0]
    out[-1] = pts_u[-1]
    return out
