"""自然三次样条 (C2 连续)。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from registry import registry
from ._common import chord_param


@registry.algorithm("natural_cubic")
def natural_cubic(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """自然三次样条 (C2 连续)。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    t = chord_param(pts)
    ts = np.linspace(t[0], t[-1], 300)
    it = CubicSpline(t, pts, bc_type='natural')
    xy = it(ts)
    return np.column_stack([xy[:, 0], xy[:, 1]])
