"""PCHIP 单调插值 (保形, 无过冲)。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from registry import registry
from ._common import chord_param


@registry.algorithm("pchip")
def pchip(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """PCHIP 单调插值（保形，无过冲）。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    t = chord_param(pts)
    ts = np.linspace(t[0], t[-1], 300)
    it = PchipInterpolator(t, pts)
    xy = it(ts)
    return np.column_stack([xy[:, 0], xy[:, 1]])
