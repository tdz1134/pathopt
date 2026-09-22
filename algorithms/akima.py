"""Akima 插值 (抗过冲, C1)。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from registry import registry
from ._common import chord_param, dedupe_consecutive


@registry.algorithm("akima")
def akima(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """Akima 插值（抗过冲）。"""
    pts = dedupe_consecutive(np.asarray(pts, dtype=np.float64))
    if len(pts) < 2:
        return pts.copy()
    t = chord_param(pts)
    ts = np.linspace(t[0], t[-1], 300)
    it = Akima1DInterpolator(t, pts)
    xy = it(ts)
    return np.column_stack([xy[:, 0], xy[:, 1]])
