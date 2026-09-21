"""折线直连 baseline（不做任何平滑）。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import resample_by_spacing


@registry.algorithm("linear")
def linear(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """折线直连 —— 所有平滑算法的零点参照。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    return resample_by_spacing(pts, spacing)
