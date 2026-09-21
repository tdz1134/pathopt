"""最小 jerk 轨迹 —— 过定点、最小化 ∫|d³P/dt³|² dt（机械臂/人臂运动经典）。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import min_derivative_path


@registry.algorithm("min_jerk")
def min_jerk(pts: np.ndarray, start_tangent: float,
             spacing: float = 0.1) -> np.ndarray:
    """最小化加加速度(jerk)平方积分，三阶差分能量最小。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    return min_derivative_path(pts, order=3, spacing=spacing)
