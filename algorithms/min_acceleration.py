"""最小加速度能量 —— 过定点、最小化 ∫|d²P/dt²|² dt（离散多项式轨迹）。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import min_derivative_path


@registry.algorithm("min_acceleration")
def min_acceleration(pts: np.ndarray, start_tangent: float,
                     spacing: float = 0.1) -> np.ndarray:
    """最小化加速度平方积分，对应二阶差分能量最小（与 min_jerk/snap 成阶次序列）。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    return min_derivative_path(pts, order=2, spacing=spacing)
