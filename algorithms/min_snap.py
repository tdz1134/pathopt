"""最小 snap 轨迹 —— 过定点、最小化 ∫|d⁴P/dt⁴|² dt（四旋翼无人机经典）。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import min_derivative_path


@registry.algorithm("min_snap")
def min_snap(pts: np.ndarray, start_tangent: float,
             spacing: float = 0.1) -> np.ndarray:
    """最小化 snap（jerk 的导数）平方积分，四阶差分能量最小。

    对应四旋翼推力变化率最小，是 Richter/Rückemann 的 minimum-snap
    轨迹生成的离散等价形式。
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    return min_derivative_path(pts, order=4, spacing=spacing)
