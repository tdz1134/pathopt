"""全局多项式拟合 —— 高阶龙格现象的典型反面教材。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import chord_param


@registry.algorithm("polynomial_global")
def polynomial_global(pts: np.ndarray, start_tangent: float,
                      spacing: float = 0.1) -> np.ndarray:
    """全局 5 次多项式最小二乘拟合 x(t)、y(t)。

    点少时勉强可用，点多/拐弯时会剧烈振荡甚至末端爆炸，
    用于对比说明"为什么路径平滑要用分段低阶而不是全局高阶"。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    t = chord_param(pts)
    if t[-1] < 1e-12:
        return pts.copy()

    # 归一化参数到 [-1, 1]，改善多项式拟合的数值条件
    tn = 2.0 * t / t[-1] - 1.0
    deg = min(5, n - 1)
    cx = np.polyfit(tn, pts[:, 0], deg)
    cy = np.polyfit(tn, pts[:, 1], deg)

    ts = np.linspace(-1.0, 1.0, 300)
    return np.column_stack([np.polyval(cx, ts), np.polyval(cy, ts)])
