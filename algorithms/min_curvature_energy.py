"""最小化曲率平方积分 ∫ κ² ds (L-BFGS-B)。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

from registry import registry
from ._common import chord_param


@registry.algorithm("min_curvature_energy")
def min_curvature_energy(pts: np.ndarray, start_tangent: float,
                         spacing: float = 0.1) -> np.ndarray:
    """最小化曲率平方积分 ∫κ² ds。

    方法：在每个中间控制点处添加微小偏移量作为优化变量，
    使得最终样条曲率能量最小。
    这是一个简化的演示版本 —— 实际可以做得更精细。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    # 优化变量：中间点的 (dx, dy) 偏移，首末点固定
    opt_pts = pts.copy()
    x0 = np.zeros((n - 2) * 2)

    def objective(offsets):
        p = pts.copy()
        p[1:-1] += offsets.reshape(-1, 2)
        s = chord_param(p)
        if s[-1] < 1e-9:
            return 1e10
        cs = CubicSpline(s, p, bc_type='natural')
        # 在细密采样上计算曲率能量
        sf = np.linspace(s[0], s[-1], 300)
        d1 = cs.derivative(1)(sf)
        d2 = cs.derivative(2)(sf)
        dx, dy = d1[:, 0], d1[:, 1]
        ddx, ddy = d2[:, 0], d2[:, 1]
        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom = np.where(denom < 1e-12, 1e-12, denom)
        kappa = (dx * ddy - dy * ddx) / denom
        # ∫κ² ds
        energy = np.trapz(kappa ** 2, sf)
        # 偏离惩罚：不让优化后点偏离太远
        deviation = np.sum((p - pts) ** 2)
        return energy + 10.0 * deviation

    sol = minimize(objective, x0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-8})

    opt_pts = pts.copy()
    opt_pts[1:-1] += sol.x.reshape(-1, 2)

    # 用优化后的点生成三次样条密采样
    s = chord_param(opt_pts)
    cs = CubicSpline(s, opt_pts, bc_type='natural')
    sf = np.linspace(s[0], s[-1], 300)
    xy = cs(sf)
    return np.column_stack([xy[:, 0], xy[:, 1]])
