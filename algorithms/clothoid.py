"""缓和曲线样条 (G2, 曲率沿弧长线性)。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

from registry import registry
from ._common import chord_param


def _clothoid_integrate(s_knot, k_knot, s_fine, p0, theta0):
    """给定节点曲率，积分出 xy 与航向。"""
    kap = np.interp(s_fine, s_knot, k_knot)
    theta = theta0 + cumulative_trapezoid(kap, s_fine, initial=0.0)
    x = p0[0] + cumulative_trapezoid(np.cos(theta), s_fine, initial=0.0)
    y = p0[1] + cumulative_trapezoid(np.sin(theta), s_fine, initial=0.0)
    return np.column_stack([x, y]), theta


@registry.algorithm("clothoid")
def clothoid(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """Clothoid 缓和曲线样条 (G2 连续)，曲率沿弧长线性。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        # Clothoid 至少需要 3 个点才有意义，否则退回直线
        return pts.copy()
    n_fine = 600
    s = chord_param(pts)
    sf = np.linspace(s[0], s[-1], n_fine)

    # 初值：自然三次样条曲率
    cs = CubicSpline(s, pts, bc_type='natural')
    d1 = cs.derivative(1)(s)
    d2 = cs.derivative(2)(s)
    dx, dy = d1[:, 0], d1[:, 1]
    ddx, ddy = d2[:, 0], d2[:, 1]
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    k0 = (dx * ddy - dy * ddx) / denom

    def resid(kk):
        xy, _ = _clothoid_integrate(s, kk, s, pts[0], start_tangent)
        return (xy[1:] - pts[1:]).ravel()

    sol = least_squares(resid, k0, method='lm')
    xy, _ = _clothoid_integrate(s, sol.x, sf, pts[0], start_tangent)
    return xy
