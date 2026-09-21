"""多项式螺旋 —— κ(s) 为分段三次多项式（Apollo 路径平滑同款思路）。

clothoid 是 κ 随弧长线性变化的特例；这里 κ(s) 在每个控制点区间
用三次 Hermite 表示（节点处 κ 与 dκ/ds 均连续 → 路径 G3 级光滑），
通过 least_squares 拟合过控制点，并正则化 dκ/ds 保持平缓。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

from registry import registry
from ._common import chord_param


def _spiral_integrate(s_knot, k_knot, kd_knot, s_fine, p0, theta0):
    """给定节点曲率及其导数，积分出 xy。"""
    kap = CubicHermiteSpline(s_knot, k_knot, kd_knot)(s_fine)
    theta = theta0 + cumulative_trapezoid(kap, s_fine, initial=0.0)
    x = p0[0] + cumulative_trapezoid(np.cos(theta), s_fine, initial=0.0)
    y = p0[1] + cumulative_trapezoid(np.sin(theta), s_fine, initial=0.0)
    return np.column_stack([x, y])


@registry.algorithm("polynomial_spiral")
def polynomial_spiral(pts: np.ndarray, start_tangent: float,
                      spacing: float = 0.1) -> np.ndarray:
    """多项式螺旋样条：曲率为弧长的分段三次多项式，G3 连续。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        # 与 clothoid 一致：少于 3 点退回折线
        return pts.copy()
    s = chord_param(pts)
    if s[-1] < 1e-12:
        return pts.copy()

    # 初值：自然三次样条在节点处的曲率及其弧长导数
    cs = CubicSpline(s, pts, bc_type='natural')
    d1 = cs.derivative(1)(s)
    d2 = cs.derivative(2)(s)
    dx, dy = d1[:, 0], d1[:, 1]
    ddx, ddy = d2[:, 0], d2[:, 1]
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    k0 = (dx * ddy - dy * ddx) / denom
    kd0 = np.gradient(k0, s)

    # 正则化权重：与弧长尺度挂钩，抑制 dκ/ds 过大
    w_reg = 0.1 * np.mean(np.diff(s))

    def resid(x):
        k, kd = x[:n], x[n:]
        xy = _spiral_integrate(s, k, kd, s, pts[0], start_tangent)
        pos_err = (xy[1:] - pts[1:]).ravel()
        return np.concatenate([pos_err, w_reg * kd])

    sol = least_squares(resid, np.concatenate([k0, kd0]), method='trf')
    k, kd = sol.x[:n], sol.x[n:]

    sf = np.linspace(s[0], s[-1], 600)
    return _spiral_integrate(s, k, kd, sf, pts[0], start_tangent)
