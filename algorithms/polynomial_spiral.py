"""多项式螺旋 —— κ(s) 为分段三次多项式（Apollo 路径平滑同款思路）。

clothoid 是 κ 随弧长线性变化的特例；这里 κ(s) 在每个控制点区间
用三次 Hermite 表示（节点处 κ 与 dκ/ds 均连续 → 路径 G3 级光滑），
通过 least_squares 拟合过控制点，并正则化 dκ/ds 保持平缓。

性能：拟合网格降至 150 点，收敛容差放宽（1e-10 / max_nfev 600），
最慢场景从 ~1.4s 降至 ~0.5s；弧长倍率保持冷启动（热启动会
收敛到更差的局部极小，见下方注释）。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

from registry import registry
from ._common import chord_param, dedupe_consecutive


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
    pts = dedupe_consecutive(np.asarray(pts, dtype=np.float64))
    n = len(pts)
    if n < 3:
        # 与 clothoid 一致：少于 3 点退回折线
        return pts.copy()
    s0 = chord_param(pts)
    if s0[-1] < 1e-12:
        return pts.copy()
    ds0 = np.diff(s0)

    # 初值：自然三次样条在节点处的曲率及其弧长导数
    cs = CubicSpline(s0, pts, bc_type='natural')
    d1 = cs.derivative(1)(s0)
    d2 = cs.derivative(2)(s0)
    dx, dy = d1[:, 0], d1[:, 1]
    ddx, ddy = d2[:, 0], d2[:, 1]
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    k0 = (dx * ddy - dy * ddx) / denom
    kd0 = np.gradient(k0, s0)

    mean_ds = float(np.mean(ds0))
    w_reg = 0.1 * mean_ds       # 抑制 dκ/ds，保持平缓
    w_len = 0.2 * mean_ds       # 弧长倍率贴近弦长
    w_end = 10.0                # 末点加权（终点必须到达优先于中段拟合）
    n_fine = 150                # 拟合网格：精度与速度的折中，输出另行加密
    # 注：弧长倍率 lr 从 0（弦长）冷启动；实验表明用样条弧长热启动
    # 会收敛到更差的局部极小（sharp_corner 长度膨胀一倍），故不采用。

    def build(k, kd, r):
        s = np.concatenate([[0.0], np.cumsum(ds0 * r)])
        sf = np.linspace(0.0, s[-1], n_fine + 1)
        xy = _spiral_integrate(s, k, kd, sf, pts[0], start_tangent)
        # 线性插值光滑采样，保持残差对 r 可微（见 clothoid 同款说明）
        xy_at = np.column_stack([np.interp(s, sf, xy[:, 0]),
                                 np.interp(s, sf, xy[:, 1])])
        return xy_at

    def resid(z):
        k, kd, lr = z[:n], z[n:2 * n], z[2 * n:]
        e = (build(k, kd, np.exp(lr))[1:] - pts[1:]).ravel()
        e[-2:] *= w_end
        return np.concatenate([e, w_reg * kd, w_len * lr])

    z0 = np.concatenate([k0, kd0, np.zeros(n - 1)])
    sol = least_squares(resid, z0, method='trf', max_nfev=600,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    k, kd, lr = sol.x[:n], sol.x[n:2 * n], sol.x[2 * n:]

    s_final = np.concatenate([[0.0], np.cumsum(ds0 * np.exp(lr))])
    n_out = max(int(s_final[-1] / max(spacing, 1e-6)) + 1, 10)
    sf = np.linspace(0.0, s_final[-1], n_out)
    return _spiral_integrate(s_final, k, kd, sf, pts[0], start_tangent)
