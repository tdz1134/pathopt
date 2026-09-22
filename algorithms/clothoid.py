"""缓和曲线样条 (G2, 曲率沿弧长线性)。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

from registry import registry
from ._common import chord_param, dedupe_consecutive


def _clothoid_integrate(s_knot, k_knot, s_fine, p0, theta0):
    """给定节点曲率，积分出 xy 与航向。"""
    kap = np.interp(s_fine, s_knot, k_knot)
    theta = theta0 + cumulative_trapezoid(kap, s_fine, initial=0.0)
    x = p0[0] + cumulative_trapezoid(np.cos(theta), s_fine, initial=0.0)
    y = p0[1] + cumulative_trapezoid(np.sin(theta), s_fine, initial=0.0)
    return np.column_stack([x, y]), theta


@registry.algorithm("clothoid")
def clothoid(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """Clothoid 缓和曲线样条 (G2 连续)，曲率沿弧长线性。

    拟合参数 = 节点曲率 k (n) + 每段弧长倍率 r = exp(lr) (n-1)：
    节点弧长不强制等于弦长，否则平滑曲线的真实弧长与折线弦长和
    失配，末点会漂移数个单位。位置残差在细网格上积分评估，
    末点残差加权以优先保证终点到达。

    性能：拟合网格降至 150 点 + 弧长倍率热启动 + 收敛容差放宽
    （1e-10 / max_nfev 600），最慢场景长尾从 ~1.3s 降至 ~0.4s。
    """
    pts = dedupe_consecutive(np.asarray(pts, dtype=np.float64))
    n = len(pts)
    if n < 3:
        # Clothoid 至少需要 3 个点才有意义，否则退回直线
        return pts.copy()
    s0 = chord_param(pts)
    if s0[-1] < 1e-12:
        return pts.copy()
    ds0 = np.diff(s0)

    # 初值：自然三次样条曲率
    cs = CubicSpline(s0, pts, bc_type='natural')
    d1 = cs.derivative(1)(s0)
    d2 = cs.derivative(2)(s0)
    dx, dy = d1[:, 0], d1[:, 1]
    ddx, ddy = d2[:, 0], d2[:, 1]
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    k0 = (dx * ddy - dy * ddx) / denom

    mean_ds = float(np.mean(ds0))
    w_reg = 0.02 * mean_ds      # 抑制曲率绝对值（帮助收敛）
    w_len = 0.2 * mean_ds       # 弧长倍率贴近弦长
    w_end = 10.0                # 末点加权（终点必须到达优先于中段拟合）
    n_fine = 150                # 拟合网格：精度与速度的折中，输出另行加密

    # 弧长倍率热启动：用初始样条各段真实弧长估计 r0，减少 trf 迭代次数
    sl = np.linspace(s0[0], s0[-1], 101)
    pl = cs(sl)
    cl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))])
    idxk = np.clip(np.rint(np.interp(s0, sl, np.arange(101.0))).astype(int), 0, 100)
    L_seg = cl[idxk[1:]] - cl[idxk[:-1]]
    r0 = np.clip(L_seg / ds0, 0.5, 1.4)
    lr0 = np.log(r0)

    def build(k, r):
        s = np.concatenate([[0.0], np.cumsum(ds0 * r)])
        sf = np.linspace(0.0, s[-1], n_fine + 1)
        xy, _ = _clothoid_integrate(s, k, sf, pts[0], start_tangent)
        # 用线性插值在节点弧长处光滑采样（保持残差对 r 可微，
        # 若用取整最近点会使 Jacobian 不连续、trf 提前停止）
        return np.column_stack([np.interp(s, sf, xy[:, 0]),
                                np.interp(s, sf, xy[:, 1])])

    def resid(z):
        k, lr = z[:n], z[n:]
        xy_at = build(k, np.exp(lr))
        e = (xy_at[1:] - pts[1:]).ravel()
        e[-2:] *= w_end
        return np.concatenate([e, w_reg * k, w_len * lr])

    z0 = np.concatenate([k0, lr0])
    sol = least_squares(resid, z0, method='trf', max_nfev=600,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    k, lr = sol.x[:n], sol.x[n:]

    s_final = np.concatenate([[0.0], np.cumsum(ds0 * np.exp(lr))])
    n_out = max(int(s_final[-1] / max(spacing, 1e-6)) + 1, 10)
    sf_out = np.linspace(0.0, s_final[-1], n_out)
    xy, _ = _clothoid_integrate(s_final, k, sf_out, pts[0], start_tangent)
    return xy
