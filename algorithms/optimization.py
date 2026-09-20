"""
基于优化的路径算法 —— 通过求解优化问题生成平滑路径。

已注册算法清单:
  - min_curvature_energy : 最小化曲率平方积分 ∫ κ² ds (L-BFGS-B)
  - elastic_band         : 弹性带方法 = 路径长度 + 曲率能量 + 偏离惩罚
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from registry import registry


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _chord_param(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


# ─── 最小化曲率能量 ──────────────────────────────────────────────────────────

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

    from scipy.interpolate import CubicSpline

    # 优化变量：中间点的 (dx, dy) 偏移，首末点固定
    opt_pts = pts.copy()
    x0 = np.zeros((n - 2) * 2)

    def objective(offsets):
        p = pts.copy()
        p[1:-1] += offsets.reshape(-1, 2)
        s = _chord_param(p)
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
    s = _chord_param(opt_pts)
    cs = CubicSpline(s, opt_pts, bc_type='natural')
    sf = np.linspace(s[0], s[-1], 300)
    xy = cs(sf)
    return np.column_stack([xy[:, 0], xy[:, 1]])


# ─── 弹性带 (Elastic Band) ──────────────────────────────────────────────────

@registry.algorithm("elastic_band")
def elastic_band(pts: np.ndarray, start_tangent: float,
                 spacing: float = 0.1) -> np.ndarray:
    """弹性带方法 —— 最小化路径长度 + 曲率惩罚。

    优化变量为路径上均匀分布的 N 个中间点坐标。
    代价函数 = w_len * 总长度 + w_curv * 曲率能量 + w_dev * 偏离原始点惩罚。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    from scipy.interpolate import CubicSpline

    # 在原始折线上均匀插出 N 个内部优化点
    N_internal = min(20, max(n, 10))
    s_orig = _chord_param(pts)
    s_internal = np.linspace(s_orig[0] + 0.01 * s_orig[-1],
                             s_orig[-1] - 0.01 * s_orig[-1], N_internal)
    # 初值：折线插值
    x0_x = np.interp(s_internal, s_orig, pts[:, 0])
    x0_y = np.interp(s_internal, s_orig, pts[:, 1])
    x0 = np.column_stack([x0_x, x0_y]).ravel()

    # 原始点在细密弧长上的投影（用于偏离惩罚）
    s_fine = np.linspace(s_orig[0], s_orig[-1], 200)
    x_orig_fine = np.interp(s_fine, s_orig, pts[:, 0])
    y_orig_fine = np.interp(s_fine, s_orig, pts[:, 1])

    w_len = 1.0
    w_curv = 0.5
    w_dev = 5.0

    def objective(offsets):
        mid = x0 + offsets
        mid_pts = mid.reshape(-1, 2)
        # 拼接首末点
        all_pts = np.vstack([pts[0:1], mid_pts, pts[-1:]])
        s = _chord_param(all_pts)
        if s[-1] < 1e-9:
            return 1e10
        # 总长度
        length = s[-1]
        # 曲率能量
        cs = CubicSpline(s, all_pts, bc_type='natural')
        sf = np.linspace(s[0], s[-1], 200)
        d1 = cs.derivative(1)(sf)
        d2 = cs.derivative(2)(sf)
        dx, dy = d1[:, 0], d1[:, 1]
        ddx, ddy = d2[:, 0], d2[:, 1]
        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom = np.where(denom < 1e-12, 1e-12, denom)
        kappa = (dx * ddy - dy * ddx) / denom
        curv_energy = np.trapz(kappa ** 2, sf)
        # 偏离惩罚
        xy_fine = cs(sf)
        dev = np.sum((xy_fine[:, 0] - x_orig_fine) ** 2 +
                     (xy_fine[:, 1] - y_orig_fine) ** 2) / len(sf)

        return w_len * length + w_curv * curv_energy + w_dev * dev

    offsets0 = np.zeros_like(x0)
    sol = minimize(objective, offsets0, method='L-BFGS-B',
                   options={'maxiter': 300, 'ftol': 1e-8})

    mid_final = (x0 + sol.x).reshape(-1, 2)
    all_pts = np.vstack([pts[0:1], mid_final, pts[-1:]])
    s = _chord_param(all_pts)
    cs = CubicSpline(s, all_pts, bc_type='natural')
    sf = np.linspace(s[0], s[-1], 300)
    xy = cs(sf)
    return np.column_stack([xy[:, 0], xy[:, 1]])
