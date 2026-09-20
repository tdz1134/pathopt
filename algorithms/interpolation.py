"""
插值类算法 —— 基于各种样条/插值方法生成密采样路径。

已注册算法清单:
  - cr_uniform      : 均匀参数化 Catmull-Rom 样条 (C1)
  - cr_chord         : 弦长参数化 Catmull-Rom (C1, 防末段曲率爆炸)
  - cr_centripetal   : 向心参数化 Catmull-Rom α=0.5 (防自交)
  - akima            : Akima 插值 (抗过冲, C1)
  - natural_cubic    : 自然三次样条 (C2 连续)
  - pchip            : PCHIP 单调插值 (保形, 无过冲)
  - clothoid         : 缓和曲线样条 (G2, 曲率沿弧长线性)
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

from registry import registry


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _chord_param(pts: np.ndarray) -> np.ndarray:
    """弦长累加参数化。"""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


# ─── Catmull-Rom 系列 ────────────────────────────────────────────────────────

@registry.algorithm("cr_uniform")
def cr_uniform(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """均匀参数化 Catmull-Rom 样条。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()

    sp = max(spacing, 1e-6)

    # 切向量
    T = np.zeros((n, 2))
    s0 = np.linalg.norm(pts[1] - pts[0])
    T[0] = s0 * np.array([np.cos(start_tangent), np.sin(start_tangent)])
    for i in range(1, n - 1):
        T[i] = 0.5 * (pts[i + 1] - pts[i - 1])
    T[n - 1] = pts[n - 1] - pts[n - 2]

    # 逐段三次 Hermite
    dense = [pts[0].copy()]
    for i in range(n - 1):
        P0, P1 = pts[i], pts[i + 1]
        M0, M1 = T[i], T[i + 1]
        seg_len = np.linalg.norm(P1 - P0)
        k = max(1, int(np.ceil(seg_len / sp)))
        for j in range(1, k + 1):
            t = j / k
            t2, t3 = t * t, t * t * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            dense.append(h00 * P0 + h10 * M0 + h01 * P1 + h11 * M1)

    dense = np.array(dense)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense


@registry.algorithm("cr_chord")
def cr_chord(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """弦长参数化 Catmull-Rom（非均匀）。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    sp = max(spacing, 1e-6)

    D = np.array([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(n - 1)])
    u = np.array([(pts[i + 1] - pts[i]) / max(D[i], 1e-9) for i in range(n - 1)])

    G = np.zeros((n, 2))
    G[0] = np.array([np.cos(start_tangent), np.sin(start_tangent)])
    for i in range(1, n - 1):
        G[i] = 0.5 * (u[i - 1] + u[i])
    G[n - 1] = u[n - 2]

    dense = [pts[0].copy()]
    for i in range(n - 1):
        P0, P1 = pts[i], pts[i + 1]
        M0, M1 = G[i] * D[i], G[i + 1] * D[i]
        k = max(1, int(np.ceil(D[i] / sp)))
        for j in range(1, k + 1):
            t = j / k
            t2, t3 = t * t, t * t * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            dense.append(h00 * P0 + h10 * M0 + h01 * P1 + h11 * M1)

    dense = np.array(dense)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense


@registry.algorithm("cr_centripetal")
def cr_centripetal(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """向心参数化 Catmull-Rom (alpha=0.5)。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    sp = max(spacing, 1e-6)
    alpha = 0.5

    dense = [pts[0].copy()]
    for i in range(n - 1):
        pm1 = pts[i - 1] if i > 0 else 2 * pts[0] - pts[1]
        p0 = pts[i]
        p1 = pts[i + 1]
        p2 = pts[i + 2] if i + 2 < n else 2 * pts[-1] - pts[-2]

        def knot_dist(a, b):
            return max(np.linalg.norm(b - a) ** alpha, 1e-6)

        t_prev = 0.0
        t_cur = t_prev + knot_dist(pm1, p0)
        t_next = t_cur + knot_dist(p0, p1)
        t_last = t_next + knot_dist(p1, p2)

        seg_len = np.linalg.norm(p1 - p0)
        k = max(1, int(np.ceil(seg_len / sp)))
        for j in range(1, k + 1):
            t = t_cur + (j / k) * (t_next - t_cur)
            A1 = (t_cur - t) / (t_cur - t_prev) * pm1 + (t - t_prev) / (t_cur - t_prev) * p0
            A2 = (t_next - t) / (t_next - t_cur) * p0 + (t - t_cur) / (t_next - t_cur) * p1
            A3 = (t_last - t) / (t_last - t_next) * p1 + (t - t_next) / (t_last - t_next) * p2
            B1 = (t_next - t) / (t_next - t_cur) * A1 + (t - t_cur) / (t_next - t_cur) * A2
            B2 = (t_last - t) / (t_last - t_next) * A2 + (t - t_next) / (t_last - t_next) * A3
            C = (t_next - t) / (t_next - t_cur) * B1 + (t - t_cur) / (t_next - t_cur) * B2
            dense.append(C)

    return np.array(dense)


# ─── SciPy 插值器 ────────────────────────────────────────────────────────────

@registry.algorithm("akima")
def akima(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """Akima 插值（抗过冲）。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    t = _chord_param(pts)
    ts = np.linspace(t[0], t[-1], 300)
    it = Akima1DInterpolator(t, pts)
    xy = it(ts)
    return np.column_stack([xy[:, 0], xy[:, 1]])


@registry.algorithm("natural_cubic")
def natural_cubic(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """自然三次样条 (C2 连续)。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    t = _chord_param(pts)
    ts = np.linspace(t[0], t[-1], 300)
    it = CubicSpline(t, pts, bc_type='natural')
    xy = it(ts)
    return np.column_stack([xy[:, 0], xy[:, 1]])


@registry.algorithm("pchip")
def pchip(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """PCHIP 单调插值（保形，无过冲）。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    t = _chord_param(pts)
    ts = np.linspace(t[0], t[-1], 300)
    it = PchipInterpolator(t, pts)
    xy = it(ts)
    return np.column_stack([xy[:, 0], xy[:, 1]])


# ─── Clothoid (缓和曲线) ─────────────────────────────────────────────────────

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
    s = _chord_param(pts)
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
