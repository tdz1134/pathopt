"""算法模块共享辅助函数（非算法，不注册）。"""
from __future__ import annotations

from math import comb

import numpy as np


def dedupe_consecutive(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """去除连续重复点 — 避免弦长参数非单调或样条节点除零。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], d > tol])
    return pts[keep]


def chord_param(pts: np.ndarray) -> np.ndarray:
    """弦长累加参数化。"""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_by_spacing(pts: np.ndarray, spacing: float,
                        min_samples: int = 10) -> np.ndarray:
    """沿折线按弧长等距重采样（首末点固定）。"""
    s = chord_param(pts)
    if s[-1] < 1e-12:
        return pts.copy()
    n = max(min_samples, int(np.ceil(s[-1] / max(spacing, 1e-6))) + 1)
    sf = np.linspace(s[0], s[-1], n)
    return np.column_stack([np.interp(sf, s, pts[:, 0]),
                            np.interp(sf, s, pts[:, 1])])


def min_derivative_path(pts: np.ndarray, order: int,
                        spacing: float | None = None,
                        min_samples: int = 120) -> np.ndarray:
    """过定点的最小导数能量路径（统一求解器）。

    在等参网格上以第 order 阶差分作为 x^(order) 的近似，
    固定控制点所在网格索引，对其余自由度解线性最小二乘：
    min ||D_k P||^2  s.t. P[fixed] = pts。order=2/3/4 分别
    对应 加速度/jerk/snap 能量最小（多项式轨迹的离散形式）。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    s = chord_param(pts)
    L = s[-1]
    if L < 1e-12:
        return pts.copy()
    if spacing is None or spacing <= 0:
        spacing = L / 240.0
    M = max(min_samples, int(np.ceil(L / spacing)) + 1)
    sd = np.linspace(s[0], s[-1], M)

    # 网格上的初始坐标（沿控制点折线插值）
    target = np.column_stack([np.interp(sd, s, pts[:, 0]),
                              np.interp(sd, s, pts[:, 1])])
    # 定位每个控制点对应的网格索引为“固定”约束点
    fixed = np.unique(np.clip(np.searchsorted(sd, s), 0, M - 1))
    for i in range(n):
        j = int(np.clip(np.searchsorted(sd, s[i]), 0, M - 1))
        target[j] = pts[i]
        fixed = np.unique(np.append(fixed, j))
    free = np.setdiff1d(np.arange(M), fixed)
    if len(free) == 0:
        return target

    k = min(order, M - 1)
    if k < 1:
        return target
    coef = np.array([(-1) ** (k - j) * comb(k, j) for j in range(k + 1)],
                    dtype=np.float64)
    D = np.zeros((M - k, M))
    for i in range(M - k):
        D[i, i:i + k + 1] = coef
    D_free = D[:, free]
    D_fix = D[:, fixed]

    out = np.zeros((M, 2))
    out[fixed] = target[fixed]
    for c in range(2):
        b = -D_fix @ target[fixed, c]
        sol, *_ = np.linalg.lstsq(D_free, b, rcond=None)
        out[free, c] = sol
    return out
