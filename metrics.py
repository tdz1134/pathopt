"""
统一评估指标 —— 对密采样路径计算各种质量指标。

所有指标函数签名统一为:
    metric(dense: np.ndarray, control_pts: np.ndarray = None) -> float
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ─── 基础几何工具 ────────────────────────────────────────────────────────────

def _arc_lengths(dense: np.ndarray) -> np.ndarray:
    """计算密采样路径的累积弧长，shape (m,)。"""
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_by_arc(dense: np.ndarray, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """按弧长均匀重采样，返回 (resampled_pts, s_uniform)。"""
    s = _arc_lengths(dense)
    if s[-1] < 1e-9:
        return dense[:1], s[:1]
    s_new = np.linspace(0.0, s[-1], n)
    x = np.interp(s_new, s, dense[:, 0])
    y = np.interp(s_new, s, dense[:, 1])
    return np.column_stack([x, y]), s_new


def _chord_param(pts: np.ndarray) -> np.ndarray:
    """弦长累加参数化。"""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


# ─── 单项指标 ────────────────────────────────────────────────────────────────

def total_length(dense: np.ndarray, **kwargs) -> float:
    """路径总长度。"""
    return float(_arc_lengths(dense)[-1])


def path_smoothness(dense: np.ndarray, **kwargs) -> float:
    """路径平滑度 = 曲率变化的积分（越小越平滑）。

    ∫ |dκ/ds| ds，即曲率变化率的 L1 范数。
    """
    s, kappa = curvature_profile(dense)
    if len(s) < 3:
        return 0.0
    dk_ds = np.gradient(kappa, s)
    return float(np.trapz(np.abs(dk_ds), s))


def max_curvature(dense: np.ndarray, **kwargs) -> float:
    """最大 |曲率|。"""
    _, kappa = curvature_profile(dense)
    return float(np.max(np.abs(kappa))) if len(kappa) > 0 else 0.0


def curvature_variance(dense: np.ndarray, **kwargs) -> float:
    """曲率方差（衡量曲率分布均匀性）。"""
    _, kappa = curvature_profile(dense)
    return float(np.var(kappa)) if len(kappa) > 0 else 0.0


def max_deviation(dense: np.ndarray, control_pts: np.ndarray = None, **kwargs) -> float:
    """路径偏离控制点折线的最大距离。"""
    if control_pts is None or len(control_pts) < 2:
        return 0.0
    # 对每个密采样点，计算到最近折线段的距离
    from scipy.spatial import cKDTree
    tree = cKDTree(control_pts)
    # 近似：用控制点 KD-tree 最近距离
    dists, _ = tree.query(dense)
    return float(np.max(dists))


def curvature_profile(dense: np.ndarray, n_resample: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """计算曲率-弧长曲线。

    Returns
    -------
    s : (n_resample,) — 均匀弧长
    kappa : (n_resample,) — 带符号曲率
    """
    xy, s = _resample_by_arc(dense, n_resample)
    if len(s) < 3:
        return s, np.zeros_like(s)
    x, y = xy[:, 0], xy[:, 1]
    dx, dy = np.gradient(x, s), np.gradient(y, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    kappa = (dx * ddy - dy * ddx) / denom
    return s, kappa


def lateral_acceleration_cost(dense: np.ndarray, velocity: float = 1.0, **kwargs) -> float:
    """横向加速度代价 = ∫ (v²κ)² ds，近似衡量乘坐舒适性。"""
    s, kappa = curvature_profile(dense)
    if len(s) < 2:
        return 0.0
    lat_acc_sq = (velocity ** 2 * kappa) ** 2
    return float(np.trapz(lat_acc_sq, s))


# ─── 指标集合 ────────────────────────────────────────────────────────────────

# 所有可用指标的注册表
ALL_METRICS = {
    "length": total_length,
    "smoothness": path_smoothness,
    "max_curvature": max_curvature,
    "curvature_var": curvature_variance,
    "max_deviation": max_deviation,
    "lat_accel_cost": lateral_acceleration_cost,
}


def evaluate(dense: np.ndarray,
             control_pts: Optional[np.ndarray] = None,
             metric_names: Optional[list[str]] = None) -> Dict[str, float]:
    """计算指定指标集合。

    Parameters
    ----------
    dense : (m, 2) 密采样路径
    control_pts : (n, 2) 原始控制点（部分指标需要）
    metric_names : 要计算的指标名列表，None 表示全部

    Returns
    -------
    dict[str, float] — 指标名 → 值
    """
    if metric_names is None:
        metric_names = list(ALL_METRICS.keys())

    results = {}
    for name in metric_names:
        if name not in ALL_METRICS:
            raise KeyError(f"未知指标 '{name}'。可用: {list(ALL_METRICS.keys())}")
        fn = ALL_METRICS[name]
        results[name] = fn(dense, control_pts=control_pts)
    return results
