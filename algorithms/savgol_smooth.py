"""Savitzky-Golay 滤波平滑 —— 先等弧长重采样，再滑动窗口多项式拟合。"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from registry import registry
from ._common import resample_by_spacing


@registry.algorithm("savgol_smooth")
def savgol_smooth(pts: np.ndarray, start_tangent: float,
                  spacing: float = 0.1) -> np.ndarray:
    """Savitzky-Golay 平滑：局部低阶多项式最小二乘，保持峰形、实现极简。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    # savgol 需要等间距采样：先按 spacing 重采样
    dense = resample_by_spacing(pts, spacing, min_samples=9)
    m = len(dense)
    if m < 5:
        return dense

    win = min(m if m % 2 == 1 else m - 1, 11)  # 奇数窗口，最长 11
    poly = min(3, win - 1)
    smooth = np.column_stack([
        savgol_filter(dense[:, 0], win, poly, mode='nearest'),
        savgol_filter(dense[:, 1], win, poly, mode='nearest'),
    ])
    # 首末点锚定回原始端点
    smooth[0] = pts[0]
    smooth[-1] = pts[-1]
    return smooth
