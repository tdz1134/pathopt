"""分段三次贝塞尔最小二乘拟合 —— 角点检测分段后端点固定、控制点最小二乘求解。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import chord_param

# 转角超过该阈值（度）视为角点，在角点处分段
_CORNER_ANGLE_DEG = 60.0


def _cubic_bezier(P0, P1, P2, P3, t):
    s = 1.0 - t
    return (s ** 3)[:, None] * P0 + (3 * s ** 2 * t)[:, None] * P1 \
        + (3 * s * t ** 2)[:, None] * P2 + (t ** 3)[:, None] * P3


@registry.algorithm("bezier_ls_fit")
def bezier_ls_fit(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """分段贝塞尔最小二乘拟合：不强制过中间点，展示"拟合"与"插值"的差异。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    sp = max(spacing, 1e-6)

    # 角点检测：相邻边方向夹角超过阈值处切分
    corners = [0]
    thresh = np.deg2rad(_CORNER_ANGLE_DEG)
    for i in range(1, n - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        if np.arccos(cosang) > thresh:
            corners.append(i)
    corners.append(n - 1)

    dense = [pts[0:1]]
    for a, b in zip(corners[:-1], corners[1:]):
        grp = pts[a:b + 1]
        m = len(grp)
        seg_len = np.linalg.norm(grp[-1] - grp[0])
        k = max(1, int(np.ceil(chord_param(grp)[-1] / sp)))
        if m <= 2 or seg_len < 1e-12:
            # 直线段
            t = np.linspace(0.0, 1.0, k + 1)[1:]
            seg = grp[0] + t[:, None] * (grp[-1] - grp[0])
        else:
            # 端点固定为 P0/P3，最小二乘求解 P1/P2
            P0, P3 = grp[0], grp[-1]
            t = chord_param(grp)
            t = (t - t[0]) / (t[-1] - t[0])
            # 基函数系数：B(t) - (1-t)³P0 - t³P3 = 3(1-t)²t·P1 + 3(1-t)t²·P2
            A1 = 3 * (1 - t) ** 2 * t
            A2 = 3 * (1 - t) * t ** 2
            A = np.column_stack([A1, A2])
            rhs = grp - np.column_stack([(1 - t) ** 3, t ** 3]) @ np.vstack([P0, P3])
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            P1, P2 = sol[0], sol[1]
            ts = np.linspace(0.0, 1.0, k + 1)[1:]
            seg = _cubic_bezier(P0, P1, P2, P3, ts)
        dense.append(seg)

    dense = np.vstack(dense)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense
