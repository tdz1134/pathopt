"""离散弯曲能最小化 —— 弧长加权二阶差分 ∫κ²ds 的 IRLS 凸求解，控制点处硬过点。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import chord_param, resample_by_spacing


@registry.algorithm("discrete_bending_qp")
def discrete_bending_qp(pts: np.ndarray, start_tangent: float,
                        spacing: float = 0.1) -> np.ndarray:
    """以密采样点为变量最小化几何弯曲能 ∫κ²ds。

    离散形式为 Σ|Δ²P|²/ds³（κ≈|Δ²P|/ds²，弧长元 ds）。由于 ds 依赖当前
    几何，用 IRLS：每轮用上一步的 ds 固定权重、对自由点解线性最小二乘。
    每步皆凸，路径不会膨胀发散。控制点所在网格索引作为硬约束保持过点。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()
    s = chord_param(pts)
    L = s[-1]
    if L < 1e-12:
        return pts.copy()

    sp = max(spacing, L / 120.0)
    P0 = resample_by_spacing(pts, sp, min_samples=24)
    M = len(P0)
    if M < 5:
        return P0

    sd = chord_param(P0)
    fixed = set(np.unique(np.clip(np.searchsorted(sd, s), 0, M - 1)).tolist())
    free = [j for j in range(M) if j not in fixed]
    if len(free) == 0:
        return P0
    free_col = {j: c for c, j in enumerate(free)}

    P = P0.copy()
    rows = M - 2
    for _ in range(8):
        dseg = np.linalg.norm(np.diff(P, axis=0), axis=1)          # M-1
        ds = 0.5 * (dseg[:-1] + dseg[1:])                          # M-2 (节点 1..M-2)
        w = 1.0 / np.maximum(ds, 1e-6) ** 1.5

        A = np.zeros((rows, len(free)))
        b = np.zeros((rows, 2))
        for r in range(rows):
            node = r + 1
            for off, cf in ((-1, 1.0), (0, -2.0), (1, 1.0)):
                j = node + off
                wr = w[r] * cf
                if j in free_col:
                    A[r, free_col[j]] += wr
                else:
                    b[r] += -wr * P[j]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        Pnew = P.copy()
        Pnew[free] = sol.reshape(-1, 2)
        for j in fixed:
            Pnew[j] = P0[j]
        P = Pnew

    P[0] = pts[0]
    P[-1] = pts[-1]
    return P
