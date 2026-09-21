"""均匀参数化 Catmull-Rom 样条 (C1)。"""
from __future__ import annotations

import numpy as np

from registry import registry


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
