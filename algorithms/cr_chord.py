"""弦长参数化 Catmull-Rom (C1, 防末段曲率爆炸)。"""
from __future__ import annotations

import numpy as np

from registry import registry


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
