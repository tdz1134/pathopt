"""分段三次贝塞尔链 —— Catmull-Rom 切向的 Bernstein 基表示，严格过点。"""
from __future__ import annotations

import numpy as np

from registry import registry


def _cubic_bezier(P0, P1, P2, P3, t):
    """三次贝塞尔求值（Bernstein 基）。"""
    s = 1.0 - t
    return (s ** 3)[:, None] * P0 + (3 * s ** 2 * t)[:, None] * P1 \
        + (3 * s * t ** 2)[:, None] * P2 + (t ** 3)[:, None] * P3


@registry.algorithm("bezier_chain")
def bezier_chain(pts: np.ndarray, start_tangent: float, spacing: float = 0.1) -> np.ndarray:
    """分段三次贝塞尔链：每段控制点由 Catmull-Rom 切向确定（P1 = P0 + T/3）。

    与 cr_uniform 数学等价，但以 Bernstein 基 / 控制点形式表示，
    便于对比贝塞尔表示与 Hermite 表示的实现差异。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    sp = max(spacing, 1e-6)

    # Catmull-Rom 切向量
    T = np.zeros((n, 2))
    s0 = np.linalg.norm(pts[1] - pts[0])
    T[0] = s0 * np.array([np.cos(start_tangent), np.sin(start_tangent)])
    for i in range(1, n - 1):
        T[i] = 0.5 * (pts[i + 1] - pts[i - 1])
    T[n - 1] = pts[n - 1] - pts[n - 2]

    dense = [pts[0].copy()]
    for i in range(n - 1):
        P0, P3 = pts[i], pts[i + 1]
        P1 = P0 + T[i] / 3.0
        P2 = P3 - T[i + 1] / 3.0
        seg_len = np.linalg.norm(P3 - P0)
        k = max(1, int(np.ceil(seg_len / sp)))
        t = np.arange(1, k + 1, dtype=np.float64) / k
        dense.append(_cubic_bezier(P0, P1, P2, P3, t))

    dense = np.vstack(dense)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense
