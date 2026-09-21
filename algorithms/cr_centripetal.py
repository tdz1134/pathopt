"""向心参数化 Catmull-Rom α=0.5 (防自交)。"""
from __future__ import annotations

import numpy as np

from registry import registry


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
