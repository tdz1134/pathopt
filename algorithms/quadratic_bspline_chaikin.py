"""Chaikin 切角细分 —— 迭代切角收敛到二次 B 样条。"""
from __future__ import annotations

import numpy as np

from registry import registry
from ._common import resample_by_spacing


@registry.algorithm("quadratic_bspline_chaikin")
def quadratic_bspline_chaikin(pts: np.ndarray, start_tangent: float,
                              spacing: float = 0.1) -> np.ndarray:
    """Chaikin 切角：每轮取每条边的 1/4 与 3/4 点，极限曲线为二次 B 样条。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return resample_by_spacing(pts, spacing) if n >= 2 else pts.copy()

    poly = pts.copy()
    for _ in range(8):
        # 内部边切角，首末点保持固定（开曲线 Chaikin）
        q = [poly[0]]
        for i in range(len(poly) - 1):
            p0, p1 = poly[i], poly[i + 1]
            q.append(0.75 * p0 + 0.25 * p1)
            q.append(0.25 * p0 + 0.75 * p1)
        q.append(poly[-1])
        poly = np.array(q)
        # 细分足够密时提前停止
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        if seg.max() < max(spacing, 1e-6):
            break

    dense = resample_by_spacing(poly, spacing)
    dense[0] = pts[0]
    dense[-1] = pts[-1]
    return dense
