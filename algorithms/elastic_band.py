"""弹性带方法 = 弧长 + 弯曲能量 + 偏离惩罚（B 样条控制网 + 解析梯度）。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline

from registry import registry
from ._common import chord_param


@registry.algorithm("elastic_band")
def elastic_band(pts: np.ndarray, start_tangent: float,
                 spacing: float = 0.1) -> np.ndarray:
    """弹性带 —— 最小化 长度 + 弯曲能量 + 偏离原始折线 的组合代价。

    优化变量取为三次 B 样条的中间控制点（首末控制点锚定路径端点），
    输出曲线即被优化对象本身，天然 C2 光滑。三项能量均有闭式梯度：
      长度  ≈ 控制多边形周长 × 节点间距
      弯曲  = Σ|Δ²P|² / ℓ³      （∫κ²ds 的控制网代理，标准做法）
      偏离  = mean|Basis·P − Q|²（Q 为原折线在 Greville 参数处的采样）
    旧实现在每次函数评估中重建样条并用数值微分估梯度，慢约百倍。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()
    s_all = chord_param(pts)
    L = s_all[-1]
    if L < 1e-12:
        return pts.copy()

    k = 3
    nc = max(16, 2 * n)                    # 控制点数
    ℓ = L / (nc - 1)                       # 均匀节点间距（参数域 [0,1]）
    internal = np.arange(1, nc - k, dtype=np.float64) / (nc - k)
    t = np.concatenate([np.zeros(k + 1), internal, np.ones(k + 1)])

    # Greville 参数处配置原折线目标点 Q
    g = np.array([t[i + 1:i + k + 1].mean() for i in range(nc)])
    Q = np.column_stack([np.interp(g * L, s_all, pts[:, 0]),
                         np.interp(g * L, s_all, pts[:, 1])])

    # 配置矩阵（端点行的基函数手工给定，避免右端点开区间问题）
    Bm = np.zeros((nc, nc))
    Bm[0, 0] = 1.0
    Bm[-1, -1] = 1.0
    Bm[1:-1] = BSpline.design_matrix(g[1:-1], t, k, extrapolate=False).toarray()

    w_len, w_curv, w_dev = 1.0, 2.0, 5.0   # 控制网尺度下的平衡权重
    # （旧实现在样条曲率上 0.5 即可，换控制网代理后需上调）

    p0, p1 = pts[0].copy(), pts[-1].copy()

    def fun_grad(x):
        P = np.vstack([p0, x.reshape(-1, 2), p1])
        d = np.diff(P, axis=0)
        Nd = np.linalg.norm(d, axis=1)
        u = d / np.maximum(Nd, 1e-12)[:, None]
        # 长度项: Σ|ΔP|·ℓ，梯度为单位切向量汇编
        g_len = np.zeros_like(P)
        g_len[:-1] -= u
        g_len[1:] += u
        f_len = float(Nd.sum()) * ℓ
        # 弯曲项: Σ|Δ²P|² / ℓ³
        D2 = P[:-2] - 2.0 * P[1:-1] + P[2:]
        f_bend = float(np.sum(D2 * D2)) / ℓ ** 3
        cd2 = (2.0 / ℓ ** 3) * D2
        g_bend = np.zeros_like(P)
        g_bend[:-2] += cd2
        g_bend[1:-1] -= 2.0 * cd2
        g_bend[2:] += cd2
        # 偏离项: mean |Bm P − Q|²（Bm 固定，梯度为矩阵-向量乘）
        r = Bm @ P - Q
        f_dev = float(np.mean(np.sum(r * r, axis=1)))
        g_dev = (2.0 / nc) * (Bm.T @ r)

        f = w_len * f_len + w_curv * f_bend + w_dev * f_dev
        g = (w_len * g_len + w_curv * g_bend + w_dev * g_dev)[1:-1]
        return f, np.ascontiguousarray(g).ravel()

    x0 = Q[1:-1].ravel()
    sol = minimize(fun_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 300, 'ftol': 1e-9})
    P = np.vstack([p0, sol.x.reshape(-1, 2), p1])

    n_out = max(int(L / max(spacing, 1e-6)) + 1, 10)
    uv = np.linspace(0.0, 1.0, n_out)
    uv[-1] = np.nextafter(1.0, 0.0)     # 样条右端点半开区间
    curve = BSpline(t, P, k)(uv)
    curve[-1] = p1                       # 端点硬锚定
    return curve
