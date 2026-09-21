"""
统一可视化模块 —— 曲线对比图、曲率图、指标热力图。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties

# ── 中文字体配置 ──
# matplotlib 3.1.x 对 .ttc 集合支持不佳，直接指定字体文件路径
_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    _cjk_font = FontProperties(fname=_CJK_FONT_PATH)
except Exception:
    _cjk_font = FontProperties()
plt.rcParams['axes.unicode_minus'] = False

from config import Result, Scenario
from metrics import curvature_profile, _chord_param
from registry import registry


# ─── 配色方案 ────────────────────────────────────────────────────────────────

# 算法 → (颜色/线型, 线宽)
DEFAULT_STYLES = {
    # Catmull-Rom 系列
    "cr_uniform":          ('b-',   2.2),
    "cr_chord":            ('c-',   2.0),
    "cr_centripetal":      ('g-',   2.0),
    # 样条插值
    "akima":               ('m-',   1.8),
    "natural_cubic":       ('orange', 1.8),
    "pchip":               ('brown', 1.8),
    "quintic_hermite":     ('teal', 1.9),
    "rbf_interp":          ('#5f9ea0', 1.8),
    # B 样条
    "bspline_uniform_cubic": ('#483d8b', 1.9),
    "bspline_chord_cubic":   ('#6a5acd', 1.9),
    "bspline_approx":        ('slateblue', 1.8),
    "reinsch_smooth_spline": ('navy', 1.8),
    "quadratic_bspline_chaikin": ('#2f4f4f', 1.8),
    # 贝塞尔
    "bezier_chain":        ('deeppink', 1.9),
    "bezier_ls_fit":       ('crimson', 1.8),
    "bezier_corner_g2":    ('#b8860b', 2.0),
    # 曲率连续 / 螺旋
    "clothoid":            ('r-',   2.0),
    "polynomial_spiral":   ('firebrick', 2.0),
    # 多项式 / 滤波 / baseline
    "polynomial_global":   ('#708090', 1.6),
    "savgol_smooth":       ('olive', 1.7),
    "linear":              ('k--',  1.4),
    # 优化 / 能量最小
    "min_curvature_energy":('darkgreen', 2.0),
    "elastic_band":        ('purple', 2.0),
    "min_acceleration":    ('#00ced1', 1.9),
    "min_jerk":            ('#1e90ff', 1.9),
    "min_snap":            ('#008b8b', 1.9),
    "discrete_bending_qp": ('#556b2f', 1.9),
    "laplacian_smooth":    ('gray', 1.7),
    "curve_shortening":    ('#8b008b', 1.8),
}

_FALLBACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def _get_style(algo_name: str, idx: int = 0):
    if algo_name in DEFAULT_STYLES:
        return DEFAULT_STYLES[algo_name]
    color = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
    return (color, 1.8)


# ─── 单场景多算法对比图 ──────────────────────────────────────────────────────

def plot_scenario_comparison(results: List[Result],
                             scenario: Scenario,
                             save_path: Optional[str] = None,
                             show: bool = True):
    """单场景下多算法对比：左=曲线叠加，右=曲率-弧长。

    Parameters
    ----------
    results : 该场景下所有算法的 Result 列表
    scenario : 场景对象
    save_path : 保存路径，None 则不保存
    show : 是否 plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # 控制点折线
    pts = scenario.control_pts
    ax1.plot(pts[:, 0], pts[:, 1], 'k--o', markersize=7, alpha=0.5,
             zorder=1, label='control pts')

    # 曲率图中跳点位置竖线
    s_hop = _chord_param(pts)
    for sh in s_hop[1:-1]:
        ax2.axvline(sh, color='gray', ls=':', linewidth=0.8, alpha=0.6)

    for idx, r in enumerate(results):
        style = _get_style(r.algorithm_name, idx)
        ax1.plot(r.dense_path[:, 0], r.dense_path[:, 1],
                 style[0], linewidth=style[1], alpha=0.85,
                 zorder=2, label=r.algorithm_name)

        s, kappa = curvature_profile(r.dense_path)
        ax2.plot(s, kappa, style[0], linewidth=style[1],
                 alpha=0.85, label=r.algorithm_name)

    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.set_title('Curves')

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('arc length s (m)')
    ax2.set_ylabel('curvature κ (1/m)')
    ax2.set_title('Curvature vs arc length')
    ax2.legend(fontsize=8)

    fig.suptitle(f"Scenario: {scenario.description}", fontsize=13,
                 fontproperties=_cjk_font)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


# ─── 多场景网格总览 ──────────────────────────────────────────────────────────

def plot_overview_grid(results: List[Result],
                       algo_names: List[str],
                       scenarios: List[Scenario],
                       save_path: Optional[str] = None,
                       show: bool = True):
    """多场景 × 多算法 网格图，每个子图画一个场景内所有算法曲线。"""
    n_sc = len(scenarios)
    n_cols = min(4, n_sc)
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 按场景名索引结果
    result_map: Dict[str, Dict[str, Result]] = {}
    for r in results:
        result_map.setdefault(r.scenario_name, {})[r.algorithm_name] = r

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        pts = sc.control_pts
        ax.plot(pts[:, 0], pts[:, 1], 'k--o', markersize=5, alpha=0.4,
                zorder=1, label='ctrl')
        for aidx, algo_name in enumerate(algo_names):
            r = result_map.get(sc.name, {}).get(algo_name)
            if r is None:
                continue
            style = _get_style(algo_name, aidx)
            ax.plot(r.dense_path[:, 0], r.dense_path[:, 1],
                    style[0], linewidth=style[1], alpha=0.8,
                    zorder=2, label=algo_name)
        ax.set_title(sc.description, fontsize=9, fontproperties=_cjk_font)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best', framealpha=0.7)

    # 隐藏多余子图
    for j in range(n_sc, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Path Optimization — Algorithm × Scenario Overview",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


# ─── 指标柱状图对比 ──────────────────────────────────────────────────────────

def plot_metric_bars(results: List[Result],
                     metric_name: str = "max_curvature",
                     save_path: Optional[str] = None,
                     show: bool = True):
    """按场景分组，每个场景内各算法的某一指标柱状图。"""
    # 按场景分组
    sc_groups: Dict[str, List[Result]] = {}
    for r in results:
        sc_groups.setdefault(r.scenario_name, []).append(r)

    n_sc = len(sc_groups)
    n_cols = min(3, n_sc)
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (sc_name, group) in enumerate(sorted(sc_groups.items())):
        ax = axes[idx]
        names = [r.algorithm_name for r in group]
        values = [r.metrics.get(metric_name, 0.0) for r in group]
        colors = [_get_style(n, i)[0].replace('-', '') if len(_get_style(n, i)[0]) <= 2
                  else _get_style(n, i)[0]
                  for i, n in enumerate(names)]
        # 处理 matplotlib line style 字符串
        bar_colors = []
        for i, n in enumerate(names):
            s = _get_style(n, i)[0]
            # 简单映射
            color_map = {'b': '#1f77b4', 'c': '#17becf', 'g': '#2ca02c',
                         'm': '#9467bd', 'r': '#d62728'}
            if s in color_map:
                bar_colors.append(color_map[s])
            elif s in ('b-', 'c-', 'g-', 'm-', 'r-'):
                bar_colors.append(color_map.get(s[0], '#7f7f7f'))
            else:
                bar_colors.append(s if s.startswith('#') else '#7f7f7f')

        bars = ax.bar(range(len(names)), values, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(metric_name)
        ax.set_title(sc_name, fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

    for j in range(n_sc, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Metric Comparison: {metric_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
