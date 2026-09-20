"""
预设场景定义 —— 覆盖各种典型路径分布情况。

已注册场景清单:
  - gentle_curve        : 均匀分布微弯              tags=[basic, smooth]
  - sharp_corner        : 直角急弯                    tags=[corner, stress]
  - uneven_spacing      : 间距悬殊 (前密后疏)    tags=[uneven, stress]
  - s_curve             : S 形曲线                    tags=[basic, smooth]
  - noisy_points        : 带噪声控制点            tags=[noise, stress]
  - two_points          : 仅两点退化                tags=[degenerate]
  - hairpin             : 回头弯 (U-turn)          tags=[extreme, corner]
  - short_last_segment  : 前段等距 + 末段极短  tags=[uneven, stress]
  - dense_hops          : 均匀密集点                tags=[basic, dense]
  - zigzag              : 锯齿形                      tags=[extreme, stress]

可通过 tags 进行分组筛选，例如 --tags stress 可取出所有压力测试场景。
"""
from __future__ import annotations

import numpy as np

from config import Scenario
from registry import registry


# ─── 基础场景 ────────────────────────────────────────────────────────────────

registry.register_scenario(Scenario(
    name="gentle_curve",
    control_pts=np.array([[0, 0], [1, 0.2], [2, 0.5], [3, 1.0], [4, 1.3]]),
    start_tangent=0.0,
    description="均匀分布微弯 — 最正常的路况",
    tags=["basic", "smooth"],
))

registry.register_scenario(Scenario(
    name="sharp_corner",
    control_pts=np.array([[0, 0], [1.5, 0], [3, 0], [3, 1.5], [3, 3]]),
    start_tangent=0.0,
    description="直角急弯 — 测试过冲/鼓包",
    tags=["corner", "stress"],
))

registry.register_scenario(Scenario(
    name="uneven_spacing",
    control_pts=np.array([[0, 0], [0.3, 0.1], [0.6, 0.3], [2.5, 2.0], [5.0, 2.5]]),
    start_tangent=0.0,
    description="间距悬殊 — 前密后疏，测试自交风险",
    tags=["uneven", "stress"],
))

registry.register_scenario(Scenario(
    name="s_curve",
    control_pts=np.array([[0, 0], [1, 1.5], [2, 1.5], [3, 0], [4, -1.5], [5, -1.5]]),
    start_tangent=np.pi / 4,
    description="S 形曲线 — CR 经典好用场景",
    tags=["basic", "smooth"],
))

# ─── 噪声/退化 ───────────────────────────────────────────────────────────────

np.random.seed(42)
_noisy_pts = np.array([[0, 0], [1, 0.5], [2, 1], [3, 0.5], [4, 0]])
_noisy_pts = _noisy_pts + np.random.randn(5, 2) * 0.3

registry.register_scenario(Scenario(
    name="noisy_points",
    control_pts=_noisy_pts,
    start_tangent=0.1,
    description="带噪声的控制点 — 模拟传感器抖动",
    tags=["noise", "stress"],
))

registry.register_scenario(Scenario(
    name="two_points",
    control_pts=np.array([[0, 0], [2, 1]]),
    start_tangent=0.3,
    description="仅两点退化 — 测试单段行为",
    tags=["degenerate"],
))

# ─── 极端场景 ────────────────────────────────────────────────────────────────

registry.register_scenario(Scenario(
    name="hairpin",
    control_pts=np.array([[0, 0], [2, 0], [3, 1], [2, 2], [0, 2], [-1, 1]]),
    start_tangent=0.0,
    description="回头弯 (U-turn) — 控制点几乎反向",
    tags=["extreme", "corner"],
))

registry.register_scenario(Scenario(
    name="short_last_segment",
    control_pts=np.array([[0, 0], [1.2, 0.2], [2.4, 0.5], [3.6, 0.9], [3.8, 1.0]]),
    start_tangent=0.0,
    description="前段等距 + 末段极短 — 模拟链式前瞻末跳被截短",
    tags=["uneven", "stress"],
))

registry.register_scenario(Scenario(
    name="dense_hops",
    control_pts=np.column_stack([
        np.linspace(0, 5, 8),
        0.8 * np.sin(np.linspace(0, np.pi, 8))
    ]),
    start_tangent=0.0,
    description="均匀密集点 — 模拟链式前瞻跳点",
    tags=["basic", "dense"],
))

registry.register_scenario(Scenario(
    name="zigzag",
    control_pts=np.array([[0, 0], [1, 2], [2, 0], [3, 2], [4, 0], [5, 2]]),
    start_tangent=np.pi / 4,
    description="锯齿形 — 测试剧烈交替转向",
    tags=["extreme", "stress"],
))
