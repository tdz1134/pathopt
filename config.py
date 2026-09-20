"""
核心数据结构定义。

Scenario  —— 一个测试场景（路径点 + 初始条件）
Algorithm —— 所有算法必须遵循的统一接口协议
Result    —— 单次运行结果（算法输出 + 指标 + 耗时）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, runtime_checkable

import numpy as np


# ─── 场景 ────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """一个测试场景。

    Attributes
    ----------
    name : str
        场景唯一标识，例如 "sharp_corner"。
    control_pts : np.ndarray, shape (n, 2)
        路径控制点坐标。
    start_tangent : float
        起点切向角 (rad)，模拟车头朝向。
    description : str
        人类可读的场景描述（用于图表标题）。
    tags : list[str]
        标签，用于筛选场景子集，例如 ["corner", "uneven"]。
    """
    name: str
    control_pts: np.ndarray
    start_tangent: float = 0.0
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.control_pts = np.asarray(self.control_pts, dtype=np.float64)
        if self.control_pts.ndim != 2 or self.control_pts.shape[1] != 2:
            raise ValueError(f"control_pts 必须是 (n, 2) 数组，当前 shape={self.control_pts.shape}")
        if not self.description:
            self.description = self.name


# ─── 算法协议 ────────────────────────────────────────────────────────────────

@runtime_checkable
class PathAlgorithm(Protocol):
    """所有路径算法必须实现的接口。

    调用约定：
        dense_pts = algorithm(control_pts, start_tangent, spacing)

    Parameters
    ----------
    control_pts : (n, 2) ndarray — 路径控制点
    start_tangent : float — 起点切向角 (rad)
    spacing : float — 密采样间距

    Returns
    -------
    dense : (m, 2) ndarray — 密采样后的路径点
    """
    name: str

    def __call__(self, control_pts: np.ndarray, start_tangent: float,
                 spacing: float = 0.1) -> np.ndarray:
        ...


# ─── 运行结果 ────────────────────────────────────────────────────────────────

@dataclass
class Result:
    """单次 (算法 × 场景) 运行结果。"""
    scenario_name: str
    algorithm_name: str
    dense_path: np.ndarray                  # (m, 2) 密采样路径
    metrics: Dict[str, float] = field(default_factory=dict)
    time_ms: float = 0.0                    # 算法执行耗时 (ms)

    def __repr__(self):
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (f"Result({self.scenario_name} × {self.algorithm_name} | "
                f"pts={len(self.dense_path)} | {metric_str} | {self.time_ms:.1f}ms)")
