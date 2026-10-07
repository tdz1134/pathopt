---
name: pathopt-framework
description: Guide work in the pathopt 路径优化算法对比框架（注册表驱动，含 algorithms、scenarios、metrics、runner、visualizer、main.py）。Use when 新增或调试路径算法、场景、评估指标，运行矩阵对比或 --bench 耗时基准，或修改注册/执行/可视化链路。Do not use for unrelated Python code.
---

# pathopt 路径优化算法对比框架

在本仓库写代码前先读本节。框架是纯标准库 + NumPy/SciPy/Matplotlib 的**扁平包结构**，所有扩展通过全局注册表 `registry` 完成，禁止直接操作内部字典。

## 架构地图

| 文件 | 职责 | 关键内容 |
|------|------|----------|
| `config.py` | 数据契约 | `Scenario` dataclass、`PathAlgorithm` Protocol（`name` + `__call__(control_pts, start_tangent, spacing)`）、`Result` dataclass |
| `registry.py` | 全局单例注册表 | `registry.algorithm(name)` 装饰器、`register_algorithm`、`register_scenario`、按 tags 过滤查询；重名注册直接抛 `ValueError` |
| `algorithms/` | 29 个算法，一个算法一个文件 | 每个文件用 `@registry.algorithm("<name>")` 注册；`algorithms/__init__.py` 统一 import 触发注册 |
| `scenarios/presets.py` | 10 个预设场景 | 模块导入时直接 `registry.register_scenario(Scenario(...))` |
| `metrics.py` | 评估指标 | 指标统一签名 `metric(dense, control_pts=None, **kwargs) -> float`，集中登记在 `ALL_METRICS` 字典，由 `evaluate()` 调度 |
| `runner.py` | 执行引擎 | `run_single`、`run_all`（场景 × 算法矩阵）、`run_benchmark`、汇总表打印 |
| `visualizer.py` | matplotlib 出图 | `plot_scenario_comparison`、`plot_overview_grid`、`plot_metric_bars`；样式走 `DEFAULT_STYLES`，未知算法名用 `_FALLBACK_COLORS` |
| `main.py` | CLI 入口 | import `algorithms` 和 `scenarios.presets` 触发注册副作用，串联 runner → metrics → visualizer |
| `CATALOG.md` | 对外清单 | 全部算法/场景/指标/tags 的约定文档，新增条目时必须同步更新 |

## 核心契约（不要破坏）

- 算法输入：`control_pts` 为 `(n, 2)` float64 数组，`start_tangent` 为起点切向角（弧度），`spacing` 为密采样间距；输出必须是 `(m, 2)` ndarray，**首末点固定为控制点首末点**。
- 场景：`Scenario(name, control_pts, start_tangent=0.0, description="", tags=[...])`；`__post_init__` 会校验 shape 必须是 `(n, 2)`。
- 结果：统一封装为 `Result`；算法抛异常时 runner 不崩溃，而是打印 traceback 并返回 `metrics={"error": 1.0}`、`dense_path=控制点副本` 的兜底 Result。因此**算法内部要自行处理退化输入**（点数 < 2、零长度段、重合点），不要指望异常中断整轮对比。
- 指标注册键即 CLI 名：`length / smoothness / max_curvature / curvature_var / max_deviation / lat_accel_cost`。

## 新增算法（按顺序做）

1. 新建 `algorithms/<algo_name>.py`，模板：

```python
"""<一句话说明，含连续阶数 C1/C2 与特点>。"""
from __future__ import annotations

import numpy as np

from registry import registry
# 共享工具优先复用 algorithms/_common.py：
# dedupe_consecutive / chord_param / resample_by_spacing / min_derivative_path
from _common import dedupe_consecutive  # 仅在需要时


@registry.algorithm("<algo_name>")
def <algo_name>(pts: np.ndarray, start_tangent: float,
                spacing: float = 0.1) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:          # 退化输入必须兜底
        return pts.copy()
    spacing = max(spacing, 1e-6)
    ...
    return dense              # (m, 2)，首末点 = pts[0] / pts[-1]
```

2. 在 `algorithms/__init__.py` 对应分类下加 `from . import <algo_name>  # noqa: F401`，并加入 `_ALGORITHM_MODULES` 列表。**漏做这步注册不会生效，也不报错。**
3. 在 `CATALOG.md` 的算法清单补一行（name + 特点 + 文件路径）。
4. 导入路径是扁平式：算法文件里写 `from registry import registry`（main.py 已把根目录插入 `sys.path`），不要写 `from pathopt.registry` 之类的包前缀。

## 新增场景

在 `scenarios/presets.py` 追加 `registry.register_scenario(Scenario(...))`，并按语义打已有 tag（basic/smooth/stress/corner/uneven/extreme/noise/degenerate/dense），便于 `--tags` 筛选；同步更新 `CATALOG.md`。场景名全局唯一，重名会抛 `ValueError`。

## 新增指标

1. 在 `metrics.py` 写 `def my_metric(dense, control_pts=None, **kwargs) -> float`，注册进 `ALL_METRICS`（键名就是 CLI `--metrics` 用的名字）。
2. 曲率类指标复用 `curvature_profile(dense, n_resample=500)`（内部已按弧长均匀重采样并防除零），不要自己另写差分。
3. 在 `CATALOG.md` 指标清单登记。

## 运行与验证

```bash
python main.py                                   # 全矩阵 + 指标表 + 出图
python main.py --no-plot                         # 只打印，最快的冒烟验证
python main.py --tags stress                     # 只跑压力场景
python main.py --algos cr_uniform,clothoid       # 只跑指定算法
python main.py --scenarios sharp_corner,hairpin  # 指定场景（优先于 --tags）
python main.py --spacing 0.05 --metrics max_curvature,smoothness
python main.py --bench 20                        # 预热+重复 N 次取中位数，只测耗时
python main.py --output ./output --no-show       # 无界面环境存图不弹窗
```

新算法的最小验证：`python main.py --algos <name> --scenarios two_points,gentle_curve,sharp_corner --no-plot`，确认无 `[ERROR]` 行、结果中没有 `error=1.0000`，再跑全场景。

## 约定与坑

- 一个算法一个文件；函数名、注册名、文件名保持一致。
- runner 的 `run_single(..., keep_path=False)` 与 `run_all(..., keep_paths=False)` 会丢弃 dense_path 以省内存；大数据量/批量场景默认走这条路径。
- `run_benchmark` 只测算法本身、不算指标，首轮为预热；耗时对比以它为准，`print_timing_summary` 含冷启动噪声只能快速参考。
- 可视化新函数统一签名 `(..., save_path: Optional[str] = None, show: bool = True)`：有 save_path 才 `plt.savefig`，show=False 时 `plt.close(fig)`；新算法的颜色/线型加进 `DEFAULT_STYLES`，不要硬编码。
- 中文标题依赖 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`，缺失时 visualizer 会回退默认字体（中文显示为方块属环境问题，不是代码 bug）。
- 改代码时保留原有实现（注释形式），不删旧逻辑；新建独立脚本用 `tdz_` 前缀。
