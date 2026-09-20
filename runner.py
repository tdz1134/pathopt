"""
核心运行引擎 —— 跑 算法 × 场景 矩阵，收集结果。
"""
from __future__ import annotations

import time
import traceback
from typing import List, Optional

from config import Result, Scenario
from metrics import evaluate
from registry import registry


def run_single(algo_name: str, scenario: Scenario, spacing: float = 0.05,
               metric_names: Optional[List[str]] = None) -> Result:
    """运行单个算法在单个场景上。

    Parameters
    ----------
    algo_name : 已注册的算法名
    scenario : 场景对象
    spacing : 密采样间距
    metric_names : 要计算的指标列表，None 表示全部

    Returns
    -------
    Result 对象
    """
    algo = registry.get_algorithm(algo_name)

    t0 = time.perf_counter()
    try:
        dense = algo(scenario.control_pts, scenario.start_tangent, spacing)
    except Exception as e:
        print(f"  [ERROR] {algo_name} × {scenario.name}: {e}")
        traceback.print_exc()
        return Result(
            scenario_name=scenario.name,
            algorithm_name=algo_name,
            dense_path=scenario.control_pts.copy(),
            metrics={"error": 1.0},
            time_ms=0.0,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metrics = evaluate(dense, control_pts=scenario.control_pts,
                       metric_names=metric_names)

    return Result(
        scenario_name=scenario.name,
        algorithm_name=algo_name,
        dense_path=dense,
        metrics=metrics,
        time_ms=elapsed_ms,
    )


def run_all(scenarios: Optional[List[Scenario]] = None,
            algo_names: Optional[List[str]] = None,
            spacing: float = 0.05,
            metric_names: Optional[List[str]] = None,
            scenario_tags: Optional[List[str]] = None) -> List[Result]:
    """运行 算法 × 场景 全矩阵对比。

    Parameters
    ----------
    scenarios : 显式场景列表（优先于 scenario_tags）
    algo_names : 算法名列表，None 表示全部已注册算法
    spacing : 密采样间距
    metric_names : 要计算的指标列表
    scenario_tags : 场景标签过滤（仅在 scenarios 为 None 时生效）

    Returns
    -------
    List[Result] — 所有结果
    """
    if scenarios is None:
        scenarios = registry.get_all_scenarios(tags=scenario_tags)
    if algo_names is None:
        algo_names = registry.list_algorithms()

    results = []
    total = len(algo_names) * len(scenarios)
    print(f"Running {len(algo_names)} algorithms × {len(scenarios)} scenarios = {total} combinations")
    print("=" * 60)

    for sc in scenarios:
        for algo_name in algo_names:
            print(f"  {algo_name:25s} × {sc.name:20s} ...", end=" ", flush=True)
            result = run_single(algo_name, sc, spacing, metric_names)
            print(f"{result.time_ms:7.1f}ms | pts={len(result.dense_path):4d}")
            results.append(result)

    print("=" * 60)
    print(f"Done. {len(results)} results collected.")
    return results


def results_to_table(results: List[Result]) -> dict:
    """将结果列表转换为便于查询的嵌套字典。

    Returns
    -------
    dict[scenario_name][algorithm_name] = Result
    """
    table = {}
    for r in results:
        if r.scenario_name not in table:
            table[r.scenario_name] = {}
        table[r.scenario_name][r.algorithm_name] = r
    return table


def print_summary(results: List[Result], metric_name: str = "max_curvature"):
    """打印指标对比表。"""
    table = results_to_table(results)
    algo_names = sorted({r.algorithm_name for r in results})
    scenario_names = sorted({r.scenario_name for r in results})

    # 表头
    header = f"{'Scenario':25s}" + "".join(f"{a:>22s}" for a in algo_names)
    print(f"\n{'─' * len(header)}")
    print(f"  Metric: {metric_name}")
    print(f"{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for sc_name in scenario_names:
        row = f"{sc_name:25s}"
        for algo_name in algo_names:
            r = table.get(sc_name, {}).get(algo_name)
            if r and metric_name in r.metrics:
                row += f"{r.metrics[metric_name]:>22.4f}"
            else:
                row += f"{'N/A':>22s}"
        print(row)

    print(f"{'─' * len(header)}")
