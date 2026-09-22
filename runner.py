"""
核心运行引擎 —— 跑 算法 × 场景 矩阵，收集结果。
"""
from __future__ import annotations

import statistics
import time
import traceback
from typing import List, Optional

from config import Result, Scenario
from metrics import evaluate
from registry import registry


def run_single(algo_name: str, scenario: Scenario, spacing: float = 0.05,
               metric_names: Optional[List[str]] = None, *,
               keep_path: bool = True) -> Result:
    """运行单个算法在单个场景上。

    Parameters
    ----------
    algo_name : 已注册的算法名
    scenario : 场景对象
    spacing : 密采样间距
    metric_names : 要计算的指标列表，None 表示全部
    keep_path : False 时不保留 dense_path（基准模式省内存）

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
        dense_path=dense if keep_path else scenario.control_pts.copy(),
        metrics=metrics,
        time_ms=elapsed_ms,
    )


def run_all(scenarios: Optional[List[Scenario]] = None,
            algo_names: Optional[List[str]] = None,
            spacing: float = 0.05,
            metric_names: Optional[List[str]] = None,
            scenario_tags: Optional[List[str]] = None,
            *, keep_paths: bool = True) -> List[Result]:
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
            result = run_single(algo_name, sc, spacing, metric_names,
                                keep_path=keep_paths)
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


def print_timing_summary(results: List[Result]):
    """从已有结果聚合每算法耗时（单次采样，仅作快速参考）。"""
    by_algo: dict = {}
    for r in results:
        if "error" in r.metrics:
            continue
        by_algo.setdefault(r.algorithm_name, []).append(r.time_ms)
    if not by_algo:
        return

    rows = sorted(by_algo.items(),
                  key=lambda kv: statistics.mean(kv[1]), reverse=True)
    width = max(len(a) for a, _ in rows)
    print(f"\n{'─' * (width + 34)}")
    print("  耗时汇总 (ms, 每场景单次采样, 含首run冷启动噪声 — 精确请用 --bench)")
    print(f"{'─' * (width + 34)}")
    print(f"{'Algorithm':{width}s}{'total':>10s}{'mean':>10s}{'max':>10s}")
    for algo, ts in rows:
        print(f"{algo:{width}s}{sum(ts):>10.1f}{statistics.mean(ts):>10.1f}"
              f"{max(ts):>10.1f}")
    print(f"{'─' * (width + 34)}")


def run_benchmark(scenarios: Optional[List[Scenario]] = None,
                  algo_names: Optional[List[str]] = None,
                  spacing: float = 0.05,
                  repeat: int = 10) -> dict:
    """耗时基准测试：每组合预热 1 次 + 重复 repeat 次，取中位数。

    只测算法本身（不含量度评估），消除单次采样噪声。

    Returns
    -------
    dict[algo_name] = {"median": ms, "mean": ms, "std": ms,
                       "min": ms, "max": ms}
    """
    if scenarios is None:
        scenarios = registry.get_all_scenarios()
    if algo_names is None:
        algo_names = registry.list_algorithms()

    print(f"Benchmark: {len(algo_names)} algorithms × {len(scenarios)} "
          f"scenarios × {repeat} repeats (预热后取每组合中位数)")
    print("=" * 60)

    out = {}
    for algo_name in algo_names:
        algo = registry.get_algorithm(algo_name)
        combo_medians = []
        for sc in scenarios:
            times = []
            try:
                for i in range(repeat + 1):   # 第 1 次为预热
                    t0 = time.perf_counter()
                    algo(sc.control_pts, sc.start_tangent, spacing)
                    dt = (time.perf_counter() - t0) * 1000.0
                    if i > 0:
                        times.append(dt)
            except Exception as e:
                print(f"  [ERROR] {algo_name} × {sc.name}: {e} — 跳过该组合")
                continue
            combo_medians.append(statistics.median(times))
        if not combo_medians:
            print(f"  {algo_name:28s} 全部场景失败，跳过")
            continue
        out[algo_name] = {
            "median": statistics.median(combo_medians),
            "mean": statistics.mean(combo_medians),
            "std": statistics.stdev(combo_medians) if len(combo_medians) > 1 else 0.0,
            "min": min(combo_medians),
            "max": max(combo_medians),
        }
        print(f"  {algo_name:28s} done")

    print("=" * 60)
    _print_benchmark_table(out)
    return out


def _print_benchmark_table(bench: dict):
    """按平均耗时降序打印基准表。"""
    rows = sorted(bench.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    width = max(len(a) for a, _ in rows)
    print(f"\n{'─' * (width + 52)}")
    print("  算法耗时基准 (ms，每场景重复取中位数后再统计；std 为场景间差异)")
    print(f"{'─' * (width + 52)}")
    print(f"{'Algorithm':{width}s}{'mean':>10s}{'median':>10s}{'std':>10s}"
          f"{'min':>10s}{'max':>10s}")
    for algo, st in rows:
        print(f"{algo:{width}s}{st['mean']:>10.2f}{st['median']:>10.2f}"
              f"{st['std']:>10.2f}{st['min']:>10.2f}{st['max']:>10.2f}")
    print(f"{'─' * (width + 52)}")
    fastest = rows[-1][0]
    slowest = rows[0][0]
    print(f"最快: {fastest} ({rows[-1][1]['mean']:.2f}ms)  "
          f"最慢: {slowest} ({rows[0][1]['mean']:.2f}ms)")
