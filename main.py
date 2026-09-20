#!/usr/bin/env python3
"""
路径优化对比框架 —— 主入口。

用法示例：
    # 运行全部算法 × 全部场景，打印指标表 + 可视化
    python main.py

    # 只看 "stress" 标签的场景
    python main.py --tags stress

    # 只跑指定算法
    python main.py --algos cr_uniform,cr_chord,clothoid

    # 只跑指定场景
    python main.py --scenarios sharp_corner,hairpin

    # 保存图片到 output 目录
    python main.py --output ./output
"""
from __future__ import annotations

import argparse
import os
import sys

# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入算法和场景（触发注册）
import algorithms.interpolation   # noqa: F401
import algorithms.optimization    # noqa: F401
import scenarios.presets          # noqa: F401

from registry import registry
from runner import run_all, print_summary, results_to_table
from visualizer import plot_scenario_comparison, plot_overview_grid, plot_metric_bars


def main():
    parser = argparse.ArgumentParser(description="Path Optimization Comparison Framework")
    parser.add_argument("--tags", type=str, default=None,
                        help="场景标签过滤，逗号分隔，例如 stress,corner")
    parser.add_argument("--algos", type=str, default=None,
                        help="算法名列表，逗号分隔")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="场景名列表，逗号分隔（优先于 --tags）")
    parser.add_argument("--spacing", type=float, default=0.05,
                        help="密采样间距 (default: 0.05)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="指标名列表，逗号分隔")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（保存图片和表格）")
    parser.add_argument("--no-plot", action="store_true",
                        help="不显示图表，只打印指标表")
    parser.add_argument("--no-show", action="store_true",
                        help="保存图片但不弹出显示")
    args = parser.parse_args()

    # 解析参数
    tags = args.tags.split(",") if args.tags else None
    algo_names = args.algos.split(",") if args.algos else None
    scenario_names = args.scenarios.split(",") if args.scenarios else None
    metric_names = args.metrics.split(",") if args.metrics else None
    show = not args.no_show and not args.no_plot

    # 确定要跑的场景
    if scenario_names:
        scenarios = [registry.get_scenario(n) for n in scenario_names]
    else:
        scenarios = registry.get_all_scenarios(tags=tags)

    if not scenarios:
        print("Error: 没有匹配的场景。")
        print(f"可用场景: {registry.list_scenarios()}")
        sys.exit(1)

    # 输出目录
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── 运行 ─────────────────────────────────────────────────────────────
    print(f"\nAlgorithms : {algo_names or registry.list_algorithms()}")
    print(f"Scenarios  : {[s.name for s in scenarios]}")
    print(f"Spacing    : {args.spacing}")
    print()

    results = run_all(
        scenarios=scenarios if scenario_names else None,
        algo_names=algo_names,
        spacing=args.spacing,
        metric_names=metric_names,
        scenario_tags=tags if not scenario_names else None,
    )

    # ── 指标表 ───────────────────────────────────────────────────────────
    for metric in (metric_names or ["max_curvature", "smoothness", "length"]):
        print_summary(results, metric_name=metric)

    # ── 可视化 ───────────────────────────────────────────────────────────
    if args.no_plot:
        return

    actual_algo_names = sorted({r.algorithm_name for r in results})

    # 1) 总览网格图
    save_path = os.path.join(output_dir, "overview_grid.png") if output_dir else None
    plot_overview_grid(results, actual_algo_names, scenarios,
                       save_path=save_path, show=show)

    # 2) 每个场景的详细对比图
    result_map = results_to_table(results)
    for sc in scenarios:
        sc_results = list(result_map.get(sc.name, {}).values())
        if not sc_results:
            continue
        save_path = os.path.join(output_dir, f"compare_{sc.name}.png") if output_dir else None
        plot_scenario_comparison(sc_results, sc, save_path=save_path, show=show)

    # 3) 指标柱状图
    for metric in (metric_names or ["max_curvature", "smoothness"]):
        save_path = os.path.join(output_dir, f"bars_{metric}.png") if output_dir else None
        plot_metric_bars(results, metric_name=metric,
                         save_path=save_path, show=show)


if __name__ == "__main__":
    main()
