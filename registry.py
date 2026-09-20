"""
全局注册表 —— 算法和场景通过装饰器 / 函数注册，运行时查询。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from config import PathAlgorithm, Scenario


class Registry:
    """集中管理所有已注册的算法和场景。"""

    def __init__(self):
        self._algorithms: Dict[str, PathAlgorithm] = {}
        self._scenarios: Dict[str, Scenario] = {}

    # ── 算法 ─────────────────────────────────────────────────────────────

    def register_algorithm(self, algo: PathAlgorithm, *, name: Optional[str] = None) -> PathAlgorithm:
        """注册一个算法实例。可作装饰器或普通调用。

        用法 1 — 装饰器：
            @registry.register_algorithm
            class MyAlgo:
                name = "my_algo"
                def __call__(self, ...): ...

        用法 2 — 带名装饰器：
            @registry.register_algorithm(name="custom_name")
            def my_func(pts, tangent, spacing): ...

        用法 3 — 直接注册：
            registry.register_algorithm(some_algo_instance)
        """
        # 支持 @registry.register_algorithm(name="xxx") 形式
        if name is not None and callable(algo):
            # 返回一个 wrapper 类实例
            original = algo
            original.name = name
            self._algorithms[name] = original
            return original

        algo_name = name or getattr(algo, 'name', None) or type(algo).__name__
        if algo_name in self._algorithms:
            raise ValueError(f"算法 '{algo_name}' 已注册，不允许重复。")
        algo.name = algo_name
        self._algorithms[algo_name] = algo
        return algo

    def algorithm(self, name: str):
        """装饰器形式注册算法函数。

        @registry.algorithm("cr_uniform")
        def cr_uniform(pts, tangent, spacing): ...
        """
        def decorator(func):
            func.name = name
            self._algorithms[name] = func
            return func
        return decorator

    def get_algorithm(self, name: str) -> PathAlgorithm:
        if name not in self._algorithms:
            raise KeyError(f"算法 '{name}' 未注册。可用: {list(self._algorithms.keys())}")
        return self._algorithms[name]

    def list_algorithms(self) -> List[str]:
        return list(self._algorithms.keys())

    # ── 场景 ─────────────────────────────────────────────────────────────

    def register_scenario(self, scenario: Scenario) -> Scenario:
        """注册一个场景。"""
        if scenario.name in self._scenarios:
            raise ValueError(f"场景 '{scenario.name}' 已注册，不允许重复。")
        self._scenarios[scenario.name] = scenario
        return scenario

    def get_scenario(self, name: str) -> Scenario:
        if name not in self._scenarios:
            raise KeyError(f"场景 '{name}' 未注册。可用: {list(self._scenarios.keys())}")
        return self._scenarios[name]

    def list_scenarios(self, tags: Optional[List[str]] = None) -> List[str]:
        """列出所有场景名，可按 tags 过滤（任一 tag 匹配即返回）。"""
        if tags is None:
            return list(self._scenarios.keys())
        return [name for name, sc in self._scenarios.items()
                if any(t in sc.tags for t in tags)]

    def get_all_scenarios(self, tags: Optional[List[str]] = None) -> List[Scenario]:
        """获取所有场景对象列表，可按 tags 过滤。"""
        names = self.list_scenarios(tags)
        return [self._scenarios[n] for n in names]

    def get_all_algorithms(self) -> List[PathAlgorithm]:
        return list(self._algorithms.values())


# ── 全局单例 ──────────────────────────────────────────────────────────────────
registry = Registry()
