"""单元测试：LangGraph 条件路由函数。

为什么单测这里特别重要
======================
路由错了的代价大：
- 把 eda_only 路由进 cleaner → 浪费 token + 引入未必相关的清洗
- 把 modeling 跳过 cleaner → 模型用脏数据训练，AUC 雪崩

这些函数纯逻辑（不依赖 API / DB / 文件系统），单测覆盖所有分支就够。
"""
from __future__ import annotations

import pytest

from app.graph.routing import should_clean, should_model
from app.graph.state import TaskPlan, new_state


# ============ should_clean ============

class TestShouldClean:
    def test_empty_state_skips(self):
        """state 里没有 task_plan：默认跳过 cleaner。"""
        state = new_state("q", ["raw_sample"], "rid-0")
        assert should_clean(state) == "skip_cleaning"

    def test_eda_only_always_skips_even_if_need_cleaning(self):
        """eda_only 优先级最高：哪怕 need_cleaning=True 也跳过。"""
        state = new_state("q", [], "rid-1")
        state["task_plan"] = TaskPlan(
            task_type="eda_only",
            target_column=None,
            need_cleaning=True,   # 故意设 True
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_clean(state) == "skip_cleaning"

    def test_modeling_with_need_cleaning_goes_to_cleaner(self):
        state = new_state("q", [], "rid-2")
        state["task_plan"] = TaskPlan(
            task_type="modeling",
            target_column="clk",
            need_cleaning=True,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_clean(state) == "cleaner"

    def test_modeling_without_need_cleaning_skips(self):
        """虽是建模任务但 Planner 觉得不用清洗 → 跳过。"""
        state = new_state("q", [], "rid-3")
        state["task_plan"] = TaskPlan(
            task_type="modeling",
            target_column="clk",
            need_cleaning=False,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_clean(state) == "skip_cleaning"

    def test_comparison_routes_per_need_cleaning(self):
        """comparison 路径完全看 need_cleaning。"""
        state = new_state("q", [], "rid-4")
        state["task_plan"] = TaskPlan(
            task_type="comparison",
            target_column=None,
            need_cleaning=True,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_clean(state) == "cleaner"
        state["task_plan"]["need_cleaning"] = False
        assert should_clean(state) == "skip_cleaning"


# ============ should_model ============

class TestShouldModel:
    def test_empty_state_skips(self):
        state = new_state("q", [], "rid-5")
        assert should_model(state) == "skip_modeling"

    def test_eda_only_skips(self):
        state = new_state("q", [], "rid-6")
        state["task_plan"] = TaskPlan(
            task_type="eda_only",
            target_column=None,
            need_cleaning=False,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_model(state) == "skip_modeling"

    def test_modeling_goes_to_modeler(self):
        state = new_state("q", [], "rid-7")
        state["task_plan"] = TaskPlan(
            task_type="modeling",
            target_column="clk",
            need_cleaning=True,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_model(state) == "modeler"

    def test_comparison_goes_to_modeler(self):
        """comparison 也走 modeler（要做分组对比的指标计算）。"""
        state = new_state("q", [], "rid-8")
        state["task_plan"] = TaskPlan(
            task_type="comparison",
            target_column=None,
            need_cleaning=False,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_model(state) == "modeler"

    def test_general_skips_modeler(self):
        """general 落到 skip（保守策略：不确定就别花算力训模型）。"""
        state = new_state("q", [], "rid-9")
        state["task_plan"] = TaskPlan(
            task_type="general",
            target_column=None,
            need_cleaning=True,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_model(state) == "skip_modeling"


# ============ 组合行为 ============

class TestRoutingCombinations:
    """两个路由函数的组合：覆盖所有"实际会走"的拓扑路径。"""

    @pytest.mark.parametrize(
        "task_type, need_cleaning, expected_clean, expected_model",
        [
            ("eda_only",   False, "skip_cleaning", "skip_modeling"),
            ("eda_only",   True,  "skip_cleaning", "skip_modeling"),  # eda 优先
            ("modeling",   True,  "cleaner",       "modeler"),         # 全链路
            ("modeling",   False, "skip_cleaning", "modeler"),         # 跳清洗
            ("comparison", True,  "cleaner",       "modeler"),
            ("comparison", False, "skip_cleaning", "modeler"),
            ("general",    True,  "cleaner",       "skip_modeling"),
            ("general",    False, "skip_cleaning", "skip_modeling"),
        ],
    )
    def test_routing_matrix(
        self, task_type, need_cleaning, expected_clean, expected_model
    ):
        state = new_state("q", [], "rid-matrix")
        state["task_plan"] = TaskPlan(
            task_type=task_type,
            target_column=None,
            need_cleaning=need_cleaning,
            relevant_tables=[],
            subtasks=[],
            rationale="",
        )
        assert should_clean(state) == expected_clean
        assert should_model(state) == expected_model
