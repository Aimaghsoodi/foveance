"""Offline validation of the paper-2 multi-model benchmark: cost accounting, the budget guard, the
foveance_expand recourse loop, and the arm-wise accuracy story -- all against an in-process mock,
no key, no network egress. This is what makes the benchmark 'ready for a real key' provable in CI.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from foveance.llm import CostAccountant, BudgetExceeded  # noqa: E402
from mock_openrouter import MockOpenRouter  # noqa: E402
from models import ModelSpec  # noqa: E402
import paper2_bench as p2  # noqa: E402


def _spec(tools=True):
    return ModelSpec("mock/model", "mock", 0.5, 1.5, tools)


def test_cost_accountant_guard_and_record():
    acc = CostAccountant(budget_usd=0.01)
    acc.record("m", 0.004)
    assert abs(acc.remaining() - 0.006) < 1e-9
    acc.guard(0.005)                       # 0.004 + 0.005 <= 0.01, fine
    try:
        acc.guard(0.007)                   # would exceed
        assert False, "should have raised"
    except BudgetExceeded:
        pass
    assert acc.by_model["m"] == 0.004


def test_arms_reproduce_the_recourse_story_offline():
    """raw recovers the fact; blind compression (disjoint) loses it; expand recovers it again."""
    with MockOpenRouter() as base_url:
        acc = CostAccountant(budget_usd=5.0)
        oob = {"hit": False}
        spec = _spec(tools=True)

        def acc_of(arm, mode, budget=300):
            row = p2.run_cell(base_url, "k", spec, arm, budget, mode, seeds=3,
                              accountant=acc, out_of_budget=oob)
            return row["accuracy"], row

        raw_acc, _ = acc_of("raw", "disjoint")
        dig_acc, _ = acc_of("digest", "disjoint")
        alloc_acc, _ = acc_of("allocator", "disjoint")
        exp_acc, exp_row = acc_of("allocator+expand", "disjoint")

        assert raw_acc == 1.0                     # uncompressed: always answerable
        assert dig_acc == 0.0                     # disjoint query: salience cannot keep the fact
        assert alloc_acc == 0.0                   # blind graded compression also drops it
        assert exp_acc == 1.0                     # recourse restores it
        assert exp_row["expand_calls"] >= 1       # the model actually used the tool
        assert acc.spent_usd > 0                  # real cost was accounted


def test_expand_arm_skipped_for_non_tool_models():
    with MockOpenRouter() as base_url:
        acc = CostAccountant(budget_usd=5.0)
        row = p2.run_cell(base_url, "k", _spec(tools=False), "allocator+expand", 300,
                          "disjoint", seeds=3, accountant=acc, out_of_budget={"hit": False})
        assert row is None                        # no tool support -> cell produces no rows


def test_budget_guard_stops_the_run():
    """A tiny cap must halt mid-run and flip the out_of_budget flag; overshoot <= one call cost."""
    per_call = 0.01
    with MockOpenRouter(cost_per_call=per_call) as base_url:
        acc = CostAccountant(budget_usd=0.025)    # affords ~2-3 calls
        oob = {"hit": False}
        p2.run_cell(base_url, "k", _spec(), "raw", 300, "overlap", seeds=20,
                    accountant=acc, out_of_budget=oob)
        assert oob["hit"] is True                 # the run was stopped by the cap
        assert acc.spent_usd <= 0.025 + per_call + 1e-9   # overshoot bounded by one call
        assert acc.calls < 20                     # it did NOT run all 20 seeds


def test_overlap_mode_digest_keeps_the_fact():
    """Sanity: when the query lexically cues the fact, salience digestion retains it (no recourse
    needed) -- the honest counterpoint to the disjoint case."""
    with MockOpenRouter() as base_url:
        acc = CostAccountant(budget_usd=5.0)
        row = p2.run_cell(base_url, "k", _spec(), "digest", 300, "overlap", seeds=3,
                          accountant=acc, out_of_budget={"hit": False})
        assert row["accuracy"] == 1.0
