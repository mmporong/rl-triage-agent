"""승인 원장으로 분모를 고정하고 누락·중복·예산 초과를 통과로 집계하지 않는다."""
import importlib.util
from copy import deepcopy

import pytest

from test_offline_replay import ROOT


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "evals"))
    spec = importlib.util.spec_from_file_location("live_summary_test", ROOT / "evals/live_loop_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "foreign", "probe"])
def test_result_rows_cannot_change_receipt_denominator(summary, mutation):
    requests = {"a": {"probe": "P_noise", "receipt": {}}, "b": {"probe": "P_reward", "receipt": {}},
                "rejected": {"probe": "P_torque", "state": "rejected"}}
    runs = [{"request_id": "a", "probe": "P_noise"}, {"request_id": "b", "probe": "P_reward"}]
    if mutation == "omit":
        runs.pop()
    elif mutation == "duplicate":
        runs.append(deepcopy(runs[0]))
    elif mutation == "foreign":
        runs[1]["request_id"] = "rejected"
    else:
        runs[1]["probe"] = "P_torque"
    with pytest.raises(ValueError):
        summary.validated_runs("case", runs, requests)


def test_denominator_uses_receipt_order_and_excludes_rejected_proposal(summary):
    requests = {"a": {"probe": "P_noise", "receipt": {}}, "rejected": {"probe": "P_torque"},
                "b": {"probe": "P_reward", "receipt": {}}}
    runs = [{"request_id": "b", "probe": "P_reward"}, {"request_id": "a", "probe": "P_noise"}]
    assert summary.validated_runs("case", runs, requests) == runs[::-1]


@pytest.mark.parametrize("field,value", [("cost_budget_status", "over"), ("cost_budget_status", "unknown"),
    ("end_to_end_budget_status", "over"), ("end_to_end_budget_status", "unknown"), ("probe_budgets_within", False)])
def test_unknown_or_over_budget_prevents_acceptance(summary, field, value):
    row = {"cost_budget_status": "within", "end_to_end_budget_status": "within", "probe_budgets_within": True}
    assert all(summary.budget_acceptance([row]).values())
    row[field] = value
    assert not all(summary.budget_acceptance([row]).values())
