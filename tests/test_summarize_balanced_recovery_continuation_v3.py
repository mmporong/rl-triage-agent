from __future__ import annotations

from copy import deepcopy

import pytest

from evals import summarize_balanced_recovery_continuation_v3 as summary


def _plan() -> dict:
    rotations = {
        "case-05": ["fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent"],
        "case-06": ["static_rule", "one_shot_llm", "iterative_agent", "fixed_sequence"],
        "case-07": ["one_shot_llm", "iterative_agent", "fixed_sequence", "static_rule"],
        "case-08": ["iterative_agent", "fixed_sequence", "static_rule", "one_shot_llm"],
    }
    return {"cases": [{"case_id": case, "rotation": methods} for case, methods in rotations.items()]}


def _timing(order: list[tuple[str, str]]) -> dict:
    return {"transitions": [
        {"case_id": case_id, "method": method} for case_id, method in order
    ]}


def test_join_accepts_actual_hold_controller_transition_and_collapses_boundary() -> None:
    plan = _plan()
    expected = summary.E._expected_order(plan)
    prior = _timing(expected[:8])
    current = _timing(expected[7:])
    assert summary.E.EXPECTED_HOLD in summary.E._observed_order(current)
    assert summary._join_order(plan, prior, current) == expected


def test_join_rejects_missing_hold_or_wrong_resumed_order() -> None:
    plan = _plan()
    expected = summary.E._expected_order(plan)
    prior = _timing(expected[:8])
    missing_hold = _timing([row for row in expected[7:] if row != summary.E.EXPECTED_HOLD])
    with pytest.raises(summary.E.ExportError, match="controller order differs"):
        summary._join_order(plan, prior, missing_hold)
    wrong = deepcopy(expected[7:])
    wrong[-1], wrong[-2] = wrong[-2], wrong[-1]
    with pytest.raises(summary.E.ExportError, match="controller order differs"):
        summary._join_order(plan, prior, _timing(wrong))


def test_join_rejects_interrupted_boundary_that_is_not_repeated_once() -> None:
    plan = _plan()
    expected = summary.E._expected_order(plan)
    with pytest.raises(summary.E.ExportError, match="controller order differs"):
        summary._join_order(plan, _timing(expected[:8]), _timing(expected[8:]))
