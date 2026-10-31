from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from lineage_post_diagnosis_finalize_v1 import terminal


def view(status="unidentifiable", left=0, remaining=None):
    return {"status": status, "budget_left": left,
            "remaining_candidates": remaining or ["actuator", "optimizer_update_count"],
            "observations": [{"outcome": "normal"} for _ in range(3)]}


def test_real_exhausted_registry_state_is_hold_without_repair_authority():
    result = terminal(view())
    assert result["hold_reason"] == "decision_limit"
    assert result["end_state"] == "HOLD" and result["approval_allowed"] is False


def test_incomplete_budget_cannot_be_reported_as_finished():
    with pytest.raises(ValueError, match="소진"):
        terminal(view(left=1))


def test_unsupported_physics_confirmation_does_not_authorize_repair():
    result = terminal(view(status="confirmed", remaining=["physics"]))
    assert result["hold_reason"] == "repair_unsupported"
    assert result["approval_allowed"] is False


def test_unknown_is_not_promoted_to_normal_or_confirmation():
    current = view()
    current["observations"][2]["outcome"] = "unknown"
    assert terminal(current)["hold_reason"] == "probe_outcome_unknown"
