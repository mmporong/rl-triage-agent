"""옵티마이저 검사부터 시작하는 명시 등록 순서 정책."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "evals", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import selection_comparison_controller_v1 as C

POLICY = "optimizer_first_registered_rule"
OPTIMIZER_PROBE = "P_optimizer_updates"
REWARD_PROBE = "P_reward"
LEGACY_RECORDED_PROBE = "P_recorded_optimizer_updates"
SOURCE_FILES = (
    "evals/registered_optimizer_first_policy_v1.py",
    "tests/test_registered_optimizer_first_policy_v1.py",
)


def _hold(checked: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema": "selection_controller_choice_v1",
        "context_digest": C.L.digest(checked["binding"]),
        "view_digest": checked["view_digest"],
        "policy": POLICY,
        "selected_by": "hold",
        "action": "hold",
        "probe": None,
        "fact_ids": [],
        "model_status": "not_used",
        "codes": [],
        "reason": reason,
        "used_for_decision": True,
        "approval_granted": False,
    }


def _continue(checked: dict[str, Any], probe: str, reason: str) -> dict[str, Any]:
    result = _hold(checked, reason)
    result.update(selected_by="registered_rule", action="continue", probe=probe)
    return result


def choose(snapshot: dict[str, Any]) -> dict[str, Any]:
    """현재 공개 view만으로 optimizer→reward→registry 순서를 선택한다."""
    checked = C.V.checked(snapshot)
    view = checked["view"]
    if view["status"] != "open" or view["budget_left"] <= 0:
        return _hold(checked, "state_" + str(view["status"]))
    eligible = view["eligible_probes"]
    observations = {row["probe"]: row.get("outcome") for row in view["observations"]}
    optimizer = observations.get(OPTIMIZER_PROBE)
    reward = observations.get(REWARD_PROBE)

    if OPTIMIZER_PROBE not in observations:
        if OPTIMIZER_PROBE not in eligible:
            return _hold(checked, "optimizer_probe_not_eligible")
        return _continue(checked, OPTIMIZER_PROBE, "optimizer_first")
    if optimizer != "normal":
        return _hold(checked, "optimizer_outcome_" + str(optimizer))

    if REWARD_PROBE not in observations:
        if REWARD_PROBE not in eligible:
            return _hold(checked, "reward_probe_not_eligible_after_optimizer_normal")
        return _continue(checked, REWARD_PROBE, "reward_after_optimizer_normal")
    if reward != "normal":
        return _hold(checked, "reward_outcome_" + str(reward))

    probe = C.V.rule_probe(view, "registry_next")
    if probe is None or probe not in eligible:
        return _hold(checked, "registered_remaining_probe_unavailable")
    return _continue(checked, probe, "registry_after_optimizer_and_reward_normal")


def request_spec(snapshot: dict[str, Any], choice: dict[str, Any]) -> dict[str, Any]:
    """선택 일치 확인 뒤 기존 호스트의 canonical 실행 후보를 반환한다."""
    expected = choose(snapshot)
    if choice != expected or expected["action"] != "continue":
        raise ValueError("현재 optimizer-first 선택과 실행 후보가 다르다")
    return C.request_spec(snapshot, choice)

