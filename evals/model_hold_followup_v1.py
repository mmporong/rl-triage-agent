"""검증된 model HOLD 뒤의 비실행 shadow 제안만 만든다.

기존 epoch와 닫힌 state는 바꾸지 않는다. 반환값에는 승인, 실행, 교정 또는
자동 재개 권한이 없으며 별도 사전등록 실험의 입력으로만 쓸 수 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import evidence_guided_selector_v1 as SEL
import reasoning_diagnostic_loop_v1 as LOOP
import registry_action_options_v2 as O2

SCHEMA = "model_hold_followup_proposal_v1"
SELECTION_ACTOR = "registered_rule_after_model_hold_proposal"
REWARD_FACT_IDS = tuple(
    f"telemetry:{role}:Episode_Reward/{name}"
    for role in ("case", "reference")
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object가 아니다: {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"파일이 없다: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _checked_ref(value: object, label: str) -> Path:
    _require(isinstance(value, dict), f"{label} ref 형식")
    item = cast(dict[str, Any], value)
    path, expected = item.get("path"), item.get("sha256")
    _require(isinstance(path, str) and isinstance(expected, str), f"{label} ref 필드")
    resolved = Path(cast(str, path)).resolve()
    _require(resolved.is_file() and _sha256(resolved) == expected, f"{label} ref/SHA 불일치")
    return resolved


def _hold(reason: str) -> tuple[dict[str, Any], str]:
    return {"action": "hold", "target": None, "fact_ids": []}, reason


def _option(options: object, *, action: str, target: str | None) -> dict[str, Any] | None:
    if not isinstance(options, list):
        return None
    for item in options:
        if isinstance(item, dict) and item.get("action") == action and item.get("target") == target:
            return cast(dict[str, Any], item)
    return None


def _model_hold_valid(
    last: dict[str, Any], dialogue: dict[str, Any], shadow: dict[str, Any],
    *, reviewer: Callable[[object, dict[str, Any]], dict[str, Any]],
    model_selector: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> bool:
    if dialogue.get("schema") == "reasoning_diagnostic_recorded_dialogue_v1":
        provider = dialogue.get("raw_response")
        stored_review = dialogue.get("review")
    else:
        calls = dialogue.get("calls")
        if not isinstance(calls, list) or not calls or not isinstance(calls[-1], dict):
            return False
        provider = calls[-1].get("result")
        stored_review = dialogue.get("final_review")
    provider_ok = (
        isinstance(provider, dict)
        and provider.get("provider_model") in LOOP.S1.CALL.ACCEPTED_PROVIDER_MODELS
        and provider.get("finish_reason") == "stop"
        and not provider.get("error_type")
        and not provider.get("infra_error")
        and isinstance(provider.get("content"), str)
        and bool(provider["content"].strip())
    )
    if not provider_ok:
        return False
    provider = cast(dict[str, Any], provider)
    review = reviewer(provider["content"], shadow)
    selected = model_selector(review, shadow)
    return bool(
        last.get("model_valid") is True
        and last.get("format_valid") is True
        and dialogue.get("model_valid") is True
        and dialogue.get("action") == {"action": "hold", "target": None, "fact_ids": []}
        and dialogue.get("provider_incomplete_hold") in (False, None)
        and stored_review == review
        and review.get("status") == "valid"
        and selected == {"action": "hold", "target": None, "fact_ids": []}
    )


def _base_result(
    *, epoch_path: Path, study_path: Path, epoch: dict[str, Any], row: dict[str, Any],
    parent_verified: bool, parent_reason: str, shadow_ref: dict[str, str] | None,
    closed_ref: dict[str, str] | None, budgets: dict[str, Any], action: dict[str, Any], reason: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "complete",
        "artifact_visibility": "private",
        "case_id": epoch.get("case_id"),
        "mode": epoch.get("mode"),
        "selection_actor": SELECTION_ACTOR,
        "parent_closed_HOLD": {
            "verified": parent_verified,
            "reason": parent_reason,
            "epoch_result": _ref(epoch_path),
            "closed_state": closed_ref,
            "decision_outcome_preserved": epoch.get("decision_outcome"),
            "model_valid_hold_preserved": row.get("model_valid_hold") is True,
        },
        "parent_study_result": _ref(study_path),
        "shadow_state_before": shadow_ref,
        "source_gate": row.get("gate"),
        "source_binding": {
            "case_checkpoint_sha256": row.get("checkpoint_sha256"),
            "reference_checkpoint_sha256": row.get("reference_checkpoint_sha256"),
        },
        "budgets": budgets,
        "proposal": action,
        "proposal_reason": reason,
        "approval_granted": False,
        "execution_allowed": False,
        "correction_allowed": False,
        "automatic_resume": False,
        "api_requests": 0,
        "sdk_role_executions": 0,
        "claims": {"recovery": False, "correction": False, "execution_effect": False},
    }


def build_proposal(
    epoch_result: Path,
    study_result: Path,
    *,
    expected_epoch_sha256: str,
    expected_study_sha256: str,
    state_checker: Callable[[object], dict[str, Any]] = SEL._state,
    context_builder: Callable[[dict[str, Any]], dict[str, Any]] = O2.context,
    hold_advancer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = SEL.advance,
    reviewer: Callable[[object, dict[str, Any]], dict[str, Any]] = O2.assess,
    model_selector: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = O2.select_model,
) -> dict[str, Any]:
    """동결된 HOLD 직전 state를 재검증하고 비실행 후속 제안을 반환한다."""
    epoch_path, study_path = Path(epoch_result).resolve(), Path(study_result).resolve()
    _require(len(expected_epoch_sha256) == 64 and _sha256(epoch_path) == expected_epoch_sha256,
             "expected epoch SHA 불일치")
    _require(len(expected_study_sha256) == 64 and _sha256(study_path) == expected_study_sha256,
             "expected study SHA 불일치")
    epoch, study = _read(epoch_path), _read(study_path)
    _require(study.get("schema") == "reasoning_diagnosis_study_result_v1", "study result schema")
    _require(study.get("status") == "complete", "study result가 complete가 아니다")
    rows = study.get("rows")
    _require(isinstance(rows, list), "study rows")
    row_items = cast(list[Any], rows)
    matches = [item for item in row_items if isinstance(item, dict)
               and item.get("case_id") == epoch.get("case_id")
               and item.get("mode") == epoch.get("mode")]
    _require(len(matches) == 1, "case/mode study row는 하나여야 한다")
    row = cast(dict[str, Any], matches[0])

    model_used = epoch.get("model_actions")
    probe_used = epoch.get("paired_probes")
    _require(type(model_used) is int and type(probe_used) is int, "epoch budget fields")
    model_used = cast(int, model_used)
    probe_used = cast(int, probe_used)
    budgets: dict[str, Any] = {
        "model_actions": {"used": model_used, "limit": LOOP.MAX_MODEL_ACTIONS,
                          "remaining": max(0, LOOP.MAX_MODEL_ACTIONS - model_used)},
        "paired_probes": {"used": probe_used, "limit": LOOP.MAX_PAIRED_PROBES,
                          "remaining": max(0, LOOP.MAX_PAIRED_PROBES - probe_used)},
        "shadow_probe_budget_left": None,
    }

    common_valid = (
        epoch.get("schema") == "reasoning_diagnostic_loop_result_v1"
        and epoch.get("status") == "complete"
        and epoch.get("decision_outcome") == "HOLD"
        and epoch.get("final_status") == "hold"
        and row.get("status") == "complete"
        and row.get("decision_outcome") == "HOLD"
        and row.get("model_valid_hold") is True
    )
    steps = epoch.get("steps")
    if not common_valid or not isinstance(steps, list) or not steps:
        action, reason = _hold("parent_epoch_incomplete_or_not_verified_model_hold")
        return _base_result(epoch_path=epoch_path, study_path=study_path, epoch=epoch, row=row,
                            parent_verified=False, parent_reason=reason, shadow_ref=None,
                            closed_ref=None, budgets=budgets, action=action, reason=reason)

    last = steps[-1]
    _require(isinstance(last, dict), "last step 형식")
    last = cast(dict[str, Any], last)
    dialogue_path = _checked_ref(last.get("dialogue"), "last dialogue")
    closed_path = _checked_ref(last.get("state_after"), "closed state")
    dialogue, closed = _read(dialogue_path), _read(closed_path)
    closed_ref = _ref(closed_path)
    last_number = last.get("step")
    _require(type(last_number) is int and last_number > 0, "last step number")
    shadow_path = epoch_path.parent / f"step-{last_number:02d}" / "state-before.json"
    shadow_ref = _ref(shadow_path)
    shadow_raw = _read(shadow_path)
    _require(shadow_raw.get("state_digest") == last.get("state_before_digest"),
             "shadow state digest가 last step과 다르다")
    _require(closed.get("state_digest") == last.get("state_after_digest")
             and closed.get("status") == "hold", "closed HOLD state가 last step과 다르다")

    shadow = state_checker(shadow_raw)
    _require(shadow.get("state_digest") == last.get("state_before_digest"),
             "replayed shadow digest가 last step과 다르다")
    hold_action = {"action": "hold", "target": None, "fact_ids": []}
    replayed_closed = hold_advancer(deepcopy(shadow), deepcopy(hold_action))
    _require(closed == replayed_closed, "closed HOLD state가 shadow+hold exact replay와 다르다")
    parent_model_valid = (
        last.get("action") == hold_action
        and _model_hold_valid(last, dialogue, shadow, reviewer=reviewer,
                              model_selector=model_selector)
    )
    if not parent_model_valid:
        action, reason = _hold("parent_hold_provider_or_contract_invalid")
        return _base_result(epoch_path=epoch_path, study_path=study_path, epoch=epoch, row=row,
                            parent_verified=False, parent_reason=reason, shadow_ref=shadow_ref,
                            closed_ref=closed_ref, budgets=budgets, action=action, reason=reason)
    ledger = shadow.get("fact_ledger")
    binding = ledger.get("source_binding") if isinstance(ledger, dict) else None
    _require(isinstance(binding, dict), "shadow source binding")
    binding = cast(dict[str, Any], binding)
    case_binding, reference_binding = binding.get("case"), binding.get("reference")
    _require(isinstance(case_binding, dict) and isinstance(reference_binding, dict),
             "shadow case/reference binding")
    case_binding = cast(dict[str, Any], case_binding)
    reference_binding = cast(dict[str, Any], reference_binding)
    _require(case_binding.get("checkpoint_sha256") == row.get("checkpoint_sha256")
             and reference_binding.get("checkpoint_sha256") == row.get("reference_checkpoint_sha256"),
             "shadow와 study case/reference checkpoint가 다르다")
    budgets["shadow_probe_budget_left"] = shadow.get("budget_left")

    gate = row.get("gate")
    source_failed = (isinstance(gate, dict) and gate.get("run") is True
                     and gate.get("source_outcome") == "not_recovered")
    if not source_failed:
        action, reason = _hold("normal_or_unverified_source_gate")
    elif budgets["model_actions"]["remaining"] == 0:
        action, reason = _hold("no_remaining_model_action_budget")
    elif shadow.get("status") != "open" or shadow.get("budget_left") == 0:
        action, reason = _hold("no_remaining_shadow_inspection_budget")
    else:
        current = context_builder(shadow)
        options = current.get("action_options")
        observed = shadow.get("observed")
        if not isinstance(observed, dict):
            action, reason = _hold("shadow_observations_invalid")
        elif shadow.get("evidence_read") is False:
            available = _option(options, action="read_evidence", target=SEL.RESOURCE)
            if available is None:
                action, reason = _hold("optimizer_count_resource_unavailable")
            else:
                action = {"action": "read_evidence", "target": SEL.RESOURCE, "fact_ids": []}
                reason = "optimizer_count_resource_not_read"
        elif SEL.NEW_PROBE not in observed:
            available = _option(options, action="run_probe", target=SEL.NEW_PROBE)
            if available is None or budgets["paired_probes"]["remaining"] == 0:
                action, reason = _hold("recorded_count_probe_unavailable_or_no_probe_budget")
            else:
                groups = current.get("count_evidence_groups")
                ids = groups[0] if isinstance(groups, list) and groups and isinstance(groups[0], list) else []
                if not ids:
                    action, reason = _hold("recorded_count_fact_group_unavailable")
                else:
                    action = {"action": "run_probe", "target": SEL.NEW_PROBE,
                              "fact_ids": list(ids)}
                    reason = "recorded_count_not_observed"
        elif observed.get(SEL.NEW_PROBE) == "normal":
            available = _option(options, action="run_probe", target="P_reward")
            facts = current.get("evidence_facts")
            ids = [item for item in REWARD_FACT_IDS
                   if isinstance(facts, dict) and facts.get(item) is not None]
            if (available is None or budgets["paired_probes"]["remaining"] == 0
                    or len(ids) != len(REWARD_FACT_IDS)):
                action, reason = _hold("reward_probe_ineligible_or_visible_facts_incomplete")
            else:
                action = {"action": "run_probe", "target": "P_reward", "fact_ids": ids}
                reason = "counts_normal_and_source_tracking_failure_gate_verified"
        else:
            action, reason = _hold("count_outcome_unknown_abnormal_or_unsupported")

    return _base_result(
        epoch_path=epoch_path, study_path=study_path, epoch=epoch, row=row,
        parent_verified=True, parent_reason="verified_model_valid_closed_hold",
        shadow_ref=shadow_ref, closed_ref=closed_ref, budgets=budgets, action=action, reason=reason,
    )


def write_proposal(result: dict[str, Any], out: Path) -> Path:
    target = Path(out).resolve()
    _require(not target.exists(), "proposal output은 새 파일이어야 한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-result", type=Path, required=True)
    parser.add_argument("--study-result", type=Path, required=True)
    parser.add_argument("--expected-epoch-sha256", required=True)
    parser.add_argument("--expected-study-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    written = write_proposal(build_proposal(
        args.epoch_result, args.study_result,
        expected_epoch_sha256=args.expected_epoch_sha256,
        expected_study_sha256=args.expected_study_sha256,
    ), args.out)
    print(json.dumps({"path": str(written), "sha256": _sha256(written)}, sort_keys=True))


if __name__ == "__main__":
    main()
