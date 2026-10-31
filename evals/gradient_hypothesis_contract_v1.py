"""저장 PPO gradient receipt를 소비하는 독립 8-subtype CPU 가설 계약."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import recorded_ppo_gradient_probe_v1 as RECORDED

from rl_triage import probe_loop as LOOP

STATE_SCHEMA = "gradient_hypothesis_contract_state_v1"
PAIR_SCHEMA = "gradient_hypothesis_source_pair_binding_v1"
SUBTYPE_SCHEMA = "gradient_hypothesis_8_subtype_v1"
PUBLIC_CONTEXT_SCHEMA = "gradient_hypothesis_public_context_v1"
COUNT_PROBE = "P_optimizer_updates"
GRADIENT_PROBE = "P_ppo_gradient"
VALUE = "optimizer_value"
COUNT = "optimizer_update_count"
SURROGATE = "optimizer_surrogate"
SUBTYPES = (
    "reward",
    "actuator",
    "exploration",
    VALUE,
    "physics",
    "termination",
    COUNT,
    SURROGATE,
)
OUTCOMES = ("normal", "abnormal", "unknown")
FUNCTION_PERMISSIONS = {
    "approval": False,
    "execution": False,
    "correction": False,
    "recovery": False,
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_digest(state: dict[str, Any]) -> str:
    return _digest({key: value for key, value in state.items() if key != "state_digest"})


def _finish(state: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(state)
    result["state_digest"] = _state_digest(result)
    return result


def _checked_state(state: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(state, dict) and state.get("schema") == STATE_SCHEMA,
            "gradient hypothesis state schema 불일치")
    require(state.get("subtype_schema") == SUBTYPE_SCHEMA,
            "7-subtype 또는 알 수 없는 subtype state는 허용하지 않는다")
    require(state.get("state_digest") == _state_digest(state), "state digest 불일치")
    require(state.get("function_permissions") == FUNCTION_PERMISSIONS,
            "function 권한 계약 불일치")
    return deepcopy(state)


def registry() -> dict[str, Any]:
    """공개 여섯 범주와 독립 count/gradient 검사를 등록한다."""
    return {
        "schema": "gradient_hypothesis_registry_v1",
        "subtype_schema": SUBTYPE_SCHEMA,
        "public_categories": list(LOOP.MECHANISMS),
        "subtypes": [
            {"subtype": name, "public_category": "optimizer" if name in (VALUE, COUNT, SURROGATE) else name}
            for name in SUBTYPES
        ],
        "outcomes": list(OUTCOMES),
        "probes": {
            COUNT_PROBE: {
                "metric": "3 complete PPO updates의 optimizer.step 및 minibatch 횟수",
                "eliminates_on_normal": [COUNT],
                "confirms_on_abnormal": COUNT,
            },
            GRADIENT_PROBE: {
                "metric": "저장 clipped-surrogate gradient와 advantage·ratio 기대값의 일치 상태",
                "eliminates_on_normal": [SURROGATE],
                "confirms_on_abnormal": SURROGATE,
            },
        },
        "evidence_kind": RECORDED.EVIDENCE_KIND,
        "function_permissions": deepcopy(FUNCTION_PERMISSIONS),
        "claim_limits": {
            "reused_development_gradient_records": True,
            "new_cause_detection": False,
            "policy_recovery": False,
            "live_model_loop": "pending",
            "correction_execution": "pending",
        },
    }


def _path_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"source pair 파일이 없다: {resolved.name}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _endpoint(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} source endpoint는 객체여야 한다")
    require(set(value) == {"role", "plan", "receipt"}, f"{name} source endpoint 계약 불일치")
    require(value["role"] == name, f"{name} role 불일치")
    require(isinstance(value["plan"], (str, Path)) and isinstance(value["receipt"], (str, Path)),
            f"{name} plan/receipt 경로 형식 불일치")
    plan, receipt = Path(value["plan"]), Path(value["receipt"])
    checked = RECORDED.checked_receipt(plan, receipt)
    require(checked.get("evidence_kind") == RECORDED.EVIDENCE_KIND,
            "development_fixture evidence_kind를 보존해야 한다")
    require(checked.get("scope") == RECORDED.SCOPE, "recorded gradient scope 불일치")
    return {
        "role": name,
        "plan": _path_ref(plan),
        "receipt": _path_ref(receipt),
        "source_binding": deepcopy(checked["source_binding"]),
        "evidence_kind": checked["evidence_kind"],
    }


def initialize(source_pair_binding: dict[str, Any], budget: int) -> dict[str, Any]:
    """고정된 recorded-gradient source pair로 새 8-subtype 상태를 연다."""
    require(isinstance(source_pair_binding, dict), "source pair binding은 객체여야 한다")
    require(set(source_pair_binding) == {"schema", "subtype_schema", "case", "reference"},
            "source pair binding 필드 불일치")
    require(source_pair_binding.get("schema") == PAIR_SCHEMA, "source pair schema 불일치")
    require(source_pair_binding.get("subtype_schema") == SUBTYPE_SCHEMA,
            "7-subtype binding의 암묵적 변환은 허용하지 않는다")
    require(type(budget) is int and 1 <= budget <= 2, "budget은 1 또는 2여야 한다")
    binding = {
        "schema": PAIR_SCHEMA,
        "subtype_schema": SUBTYPE_SCHEMA,
        "case": _endpoint("case", source_pair_binding["case"]),
        "reference": _endpoint("reference", source_pair_binding["reference"]),
    }
    return _finish({
        "schema": STATE_SCHEMA,
        "subtype_schema": SUBTYPE_SCHEMA,
        "status": "open",
        "conclusion": None,
        "hypotheses": list(SUBTYPES),
        "observed": {},
        "history": [],
        "budget_left": budget,
        "source_pair_binding": binding,
        "evidence_kind": RECORDED.EVIDENCE_KIND,
        "function_permissions": deepcopy(FUNCTION_PERMISSIONS),
    })


def _same_path_ref(expected: dict[str, str], supplied: Path, label: str) -> None:
    resolved = supplied.resolve()
    require(str(resolved) == expected["path"], f"{label} role/path 불일치")
    require(resolved.is_file() and _sha(resolved) == expected["sha256"],
            f"{label} SHA 불일치")


def _checked_pair(
    state: dict[str, Any],
    case_plan: Path,
    case_receipt: Path,
    ref_plan: Path,
    ref_receipt: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = {
        "case": (Path(case_plan), Path(case_receipt)),
        "reference": (Path(ref_plan), Path(ref_receipt)),
    }
    result = []
    for role, (plan, receipt) in supplied.items():
        expected = state["source_pair_binding"][role]
        _same_path_ref(expected["plan"], plan, f"{role} plan")
        _same_path_ref(expected["receipt"], receipt, f"{role} receipt")
        checked = RECORDED.checked_receipt(plan, receipt)
        require(checked.get("source_binding") == expected["source_binding"],
                f"{role} source binding 불일치")
        require(checked.get("evidence_kind") == state["evidence_kind"],
                f"{role} evidence_kind 불일치")
        result.append(checked)
    return result[0], result[1]


def _advance(
    state: dict[str, Any],
    probe: str,
    case_plan: Path,
    case_receipt: Path,
    ref_plan: Path,
    ref_receipt: Path,
) -> dict[str, Any]:
    current = _checked_state(state)
    require(current["status"] == "open", "열린 상태에서만 advance할 수 있다")
    require(current["budget_left"] > 0, "probe budget이 없다")
    require(probe not in current["observed"], "같은 probe는 한 번만 소비할 수 있다")
    case, reference = _checked_pair(current, case_plan, case_receipt, ref_plan, ref_receipt)
    case_outcome = case["count_status"] if probe == COUNT_PROBE else case["outcome"]
    reference_outcome = reference["count_status"] if probe == COUNT_PROBE else reference["outcome"]
    require(case_outcome in OUTCOMES and reference_outcome in OUTCOMES,
            "receipt outcome이 지원 범위 밖이다")

    effective = case_outcome if reference_outcome == "normal" else "unknown"
    hypotheses = list(current["hypotheses"])
    target = COUNT if probe == COUNT_PROBE else SURROGATE
    status, conclusion = "open", None
    if effective == "normal":
        hypotheses = [name for name in hypotheses if name != target]
    elif effective == "abnormal":
        hypotheses = [target]
        status, conclusion = "confirmed", target
    else:
        status = "hold"

    metrics = (
        {
            "case_status": case_outcome,
            "reference_status": reference_outcome,
            "case_primitives": deepcopy(case["count_primitives"]),
            "reference_primitives": deepcopy(reference["count_primitives"]),
        }
        if probe == COUNT_PROBE
        else {
            "case_status": case_outcome,
            "reference_status": reference_outcome,
            "case_primitive": deepcopy(case["gradient_primitive"]),
            "reference_primitive": deepcopy(reference["gradient_primitive"]),
        }
    )
    observed = {**current["observed"], probe: effective}
    budget_left = current["budget_left"] - 1
    if status == "open" and budget_left == 0:
        status = "hold"
    history = [*current["history"], {
        "probe": probe,
        "outcome": effective,
        "metrics": metrics,
        "evidence_kind": current["evidence_kind"],
    }]
    return _finish({
        **{key: value for key, value in current.items() if key != "state_digest"},
        "status": status,
        "conclusion": conclusion,
        "hypotheses": hypotheses,
        "observed": observed,
        "history": history,
        "budget_left": budget_left,
    })


def advance_count(
    state: dict[str, Any],
    case_plan: Path,
    case_receipt: Path,
    ref_plan: Path,
    ref_receipt: Path,
) -> dict[str, Any]:
    return _advance(state, COUNT_PROBE, case_plan, case_receipt, ref_plan, ref_receipt)


def advance_gradient(
    state: dict[str, Any],
    case_plan: Path,
    case_receipt: Path,
    ref_plan: Path,
    ref_receipt: Path,
) -> dict[str, Any]:
    return _advance(state, GRADIENT_PROBE, case_plan, case_receipt, ref_plan, ref_receipt)


def public_context(state: dict[str, Any]) -> dict[str, Any]:
    """모델에 등록 probe와 안전한 집계 지표만 투영한다."""
    current = _checked_state(state)
    reg = registry()
    eligible = [name for name in reg["probes"] if name not in current["observed"]]
    return {
        "schema": PUBLIC_CONTEXT_SCHEMA,
        "public_categories": list(LOOP.MECHANISMS),
        "status": current["status"],
        "budget_left": current["budget_left"],
        "eligible_probes": eligible,
        "probe_metrics": {name: spec["metric"] for name, spec in reg["probes"].items()},
        "observations": [
            {"probe": row["probe"], "outcome": row["outcome"], "metrics": deepcopy(row["metrics"])}
            for row in current["history"]
        ],
        "evidence_kind": current["evidence_kind"],
        "function_permissions": deepcopy(FUNCTION_PERMISSIONS),
        "live_model_loop": "pending",
        "correction_execution": "pending",
    }


__all__ = [
    "PAIR_SCHEMA",
    "SUBTYPE_SCHEMA",
    "advance_count",
    "advance_gradient",
    "initialize",
    "public_context",
    "registry",
]
