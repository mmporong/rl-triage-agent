"""등록 option_id 모델 계약을 기존 selector와 paired probe host에 연결한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import evidence_guided_probe_host_v1 as HOST
import evidence_guided_selector_v1 as SEL
import registry_action_options_v1 as O

SCHEMA = "registry_action_loop_result_v1"
MAX_MODEL_ACTIONS = 4
MAX_PAIRED_PROBES = 3
SOURCE_FILES = (
    "evals/registry_action_loop_v1.py",
    "tests/test_registry_action_loop_v1.py",
    "evals/registry_action_options_v1.py",
    "tests/test_registry_action_options_v1.py",
    "evals/evidence_guided_selector_v1.py",
    "tests/test_evidence_guided_selector_v1.py",
    "evals/evidence_guided_probe_host_v1.py",
    "tests/test_evidence_guided_probe_host_v1.py",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": sha(path)}


def _pair_proof(
    pair: dict[str, Any], state: dict[str, Any], action: dict[str, Any], probe_out: Path,
) -> dict[str, Any]:
    names = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
    require(all(isinstance(pair.get(name), (str, Path)) for name in names),
            "paired host가 plan/receipt 네 경로를 반환하지 않았다")
    probe_out = Path(probe_out).resolve()
    require(probe_out.is_dir(), "paired probe output 폴더가 없다")
    paths = {name: Path(pair[name]).resolve() for name in names}
    require(all(path.is_relative_to(probe_out) for path in paths.values()),
            "paired plan/receipt가 명시 probe output을 이탈했다")
    proof = {name: ref(path) for name, path in paths.items()}
    approval_path = probe_out / "root-approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    require(isinstance(approval, dict) and approval.get("schema") == HOST.APPROVAL_SCHEMA
            and approval.get("state_digest") == state["state_digest"]
            and approval.get("action") == action
            and approval.get("request_spec") == SEL.request_spec(state, action)
            and approval.get("plans") == {
                "case": proof["case_plan"], "reference": proof["reference_plan"]}
            and approval.get("approval_granted") is True,
            "root approval이 selector state/action/plan과 다르다")
    return {**proof, "root_approval": ref(approval_path)}


def _time(pair: dict[str, Any], name: str) -> float:
    value = pair.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("paired host 시간은 수치여야 한다")
    numeric = float(value)
    require(math.isfinite(numeric) and numeric >= 0, "paired host 시간은 확인된 비음수 수치여야 한다")
    return numeric


def run_epoch(
    initial_state: dict,
    *,
    run_id: str,
    out: Path,
    verify_context: Callable[[], object],
    complete_factory: Callable[[str, dict], Callable],
    execute_pair: Callable[[dict, dict, Path], dict],
) -> dict[str, Any]:
    """최대 네 모델 행동으로 option→기존 action→receipt→state 전이를 수행한다."""
    require(isinstance(run_id, str) and bool(run_id.strip()), "run_id가 없다")
    require(callable(verify_context) and callable(complete_factory) and callable(execute_pair),
            "검증·모델·paired host callback이 필요하다")
    out = Path(out).resolve()
    require(not out.exists(), "epoch out은 fresh 폴더여야 한다")
    verify_context()
    O.context(initial_state)
    require(initial_state["budget_left"] <= MAX_PAIRED_PROBES, "selector probe 예산 상한 초과")
    out.mkdir(parents=True, exist_ok=False)
    state = deepcopy(initial_state)
    steps: list[dict[str, Any]] = []
    model_actions = 0
    paired_probes = 0
    parent_wall_s = 0.0
    owned_child_wall_s = 0.0
    started = time.monotonic()
    stop_reason: str | None = None
    try:
        while state["status"] == "open" and state["budget_left"] > 0:
            if model_actions >= MAX_MODEL_ACTIONS:
                stop_reason = "model_action_budget_exhausted"
                break
            verify_context()
            O.context(state)
            model_actions += 1
            index = model_actions
            stage = out / f"step-{index:02d}"
            write_new(stage / "state-before.json", state)
            option_context = O.context(state)
            write_new(stage / "option-context.json", option_context)
            complete = complete_factory(f"{run_id}.d{index}", deepcopy(state))
            dialogue = O.dialogue(state, complete, verify_context=verify_context)
            write_new(stage / "dialogue.json", dialogue)
            action = dialogue["action"]
            selected_option = dialogue.get("final_review", {}).get("selected_option_id")
            selection = {
                "selected_by": "model_option_id",
                "selected_option_id": selected_option,
                "model_valid": dialogue["model_valid"],
                "state_digest": state["state_digest"],
                "option_context_digest": dialogue["context_digest"],
                "dialogue": ref(stage / "dialogue.json"),
                "action": deepcopy(action),
                "approval_granted": False,
            }
            write_new(stage / "selected-action.json", selection)
            before = deepcopy(state)
            if action["action"] == "run_probe":
                require(paired_probes < MAX_PAIRED_PROBES, "paired probe 상한 초과")
                request = SEL.request_spec(before, action)
                write_new(stage / "request-spec.json", request)
                verify_context()
                pair = execute_pair(before, action, stage / "probe")
                verify_context()
                proof = _pair_proof(pair, before, action, stage / "probe")
                write_new(stage / "paired-proof.json", proof)
                state = SEL.advance(
                    before, action,
                    case_plan=Path(pair["case_plan"]), case_receipt=Path(pair["case_receipt"]),
                    reference_plan=Path(pair["reference_plan"]),
                    reference_receipt=Path(pair["reference_receipt"]),
                )
                SEL.context(state)
                paired_probes += 1
                parent_wall_s += _time(pair, "parent_wall_s")
                owned_child_wall_s += _time(pair, "owned_child_wall_s")
                write_new(stage / "host-action.json", {
                    "action": deepcopy(action), "request_spec": request,
                    "root_approval": proof["root_approval"], "paired_proof": ref(stage / "paired-proof.json")})
            else:
                state = SEL.advance(before, action)
                SEL.context(state)
                write_new(stage / "host-action.json", {
                    "action": deepcopy(action), "request_spec": None,
                    "root_approval": None, "paired_proof": None})
            write_new(stage / "state-after.json", state)
            steps.append({
                "step": index, "selected_by": "model_option_id",
                "selected_option_id": selected_option, "model_valid": dialogue["model_valid"],
                "action": deepcopy(action), "state_before_digest": before["state_digest"],
                "state_after_digest": state["state_digest"],
                "dialogue": ref(stage / "dialogue.json"),
                "host_action": ref(stage / "host-action.json"),
                "state_after": ref(stage / "state-after.json"),
            })
        verify_context()
        SEL.context(state)
        write_new(out / "final-state.json", state)
        decision = "confirmed" if state["status"] == "confirmed" else "HOLD"
        if stop_reason is None and decision == "HOLD":
            stop_reason = ("model_invalid_hold" if state["status"] == "hold"
                           and steps and steps[-1]["model_valid"] is False
                           else state["status"])
        result = {
            "schema": SCHEMA, "status": "complete", "run_id": run_id,
            "selection_subject": "model_option_id", "model_actions": model_actions,
            "paired_probes": paired_probes, "decision_outcome": decision,
            "stop_reason": stop_reason, "final_status": state["status"],
            "confirmed_family": (state["hypotheses"][0]
                                 if state["status"] == "confirmed" else None),
            "steps": steps, "parent_probe_wall_s": parent_wall_s,
            "owned_child_wall_s": owned_child_wall_s,
            "epoch_wall_s": time.monotonic() - started,
            "automatic_repair": False,
            "claims": {"diagnosis_improvement": False, "recovery": False,
                       "new_probe_execution": paired_probes > 0},
        }
        write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        write_new(out / "failure.json", {
            "schema": SCHEMA, "status": "invalid", "run_id": run_id,
            "error_type": type(error).__name__, "error": str(error),
            "model_actions": model_actions, "paired_probes": paired_probes,
            "automatic_retry": False, "recovery": False,
        })
        raise


__all__ = ["MAX_MODEL_ACTIONS", "MAX_PAIRED_PROBES", "SCHEMA", "run_epoch"]
