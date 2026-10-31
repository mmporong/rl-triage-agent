"""근거 기반 selector의 단일 paired probe 요청을 기존 실행 경계에 연결한다."""
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
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import evidence_guided_selector_v1 as SEL
import lineage_diagnosis_epoch_v1 as E0
import prospective_diagnosis_inputs_v1 as INPUT
import recorded_optimizer_probe_v1 as RECORDED
import runtime_observation_probe_v1 as OBS
import runtime_observation_view_v1 as RV
import runtime_training_inputs_v1 as BASE
import selection_comparison_controller_v1 as C

SCHEMA = "evidence_guided_probe_host_result_v1"
APPROVAL_SCHEMA = "evidence_guided_probe_host_approval_v1"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, value: object) -> None:
    path = Path(path).resolve()
    _require(not path.exists(), f"{path.name}은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _source(path: Path, runtime: Path, role: str, spec: dict[str, Any]) -> tuple[Path, Path]:
    """Prospective envelope를 원래 registered-training 입력으로 되돌려 source를 대조한다."""
    public, metadata = INPUT.checked(Path(path).resolve())
    base_ref = metadata.get("base_collected")
    runtime_refs = metadata.get("runtime_facts")
    _require(isinstance(base_ref, dict) and isinstance(runtime_refs, dict),
             "prospective 입력의 base/runtime ref가 없다")
    base_ref = cast(dict[str, Any], base_ref)
    base_path = Path(base_ref["path"]).resolve()
    runtime_path = Path(runtime).resolve()
    runtime_refs = cast(dict[str, Any], runtime_refs)
    expected_runtime = runtime_refs.get(role)
    _require(isinstance(expected_runtime, dict)
             and runtime_path == Path(expected_runtime["path"]).resolve()
             and _sha(runtime_path) == expected_runtime.get("sha256"),
             f"{role} runtime artifact가 prospective 입력과 다르다")
    base, _base_metadata = BASE.checked(base_path)
    binding = spec["source_binding"].get(role)
    _require(isinstance(binding, dict)
             and public["source_binding"]["checkpoint_sha256"]
             == base["source_binding"]["checkpoint_sha256"]
             == binding.get("checkpoint_sha256"),
             f"{role} selector/input checkpoint가 다르다")
    binding = cast(dict[str, Any], binding)
    envelope, _facts = RECORDED.FACT.checked(
        runtime_path, expected_checkpoint_sha256=binding["checkpoint_sha256"])
    _require(envelope["source_binding"]["binding_digest"] == binding.get("binding_digest"),
             f"{role} selector/runtime binding이 다르다")
    return base_path, runtime_path


def _context(callback: Callable[[dict[str, Any]], object], *, phase: str, role: str,
             state: dict[str, Any], action: dict[str, Any], spec: dict[str, Any],
             plan: Path | None = None) -> dict[str, Any]:
    request = {
        "schema": "evidence_guided_probe_host_context_v1", "phase": phase, "role": role,
        "state_digest": state["state_digest"], "action": deepcopy(action),
        "request_spec": deepcopy(spec), "plan": None if plan is None else _ref(plan),
    }
    value = callback(deepcopy(request))
    _require(isinstance(value, dict) and value.get("approved") is True,
             f"{role} {phase} context 재검증이 승인되지 않았다")
    return cast(dict[str, Any], value)


def _approval(out: Path, state: dict[str, Any], action: dict[str, Any], spec: dict[str, Any],
              plans: dict[str, Path], approver: str) -> Path:
    _require(isinstance(approver, str) and bool(approver.strip()), "승인 주체가 없다")
    path = out / "root-approval.json"
    _write(path, {
        "schema": APPROVAL_SCHEMA, "actor": approver,
        "selector_contract": "new_evidence_guided_selector_not_legacy_model_dialogue",
        "state_digest": state["state_digest"], "action": deepcopy(action),
        "request_spec": deepcopy(spec),
        "plans": {role: _ref(plan) for role, plan in plans.items()},
        "approval_granted": True,
    })
    return path


def _recorded(
    state: dict[str, Any], action: dict[str, Any], spec: dict[str, Any], sources: dict[str, tuple[Path, Path]],
    out: Path, verify_context: Callable[[dict[str, Any]], object], approver: str,
) -> tuple[dict[str, Path], float]:
    plans: dict[str, Path] = {}
    request_ids = spec.get("request_ids")
    _require(isinstance(request_ids, dict) and set(request_ids) == {"case", "reference"},
             "recorded selector request ID 결합이 없다")
    request_ids = cast(dict[str, str], request_ids)
    for role in ("case", "reference"):
        _context(verify_context, phase="before_role", role=role, state=state, action=action, spec=spec)
        folder = out / role
        RECORDED.prepare(
            sources[role][0], sources[role][1], role=role,
            request_id=request_ids[role], params=RECORDED.PARAMS,
            ledger_path=folder / "ledger.jsonl", evidence_kind="actual_training_reuse",
            out=folder / "plan",
        )
        plans[role] = folder / "plan/plan.json"
    approval = _approval(out, state, action, spec, plans, approver)
    for role in ("case", "reference"):
        folder, plan = out / role, plans[role]
        context = _context(verify_context, phase="before_execution", role=role, state=state,
                           action=action, spec=spec, plan=plan)
        _require(context.get("approval_sha256", _sha(approval)) == _sha(approval),
                 "context의 root approval 결합이 다르다")
        auth = folder / "authorization.json"
        RECORDED.authorize(plan, approved_by=f"{approver}@{_sha(approval)[:16]}", out=auth)
        receipt = folder / "receipt.json"
        RECORDED.execute(plan, auth, out=receipt)
        RECORDED.checked_receipt(plan, receipt)
    return {
        "case_plan": plans["case"], "case_receipt": out / "case/receipt.json",
        "reference_plan": plans["reference"],
        "reference_receipt": out / "reference/receipt.json",
    }, 0.0


def _legacy_choice(snapshot: dict[str, Any], target: str) -> dict[str, Any]:
    checked = C.V.checked(snapshot)
    _require(target in checked["view"]["eligible_probes"],
             "new selector target이 legacy snapshot에서 실행 가능하지 않다")
    base = C.choose(checked, "registry_next")
    return {
        **base, "action": "continue", "probe": target, "selected_by": "registered_rule",
        "model_status": "not_used", "codes": [], "reason": "evidence_guided_host_normalized",
        "approval_granted": False,
    }


def _legacy(
    state: dict[str, Any], action: dict[str, Any], spec: dict[str, Any], sources: dict[str, tuple[Path, Path]],
    probe_protocol: Path, anchor: str, lab: Path, out: Path,
    verify_context: Callable[[dict[str, Any]], object], approver: str,
) -> tuple[dict[str, Path], float]:
    target = cast(str, action["target"])
    plans: dict[str, Path] = {}
    for role in ("case", "reference"):
        _context(verify_context, phase="before_role", role=role, state=state, action=action, spec=spec)
        stage = out / role
        OBS.prepare_probe(sources[role][0], target, stage=stage, lab=lab,
                          protocol_path=probe_protocol, anchor=anchor, role=role)
        plans[role] = stage / "plan.json"

    ledger_path = out / "probes.ledger.jsonl"
    snapshot = C.V.build_snapshot(
        state["initial_packet"], f"evidence-guided-{state['state_digest'][:16]}",
        "registry_next", C.V.file_digest(Path(C.V.PROTOCOL)), ledger_path,
    )
    choice = _legacy_choice(snapshot, target)
    legacy_spec = C.request_spec(snapshot, choice)
    _require(legacy_spec["probe"] == target and legacy_spec["args"] == spec["host_args"]["params"],
             "new selector host args와 legacy registry args가 다르다")
    approval = _approval(out, state, action, spec, plans, approver)
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    request = ledger.propose(legacy_spec["prereg"], target, int(legacy_spec["budget_gpu_s"]))
    ledger.approve(request["request_id"], f"{approver}@{_sha(approval)[:16]}")
    ledger.consume(request["request_id"], legacy_spec["prereg"], target)
    auth = out / "authorization.json"
    _write(auth, {
        "schema": OBS.AUTH_SCHEMA,
        "plan_sha256": {role: _sha(path) for role, path in plans.items()},
        "snapshot": snapshot, "choice": choice, "request_spec": legacy_spec,
        "ledger_path": str(ledger_path.resolve()), "request_id": request["request_id"],
    })

    receipts: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for role in ("reference", "case"):
        context = _context(verify_context, phase="before_execution", role=role, state=state,
                           action=action, spec=spec, plan=plans[role])
        _require(context.get("approval_sha256", _sha(approval)) == _sha(approval),
                 "context의 root approval 결합이 다르다")
        owned = context.get("owned")
        gpu_reader = context.get("gpu_reader")
        _require(owned is None or callable(owned), "owned context 값이 callable이 아니다")
        _require(gpu_reader is None or callable(gpu_reader), "gpu_reader context 값이 callable이 아니다")
        try:
            receipts[role] = OBS.execute_probe(
                plans[role], auth,
                owned=cast(Callable[..., object] | None, owned),
                gpu_reader=cast(Callable[..., object] | None, gpu_reader),
            )
        except Exception:
            receipt_path = plans[role].parent / "probe_receipt.json"
            if not receipt_path.is_file():
                raise
            receipts[role] = OBS.read(receipt_path)
    parent_wall = time.perf_counter() - started
    E0._settle(ledger, request["request_id"], target, receipts, parent_wall)
    for role in ("case", "reference"):
        OBS.verify_receipt(plans[role], plans[role].parent / "probe_receipt.json")
    owned_wall = sum(float(receipts[role].get("execution", {}).get("wall_s", 0.0))
                     for role in ("case", "reference"))
    _require(math.isfinite(owned_wall) and owned_wall >= 0, "owned child wall이 유한하지 않다")
    return {
        "case_plan": plans["case"], "case_receipt": plans["case"].parent / "probe_receipt.json",
        "reference_plan": plans["reference"],
        "reference_receipt": plans["reference"].parent / "probe_receipt.json",
    }, owned_wall


def execute_pair(
    state: dict[str, Any], action: dict[str, Any], *, case_input: Path, reference_input: Path,
    case_runtime: Path, reference_runtime: Path, probe_protocol: Path, view_protocol: Path,
    anchor: str, lab: Path, out: Path, verify_context: Callable[[dict[str, Any]], object],
    approver: str = "codex-main-user-delegated",
) -> dict[str, Any]:
    """검증된 selector 요청 하나를 독립 paired receipt 네 경로로 실행한다."""
    started = time.perf_counter()
    out = Path(out).resolve()
    _require(not out.exists(), "paired probe out은 fresh 경로여야 한다")
    _require(callable(verify_context), "verify_context callback이 필요하다")
    spec = SEL.request_spec(state, action)
    _require(action["action"] == "run_probe" and isinstance(action["target"], str),
             "host는 run_probe action만 실행한다")
    sources = {
        "case": _source(case_input, case_runtime, "case", spec),
        "reference": _source(reference_input, reference_runtime, "reference", spec),
    }
    out.mkdir(parents=True, exist_ok=False)
    with RV.view_scope(Path(view_protocol), anchor):
        if action["target"] == RECORDED.PROBE:
            paths, owned_wall = _recorded(
                state, action, spec, sources, out, verify_context, approver)
            scope = "recorded_verified_training_cpu_reuse"
        else:
            paths, owned_wall = _legacy(
                state, action, spec, sources, Path(probe_protocol), anchor, Path(lab).resolve(),
                out, verify_context, approver)
            scope = "legacy_registered_paired_probe"
    result = {
        "schema": SCHEMA, **{name: str(path.resolve()) for name, path in paths.items()},
        "parent_wall_s": time.perf_counter() - started, "owned_child_wall_s": owned_wall,
        "scope": scope,
    }
    _write(out / "host-result.json", result)
    return deepcopy(result)
