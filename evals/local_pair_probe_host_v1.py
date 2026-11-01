"""Public 48-fact 선택을 runtime-view 64-fact OBS paired probe에 단회 연결한다."""
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
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import lineage_diagnosis_epoch_v1 as DIAG
import runtime_observation_probe_v1 as OBS
import runtime_observation_view_v1 as RV
import selection_comparison_controller_v1 as C

SCHEMA = "local_pair_probe_host_result_v1"
AUTH_SCHEMA = OBS.AUTH_SCHEMA
ROLES = ("reference", "case")
SOURCE_FILES = (
    "evals/local_pair_probe_host_v1.py",
    "tests/test_local_pair_probe_host_v1.py",
    "evals/runtime_observation_probe_v1.py",
    "evals/runtime_observation_view_v1.py",
    "evals/lineage_diagnosis_epoch_v1.py",
    "evals/selection_comparison_controller_v1.py",
    "evals/selection_comparison_view_v1.py",
    "src/rl_triage/mechanism_registry.py",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")))
    require(isinstance(value, dict), "JSON object required")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: object) -> str:
    return C.L.digest(value)


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def file_ref(path: Path) -> dict[str, str]:
    checked = Path(path).resolve(strict=True)
    require(checked.is_file() and not checked.is_symlink(), "unsafe file ref")
    return {"path": str(checked), "sha256": sha(checked)}


def _packets(public: dict[str, Any], backend: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    public_checked = C.V.S.validate_packet(public)
    backend_checked = C.V.S.validate_packet(backend)
    public_facts, backend_facts = public_checked["facts"], backend_checked["facts"]
    runtime = {key: value for key, value in backend_facts.items() if key.startswith("runtime:")}
    common = {key: value for key, value in backend_facts.items() if not key.startswith("runtime:")}
    require(len(public_facts) == 48 and common == public_facts and len(runtime) == 16,
            "public48/backend64 fact projection")
    public_without_facts = {key: value for key, value in public_checked.items() if key != "facts"}
    backend_without_facts = {key: value for key, value in backend_checked.items() if key != "facts"}
    require(public_without_facts == backend_without_facts, "public/backend packet scope differs")
    return public_checked, backend_checked


def _adapter_choice(snapshot: dict[str, Any], external_choice: dict[str, Any], probe: str) -> dict[str, Any]:
    return {"schema": "selection_controller_choice_v1",
            "context_digest": digest(snapshot["binding"]), "view_digest": snapshot["view_digest"],
            "policy": snapshot["inputs"]["policy"], "selected_by": "registered_rule",
            "action": "continue", "probe": probe, "fact_ids": [], "model_status": "not_used",
            "codes": [], "reason": "external_public_choice_adapter",
            "used_for_decision": True, "approval_granted": False,
            "external_selected_by": external_choice.get("selected_by")}


def _approval(callback: Callable, request: dict[str, Any]) -> dict[str, Any]:
    value = callback(deepcopy(request))
    require(isinstance(value, dict) and value.get("approved") is True, "host approval denied")
    require(set(value) <= {"approved", "owned", "gpu_reader"}, "host approval response fields")
    return value


def _ledger_start(path: Path, spec: dict[str, Any], approver: str, *, fresh: bool) -> tuple[Any, str]:
    ledger = C.S.M.RegistryLedger(path, C.RC.registry())
    requests = ledger.requests()
    if fresh:
        require(not ledger.events() and not requests, "backend probe ledger must start empty")
    else:
        require(all(row.get("state") == "done" for row in requests.values()),
                "public probe ledger has unfinished prefix")
    request = ledger.propose(spec["prereg"], spec["probe"], int(spec["budget_gpu_s"]))
    ledger.approve(request["request_id"], approver)
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    return ledger, request["request_id"]


def _prefix_proof(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected is None:
        prefix = raw
    else:
        length = expected.get("byte_length")
        require(type(length) is int and 0 <= length <= len(raw), "ledger prefix length")
        prefix = raw[:length]
        require(hashlib.sha256(prefix).hexdigest() == expected.get("sha256"),
                "ledger prefix changed")
    return {"byte_length": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest()}


def _done(path: Path, binding: dict[str, Any], packet: dict[str, Any], request_id: str,
          spec: dict[str, Any], *, prior: list[dict[str, Any]], outcome: str,
          allow_suffix: bool, prefix_proof: dict[str, Any] | None = None) -> dict[str, Any]:
    rows, _ledger_digest, pending = C.V.read_history(path, binding, packet["execution_budget_s"])
    ledger = C.S.M.RegistryLedger(path, C.RC.registry())
    request = ledger.requests().get(request_id, {})
    events = [row for row in ledger.events() if row.get("request_id") == request_id]
    public_rows = [{"probe": row["probe"], "outcome": row["outcome"]} for row in rows]
    expected_rows = [*deepcopy(prior), {"probe": spec["probe"], "outcome": outcome}]
    require(pending is None and (public_rows[:len(expected_rows)] == expected_rows)
            and (allow_suffix or len(public_rows) == len(expected_rows))
            and request.get("probe") == spec["probe"]
            and request.get("state") == "done"
            and request.get("prereg_digest") == digest(spec["prereg"])
            and request.get("args_digest") == digest(spec["args"])
            and [row.get("event") for row in events]
            == ["proposed", "approved", "consumed", "receipt"], "ledger not single done request")
    proof = _prefix_proof(path, prefix_proof)
    return {"path": str(path.resolve()), "prefix": proof, "request_id": request_id,
            "observation_index": len(prior), "receipt": deepcopy(events[-1])}


def _owned_child_wall(receipts: dict[str, dict[str, Any]]) -> tuple[float | None, float, list[str]]:
    values: list[float] = []
    unknown: list[str] = []
    for role in ROLES:
        value = receipts[role].get("execution", {}).get("wall_s")
        if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
            unknown.append(role)
        else:
            values.append(float(value))
    known = sum(values)
    return (known if not unknown else None), known, unknown


def execute_pair(snapshot: dict, choice: dict, *, backend_packet: dict,
                 case_input: Path, reference_input: Path, probe_protocol: Path,
                 view_protocol: Path, anchor: str, lab: Path, out: Path,
                 verify_context: Callable, approver: str = "codex-main-user-delegated",
                 owned: Any = None, gpu_reader: Any = None) -> dict[str, Any]:
    """승인된 public 선택 하나를 OBS case/reference에 실행하고 두 원장을 한 번 정산한다."""
    require(isinstance(approver, str) and bool(approver.strip()), "approver")
    frozen = C.V.checked(snapshot)
    public_spec = C.request_spec(frozen, choice)
    public_packet, checked_backend = _packets(frozen["inputs"]["packet"], backend_packet)
    out = Path(out).resolve()
    require(not out.exists(), "host output must be fresh")
    out.mkdir(parents=True)
    backend_ledger_path = out / "backend/probes.ledger.jsonl"
    public_ledger_path = Path(frozen["inputs"]["ledger_path"]).resolve()
    with RV.view_scope(view_protocol, anchor):
        backend_snapshot = C.V.build_snapshot(
            checked_backend, "local_pair_backend_" + digest(frozen["binding"])[:12],
            frozen["inputs"]["policy"], C.V.file_digest(view_protocol), backend_ledger_path)
        backend_choice = _adapter_choice(backend_snapshot, choice, public_spec["probe"])
        backend_spec = C.request_spec(backend_snapshot, backend_choice)
        require({key: backend_spec[key] for key in ("probe", "args", "checkpoint_sha256",
                                                    "reference_checkpoint_sha256", "budget_gpu_s")}
                == {key: public_spec[key] for key in ("probe", "args", "checkpoint_sha256",
                                                     "reference_checkpoint_sha256", "budget_gpu_s")},
                "public/backend request spec differs")
        plans: dict[str, dict[str, Any]] = {}
        plan_paths: dict[str, Path] = {}
        inputs = {"case": Path(case_input), "reference": Path(reference_input)}
        for role in ROLES:
            stage = out / "probe" / role
            request = {"phase": "prepare", "role": role, "probe": public_spec["probe"],
                       "public_choice_sha256": digest(choice),
                       "backend_spec_sha256": digest(backend_spec)}
            _approval(verify_context, request)
            plans[role] = OBS.prepare_probe(inputs[role], public_spec["probe"], stage=stage,
                                            lab=Path(lab), protocol_path=Path(probe_protocol),
                                            anchor=anchor, role=role)
            plan_paths[role] = stage / "plan.json"
            require(plan_paths[role].is_file(), "OBS plan missing")
        backend_ledger, backend_request_id = _ledger_start(
            backend_ledger_path, backend_spec, approver, fresh=True)
        public_ledger, public_request_id = _ledger_start(
            public_ledger_path, public_spec, approver, fresh=False)
        authorization = {"schema": AUTH_SCHEMA,
                         "plan_sha256": {role: sha(path) for role, path in plan_paths.items()},
                         "snapshot": backend_snapshot, "choice": backend_choice,
                         "request_spec": backend_spec, "ledger_path": str(backend_ledger_path),
                         "request_id": backend_request_id}
        authorization_path = out / "probe/authorization.json"
        write_new(authorization_path, authorization)
        receipts: dict[str, dict[str, Any]] = {}
        receipt_refs: dict[str, dict[str, str] | None] = {}
        started = time.perf_counter()
        for role in ROLES:
            response = _approval(verify_context, {
                "phase": "execute", "role": role, "probe": public_spec["probe"],
                "plan_sha256": sha(plan_paths[role]), "authorization_sha256": sha(authorization_path),
                "public_choice_sha256": digest(choice), "backend_spec_sha256": digest(backend_spec)})
            role_owned = response.get("owned", owned)
            role_gpu_reader = response.get("gpu_reader", gpu_reader)
            try:
                receipts[role] = OBS.execute_probe(plan_paths[role], authorization_path,
                                                   owned=role_owned, gpu_reader=role_gpu_reader)
                receipt_path = plan_paths[role].parent / "probe_receipt.json"
                require(receipt_path.is_file(), "OBS receipt missing")
                receipt_refs[role] = file_ref(receipt_path)
            except Exception as error:  # noqa: BLE001 - no retry; outer unknown is settled once.
                receipts[role] = {"status": "unknown", "measurement": None,
                                  "execution": {}, "role": role,
                                  "error_type": type(error).__name__}
                receipt_refs[role] = None
        parent_wall = time.perf_counter() - started
        backend_outcome, backend_receipt = DIAG._settle(
            backend_ledger, backend_request_id, public_spec["probe"], receipts, parent_wall)
        backend_done = _done(
            backend_ledger_path, backend_snapshot["binding"], checked_backend,
            backend_request_id, backend_spec, prior=[], outcome=backend_outcome,
            allow_suffix=False)
        verified_receipts = {}
        for role in ROLES:
            if receipt_refs[role] is None:
                verified_receipts[role] = receipts[role]
            else:
                receipt_ref = receipt_refs[role]
                if receipt_ref is None:
                    raise AssertionError("unreachable receipt branch")
                verified_receipts[role] = OBS.verify_receipt(
                    plan_paths[role], Path(receipt_ref["path"]))
        verified_unknown = any(row.get("status") != "complete"
                               or not isinstance(row.get("measurement"), dict)
                               for row in verified_receipts.values())
        require((backend_outcome == "unknown") == verified_unknown,
                "verified OBS receipts differ from backend settlement")
        public_outcome, public_receipt = DIAG._settle(
            public_ledger, public_request_id, public_spec["probe"], verified_receipts, parent_wall)
        require(backend_outcome == public_outcome, "public/backend settlement differs")
        public_done = _done(
            public_ledger_path, frozen["binding"], public_packet, public_request_id, public_spec,
            prior=frozen["view"]["observations"], outcome=public_outcome, allow_suffix=False)
    owned_child_wall, owned_child_known_wall, owned_child_unknown_roles = _owned_child_wall(
        verified_receipts)
    result = {"schema": SCHEMA,
              "status": "complete" if backend_outcome != "unknown" else "unknown_hold",
              "outcome": backend_outcome, "probe": public_spec["probe"],
              "evidence_kind": "actual_runtime_observation",
              "request_id": public_request_id, "backend_request_id": backend_request_id,
              "public_snapshot_sha256": digest(frozen), "public_choice": deepcopy(choice),
              "public_choice_sha256": digest(choice), "public_selected_by": choice.get("selected_by"),
              "backend_packet_sha256": digest(checked_backend),
              "backend_choice": backend_choice, "backend_choice_role": "registered_rule_adapter_not_model",
              "request_spec_sha256": digest(public_spec), "authorization": file_ref(authorization_path),
              "probe_protocol": file_ref(Path(probe_protocol)),
              "view_protocol": file_ref(Path(view_protocol)), "view_anchor": anchor,
              "plans": {role: file_ref(path) for role, path in plan_paths.items()},
              "receipts": receipt_refs, "outer_unknown_receipts": {
                  role: deepcopy(receipts[role]) if receipt_refs[role] is None else None for role in ROLES},
              "public_ledger": public_done, "backend_ledger": backend_done,
              "public_ledger_receipt": public_receipt, "backend_ledger_receipt": backend_receipt,
              "parent_wall_s": parent_wall, "owned_child_wall_s": owned_child_wall,
              "owned_child_known_wall_s": owned_child_known_wall,
              "owned_child_unknown_roles": owned_child_unknown_roles, "retry_count": 0,
              "correction_executed": False, "recovery_claimed": False}
    result_path = out / "result.json"
    write_new(result_path, result)
    returned = deepcopy(result)
    returned["result_ref"] = file_ref(result_path)
    return returned


def verify_pair(result_path: Path, snapshot: dict, choice: dict) -> dict[str, Any]:
    """현재 raw files와 두 done ledger에서 paired host 결과를 read-only 재검증한다."""
    result_path = Path(result_path).resolve(strict=True)
    result = read(result_path)
    require(result.get("schema") == SCHEMA and result.get("status") in {"complete", "unknown_hold"}
            and result.get("public_snapshot_sha256") == digest(snapshot)
            and result.get("public_choice") == choice
            and result.get("public_choice_sha256") == digest(choice)
            and result.get("retry_count") == 0
            and result.get("evidence_kind") == "actual_runtime_observation",
            "host result/public choice binding")
    public_spec = C._frozen_request_spec(snapshot, choice)
    require(result.get("probe") == public_spec["probe"]
            and result.get("request_spec_sha256") == digest(public_spec), "host request spec")
    authorization_ref = result.get("authorization")
    if not isinstance(authorization_ref, dict):
        raise TypeError("authorization ref")
    authorization_path = Path(authorization_ref["path"]).resolve(strict=True)
    require(file_ref(authorization_path) == authorization_ref, "authorization SHA")
    authorization = read(authorization_path)
    backend_snapshot, backend_choice = authorization["snapshot"], authorization["choice"]
    require(result.get("backend_choice") == backend_choice
            and result.get("backend_choice_role") == "registered_rule_adapter_not_model"
            and backend_choice.get("selected_by") == "registered_rule"
            and backend_choice.get("external_selected_by") == choice.get("selected_by"),
            "backend adapter provenance")
    probe_protocol_ref, view_protocol_ref = result.get("probe_protocol"), result.get("view_protocol")
    if not isinstance(probe_protocol_ref, dict) or not isinstance(view_protocol_ref, dict):
        raise TypeError("protocol refs")
    require(file_ref(Path(probe_protocol_ref["path"])) == probe_protocol_ref
            and file_ref(Path(view_protocol_ref["path"])) == view_protocol_ref,
            "protocol SHA")
    verified_receipts = {}
    for role in ROLES:
        plan_ref = result["plans"][role]
        plan_path = Path(plan_ref["path"]).resolve(strict=True)
        require(file_ref(plan_path) == plan_ref, f"{role} plan SHA")
        receipt_ref = result["receipts"][role]
        if receipt_ref is None:
            unknown = result["outer_unknown_receipts"][role]
            require(isinstance(unknown, dict) and unknown.get("status") == "unknown",
                    f"{role} explicit unknown")
            verified_receipts[role] = unknown
        else:
            if not isinstance(receipt_ref, dict):
                raise ValueError(f"{role} receipt ref")
            receipt_path = Path(receipt_ref["path"]).resolve(strict=True)
            require(file_ref(receipt_path) == receipt_ref, f"{role} receipt SHA")
            verified_receipts[role] = OBS.verify_receipt(plan_path, receipt_path)
    backend_spec = C._frozen_request_spec(backend_snapshot, backend_choice)
    public_done = _done(
        Path(result["public_ledger"]["path"]), snapshot["binding"],
        snapshot["inputs"]["packet"], result["public_ledger"]["request_id"], public_spec,
        prior=snapshot["view"]["observations"], outcome=result["outcome"], allow_suffix=True,
        prefix_proof=result["public_ledger"]["prefix"])
    with RV.view_scope(Path(view_protocol_ref["path"]), result.get("view_anchor")):
        backend_done = _done(
            Path(result["backend_ledger"]["path"]), backend_snapshot["binding"],
            backend_snapshot["inputs"]["packet"], result["backend_ledger"]["request_id"],
            backend_spec, prior=[], outcome=result["outcome"], allow_suffix=False,
            prefix_proof=result["backend_ledger"]["prefix"])
    require(public_done == result["public_ledger"] and backend_done == result["backend_ledger"],
            "current done ledger refs")
    expected_unknown = any(row.get("status") != "complete"
                           or not isinstance(row.get("measurement"), dict)
                           for row in verified_receipts.values())
    require((result["outcome"] == "unknown") == expected_unknown
            and result["status"] == ("unknown_hold" if expected_unknown else "complete"),
            "paired outcome/status")
    current = C.V.build_snapshot(**snapshot["inputs"])
    observations = current.get("view", {}).get("observations", [])
    observation_index = result["public_ledger"].get("observation_index")
    require(type(observation_index) is int and observation_index < len(observations)
            and observations[observation_index] == {
                "probe": result["probe"], "outcome": result["outcome"]},
            "public ledger bound observation differs")
    require(result.get("request_id") == public_done["request_id"]
            and result.get("backend_request_id") == backend_done["request_id"],
            "request identifiers")
    owned_child_wall, known_wall, unknown_roles = _owned_child_wall(verified_receipts)
    require(result.get("owned_child_wall_s") == owned_child_wall
            and result.get("owned_child_known_wall_s") == known_wall
            and result.get("owned_child_unknown_roles") == unknown_roles,
            "owned child wall recomputation")
    verified = deepcopy(result)
    verified["result_ref"] = file_ref(result_path)
    return verified
