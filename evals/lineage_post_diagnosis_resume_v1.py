"""완료된 모델 응답 뒤 중단된 epoch 2를 세 번째 probe부터 재개한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_diagnosis_epoch_v1 as E0
import lineage_nim_transport_v1 as T
import lineage_post_diagnosis_epoch_v1 as POST_EPOCH
import lineage_post_diagnosis_epoch_v3 as E3
import lineage_post_parent_binding_v1 as PARENT
import lineage_post_probe_backend_v1 as POST_BACKEND
import lineage_post_recovery_candidate_v1 as CANDIDATE2
import lineage_recovery_candidate_v1 as CANDIDATE1
import selection_comparison_controller_v1 as C

PROTOCOL_SCHEMA = "lineage_post_diagnosis_resume_protocol_v1"
PLAN_SCHEMA = "lineage_post_diagnosis_resume_plan_v1"
RESULT_SCHEMA = "lineage_post_diagnosis_resume_result_v1"
SOURCE_FILES = (
    "evals/lineage_post_diagnosis_resume_v1.py",
    "tests/test_lineage_post_diagnosis_resume_v1.py",
    "evals/lineage_post_diagnosis_epoch_v3.py",
    "evals/lineage_post_parent_binding_v1.py",
    "evals/lineage_post_probe_backend_v1.py",
    "evals/lineage_post_recovery_candidate_v1.py",
    "evals/lineage_recovery_candidate_v1.py",
    "evals/lineage_nim_transport_v1.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
ROLES = ("case", "reference")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("protocol은 저장소 파일이어야 한다") from exc


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def prepare_protocol(parent_v3_protocol: Path) -> dict:
    parent = Path(parent_v3_protocol).resolve()
    _require(_read(parent).get("schema") == E3.PROTOCOL_SCHEMA,
             "v3 parent protocol schema가 다르다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "component_v3_protocol": {"path": _relative(parent), "sha256_lf": _sha(parent, lf=True)},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "scope": {
            "resume_reason": "recorded_response_completed_after_host_file_race",
            "reused_model_responses": 3,
            "reused_completed_logical_probes": 2,
            "new_logical_probe_cap": 1,
            "fresh_api_requests": 0,
            "correction_execution": False,
            "second_training": False,
        },
    }


def verify_protocol(protocol_path: Path, *, lab: Path, anchor: str) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "resume anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    component = value.get("component_v3_protocol")
    _require(isinstance(component, dict) and set(component) == {"path", "sha256_lf"},
             "v3 parent protocol 결합이 없다")
    parent = ROOT / str(component["path"])
    _require(value == prepare_protocol(parent), "resume protocol 내용 또는 SHA가 다르다")
    E3.verify_protocol(parent, lab=Path(lab), anchor=anchor)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), component["path"]: component["sha256_lf"],
             **value["source_sha256_lf"]}
    for relative, expected in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected,
                 f"현재 resume source/blob이 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected,
                 f"resume anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def _recorded_dialogue(snapshot: dict, queue: Path, slot_id: str) -> tuple[dict, dict, dict]:
    inputs = C.dialogue_input(snapshot, "production")
    _require(inputs["needs_model"], "기록 응답을 소비할 model dialogue가 필요하지 않다")
    queue = Path(queue).resolve()
    session = T._session(queue)
    matches = []
    for request_path in sorted((queue / "requests").glob(f"{slot_id}-*.json")):
        request = _read(request_path)
        if request.get("slot_id") == slot_id:
            matches.append((request_path, request))
    _require(len(matches) == 1, "logical slot의 기록 request가 없거나 중복됐다")
    request_path, request = matches[0]
    request_id = request_path.stem
    _require(request.get("schema") == T.REQUEST_SCHEMA
             and request.get("messages") == inputs["messages"]
             and request.get("params") == inputs["params"]
             and request.get("context_digest") == C.L.digest(snapshot["binding"])
             and request.get("view_digest") == snapshot["view_digest"],
             "기록 request가 현재 snapshot/messages/params와 다르다")
    dispatch_path = queue / "dispatches" / request_path.name
    response_path = queue / "responses" / request_path.name
    sha_path = queue / "response-sha" / f"{request_id}.sha256"
    dispatch, response, response_sha = _read(dispatch_path), _read(response_path), _read(sha_path)
    _require(dispatch.get("request_id") == request_id
             and dispatch.get("request_sha256") == _sha(request_path),
             "기록 dispatch가 request와 다르다")
    _require(response_sha == {"request_id": request_id, "response_sha256": _sha(response_path)}
             and response.get("schema") == T.RESPONSE_SCHEMA
             and response.get("request_id") == request_id
             and response.get("request_sha256") == _sha(request_path)
             and response.get("dispatch_sha256") == _sha(dispatch_path),
             "기록 response 또는 SHA가 request/dispatch와 다르다")
    result = T._validate_result(response.get("result"), session)
    calls = 0

    def complete(messages: list[dict], params: dict) -> dict:
        nonlocal calls
        calls += 1
        _require(calls == 1 and messages == request["messages"] and params == request["params"],
                 "recorded NIM callback 입력 또는 호출 수가 다르다")
        return deepcopy(result)

    engine = C.CALL.engine(snapshot, inputs["view"], inputs["messages"], "production", complete)
    _require(calls == 1 and engine.get("requests") == 1, "recorded NIM 응답은 한 번만 소비해야 한다")
    dialogue = {
        "schema": "selection_controller_dialogue_v1",
        "input_digest": C.L.digest(inputs),
        "safe_results": [C.CALL.safe_result(row["result"]) for row in engine["calls"]],
        "engine": engine,
    }
    choice = C.choose(snapshot, "model_adaptive", dialogue)
    return choice, engine, {
        "request": _ref(request_path), "dispatch": _ref(dispatch_path),
        "response": _ref(response_path), "response_sha": _ref(sha_path),
        "provider_model": result["provider_model"], "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"], "total_tokens": result["total_tokens"],
        "model_wall_s": result["wall_s"], "transport_wall_s": response["transport_wall_s"],
    }


def _validate_prefix(old_stage: Path, queue: Path, post_input: Path, reference_input: Path,
                     packet: dict, binding: dict, run_id: str, policy: str) -> tuple[list[dict], list[dict], list[dict]]:
    ledger_path = old_stage / "probes.ledger.jsonl"
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    events, requests = ledger.events(), ledger.requests()
    _require(len(events) == 8 and len(requests) == 2
             and all(row.get("state") == "done" for row in requests.values()),
             "기존 prefix는 완료된 두 probe와 8개 event여야 한다")
    rows: list[dict] = []
    prefix_events: list[dict] = []
    evidence: list[dict] = []
    model_evidence: list[dict] = []
    with CANDIDATE2.typed_backend_scope():
        for decision, expected_probe in enumerate(("P_episode", "P_noise"), 1):
            step_root = old_stage / "probes" / f"step_{decision:02d}"
            auth_path = step_root / "authorization.json"
            auth = _read(auth_path)
            snapshot, choice = auth.get("snapshot"), auth.get("choice")
            _require(isinstance(snapshot, dict) and isinstance(choice, dict),
                     "기존 authorization snapshot/choice가 객체가 아니다")
            CANDIDATE1._snapshot_at(snapshot, prefix_events, rows)
            _require(snapshot["inputs"]["packet"] == packet
                     and snapshot["inputs"]["run_id"] == run_id
                     and snapshot["inputs"]["policy"] == policy
                     and Path(snapshot["inputs"]["ledger_path"]).resolve() == ledger_path.resolve(),
                     "기존 authorization packet/run/policy/ledger가 다르다")
            with tempfile.TemporaryDirectory(prefix="lineage-resume-prefix-") as temporary:
                prefix_ledger = Path(temporary) / "ledger.jsonl"
                prefix_ledger.write_text("".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in prefix_events), encoding="utf-8", newline="\n")
                replay_snapshot = C.V.build_snapshot(
                    packet, run_id, policy, snapshot["inputs"]["protocol_sha256"], prefix_ledger)
                _require(replay_snapshot["binding"] == snapshot["binding"]
                         and replay_snapshot["view"] == snapshot["view"],
                         "prefix 별칭 snapshot의 의미가 기존 authorization과 다르다")
                recorded_choice, _engine, model_ref = _recorded_dialogue(
                    replay_snapshot, queue, f"case-07.model.{decision}")
            _require(recorded_choice == choice and choice.get("probe") == expected_probe,
                     "기존 모델 선택과 authorization 선택이 다르다")
            spec = C._frozen_request_spec(snapshot, choice)
            request_id = auth.get("request_id")
            _require(isinstance(request_id, str) and request_id in requests,
                     "기존 request ID가 ledger에 없다")
            request_events = [row for row in events if row.get("request_id") == request_id]
            CANDIDATE1._done_authorization(
                auth_path, _sha(auth_path), snapshot, choice, spec, request_id,
                ledger_path.resolve(), requests[request_id], request_events,
            )
            paired, measurements, role_refs = {}, {}, {}
            for role, input_path in (("case", post_input), ("reference", reference_input)):
                plan_path = step_root / role / "plan.json"
                plan = POST_BACKEND._verify_plan(plan_path)
                receipt = _read(step_root / role / "probe_receipt.json")
                _require(auth["plan_sha256"][role] == _sha(plan_path)
                         and plan["probe"] == expected_probe and plan["role"] == role,
                         "기존 paired plan/auth/probe 결합이 다르다")
                measurement, role_ref = CANDIDATE1._receipt(
                    receipt, role=role, plan_path=plan_path, plan=plan,
                    auth_path=auth_path, auth_sha=_sha(auth_path), request_id=request_id,
                    collected_input=input_path, source_binding=binding[role],
                )
                paired[role], measurements[role], role_refs[role] = receipt, measurement, role_ref
            outcome = C.classify_raw(expected_probe, measurements["case"], measurements["reference"])
            _require(outcome == "normal" == requests[request_id]["receipt"]["outcome"],
                     "기존 paired raw 재분류는 normal이어야 한다")
            rows.append({"probe": expected_probe, "outcome": outcome,
                         "measurement": measurements["case"],
                         "gpu_s": requests[request_id]["receipt"]["gpu_s"]})
            prefix_events.extend(request_events)
            evidence.append({"decision": decision, "probe": expected_probe,
                             "authorization": _ref(auth_path), "roles": role_refs})
            model_evidence.append(model_ref)
    return rows, evidence, model_evidence


def _prepare_value(*, old_stage: Path, transport_root: Path, operator_failure: Path,
                   post_input: Path, reference_input: Path, gate_path: Path,
                   gate_sha256: str, lab: Path, protocol_path: Path, anchor: str,
                   run_id: str, policy: str) -> dict:
    _require(policy == "model_adaptive", "resume은 model_adaptive epoch2만 허용한다")
    verify_protocol(protocol_path, lab=lab, anchor=anchor)
    old_stage, transport_root = Path(old_stage).resolve(), Path(transport_root).resolve()
    failure = _read(operator_failure)
    _require(failure.get("status") == "invalid" and failure.get("automatic_retry") is False
             and failure.get("error") == "epoch2 exit=1", "기존 operator failure 결합이 다르다")
    closed, worker = _read(transport_root / "closed.json"), _read(transport_root / "worker/result.json")
    _require(closed.get("schema") == "lineage_nim_transport_closed_v1"
             and worker.get("status") == "closed" and worker.get("posts") == 3
             and worker.get("queue_status", {}).get("responses") == 3
             and worker.get("queue_status", {}).get("pending") == 0,
             "기록 NIM queue가 3응답 완료·closed 상태가 아니다")
    _gate, gate_case = E0._gate(gate_path, gate_sha256, "case-07")
    _require(gate_case.get("arm") == "COMPOUND" and gate_case.get("diagnosis_eligible") is True,
             "resume 대상은 eligible COMPOUND case-07이어야 한다")
    envelope = POST_EPOCH.POST.make_selection_packet(
        post_input, reference_input, budget_left=3, execution_budget_s=1200)
    binding = POST_EPOCH._checked_post_binding(envelope, gate_case)
    packet = envelope["packet"]
    with PARENT.parent_scope(post_input, reference_input, run_id, policy):
        rows, prefix, model_prefix = _validate_prefix(
            old_stage, transport_root, post_input, reference_input,
            packet, binding, run_id, policy,
        )
        snapshot = C.V.build_snapshot(packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL),
                                      old_stage / "probes.ledger.jsonl")
        _require([row["probe"] for row in snapshot["view"]["observations"]]
                 == ["P_episode", "P_noise"], "세 번째 선택 전 관측 prefix가 다르다")
        choice, engine, model_third = _recorded_dialogue(snapshot, transport_root,
                                                         "case-07.model.3")
        spec = C.request_spec(snapshot, choice)
    _require(choice.get("probe") == spec.get("probe") == "P_physics"
             and choice.get("selected_by") == "model" and choice.get("model_status") == "valid",
             "기록된 세 번째 선택은 유효한 P_physics여야 한다")
    return {
        "schema": PLAN_SCHEMA, "status": "prepared", "case_id": "case-07",
        "run_id": run_id, "policy": policy, "anchor": anchor,
        "protocol": _ref(protocol_path), "lab": str(Path(lab).resolve()),
        "inputs": {"post": _ref(post_input), "reference": _ref(reference_input),
                   "gate": _ref(gate_path), "gate_sha256": gate_sha256},
        "old_execution": {"stage": str(old_stage), "ledger": _ref(old_stage / "probes.ledger.jsonl"),
                          "operator_failure": _ref(operator_failure),
                          "transport_root": str(transport_root), "closed": _ref(transport_root / "closed.json"),
                          "worker_result": _ref(transport_root / "worker/result.json")},
        "packet_digest": C.L.digest(packet), "binding": binding,
        "prefix_rows": rows, "prefix_steps": prefix,
        "recorded_model": {"first_two": model_prefix, "third": model_third,
                           "third_engine": engine, "fresh_api_requests": 0},
        "third_choice": choice, "third_request_spec": spec,
        "execution_contract": {"newly_executed_logical_probe": 1, "total_logical_probes": 3,
                               "epoch2_diagnosis_only": True, "correction_execution": False,
                               "second_training": False, "actual_backend_reused_recorded_nim": True},
    }


def prepare_resume(*, old_stage: Path, transport_root: Path, operator_failure: Path,
                   post_input: Path, reference_input: Path, gate_path: Path,
                   gate_sha256: str, lab: Path, protocol_path: Path, anchor: str,
                   run_id: str, policy: str, out: Path) -> dict:
    out = Path(out).resolve()
    _require(not out.exists(), "resume plan 출력은 fresh 파일이어야 한다")
    value = _prepare_value(old_stage=old_stage, transport_root=transport_root,
                           operator_failure=operator_failure, post_input=post_input,
                           reference_input=reference_input, gate_path=gate_path,
                           gate_sha256=gate_sha256, lab=lab, protocol_path=protocol_path,
                           anchor=anchor, run_id=run_id, policy=policy)
    _write_new(out, value)
    return deepcopy(value)


def _assert_ref(ref: dict) -> Path:
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "파일 ref 형식이 다르다")
    path = Path(ref["path"]).resolve()
    _require(path.is_file() and _sha(path) == ref["sha256"], "파일 ref SHA가 바뀌었다")
    return path


def _execution_scope(probe_executor: Callable | None) -> tuple[str, bool]:
    return (("actual_backend", True) if probe_executor is None else ("cpu_fixture", False))


def execute_resume(plan_path: Path, *, out: Path, probe_protocol: Path,
                   approver: str = "codex-main-user-delegated",
                   probe_executor: Callable | None = None) -> dict:
    plan_path = Path(plan_path).resolve()
    plan = _read(plan_path)
    _require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "prepared",
             "검증된 resume plan이 아니다")
    post_input = _assert_ref(plan["inputs"]["post"])
    reference_input = _assert_ref(plan["inputs"]["reference"])
    gate_path = _assert_ref(plan["inputs"]["gate"])
    protocol_path = _assert_ref(plan["protocol"])
    ledger_source = _assert_ref(plan["old_execution"]["ledger"])
    operator_failure = _assert_ref(plan["old_execution"]["operator_failure"])
    current = _prepare_value(
        old_stage=Path(plan["old_execution"]["stage"]),
        transport_root=Path(plan["old_execution"]["transport_root"]),
        operator_failure=operator_failure, post_input=post_input, reference_input=reference_input,
        gate_path=gate_path, gate_sha256=plan["inputs"]["gate_sha256"],
        lab=Path(plan["lab"]), protocol_path=protocol_path, anchor=plan["anchor"],
        run_id=plan["run_id"], policy=plan["policy"],
    )
    _require(current == plan, "resume plan 재검증 결과가 다르다")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out / "probes.ledger.jsonl"
    ledger_path.write_bytes(ledger_source.read_bytes())
    packet = POST_EPOCH.POST.make_selection_packet(
        post_input, reference_input, budget_left=3, execution_budget_s=1200)["packet"]
    evidence_scope, reused_actual_nim = _execution_scope(probe_executor)
    executor = POST_BACKEND.execute_probe if probe_executor is None else probe_executor
    with PARENT.parent_scope(post_input, reference_input, plan["run_id"], plan["policy"]):
        snapshot = C.V.build_snapshot(packet, plan["run_id"], plan["policy"],
                                      C.V.file_digest(C.V.PROTOCOL), ledger_path)
        choice, spec = plan["third_choice"], plan["third_request_spec"]
        _require(C.request_spec(snapshot, choice) == spec and spec["probe"] == "P_physics",
                 "fresh ledger의 세 번째 선택/spec이 plan과 다르다")
        step_root = out / "probes" / "step_03"
        plans = {
            "case": POST_BACKEND.prepare_probe(post_input, "P_physics", stage=step_root / "case",
                                                lab=Path(plan["lab"]), protocol_path=probe_protocol,
                                                anchor=plan["anchor"], role="case"),
            "reference": POST_BACKEND.prepare_probe(reference_input, "P_physics",
                                                     stage=step_root / "reference",
                                                     lab=Path(plan["lab"]), protocol_path=probe_protocol,
                                                     anchor=plan["anchor"], role="reference"),
        }
        plan_paths = {role: E0._plan_path(value) for role, value in plans.items()}
        ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
        _require(len(ledger.requests()) == 2
                 and all(row["state"] == "done" for row in ledger.requests().values()),
                 "복사 ledger prefix가 두 완료 request가 아니다")
        request = ledger.propose(spec["prereg"], spec["probe"], int(spec["budget_gpu_s"]))
        ledger.approve(request["request_id"], approver)
        ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
        authorization = {
            "schema": POST_BACKEND.AUTH_SCHEMA,
            "plan_sha256": {role: _sha(path) for role, path in plan_paths.items()},
            "snapshot": snapshot, "choice": choice, "request_spec": spec,
            "ledger_path": str(ledger_path), "request_id": request["request_id"],
        }
        authorization_path = step_root / "authorization.json"
        _write_new(authorization_path, authorization)
        receipts, parent_wall = E0._execute_pair(executor, plan_paths, authorization_path)
        outcome, ledger_receipt = E0._settle(
            ledger, request["request_id"], "P_physics", receipts, parent_wall)
        final = C.V.build_snapshot(packet, plan["run_id"], plan["policy"],
                                   C.V.file_digest(C.V.PROTOCOL), ledger_path)
    _require(outcome in {"normal", "abnormal", "unknown"}, "세 번째 probe outcome이 다르다")
    if outcome == "abnormal":
        end_state, hold_reason = "HOLD", "repair_unsupported"
        _require(final["view"]["status"] == "confirmed"
                 and final["view"]["remaining_candidates"] == ["physics"],
                 "P_physics abnormal은 physics 단일 확인이어야 한다")
    elif outcome == "normal":
        end_state, hold_reason = "HOLD", "decision_limit"
        _require(final["view"]["status"] == "open", "세 번째 normal 뒤 상태는 open이어야 한다")
    else:
        end_state, hold_reason = "HOLD", "probe_outcome_unknown"
    result = {
        "schema": RESULT_SCHEMA, "status": "complete", "case_id": "case-07",
        "epoch": 2, "end_state": end_state, "hold_reason": hold_reason,
        "probe_order": ["P_episode", "P_noise", "P_physics"],
        "outcomes": ["normal", "normal", outcome],
        "newly_executed_logical_probe": 1, "total_logical_probes": 3,
        "fresh_api_requests": 0, "evidence_scope": evidence_scope,
        "actual_backend_reused_recorded_nim": reused_actual_nim,
        "epoch2_diagnosis_only": True, "correction_execution": False,
        "second_training": False, "repair_supported": False,
        "plan": _ref(plan_path), "copied_prefix_ledger": _ref(ledger_path),
        "step_03": {"authorization": _ref(authorization_path),
                    "request_id": request["request_id"], "outcome": outcome,
                    "ledger_receipt": ledger_receipt, "paired_receipts": receipts},
        "final_snapshot_digest": C.L.digest(final),
        "remaining_candidates": deepcopy(final["view"]["remaining_candidates"]),
        "model_usage": {"reused_responses": 3,
                        "input_tokens": sum(row["input_tokens"] for row in [
                            *plan["recorded_model"]["first_two"], plan["recorded_model"]["third"]]),
                        "output_tokens": sum(row["output_tokens"] for row in [
                            *plan["recorded_model"]["first_two"], plan["recorded_model"]["third"]]),
                        "fresh_requests": 0},
    }
    _write_new(out / "result.json", result)
    return deepcopy(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    protocol = commands.add_parser("prepare-protocol")
    protocol.add_argument("--parent-v3-protocol", type=Path, required=True)
    protocol.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    for item in ("old-stage", "transport-root", "operator-failure", "post-input",
                 "reference-input", "gate", "lab", "protocol"):
        prepare.add_argument("--" + item, type=Path, required=True)
    prepare.add_argument("--gate-sha256", required=True)
    prepare.add_argument("--anchor", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--policy", choices=("model_adaptive",), required=True)
    prepare.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--probe-protocol", type=Path, required=True)
    execute.add_argument("--approver", default="codex-main-user-delegated")
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.parent_v3_protocol))
    elif args.command == "prepare":
        prepare_resume(old_stage=args.old_stage, transport_root=args.transport_root,
                       operator_failure=args.operator_failure, post_input=args.post_input,
                       reference_input=args.reference_input, gate_path=args.gate,
                       gate_sha256=args.gate_sha256, lab=args.lab,
                       protocol_path=args.protocol, anchor=args.anchor, run_id=args.run_id,
                       policy=args.policy, out=args.out)
    else:
        execute_resume(args.plan, out=args.out, probe_protocol=args.probe_protocol,
                       approver=args.approver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
