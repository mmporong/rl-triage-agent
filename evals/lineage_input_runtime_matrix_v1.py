# pyright: reportArgumentType=false, reportMissingImports=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""검증된 lineage source로 입력·품질·busy·transport CPU 경계 matrix를 실행한다."""
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
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import agentic_frame_quality_guard_v1 as FRAME
import lineage_five_policy_driver_v1 as DRIVER
import lineage_five_policy_nim_transport_v1 as TRANSPORT
import local_pair_input_runtime_guard_v1 as GUARD

SCHEMA = "lineage_input_runtime_matrix_v1"
RESULT_SCHEMA = SCHEMA + "_result"
MANIFEST_SCHEMA = SCHEMA + "_manifest"
SOURCE_FILES = tuple(sorted({
    *DRIVER.SOURCE_FILES, *TRANSPORT.SOURCE_FILES,
    "evals/local_pair_input_runtime_guard_v1.py",
    "tests/test_local_pair_input_runtime_guard_v1.py",
    "evals/agentic_frame_quality_guard_v1.py",
    "tests/test_agentic_frame_quality_guard_v1.py",
    "evals/lineage_input_runtime_matrix_v1.py",
    "tests/test_lineage_input_runtime_matrix_v1.py",
}))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    target = Path(path).resolve()
    require(not target.exists(), f"{target.name}은 fresh 파일이어야 한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    require(target.is_file(), f"근거 파일이 없다: {target.name}")
    return {"path": str(target), "sha256": sha(target)}


def _callbacks(allow_fixture: bool, validators: dict[str, Callable] | None) -> dict[str, Callable]:
    if not allow_fixture:
        require(validators is None, "production matrix validator는 교체할 수 없다")
        return {
            "source": lambda value: DRIVER.checked_source(value, fixture=False),
            "protocol": DRIVER.verify_protocol,
            "transport_initialize": TRANSPORT.initialize,
            "transport_complete": TRANSPORT.complete,
            "transport_status": TRANSPORT.status,
            "transport_verify": TRANSPORT.verify_call,
        }
    required = {"source", "protocol", "transport_initialize", "transport_complete",
                "transport_status", "transport_verify"}
    require(isinstance(validators, dict), "fixture matrix validator가 필요하다")
    typed = cast(dict[str, Callable], validators)
    require(set(typed) == required and all(callable(item) for item in typed.values()),
            "fixture matrix validator 집합이 다르다")
    return typed


def _registered_source(source_path: Path, index_path: Path, protocol_path: Path, anchor: str, *,
                       allow_fixture: bool, callbacks: dict[str, Callable]) -> tuple[dict, dict]:
    source_path = Path(source_path).resolve(strict=True)
    index_path = Path(index_path).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    raw = read(source_path)
    protocol = callbacks["protocol"](protocol_path, anchor)
    checked = callbacks["source"](deepcopy(raw))
    require(checked == raw, "source validator가 입력을 바꿨다")
    index_ref = checked.get("index")
    require(isinstance(index_ref, dict)
            and Path(str(index_ref.get("path", ""))).resolve() == index_path
            and index_ref.get("sha256") == sha(index_path),
            "source/index SHA 결합이 다르다")
    if allow_fixture:
        require(protocol.get("fixture") is True, "fixture protocol 표식이 없다")
    else:
        key = DRIVER.source_registration_key(checked, checked.get("matrix_policy", "fixed_eligible"))
        require(protocol.get("source_digests", {}).get(key) == DRIVER.BASE.digest(checked),
                "source가 protocol에 등록되지 않았다")
    return checked, protocol


def _row(case: str, evidence_class: str, result: dict[str, Any], *,
         claim: str, notes: str | None = None) -> dict[str, Any]:
    decision = result.get("decision")
    status = "HOLD" if decision == "HOLD" else "continue" if decision == "continue" else (
        "not_run" if result.get("status") == "not_run" else "verified")
    return {"case": case, "evidence_class": evidence_class, "status": status,
            "claim": claim, "result": deepcopy(result), "notes": notes,
            "approval_granted": False, "dispatch_granted": False,
            "automatic_retry": False}


def _quality_receipt() -> tuple[dict[str, Any], float]:
    raw = b"P5\n8 8\n255\n" + bytes([128] * 64)
    variance = FRAME.laplacian_variance(raw)
    require(variance == 0.0, "통제 blur의 Laplacian variance가 0이 아니다")
    receipt = {
        "schema": FRAME.SCHEMA + "_evidence_receipt", "approval_granted": False,
        "visual_outcome": "unknown", "frame_refs": [], "image_count": 0,
        "quality_receipts": [{"outcome": "quality_below_registered_floor",
                              "metric": {"name": FRAME.METRIC, "value": variance,
                                         "unit": FRAME.UNIT}}],
    }
    return receipt, variance


def _geometry(status: str) -> dict[str, str]:
    gates = {"consistent": "reviewed_consistent", "inconsistent": "quarantine_visual_evidence",
             "unclear": "HOLD_visual_evidence"}
    return {"schema": "visual_localization_host_decision_v2", "status": status,
            "gate": gates[status]}


def _request(source: dict[str, Any], output: Path) -> dict[str, Any]:
    packet = source["packet"]
    snapshot = DRIVER.C.V.build_snapshot(
        packet, "matrix_transport", "model_adaptive",
        DRIVER.C.V.file_digest(DRIVER.C.V.PROTOCOL), output / "empty-ledger.jsonl")
    context = DRIVER.A.prepare(
        snapshot, goal=DRIVER.BASE.P.PAIR_GOAL, decisions_left=3)
    messages = DRIVER.CALL._messages(context, DRIVER.A.messages(context))
    return DRIVER._request(source, context, messages, "matrixtransport", "logs_only", 1)


def run_matrix(source_path: Path, index_path: Path, protocol_path: Path, anchor: str,
               out: Path, *, actual_call_dirs: tuple[Path, ...] = (),
               allow_fixture: bool = False,
               validators: dict[str, Callable] | None = None) -> dict[str, Any]:
    """원문을 바꾸지 않고 guard/driver/fixture-transport matrix를 fresh 폴더에 쓴다."""
    output = Path(out).resolve()
    require(not output.exists(), "matrix output은 fresh 경로여야 한다")
    callbacks = _callbacks(allow_fixture, validators)
    originals = {"source": sha(Path(source_path)), "index": sha(Path(index_path)),
                 "protocol": sha(Path(protocol_path))}
    source, protocol = _registered_source(
        source_path, index_path, protocol_path, anchor,
        allow_fixture=allow_fixture, callbacks=callbacks)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    if actual_call_dirs:
        verified_calls = []
        for call_dir in actual_call_dirs:
            call = Path(call_dir).resolve(strict=True)
            receipt_before = sha(call / "receipt.json")
            verified = callbacks["transport_verify"](call)
            gate = GUARD.gate_transport(verified)
            require(sha(call / "receipt.json") == receipt_before,
                    "actual transport receipt가 matrix 검증 중 바뀌었다")
            verified_calls.append({"call_ref": deepcopy(verified["call_ref"]),
                                   "live": verified["live"], "status": verified["status"],
                                   "outcome": verified["outcome"], "gate": gate})
        rows.append(_row("actual_model_sdk_receipts", "actual_model_sdk_raw",
                         {"status": "verified", "calls": verified_calls},
                         claim="입력으로 제공된 기존 call 원문만 read-only 검증"))
    else:
        rows.append(_row("actual_model_sdk_receipts", "actual_model_sdk_raw",
                         {"status": "not_run", "reason": "no_registered_call_input"},
                         claim="actual API outage 또는 성공을 주장하지 않음"))

    controlled = output / "controlled-inputs"
    controlled.mkdir()
    for name, error_type in (("registered_input_missing", FileNotFoundError),
                             ("registered_input_permission", PermissionError)):
        target = controlled / (name + ".json")

        def unavailable(_target: Path = target, _error: type[OSError] = error_type) -> None:
            raise _error(2 if _error is FileNotFoundError else 13,
                         "controlled matrix input", str(_target))

        guarded = GUARD.invoke_guarded(
            "input", unavailable, registered_input_roots=(controlled,))
        rows.append(_row(name, "controlled_perturbation", guarded,
                         claim="등록된 matrix copy root의 주입형 입력 오류"))

    quality_receipt, variance = _quality_receipt()
    quality = GUARD.gate_quality(quality_receipt)
    rows.append(_row("blur_frame_quarantine", "controlled_perturbation", quality,
                     claim="통제 constant PGM CPU blur metric; 실제 source frame 의미 인식 아님",
                     notes=f"laplacian_variance={variance}; frame_refs=0; image_count=0"))
    conflict = GUARD.gate_geometry(_geometry("consistent"), "inconsistent")
    rows.append(_row("visual_numeric_conflict", "controlled_perturbation", conflict,
                     claim="검증 형식의 통제 visual/numeric 불일치"))
    unknown = GUARD.gate_geometry(_geometry("unclear"), "unknown")
    rows.append(_row("visual_numeric_unknown", "controlled_perturbation", unknown,
                     claim="검증 형식의 통제 unknown"))

    busy_calls = {"busy": 0, "dispatch": 0}

    def busy_reader(_operation_id: str, _intent: dict) -> dict[str, str]:
        busy_calls["busy"] += 1
        return {"status": "busy", "observation_id": "matrix-injected-busy-1"}

    def forbidden_probe(*_args: Any, **_kwargs: Any) -> dict:
        busy_calls["dispatch"] += 1
        raise AssertionError("busy matrix는 probe dispatch 전에 멈춰야 한다")

    policy = source.get("matrix_policy", "fixed_eligible")
    busy_result = DRIVER.run_arm(
        source, arm=policy, run_id="input_matrix_busy", output=output / "driver-busy",
        protocol_path=Path(protocol_path), execution_anchor=anchor, complete=None,
        execute_probe=forbidden_probe, verify_probe=lambda *_args: {},
        busy_reader=busy_reader, allow_fixture=allow_fixture,
        registered_input_roots=(Path(index_path).resolve().parent,))
    busy_gate = read(output / "driver-busy/step01/busy-gate.json") if (
        output / "driver-busy/step01/busy-gate.json").is_file() else {
            "decision": "HOLD", "reason": busy_result.get("hold_reason"),
            "stage": "busy_not_reached", "evidence": {"calls": busy_calls}}
    rows.append(_row("gpu_busy_pre_dispatch", "controlled_injection", busy_gate,
                     claim="GPU 상태 실측이 아닌 주입 busy callback",
                     notes=f"driver_scope={busy_result['scope']}; dispatch_calls={busy_calls['dispatch']}"))

    timeout_calls = {"count": 0}

    def typed_timeout() -> None:
        timeout_calls["count"] += 1
        raise TimeoutError("controlled typed timeout")

    timeout = GUARD.invoke_guarded("transport", typed_timeout)
    rows.append(_row("typed_transport_timeout", "controlled_perturbation", timeout,
                     claim="주입 TimeoutError; 실제 API outage 아님",
                     notes=f"callback_calls={timeout_calls['count']}"))
    pending_calls = {"count": 0}

    def pending() -> None:
        pending_calls["count"] += 1
        raise RuntimeError(GUARD.PENDING_NO_RESEND)

    pending_result = GUARD.invoke_guarded("transport", pending)
    rows.append(_row("pending_no_automatic_resend", "controlled_perturbation", pending_result,
                     claim="등록 pending exception 1회 호출; 자동 재전송 없음",
                     notes=f"callback_calls={pending_calls['count']}"))

    transport_root = output / "fixture-transport"
    callbacks["transport_initialize"](
        transport_root, protocol_path=Path(protocol_path), execution_anchor=anchor,
        api_cap=protocol.get("api_cap", TRANSPORT.API_CAP))
    request = _request(source, output)
    sends = {"count": 0}

    def fixture_send(_body: dict, _call: Path) -> dict[str, Any]:
        sends["count"] += 1
        return {"live": False, "http_status": 200, "wall_s": 0.0,
                "provider_response": {
                    "id": "cpu-matrix-fixture", "model": TRANSPORT.MODEL_CALL.MODEL,
                    "choices": [{"message": {"content": '{"action":"hold"}'},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}}

    first = callbacks["transport_complete"](transport_root, request, send=fixture_send)

    def duplicate_send_forbidden(*_args: Any, **_kwargs: Any) -> dict:
        raise AssertionError("완료 receipt 재사용은 send를 다시 호출하면 안 된다")

    second = callbacks["transport_complete"](
        transport_root, request, send=duplicate_send_forbidden)
    status = callbacks["transport_status"](transport_root)
    require(first == second and sends["count"] == 1
            and status.get("completed") == 1 and status.get("automatic_retries") == 0,
            "fixture transport 완료 receipt 재사용 계약이 다르다")
    rows.append(_row("completed_receipt_reuse", "fixture_transport",
                     {"status": "verified", "call_ref": deepcopy(first.get("call_ref")),
                      "send_calls": sends["count"], "transport_status": status,
                      "actual_backend_admissible": False},
                     claim="live=false CPU fixture transport; actual model 성공으로 사용 금지"))

    current = {"source": sha(Path(source_path)), "index": sha(Path(index_path)),
               "protocol": sha(Path(protocol_path))}
    require(current == originals, "matrix 실행 중 등록 원문 SHA가 바뀌었다")
    result = {
        "schema": RESULT_SCHEMA, "status": "complete",
        "scope": "development_fixture" if allow_fixture else "actual_source_cpu_matrix",
        "source": ref(Path(source_path)), "index": ref(Path(index_path)),
        "protocol": ref(Path(protocol_path)), "anchor": anchor,
        "source_digest": DRIVER.BASE.digest(source), "rows": rows,
        "row_count": len(rows), "actual_api_calls": 0, "gpu_operations": 0,
        "source_files_mutated": False, "fixture_results_promoted_to_actual": False,
        "claims": {"actual_api_outage": False, "semantic_blur_recognition": False,
                   "actual_gpu_busy": False, "model_effect": False},
    }
    write_new(output / "matrix.json", result)
    manifest = {"schema": MANIFEST_SCHEMA, "files": {
        "matrix.json": sha(output / "matrix.json"),
        "driver-busy/result.json": sha(output / "driver-busy/result.json"),
        "fixture-transport/manifest.json": sha(output / "fixture-transport/manifest.json")}}
    write_new(output / "manifest.json", manifest)
    return deepcopy(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actual-call-dir", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    run_matrix(args.source, args.index, args.protocol, args.anchor, args.out,
               actual_call_dirs=tuple(args.actual_call_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
