"""E2E 정적 검증과 30초 native resource proof 사이의 짧은 handoff를 제공한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    sys.path.insert(0, str(folder))

import run_vision_corrected_pull_record_v2 as RUNNER2
import vision_transferred_e2e_v2 as E

SCHEMA = "vision_corrected_native_handoff_ticket_v2"
INTENT_SCHEMA = "vision_corrected_native_handoff_intent_v2"
FAILURE_SCHEMA = "vision_corrected_native_handoff_failure_v2"
BRIDGE_RELATIVE = "evals/vision_corrected_native_handoff_v2.py"
RUNNER_RELATIVE = "bench/run_vision_corrected_pull_record_v2.py"
TTL_S = 120
GROUPS = {"source-01": 2220, "source-02": 2221}
HEX40 = E.CALL._HEX40


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "handoff 입력은 regular file이어야 함")
    return {"path": str(path), "sha256": E.ADAPTER.file_sha(path)}


def _read_ref(value: Mapping[str, Any]) -> tuple[Path, Any]:
    require(set(value) == {"path", "sha256"}, "handoff ref 형식 불일치")
    path = Path(str(value["path"])).resolve(strict=True)
    require(path.is_file() and not path.is_symlink()
            and E.ADAPTER.file_sha(path) == value["sha256"], "handoff ref 원문/SHA 불일치")
    return path, _read(path)


def _utc(value: str, name: str) -> datetime:
    parsed = E.ADAPTER._time(value, name)
    require(parsed.tzinfo is not None, f"{name} timezone 누락")
    return parsed.astimezone(timezone.utc)


def _inside(path: Path, parent: Path, name: str) -> Path:
    path = Path(path).absolute()
    parent = Path(parent).resolve()
    require(path == parent or parent in path.parents, f"{name}이 runtime 경계 밖")
    return path


def _bridge_source(repo: Path, protocol: Mapping[str, Any], *,
                   evidence_factory: Callable[[Path], Any] = E.ObjectEvidence) -> dict[str, Any]:
    source = protocol.get("source_sha256_lf", {})
    commit = str(protocol.get("source_commit", ""))
    require(HEX40.fullmatch(commit) is not None, "native protocol source commit 형식 오류")
    evidence = evidence_factory(repo)
    files: dict[str, str] = {}
    for relative in (BRIDGE_RELATIVE, RUNNER_RELATIVE):
        current = E.ADAPTER.file_sha(repo / relative, lf=True)
        require(source.get(relative) == current, f"native v2 source closure 누락: {relative}")
        _, blob = evidence.file(commit, relative)
        committed = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
        require(committed == current, f"native v2 commit blob 불일치: {relative}")
        files[relative] = current
    return {"source_commit": commit, "files_sha256_lf": files}


def _context(repo: Path, execution_protocol: Path, execution_sha: str,
             lease: Path, lease_sha: str) -> dict:
    def portable(path: Path, sha: str) -> dict[str, str]:
        path = Path(path).resolve(strict=True)
        require(repo in path.parents, "execution 입력이 repo 밖")
        return {"path": path.relative_to(repo).as_posix(), "sha256": sha}
    return E.ADAPTER.verify_context(repo, portable(execution_protocol, execution_sha),
                                    portable(lease, lease_sha), require_active=False)


def prepare(
    repo: Path,
    plan_path: Path,
    scope_path: Path,
    execution_protocol: Path,
    execution_protocol_sha256: str,
    lease: Path,
    lease_sha256: str,
    bound_protocol: Path,
    registration_reference_path: Path,
    registration_anchor: str,
    group: str,
    output: Path,
    *,
    now: datetime | None = None,
    runtime_factory: Callable[..., Any] = E.Runtime,
    evidence_factory: Callable[[Path], Any] = E.ObjectEvidence,
) -> dict[str, Any]:
    """비싼 E2E 검증을 끝낸 뒤 120초짜리 record ticket을 write-once로 만든다."""
    repo = Path(repo).resolve(strict=True)
    require(group in GROUPS and HEX40.fullmatch(registration_anchor) is not None, "handoff group/anchor 불일치")
    context = _context(repo, execution_protocol, execution_protocol_sha256, lease, lease_sha256)
    runtime = runtime_factory(repo, Path(plan_path).resolve(strict=True), context,
                              Path(scope_path).resolve(strict=True), require_active=False)
    bindings = runtime.recording_bindings()
    logical = f"corrected-s{GROUPS[group]}"
    require(logical in bindings, "handoff corrected binding 누락")
    binding = bindings[logical]
    runner, native = E.native_components()
    protocol_path = Path(bound_protocol).resolve(strict=True)
    registration_ref_path = Path(registration_reference_path).resolve(strict=True)
    registration_ref = _read(registration_ref_path)
    require(isinstance(registration_ref, dict), "native registration ref 파일 형식 오류")
    template, _ = E.ADAPTER._ref(
        repo, context["protocol"]["execution_bindings"]["corrected_render_plan"],
        "pretraining corrected native template",
    )
    E.verify_native_registration(repo, registration_ref, registration_anchor,
                                 protocol_path, template)
    protocol = runner.validate_protocol(protocol_path)
    E.ADAPTER.verify_record_service(context, native.NativeCorrectedRecordService(protocol_path).record)
    expected_binding = runner._verified_binding(
        logical, Path(binding["correction_result"]), Path(binding["approval_receipt"])
    )
    catalog = [row for row in protocol["checkpoints"] if row["logical_id"] == logical]
    require(catalog == [expected_binding], "ticket 교정 결과/승인과 bound checkpoint 불일치")
    closure = _bridge_source(repo, protocol, evidence_factory=evidence_factory)
    runtime_root = runtime.path.resolve()
    expected = {
        "stage": str(runtime_root / "native-stage-v2" / group),
        "cache": str(runtime_root / "native-record" / f"{group}.json"),
        "intent": str(runtime_root / "intents" / f"native-v2-{group}.json"),
        "receipt_output": str(runtime_root / "native-receipt-v2" / group),
        "failure": str(runtime_root / "failures" / f"native-v2-{group}.json"),
    }
    for name, path in expected.items():
        checked = _inside(Path(path), runtime_root, name)
        require(not checked.exists() and not checked.is_symlink(), f"새 {name} 경로가 필요함")
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    value = {
        "schema": SCHEMA,
        "created_at_utc": created.isoformat(),
        "expires_at_utc": (created + timedelta(seconds=TTL_S)).isoformat(),
        "ttl_s": TTL_S,
        "group": group,
        "logical_id": logical,
        "registration_anchor": registration_anchor,
        "inputs": {
            "plan": _ref(plan_path),
            "scope": _ref(scope_path),
            "execution_protocol": _ref(execution_protocol),
            "lease": _ref(lease),
            "bound_protocol": _ref(protocol_path),
            "registration_reference": _ref(registration_ref_path),
        },
        "binding": {name: _ref(path) for name, path in binding.items()},
        "expected": expected,
        "bridge_source": closure,
        "intent_policy": "write_once_before_gpu_no_automatic_rerun",
        "cache_policy": "write_once_after_recheck_native_result_only",
    }
    E.ADAPTER._write_new(output, value)
    return value


def _ticket(repo: Path, ticket_path: Path, *, now: datetime,
            evidence_factory: Callable[[Path], Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ticket = _read(ticket_path)
    required = {"schema", "created_at_utc", "expires_at_utc", "ttl_s", "group", "logical_id",
                "registration_anchor", "inputs", "binding", "expected", "bridge_source",
                "intent_policy", "cache_policy"}
    require(isinstance(ticket, dict) and set(ticket) == required and ticket["schema"] == SCHEMA
            and ticket["ttl_s"] == TTL_S and ticket["group"] in GROUPS
            and ticket["logical_id"] == f"corrected-s{GROUPS[ticket['group']]}"
            and ticket["intent_policy"] == "write_once_before_gpu_no_automatic_rerun"
            and ticket["cache_policy"] == "write_once_after_recheck_native_result_only",
            "native handoff ticket 형식 불일치")
    created, expires = _utc(ticket["created_at_utc"], "ticket created"), _utc(ticket["expires_at_utc"], "ticket expires")
    require(expires - created == timedelta(seconds=TTL_S) and created <= now < expires,
            "native handoff ticket TTL 밖")
    documents = {name: _read_ref(value) for name, value in ticket["inputs"].items()}
    _protocol_path, protocol = documents["bound_protocol"]
    closure = _bridge_source(repo, protocol, evidence_factory=evidence_factory)
    require(ticket["bridge_source"] == closure, "ticket bridge source closure 변경")
    return ticket, {name: {"path": path, "value": value} for name, (path, value) in documents.items()}


def record(
    repo: Path,
    ticket_path: Path,
    ticket_sha256: str,
    provider_inventory: Path,
    gpu_csv: Path,
    process_csv: Path,
    observed_at_utc: str,
    expires_at_utc: str,
    resource_proof_output: Path,
    isaaclab: Path,
    python: Path,
    *,
    now: datetime | None = None,
    evidence_factory: Callable[[Path], Any] = E.ObjectEvidence,
    components: Callable[[], tuple[Any, Any]] = E.native_components,
) -> dict[str, Any]:
    """fresh proof를 만든 뒤 native recorder 한 건을 소비하고 검증된 cache만 쓴다."""
    repo = Path(repo).resolve(strict=True)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ticket_path = Path(ticket_path).resolve(strict=True)
    require(E.ADAPTER.file_sha(ticket_path) == ticket_sha256,
            "native handoff ticket 원문 SHA가 prepare 영수증과 다름")
    ticket, documents = _ticket(repo, ticket_path, now=current,
                                evidence_factory=evidence_factory)
    runner, native = components()
    protocol_path = documents["bound_protocol"]["path"]
    protocol = documents["bound_protocol"]["value"]
    logical, group = ticket["logical_id"], ticket["group"]
    binding_paths = {name: _read_ref(value)[0] for name, value in ticket["binding"].items()}
    expected_binding = runner._verified_binding(
        logical, binding_paths["correction_result"], binding_paths["approval_receipt"]
    )
    require(expected_binding in protocol["checkpoints"], "record 시점 교정 결과/승인 변경")
    expected = {name: _inside(Path(path), Path(ticket["expected"]["cache"]).parent.parent, name)
                for name, path in ticket["expected"].items()}
    for name in ("stage", "cache", "intent", "receipt_output", "failure"):
        require(not expected[name].exists() and not expected[name].is_symlink(),
                f"기존 {name} 때문에 자동 재실행 금지")
    proof_path = Path(resource_proof_output).absolute()
    require(not proof_path.exists() and not proof_path.is_symlink(), "resource proof 출력은 새 파일이어야 함")
    proof = runner.build_resource_proof(
        protocol_path,
        provider_inventory_path=provider_inventory,
        nvidia_gpu_csv_path=gpu_csv,
        nvidia_process_csv_path=process_csv,
        observed_at_utc=observed_at_utc,
        expires_at_utc=expires_at_utc,
        repo=repo,
    )
    E.ADAPTER._write_new(proof_path, proof)
    proof_sha = E.ADAPTER.file_sha(proof_path)
    expected["intent"].parent.mkdir(parents=True, exist_ok=True)
    intent = {
        "schema": INTENT_SCHEMA,
        "ticket": _ref(ticket_path),
        "protocol": _ref(protocol_path),
        "logical_id": logical,
        "resource_proof": _ref(proof_path),
        "stage": str(expected["stage"]),
        "receipt_output": str(expected["receipt_output"]),
        "used_for_decision": False,
        "created_at_utc": current.isoformat(),
    }
    E.ADAPTER._write_new(expected["intent"], intent)
    service = native.NativeCorrectedRecordService(protocol_path)
    kwargs = {
        "resource_proof": proof_path,
        "resource_sha256": proof_sha,
        "isaaclab": Path(isaaclab),
        "python": Path(python),
        "stage": expected["stage"],
        "logical_id": logical,
        "correction_result": binding_paths["correction_result"],
        "approval_receipt": binding_paths["approval_receipt"],
        "receipt_output": expected["receipt_output"],
        "normal_reference": None,
    }
    try:
        result = service.record(**kwargs, run=RUNNER2.run_job)
        if not isinstance(result, dict) or result.get("status") != "complete":
            failure = {"schema": FAILURE_SCHEMA, "status": "HOLD", "reason": "native_record_incomplete",
                       "group": group, "intent": _ref(expected["intent"]), "result": result}
            expected["failure"].parent.mkdir(parents=True, exist_ok=True)
            E.ADAPTER._write_new(expected["failure"], failure)
            return failure
        E.recheck_native_result(result, protocol_path, logical,
                                binding_paths["correction_result"], binding_paths["approval_receipt"])
        expected["cache"].parent.mkdir(parents=True, exist_ok=True)
        E.ADAPTER._write_new(expected["cache"], result)
        return result
    except Exception as error:
        if not expected["failure"].exists():
            expected["failure"].parent.mkdir(parents=True, exist_ok=True)
            E.ADAPTER._write_new(expected["failure"], {
                "schema": FAILURE_SCHEMA, "status": "HOLD", "reason": "native_record_failed",
                "group": group, "intent": _ref(expected["intent"]), "error_type": type(error).__name__,
            })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("repo", "plan", "scope", "execution-protocol", "lease", "bound-protocol",
                 "recording-registration-ref", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--execution-protocol-sha256", required=True)
    prep.add_argument("--lease-sha256", required=True)
    prep.add_argument("--recording-registration-anchor", required=True)
    prep.add_argument("--group", choices=tuple(GROUPS), required=True)
    run = sub.add_parser("record")
    for name in ("repo", "ticket", "provider-inventory", "gpu-csv", "process-csv",
                 "resource-proof-output", "isaaclab", "python"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--ticket-sha256", required=True)
    run.add_argument("--observed-at-utc", required=True)
    run.add_argument("--expires-at-utc", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.repo, args.plan, args.scope, args.execution_protocol,
                        args.execution_protocol_sha256, args.lease, args.lease_sha256,
                        args.bound_protocol, args.recording_registration_ref,
                        args.recording_registration_anchor, args.group, args.output)
        print(json.dumps({"status": "ticket_ready", "group": value["group"],
                          "expires_at_utc": value["expires_at_utc"],
                          "ticket_sha256": E.ADAPTER.file_sha(args.output)}))
    else:
        value = record(args.repo, args.ticket, args.ticket_sha256, args.provider_inventory, args.gpu_csv,
                       args.process_csv, args.observed_at_utc, args.expires_at_utc,
                       args.resource_proof_output, args.isaaclab, args.python)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
