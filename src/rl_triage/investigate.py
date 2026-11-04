"""단일 조사 진입점: ``python -m rl_triage.investigate <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rl_triage.investigation_session import (
    InvestigationSession,
    canonical,
    file_sha,
    reference,
    write_new,
)

REGISTERED_BACKEND = "registered_diagnostic_v1"
FIXTURE_BACKEND = "cpu_fixture_v1"
_REQUIRED = {
    "backend_id", "prepare_source", "verify_source", "authorize", "execute", "verify_receipt",
}
_OPTIONAL = {"complete", "observation_gate", "collect_metrics"}
RECOVERY_COMMANDS = (
    "recovery-prepare", "recovery-propose", "recovery-approve", "recovery-dispatch",
    "recovery-settle", "recovery-status", "recovery-post", "recovery-review",
)


def _fixture_backend() -> dict[str, Any]:
    def prepare_source(payload: dict) -> dict:
        return {
            "source_path": payload["source_path"], "registry": payload["registry"],
            "model_binding": payload.get("model_binding"), "execution_class": "fixture",
        }

    def verify_source(path: Path) -> dict:
        source = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "verified", "verifier_id": FIXTURE_BACKEND,
            "source_sha256": file_sha(path), "checkpoint_sha256": source["checkpoint_sha256"],
            "execution_class": "fixture", "case_id": source["case_id"],
        }

    def authorize(request: dict, actor: str, authorization: dict) -> dict:
        if actor != "fixture_actor" or authorization != {"fixture": True}:
            return {"status": "denied"}
        return {
            "status": "authorized", "authority_id": FIXTURE_BACKEND, "actor": actor,
            "request_id": request["request_id"], "effect_key": request["effect_key"],
        }

    def complete(messages: list, _params: dict) -> dict:
        user = json.loads(messages[-1]["content"])
        probe = next(iter(user["catalog"]))
        fact_id = next(iter(user["facts"]))
        one_shot = "fixed ordered plan" in messages[0]["content"]
        choice = {"action": "plan" if one_shot else "run_probe",
                  "probe": None if one_shot else probe,
                  "probes": [probe] if one_shot else [], "fact_ids": [fact_id]}
        return {"content": json.dumps(choice)}

    def execute(intent: dict) -> dict:
        request, source = intent["request"], intent["source"]
        outcome = source["binding"].get("fixture_outcomes", {}).get(request["probe"], "unknown")
        artifact = {
            "outcome": outcome, "measurement": {"fixture": True}, "gpu_s": 0.0,
            "execution_wall_s": 0.0, "exit_code": 0,
            "source_sha256": intent["preregistration"]["source_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"], "case_id": source["case_id"],
            "probe": request["probe"], "request_id": request["request_id"],
            "effect_key": request["effect_key"], "registry_sha256": intent["registry_sha256"],
            "execution_class": "fixture",
        }
        write_new(Path(intent["host_receipt_path"]), artifact)
        return artifact

    def verify_receipt(raw: dict, intent: dict) -> dict:
        return {**raw, "receipt_ref": reference(Path(intent["host_receipt_path"]))}

    return {
        "backend_id": FIXTURE_BACKEND, "prepare_source": prepare_source,
        "verify_source": verify_source, "authorize": authorize, "complete": complete,
        "execute": execute, "verify_receipt": verify_receipt,
        "observation_gate": lambda _context: {
            "decision": "allow", "gate_id": FIXTURE_BACKEND, "reason": "fixture",
        },
        "collect_metrics": lambda _event: None,
    }


def _load_backend(name: str, fixture: bool) -> tuple[dict[str, Any] | None, str | None]:
    if name == FIXTURE_BACKEND:
        if not fixture:
            return None, "fixture_requires_explicit_flag"
        backend = _fixture_backend()
    elif name == REGISTERED_BACKEND:
        if fixture:
            return None, "fixture_flag_backend_mismatch"
        try:
            # 허용된 단일 production namespace 외의 module/path/command는 받지 않는다.
            from rl_triage import registered_diagnostic_backend as registered
        except ImportError:
            return None, "registered_backend_not_installed"
        try:
            backend = registered.build_backend()
        except Exception as error:  # noqa: BLE001 - backend 초기화 실패는 제품 실행 성공이 아니다.
            return None, f"registered_backend_init:{type(error).__name__}"
    else:
        return None, "backend_name_not_registered"
    if not isinstance(backend, dict) or set(backend) - (_REQUIRED | _OPTIONAL) or not _REQUIRED <= set(backend):
        return None, "backend_contract_mismatch"
    if backend.get("backend_id") != name:
        return None, "backend_identity_mismatch"
    if any(not callable(backend[key]) for key in set(backend) - {"backend_id"}):
        return None, "backend_callable_mismatch"
    return backend, None


def _payload(path: Path | None) -> dict:
    text = path.read_text(encoding="utf-8") if path else ""
    if not text.strip():
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("input JSON must be an object")
    canonical(value)
    return value


def _blocked(reason: str) -> dict:
    return {"status": "blocked_backend", "reason": reason, "completed": False}


def _run_recovery(command: str, session_path: Path, payload: dict, *, backend_name: str,
                  fixture: bool) -> dict:
    if fixture or backend_name != REGISTERED_BACKEND:
        return _blocked("recovery_requires_registered_host")
    from rl_triage import investigation_recovery as recovery

    operation = command.removeprefix("recovery-")
    expected_fields = {
        "prepare": {"recovery_payload", "folder"}, "propose": set(),
        "approve": {"request_id", "actor", "approval_ref"}, "dispatch": {"request_id"},
        "settle": {"request_id", "receipt"}, "status": set(), "post": {"request_id"},
        "review": {"request_id", "host_gate_sources"},
    }
    if operation not in expected_fields or set(payload) != expected_fields[operation]:
        raise ValueError("recovery command input fields differ")
    if operation == "prepare":
        return recovery.prepare(session_path, payload["recovery_payload"], Path(payload["folder"]))
    if operation == "propose":
        return recovery.propose(session_path)
    if operation == "approve":
        return recovery.approve(session_path, payload["request_id"], payload["actor"],
                                payload["approval_ref"])
    if operation == "dispatch":
        return recovery.dispatch(session_path, payload["request_id"])
    if operation == "settle":
        return recovery.settle(session_path, payload["request_id"], payload["receipt"])
    if operation == "status":
        return recovery.status(session_path)
    if operation == "review":
        return recovery.review(session_path, payload["request_id"], payload["host_gate_sources"])
    return recovery.verified_post_observation(session_path, payload["request_id"])


def run(command: str, session_path: Path, payload: dict, *, backend_name: str,
        fixture: bool) -> dict:
    if command in RECOVERY_COMMANDS:
        return _run_recovery(command, session_path, payload, backend_name=backend_name,
                             fixture=fixture)
    if command in {"status", "export"}:
        session = InvestigationSession(session_path)
        if command == "status":
            return session.status()
        return session.export(Path(payload["path"]))
    backend, reason = _load_backend(backend_name, fixture)
    if backend is None:
        return _blocked(reason or "backend_unavailable")
    if command == "prepare":
        prepared = backend["prepare_source"](payload)
        required = {"source_path", "registry", "model_binding", "execution_class"}
        if not isinstance(prepared, dict) or set(prepared) != required:
            return _blocked("prepare_source_contract_mismatch")
        execution_class = prepared["execution_class"]
        if execution_class not in ({"fixture"} if fixture else {"actual", "recorded"}):
            return _blocked("execution_class_mismatch")
        session = InvestigationSession.prepare(
            Path(prepared["source_path"]), session_path, registry=prepared["registry"],
            method=payload["method"], max_checks=payload["max_checks"],
            gpu_budget_per_check_s=payload["gpu_budget_per_check_s"],
            source_verifier=backend["verify_source"], model_binding=prepared["model_binding"],
            execution_class=execution_class,
        )
        return session.status()
    session = InvestigationSession(session_path)
    collector = backend.get("collect_metrics")
    if command == "next":
        if session.manifest["method"] in {"one_shot_llm", "iterative_agent"} and "complete" not in backend:
            return _blocked("model_backend_not_registered")
        return session.next(
            complete=backend.get("complete"), source_verifier=backend["verify_source"],
            observation_gate=backend.get("observation_gate"), metrics_collector=collector,
        )
    if command == "approve":
        return session.approve(
            payload["request_id"], actor=payload["actor"], authorization=payload["authorization"],
            authorize=backend["authorize"], source_verifier=backend["verify_source"],
        )
    if command == "dispatch":
        return session.dispatch(
            payload["request_id"], execute=backend["execute"],
            verify_receipt=backend["verify_receipt"], source_verifier=backend["verify_source"],
            metrics_collector=collector,
        )
    if command == "settle":
        return session.settle(
            payload["request_id"], host_receipt=payload.get("host_receipt"),
            verify_receipt=backend["verify_receipt"], source_verifier=backend["verify_source"],
            metrics_collector=collector,
        )
    raise ValueError(f"unsupported command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "next", "approve", "dispatch", "settle",
                                          "status", "export", *RECOVERY_COMMANDS))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--backend", default=REGISTERED_BACKEND)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.command, args.session, _payload(args.input),
                     backend_name=args.backend, fixture=args.fixture)
    except Exception as error:  # noqa: BLE001 - CLI 경계는 안전한 오류 JSON으로 변환한다.
        result = {"status": "rejected", "error_type": type(error).__name__,
                  "reason": str(error), "completed": False}
    sys.stdout.write(canonical(result).decode() + "\n")
    return 0 if result.get("status") not in {"rejected"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
