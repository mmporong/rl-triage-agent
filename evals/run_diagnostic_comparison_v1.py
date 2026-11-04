"""사전등록된 네 진단 arm을 기존 investigation 진입점으로 순차 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

_ENTRY_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ENTRY_ROOT), *(str(_ENTRY_ROOT / name) for name in ("src", "bench", "evals"))]

from rl_triage import investigate
from rl_triage.investigation_session import _journal_module as _decision_journal_module
from rl_triage.investigation_session import file_sha, reference, write_new

SCHEMA = "diagnostic_comparison_plan_v1"
JOB_SCHEMA = "diagnostic_comparison_job_v1"
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
ACTUAL_BACKEND = investigate.REGISTERED_BACKEND
FIXTURE_BACKEND = investigate.FIXTURE_BACKEND
TERMINAL = {
    "confirmed",
    "none_supported",
    "unidentifiable",
    "hold_unknown",
    "hold_budget_exhausted",
    "host_no_intervention",
    "hold_observation_unqualified",
    "hold_no_registered_probe",
    "model_hold",
    "hold_blocked_pending",
    "hold_paused_infra",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MIN_MODEL_WATCHDOG_S = 660


class ComparisonContractError(ValueError):
    """Plan, source, protocol, 또는 stage 격리 계약이 깨졌을 때 발생한다."""


class GpuSampler(Protocol):
    def start(self) -> dict: ...

    def stop(self) -> dict: ...


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _lf_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(_lf_bytes(path.read_bytes())).hexdigest()


def _relative_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or value in {"", "."} or "\\" in value:
        raise ComparisonContractError(f"{label} must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ComparisonContractError(f"{label} escapes repo")
    return path


def _read_ref(value: object, label: str) -> tuple[Path, dict]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ComparisonContractError(f"{label} reference fields differ")
    candidate = Path(str(value["path"]))
    if candidate.is_symlink():
        raise ComparisonContractError(f"{label} reference is a symlink")
    path = candidate.resolve(strict=True)
    if not isinstance(value["sha256"], str) or _SHA.fullmatch(value["sha256"]) is None:
        raise ComparisonContractError(f"{label} reference is invalid")
    if file_sha(path) != value["sha256"]:
        raise ComparisonContractError(f"{label} SHA changed")
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")),
    )
    if not isinstance(parsed, dict):
        raise ComparisonContractError(f"{label} must be a JSON object")
    canonical(parsed)
    return path, parsed


def _git(call: Callable[..., object], args: list[str], repo_root: Path) -> object:
    result = call(args, cwd=repo_root, capture_output=True, check=False, timeout=10)
    if getattr(result, "returncode", None) != 0:
        raise ComparisonContractError(f"git verification failed: {' '.join(args[1:])}")
    return result


def _blob(call: Callable[..., object], anchor: str, path: Path, repo_root: Path) -> bytes:
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ComparisonContractError("committed input is outside repo") from exc
    result = _git(call, ["git", "show", f"{anchor}:{relative}"], repo_root)
    stdout = getattr(result, "stdout", None)
    if isinstance(stdout, str):
        return stdout.encode("utf-8")
    if not isinstance(stdout, bytes):
        raise ComparisonContractError("git show did not return blob bytes")
    return stdout


def _validate_plan(value: dict) -> None:
    fields = {
        "schema",
        "source_anchor",
        "source_closure",
        "methods",
        "cases",
        "limits",
        "inputs",
        "backend",
        "authorization",
    }
    if set(value) != fields or value.get("schema") != SCHEMA:
        raise ComparisonContractError("plan fields differ")
    if _COMMIT.fullmatch(str(value["source_anchor"])) is None:
        raise ComparisonContractError("source anchor is invalid")
    closure = value["source_closure"]
    if not isinstance(closure, dict) or not closure:
        raise ComparisonContractError("source closure is invalid")
    for path, sha256 in closure.items():
        _relative_path(path, "source closure path")
        if _SHA.fullmatch(str(sha256)) is None:
            raise ComparisonContractError("source closure LF SHA is invalid")
    if value["methods"] != list(METHODS):
        raise ComparisonContractError("four methods are not fixed")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != {"max_checks", "api_cap", "watchdog_s"}:
        raise ComparisonContractError("limit fields differ")
    if limits["max_checks"] != 3 or type(limits["api_cap"]) is not int or not 1 <= limits["api_cap"] <= 40:
        raise ComparisonContractError("check or API cap differs")
    watchdog = limits["watchdog_s"]
    if isinstance(watchdog, bool) or not isinstance(watchdog, (int, float)):
        raise ComparisonContractError("watchdog is invalid")
    if not math.isfinite(watchdog) or not MIN_MODEL_WATCHDOG_S <= watchdog <= 3600:
        raise ComparisonContractError("watchdog is not bounded")
    cases = value["cases"]
    if not isinstance(cases, list) or not cases:
        raise ComparisonContractError("cases are missing")
    seen_cases: set[str] = set()
    for row in cases:
        if not isinstance(row, dict) or set(row) != {"case_id", "rotation"}:
            raise ComparisonContractError("case rotation fields differ")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise ComparisonContractError("case ID is missing or duplicated")
        seen_cases.add(case_id)
        if len(row["rotation"]) != 4 or set(row["rotation"]) != set(METHODS):
            raise ComparisonContractError("case rotation is not a four-arm permutation")
    inputs = value["inputs"]
    roles: set[str] = set()
    if not isinstance(inputs, list) or not inputs:
        raise ComparisonContractError("plan inputs are missing")
    for row in inputs:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256"}:
            raise ComparisonContractError("plan input fields differ")
        _relative_path(row["path"], "plan input path")
        role = row["role"]
        if not isinstance(role, str) or not role or role in roles or _SHA.fullmatch(str(row["sha256"])) is None:
            raise ComparisonContractError("plan input role or SHA is invalid")
        roles.add(role)
    for case_id in seen_cases:
        if f"protocol:{case_id}" not in roles and not (len(seen_cases) == 1 and "protocol" in roles):
            raise ComparisonContractError(f"protocol input is not registered for {case_id}")
    if value["backend"] != ACTUAL_BACKEND:
        raise ComparisonContractError("production backend is not fixed")
    policy = value["authorization"]
    if not isinstance(policy, dict) or set(policy) != {"actor", "probe_scope"}:
        raise ComparisonContractError("authorization policy fields differ")
    if not isinstance(policy["actor"], str) or not policy["actor"]:
        raise ComparisonContractError("authorization actor is missing")
    scope = policy["probe_scope"]
    if not isinstance(scope, list) or not scope or len(scope) != len(set(scope)) or any(
        not isinstance(probe, str) or not probe for probe in scope
    ):
        raise ComparisonContractError("probe scope is invalid")


def load_plan(
    plan_ref: dict,
    *,
    repo_root: Path,
    registration_anchor: str | None = None,
    git_call: Callable[..., object] = subprocess.run,
    require_committed: bool = True,
) -> dict:
    """Code anchor와 별도 registration anchor의 plan/input blob을 검증한다."""
    repo = Path(repo_root).resolve(strict=True)
    if not isinstance(plan_ref, dict) or set(plan_ref) != {"path", "sha256"}:
        raise ComparisonContractError("plan reference fields differ")
    relative_plan = _relative_path(plan_ref["path"], "plan path")
    if _SHA.fullmatch(str(plan_ref["sha256"])) is None:
        raise ComparisonContractError("plan reference LF SHA is invalid")
    plan_candidate = repo / relative_plan
    if plan_candidate.is_symlink():
        raise ComparisonContractError("plan path is a symlink")
    plan_path = plan_candidate.resolve(strict=True)
    if _lf_sha(plan_path) != plan_ref["sha256"]:
        raise ComparisonContractError("plan SHA changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ComparisonContractError("plan must be a JSON object")
    _validate_plan(plan)
    for row in plan["inputs"]:
        candidate = repo / _relative_path(row["path"], "plan input path")
        if candidate.is_symlink():
            raise ComparisonContractError(f"plan input is a symlink: {row['role']}")
        path = candidate.resolve(strict=True)
        if _lf_sha(path) != row["sha256"]:
            raise ComparisonContractError(f"plan input changed: {row['role']}")
    closure_paths = {_relative_path(path, "source closure path") for path in plan["source_closure"]}
    input_paths = {_relative_path(row["path"], "plan input path") for row in plan["inputs"]}
    if relative_plan in closure_paths or closure_paths & input_paths:
        raise ComparisonContractError("registration plan/inputs cannot self-bind in source closure")
    for relative_text, expected in plan["source_closure"].items():
        relative = _relative_path(relative_text, "source closure path")
        candidate = repo / relative
        if candidate.is_symlink():
            raise ComparisonContractError(f"source closure is a symlink: {relative.as_posix()}")
        current = candidate.resolve(strict=True)
        if _lf_sha(current) != expected:
            raise ComparisonContractError(f"source closure current file changed: {relative.as_posix()}")
    if require_committed:
        if registration_anchor is None or _COMMIT.fullmatch(registration_anchor) is None:
            raise ComparisonContractError("registration anchor is required for actual execution")
        code_anchor = plan["source_anchor"]
        _git(git_call, ["git", "merge-base", "--is-ancestor", code_anchor, "HEAD"], repo)
        _git(git_call, ["git", "merge-base", "--is-ancestor", registration_anchor, "HEAD"], repo)
        _git(git_call, ["git", "merge-base", "--is-ancestor", code_anchor, registration_anchor], repo)
        if hashlib.sha256(_lf_bytes(_blob(git_call, registration_anchor, plan_path, repo))).hexdigest() != plan_ref["sha256"]:
            raise ComparisonContractError("committed plan blob differs")
        for row in plan["inputs"]:
            path = (repo / _relative_path(row["path"], "plan input path")).resolve(strict=True)
            if hashlib.sha256(_lf_bytes(_blob(git_call, registration_anchor, path, repo))).hexdigest() != row["sha256"]:
                raise ComparisonContractError(f"committed input blob differs: {row['role']}")
        for relative, expected in plan["source_closure"].items():
            path = (repo / _relative_path(relative, "source closure path")).resolve(strict=True)
            if hashlib.sha256(_lf_bytes(_blob(git_call, code_anchor, path, repo))).hexdigest() != expected:
                raise ComparisonContractError(f"code anchor source closure differs: {relative}")
    return plan


def _under(path: Path, root: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ComparisonContractError(f"{label} must be absolute")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ComparisonContractError(f"{label} escapes output root") from exc
    if path.exists() and path.is_symlink():
        raise ComparisonContractError(f"{label} is a symlink")
    return resolved


def _validate_binding(
    binding: object, plan_ref: dict, plan: dict, output_root: Path, repo_root: Path
) -> dict:
    if not isinstance(binding, dict) or set(binding) != {"schema", "plan", "api_root", "cases"}:
        raise ComparisonContractError("job binding fields differ")
    if binding["schema"] != JOB_SCHEMA or binding["plan"] != plan_ref:
        raise ComparisonContractError("job plan binding differs")
    api_root = binding["api_root"]
    if not isinstance(api_root, str) or not api_root or "://" in api_root or api_root.startswith("\\\\"):
        raise ComparisonContractError("shared API root is invalid")
    shared_api = Path(api_root)
    if not shared_api.is_absolute() or (shared_api.exists() and shared_api.is_symlink()):
        raise ComparisonContractError("shared API root must be an absolute local path")
    shared_api = shared_api.resolve(strict=False)
    by_case = {row["case_id"]: row for row in plan["cases"]}
    rows = binding["cases"]
    if not isinstance(rows, list) or {row.get("case_id") for row in rows if isinstance(row, dict)} != set(by_case):
        raise ComparisonContractError("job cases differ from plan")
    normalized: dict[str, dict[str, dict]] = {}
    paths: set[Path] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"case_id", "arms"} or not isinstance(row["arms"], list):
            raise ComparisonContractError("job case fields differ")
        arms: dict[str, dict] = {}
        for arm in row["arms"]:
            if not isinstance(arm, dict) or set(arm) != {"method", "source_path", "session_path", "prepare_payload"}:
                raise ComparisonContractError("job arm fields differ")
            method = arm["method"]
            if method not in METHODS or method in arms or not isinstance(arm["prepare_payload"], dict):
                raise ComparisonContractError("job arm method or payload differs")
            source = _under(Path(arm["source_path"]), output_root, "source path")
            session = _under(Path(arm["session_path"]), output_root, "session path")
            expected = output_root / "bound" / row["case_id"] / method
            if source != expected / "source.json" or session != expected / "session":
                raise ComparisonContractError("arm source/session stage layout differs")
            if source in paths or session in paths or source == session:
                raise ComparisonContractError("arm stage paths are not isolated")
            paths.update((source, session))
            payload = deepcopy(arm["prepare_payload"])
            if payload.get("source_path") != str(source):
                raise ComparisonContractError("prepare source path differs")
            if payload.get("method") != method or payload.get("case_id", row["case_id"]) != row["case_id"]:
                raise ComparisonContractError("prepare case or method differs")
            if payload.get("max_checks") != plan["limits"]["max_checks"]:
                raise ComparisonContractError("prepare check budget differs")
            protocol_ref = payload.get("protocol")
            if payload.get("execution_class") in {"actual", "recorded"} and protocol_ref is None:
                raise ComparisonContractError("registered arm protocol is missing")
            if protocol_ref is not None:
                expected_protocol = _protocol_input(plan, row["case_id"])
                if not isinstance(protocol_ref, dict) or set(protocol_ref) != {"path", "sha256"}:
                    raise ComparisonContractError("arm protocol reference fields differ")
                private_path = Path(protocol_ref["path"])
                if not private_path.is_absolute():
                    raise ComparisonContractError("private protocol reference must be absolute")
                actual_path = private_path.resolve(strict=True)
                registered_path = (
                    repo_root / _relative_path(expected_protocol["path"], "case protocol path")
                ).resolve(strict=True)
                if actual_path != registered_path or _lf_sha(actual_path) != expected_protocol["sha256"]:
                    raise ComparisonContractError("arm protocol differs from case registration")
            if method in {"one_shot_llm", "iterative_agent"}:
                if payload.get("api_root") != str(shared_api):
                    raise ComparisonContractError("model arm API root differs from shared binding")
            elif payload.get("api_root") is not None:
                raise ComparisonContractError("non-model arm must not have an API root")
            arms[method] = {
                **arm,
                "case_id": row["case_id"],
                "source_path": str(source),
                "session_path": str(session),
                "prepare_payload": payload,
            }
        if set(arms) != set(METHODS):
            raise ComparisonContractError("job does not bind all four arms")
        normalized[row["case_id"]] = arms
    return normalized


def _rusage() -> dict | None:
    try:
        resource = importlib.import_module("resource")
        value = resource.getrusage(resource.RUSAGE_SELF)
        return {"user_s": value.ru_utime, "system_s": value.ru_stime, "max_rss": value.ru_maxrss}
    except (ImportError, OSError):
        return None


def _append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_once_or_match(path: Path, value: dict) -> None:
    if path.exists():
        _path, existing = _read_ref(reference(path), path.name)
        if existing != value:
            raise ComparisonContractError(f"existing {path.name} differs")
        return
    write_new(path, value)


def _invoke_step(
    journal: Path,
    label: str,
    call: Callable[[], dict],
    *,
    clock: Callable[[], float],
    cpu_clock: Callable[[], float],
) -> tuple[dict | None, Exception | None]:
    wall_start, cpu_start, usage_start = clock(), cpu_clock(), _rusage()
    result: dict | None = None
    error: Exception | None = None
    try:
        candidate = call()
        if not isinstance(candidate, dict):
            raise TypeError("investigate result is not an object")
        result = candidate
    except Exception as caught:  # noqa: BLE001 - transport ambiguity is recorded, never retried.
        error = caught
    event = {
        "schema": "diagnostic_comparison_timing_v1",
        "step": label,
        "wall_start_s": wall_start,
        "wall_end_s": clock(),
        "process_cpu_start_s": cpu_start,
        "process_cpu_end_s": cpu_clock(),
        "rusage_before": usage_start,
        "rusage_after": _rusage(),
        "status": "error" if error else "returned",
        "error_type": type(error).__name__ if error else None,
    }
    _append(journal, event)
    return result, error


def _raise_actual_integrity(error: Exception | None, fixture: bool) -> None:
    if not fixture and isinstance(error, (ComparisonContractError, ValueError)):
        raise ComparisonContractError(f"registered source/protocol integrity: {type(error).__name__}") from error


def _approval(
    stage: Path,
    *,
    plan: dict,
    protocol_ref: dict,
    source_path: Path,
    request: dict,
    fixture: bool,
) -> tuple[dict, dict]:
    probe = request.get("probe")
    if probe not in plan["authorization"]["probe_scope"]:
        raise PermissionError("pending probe is outside preregistered scope")
    actor = plan["authorization"]["actor"]
    source_ref = reference(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if fixture:
        body = {
            "schema": "diagnostic_comparison_fixture_approval_v1",
            "actor": actor,
            "request_id": request["request_id"],
            "effect_key": request["effect_key"],
            "probe": probe,
            "approved": True,
        }
        authorization: dict[str, Any] = {"fixture": True}
    else:
        body = {
            "schema": "registered_diagnostic_approval_v1",
            "approval_id": "diagnostic-comparison:" + digest(request),
            "protocol_sha256": protocol_ref["sha256"],
            "source_sha256": source_ref["sha256"],
            "case_id": source["case_id"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "probe": probe,
            "request_id": request["request_id"],
            "effect_key": request["effect_key"],
            "registry_sha256": request["registry_digest"],
            "actor": actor,
            "approved": True,
        }
        authorization = {"protocol": protocol_ref, "approval": {}, "source": source_ref}
    path = stage / "runner_approvals" / f"{request['request_id']}.json"
    _write_once_or_match(path, body)
    approval_ref = reference(path)
    if not fixture:
        authorization["approval"] = approval_ref
    return {"request_id": request["request_id"], "actor": actor,
            "authorization": authorization}, approval_ref


def _json_file(path: Path, label: str) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")),
    )
    if not isinstance(value, dict):
        raise ComparisonContractError(f"{label} must be a JSON object")
    canonical(value)
    return value


def _session_events(session: Path) -> list[dict]:
    path = session / "execution.jsonl"
    if not path.is_file() or path.is_symlink():
        return []
    events: list[dict] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON: {token}")
            ),
        )
        if not isinstance(value, dict):
            raise ComparisonContractError(f"execution event {index} is not an object")
        canonical(value)
        events.append(value)
    return events


def _number_or_none(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonContractError(f"{label} is not a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ComparisonContractError(f"{label} is not a finite non-negative number")
    return number


def _token_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _http_response_evidence(result: object) -> tuple[int | None, str | None]:
    if not isinstance(result, dict):
        return None, None
    fields = [key for key in ("http_status", "status_code") if key in result]
    known = [key for key in fields if result[key] is not None]
    if len(known) == 2 and result[known[0]] != result[known[1]]:
        raise ComparisonContractError("transport HTTP status fields disagree")
    if not fields:
        return None, None
    key = known[0] if known else fields[0]
    value = result[key]
    evidence_scope = f"safe_result.{key}"
    if value is None:
        return 0, evidence_scope
    if type(value) is int and 100 <= value <= 599:
        return 1, evidence_scope
    return None, evidence_scope


def _readonly_model_journal(
    root: Path,
    api_root: Path,
    session_manifest: dict,
    *,
    case_id: str,
    method: str,
    ordinal: int,
) -> tuple[dict, dict, Any]:
    """파일을 생성하는 생성자 없이 원장 validator를 복원한다."""
    journal_module = _decision_journal_module()
    manifest_path = root / "manifest.json"
    manifest = journal_module._read(manifest_path)
    binding = session_manifest.get("model_binding")
    verification = session_manifest.get("source_verification")
    if not isinstance(binding, dict) or not isinstance(verification, dict):
        raise ComparisonContractError("model journal has no session binding")
    expected_api_root = Path(str(binding["api_root"])) if "api_root" in binding else root.parents[1] / "api"
    expected_api_root = expected_api_root.resolve(strict=False)
    if api_root.resolve(strict=False) != expected_api_root:
        raise ComparisonContractError("model journal API root differs from session binding")
    decision_binding = manifest.get("binding")
    if not isinstance(decision_binding, dict) or (
        decision_binding.get("run_id") != f"{case_id}:{method}:{ordinal}"
        or decision_binding.get("source_anchor") != binding.get("source_anchor")
        or decision_binding.get("protocol_sha256") != binding.get("protocol_sha256")
        or decision_binding.get("checkpoint_sha256") != verification.get("checkpoint_sha256")
        or manifest.get("params") != binding.get("params")
        or manifest.get("api_cap") != binding.get("api_cap")
    ):
        raise ComparisonContractError("model journal binding differs from session")

    journal = object.__new__(journal_module.DecisionJournal)
    journal.root = root.resolve(strict=True)
    journal.api_root = api_root.resolve(strict=True)
    journal.binding = deepcopy(decision_binding)
    journal.params = deepcopy(manifest["params"])
    journal.retry_delay_s = float(manifest["retry_delay_s"])
    journal.api_cap = manifest["api_cap"]
    journal.manifest_path = manifest_path.resolve(strict=True)
    journal.api_manifest_path = (api_root / "manifest.json").resolve(strict=True)
    journal.manifest_sha256 = journal_module.file_sha(journal.manifest_path)
    journal.binding_sha256 = journal_module.digest(journal.binding)
    journal.params_sha256 = journal_module.digest(journal.params)
    try:
        journal._verify_prepared()
        audit = journal._audit_api()
    except journal_module.JournalError as error:
        raise ComparisonContractError("model journal integrity audit failed") from error

    terminal_path = root / "terminal.json"
    terminal = None
    if terminal_path.exists():
        try:
            terminal = journal_module._read(terminal_path)
        except journal_module.JournalError as error:
            raise ComparisonContractError("model terminal integrity audit failed") from error
        required = {
            "schema", "manifest_sha256", "status", "engine_result", "calls_sha256",
            "completed_calls",
        }
        completed = [row["response"]["result"] for row in audit["own"] if row["response"] is not None]
        if (
            set(terminal) != required
            or terminal.get("schema") != "model_decision_terminal_v1"
            or terminal.get("manifest_sha256") != journal.manifest_sha256
            or terminal.get("status") not in journal_module.TERMINAL_STATUSES
            or not isinstance(terminal.get("engine_result"), dict)
            or terminal.get("completed_calls") != len(completed)
            or terminal.get("calls_sha256") != journal_module.digest(completed)
        ):
            raise ComparisonContractError("model terminal/API snapshot binding differs")
    return manifest, audit, terminal


def _model_costs(
    session: Path,
    session_manifest: dict,
    max_checks: int,
    *,
    case_id: str,
    method: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    binding = session_manifest.get("model_binding")
    if binding is None:
        return [], [], []
    if not isinstance(binding, dict):
        raise ComparisonContractError("session model binding is invalid")
    api_root = Path(str(binding.get("api_root", session / "api")))
    if not api_root.is_absolute() or api_root.is_symlink():
        raise ComparisonContractError("session API root is invalid")

    events: list[dict] = []
    journals: list[dict] = []
    flat_refs: list[dict] = []
    seen_reservations: set[str] = set()
    for ordinal in range(max_checks):
        root = session / "model" / f"{ordinal:04d}"
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            continue
        if manifest_path.is_symlink():
            raise ComparisonContractError("model journal manifest is a symlink")
        manifest, audit, terminal = _readonly_model_journal(
            root, api_root, session_manifest, case_id=case_id, method=method, ordinal=ordinal
        )
        journal_refs: dict[str, Any] = {
            "manifest": reference(manifest_path),
            "api_manifest": reference(api_root / "manifest.json"),
            "terminal": reference(root / "terminal.json") if terminal is not None else None,
            "attempts": [],
        }
        flat_refs.extend([journal_refs["manifest"], journal_refs["api_manifest"]])
        if journal_refs["terminal"] is not None:
            flat_refs.append(journal_refs["terminal"])
        for row in audit["own"]:
            reservation_path = row["reservation_path"]
            reservation = row["reservation"]
            response_path = row["response_path"]
            response = row["response"]
            reservation_sha = file_sha(reservation_path)
            if reservation_sha in seen_reservations:
                raise ComparisonContractError("model API reservation was counted twice")
            seen_reservations.add(reservation_sha)
            response_ref = reference(response_path) if response_path is not None else None
            receipt_path = api_root / "receipts" / reservation_path.name
            receipt_ref = reference(receipt_path) if response_path is not None else None
            attempt_refs = {
                "reservation": reference(reservation_path),
                "response": response_ref,
                "receipt": receipt_ref,
            }
            journal_refs["attempts"].append(attempt_refs)
            flat_refs.extend(ref for ref in attempt_refs.values() if ref is not None)

            result = response["result"] if response is not None else None
            usage = result.get("usage") if isinstance(result, dict) else None
            raw_usage = deepcopy(usage) if isinstance(usage, dict) else None
            input_tokens = _token_or_none(usage.get("prompt_tokens")) if isinstance(usage, dict) else None
            output_tokens = (
                _token_or_none(usage.get("completion_tokens")) if isinstance(usage, dict) else None
            )
            response_wall = (
                _number_or_none(result.get("wall_s"), "model response wall_s")
                if isinstance(result, dict) and "wall_s" in result
                else None
            )
            confirmed_http_response, http_evidence_scope = _http_response_evidence(result)
            safe_result = result if isinstance(result, dict) else {}
            result_class = (
                "unconfirmed"
                if response is None
                else "infra_error"
                if response["status"] == "infra_error"
                else "failed"
                if safe_result.get("status") == "error"
                or isinstance(safe_result.get("error_type"), str)
                else "normal"
            )
            status = (
                "unknown" if result_class == "unconfirmed"
                else "completed" if result_class == "normal"
                else "failed"
            )
            events.append(
                {
                    "schema": "diagnostic_resource_cost_event_v1",
                    "phase_id": (
                        f"model:{case_id}:{method}:{ordinal}:{reservation['index']}"
                    ),
                    "attempt_id": f"model:{reservation_sha}",
                    "receipt_id": f"model:{reservation_sha}",
                    "status": status,
                    "costs": {
                        "model_calls": 1 if response is not None else None,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "gpu_wall_s": 0.0,
                        "cpu_s": None,
                        "human_review_s": None,
                        "execution_wall_s": response_wall,
                        "simulator_steps": 0,
                    },
                    "model_api": {
                        "model_requested": session_manifest["model_binding"]["params"].get("model"),
                        "model_returned": safe_result.get("model_returned"),
                        "finish_reason": safe_result.get("finish_reason"),
                        "global_index": reservation["index"],
                        "decision_manifest_sha256": file_sha(manifest_path),
                        "reservation_sha256": reservation_sha,
                        "reservation_consumed_budget": True,
                        "completion_attempt_returned": response is not None,
                        "result_class": result_class,
                        "model_call_scope": (
                            "completion_attempt_returned"
                            if response is not None
                            else "unconfirmed_after_reservation"
                        ),
                        "confirmed_http_response": confirmed_http_response,
                        "http_response_evidence_scope": http_evidence_scope,
                        "response_wall_s": response_wall,
                        "response_wall_scope": "nim_transport.result.wall_s" if response_wall is not None else None,
                        "usage": raw_usage,
                        "hosted_provider_gpu_s": None,
                        "actual_billing_usd": None,
                        "reproducible_estimate_usd": None,
                        "raw_refs": attempt_refs,
                    },
                }
            )
        journals.append(
            {
                "ordinal": ordinal,
                "run_id": manifest["binding"]["run_id"],
                "reserved_global": audit["reserved"],
                "pending_global": audit["pending"],
                "own_attempts": len(audit["own"]),
                "refs": journal_refs,
            }
        )
    unique_refs = {item["path"]: item for item in flat_refs}
    return events, journals, list(unique_refs.values())


def _request_cost(
    session: Path,
    session_manifest: dict,
    request: dict,
    logical_receipt: dict | None,
) -> tuple[dict, dict | None]:
    request_id = request["request_id"]
    effect_key = request["effect_key"]
    source_ref = session_manifest.get("source")
    source_sha = source_ref.get("sha256") if isinstance(source_ref, dict) else None
    if not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None:
        raise ComparisonContractError("session source SHA is invalid")
    stage = session / "actual_stages" / request_id
    host_path = session / "host_receipts" / f"{request_id}.json"
    host = _json_file(host_path, "host receipt") if host_path.is_file() and not host_path.is_symlink() else None
    host_ref = reference(host_path) if host is not None else None
    if host is not None and (
        host.get("request_id") != request_id
        or host.get("effect_key") != effect_key
        or host.get("source_sha256") != source_sha
    ):
        raise ComparisonContractError("host receipt request/effect/source binding differs")
    logical_gpu_s = _number_or_none(
        logical_receipt.get("gpu_s") if logical_receipt is not None else None,
        "logical settlement gpu_s",
    )
    if logical_receipt is not None and logical_receipt.get("effect_key") != effect_key:
        raise ComparisonContractError("logical settlement effect binding differs")
    logical = {
        "receipt_present": host is not None,
        "receipt_ref": host_ref,
        "ledger_settlement_present": logical_receipt is not None,
        "ledger_settlement_gpu_s": logical_gpu_s,
        "ledger_outcome": logical_receipt.get("outcome") if logical_receipt else None,
    }

    if not stage.exists():
        execution_class = session_manifest.get("execution_class")
        if host is not None:
            assert host_ref is not None
            gpu_wall = _number_or_none(host.get("gpu_s"), "host receipt gpu_s")
            execution_wall = _number_or_none(
                host.get("execution_wall_s"), "host receipt execution_wall_s"
            )
            event_status = "completed"
            receipt_id = f"{request_id}:{host_ref['sha256']}"
            missing: list[str] = []
        else:
            gpu_wall = None
            execution_wall = None
            event_status = "unknown"
            receipt_id = f"missing:{request_id}"
            missing = ["actual_stage_and_host_receipt_missing"]
        owned_count: int | None = 0 if execution_class in {"fixture", "recorded"} else None
        resource = {
            "schema": "diagnostic_request_resource_cost_v1",
            "phase_id": f"request:{request_id}:{effect_key}",
            "request_id": request_id,
            "effect_key": effect_key,
            "source_sha256": source_sha,
            "logical_settlement": logical,
            "actual_owned": {
                "stage_present": False,
                "owned_sdk_execution_count": owned_count,
                "owned_wall_s": None if owned_count is None else 0.0,
                "natural_exit_count": None if owned_count is None else 0,
                "non_natural_exit_count": None if owned_count is None else 0,
                "failure_count": None if owned_count is None else 0,
                "incomplete_count": None if owned_count is None else 0,
                "missing_reasons": missing,
            },
            "effective_cost_source": "host_receipt" if host is not None else None,
            "double_count_prevented": True,
            "raw_refs": {"host_receipt": host_ref, "stage": None},
        }
        return resource, {
            "schema": "diagnostic_resource_cost_event_v1",
            "phase_id": resource["phase_id"],
            "attempt_id": request_id,
            "receipt_id": receipt_id,
            "status": event_status,
            "costs": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "gpu_wall_s": gpu_wall,
                "cpu_s": None,
                "human_review_s": None,
                "execution_wall_s": execution_wall,
                "simulator_steps": None,
            },
        }
    if not stage.is_dir() or stage.is_symlink():
        raise ComparisonContractError("actual stage is not a real directory")

    selected = {
        "manifest": stage / "manifest.json",
        "execution": stage / "execution.json",
        "failure": stage / "output" / "failure.json",
        "training_failure": stage / "output" / "training_failure.json",
        "training_evidence": stage / "output" / "training_evidence.json",
        "records": stage / "output" / "records.jsonl",
    }
    stage_refs = {
        name: reference(path)
        for name, path in selected.items()
        if path.is_file() and not path.is_symlink()
    }
    request_path = stage / "request.json"
    if not request_path.is_file() or request_path.is_symlink():
        raise ComparisonContractError("actual stage request binding is missing")
    stage_request = _json_file(request_path, "actual stage request")
    nested = stage_request.get("request")
    if (
        stage_request.get("request_id") != request_id
        or stage_request.get("effect_key") != effect_key
        or not isinstance(nested, dict)
        or nested.get("request_id") != request_id
        or nested.get("effect_key") != effect_key
    ):
        raise ComparisonContractError("actual stage request/effect binding differs")
    stage_refs["request"] = reference(request_path)

    missing_reasons: list[str] = []
    stage_manifest = None
    manifest_path = selected["manifest"]
    if manifest_path.is_file() and not manifest_path.is_symlink():
        stage_manifest = _json_file(manifest_path, "actual stage manifest")
        source = stage_manifest.get("source")
        if (
            stage_manifest.get("request_id") != request_id
            or stage_manifest.get("effect_key") != effect_key
            or not isinstance(source, dict)
            or source.get("source_sha256") != source_sha
        ):
            raise ComparisonContractError("actual stage manifest request/effect/source differs")
    else:
        missing_reasons.append("stage_manifest_missing")

    execution = None
    execution_path = selected["execution"]
    if execution_path.is_file() and not execution_path.is_symlink():
        execution = _json_file(execution_path, "actual stage execution")
    else:
        missing_reasons.append("stage_execution_missing")

    failure_artifacts = [name for name in ("failure", "training_failure") if name in stage_refs]
    if execution is None:
        owned_count = owned_wall = natural_count = non_natural_count = None
        failure_count = incomplete_count = None
        effective_gpu = effective_execution = None
        receipt_id = f"missing-execution:{request_id}"
        event_status = "unknown"
    else:
        owned_wall = _number_or_none(execution.get("wall_s"), "owned execution wall_s")
        natural = execution.get("natural_exit")
        if not isinstance(natural, bool):
            raise ComparisonContractError("owned execution natural_exit is not bool")
        owned_count = 1
        natural_count = int(natural)
        non_natural_count = int(not natural)
        failure_count = int(bool(failure_artifacts))
        incomplete_count = int(bool(failure_artifacts) or host is None)
        effective_gpu = owned_wall
        effective_execution = owned_wall
        receipt_id = f"{request_id}:{stage_refs['execution']['sha256']}"
        event_status = "failed" if failure_count else "completed" if host is not None else "unknown"
        if host is not None:
            host_gpu = _number_or_none(host.get("gpu_s"), "host receipt gpu_s")
            host_wall = _number_or_none(host.get("execution_wall_s"), "host receipt execution_wall_s")
            if host_gpu != owned_wall or host_wall != owned_wall:
                raise ComparisonContractError("host receipt and actual owned wall differ")

    resource = {
        "schema": "diagnostic_request_resource_cost_v1",
        "phase_id": f"request:{request_id}:{effect_key}",
        "request_id": request_id,
        "effect_key": effect_key,
        "source_sha256": source_sha,
        "logical_settlement": logical,
        "actual_owned": {
            "stage_present": True,
            "owned_sdk_execution_count": owned_count,
            "owned_wall_s": owned_wall,
            "natural_exit_count": natural_count,
            "non_natural_exit_count": non_natural_count,
            "failure_count": failure_count,
            "incomplete_count": incomplete_count,
            "failure_artifact_count": len(failure_artifacts),
            "missing_reasons": missing_reasons,
        },
        "effective_cost_source": "actual_stage_execution" if execution is not None else None,
        "double_count_prevented": True,
        "raw_refs": {"host_receipt": host_ref, "stage": stage_refs},
    }
    event = {
        "schema": "diagnostic_resource_cost_event_v1",
        "phase_id": resource["phase_id"],
        "attempt_id": request_id,
        "receipt_id": receipt_id,
        "status": event_status,
        "costs": {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "gpu_wall_s": effective_gpu,
            "cpu_s": None,
            "human_review_s": None,
            "execution_wall_s": effective_execution,
            "simulator_steps": None,
        },
    }
    return resource, event


def _raw_refs(session: Path, max_checks: int, *, case_id: str, method: str) -> dict:
    refs: dict[str, Any] = {
        "host_receipts": [], "api_journals": [], "model_journals": [], "actual_stages": []
    }
    receipt_dir = session / "host_receipts"
    if receipt_dir.is_dir() and not receipt_dir.is_symlink():
        for path in sorted(receipt_dir.glob("*.json")):
            if path.is_file() and not path.is_symlink():
                refs["host_receipts"].append(reference(path))
    manifest_path = session / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        refs["resource_cost"] = {
            "schema": "diagnostic_arm_resource_cost_v1",
            "phase_id": f"arm:{case_id}:{method}",
            "events": [],
            "missing_reasons": ["session_manifest_missing"],
        }
        return refs
    session_manifest = _json_file(manifest_path, "session manifest")
    model_events, model_journals, api_refs = _model_costs(
        session, session_manifest, max_checks, case_id=case_id, method=method
    )
    refs["model_journals"] = model_journals
    refs["api_journals"] = api_refs
    events = _session_events(session)
    requests: dict[str, dict] = {}
    receipts: dict[str, dict] = {}
    for event in events:
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            continue
        if event.get("event") == "proposed":
            requests[request_id] = event
        elif event.get("event") == "receipt":
            receipts[request_id] = event
    cost_events: list[dict] = []
    for request_id, request in requests.items():
        resource, cost_event = _request_cost(
            session, session_manifest, request, receipts.get(request_id)
        )
        refs["actual_stages"].append(resource)
        if cost_event is not None:
            cost_event["attempt_id"] = f"{case_id}:{method}:{cost_event['attempt_id']}"
            cost_event["receipt_id"] = f"{case_id}:{method}:{cost_event['receipt_id']}"
            cost_events.append(cost_event)
    cost_events.extend(model_events)
    actual_owned = [row["actual_owned"] for row in refs["actual_stages"]]

    def total(field: str) -> int | float | None:
        values = [row[field] for row in actual_owned]
        return None if any(value is None for value in values) else sum(values)

    def known_total(field: str) -> int | float:
        return sum(row[field] for row in actual_owned if row[field] is not None)

    refs["resource_cost"] = {
        "schema": "diagnostic_arm_resource_cost_v1",
        "phase_id": f"arm:{case_id}:{method}",
        "events": cost_events,
        "request_costs": refs["actual_stages"],
        "actual_owned_summary": {
            "owned_sdk_execution_count": total("owned_sdk_execution_count"),
            "whole_owned_wall_s": total("owned_wall_s"),
            "failure_count": total("failure_count"),
            "incomplete_count": total("incomplete_count"),
            "known_owned_sdk_execution_count": known_total("owned_sdk_execution_count"),
            "known_owned_wall_s": known_total("owned_wall_s"),
            "missing_request_count": sum(
                any(value is None for value in (
                    row["owned_sdk_execution_count"], row["owned_wall_s"],
                    row["failure_count"], row["incomplete_count"],
                ))
                for row in actual_owned
            ),
        },
        "model_api_summary": {
            "reserved_attempts": len(model_events),
            "completed_attempts": sum(
                event["model_api"]["completion_attempt_returned"] for event in model_events
            ),
            "normal_results": sum(
                event["model_api"]["result_class"] == "normal" for event in model_events
            ),
            "infra_error_results": sum(
                event["model_api"]["result_class"] == "infra_error" for event in model_events
            ),
            "failed_results": sum(
                event["model_api"]["result_class"] == "failed" for event in model_events
            ),
            "unconfirmed_reservations": sum(
                event["model_api"]["result_class"] == "unconfirmed" for event in model_events
            ),
            "model_calls": (
                None
                if any(event["costs"]["model_calls"] is None for event in model_events)
                else sum(event["costs"]["model_calls"] for event in model_events)
            ),
            "known_model_calls": sum(
                event["costs"]["model_calls"]
                for event in model_events
                if event["costs"]["model_calls"] is not None
            ),
            "confirmed_http_responses": sum(
                event["model_api"]["confirmed_http_response"] == 1 for event in model_events
            ),
            "confirmed_no_http_responses": sum(
                event["model_api"]["confirmed_http_response"] == 0 for event in model_events
            ),
            "unknown_http_response_count": sum(
                event["model_api"]["confirmed_http_response"] is None for event in model_events
            ),
            "input_tokens": (
                0 if not model_events else None
                if any(event["costs"]["input_tokens"] is None for event in model_events)
                else sum(event["costs"]["input_tokens"] for event in model_events)
            ),
            "output_tokens": (
                0 if not model_events else None
                if any(event["costs"]["output_tokens"] is None for event in model_events)
                else sum(event["costs"]["output_tokens"] for event in model_events)
            ),
            "hosted_provider_gpu_s": None,
            "actual_billing_usd": None,
            "reproducible_estimate_usd": None,
        },
        "aggregation": "one_effective_event_per_request_stage_preferred_over_host_receipt",
    }
    return refs


def _protocol_input(plan: dict, case_id: str) -> dict:
    exact = next((row for row in plan["inputs"] if row["role"] == f"protocol:{case_id}"), None)
    if exact is not None:
        return exact
    generic = [row for row in plan["inputs"] if row["role"] == "protocol"]
    if len(plan["cases"]) == 1 and len(generic) == 1:
        return generic[0]
    raise ComparisonContractError(f"protocol input is ambiguous for {case_id}")


def _validate_child_watchdogs(protocol: dict, payload: dict, plan_watchdog_s: float) -> None:
    components = protocol.get("components")
    resolver = payload.get("artifact_resolver")
    if not isinstance(components, dict) or not isinstance(resolver, dict):
        raise ComparisonContractError("protocol components or artifact resolver differ")
    for component in components.values():
        if not isinstance(component, dict) or component.get("execution_class") != "actual":
            continue
        artifacts = component.get("artifacts")
        execution = artifacts.get("execution") if isinstance(artifacts, dict) else None
        artifact_id = execution.get("artifact_id") if isinstance(execution, dict) else None
        configured = resolver.get(artifact_id) if isinstance(artifact_id, str) else None
        if not isinstance(configured, dict):
            raise ComparisonContractError("actual child execution config is not resolved")
        _config_path, config = _read_ref(configured, "actual child execution config")
        watchdog = config.get("watchdog_s")
        if isinstance(watchdog, bool) or not isinstance(watchdog, (int, float)):
            raise ComparisonContractError("actual child watchdog is invalid")
        if not math.isfinite(watchdog) or not 0 < watchdog <= plan_watchdog_s:
            raise ComparisonContractError("actual child watchdog exceeds plan")


def _run_arm(
    plan: dict,
    arm: dict,
    *,
    repo_root: Path,
    stage: Path,
    fixture: bool,
    invoke: Callable[..., dict],
    clock: Callable[[], float],
    cpu_clock: Callable[[], float],
    gpu_sampler: GpuSampler | None,
) -> dict:
    stage.mkdir(parents=True, exist_ok=True)
    journal = stage / "controller-timing.jsonl"
    session = Path(arm["session_path"])
    payload = deepcopy(arm["prepare_payload"])
    backend = FIXTURE_BACKEND if fixture else ACTUAL_BACKEND
    protocol_ref = payload.get("protocol")
    if not fixture and (not isinstance(protocol_ref, dict) or set(protocol_ref) != {"path", "sha256"}):
        raise ComparisonContractError("production prepare protocol reference differs")
    if fixture:
        protocol_ref = {"path": "fixture", "sha256": "0" * 64}
    else:
        registered_protocol = _protocol_input(plan, arm["case_id"])
        expected_protocol = (repo_root / _relative_path(
            registered_protocol["path"], "case protocol path"
        )).resolve(strict=True)
        if not isinstance(protocol_ref, dict) or Path(protocol_ref["path"]).resolve(strict=True) != expected_protocol:
            raise ComparisonContractError("prepare protocol differs from the plan input")
        protocol_path, protocol = _read_ref(protocol_ref, "prepare protocol")
        if _lf_sha(protocol_path) != registered_protocol["sha256"]:
            raise ComparisonContractError("prepare protocol LF SHA differs from plan")
        model = protocol.get("model")
        actors = protocol.get("actors")
        components = protocol.get("components")
        if not isinstance(model, dict) or model.get("api_cap") != plan["limits"]["api_cap"]:
            raise ComparisonContractError("protocol API cap differs from plan")
        if model.get("source_anchor") != plan["source_anchor"]:
            raise ComparisonContractError("protocol code source anchor differs from plan")
        if not isinstance(actors, list) or plan["authorization"]["actor"] not in actors:
            raise ComparisonContractError("protocol actor differs from plan")
        if not isinstance(components, dict) or not set(plan["authorization"]["probe_scope"]) <= set(components):
            raise ComparisonContractError("protocol probe scope differs from plan")
        _validate_child_watchdogs(protocol, payload, float(plan["limits"]["watchdog_s"]))
    started_existing = session.exists()
    end_to_end_start, cpu_start, usage_start = clock(), cpu_clock(), _rusage()
    deadline = end_to_end_start + float(plan["limits"]["watchdog_s"])
    sampler_start = None
    sampler_stop = None
    if gpu_sampler is not None:
        if fixture or payload.get("execution_class") != "actual":
            raise ComparisonContractError("GPU sampler is allowed only for actual execution")
        sampler_start = gpu_sampler.start()
        if not isinstance(sampler_start, dict) or sampler_start.get("scope") != "device_wide":
            raise ComparisonContractError("GPU sampler start scope differs")
    result: dict[str, Any] = {"method": arm["method"], "status": "not_started"}
    try:
        if clock() > deadline:
            result = {**result, "status": "hold_watchdog", "deadline_s": deadline}
            return result
        if not session.exists():
            prepare_attempt = stage / "prepare-attempt.json"
            if prepare_attempt.exists():
                result = {**result, "status": "hold_prepare_unknown", "reexecute_allowed": False}
                return result
            _write_once_or_match(prepare_attempt, {
                "schema": "diagnostic_comparison_prepare_attempt_v1",
                "method": arm["method"],
                "reexecute_allowed": False,
            })
            prepared, error = _invoke_step(
                journal,
                "prepare",
                lambda: invoke("prepare", session, payload, backend_name=backend, fixture=fixture),
                clock=clock,
                cpu_clock=cpu_clock,
            )
            if error or prepared is None:
                _raise_actual_integrity(error, fixture)
                result = {**result, "status": "prepare_transport_failure",
                          "error_type": type(error).__name__ if error else "MissingResult"}
                return result
            if prepared.get("status") in {"blocked_backend", "rejected"}:
                result = {**result, "status": prepared["status"], "prepare": prepared}
                return result

        just_dispatched = False
        for _transition in range(32):
            if clock() > deadline:
                result = {**result, "status": "hold_watchdog", "deadline_s": deadline}
                break
            state, error = _invoke_step(
                journal,
                "status",
                lambda: invoke("status", session, {}, backend_name=backend, fixture=fixture),
                clock=clock,
                cpu_clock=cpu_clock,
            )
            if error or state is None:
                _raise_actual_integrity(error, fixture)
                result = {**result, "status": "status_failure",
                          "error_type": type(error).__name__ if error else "MissingResult"}
                return result
            if clock() > deadline:
                result = {**result, "status": "hold_watchdog", "deadline_s": deadline,
                          "state": state}
                break
            status = state.get("status")
            if status in TERMINAL:
                result = {**result, "status": status, "state": state}
                break
            if status == "open":
                next_ordinal = len(state.get("observed", {}))
                next_attempt = stage / "next_attempts" / f"{next_ordinal:04d}.json"
                if next_attempt.exists():
                    result = {**result, "status": "hold_next_unknown", "reexecute_allowed": False}
                    break
                _write_once_or_match(next_attempt, {
                    "schema": "diagnostic_comparison_next_attempt_v1",
                    "ordinal": next_ordinal,
                    "reexecute_allowed": False,
                })
                next_state, error = _invoke_step(
                    journal,
                    "next",
                    lambda: invoke("next", session, {}, backend_name=backend, fixture=fixture),
                    clock=clock,
                    cpu_clock=cpu_clock,
                )
                if error or next_state is None:
                    _raise_actual_integrity(error, fixture)
                    result = {**result, "status": "next_transport_failure",
                              "error_type": type(error).__name__ if error else "MissingResult"}
                    return result
                if clock() > deadline:
                    result = {**result, "status": "hold_watchdog", "deadline_s": deadline,
                              "state": next_state}
                    break
                if next_state.get("status") in TERMINAL or str(next_state.get("status", "")).startswith("hold_"):
                    result = {**result, "status": next_state["status"], "state": next_state}
                    break
                continue
            pending = state.get("pending")
            if not isinstance(pending, dict) or not isinstance(pending.get("request_id"), str):
                result = {**result, "status": "state_contract_failure", "state": state}
                break
            request_id = pending["request_id"]
            if status == "approval_pending":
                try:
                    approval_payload, approval_ref = _approval(
                        stage,
                        plan=plan,
                        protocol_ref=protocol_ref,
                        source_path=Path(arm["source_path"]),
                        request=pending,
                        fixture=fixture,
                    )
                except PermissionError:
                    result = {**result, "status": "rejected_probe_scope", "request": pending}
                    break
                approval_attempt = stage / "approval_attempts" / f"{request_id}.json"
                if approval_attempt.exists():
                    result = {**result, "status": "hold_approval_unknown",
                              "request_id": request_id, "reexecute_allowed": False}
                    break
                _write_once_or_match(approval_attempt, {
                    "schema": "diagnostic_comparison_approval_attempt_v1",
                    "request_id": request_id,
                    "effect_key": pending["effect_key"],
                    "reexecute_allowed": False,
                })
                approved, error = _invoke_step(
                    journal,
                    "approval",
                    lambda approval_payload=approval_payload: invoke(
                        "approve", session, approval_payload, backend_name=backend, fixture=fixture
                    ),
                    clock=clock,
                    cpu_clock=cpu_clock,
                )
                if error or approved is None:
                    _raise_actual_integrity(error, fixture)
                    result = {**result, "status": "approval_transport_failure",
                              "approval": approval_ref,
                              "error_type": type(error).__name__ if error else "MissingResult"}
                    return result
                continue
            if status == "ready_to_dispatch":
                attempt = stage / "dispatch_attempts" / f"{request_id}.json"
                if attempt.exists():
                    result = {**result, "status": "hold_transport_unknown",
                              "request_id": request_id, "reexecute_allowed": False}
                    break
                _write_once_or_match(attempt, {
                    "schema": "diagnostic_comparison_dispatch_attempt_v1",
                    "request_id": request_id,
                    "effect_key": pending["effect_key"],
                    "reexecute_allowed": False,
                })
                dispatched, error = _invoke_step(
                    journal,
                    "dispatch",
                    lambda request_id=request_id: invoke(
                        "dispatch",
                        session,
                        {"request_id": request_id},
                        backend_name=backend,
                        fixture=fixture,
                    ),
                    clock=clock,
                    cpu_clock=cpu_clock,
                )
                if error or dispatched is None:
                    result = {**result, "status": "dispatch_transport_unknown",
                              "request_id": request_id, "reexecute_allowed": False,
                              "error_type": type(error).__name__ if error else "MissingResult"}
                    return result
                if clock() > deadline:
                    result = {**result, "status": "hold_watchdog", "deadline_s": deadline,
                              "state": dispatched, "reexecute_allowed": False}
                    break
                just_dispatched = True
                if dispatched.get("status") == "pending_result":
                    dispatch_error = session / "dispatch_errors" / f"{request_id}.json"
                    if not fixture and dispatch_error.is_file():
                        error_body = json.loads(dispatch_error.read_text(encoding="utf-8"))
                        if error_body.get("error_type") in {"BackendBlocked", "ValueError"}:
                            raise ComparisonContractError("registered source or lease integrity failure")
                    result = {**result, "status": "resume_required", "state": dispatched,
                              "request_id": request_id, "reexecute_allowed": False}
                    break
                continue
            if status == "pending_result":
                if just_dispatched and not started_existing:
                    result = {**result, "status": "resume_required", "state": state,
                              "request_id": request_id, "reexecute_allowed": False}
                    break
                receipt_path = session / "host_receipts" / f"{request_id}.json"
                receipt = None
                if receipt_path.is_file() and not receipt_path.is_symlink():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                settle_attempt = stage / "settle_attempts" / f"{request_id}.json"
                if settle_attempt.exists():
                    result = {**result, "status": "hold_settle_unknown",
                              "request_id": request_id, "reexecute_allowed": False}
                    break
                _write_once_or_match(settle_attempt, {
                    "schema": "diagnostic_comparison_settle_attempt_v1",
                    "request_id": request_id,
                    "receipt_present": receipt is not None,
                    "reexecute_allowed": False,
                })
                settled, settle_error = _invoke_step(
                    journal,
                    "settle",
                    lambda request_id=request_id, receipt=receipt: invoke(
                        "settle",
                        session,
                        {"request_id": request_id, "host_receipt": receipt},
                        backend_name=backend,
                        fixture=fixture,
                    ),
                    clock=clock,
                    cpu_clock=cpu_clock,
                )
                if settle_error or settled is None:
                    _raise_actual_integrity(settle_error, fixture)
                    result = {**result, "status": "settle_failure", "request_id": request_id,
                              "error_type": type(settle_error).__name__ if settle_error else "MissingResult"}
                    return result
                if clock() > deadline:
                    result = {**result, "status": "hold_watchdog", "deadline_s": deadline,
                              "state": settled, "settled_without_reexecution": True}
                    break
                result = {**result, "status": settled.get("status", "hold_unknown"),
                          "state": settled, "settled_without_reexecution": True}
                if receipt is None:
                    break
                continue
            result = {**result, "status": "state_contract_failure", "state": state}
            break
        else:
            result = {**result, "status": "controller_transition_limit"}

        export_path = stage / "session-export.json"
        if session.exists() and not export_path.exists():
            exported, export_error = _invoke_step(
                journal,
                "export",
                lambda: invoke(
                    "export", session, {"path": str(export_path)}, backend_name=backend, fixture=fixture
                ),
                clock=clock,
                cpu_clock=cpu_clock,
            )
            if export_error:
                result["export_error"] = type(export_error).__name__
            elif exported is not None:
                result["export"] = reference(export_path)
        elif export_path.is_file():
            result["export"] = reference(export_path)
        if session.exists():
            raw = _raw_refs(
                session,
                plan["limits"]["max_checks"],
                case_id=arm["case_id"],
                method=arm["method"],
            )
            result["raw_refs"] = raw
            result["resource_cost"] = raw["resource_cost"]
        return result
    finally:
        if gpu_sampler is not None and sampler_start is not None:
            sampler_stop = gpu_sampler.stop()
        result["timing"] = {
            "end_to_end_start_s": end_to_end_start,
            "end_to_end_end_s": clock(),
            "process_cpu_start_s": cpu_start,
            "process_cpu_end_s": cpu_clock(),
            "rusage_before": usage_start,
            "rusage_after": _rusage(),
        }
        if journal.is_file():
            result["controller_timing"] = reference(journal)
        if session.exists():
            raw = _raw_refs(
                session,
                plan["limits"]["max_checks"],
                case_id=arm["case_id"],
                method=arm["method"],
            )
            result["raw_refs"] = raw
            result["resource_cost"] = raw["resource_cost"]
        result["gpu_sampling"] = {"start": sampler_start, "stop": sampler_stop,
                                  "scope": "device_wide" if sampler_start else None}


def run_comparison(
    plan_ref: dict,
    binding: dict,
    *,
    repo_root: Path,
    output_root: Path,
    registration_anchor: str | None = None,
    fixture: bool = False,
    invoke: Callable[..., dict] = investigate.run,
    git_call: Callable[..., object] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
    gpu_sampler: GpuSampler | None = None,
) -> dict:
    """각 case의 preregistered rotation을 순차 실행하고 독립 arm 실패를 보존한다."""
    repo = Path(repo_root).resolve(strict=True)
    root = Path(output_root).resolve(strict=False)
    if root.exists() and root.is_symlink():
        raise ComparisonContractError("output root is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    plan = load_plan(
        plan_ref,
        repo_root=repo,
        registration_anchor=registration_anchor,
        git_call=git_call,
        require_committed=not fixture,
    )
    if fixture and plan["authorization"]["actor"] != "fixture_actor":
        raise ComparisonContractError("fixture actor must be fixture_actor")
    arms = _validate_binding(binding, plan_ref, plan, root, repo)
    batch_start, cpu_start = clock(), cpu_clock()
    results: list[dict] = []
    batch_status = "completed"
    for case in plan["cases"]:
        case_id = case["case_id"]
        for ordinal, method in enumerate(case["rotation"]):
            stage = root / case_id / f"{ordinal:02d}-{method}"
            try:
                arm_result = _run_arm(
                    plan,
                    arms[case_id][method],
                    repo_root=repo,
                    stage=stage,
                    fixture=fixture,
                    invoke=invoke,
                    clock=clock,
                    cpu_clock=cpu_clock,
                    gpu_sampler=gpu_sampler,
                )
            except ComparisonContractError as error:
                results.append({"case_id": case_id, "method": method,
                                "status": "integrity_failure", "reason": str(error)})
                batch_status = "stopped_integrity_failure"
                break
            results.append({"case_id": case_id, **arm_result})
        if batch_status != "completed":
            break
    report = {
        "schema": "diagnostic_comparison_result_v1",
        "plan": plan_ref,
        "registration_anchor": registration_anchor,
        "execution_class": "fixture" if fixture else "registered",
        "status": batch_status,
        "arms": results,
        "timing": {
            "end_to_end_start_s": batch_start,
            "end_to_end_end_s": clock(),
            "process_cpu_start_s": cpu_start,
            "process_cpu_end_s": cpu_clock(),
        },
        "automatic_retry": False,
    }
    latest = root / "comparison.latest.json"
    temporary = root / f".comparison.{os.getpid()}.tmp"
    temporary.write_bytes(canonical(report) + b"\n")
    os.replace(temporary, latest)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--anchor")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--no-device-sampling", action="store_true",
                        help="장치 계측을 생략하고 사용률·전력·메모리를 미측정으로 남긴다")
    args = parser.parse_args(argv)
    try:
        repo = args.repo.resolve(strict=True)
        relative_plan = args.plan.resolve(strict=True).relative_to(repo).as_posix()
        plan_ref = {"path": relative_plan, "sha256": _lf_sha(args.plan.resolve(strict=True))}
        binding = json.loads(args.binding.read_text(encoding="utf-8"))
        sampler = None
        if not args.fixture and not args.no_device_sampling:
            from rl_triage.gpu_resource_sampler import PeriodicDeviceSampler

            sampler = PeriodicDeviceSampler(args.output.resolve() / "device-samples")
        result = run_comparison(
            plan_ref,
            binding,
            repo_root=args.repo,
            output_root=args.output,
            registration_anchor=args.anchor,
            fixture=args.fixture,
            gpu_sampler=sampler,
        )
    except Exception as error:  # noqa: BLE001 - CLI는 실패를 구조화하고 backend를 재실행하지 않는다.
        result = {"status": "rejected", "error_type": type(error).__name__, "reason": str(error)}
    print(canonical(result).decode("utf-8"))
    return 0 if result.get("status") != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
