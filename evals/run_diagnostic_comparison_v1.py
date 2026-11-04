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
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from rl_triage import investigate
from rl_triage.investigation_session import file_sha, reference, write_new

SCHEMA = "diagnostic_comparison_plan_v1"
JOB_SCHEMA = "diagnostic_comparison_job_v1"
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
ACTUAL_BACKEND = investigate.REGISTERED_BACKEND
FIXTURE_BACKEND = investigate.FIXTURE_BACKEND
TERMINAL = {
    "confirmed",
    "none_supported",
    "hold_unknown",
    "hold_budget_exhausted",
    "host_no_intervention",
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


def _raw_refs(session: Path, max_checks: int) -> dict:
    refs: dict[str, list[dict]] = {"host_receipts": [], "api_journals": []}
    receipt_dir = session / "host_receipts"
    if receipt_dir.is_dir() and not receipt_dir.is_symlink():
        for path in sorted(receipt_dir.glob("*.json")):
            if path.is_file() and not path.is_symlink():
                refs["host_receipts"].append(reference(path))
    for ordinal in range(max_checks):
        for name in ("terminal.json", "attempts.jsonl"):
            path = session / "model" / f"{ordinal:04d}" / name
            if path.is_file() and not path.is_symlink():
                refs["api_journals"].append(reference(path))
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
        result["raw_refs"] = _raw_refs(session, plan["limits"]["max_checks"]) if session.exists() else {}
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
            result["raw_refs"] = _raw_refs(session, plan["limits"]["max_checks"])
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
    args = parser.parse_args(argv)
    try:
        repo = args.repo.resolve(strict=True)
        relative_plan = args.plan.resolve(strict=True).relative_to(repo).as_posix()
        plan_ref = {"path": relative_plan, "sha256": _lf_sha(args.plan.resolve(strict=True))}
        binding = json.loads(args.binding.read_text(encoding="utf-8"))
        result = run_comparison(
            plan_ref,
            binding,
            repo_root=args.repo,
            output_root=args.output,
            registration_anchor=args.anchor,
            fixture=args.fixture,
        )
    except Exception as error:  # noqa: BLE001 - CLI는 실패를 구조화하고 backend를 재실행하지 않는다.
        result = {"status": "rejected", "error_type": type(error).__name__, "reason": str(error)}
    print(canonical(result).decode("utf-8"))
    return 0 if result.get("status") != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
