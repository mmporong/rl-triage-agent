"""V05 E2E의 Brev 명령 실행과 actual 원문 검증 경계."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import selection_comparison_controller_v1 as CONTROLLER
import vision_brev_backend_v1 as RECORD
import vision_evidence_contract_v1 as VISION
import vision_execution_evidence_v1 as EVIDENCE  # pyright: ignore[reportMissingImports]
import vision_selection_execution_v1 as EXECUTION

from rl_triage import grounded_recovery as GROUNDED
from rl_triage import recovery as RECOVERY

UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 compatibility
PROTOCOL_SCHEMA = "vision_selection_e2e_execution_protocol_v1"
LEASE_SCHEMA = "vision_selection_e2e_lease_v1"
PAIR_SCHEMA = "vision_selection_probe_pair_request_v1"
PAIR_RESULT_SCHEMA = "vision_selection_probe_pair_result_v1"
PAIR_EVENTS = ("proposed", "approved", "consumed")
PAIR_KEYS = {
    "schema", "group", "probe", "seed", "arms", "checkpoints", "selection",
    "protocols", "execution", "commands", "commands_sha256", "budget_gpu_s", "resource_sha256",
}
GROUP_KEYS = {
    "group", "seed", "source_protocol_sha256_lf", "checkpoints", "corrected_job_id",
    "probe_budget_gpu_s", "action_limits", "execution",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{target.name}은 regular file이어야 한다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{target.name}은 단일 regular file이어야 한다")
    value = json.loads(target.read_text(encoding="utf-8"),
                       parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    if not isinstance(value, dict):
        raise TypeError(f"{target.name}은 JSON object여야 한다")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ref(root: Path, value: object, label: str, *, lf: bool = False) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    raw, expected = value.get("path"), value.get("sha256")
    if not isinstance(raw, str) or Path(raw).is_absolute() or "\\" in raw or ".." in Path(raw).parts:
        raise ValueError(f"{label} path가 portable 형식이 아니다")
    path = (root / raw).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    if not isinstance(expected, str) or file_sha(path, lf=lf) != expected:
        raise ValueError(f"{label} SHA가 다르다")
    return path, _read(path)


def _portable_reference(root: Path, path: Path) -> dict[str, str]:
    repository = Path(root).resolve(strict=True)
    target = Path(path).resolve(strict=True)
    target.relative_to(repository)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{target.name}은 portable regular file이어야 한다")
    return {"path": target.relative_to(repository).as_posix(), "sha256": file_sha(target)}


def _file_ref(root: Path, value: object, label: str, *, lf: bool = False) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    raw, expected = value.get("path"), value.get("sha256")
    if not isinstance(raw, str) or Path(raw).is_absolute() or "\\" in raw or ".." in Path(raw).parts:
        raise ValueError(f"{label} path가 portable 형식이 아니다")
    path = (root / raw).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    if path.is_symlink() or not path.is_file() or file_sha(path, lf=lf) != expected:
        raise ValueError(f"{label} SHA가 다르다")
    return path


def verify_bound_method(root: Path, reference: Mapping[str, Any], method: object,
                        expected_name: str) -> None:
    """등록 source 파일에 정의된 bound method만 외부 action service로 허용한다."""
    path = _file_ref(
        Path(root), {"path": reference.get("path"), "sha256": reference.get("sha256_lf")},
        expected_name, lf=True,
    )
    function = getattr(method, "__func__", None)
    owner = getattr(method, "__self__", None)
    source = inspect.getsourcefile(function) if function is not None else None
    if (owner is None or function is None or function.__name__ != expected_name
            or source is None or Path(source).resolve() != path):
        raise ValueError(f"{expected_name} service가 등록 source의 bound method가 아니다")


def _time(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{label} 시각이 문자열이 아니다")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10
    if parsed.tzinfo is None:
        raise ValueError(f"{label} 시각에 timezone이 없다")
    return parsed.astimezone(UTC_TZ)


def verify_context(root: Path, protocol_reference: object, lease_reference: object,
                   *, now: datetime | None = None, require_active: bool = True) -> dict[str, Any]:
    """새 E2E protocol과 그 protocol에 발급된 유효 lease만 승인한다."""
    if not isinstance(protocol_reference, Mapping) or not isinstance(lease_reference, Mapping):
        raise TypeError("execution protocol/lease reference가 object가 아니다")
    repository = Path(root).resolve(strict=True)
    protocol_path, protocol = _ref(repository, protocol_reference, "execution protocol")
    lease_path, lease = _ref(repository, lease_reference, "execution lease")
    protocol_keys = {"schema", "protocol_id", "registered_before_execution", "deadline_utc",
                     "source_anchor", "controller_sources_lf", "source_plan_sha256",
                     "selection_plan_sha256", "probe_execution", "plan_references",
                     "correction", "execution_bindings", "limits", "resources", "groups"}
    lease_keys = {"schema", "status", "execution_id", "protocol_sha256", "resource", "budget",
                  "issued_at_utc", "expires_at_utc"}
    limits = protocol.get("limits")
    groups = protocol.get("groups")
    if (set(protocol) != protocol_keys or protocol.get("schema") != PROTOCOL_SCHEMA
            or protocol.get("registered_before_execution") is not True
            or not isinstance(protocol.get("protocol_id"), str)
            or not isinstance(limits, Mapping)
            or limits != {"probe_pairs": 2, "gpu_probe_runs": 4, "corrected_runs": 2,
                          "fixed_evaluations": 2, "api_requests": 32}
            or not isinstance(groups, list) or len(groups) != 2
            or {row.get("group") for row in groups if isinstance(row, Mapping)}
            != {"source-01", "source-02"}
            or any(set(row) != GROUP_KEYS for row in groups if isinstance(row, Mapping))
            or any(not isinstance(row.get("action_limits"), Mapping)
                   or set(row["action_limits"]) != {"probe_pair", "correction", "record"}
                   for row in groups if isinstance(row, Mapping))
            or protocol.get("correction") != {"id": "restore_body_frame_reward", "mode": "restart"}):
        raise ValueError("E2E execution protocol 등록값이 다르다")
    anchor = protocol.get("source_anchor")
    controllers = protocol.get("controller_sources_lf")
    required_controllers = {
        "evals/vision_selection_brev_adapter_v1.py",
        "evals/vision_selection_e2e_caller_v1.py",
        "evals/run_vision_selection_e2e_v1.py",
        "evals/vision_selection_e2e_v1.py",
        "evals/vision_selection_call_v1.py",
    }
    if (not isinstance(anchor, str) or len(anchor) != 40
            or any(ch not in "0123456789abcdef" for ch in anchor)
            or not isinstance(controllers, Mapping) or set(controllers) != required_controllers):
        raise ValueError("E2E controller source 등록값이 다르다")
    for relative, expected in controllers.items():
        _file_ref(repository, {"path": relative, "sha256": expected},
                  f"controller {relative}", lf=True)
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{anchor}:{relative}"],
            check=False, capture_output=True,
        )
        if completed.returncode != 0 or hashlib.sha256(
                completed.stdout.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"controller {relative}가 source anchor blob과 다르다")
    bindings = protocol.get("execution_bindings")
    plan_references = protocol.get("plan_references")
    if not isinstance(plan_references, Mapping) or set(plan_references) != {"source", "selection"}:
        raise ValueError("E2E plan reference 등록값이 다르다")
    source_plan_path = _file_ref(repository, plan_references["source"], "source E2E plan")
    selection_plan_path = _file_ref(repository, plan_references["selection"], "selection plan")
    if (file_sha(source_plan_path) != protocol.get("source_plan_sha256")
            or file_sha(selection_plan_path) != protocol.get("selection_plan_sha256")):
        raise ValueError("E2E plan reference와 등록 SHA가 다르다")
    if (not isinstance(bindings, Mapping)
            or set(bindings) != {"source_protocol", "correction_backend", "correction_service", "record_backend", "record_verifier",
                                 "evidence_contract", "fixed_eval_protocol", "corrected_render_plan"}):
        raise ValueError("E2E backend/verifier 등록값이 다르다")
    for name in ("source_protocol", "correction_backend", "correction_service", "record_backend", "record_verifier", "evidence_contract", "fixed_eval_protocol"):
        value = bindings[name]
        expected_keys = ({"path", "sha256_lf", "class"}
                         if name in {"correction_backend", "record_verifier"}
                         else {"path", "sha256_lf"})
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError(f"{name} 등록 형식이 다르다")
        if "class" in value and (not isinstance(value["class"], str) or not value["class"]):
            raise ValueError(f"{name} class 등록값이 없다")
        _file_ref(repository, {"path": value["path"], "sha256": value["sha256_lf"]}, name,
                  lf=True)
    _ref(repository, bindings["corrected_render_plan"], "corrected render plan")
    resources = protocol.get("resources")
    if (not isinstance(resources, Mapping)
            or set(resources) != {"compute_quote_usd_per_h", "whole_batch_budget_usd"}
            or any(isinstance(resources[key], bool) or not isinstance(resources[key], (int, float))
                   or float(resources[key]) <= 0 for key in resources)):
        raise ValueError("E2E 비용 계약이 다르다")
    for row in groups:
        for name, limit in row["action_limits"].items():
            if (not isinstance(limit, Mapping) or set(limit) != {"max_wall_s", "max_compute_usd"}
                    or isinstance(limit["max_wall_s"], bool)
                    or not isinstance(limit["max_wall_s"], (int, float))
                    or isinstance(limit["max_compute_usd"], bool)
                    or not isinstance(limit["max_compute_usd"], (int, float))
                    or float(limit["max_wall_s"]) <= 0 or float(limit["max_compute_usd"]) <= 0
                    or float(limit["max_compute_usd"]) + 1e-9 <
                    float(limit["max_wall_s"]) * float(resources["compute_quote_usd_per_h"]) / 3600):
                raise ValueError(f"{name} 비용 상한이 wall/quote를 덮지 않는다")
    protocol_sha = file_sha(protocol_path)
    if (set(lease) != lease_keys or lease.get("schema") != LEASE_SCHEMA
            or lease.get("status") != "granted" or lease.get("execution_id") != protocol["protocol_id"]
            or lease.get("protocol_sha256") != protocol_sha):
        raise ValueError("E2E lease가 execution protocol과 다르다")
    _, resource = _ref(repository, lease.get("resource"), "E2E resource")
    _, budget = _ref(repository, lease.get("budget"), "E2E budget")
    if (resource.get("schema") != "brev_resource_proof_v1" or resource.get("status") != "granted"
            or resource.get("protocol_sha256") != protocol_sha
            or resource.get("lease_id") is None):
        raise ValueError("E2E resource가 새 protocol에 발급되지 않았다")
    budget_fields = {"schema", "status", "lease_id", "issued_at_utc", "expires_at_utc",
                     "compute_started_at_utc", "protocol_sha256", "whole_batch_budget_usd",
                     "compute_quote_usd_per_h", "cost_scope"}
    if (set(budget) != budget_fields or budget.get("schema") != "brev_budget_admission_v1"
            or budget.get("status") != "granted" or budget.get("lease_id") != resource.get("lease_id")
            or budget.get("protocol_sha256") != protocol_sha
            or resource.get("compute_started_at_utc") != budget.get("compute_started_at_utc")
            or protocol["resources"] != {
                "compute_quote_usd_per_h": budget.get("compute_quote_usd_per_h"),
                "whole_batch_budget_usd": budget.get("whole_batch_budget_usd"),
            }):
        raise ValueError("E2E budget/resource/protocol 결합이 다르다")
    current = (now or datetime.now(UTC_TZ)).astimezone(UTC_TZ)
    issued, expires, deadline = (_time(lease["issued_at_utc"], "lease issued"),
                                 _time(lease["expires_at_utc"], "lease expires"),
                                 _time(protocol["deadline_utc"], "protocol deadline"))
    if not issued < expires <= deadline:
        raise ValueError("E2E lease 발급·만료·deadline 순서가 다르다")
    if require_active and not issued <= current < expires:
        raise ValueError("E2E lease가 아직 유효하지 않거나 만료됐다")
    if (resource.get("issued_at_utc") != lease["issued_at_utc"]
            or resource.get("expires_at_utc") != lease["expires_at_utc"]
            or budget.get("issued_at_utc") != lease["issued_at_utc"]
            or budget.get("expires_at_utc") != lease["expires_at_utc"]):
        raise ValueError("resource와 lease 시간 결합이 다르다")
    if require_active:
        lifecycle_s = current.timestamp() - _time(
            budget["compute_started_at_utc"], "compute started",
        ).timestamp()
        lifecycle_cost = lifecycle_s * float(budget["compute_quote_usd_per_h"]) / 3600.0
        if lifecycle_s < 0 or lifecycle_cost > float(budget["whole_batch_budget_usd"]) + 1e-9:
            raise ValueError("E2E resource lifecycle 비용이 등록 예산을 넘었다")
    return {"root": repository, "protocol_reference": dict(protocol_reference),
            "lease_reference": dict(lease_reference),
            "protocol_path": protocol_path, "protocol": protocol,
            "protocol_sha256": file_sha(protocol_path), "lease_path": lease_path,
            "lease_sha256": file_sha(lease_path), "lease": lease, "resource": resource,
            "budget": budget}


def refresh_context(context: Mapping[str, Any], *, require_active: bool = True) -> dict[str, Any]:
    """원문 동일성을 다시 검증하고 새 action일 때만 현재 admission을 요구한다."""
    fresh = verify_context(Path(context["root"]), context.get("protocol_reference"),
                           context.get("lease_reference"), require_active=require_active)
    fields = ("protocol_sha256", "lease_sha256", "protocol", "lease", "resource", "budget")
    if any(fresh[field] != context.get(field) for field in fields):
        raise ValueError("execution context 원문이 검증 뒤 바뀌었다")
    return fresh


def verify_pair_prefix(root: Path, reference: object) -> dict[str, Any]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256", "request_id",
                                                                  "prefix_sha256"}:
        raise ValueError("probe pair 승인 reference 형식이 다르다")
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve(strict=True)
    path.relative_to(Path(root).resolve(strict=True))
    if file_sha(path) != reference["sha256"]:
        raise ValueError("probe pair 원장 SHA가 다르다")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    request_rows = [row for row in rows if row.get("request_id") == reference["request_id"]]
    if ([row.get("event") for row in request_rows] != list(PAIR_EVENTS)
            or digest(request_rows) != reference["prefix_sha256"]):
        raise ValueError("probe pair 승인은 proposed→approved→consumed prefix여야 한다")
    request = request_rows[0].get("request")
    if (not isinstance(request, dict) or set(request) != PAIR_KEYS
            or request.get("schema") != PAIR_SCHEMA
            or request_rows[1].get("request_sha256") != digest(request)
            or request_rows[2].get("request_sha256") != digest(request)):
        raise ValueError("probe pair 승인 request가 바뀌었다")
    return request


def _registered_pair(context: Mapping[str, Any], request: Mapping[str, Any], *,
                     require_active: bool = True) -> None:
    refresh_context(context, require_active=require_active)
    protocol = context.get("protocol")
    lease = context.get("lease")
    if not isinstance(protocol, Mapping) or not isinstance(lease, Mapping):
        raise TypeError("검증된 execution context가 아니다")
    rows = [row for row in protocol.get("groups", [])
            if isinstance(row, Mapping) and row.get("group") == request.get("group")]
    if len(rows) != 1:
        raise ValueError("probe pair group이 execution protocol에 유일하지 않다")
    row = rows[0]
    expected = {
        "seed": row["seed"],
        "checkpoints": row["checkpoints"],
        "source_protocol": row["source_protocol_sha256_lf"],
        "probe_protocol": protocol["probe_execution"],
        "execution": row["execution"],
        "budget": row["probe_budget_gpu_s"],
        "resource": lease["resource"]["sha256"],
    }
    actual = {
        "seed": request.get("seed"),
        "checkpoints": request.get("checkpoints"),
        "source_protocol": request.get("protocols", {}).get("source"),
        "probe_protocol": request.get("protocols", {}).get("probe"),
        "execution": request.get("execution"),
        "budget": request.get("budget_gpu_s"),
        "resource": request.get("resource_sha256"),
    }
    if actual != expected:
        raise ValueError("probe pair가 등록 seed/checkpoint/protocol/resource/execution과 다르다")
    if (request.get("arms") != ["NONE", "F1"]
            or not isinstance(request.get("selection"), Mapping)
            or set(request["selection"]) != {"terminal_sha256", "activation_bridge_sha256"}
            or any(not isinstance(request["selection"][key], str)
                   or len(request["selection"][key]) != 64 for key in request["selection"])):
        raise ValueError("probe pair selection/arm 결합이 다르다")


def _run(argv: Sequence[str], *, cwd: Path,
         runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> None:
    if (not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv
            or any(not isinstance(part, str) or not part for part in argv)):
        raise ValueError("실행 argv가 비어 있거나 문자열 배열이 아니다")
    completed = runner(list(argv), cwd=str(cwd), check=False, capture_output=True, text=True,
                       encoding="utf-8", errors="strict")
    if completed.returncode != 0:
        raise RuntimeError(f"Brev command exit={completed.returncode}")


def validate_pair_result(context: Mapping[str, Any], request: Mapping[str, Any],
                         result: Mapping[str, Any], expected_source: Mapping[str, Any],
                         authorization_sha256: str) -> dict[str, Any]:
    """완료 cache도 actual 6원문에서 측정·분류·비용을 다시 만든다."""
    refresh_context(context, require_active=False)
    _registered_pair(context, request, require_active=False)
    root = Path(context["root"])
    actuals: dict[str, dict[str, Any]] = {}
    for arm, side in (("NONE", "reference"), ("F1", "case")):
        arm_result = result.get(arm)
        if not isinstance(arm_result, Mapping):
            raise TypeError("cached probe arm 결과가 없다")
        descriptor_ref = arm_result.get("descriptor")
        if not isinstance(descriptor_ref, Mapping):
            raise TypeError("cached probe descriptor reference가 없다")
        descriptor_path, descriptor = _ref(root, descriptor_ref, f"cached {arm} descriptor")
        if descriptor_path != (root / str(descriptor_ref["path"])).resolve():
            raise ValueError("cached descriptor 경로가 바뀌었다")
        actuals[arm] = EXECUTION._brev_actual_probe(
            root, descriptor, side=side, probe=request["probe"],
            checkpoint_sha256=request["checkpoints"][arm], expected_source=dict(expected_source),
        )
    outcomes = {
        "NONE": CONTROLLER.classify_raw(
            request["probe"], actuals["NONE"]["measurement"], actuals["NONE"]["measurement"],
        ),
        "F1": CONTROLLER.classify_raw(
            request["probe"], actuals["F1"]["measurement"], actuals["NONE"]["measurement"],
        ),
    }
    gpu_s = sum(float(actuals[arm]["gpu_s"]) for arm in ("NONE", "F1"))
    if gpu_s > request["budget_gpu_s"]:
        raise ValueError("cached probe GPU 시간이 등록 예산을 넘었다")
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == request["group"])
    _action_cost(context, entry, "probe_pair", gpu_s)
    for arm in ("NONE", "F1"):
        raw = result[arm]
        receipt = result.get(arm.lower())
        descriptor = _read(root / raw["descriptor"]["path"])
        execution = _ref(root, descriptor["sources"]["execution"], f"cached {arm} execution")[1]
        expected_receipt = {
            "status": "complete", "outcome": outcomes[arm], "seed": request["seed"], "arm": arm,
            "checkpoint_sha256": request["checkpoints"][arm],
            "job_sha256": descriptor["sources"]["job"]["sha256"],
            "worker_sha256": descriptor["sources"]["worker"]["sha256"],
            "execution_sha256": descriptor["sources"]["execution"]["sha256"],
            "report_jobs_sha256": descriptor["sources"]["report_jobs"]["sha256"],
            "report_sha256": descriptor["sources"]["report"]["sha256"],
            "gpu_before_sha256": digest(execution["gpu_before"]),
            "gpu_after_sha256": digest(execution["gpu_after"]),
            "authorization_sha256": authorization_sha256,
            "seal_sha256": raw["descriptor"]["sha256"],
        }
        if (raw.get("outcome") != outcomes[arm]
                or raw.get("measurement") != actuals[arm]["measurement"]
                or receipt != expected_receipt):
            raise ValueError("cached probe 결과가 actual 원문 재계산과 다르다")
    required = {"schema", "status", "request_sha256", "probe", "seed", "NONE", "F1",
                "none", "f1", "gpu_runs", "gpu_s"}
    if (set(result) != required or result.get("schema") != PAIR_RESULT_SCHEMA
            or result.get("status") != "complete" or result.get("request_sha256") != digest(request)
            or result.get("probe") != request["probe"] or result.get("seed") != request["seed"]
            or result.get("gpu_runs") != 2 or result.get("gpu_s") != gpu_s):
        raise ValueError("cached probe pair 집계가 다르다")
    return dict(result)


def validate_partial_pair_result(context: Mapping[str, Any], request: Mapping[str, Any],
                                 value: object, expected_source: Mapping[str, Any],
                                 authorization_sha256: str) -> dict[str, Any]:
    refresh_context(context, require_active=False)
    if (not isinstance(value, Mapping)
            or set(value) != {"schema", "status", "reason", "request_sha256",
                              "authorization_sha256", "completed_arm", "descriptor",
                              "actual", "gpu_s"}
            or value.get("schema") != PAIR_RESULT_SCHEMA or value.get("status") != "HOLD"
            or value.get("reason") != "next_arm_window_insufficient"
            or value.get("request_sha256") != digest(request)
            or value.get("authorization_sha256") != authorization_sha256
            or value.get("completed_arm") != "NONE"):
        raise ValueError("partial probe pair identity/schema가 다르다")
    root = Path(context["root"])
    descriptor_path = _file_ref(root, value.get("descriptor"), "partial NONE descriptor")
    actual = EXECUTION._brev_actual_probe(
        root, _read(descriptor_path), side="reference", probe=str(request["probe"]),
        checkpoint_sha256=str(request["checkpoints"]["NONE"]),
        expected_source=dict(expected_source),
    )
    entry = next(row for row in context["protocol"]["groups"]
                 if row["group"] == request["group"])
    plan = request["execution"]["NONE"]
    gpu_s = float(actual["gpu_s"])
    if (actual != value.get("actual") or actual["seed"] != request["seed"]
            or actual["stage"] != plan["stage"] or actual["run_name"] != plan["run_name"]
            or value.get("gpu_s") != gpu_s or gpu_s > request["budget_gpu_s"]):
        raise ValueError("partial NONE actual/비용이 승인 request와 다르다")
    _action_cost(context, entry, "probe_pair", gpu_s)
    return dict(value)


def run_probe_pair(context: Mapping[str, Any], pair_reference: object,
                   commands: Mapping[str, Mapping[str, Any]], expected_source: Mapping[str, Any],
                   stage: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
                   ) -> dict[str, Any]:
    """승인된 NONE/F1 두 명령을 한 번만 실행하고 actual 6원문을 재검산한다."""
    root = Path(context["root"])
    request = verify_pair_prefix(root, pair_reference)
    if not isinstance(pair_reference, Mapping):
        raise TypeError("probe pair reference가 object가 아니다")
    output = Path(stage)
    result_path = output / "result.json"
    partial_path = output / "partial.json"
    if result_path.is_file():
        cached = _read(result_path)
        return validate_pair_result(
            context, request, cached, expected_source, str(pair_reference["sha256"]),
        )
    if partial_path.is_file():
        return validate_partial_pair_result(
            context, request, _read(partial_path), expected_source,
            str(pair_reference["sha256"]),
        )
    if output.exists() and any(output.iterdir()):
        return {"schema": PAIR_RESULT_SCHEMA, "status": "HOLD", "reason": "partial_stage",
                "request_sha256": digest(request)}
    _registered_pair(context, request)
    if set(commands) != {"NONE", "F1"} or request.get("arms") != ["NONE", "F1"]:
        raise ValueError("probe pair는 NONE/F1 exact pair여야 한다")
    if request.get("commands") != dict(commands) or request.get("commands_sha256") != digest(commands):
        raise ValueError("probe command가 승인 request와 다르다")
    output.mkdir(parents=True, exist_ok=True)
    verified: dict[str, dict[str, Any]] = {}
    descriptors: dict[str, dict[str, str]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == request["group"])
    try:
        for arm, side in (("NONE", "reference"), ("F1", "case")):
            try:
                require_action_window(context, entry, "probe_pair")
            except ValueError:
                if verified:
                    hold = {
                        "schema": PAIR_RESULT_SCHEMA, "status": "HOLD",
                        "reason": "next_arm_window_insufficient",
                        "request_sha256": digest(request),
                        "authorization_sha256": str(pair_reference["sha256"]),
                        "completed_arm": "NONE", "descriptor": descriptors["NONE"],
                        "actual": verified["NONE"],
                        "gpu_s": float(verified["NONE"]["gpu_s"]),
                    }
                    _action_cost(context, entry, "probe_pair", hold["gpu_s"])
                    _write_new(partial_path, hold)
                    return hold
                raise
            spec = commands[arm]
            if set(spec) != {"argv", "descriptor"}:
                raise ValueError("probe command 형식이 다르다")
            descriptor_path = Path(str(spec["descriptor"])).resolve()
            descriptor_path.relative_to(output.resolve())
            _run(spec["argv"], cwd=root, runner=runner)
            descriptor = _read(descriptor_path)
            checkpoint = request["checkpoints"][arm]
            actual = EXECUTION._brev_actual_probe(
                root, descriptor, side=side, probe=request["probe"],
                checkpoint_sha256=checkpoint, expected_source=dict(expected_source),
            )
            plan = request["execution"][arm]
            if (actual["seed"] != request["seed"] or actual["stage"] != plan["stage"]
                    or actual["run_name"] != plan["run_name"]):
                raise ValueError("probe actual이 승인 stage/run/seed와 다르다")
            verified[arm] = actual
            descriptors[arm] = {"path": descriptor_path.relative_to(root).as_posix(),
                                "sha256": file_sha(descriptor_path)}
        outcome = CONTROLLER.classify_raw(
            request["probe"], verified["F1"]["measurement"], verified["NONE"]["measurement"],
        )
        none_outcome = CONTROLLER.classify_raw(
            request["probe"], verified["NONE"]["measurement"], verified["NONE"]["measurement"],
        )
        outcomes = {"NONE": none_outcome, "F1": outcome}
        total_gpu_s = sum(float(verified[arm]["gpu_s"]) for arm in ("NONE", "F1"))
        if total_gpu_s > request["budget_gpu_s"]:
            raise ValueError("probe pair 실제 GPU 시간이 등록 예산을 넘었다")
        _action_cost(context, entry, "probe_pair", total_gpu_s)
        for arm in ("NONE", "F1"):
            descriptor = _read(root / descriptors[arm]["path"])
            sources = descriptor["sources"]
            execution_path, execution = _ref(root, sources["execution"], f"{arm} execution")
            del execution_path
            receipts[arm] = {
                "status": "complete", "outcome": outcomes[arm], "seed": request["seed"],
                "arm": arm, "checkpoint_sha256": request["checkpoints"][arm],
                "job_sha256": sources["job"]["sha256"],
                "worker_sha256": sources["worker"]["sha256"],
                "execution_sha256": sources["execution"]["sha256"],
                "report_jobs_sha256": sources["report_jobs"]["sha256"],
                "report_sha256": sources["report"]["sha256"],
                "gpu_before_sha256": digest(execution["gpu_before"]),
                "gpu_after_sha256": digest(execution["gpu_after"]),
                "authorization_sha256": str(pair_reference["sha256"]),
                "seal_sha256": descriptors[arm]["sha256"],
            }
        body = {"schema": PAIR_RESULT_SCHEMA, "status": "complete",
                "request_sha256": digest(request), "probe": request["probe"],
                "seed": request["seed"], "NONE": {"outcome": none_outcome,
                "measurement": verified["NONE"]["measurement"], "descriptor": descriptors["NONE"]},
                "F1": {"outcome": outcome, "measurement": verified["F1"]["measurement"],
                       "descriptor": descriptors["F1"]},
                "none": receipts["NONE"], "f1": receipts["F1"],
                "gpu_runs": 2, "gpu_s": total_gpu_s}
        with result_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(body, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        return body
    except BaseException:  # noqa: TRY203 - 실패한 partial 원문을 보존한다.
        # partial은 재실행하지 않는다. 원문은 진단을 위해 보존한다.
        raise


def _expected_correction_source(context: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(context["root"])
    binding = context["protocol"]["execution_bindings"]["source_protocol"]
    protocol_path = _file_ref(
        root, {"path": binding["path"], "sha256": binding["sha256_lf"]},
        "registered correction source protocol", lf=True,
    )
    protocol = _read(protocol_path)
    environment = protocol.get("environment")
    if not isinstance(environment, Mapping):
        raise TypeError("registered correction environment가 없다")
    expected_environment = dict(environment)
    expected_environment.update(cuda_available=True, cuda_device_count=1)
    packages = {key: value for key, value in environment.items()
                if key not in {"python", "isaaclab_sha"}}
    return {
        "lab_head": environment.get("isaaclab_sha"), "lab_tracked_clean": True,
        "source_anchor": context["protocol"]["source_anchor"],
        "protocol": {"path": binding["path"], "sha256_lf": binding["sha256_lf"]},
        "environment": expected_environment,
        "selected_python": environment.get("python"), "selected_packages": packages,
        "source_sha256_lf": protocol.get("source_sha256_lf"),
        "runtime_source_sha256_lf": protocol.get("runtime_source_sha256_lf"),
        "python_package_source_sha256_lf": protocol.get("python_package_source_sha256_lf"),
    }


def _validate_registered_source(context: Mapping[str, Any], execution: Mapping[str, Any],
                                label: str) -> None:
    expected = _expected_correction_source(context)
    source = execution.get("source_before")
    selected = source.get("selected_interpreter") if isinstance(source, Mapping) else None
    if (not isinstance(source, Mapping) or not isinstance(selected, Mapping)
            or source.get("lab_head") != expected["lab_head"]
            or source.get("lab_tracked_clean") is not True
            or source.get("source_anchor") != expected["source_anchor"]
            or source.get("protocol") != expected["protocol"]
            or source.get("environment") != expected["environment"]
            or selected.get("python") != expected["selected_python"]
            or selected.get("packages") != expected["selected_packages"]
            or source.get("source_sha256_lf") != expected["source_sha256_lf"]
            or source.get("runtime_source_sha256_lf") != expected["runtime_source_sha256_lf"]
            or source.get("python_package_source_sha256_lf")
            != expected["python_package_source_sha256_lf"]):
        raise ValueError(f"{label} source identity가 등록 protocol과 다르다")


def _execution_gpu_s(value: object, label: str, context: Mapping[str, Any], *,
                     bind_registered_source: bool = False) -> float:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} execution이 object가 아니다")
    required = {"completed", "exit_code", "timed_out", "cleanup_verified", "gpu_before",
                "gpu_after", "source_before", "source_after", "wall_s", "finished_at"}
    wall = value.get("wall_s")
    if (not required <= set(value) or value.get("completed") is not True
            or value.get("exit_code") != 0 or value.get("timed_out") is not False
            or value.get("cleanup_verified") is not True or value.get("gpu_before") != []
            or value.get("gpu_after") != [] or value.get("source_before") != value.get("source_after")
            or isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0):
        raise ValueError(f"{label} 완료·GPU·cleanup receipt가 다르다")
    finished = _time(value["finished_at"], f"{label} finished")
    started = finished.timestamp() - float(wall)
    issued = _time(context["lease"]["issued_at_utc"], "lease issued").timestamp()
    boundary = min(_time(context["lease"]["expires_at_utc"], "lease expires"),
                   _time(context["protocol"]["deadline_utc"], "protocol deadline"))
    if started < issued or finished > boundary:
        raise ValueError(f"{label} 실행시간이 lease/deadline 밖이다")
    if bind_registered_source:
        _validate_registered_source(context, value, label)
    return float(wall)


def _action_cost(context: Mapping[str, Any], entry: Mapping[str, Any], action: str,
                 gpu_s: float) -> float:
    limit = entry["action_limits"][action]
    cost = gpu_s * float(context["protocol"]["resources"]["compute_quote_usd_per_h"]) / 3600.0
    if gpu_s > float(limit["max_wall_s"]) or cost > float(limit["max_compute_usd"]) + 1e-9:
        raise ValueError(f"{action} 실제 시간·비용이 등록 상한을 넘었다")
    return cost


def require_action_window(context: Mapping[str, Any], entry: Mapping[str, Any], action: str) -> None:
    """GPU action 전체 등록 wall이 현재 lease/deadline 안에 들어갈 때만 허용한다."""
    fresh = refresh_context(context)
    now = datetime.now(UTC_TZ)
    boundary = min(_time(fresh["lease"]["expires_at_utc"], "lease expires"),
                   _time(fresh["protocol"]["deadline_utc"], "protocol deadline"))
    max_wall = float(entry["action_limits"][action]["max_wall_s"])
    if now.timestamp() + max_wall > boundary.timestamp():
        raise ValueError(f"{action} 실행 예약 창이 lease/deadline보다 짧다")


class CorrectionService:
    """등록 backend의 실제 train/evaluate 인터페이스를 E2E 한 건으로 고정한다."""

    def __init__(self, backend: Any, runtime: Path, none_job_id: str, run_name: str):
        self.backend = backend
        self.runtime = Path(runtime)
        self.none_job_id = none_job_id
        self.run_name = run_name
        self._none_artifact: dict[str, Any] | None = None

    def preflight(self) -> bool:
        """등록 NONE 완료 cache만 읽으며 새 학습은 시작하지 않는다."""
        completed = getattr(self.backend, "_completed", None)
        if not callable(completed):
            raise TypeError("production backend에 cache-only 완료 조회가 없다")
        value = completed(self.none_job_id)
        if value is None:
            return False
        if not isinstance(value, dict) or value.get("arm") != "NONE":
            raise ValueError("등록 NONE 완료 artifact가 다르다")
        self._none_artifact = value
        return True

    def train(self, job_id: str, *, approval_reference: Mapping[str, Any]) -> dict[str, Any]:
        return self.backend.train(job_id, approval_reference=approval_reference)

    def training_evidence(self, job_id: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
        events = self.backend._events()  # production backend performs ledger schema/order validation
        matches = [row for row in events if row.get("job_id") == job_id]
        if not matches or matches[-1].get("event") != "completed":
            raise ValueError("CORRECTED 완료 training ledger event가 없다")
        event = matches[-1]
        artifact_path = Path(str(event.get("artifact_path", "")))
        inventory = event.get("inventory")
        if not isinstance(inventory, Mapping):
            raise TypeError("CORRECTED training inventory가 없다")
        required = inventory.get("required_files")
        if not isinstance(required, Mapping) or set(required) != {"readback", "records", "execution"}:
            raise ValueError("CORRECTED training required files가 다르다")
        root = Path(__file__).resolve().parents[1]
        files = {
            "artifact": _portable_reference(root, artifact_path),
            "checkpoint": _portable_reference(root, Path(str(artifact["checkpoint"]["path"]))),
            "telemetry": _portable_reference(root, Path(str(artifact["telemetry"]["path"]))),
            "manifest": _portable_reference(root, Path(str(inventory["manifest"]["path"]))),
            **{name: _portable_reference(root, Path(str(required[name]["path"])))
               for name in ("readback", "records", "execution")},
        }
        return {"schema": "vision_selection_training_evidence_v1",
                "ledger": _portable_reference(root, self.backend.ledger_path),
                "completed_event_sha256": digest(event), "files": files}

    def evaluate(self, artifact: dict[str, Any]) -> dict[str, Any]:
        none = self._none_artifact
        if none is None:
            raise ValueError("NONE cache preflight가 완료되지 않았다")
        raw = self.backend.evaluate(
            self.runtime / "fixed-eval" / self.run_name,
            {"NONE": none, "CORRECTED": artifact}, self.run_name,
        )
        reports = raw.get("reports") if isinstance(raw, Mapping) else None
        if not isinstance(reports, Mapping) or set(reports) != {"NONE", "CORRECTED"}:
            raise ValueError("production fixed evaluation arm이 다르다")
        stage = self.runtime / "fixed-eval" / self.run_name
        root = Path(__file__).resolve().parents[1]
        evidence = {name: _portable_reference(root, stage / filename) for name, filename in (
            ("job", "job.json"), ("worker", "worker.json"), ("execution", "execution.json"),
            ("none_report", "reports/NONE.json"),
            ("corrected_report", "reports/CORRECTED.json"),
        )}
        return {"reports": {"none": reports["NONE"], "corrected": reports["CORRECTED"]},
                "execution": raw.get("execution"), "evidence": evidence}


def verify_correction_service(context: Mapping[str, Any], service: object) -> None:
    bindings = context["protocol"]["execution_bindings"]
    typed = service  # pyright: ignore[reportUnknownVariableType]
    verify_bound_method(Path(context["root"]), bindings["correction_service"],
                        getattr(typed, "preflight", None), "preflight")
    verify_bound_method(Path(context["root"]), bindings["correction_service"],
                        getattr(typed, "train", None), "train")
    verify_bound_method(Path(context["root"]), bindings["correction_service"],
                        getattr(typed, "training_evidence", None), "training_evidence")
    verify_bound_method(Path(context["root"]), bindings["correction_service"],
                        getattr(typed, "evaluate", None), "evaluate")
    backend = getattr(service, "backend", None)
    source = inspect.getsourcefile(type(backend)) if backend is not None else None
    registered = _file_ref(
        Path(context["root"]),
        {"path": bindings["correction_backend"]["path"],
         "sha256": bindings["correction_backend"]["sha256_lf"]},
        "correction backend class", lf=True,
    )
    if source is None or Path(source).resolve() != registered:
        raise ValueError("correction service backend가 등록 production class가 아니다")
    if type(backend).__name__ != bindings["correction_backend"]["class"]:
        raise ValueError("correction service backend class 이름이 등록값과 다르다")


def _fixed_labels(context: Mapping[str, Any], reports: object,
                  none_sha: str, corrected_sha: str) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(reports, Mapping) or set(reports) != {"none", "corrected"}:
        raise ValueError("fixed evaluation reports arm이 다르다")
    protocol_sha = context["protocol"]["execution_bindings"]["fixed_eval_protocol"]["sha256_lf"]
    checked: dict[str, Any] = {}
    for arm, checkpoint_sha in (("none", none_sha), ("corrected", corrected_sha)):
        report = reports[arm]
        if not isinstance(report, dict):
            raise TypeError("fixed report가 object가 아니다")
        checked[arm] = GROUNDED.validate_fixed_report(report, {
            "name": arm, "checkpoint_file": report.get("checkpoint", {}).get("file"),
            "checkpoint_sha256": checkpoint_sha, "action_scale": 0.25,
            "protocol_sha256_lf": protocol_sha,
        })
    none_behavior = GROUNDED._behavior(checked["none"])
    corrected_behavior = GROUNDED._behavior(checked["corrected"])
    bands = RECOVERY.derive_bands([RECOVERY.compare(none_behavior, none_behavior)])
    labels = {
        "none": RECOVERY.verdict(RECOVERY.compare(none_behavior, none_behavior), bands)["label"],
        "corrected": RECOVERY.verdict(RECOVERY.compare(corrected_behavior, none_behavior), bands)["label"],
    }
    return labels, checked


def run_correction(context: Mapping[str, Any], group: str,
                   approval_reference: Mapping[str, Any], job_id: str,
                   train: Callable[..., dict[str, Any]], evaluate: Callable[[dict[str, Any]], dict[str, Any]]
                   ) -> dict[str, Any]:
    refresh_context(context)
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == group)
    owner = getattr(train, "__self__", None)
    if owner is None or getattr(evaluate, "__self__", None) is not owner:
        raise ValueError("train/evaluate가 같은 correction service가 아니다")
    verify_correction_service(context, owner)
    require_action_window(context, entry, "correction")
    artifact = train(job_id, approval_reference=approval_reference)
    checkpoint = artifact.get("checkpoint") if isinstance(artifact, Mapping) else None
    if (not isinstance(artifact, dict) or artifact.get("arm") != "CORRECTED"
            or artifact.get("seed") != entry["seed"] or not isinstance(checkpoint, Mapping)
            or artifact.get("source_checkpoint_sha256") != entry["checkpoints"]["F1"]):
        raise ValueError("CORRECTED training artifact가 등록 job과 다르다")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if (not checkpoint_path.is_file() or checkpoint_path.is_symlink()
            or file_sha(checkpoint_path) != checkpoint.get("sha256")):
        raise ValueError("CORRECTED checkpoint 원문/SHA가 다르다")
    train_gpu_s = _execution_gpu_s(artifact.get("execution"), "training", context,
                                      bind_registered_source=True)
    try:
        require_action_window(context, entry, "correction")
    except ValueError:
        return {"schema": "vision_selection_correction_hold_v1", "status": "HOLD",
                "reason": "fixed_evaluation_window_insufficient", "group": group,
                "job_id": job_id, "training": artifact, "gpu_s": train_gpu_s}
    training_evidence = owner.training_evidence(job_id, artifact)
    fixed = evaluate(artifact)
    if not isinstance(fixed, dict) or set(fixed) != {"reports", "execution", "evidence"}:
        raise ValueError("fixed evaluation 결과 형식이 다르다")
    eval_gpu_s = _execution_gpu_s(fixed["execution"], "fixed evaluation", context,
                                     bind_registered_source=True)
    labels, reports = _fixed_labels(
        context, fixed["reports"], entry["checkpoints"]["NONE"], str(checkpoint["sha256"]),
    )
    gpu_s = train_gpu_s + eval_gpu_s
    compute_usd = _action_cost(context, entry, "correction", gpu_s)
    return {"schema": "vision_selection_correction_result_v1", "status": "complete",
            "group": group, "job_id": job_id, "checkpoint_sha256": checkpoint["sha256"],
            "training": artifact, "training_evidence": training_evidence,
            "evaluation": {"reports": reports, "execution": fixed["execution"],
                           "evidence": fixed["evidence"]},
            "fixed_eval": labels, "gpu_s": gpu_s, "compute_usd": compute_usd, "exit_code": 0}


def _validate_training_evidence(root: Path, job_id: str, training: Mapping[str, Any],
                                evidence: object) -> None:
    if (not isinstance(evidence, Mapping)
            or set(evidence) != {"schema", "ledger", "completed_event_sha256", "files"}
            or evidence.get("schema") != "vision_selection_training_evidence_v1"):
        raise ValueError("cached training evidence schema가 다르다")
    ledger_path = _file_ref(root, evidence["ledger"], "training ledger")
    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    matches = [row for row in events if isinstance(row, Mapping) and row.get("job_id") == job_id]
    if ([row.get("event") for row in matches] != ["reserved", "started", "completed"]
            or len({row.get("plan_sha256") for row in matches}) != 1
            or digest(matches[-1]) != evidence.get("completed_event_sha256")):
        raise ValueError("cached training 완료 event가 원장과 다르다")
    event = matches[-1]
    inventory = event.get("inventory")
    files = evidence.get("files")
    if (not isinstance(inventory, Mapping) or not isinstance(files, Mapping)
            or set(files) != {"artifact", "checkpoint", "telemetry", "manifest",
                             "readback", "records", "execution"}):
        raise ValueError("cached training inventory/evidence files가 다르다")
    opened = {name: _file_ref(root, reference, f"training {name}")
              for name, reference in files.items()}
    raw_artifact = _read(opened["artifact"] )
    if (raw_artifact != dict(training)
            or event.get("artifact_sha256") != files["artifact"].get("sha256")):
        raise ValueError("cached training artifact 원문이 결과와 다르다")
    required = inventory.get("required_files")
    if not isinstance(required, Mapping):
        raise TypeError("cached training required files inventory가 없다")
    expected_sha = {
        "checkpoint": inventory.get("checkpoint", {}).get("sha256"),
        "telemetry": inventory.get("telemetry", {}).get("sha256"),
        "manifest": inventory.get("manifest", {}).get("sha256"),
        **{name: required.get(name, {}).get("sha256")
           for name in ("readback", "records", "execution")},
    }
    if set(expected_sha) != set(opened) - {"artifact"} or any(
            files[name].get("sha256") != expected_sha[name] for name in expected_sha):
        raise ValueError("cached training inventory SHA가 portable evidence와 다르다")
    if (_read(opened["manifest"]) != training.get("manifest")
            or _read(opened["readback"]) != training.get("readback")
            or _read(opened["execution"]) != training.get("execution")):
        raise ValueError("cached training manifest/readback/execution 원문이 다르다")


def _validate_evaluation_evidence(root: Path, evaluation: Mapping[str, Any],
                                  none_sha: str, corrected_sha: str) -> None:
    evidence = evaluation.get("evidence")
    if (not isinstance(evidence, Mapping)
            or set(evidence) != {"job", "worker", "execution", "none_report",
                                 "corrected_report"}):
        raise ValueError("cached fixed evaluation evidence schema가 다르다")
    raw = {name: _ref(root, reference, f"fixed evaluation {name}")[1]
           for name, reference in evidence.items()}
    artifacts = raw["job"].get("artifacts")
    if (raw["execution"] != evaluation.get("execution")
            or raw["job"].get("mode") != "evaluate" or raw["worker"].get("completed") is not True
            or not isinstance(artifacts, Mapping) or set(artifacts) != {"NONE", "CORRECTED"}
            or artifacts["NONE"].get("checkpoint", {}).get("sha256") != none_sha
            or artifacts["CORRECTED"].get("checkpoint", {}).get("sha256") != corrected_sha):
        raise ValueError("cached fixed evaluation 실행 원문이 다르다")
    descriptors = raw["worker"].get("reports")
    reports = evaluation.get("reports")
    if not isinstance(descriptors, Mapping) or not isinstance(reports, Mapping):
        raise TypeError("cached fixed evaluation report descriptor가 없다")
    for arm, key, backend_arm in (("none", "none_report", "NONE"),
                                  ("corrected", "corrected_report", "CORRECTED")):
        descriptor = descriptors.get(backend_arm)
        if (not isinstance(descriptor, Mapping)
                or descriptor.get("sha256") != evidence[key].get("sha256")
                or raw[key] != reports.get(arm)):
            raise ValueError(f"cached fixed evaluation {arm} report 원문이 다르다")


def validate_correction_result(context: Mapping[str, Any], group: str,
                               result: Mapping[str, Any]) -> dict[str, Any]:
    refresh_context(context, require_active=False)
    required = {"schema", "status", "group", "job_id", "checkpoint_sha256", "training",
                 "training_evidence", "evaluation", "fixed_eval", "gpu_s",
                 "compute_usd", "exit_code"}
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == group)
    training, evaluation = result.get("training"), result.get("evaluation")
    checkpoint = training.get("checkpoint") if isinstance(training, Mapping) else None
    if (set(result) != required or result.get("schema") != "vision_selection_correction_result_v1"
            or result.get("status") != "complete" or result.get("group") != group
            or result.get("job_id") != entry["corrected_job_id"]
            or result.get("exit_code") != 0 or not isinstance(training, Mapping)
            or training.get("arm") != "CORRECTED" or training.get("seed") != entry["seed"]
            or training.get("source_checkpoint_sha256") != entry["checkpoints"]["F1"]
            or not isinstance(checkpoint, Mapping)
            or result.get("checkpoint_sha256") != checkpoint.get("sha256")
            or not isinstance(evaluation, Mapping)
            or set(evaluation) != {"reports", "execution", "evidence"}):
        raise ValueError("cached correction identity/schema가 다르다")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if (not checkpoint_path.is_file() or checkpoint_path.is_symlink()
            or file_sha(checkpoint_path) != checkpoint.get("sha256")):
        raise ValueError("cached CORRECTED checkpoint 원문/SHA가 다르다")
    _validate_training_evidence(Path(context["root"]), str(result["job_id"]), training,
                                result.get("training_evidence"))
    _validate_evaluation_evidence(
        Path(context["root"]), evaluation, entry["checkpoints"]["NONE"],
        str(checkpoint["sha256"]),
    )
    gpu_s = (_execution_gpu_s(training.get("execution"), "cached training", context,
                              bind_registered_source=True)
             + _execution_gpu_s(evaluation.get("execution"), "cached fixed evaluation", context,
                                bind_registered_source=True))
    labels, reports = _fixed_labels(
        context, evaluation.get("reports"), entry["checkpoints"]["NONE"], str(checkpoint["sha256"]),
    )
    compute_usd = _action_cost(context, entry, "correction", gpu_s)
    if (evaluation.get("reports") != reports or result.get("fixed_eval") != labels
            or result.get("gpu_s") != gpu_s or result.get("compute_usd") != compute_usd):
        raise ValueError("cached correction 평가/비용 재계산이 다르다")
    return dict(result)


def verify_record_service(context: Mapping[str, Any], record: object) -> None:
    bindings = context["protocol"]["execution_bindings"]
    verify_bound_method(Path(context["root"]), bindings["record_backend"], record, "record")
    owner = getattr(record, "__self__", None)
    class_source = inspect.getsourcefile(type(owner)) if owner is not None else None
    registered = _file_ref(
        Path(context["root"]),
        {"path": bindings["record_verifier"]["path"],
         "sha256": bindings["record_verifier"]["sha256_lf"]},
        "record verifier class", lf=True,
    )
    if (class_source is None or Path(class_source).resolve() != registered
            or type(owner).__name__ != bindings["record_verifier"]["class"]):
        raise ValueError("record service class가 등록 verifier source와 다르다")


def validate_record_result(context: Mapping[str, Any], value: object, *,
                           record_job: Mapping[str, Any] | None = None) -> dict[str, Any]:
    refresh_context(context, require_active=False)
    if (not isinstance(value, Mapping)
            or set(value) not in ({"execution", "result"},
                                  {"execution", "result", "gpu_s", "compute_usd"})):
        raise ValueError("record 결과 reference 형식이 다르다")
    root = Path(context["root"])
    execution_path, execution = _ref(root, value["execution"], "record execution")
    result_path, result = _ref(root, value["result"], "record result")
    gpu_s = _execution_gpu_s(execution, "record", context)
    if (execution.get("result_sha256") != file_sha(result_path)
            or result.get("completed") is not True):
        raise ValueError("record execution/result 결합이 다르다")
    compute_usd = 0.0
    if record_job is not None:
        checkpoint = record_job.get("checkpoint")
        if (not isinstance(checkpoint, Mapping)
                or result.get("checkpoint_sha256") != checkpoint.get("sha256")):
            raise ValueError("record result checkpoint가 등록 job과 다르다")
        groups = [row for row in context["protocol"]["groups"]
                  if row.get("corrected_job_id") == record_job.get("source_job_id")]
        if len(groups) != 1:
            raise ValueError("record job의 corrected lineage가 등록되지 않았다")
        job_path = execution_path.parent / "job.json"
        if _read(job_path) != dict(record_job):
            raise ValueError("record execution job 원문이 등록 job과 다르다")
        verifier_path = _file_ref(
            Path(context["root"]),
            {"path": context["protocol"]["execution_bindings"]["record_verifier"]["path"],
             "sha256": context["protocol"]["execution_bindings"]["record_verifier"]["sha256_lf"]},
            "record verifier", lf=True,
        )
        if verifier_path.name in {"vision_brev_render_backend_v4.py", "vision_brev_render_backend_v5.py"}:
            auxiliary = execution.get("auxiliary")
            if (execution.get("schema") != "vision_brev_record_execution_v2"
                    or not isinstance(auxiliary, Mapping)
                    or set(auxiliary) != {"file", "sha256"}):
                raise ValueError("record atomic attestation receipt가 없다")
            manifest_path = execution_path.parent / str(auxiliary["file"])
            if file_sha(manifest_path) != auxiliary.get("sha256"):
                raise ValueError("record atomic attestation SHA가 다르다")
            EVIDENCE.verify_manifest(manifest_path, job_path, result_path)
            RECORD._verify_record_result(
                result_path, dict(record_job), str(execution.get("job_sha256")),
            )
        compute_usd = _action_cost(context, groups[0], "record", gpu_s)
    if (("gpu_s" in value or "compute_usd" in value)
            and (value.get("gpu_s") != gpu_s or value.get("compute_usd") != compute_usd)):
        raise ValueError("record cache 시간·비용 재계산이 다르다")
    return {"execution": dict(value["execution"]), "result": dict(value["result"]),
            "gpu_s": gpu_s, "compute_usd": compute_usd}


def materialize_record_contract(context: Mapping[str, Any], value: object, *,
                                record_job: Mapping[str, Any], output_dir: Path,
                                window: int = 1) -> Path:
    """검증된 record 원문에서 V01 관찰 계약을 결정적으로 만든다."""
    if window not in (1, 2):
        raise ValueError("record window는 1 또는 2여야 한다")
    checked = validate_record_result(context, value, record_job=record_job)
    root = Path(context["root"]).resolve(strict=True)
    bindings = context["protocol"]["execution_bindings"]
    registered = _file_ref(
        root,
        {"path": bindings["evidence_contract"]["path"],
         "sha256": bindings["evidence_contract"]["sha256_lf"]},
        "evidence contract materializer", lf=True,
    )
    source = inspect.getsourcefile(VISION.prepare)
    if source is None or Path(source).resolve() != registered:
        raise ValueError("V01 materializer가 등록 source와 다르다")

    result_path, result = _ref(root, checked["result"], "record result")
    artifacts = result.get("artifacts")
    logical_id = f"spec_{window}"
    descriptor = artifacts.get(logical_id) if isinstance(artifacts, Mapping) else None
    if (not isinstance(descriptor, Mapping)
            or set(descriptor) != {"file", "sha256", "bytes"}
            or not isinstance(descriptor.get("file"), str)
            or Path(str(descriptor["file"])).name != descriptor["file"]):
        raise ValueError("record V01 spec descriptor가 다르다")
    matches = [path for path in result_path.parent.rglob(str(descriptor["file"]))
               if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise ValueError("record V01 spec 원문이 유일하지 않다")
    spec_path = matches[0].resolve(strict=True)
    spec_path.relative_to(result_path.parent.resolve(strict=True))
    if (file_sha(spec_path) != descriptor.get("sha256")
            or spec_path.stat().st_size != descriptor.get("bytes")):
        raise ValueError("record V01 spec 원문/SHA/크기가 다르다")

    output = Path(output_dir).resolve()
    runtime_root = (root / ".triage-runtime" / "vision-selection-e2e").resolve()
    output.relative_to(runtime_root)
    for parent in (output, *output.parents):
        if parent == runtime_root.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise ValueError("V01 output 경로에 symlink가 있다")
    contract_path = output / "contract.json"
    if output.exists():
        if not contract_path.is_file() or contract_path.is_symlink():
            raise ValueError("기존 V01 output이 완료 계약이 아니다")
        contract = VISION.verify(contract_path)
    else:
        contract_path = VISION.prepare(spec_path, output)
        contract = VISION.verify(contract_path)
    provenance = contract.get("provenance")
    checkpoint = record_job.get("checkpoint")
    if (not isinstance(provenance, Mapping) or not isinstance(checkpoint, Mapping)
            or provenance.get("checkpoint", {}).get("sha256") != checkpoint.get("sha256")
            or provenance.get("video", {}).get("sha256") != result.get("video_sha256")
            or provenance.get("play_run_id") != f"{record_job.get('job_id')}-w{window}"):
        raise ValueError("V01 계약이 record job/checkpoint/video와 다르다")
    return contract_path


def run_post_evidence(context: Mapping[str, Any], correction: Mapping[str, Any],
                      record_job: Mapping[str, Any], record: Callable[[Mapping[str, Any]], dict[str, Any]],
                      observe: Callable[[dict[str, Any]], dict[str, Any]],
                      ) -> dict[str, Any]:
    refresh_context(context)
    verify_record_service(context, record)
    fixed = correction.get("fixed_eval")
    if (not isinstance(fixed, Mapping) or fixed.get("none") != "healthy"
            or fixed.get("corrected") != "healthy"):
        return {"status": "HOLD", "reason": "fixed_eval_not_recovered"}
    groups = [row for row in context["protocol"]["groups"]
              if row.get("corrected_job_id") == record_job.get("source_job_id")]
    if len(groups) != 1:
        raise ValueError("record job의 corrected lineage가 등록되지 않았다")
    require_action_window(context, groups[0], "record")
    checked = validate_record_result(context, record(record_job), record_job=record_job)
    observation = observe(checked)
    if observation.get("used_for_decision") is not False:
        raise ValueError("후속 VL 관찰은 회복 판정에 사용할 수 없다")
    return {"status": "complete", "record": checked, "observation": observation,
            "used_for_decision": False}
