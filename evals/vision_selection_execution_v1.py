"""동결 영상 선택 비교의 비실행 점검, 단일 명령 생성, 검증 집계."""
from __future__ import annotations

import argparse
import json
import math
import stat
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "src", ROOT / "evals", ROOT / "bench"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import model_decision_journal_v1 as JOURNAL
import residual_demo_v1 as RECOVERY
import run_vision_brev_batch_v1 as SOURCE  # pyright: ignore[reportMissingImports]
import selection_comparison_controller_v1 as CONTROLLER
import vision_remote_gpu_v1 as REMOTE
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMPARISON
import vision_selection_sources_v1 as INPUTS
import vision_selection_worker_v1 as WORKER

from rl_triage import grounded_recovery as GROUNDED

EVIDENCE_SCHEMA = "vision_selection_evidence_locations_v2"
UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 compatibility


def _read(path: Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError(f"{target.name}은 symlink가 아닌 regular file이어야 한다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{target.name}은 단일 regular file이어야 한다")

    class _Pairs(list):
        pass

    encoded = json.loads(
        target.read_text(encoding="utf-8"), object_pairs_hook=_Pairs,
        parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(f"JSON constant 거부: {raw}")),
    )

    def restore(value: Any) -> Any:
        if isinstance(value, _Pairs):
            result: dict[str, Any] = {}
            for key, item in value:
                if not isinstance(key, str) or key in result:
                    raise ValueError("JSON object의 키가 중복됐거나 문자열이 아니다")
                result[key] = restore(item)
            return result
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    restored = restore(encoded)
    if not isinstance(restored, dict):
        raise TypeError(f"{target.name}은 JSON object여야 한다")
    return restored


def _ref(root: Path, value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    raw, digest = value["path"], value["sha256"]
    if not isinstance(raw, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} reference 값이 다르다")
    if Path(raw).is_absolute() or "\\" in raw or ".." in Path(raw).parts:
        raise ValueError(f"{label} path는 portable repo-relative 형식이어야 한다")
    path = (root / raw).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    if JOURNAL.file_sha(path) != digest:
        raise ValueError(f"{label} SHA가 다르다")
    return path, digest


def load_context(protocol: Path, protocol_sha_lf: str, source_anchor: str,
                 input_locations: Path) -> dict[str, Any]:
    """기존 worker 준비 계약을 그대로 통과한 읽기 전용 실행 문맥을 만든다."""
    prepared = WORKER.prepare(protocol, protocol_sha_lf, source_anchor, input_locations)
    repository, stage, manifest = CALL._plan(prepared["plan"])
    return {
        "repository": repository, "stage": stage, "manifest": manifest,
        "prepared": prepared, "protocol_path": Path(protocol).resolve(strict=True),
        "protocol_sha256_lf": protocol_sha_lf, "source_anchor": source_anchor,
        "input_locations": Path(input_locations).resolve(strict=True),
    }


def _owner_present(stage: Path, decision: Path | None = None) -> bool:
    paths = [stage / "worker.identity.json", stage / ".active", stage / "api" / ".active"]
    if decision is not None:
        paths.append(decision / ".active")
    return any(path.exists() or path.is_symlink() for path in paths)


def _existing_journal(context: dict[str, Any], row: dict, frozen: dict) -> Any | None:
    stage = Path(context["stage"])
    decision = stage / "decisions" / row["call_id"]
    decision_manifest, api_manifest = decision / "manifest.json", stage / "api" / "manifest.json"
    if not decision_manifest.exists() and not api_manifest.exists():
        return None
    if not decision_manifest.is_file() or not api_manifest.is_file():
        raise JOURNAL.JournalIntegrityError("decision/API manifest가 부분 생성됐다")
    journal = CALL._journal(stage, context["manifest"], row, frozen)
    journal._verify_prepared()
    return journal


def inspect_run(context: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any]:
    """terminal은 provider 없이 replay하고 첫 미완료 판단의 안전 상태만 읽는다."""
    now = (now_utc or datetime.now(UTC_TZ)).astimezone(UTC_TZ)
    stage, manifest, repository = Path(context["stage"]), context["manifest"], Path(context["repository"])
    if _owner_present(stage):
        return _inspection("blocked_pending", manifest, None, None, 0, 0, 0, 0)
    terminal = semantic = infra = 0
    first: tuple[dict, dict] | None = None
    last_audit = {"reserved": 0, "pending": 0, "remaining": manifest["api_cap"], "rows": []}
    for row in manifest["rows"]:
        _, _, frozen = CALL._row(repository, manifest, row["call_id"])
        terminal_path = stage / "decisions" / row["call_id"] / "terminal.json"
        if terminal_path.is_file():
            replay = CALL._replay(stage, manifest, row, frozen)
            terminal += 1
            result = replay["engine_result"]
            if result.get("hold_reason") == "selection_invalid":
                semantic += 1
            if str(result.get("hold_reason", "")).startswith("transport_") or result.get("stop_reason") == "budget_exhausted":
                infra += 1
            journal = _existing_journal(context, row, frozen)
            if journal is not None:
                last_audit = journal._audit_api()
            continue
        first = (row, frozen)
        break
    if first is None:
        return _inspection("complete", manifest, None, None, terminal, semantic, infra, last_audit["reserved"], last_audit)
    row, frozen = first
    decision = stage / "decisions" / row["call_id"]
    if _owner_present(stage, decision):
        return _inspection("blocked_pending", manifest, row["call_id"], None, terminal, semantic, infra,
                           last_audit["reserved"], last_audit)
    journal = _existing_journal(context, row, frozen)
    if journal is not None:
        audit = journal._audit_api()
        last_audit = audit
        if audit["pending"]:
            return _inspection("blocked_pending", manifest, row["call_id"], None, terminal, semantic, infra,
                               audit["reserved"], audit)
        own = audit["own"]
        if own and own[-1]["response"] is not None:
            retry = own[-1]["response"].get("retry_not_before")
            if retry is not None and now < JOURNAL._parse_stamp(retry):
                return _inspection("paused_infra", manifest, row["call_id"], retry, terminal, semantic, infra,
                                   audit["reserved"], audit)
    if last_audit["remaining"] <= 0:
        return _inspection("cap_exhausted", manifest, row["call_id"], None, terminal, semantic, infra,
                           last_audit["reserved"], last_audit)
    return _inspection("runnable", manifest, row["call_id"], None, terminal, semantic, infra,
                       last_audit["reserved"], last_audit)


def _inspection(status: str, manifest: dict, call_id: str | None, not_before: str | None,
                terminal: int, semantic: int, infra: int, reserved: int,
                audit: dict | None = None) -> dict[str, Any]:
    public = audit or {"pending": 0, "remaining": manifest["api_cap"] - reserved}
    return {
        "status": status, "next_call_id": call_id, "not_before": not_before,
        "api": {"cap": manifest["api_cap"], "reserved": reserved,
                "pending": public.get("pending", 0), "remaining": public.get("remaining", manifest["api_cap"] - reserved)},
        "counts": {"terminal": terminal, "semantic_invalid": semantic, "infra_terminal": infra,
                   "unstarted": len(manifest["rows"]) - terminal},
    }


def remaining_commands(context: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any]:
    state = inspect_run(context, now_utc)
    commands: list[list[str]] = []
    if state["status"] == "runnable":
        commands.append([
            "setsid", str(Path(sys.executable).resolve()), str((ROOT / "evals/vision_selection_worker_v1.py").resolve()),
            "advance", "--protocol", str(context["protocol_path"]),
            "--protocol-sha256-lf", context["protocol_sha256_lf"],
            "--source-anchor", context["source_anchor"], "--input-locations", str(context["input_locations"]),
            "--call-id", state["next_call_id"],
        ])
    return {**state, "commands": commands}


def project_results(context: dict[str, Any]) -> list[dict[str, Any]]:
    state = inspect_run(context)
    stage, manifest = Path(context["stage"]), context["manifest"]
    inputs = {row["call_id"]: row for row in context["prepared"]["protocol"]["rows"]}
    rows: list[dict[str, Any]] = []
    for plan_row in manifest["rows"]:
        terminal_path = stage / "decisions" / plan_row["call_id"] / "terminal.json"
        if not terminal_path.is_file():
            continue
        terminal = CALL._read_canonical(terminal_path)
        result = terminal["engine_result"]
        if result.get("model_status") == "valid":
            model_status = "valid"
        elif result.get("hold_reason") == "selection_invalid":
            model_status = "semantic_invalid"
        elif str(result.get("hold_reason", "")).startswith("transport_") or result.get("stop_reason") == "budget_exhausted":
            model_status = "infrastructure_failed"
        else:
            raise ValueError("terminal 상태를 공개 상태로 투영할 수 없다")
        source = inputs[plan_row["call_id"]]
        selection = result.get("selection", {})
        rows.append({
            "slot_id": source["slot_id"], "view_kind": source["view_kind"], "call_id": plan_row["call_id"],
            "capsule_sha256": plan_row["capsule_sha256"], "messages_sha256": source["messages_sha256"],
            "terminal_sha256": JOURNAL.file_sha(terminal_path), "model_status": model_status,
            "action": selection.get("action"), "probe": selection.get("probe"),
            "fact_ids": deepcopy(selection.get("fact_ids", [])),
        })
    if state["status"] == "blocked_pending" and state["api"]["pending"] == 1 and state["next_call_id"]:
        source = inputs[state["next_call_id"]]
        plan_row = next(row for row in manifest["rows"] if row["call_id"] == state["next_call_id"])
        rows.append({"slot_id": source["slot_id"], "view_kind": source["view_kind"],
                     "call_id": state["next_call_id"], "capsule_sha256": plan_row["capsule_sha256"],
                     "messages_sha256": source["messages_sha256"], "terminal_sha256": None,
                     "model_status": "unknown_pending", "action": None, "probe": None, "fact_ids": []})
    return rows


def _usage(context: dict[str, Any]) -> dict[str, Any]:
    api = Path(context["stage"]) / "api" / "responses"
    paths = sorted(api.glob("*.json")) if api.exists() else []
    known = missing = tokens = 0
    wall = 0.0
    for path in paths:
        result = _read(path).get("result", {})
        total = result.get("total_tokens") if isinstance(result, dict) else None
        if type(total) is int and total >= 0:
            known += 1
            tokens += total
        else:
            missing += 1
        seconds = result.get("wall_s") if isinstance(result, dict) else None
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0:
            wall += float(seconds)
    return {"requests": len(paths), "tokens": {"known": known, "missing": missing, "sum": tokens},
            "provider_wall_s_sum": wall}


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} 값이 유한수가 아니다")
    return float(value)


def _source_artifact(value: dict, row: dict, expected_arm: str, telemetry_sha256: str) -> None:
    checkpoint, telemetry = value.get("checkpoint"), value.get("telemetry")
    if (value.get("arm") != expected_arm or value.get("seed") != row["source_seed"]
            or not isinstance(checkpoint, dict) or checkpoint.get("sha256") != row["checkpoint_sha256"]
            or not isinstance(telemetry, dict) or telemetry.get("sha256") != telemetry_sha256):
        raise ValueError("source training artifact와 현재 comparison input 계보가 다르다")


def _input_artifacts(root: Path, context: dict[str, Any], group: str,
                     input_rows: list[dict], source_protocol: dict) -> dict[str, dict]:
    """동결 input-locations가 가리키는 readiness에서 학습 artifact를 다시 결합한다."""
    locations = _read(Path(context["input_locations"]))
    if (set(locations) != {"schema", "execution_id", "rows"}
            or not isinstance(locations["rows"], list)
            or len(locations["rows"]) != len(input_rows)):
        raise ValueError("input locations 형식·row 수가 다르다")
    by_call = {}
    for public, location in zip(input_rows, locations["rows"], strict=True):
        if (not isinstance(location, dict) or set(location) != {"call_id", "input_manifest", "capsule"}
                or location.get("call_id") != public["call_id"]):
            raise ValueError("input locations 순서·call 결합이 다르다")
        by_call[public["call_id"]] = location
    outputs: dict[str, dict] = {}
    for role, arm in (("normal", "NONE"), ("registered-f1", "F1")):
        rows = [row for row in input_rows if row["seed_group"] == group and row["source_role"] == role]
        if not rows or any(row["checkpoint_sha256"] != rows[0]["checkpoint_sha256"] for row in rows):
            raise ValueError("comparison input checkpoint가 role 안에서 다르다")
        identities = set()
        selected: tuple[dict, str] | None = None
        for row in rows:
            location = by_call[row["call_id"]]
            manifest_path, _ = _ref(root, location["input_manifest"], "input manifest")
            manifest = _read(manifest_path)
            if set(manifest) != {"schema", "snapshot", "contract", "trace", "camera_pose_audit", "model_readiness"}:
                raise ValueError("input manifest 형식이 다르다")
            readiness_path, readiness_sha = _ref(root, manifest["model_readiness"], "model readiness")
            readiness = _read(readiness_path)
            proof = INPUTS.verify_model_readiness(root, readiness_path, source_protocol)
            if (proof.get("checkpoint_sha256") != row["checkpoint_sha256"]
                    or proof.get("source_seed") != row["source_seed"]
                    or proof.get("source_role") != role):
                raise ValueError("model readiness 재검증과 comparison input 계보가 다르다")
            artifact_path, artifact_sha = _ref(
                root, readiness.get("source_training_artifact"), "source training artifact",
            )
            artifact = _read(artifact_path)
            snapshot_path, _ = _ref(root, manifest["snapshot"], "selection snapshot")
            snapshot = _read(snapshot_path)
            bindings = snapshot.get("inputs", {}).get("packet", {}).get("input_bindings")
            case = bindings.get("case") if isinstance(bindings, dict) else None
            telemetry_sha = artifact.get("telemetry", {}).get("sha256")
            if (not isinstance(case, dict)
                    or case != {"checkpoint_sha256": row["checkpoint_sha256"],
                                "telemetry_sha256": telemetry_sha}):
                raise ValueError("snapshot case와 readiness 학습 artifact 계보가 다르다")
            _source_artifact(artifact, row, arm, str(telemetry_sha))
            identity = (readiness_sha, artifact_sha, str(telemetry_sha))
            identities.add(identity)
            selected = (artifact, artifact_sha)
        if len(identities) != 1 or selected is None:
            raise ValueError("같은 role 입력이 다른 readiness 학습 artifact를 쓴다")
        outputs[arm] = selected[0]
    return outputs


def _evaluate_evidence(root: Path, context: dict[str, Any], group: str, evaluation: dict,
                       source_protocol: dict, input_rows: list[dict]) -> tuple[dict, dict]:
    fields = {"intent", "raw", "receipt", "resource",
              "gpu_before_observation", "gpu_after_observation"}
    if not isinstance(evaluation, dict) or set(evaluation) != fields:
        raise ValueError("evaluate evidence portable reference가 완전하지 않다")
    intent_path, intent_sha = _ref(root, evaluation["intent"], f"{group} evaluate intent")
    raw_path, raw_sha = _ref(root, evaluation["raw"], f"{group} evaluate raw")
    receipt_path, _ = _ref(root, evaluation["receipt"], f"{group} evaluate receipt")
    resource_path, resource_sha = _ref(root, evaluation["resource"], f"{group} resource")
    before_path, before_sha = _ref(root, evaluation["gpu_before_observation"], f"{group} GPU before")
    after_path, after_sha = _ref(root, evaluation["gpu_after_observation"], f"{group} GPU after")
    intent, raw, receipt, resource = map(_read, (intent_path, raw_path, receipt_path, resource_path))
    seeds = {row["source_seed"] for row in input_rows if row["seed_group"] == group}
    if len(seeds) != 1:
        raise ValueError("comparison input source seed가 group 안에서 유일하지 않다")
    seed = next(iter(seeds))
    actions = [(ordinal, action) for ordinal, action in enumerate(source_protocol["source_action_order"])
               if action.get("kind") == "evaluate" and action.get("seed") == seed]
    if len(actions) != 1:
        raise ValueError("source protocol evaluate action이 유일하지 않다")
    ordinal, action = actions[0]
    limit = source_protocol["resources"]["action_limits"]["evaluate"]
    expected_intent = {
        "schema": SOURCE.INTENT_SCHEMA, "action_id": action["action_id"], "kind": "evaluate",
        "ordinal": ordinal, "protocol_sha256_lf": source_protocol["_verified_sha256_lf"],
        "resource_sha256": resource_sha, "reserved_max_wall_s": limit["max_wall_s"],
        "reserved_max_compute_usd": limit["max_compute_usd"],
        "cost_scope": source_protocol["resources"]["cost_scope"],
    }
    if (set(intent) != {*expected_intent, "cost_before_usd", "started_at_utc"}
            or any(intent.get(key) != value for key, value in expected_intent.items())):
        raise ValueError("evaluate intent와 등록 action/resource 결합이 다르다")
    if (set(receipt) != {"schema", "action_id", "intent_sha256", "raw_sha256",
                         "cost_after_usd", "finished_at_utc"}
            or receipt.get("schema") != SOURCE.RECEIPT_SCHEMA
            or receipt.get("action_id") != action["action_id"]
            or receipt.get("intent_sha256") != intent_sha or receipt.get("raw_sha256") != raw_sha):
        raise ValueError("evaluate receipt SHA 결합이 다르다")
    started, finished = SOURCE._stamp(intent["started_at_utc"]), SOURCE._stamp(receipt["finished_at_utc"])
    compute_started = SOURCE._stamp(resource["compute_started_at_utc"])
    deadline = SOURCE._stamp(source_protocol["resources"]["deadline_utc"])
    quote = _finite(source_protocol["resources"]["compute_quote_usd_per_h"], "compute quote")
    before_cost = (started - compute_started).total_seconds() * quote / 3600
    after_cost = (finished - compute_started).total_seconds() * quote / 3600
    if (not 0 <= (finished - started).total_seconds() <= limit["max_wall_s"] or finished > deadline
            or abs(_finite(intent["cost_before_usd"], "cost before") - before_cost) > 1e-9
            or abs(_finite(receipt["cost_after_usd"], "cost after") - after_cost) > 1e-9
            or not 0 <= before_cost <= after_cost <= source_protocol["resources"]["whole_batch_budget_usd"]):
        raise ValueError("evaluate receipt 시각·비용이 등록 자원과 다르다")
    execution = raw.get("evaluation", {}).get("execution")
    before_ref = execution.get("gpu_before_observation") if isinstance(execution, dict) else None
    after_ref = execution.get("gpu_after_observation") if isinstance(execution, dict) else None
    expected_execution = {
        "schema": "brev_recovery_posix_execution_v1", "mode": "evaluate", "completed": True,
        "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "gpu_before": [], "gpu_after": [],
    }
    if (raw.get("kind") != "evaluate" or raw.get("action_id") != action["action_id"]
            or raw.get("seed") != seed or not isinstance(execution, dict)
            or any(execution.get(key) != value for key, value in expected_execution.items())
            or execution.get("source_before") != execution.get("source_after")
            or not isinstance(execution.get("launch"), dict)
            or _finite(execution.get("wall_s"), "evaluate wall") <= 0
            or not isinstance(before_ref, dict) or before_ref.get("sha256") != before_sha
            or not isinstance(after_ref, dict) or after_ref.get("sha256") != after_sha):
        raise ValueError("evaluate raw 실행 완료·GPU 해제 결합이 다르다")
    resource_id, gpu_uuid = resource.get("resource_id"), resource.get("gpu_uuid")
    if not isinstance(resource_id, str) or not isinstance(gpu_uuid, str):
        raise TypeError("evaluate resource identity가 없다")
    REMOTE.verify_observation_reference({"path": str(before_path), "sha256": before_sha}, resource_id, gpu_uuid)
    REMOTE.verify_observation_reference({"path": str(after_path), "sha256": after_sha}, resource_id, gpu_uuid)
    reports = raw.get("evaluation", {}).get("reports")
    if not isinstance(reports, dict) or set(reports) != {"NONE", "F1"}:
        raise ValueError("evaluate NONE/F1 원본 report가 없다")
    bands_path = (root / source_protocol["evaluation"]["bands_path"]).resolve(strict=True)
    bands_path.relative_to(root.resolve(strict=True))
    bands = _read(bands_path).get("bands")
    if not isinstance(bands, dict) or raw.get("source_gate") != RECOVERY.recovery(
            reports["F1"], reports["NONE"], bands):
        raise ValueError("evaluate source failure gate 재계산이 raw와 다르다")
    artifacts = _input_artifacts(root, context, group, input_rows, source_protocol)
    fixed_path = (root / source_protocol["fixed_eval_protocol"]).resolve(strict=True)
    fixed_path.relative_to(root.resolve(strict=True))
    fixed_sha = SOURCE.sha256(fixed_path, lf=True)
    for arm in ("NONE", "F1"):
        checkpoint = artifacts[arm]["checkpoint"]
        reports[arm] = GROUNDED.validate_fixed_report(reports[arm], {
            "name": arm, "checkpoint_file": Path(checkpoint["path"]).name,
            "checkpoint_sha256": checkpoint["sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": fixed_sha,
        })
    training = {row["job_id"]: row for row in source_protocol["training_plan"]}
    if (training[action["none_job_id"]]["seed"] != seed or training[action["none_job_id"]]["arm"] != "NONE"
            or training[action["f1_job_id"]]["seed"] != seed or training[action["f1_job_id"]]["arm"] != "F1"):
        raise ValueError("evaluate action의 NONE/F1 training 계보가 다르다")
    return reports, artifacts


def _brev_actual_probe(root: Path, descriptor: dict, *, side: str, probe: str,
                       checkpoint_sha256: str, expected_source: dict) -> dict:
    descriptor_keys = {"schema", "side", "probe", "checkpoint_sha256", "sources"}
    source_keys = {"job", "worker", "execution", "backend_result", "report_jobs", "report"}
    if (not isinstance(descriptor, dict) or set(descriptor) != descriptor_keys
            or descriptor.get("schema") != "vision_selection_brev_probe_evidence_v1"
            or descriptor.get("side") != side or descriptor.get("probe") != probe
            or descriptor.get("checkpoint_sha256") != checkpoint_sha256
            or not isinstance(descriptor.get("sources"), dict)
            or set(descriptor["sources"]) != source_keys):
        raise ValueError("Brev actual probe descriptor 결합이 다르다")
    paths = {name: _ref(root, value, f"Brev {side} {name}")[0]
             for name, value in descriptor["sources"].items()}
    values = {name: _read(path) for name, path in paths.items()}
    job, worker, execution = values["job"], values["worker"], values["execution"]
    backend, jobs, report = values["backend_result"], values["report_jobs"], values["report"]
    job_keys = {"schema", "mode", "stage", "lab", "repo", "protocol_path", "protocol_sha256_lf",
                "source_commit_sha", "source_sha256_lf", "runtime_source_sha256_lf",
                "python_package_source_sha256_lf", "arm", "seed", "run_name", "probe",
                "checkpoint_path", "checkpoint_sha256", "report_path", "training_manifest_path",
                "inspection_manifest_path", "records_path", "readback_path"}
    worker_keys = {"schema", "completed", "probe", "measurement", "report", "inspection_manifest", "wall_s"}
    execution_keys = {"schema", "mode", "completed", "started_at", "finished_at", "exit_code",
                      "timed_out", "cleanup_verified", "launch", "gpu_before", "gpu_after", "wall_s",
                      "source_before", "source_after"}
    if (set(job) != job_keys or set(worker) != worker_keys or set(execution) != execution_keys
            or set(backend) != {"execution", "measurement", "report"}
            or set(jobs) != {"fault", "seed", "task", "jobs"}
            or set(report) != {"name", "checkpoint", "task", "seed", "num_envs", "probes"}):
        raise ValueError("Brev actual probe 원본 필드가 다르다")
    expected_arm, expected_fault = ("F1", "F1") if side == "case" else ("NONE", "NONE")
    rows = jobs.get("jobs")
    report_probe = report.get("probes", {}).get(probe) if isinstance(report.get("probes"), dict) else None
    source = execution.get("source_before")
    expected_job = {
        "name": job.get("run_name"), "checkpoint": job.get("checkpoint_path"),
        "probes": [probe], "output": job.get("report_path"),
    }
    worker_report = worker.get("report")
    selected = source.get("selected_interpreter") if isinstance(source, dict) else None
    environment = source.get("environment") if isinstance(source, dict) else None
    packages = selected.get("packages") if isinstance(selected, dict) else None
    package_versions = {key: value for key, value in expected_source["environment"].items()
                        if key not in {"python", "isaaclab_sha", "cuda_available", "cuda_device_count"}}
    if (job.get("schema") != "brev_recovery_worker_job_v1" or job.get("mode") != "probe"
            or job.get("arm") != expected_arm or job.get("probe") != probe
            or job.get("checkpoint_sha256") != checkpoint_sha256
            or type(job.get("seed")) is not int or job["seed"] < 0
            or worker.get("schema") != "brev_recovery_probe_worker_v1"
            or worker.get("completed") is not True or worker.get("probe") != probe
            or execution.get("schema") != "brev_recovery_posix_execution_v1"
            or execution.get("mode") != "probe" or execution.get("completed") is not True
            or execution.get("exit_code") != 0 or execution.get("timed_out") is not False
            or execution.get("cleanup_verified") is not True
            or execution.get("gpu_before") != [] or execution.get("gpu_after") != []
            or execution.get("source_before") != execution.get("source_after")
            or _finite(execution.get("wall_s"), "Brev probe wall") <= 0
            or not isinstance(source, dict)
            or source.get("lab_head") != expected_source["environment"]["isaaclab_sha"]
            or source.get("lab_tracked_clean") is not True
            or source.get("source_anchor") != job.get("source_commit_sha")
            or source.get("protocol", {}).get("sha256_lf") != job.get("protocol_sha256_lf")
            or source.get("source_anchor") != expected_source["source_anchor"]
            or source.get("protocol") != expected_source["protocol"]
            or environment != expected_source["environment"]
            or not isinstance(selected, dict) or selected.get("python") != expected_source["environment"]["python"]
            or packages != package_versions
            or source.get("source_sha256_lf") != job.get("source_sha256_lf")
            or source.get("source_sha256_lf") != expected_source["source_sha256_lf"]
            or source.get("runtime_source_sha256_lf") != job.get("runtime_source_sha256_lf")
            or source.get("runtime_source_sha256_lf") != expected_source["runtime_source_sha256_lf"]
            or source.get("python_package_source_sha256_lf") != job.get("python_package_source_sha256_lf")
            or source.get("python_package_source_sha256_lf") != expected_source["python_package_source_sha256_lf"]
            or not str(job.get("protocol_path", "")).replace("\\", "/").endswith(
                "/" + expected_source["protocol"]["path"])
            or jobs.get("fault") != expected_fault or jobs.get("seed") != job.get("seed")
            or jobs.get("task") != expected_source["task"]
            or not isinstance(rows, list) or rows != [expected_job]
            or report.get("name") != job.get("run_name") or report.get("seed") != job.get("seed")
            or report.get("task") != jobs.get("task") or set(report.get("probes", {})) != {probe}
            or not isinstance(report_probe, dict) or report_probe.get("exit") != 0
            or not isinstance(report_probe.get("measurement"), dict)
            or report.get("checkpoint") != {"file": Path(str(job.get("checkpoint_path"))).name,
                                                "sha256": checkpoint_sha256}
            or not isinstance(worker_report, dict) or worker_report.get("path") != job.get("report_path")
            or worker_report.get("sha256") != JOURNAL.file_sha(paths["report"])
            or worker.get("measurement") != report_probe["measurement"]
            or backend != {"execution": execution, "measurement": worker["measurement"], "report": report}):
        raise ValueError("Brev actual probe 완료·계보·GPU 원본 결합이 다르다")
    return {
        "measurement": deepcopy(worker["measurement"]), "seed": job["seed"],
        "run_name": job["run_name"], "stage": job["stage"], "gpu_s": float(execution["wall_s"]),
        "source_identity": {
            key: deepcopy(source[key]) for key in (
                "source_anchor", "protocol", "environment", "selected_interpreter", "source_sha256_lf",
                "runtime_source_sha256_lf", "python_package_source_sha256_lf",
            )
        },
    }


def _probe_execution_source(root: Path, registration: object) -> dict:
    if not isinstance(registration, dict) or set(registration) != {"source_anchor", "protocol"}:
        raise ValueError("probe execution registration 형식이 다르다")
    source_anchor = registration.get("source_anchor")
    if (not isinstance(source_anchor, str) or len(source_anchor) != 40
            or any(character not in "0123456789abcdef" for character in source_anchor)):
        raise ValueError("probe execution source anchor가 다르다")
    protocol_name, _, protocol = WORKER._protocol_reference(
        root, registration.get("protocol"), "probe execution protocol",
    )
    training = protocol.get("training")
    environment = protocol.get("environment")
    resources = protocol.get("resources")
    if (protocol.get("schema") != "brev_recovery_protocol_v4"
            or protocol.get("registered_before_measurement") is not True
            or not isinstance(training, dict) or not isinstance(training.get("task"), str)
            or not isinstance(environment, dict)
            or not isinstance(resources, dict) or resources.get("instance_type") != "l40s-48gb.1x"
            or resources.get("max_owned_instances") != 1):
        raise ValueError("probe execution protocol 등록값이 다르다")
    expected_environment = deepcopy(environment)
    expected_environment.update(cuda_available=True, cuda_device_count=1)
    return {
        "source_anchor": source_anchor,
        "protocol": {"path": protocol_name, "sha256_lf": registration["protocol"]["sha256_lf"]},
        "environment": expected_environment,
        "source_sha256_lf": deepcopy(protocol.get("source_sha256_lf")),
        "runtime_source_sha256_lf": deepcopy(protocol.get("runtime_source_sha256_lf")),
        "python_package_source_sha256_lf": deepcopy(protocol.get("python_package_source_sha256_lf")),
        "task": training["task"],
    }


def _portable_probe(root: Path, item: dict, snapshot: dict, case_checkpoint: str,
                    reference_checkpoint: str, expected_source: dict) -> tuple[dict, str]:
    names = {"seal", "authorization", "ledger", "case", "reference"}
    if not isinstance(item, dict) or set(item) != names:
        raise ValueError("probe portable evidence 필드가 다르다")
    paths = {name: _ref(root, item[name], f"probe {name}") for name in names}
    values = {name: _read(path) for name, (path, _) in paths.items() if name != "ledger"}
    seal, auth = values["seal"], values["authorization"]
    seal_sha = paths["seal"][1]
    seal_keys = {"schema", "status", "authorization_path", "authorization_sha256", "arm", "epoch",
                 "evidence_kind", "snapshot_digest", "snapshot", "case_path", "case_sha256",
                 "reference_path", "reference_sha256", "request_id", "receipt"}
    auth_keys = {"schema", "arm", "epoch", "evidence_kind", "snapshot_digest", "snapshot", "choice",
                 "prereg", "execution_plan", "request_id"}
    if (set(seal) != seal_keys or set(auth) != auth_keys
            or seal.get("schema") != "selection_probe_seal_v1" or seal.get("status") != "completed"
            or seal.get("snapshot") != snapshot or seal.get("snapshot_digest") != CONTROLLER.L.digest(snapshot)
            or auth.get("schema") != "selection_probe_authorization_v1"
            or auth.get("snapshot") != snapshot or auth.get("snapshot_digest") != seal["snapshot_digest"]
            or seal.get("authorization_sha256") != paths["authorization"][1]
            or seal.get("case_sha256") != paths["case"][1]
            or seal.get("reference_sha256") != paths["reference"][1]
            or seal.get("request_id") != auth.get("request_id")
            or seal.get("evidence_kind") != CONTROLLER.ACTUAL_EVIDENCE
            or auth.get("evidence_kind") != CONTROLLER.ACTUAL_EVIDENCE
            or seal.get("arm") != auth.get("arm") or seal.get("epoch") != auth.get("epoch")):
        raise ValueError("probe seal/auth/raw portable 결합이 다르다")
    choice = auth.get("choice")
    if not isinstance(choice, dict):
        raise TypeError("probe authorization choice가 object가 아니다")
    spec = CONTROLLER._frozen_request_spec(snapshot, choice)
    if auth.get("prereg") != spec["prereg"]:
        raise ValueError("probe authorization prereg/evidence 결합이 다르다")
    plan = auth.get("execution_plan")
    plan_keys = {"stage", "run_name", "checkpoint_sha256", "seed", "args", "gpu_contract", "reference"}
    reference_keys = {"evidence_path", "evidence_sha256", "checkpoint_sha256", "stage",
                      "run_name", "seed", "gpu_contract"}
    reference_plan = plan.get("reference") if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or set(plan) != plan_keys
            or not isinstance(reference_plan, dict) or set(reference_plan) != reference_keys
            or plan.get("checkpoint_sha256") != case_checkpoint
            or plan.get("reference", {}).get("checkpoint_sha256") != reference_checkpoint
            or plan.get("reference", {}).get("evidence_sha256") != paths["reference"][1]
            or seal.get("reference_path") != reference_plan.get("evidence_path")
            or plan.get("args") != spec["args"] or type(plan.get("seed")) is not int
            or type(reference_plan.get("seed")) is not int
            or plan.get("gpu_contract") not in CONTROLLER.GPU_CONTRACTS
            or reference_plan.get("gpu_contract") not in CONTROLLER.GPU_CONTRACTS):
        raise ValueError("probe execution plan과 현재 F1/NONE checkpoint가 다르다")
    request_id = auth["request_id"]
    ledger = CONTROLLER.S.M.RegistryLedger(paths["ledger"][0], CONTROLLER.RC.registry())
    request = ledger.requests().get(request_id, {})
    events = [row for row in ledger.events() if row.get("request_id") == request_id]
    registry_digest = ledger.registry_digest
    effect_key = request.get("effect_key")
    args_digest = CONTROLLER.L.digest(spec["args"])
    if ([row.get("event") for row in events] != ["proposed", "approved", "consumed", "receipt"]
            or events[-1] != seal.get("receipt") or request.get("state") != "done"
            or request.get("prereg_digest") != CONTROLLER.L.digest(spec["prereg"])
            or request.get("probe") != spec["probe"]
            or request.get("args_digest") != args_digest
            or any(row.get("registry_digest") != registry_digest
                   or row.get("effect_key") != effect_key for row in events)
            or any(row.get("args_digest") != args_digest for row in (events[0], events[2], events[3]))):
        raise ValueError("probe 승인·소비·receipt 원장이 완결되지 않았다")
    receipt = seal["receipt"]
    case, reference = values["case"], values["reference"]
    probe = spec["probe"]
    case_actual = _brev_actual_probe(root, case, side="case", probe=probe,
                                     checkpoint_sha256=case_checkpoint, expected_source=expected_source)
    reference_actual = _brev_actual_probe(root, reference, side="reference", probe=probe,
                                          checkpoint_sha256=reference_checkpoint, expected_source=expected_source)
    if (case_actual["seed"] != plan["seed"] or reference_actual["seed"] != reference_plan["seed"]
            or case_actual["run_name"] != plan["run_name"]
            or reference_actual["run_name"] != reference_plan["run_name"]
            or case_actual["stage"] != plan["stage"] or reference_actual["stage"] != reference_plan["stage"]
            or case_actual["source_identity"] != reference_actual["source_identity"]):
        raise ValueError("Brev case/reference 실행 계획·자원 계보가 다르다")
    case_measurement = case_actual["measurement"]
    reference_measurement = reference_actual["measurement"]
    if not isinstance(case_measurement, dict) or not isinstance(reference_measurement, dict):
        raise TypeError("probe case/reference measurement가 object가 아니다")
    outcome = CONTROLLER.classify_raw(probe, case_measurement, reference_measurement)
    receipt_gpu_s = _finite(receipt.get("gpu_s"), "probe gpu_s")
    if (receipt.get("outcome") != outcome or receipt.get("measurement") != case_measurement
            or not case_actual["gpu_s"] <= receipt_gpu_s <= spec["budget_gpu_s"]
            or receipt.get("exit_code") != 0
            or receipt.get("over_budget") is not False):
        raise ValueError("probe 분류와 완료 receipt가 다르다")
    return {"probe": probe, "outcome": outcome, "measurement": case_measurement,
            "evidence_kind": auth["evidence_kind"], "request_id": request_id,
            "seal_sha256": seal_sha}, seal_sha


def _evidence(context: dict[str, Any], path: Path) -> tuple[dict, dict, str]:
    manifest = _read(path)
    expected = {"schema", "protocol_sha256_lf", "probe_execution", "source_groups"}
    if set(manifest) != expected or manifest.get("schema") != EVIDENCE_SCHEMA \
            or manifest.get("protocol_sha256_lf") != context["protocol_sha256_lf"]:
        raise ValueError("evidence locations와 actual protocol 결합이 다르다")
    groups = manifest["source_groups"]
    if not isinstance(groups, dict) or set(groups) != set(COMPARISON.SEED_GROUPS):
        raise ValueError("두 source group evidence가 필요하다")
    labels, probes = {}, {}
    input_rows = context["prepared"]["protocol"]["rows"]
    checkpoints = {(row["seed_group"], row["source_role"]): row["checkpoint_sha256"] for row in input_rows}
    source_protocol_sha = input_rows[0]["source_protocol_sha256"]
    source_ref = context["prepared"]["protocol"]["source_protocol"]
    _, _, source_protocol = WORKER._protocol_reference(ROOT, source_ref, "source protocol")
    source_protocol["_verified_sha256_lf"] = source_ref["sha256_lf"]
    expected_probe_source = _probe_execution_source(ROOT, manifest["probe_execution"])
    bands = _read((ROOT / source_protocol["evaluation"]["bands_path"]).resolve(strict=True))["bands"]
    for group in COMPARISON.SEED_GROUPS:
        item = groups[group]
        if not isinstance(item, dict) or set(item) != {"evaluation", "probe_evidence"}:
            raise ValueError("source group evidence 필드가 다르다")
        reports, _ = _evaluate_evidence(ROOT, context, group, item["evaluation"], source_protocol, input_rows)
        measured = RECOVERY.recovery(reports["F1"], reports["NONE"], bands)
        normal = RECOVERY.recovery(reports["NONE"], reports["NONE"], bands)
        label = {"recovered": "healthy", "not_recovered": "unhealthy", "unknown": "unknown"}
        labels[group] = {
            "normal": {"checkpoint_sha256": checkpoints[(group, "normal")],
                       "fixed_eval_artifact_sha256": JOURNAL.digest(reports["NONE"]),
                       "label": label[normal["outcome"]]},
            "registered-f1": {"checkpoint_sha256": checkpoints[(group, "registered-f1")],
                              "fixed_eval_artifact_sha256": JOURNAL.digest(reports["F1"]),
                              "label": label[measured["outcome"]]},
        }
        auth_ref = item["probe_evidence"].get("authorization") if isinstance(item["probe_evidence"], dict) else None
        auth_path, _ = _ref(ROOT, auth_ref, f"{group} probe authorization")
        snapshot = _read(auth_path).get("snapshot")
        if not isinstance(snapshot, dict):
            raise TypeError("probe authorization snapshot이 object가 아니다")
        verified, seal_sha = _portable_probe(
            ROOT, item["probe_evidence"], snapshot, checkpoints[(group, "registered-f1")],
            checkpoints[(group, "normal")], expected_probe_source,
        )
        probes[group] = {"checkpoint_sha256": checkpoints[(group, "registered-f1")],
                         "receipt_sha256": seal_sha,
                         "authorization_sha256": item["probe_evidence"]["authorization"]["sha256"],
                         "probe": verified["probe"], "outcome": verified["outcome"],
                         "approval_consumed": True, "usable": verified["outcome"] != "unknown"}
    return labels, probes, source_protocol_sha


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = CALL._read_canonical(path.resolve(strict=True))
        except (TypeError, ValueError) as error:
            raise FileExistsError(f"기존 {path.name}을 검증할 수 없다") from error
        if existing != value:
            raise FileExistsError(f"기존 {path.name}과 새 집계가 다르다")
        return
    CALL._write_new(path, value)


def _check_output_pair(output: Path, choice: dict[str, Any], summary: dict[str, Any]) -> bool:
    paths = ((output / "choice_results.json", choice), (output / "comparison_summary.json", summary))
    present = [path.exists() or path.is_symlink() for path, _ in paths]
    if any(present) and not all(present):
        raise FileExistsError("집계 출력 두 파일이 부분 생성 상태다")
    if all(present):
        for path, expected in paths:
            _write_once(path, expected)
        return True
    return False


def summarize(context: dict[str, Any], evidence_manifest: Path, output_dir: Path,
              finalize: bool = False) -> dict[str, Any]:
    """원본 판단·평가·승인 probe를 재검증해 경로 없는 집계만 만든다."""
    state = inspect_run(context)
    stage = Path(context["stage"])
    decision = stage / "decisions" / state["next_call_id"] if state.get("next_call_id") else None
    if _owner_present(stage, decision):
        raise RuntimeError("owner가 남은 실행은 집계할 수 없다")
    labels, probes, source_protocol_sha = _evidence(context, evidence_manifest)
    choices = project_results(context)
    summary = COMPARISON.summarize(
        context["prepared"]["protocol"]["rows"], choices,
        method_protocol_sha256=context["prepared"]["protocol"]["rows"][0]["method_protocol_sha256"],
        source_protocol_sha256=source_protocol_sha, source_labels=labels, probe_support=probes,
    )
    summary["api_usage"] = _usage(context)
    summary["evidence_manifest_sha256"] = JOURNAL.file_sha(evidence_manifest)
    result = {"status": state["status"], "choice_results": choices, "comparison_summary": summary,
              "finalized": False}
    if not finalize:
        return result
    terminal_boundary = state["counts"]["terminal"] == 24 or state["api"]["remaining"] == 0
    permanent_orphan = state["status"] == "blocked_pending" and state["api"]["pending"] > 0
    if not (terminal_boundary or permanent_orphan):
        raise RuntimeError("finalize 경계에 도달하지 않았다")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    choice_doc = {"schema": "vision_selection_choice_results_v1", "rows": choices,
                  "api_usage": summary["api_usage"]}
    if not _check_output_pair(output, choice_doc, summary):
        _write_once(output / "choice_results.json", choice_doc)
        _write_once(output / "comparison_summary.json", summary)
    result["finalized"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "remaining_commands", "summarize"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--input-locations", required=True, type=Path)
    parser.add_argument("--evidence-locations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    context = load_context(args.protocol, args.protocol_sha256_lf, args.source_anchor, args.input_locations)
    if args.command == "verify":
        result = inspect_run(context)
    elif args.command == "remaining_commands":
        result = remaining_commands(context)
    else:
        if args.evidence_locations is None or args.output_dir is None:
            parser.error("summarize에는 --evidence-locations와 --output-dir가 필요하다")
        result = summarize(context, args.evidence_locations, args.output_dir, args.finalize)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
