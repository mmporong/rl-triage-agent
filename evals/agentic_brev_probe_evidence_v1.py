"""POSIX probe 원문을 읽어 검증하는 CPU evidence. 실행·승인·분류는 수행하지 않는다."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import brev_recovery_backend_v4 as B
import model_decision_journal_v1 as J
import vision_remote_gpu_v1 as GPU
from lease_timestamps import parse_lease_timestamp

SCHEMA = "agentic_brev_probe_evidence_v1"
REQUEST_SCHEMA = "agentic_brev_probe_request_v1"
ACTUAL, FIXTURE = "actual_brev_posix_probe_v1", "cpu_fixture"
GPU_CONTRACT = "brev_posix_v1"
PROBES = (*B.LEGACY_PROBES, B.OPTIMIZER_PROBE)
FILES = {"job", "worker", "report", "execution", "backend_result", "report_jobs"}
HASHES = ("source_sha256_lf", "runtime_source_sha256_lf", "python_package_source_sha256_lf")
JOB_KEYS = {"schema", "mode", "stage", "lab", "repo", "protocol_path", "protocol_sha256_lf",
    "source_commit_sha", *HASHES, "arm", "seed", "run_name", "probe", "checkpoint_path", "checkpoint_sha256",
    "report_path", "training_manifest_path", "inspection_manifest_path", "records_path", "readback_path"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest(value: object) -> str:
    return J.digest(value)


def _pairs(rows: list) -> dict:
    value: dict = {}
    for key, item in rows:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def loads(raw: str) -> dict:
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    require(isinstance(value, dict), "JSON object required")
    J._validate_json(value, "probe evidence")
    return value


def _logical(value: str) -> PurePosixPath:
    require(isinstance(value, str) and value.startswith("/") and not value.startswith("//")
            and "\\" not in value and "\x00" not in value
            and all(p not in {"", ".", ".."} for p in value[1:].split("/")), "canonical POSIX path required")
    return PurePosixPath(value)


def _stamp(value: str) -> datetime:
    return parse_lease_timestamp(value)


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


class Resolver:
    """등록된 disjoint 원격 root만 해석한다. raw path와 파일은 바꾸지 않는다."""

    def __init__(self, roots: list, expected_sha256: str):
        require(digest(roots) == expected_sha256 and isinstance(roots, list) and bool(roots), "root-map SHA/list")
        self.rows: list[tuple[PurePosixPath, Path]] = []
        self.inventory: dict[str, str] = {}
        for row in roots:
            require(isinstance(row, dict) and set(row) == {"remote", "local"}, "root-map fields")
            remote, local = _logical(row["remote"]), Path(row["local"])
            require(local.is_absolute() and local == local.resolve(strict=True) and local.is_dir()
                    and not any(p.is_symlink() for p in (local, *local.parents)), "root-map local root")
            require(all(remote != old and remote not in old.parents and old not in remote.parents
                        for old, _ in self.rows), "duplicate/overlapping remote root")
            self.rows.append((remote, local))

    def path(self, logical: str) -> Path:
        source = _logical(logical)
        matches = [(remote, local) for remote, local in self.rows if remote == source or remote in source.parents]
        require(len(matches) == 1, "path outside registered root-map")
        remote, local = matches[0]
        path = local.joinpath(*source.relative_to(remote).parts)
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                and path.stat().st_nlink == 1 and path.resolve().is_relative_to(local), "single regular source required")
        return path

    def file(self, reference: dict, *, lf: bool = False) -> Path:
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}
                and re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256"))) is not None, "raw reference fields")
        path = self.path(reference["path"])
        raw = path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        require(hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest() == reference["sha256"],
                "raw source SHA mismatch")
        require(reference["path"] not in self.inventory or self.inventory[reference["path"]] == raw_sha,
                "source changed during verification")
        self.inventory[reference["path"]] = raw_sha
        return path

    def read(self, reference: dict) -> dict:
        return loads(self.file(reference).read_text(encoding="utf-8"))

    def observation(self, reference: dict, resource: dict) -> dict:
        path = self.file(reference)
        receipt = self.read(reference)
        for kind in ("gpu", "process"):
            name = receipt[f"{kind}_stdout_file"]
            require(isinstance(name, str) and PurePosixPath(name).name == name, "GPU CSV filename")
            logical = str(_logical(reference["path"]).parent / name)
            self.file({"path": logical, "sha256": receipt[f"{kind}_stdout_sha256"]})
        return GPU.verify_observation_reference({"path": str(path), "sha256": reference["sha256"]},
                                                resource["resource_id"], resource["gpu_uuid"])


def _source(request: dict, resolver: Resolver, job: dict) -> dict:
    source = request["source"]
    require(set(source) == {"artifact", "checkpoint", "telemetry", "manifest", "lineage_id"}
            and isinstance(source["lineage_id"], str) and bool(source["lineage_id"]), "source binding fields")
    artifact = resolver.read(source["artifact"])
    resolver.file(source["checkpoint"])
    resolver.read(source["telemetry"])
    manifest = resolver.read(source["manifest"])
    require(artifact["seed"] == request["seed"] == job["seed"]
            and artifact["arm"] == request["arm"] == job["arm"], "wrong source seed/arm")
    for key in ("checkpoint", "telemetry"):
        require(artifact[key]["sha256"] == source[key]["sha256"], "wrong source checkpoint/telemetry")
        require(resolver.file({"path": artifact[key]["path"], "sha256": source[key]["sha256"]})
                == resolver.file(source[key]), "source root relocation mismatch")
    require(resolver.file({"path": job["checkpoint_path"], "sha256": job["checkpoint_sha256"]})
            == resolver.file(source["checkpoint"])
            and job["checkpoint_sha256"] == source["checkpoint"]["sha256"], "wrong job checkpoint")
    require(resolver.file({"path": job["training_manifest_path"], "sha256": source["manifest"]["sha256"]})
            == resolver.file(source["manifest"])
            == resolver.file({"path": artifact["manifest_path"], "sha256": source["manifest"]["sha256"]})
            and manifest["seed"] == request["seed"] and manifest["arm"] == request["arm"], "training manifest binding")
    return {**artifact, "manifest": manifest}


def _execution(request: dict, resolver: Resolver, job: dict, execution: dict, worker: dict) -> None:
    require(set(execution) == {"schema", "mode", "completed", "started_at", "finished_at", "exit_code",
        "timed_out", "cleanup_verified", "launch", "gpu_before", "gpu_after", "wall_s", "source_before",
        "source_after", "gpu_before_observation", "gpu_after_observation"}, "incomplete POSIX execution fields")
    require(execution.get("schema") == "brev_recovery_posix_execution_v1"
            and execution.get("mode") == "probe" and execution.get("completed") is True
            and type(execution.get("exit_code")) is int and execution["exit_code"] == 0
            and execution.get("timed_out") is False and execution.get("cleanup_verified") is True
            and execution.get("gpu_before") == [] and execution.get("gpu_after") == []
            and _positive(execution.get("wall_s")) and _positive(worker.get("wall_s")), "incomplete probe/cleanup")
    identity = execution.get("launch", {})
    require(type(identity.get("pid")) is int and identity["pid"] > 0
            and identity.get("pgid") == identity["pid"] == identity.get("sid")
            and identity.get("posix_group_verified") is True
            and type(identity.get("start_ticks")) is int and identity["start_ticks"] > 0, "owned POSIX identity required")
    started, finished = _stamp(execution["started_at"]), _stamp(execution["finished_at"])
    require(_positive(request["max_wall_s"]) and started < finished
            and (finished - started).total_seconds() <= request["max_wall_s"]
            and execution["wall_s"] <= request["max_wall_s"]
            and worker["wall_s"] <= execution["wall_s"], "registered probe time window")
    expected = request["expected_source"]
    require(re.fullmatch(r"[0-9a-f]{40}", str(job["source_commit_sha"])) is not None,
            "source anchor SHA format")
    for key in HASHES:
        require(isinstance(job[key], dict) and bool(job[key]) and all(
            isinstance(name, str) and not PurePosixPath(name).is_absolute() and "\\" not in name
            and all(part not in {"", ".", ".."} for part in name.split("/"))
            and re.fullmatch(r"[0-9a-f]{64}", str(sha)) is not None for name, sha in job[key].items()), "source hash map")
    require(execution["source_before"] == execution["source_after"] == expected
            and expected.get("lab_tracked_clean") is True
            and expected.get("lab_head") == expected["environment"]["isaaclab_sha"]
            and expected["source_anchor"] == job["source_commit_sha"]
            and expected["protocol"]["sha256_lf"] == job["protocol_sha256_lf"]
            and str(_logical(job["protocol_path"]).relative_to(_logical(job["repo"])))
            == expected["protocol"]["path"]
            and all(expected[key] == job[key] and bool(job[key]) for key in HASHES), "wrong probe source snapshot")
    selected, environment = expected["selected_interpreter"], expected["environment"]
    require(selected["python"] == environment["python"]
            and selected["packages"] == {k: v for k, v in environment.items()
                if k not in {"python", "isaaclab_sha", "cuda_available", "cuda_device_count"}}, "selected interpreter binding")
    resolver.file({"path": job["protocol_path"], "sha256": job["protocol_sha256_lf"]}, lf=True)
    resource = request["resource"]
    proof = resolver.read(resource["proof"])
    require(proof.get("schema") == "brev_resource_proof_v1" and proof.get("status") == "granted"
            and all(proof.get(k) == resource[k] for k in ("resource_id", "gpu_uuid", "lease_id", "protocol_sha256"))
            and isinstance(resource["lease_id"], str) and bool(resource["lease_id"])
            and _stamp(proof["issued_at_utc"]) <= started < finished <= _stamp(proof["expires_at_utc"]),
            "resource/lease binding or execution outside lease")
    resolver.file(resource["protocol"])
    require(resource["protocol"]["sha256"] == resource["protocol_sha256"], "resource execution protocol SHA")
    capture = resource["capture"]
    require(set(capture) == GPU.CAPTURE_KEYS and capture["resource_id"] == resource["resource_id"]
            and capture["gpu_uuid"] == resource["gpu_uuid"]
            and all(proof.get(k) == v for k, v in capture.items()), "registered capture/resource proof")
    relocated = dict(capture)
    for prefix in ("provider_inventory", "nvidia_gpu_csv", "nvidia_process_csv"):
        relocated[prefix + "_path"] = str(resolver.file(
            {"path": capture[prefix + "_path"], "sha256": capture[prefix + "_sha256"]}))
    GPU.verify_capture_proof(relocated)  # 원문 변경 없이 등록 root-map으로 찾은 파일만 전달한다.
    before = resolver.observation(execution["gpu_before_observation"], resource)
    after = resolver.observation(execution["gpu_after_observation"], resource)
    require(before["ordinal"] < after["ordinal"], "duplicate/reversed GPU observations")


def _optimizer(request: dict, resolver: Resolver, job: dict, worker: dict, artifact: dict, report: dict) -> dict:
    """원래 observer assembler로 3 update를 재계산한다. POSIX 원문 경로는 유지한다."""
    descriptor = worker["inspection_manifest"]
    require(descriptor == request["files"]["inspection"], "optimizer inspection reference")
    inspection = resolver.read(descriptor)
    training = artifact["manifest"]
    expected = {**training, "mode": "inspection", "max_iterations": 3, "run_name": job["run_name"],
        "records_path": job["records_path"], "readback_path": job["readback_path"],
        "source_checkpoint_sha256": job["checkpoint_sha256"], "source_commit_sha": job["source_commit_sha"],
        "protocol_path": request["expected_source"]["protocol"]["path"],
        "protocol_sha256_lf": job["protocol_sha256_lf"], "source_sha256_lf": job["source_sha256_lf"],
        "runtime_source_sha256_lf": job["runtime_source_sha256_lf"]}
    require(inspection == expected and set(inspection) == B.RESIDUAL.REQUIRED
            and all(inspection[k] == v for k, v in B.RESIDUAL.FIXED.items()), "optimizer inspection/source binding")
    for key, name in (("inspection_manifest_path", "inspection"), ("records_path", "records"), ("readback_path", "readback")):
        require(job[key] == request["files"][name]["path"], "optimizer stage path binding")
    rows: list[Mapping[str, Any]] = [loads(line) for line in resolver.file(
        request["files"]["records"]).read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 3 and [r.get("update_index") for r in rows] == [0, 1, 2], "incomplete/duplicate optimizer updates")
    resolver.read(request["files"]["readback"])
    measured = B.OPTIMIZER.assemble_inspection(rows, B.RESIDUAL.observer_manifest(inspection))
    require(report == measured, "optimizer raw records/report mismatch")
    return measured


def build(request: dict, roots: list, *, expected_request_sha256: str,
          expected_roots_sha256: str, allow_fixture: bool = False) -> dict:
    """등록 request SHA는 caller가 신뢰 경계 밖에서 제공한다. 이 함수는 권한을 발급하지 않는다."""
    require(digest(request) == expected_request_sha256 and request.get("schema") == REQUEST_SCHEMA, "registered request SHA/schema")
    require(type(allow_fixture) is bool and request.get("evidence_kind") in {ACTUAL, FIXTURE}
            and (request["evidence_kind"] == ACTUAL or allow_fixture), "fixture requires explicit allow_fixture")
    require(re.fullmatch(r"[0-9a-f]{64}", str(request.get("operation_id"))) is not None
            and request.get("action") == "probe" and request.get("probe") in PROBES
            and request.get("side") in {"case", "reference"}
            and request.get("arm") == ("F1" if request["side"] == "case" else "NONE")
            and type(request.get("seed")) is int and request["seed"] >= 0, "registered probe operation/identity")
    resolver = Resolver(roots, expected_roots_sha256)
    optimizer = request["probe"] == B.OPTIMIZER_PROBE
    expected_files = (FILES - {"report_jobs"}) | {"inspection", "records", "readback"} if optimizer else FILES
    require(set(request["files"]) == expected_files, "missing/extra probe file inventory")
    paths = {name: resolver.file(reference) for name, reference in request["files"].items()}
    require(len(set(paths.values())) == len(paths), "duplicate probe source file")
    values = {name: resolver.read(request["files"][name]) for name in expected_files - {"records"}}
    job, worker, execution, report = (values[k] for k in ("job", "worker", "execution", "report"))
    stage = _logical(request["stage"])
    require(all(_logical(v["path"]).parent == stage for v in request["files"].values()), "raw files outside registered stage")
    require(set(job) == JOB_KEYS and job.get("schema") == "brev_recovery_worker_job_v1" and job.get("mode") == "probe"
            and job.get("stage") == request["stage"] and job.get("probe") == request["probe"]
            and job.get("run_name") == request["run_name"]
            and job.get("report_path") == request["files"]["report"]["path"]
            and set(worker) == {"schema", "completed", "probe", "measurement", "report", "inspection_manifest", "wall_s"}
            and worker.get("schema") == "brev_recovery_probe_worker_v1" and worker.get("completed") is True
            and worker.get("probe") == request["probe"] and worker.get("report") == request["files"]["report"], "job/worker identity or report reference")
    artifact = _source(request, resolver, job)
    _execution(request, resolver, job, execution, worker)
    backend = values["backend_result"]
    if optimizer:
        measured = _optimizer(request, resolver, job, worker, artifact, report)
        descriptor = {**worker["inspection_manifest"], "body": values["inspection"]}
        expected_backend = {"execution": execution, "measurement": measured, "report": report, "inspection_manifest": descriptor}
    else:
        rows = values["report_jobs"]
        expected_job = {"name": request["run_name"], "checkpoint": job["checkpoint_path"],
                        "probes": [request["probe"]], "output": job["report_path"]}
        require(rows == {"fault": request["arm"], "seed": request["seed"], "task": B.TASK, "jobs": [expected_job]}
                and worker["inspection_manifest"] is None
                and report.get("checkpoint") == {"file": _logical(job["checkpoint_path"]).name,
                                                   "sha256": job["checkpoint_sha256"]}, "probe jobs/checkpoint binding")
        measured = B.validate_probe_result(artifact, request["probe"], request["run_name"],
            {"measurement": worker["measurement"], "report": report, "execution": execution})
        expected_backend = {"execution": execution, "measurement": measured, "report": report}
    require(worker["measurement"] == measured and backend == expected_backend, "worker/backend measurement mismatch")
    for logical, expected_sha in resolver.inventory.items():
        require(J.file_sha(resolver.path(logical)) == expected_sha, "raw changed during verification")
    return {"schema": SCHEMA, "request": deepcopy(request), "request_sha256": expected_request_sha256,
        "root_map_sha256": expected_roots_sha256, "evidence_kind": request["evidence_kind"],
        "gpu_contract": GPU_CONTRACT, "probe": request["probe"], "checkpoint_sha256": job["checkpoint_sha256"],
        "seed": job["seed"], "run_name": job["run_name"], "stage": job["stage"], "side": request["side"],
        "measurement": measured, "wall_s": execution["wall_s"], "raw_files_sha256": dict(resolver.inventory),
        "source_snapshot_sha256": digest(execution["source_before"]), "approval_granted": False,
        "repair_supported": False, "execution_permitted": False}


def verify(evidence: dict, roots: list, *, expected_request_sha256: str,
           expected_roots_sha256: str, allow_fixture: bool = False) -> dict:
    rebuilt = build(evidence["request"], roots, expected_request_sha256=expected_request_sha256,
                    expected_roots_sha256=expected_roots_sha256, allow_fixture=allow_fixture)
    require(evidence == rebuilt, "evidence differs from raw recomputation")
    return rebuilt


def write_new(path: Path, evidence: dict) -> None:
    J._write_new(path, evidence)


def action_support(action: str) -> dict:
    return {"status": "supported_for_verification" if action == "probe" else "HOLD",
            "reason": None if action == "probe" else "repair_unsupported",
            "execution_permitted": False, "approval_granted": False}
