"""V04 source 배치 전에 같은 runtime·GPU lock에서 NONE 20회 학습과 render-sync smoke를 점검한다.

사전 점검은 인프라 확인 자료다. 모델 판단·회복 측정 자료로 쓰지 않으며 core 학습 원장과
source action 목록에 기록하지 않는다. 별도 ``preflight`` namespace에 intent/raw/receipt를
write-once로 남기고, 결과가 확정되지 않은 단계는 자동으로 다시 실행하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import brev_recovery_backend_v4 as B  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as RECORD
import run_vision_brev_render_v2 as RENDER
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]
import vision_selection_sources_v1 as SOURCES  # pyright: ignore[reportMissingImports]

SOURCE = RENDER.SOURCE
SCHEMA = "vision_brev_preflight_protocol_v1"
METHOD = "same_runtime_none20_then_render_sync_smoke_v1"
SCOPE = "infrastructure_preflight_not_model_or_recovery_evidence_v1"
INTENT_SCHEMA = "vision_brev_preflight_stage_intent_v1"
RAW_SCHEMA = "vision_brev_preflight_stage_raw_v1"
RECEIPT_SCHEMA = "vision_brev_preflight_stage_receipt_v1"
FAILURE_SCHEMA = "vision_brev_preflight_stage_failure_v1"
INVALID_SCHEMA = "vision_brev_preflight_stage_invalid_v1"
COMPLETION_SCHEMA = "vision_brev_preflight_completion_v1"
TRAIN_STAGE, SMOKE_STAGE = "train_none20", "render_sync_smoke"
STAGES = (TRAIN_STAGE, SMOKE_STAGE)
STAGE_WALL_S = {TRAIN_STAGE: 1800, SMOKE_STAGE: 900}
PREFLIGHT_WALL_S = 2700
PREFLIGHT_ITERATIONS = 20
NUM_ENVS = 4096
SOURCE_FIRST_ACTION_RESERVE_S = SOURCE.ACTION_AUTHORIZATION_S
TOLERANCE = 1e-9
CONTROLLER_SOURCES = frozenset({
    "bench/run_vision_brev_preflight_v1.py",
    "bench/run_vision_brev_render_v2.py",
    "bench/run_vision_brev_batch_v1.py",
    "bench/brev_recovery_worker_v2.py",
    "bench/brev_compute_probe_v1.py",
    "evals/brev_recovery_backend_v4.py",
    "evals/vision_brev_backend_v1.py",
    "evals/vision_brev_render_backend_v2.py",
    "evals/vision_remote_gpu_v1.py",
    "evals/preregistered_seed_guard.py",
})
PROTOCOL_KEYS = {
    "schema", "protocol_id", "registered_before_measurement", "method", "scope", "model_use_ready",
    "source_protocol", "render_protocol", "record_profile_sha256", "controller_sources_lf",
    "training", "smoke", "order", "limits", "reservation",
}
INTENT_KEYS = {
    "schema", "protocol_id", "stage", "ordinal", "job_id", "preflight_protocol_sha256_lf",
    "source_protocol_sha256_lf", "render_protocol_sha256_lf", "resource_sha256",
    "job_binding_sha256", "started_at_utc", "cost_before_usd", "reserved_max_wall_s",
    "reserved_max_compute_usd", "cost_scope",
}
RECEIPT_KEYS = {
    "schema", "stage", "job_id", "intent_sha256", "raw_sha256", "started_at_utc",
    "finished_at_utc", "elapsed_wall_s", "cost_before_usd", "cost_after_usd", "recovered_from_cache",
}
ARTIFACT_KEYS = {
    "checkpoint", "telemetry", "seed", "arm", "original_faults", "applied_corrections", "readback",
    "execution", "manifest_path", "manifest_sha256", "manifest", "source_checkpoint_sha256",
}


def preflight_ids(seed: int) -> dict[str, str]:
    """core source/render ID와 겹치지 않는 seed 고정 파생 ID."""
    return {
        "train_job_id": f"preflight-train-s{seed}", "run_name": f"preflight-none20-s{seed}",
        "smoke_job_id": f"preflight-smoke-s{seed}", "checkpoint_logical_id": f"preflight-policy-s{seed}",
    }


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _matches(actual: Any, expected: Any) -> bool:
    if _number(expected):
        return _number(actual) and abs(float(actual) - float(expected)) <= TOLERANCE
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_matches(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_matches(left, right) for left, right in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def _core_identifiers(source: Mapping[str, Any], render: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for row in source["training_plan"]:
        names.update((row["job_id"], row["run_name"]))
    for action in source["source_action_order"]:
        names.update(value for value in action.values() if isinstance(value, str))
    for row in render["smoke_plan"]:
        names.update(row.values())
    return names


def _limits(quote: float) -> dict[str, Any]:
    return {
        TRAIN_STAGE: {"max_wall_s": STAGE_WALL_S[TRAIN_STAGE],
                      "max_compute_usd": quote * STAGE_WALL_S[TRAIN_STAGE] / 3600.0},
        SMOKE_STAGE: {"max_wall_s": STAGE_WALL_S[SMOKE_STAGE],
                      "max_compute_usd": quote * STAGE_WALL_S[SMOKE_STAGE] / 3600.0},
        "total_max_wall_s": PREFLIGHT_WALL_S,
        "total_max_compute_usd": quote * PREFLIGHT_WALL_S / 3600.0,
    }


def _reservation(source: Mapping[str, Any], render: Mapping[str, Any]) -> dict[str, Any]:
    resources = source["resources"]
    limits = resources["action_limits"]
    return {
        "source_actions_usd": sum(float(limits[action["kind"]]["max_compute_usd"])
                                  for action in source["source_action_order"]),
        "render_smokes_usd": len(render["smoke_plan"]) * float(render["smoke_limits"]["max_compute_usd"]),
        "preflight_usd": float(resources["compute_quote_usd_per_h"]) * PREFLIGHT_WALL_S / 3600.0,
        "whole_batch_budget_usd": float(resources["whole_batch_budget_usd"]),
        "source_first_action_reserve_s": SOURCE_FIRST_ACTION_RESERVE_S,
    }


def _expected_protocol(source: Mapping[str, Any], render: Mapping[str, Any], source_reference: Mapping[str, str],
                       render_reference: Mapping[str, str], seed: int,
                       controller_sources_lf: Mapping[str, str]) -> dict[str, Any]:
    ids = preflight_ids(seed)
    return {
        "schema": SCHEMA, "protocol_id": source["protocol_id"], "registered_before_measurement": True,
        "method": METHOD, "scope": SCOPE, "model_use_ready": False,
        "source_protocol": dict(source_reference), "render_protocol": dict(render_reference),
        "record_profile_sha256": RECORD.digest(RECORD.fixed_profile()),
        "controller_sources_lf": dict(controller_sources_lf),
        "training": {
            "job_id": ids["train_job_id"], "arm": "NONE", "seed": seed, "run_name": ids["run_name"],
            "iterations": PREFLIGHT_ITERATIONS, "num_envs": NUM_ENVS,
            "stage_relative": f"preflight/training/{ids['train_job_id']}",
        },
        "smoke": {
            "job_id": ids["smoke_job_id"], "mode": RECORD.RENDER_SYNC_SMOKE_MODE,
            "checkpoint_logical_id": ids["checkpoint_logical_id"], "policy_steps": 0,
            "capture_phase_physics_steps": 0, "renderer_combination_only": True,
        },
        "order": list(STAGES),
        "limits": _limits(float(source["resources"]["compute_quote_usd_per_h"])),
        "reservation": _reservation(source, render),
    }


def build_protocol(source_protocol: Mapping[str, Any], source_reference: Mapping[str, Any],
                   render_protocol: Mapping[str, Any], render_reference: Mapping[str, Any], seed: int,
                   controller_sources_lf: Mapping[str, str]) -> dict[str, Any]:
    """검증된 source/render protocol에서 preflight protocol을 결정적으로 만든다. 파일은 쓰지 않는다."""
    source = SOURCE.validate_protocol(source_protocol)
    source_ref = RENDER._source_reference(source_reference)
    render = RENDER.validate_protocol(render_protocol, source, source_ref)
    render_ref = RENDER._source_reference(render_reference)
    if type(seed) is not int or seed < 0:
        raise ValueError("preflight seed는 음이 아닌 정수여야 한다")
    protocol = _expected_protocol(source, render, source_ref, render_ref, seed, controller_sources_lf)
    return validate_protocol(protocol, source, source_ref, render, render_ref)


def validate_protocol(value: Mapping[str, Any], source_protocol: Mapping[str, Any],
                      source_reference: Mapping[str, Any], render_protocol: Mapping[str, Any],
                      render_reference: Mapping[str, Any]) -> dict[str, Any]:
    source = SOURCE.validate_protocol(source_protocol)
    source_ref = RENDER._source_reference(source_reference)
    render = RENDER.validate_protocol(render_protocol, source, source_ref)
    render_ref = RENDER._source_reference(render_reference)
    if set(value) != PROTOCOL_KEYS:
        raise ValueError("preflight protocol 필드가 다르다")
    training = value.get("training")
    seed = training.get("seed") if isinstance(training, dict) else None
    if type(seed) is not int or seed < 0:
        raise ValueError("preflight seed가 잘못됐다")
    controller = SOURCE._hash_map(value.get("controller_sources_lf"), "controller_sources_lf")
    if not CONTROLLER_SOURCES <= set(controller):
        raise ValueError("preflight controller/backend/worker source 결합이 빠졌다")
    expected = _expected_protocol(source, render, source_ref, render_ref, seed, controller)
    if not _matches(dict(value), expected):
        raise ValueError("preflight protocol 값이 source/render 파생 계약과 다르다")
    if seed in {row["seed"] for row in source["training_plan"]}:
        raise ValueError("preflight seed가 core 학습 seed와 겹친다")
    if set(preflight_ids(seed).values()) & _core_identifiers(source, render):
        raise ValueError("preflight job/run/checkpoint ID가 source/render ID와 충돌한다")
    reservation = expected["reservation"]
    total = reservation["source_actions_usd"] + reservation["render_smokes_usd"] + reservation["preflight_usd"]
    if total > reservation["whole_batch_budget_usd"] + TOLERANCE:
        raise ValueError("source10+render4+preflight 최대 예약 합이 whole batch 상한을 넘는다")
    return dict(value)


def controller_source_hashes(repo: Path) -> dict[str, str]:
    """메인이 protocol을 만들 때 쓰는 controller source LF SHA 목록."""
    return {name: SOURCE.sha256(Path(repo) / name, lf=True) for name in sorted(CONTROLLER_SOURCES)}


def _verify_controller_binding(repo: Path, anchor: str, preflight_path: Path, preflight_sha: str,
                               controller: Mapping[str, str]) -> None:
    relative = preflight_path.relative_to(repo).as_posix()
    blob = SOURCE._git(repo, "show", f"{anchor}:{relative}").stdout.encode().replace(b"\r\n", b"\n")
    if hashlib.sha256(blob).hexdigest() != preflight_sha:
        raise ValueError("source anchor의 preflight protocol blob SHA가 다르다")
    for name, expected in controller.items():
        path = repo / name
        if not path.is_file() or path.is_symlink() or SOURCE.sha256(path, lf=True) != expected:
            raise ValueError("preflight controller source LF SHA가 다르다")
    names = sorted({relative, *controller})
    SOURCE._git(repo, "ls-files", "--error-unmatch", "--", *names)
    if SOURCE._git(repo, "status", "--porcelain", "--", *names).stdout.strip():
        raise ValueError("preflight 결합 파일의 index 또는 working tree가 clean 상태가 아니다")
    head = SOURCE._git(repo, "rev-parse", "HEAD").stdout.strip()
    specs = [f"{commit}{name}" for commit in (f"{anchor}:", f"{head}:", ":") for name in names]
    blobs = SOURCE._git(repo, "rev-parse", *specs).stdout.split()
    count = len(names)
    if len(blobs) != 3 * count or blobs[:count] != blobs[count:2 * count] or blobs[:count] != blobs[2 * count:]:
        raise ValueError("preflight 결합 blob이 source anchor 뒤 변경됐다")


def _verify_gpu_observations(execution: Mapping[str, Any], resource: Mapping[str, Any]) -> None:
    for key in ("gpu_before_observation", "gpu_after_observation"):
        reference = execution.get(key)
        if not isinstance(reference, Mapping):
            raise TypeError(f"{key} 원문 참조가 없다")
        REMOTE.verify_observation_reference(reference, str(resource["resource_id"]), str(resource["gpu_uuid"]))


def _verify_capture_files(repo: Path, references: Mapping[str, Any], source: Mapping[str, Any],
                          source_path: Path, source_sha: str, resource: Mapping[str, Any],
                          resource_sha: str) -> None:
    """backend 없이 기존 capture/smoke 검증기로 attested 원문을 다시 읽는다."""
    protocol = B.validate_protocol({**source, "protocol_path": str(source_path), "protocol_sha256_lf": source_sha})
    protocol["_verified_sha256_lf"] = source_sha
    capture = SOURCES._verify_capture(
        repo, dict(references), RECORD.RENDER_SYNC_SMOKE_MODE, protocol, resource_sha,
        str(resource["raw_gpu_receipt_sha256"]), dict(resource),
    )
    SOURCES._verify_smoke(capture)


def _capture_references_offline(repo: Path, runtime: Path, job_id: str) -> dict[str, dict[str, str]]:
    """완료 capture의 canonical ref를 backend·worker 없이 기존 보관 규칙으로 파생한다."""
    import vision_render_readiness_v2 as FINALIZE  # pyright: ignore[reportMissingImports]

    return FINALIZE.archive_capture_references(repo, runtime, job_id)


def _resource_gate(source: Mapping[str, Any], source_sha: str, resource: Mapping[str, Any],
                   at: datetime) -> dict[str, Any]:
    """기존 SourceBatch resource 검사를 주어진 시각으로 재사용한다."""
    checker = SOURCE.SourceBatch.__new__(SOURCE.SourceBatch)
    checker.protocol, checker.protocol_sha = dict(source), source_sha
    checker.now = lambda: at
    return checker._verify_resource(resource)


def _failure_kind(error: BaseException) -> str:
    kind = getattr(error, "kind", None)
    return kind if isinstance(error, B.BackendError) and isinstance(kind, str) else type(error).__name__


class PreflightLedger:
    """사전 점검 원장을 GPU·provider 호출 없이 원문 SHA와 의미로 재검증한다."""

    def __init__(self, repo: Path, runtime: Path, source: Mapping[str, Any], source_path: Path,
                 source_sha: str, render_path: Path, render_sha: str, preflight: Mapping[str, Any],
                 preflight_path: Path, preflight_sha: str, anchor: str, resource: Mapping[str, Any],
                 resource_sha: str):
        self.repo, self.runtime = Path(repo), Path(runtime)
        self.source, self.source_path, self.source_sha = dict(source), Path(source_path), source_sha
        self.render_path, self.render_sha = Path(render_path), render_sha
        self.preflight, self.preflight_path, self.preflight_sha = dict(preflight), Path(preflight_path), preflight_sha
        self.anchor, self.resource, self.resource_sha = anchor, dict(resource), resource_sha
        self.folder = self.runtime / "preflight"
        current = self.repo
        for part in self.folder.relative_to(self.repo).parts:
            current /= part
            if current.exists() and current.is_symlink():
                raise ValueError("preflight runtime 경로에 symlink가 있다")
        resources = self.source["resources"]
        self.quote = float(resources["compute_quote_usd_per_h"])
        self.cap = float(resources["whole_batch_budget_usd"])
        self.deadline = SOURCE._stamp(resources["deadline_utc"])
        self.expires = SOURCE._stamp(self.resource["expires_at_utc"])
        self.compute_started = SOURCE._stamp(self.resource["compute_started_at_utc"])
        self.training = self.preflight["training"]
        self.smoke = self.preflight["smoke"]

    # 경로와 결합값
    def stage_folder(self, index: int) -> Path:
        return self.folder / "stages" / f"{index:02d}_{STAGES[index]}"

    def train_stage(self) -> Path:
        return SOURCE.VISION._owned(self.runtime / self.training["stage_relative"], self.runtime, "preflight stage")

    def smoke_inputs(self) -> Path:
        return self.folder / "smoke-inputs" / self.smoke["job_id"]

    def smoke_stage(self) -> Path:
        return self.runtime / "record" / self.smoke["job_id"]

    def source_started(self) -> bool:
        return any((self.runtime / name).exists() for name in (
            "actions", "render-actions", "freshness.json", "manifest.json", "training.ledger.jsonl",
        ))

    def train_request(self) -> dict[str, Any]:
        return {"job_id": self.training["job_id"], "stage": str(self.train_stage()), "arm": "NONE",
                "seed": self.training["seed"], "run_name": self.training["run_name"],
                "iterations": PREFLIGHT_ITERATIONS, "num_envs": NUM_ENVS}

    def job_id(self, index: int) -> str:
        return str(self.training["job_id"] if index == 0 else self.smoke["job_id"])

    def lifecycle(self, at: datetime) -> float:
        elapsed = (at.astimezone(SOURCE.UTC_TZ) - self.compute_started).total_seconds()
        if elapsed < 0:
            raise ValueError("compute 시작 전 시각으로 lifecycle 비용을 계산할 수 없다")
        return elapsed * self.quote / 3600.0

    def source_reserve(self) -> float:
        reservation = self.preflight["reservation"]
        return float(reservation["source_actions_usd"]) + float(reservation["render_smokes_usd"])

    def remaining_reserve(self, index: int) -> float:
        later = sum(float(self.preflight["limits"][stage]["max_compute_usd"]) for stage in STAGES[index:])
        return later + self.source_reserve()

    def intent_body(self, index: int, binding: str, started_at: Any, cost_before: Any) -> dict[str, Any]:
        limit = self.preflight["limits"][STAGES[index]]
        return {
            "schema": INTENT_SCHEMA, "protocol_id": self.preflight["protocol_id"], "stage": STAGES[index],
            "ordinal": index, "job_id": self.job_id(index),
            "preflight_protocol_sha256_lf": self.preflight_sha, "source_protocol_sha256_lf": self.source_sha,
            "render_protocol_sha256_lf": self.render_sha, "resource_sha256": self.resource_sha,
            "job_binding_sha256": binding, "started_at_utc": started_at, "cost_before_usd": cost_before,
            "reserved_max_wall_s": limit["max_wall_s"], "reserved_max_compute_usd": limit["max_compute_usd"],
            "cost_scope": self.source["resources"]["cost_scope"],
        }

    def stage_state(self, index: int) -> str:
        folder = self.stage_folder(index)
        for name, state in (("failure", "failed"), ("invalid", "invalid"), ("receipt", "complete"),
                            ("intent", "pending")):
            if (folder / f"{name}.json").exists():
                return state
        return "absent"

    def reference(self, path: Path, anchor: str) -> dict[str, str]:
        return RENDER._path_reference(self.repo, path, anchor)

    # 학습 단계
    def verify_train_artifact(self, artifact: Any) -> dict[str, Any]:
        """20회 NONE 학습의 checkpoint·telemetry·manifest·readback·execution 원문을 대조한다."""
        if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
            raise ValueError("preflight 학습 artifact 필드가 다르다")
        request = self.train_request()
        seed = request["seed"]
        if (artifact["arm"] != "NONE" or artifact["seed"] != seed or artifact["original_faults"] != []
                or artifact["applied_corrections"] != [] or artifact["source_checkpoint_sha256"] is not None):
            raise ValueError("preflight 학습 artifact identity가 NONE 20회 계약과 다르다")
        expected_manifest = {"schema": "brev_recovery_dryrun_manifest_v1", "arm": "NONE", "seed": seed,
                             "iterations": PREFLIGHT_ITERATIONS}
        expected_readback = {"schema": "brev_recovery_dryrun_readback_v1", "remaining_faults": []}
        if artifact["manifest"] != expected_manifest or artifact["readback"] != expected_readback:
            raise ValueError("preflight manifest/readback이 NONE 20회 계약과 다르다")
        stage = Path(request["stage"])
        names = {"job": "job.json", "worker": "worker.json", "execution": "execution.json",
                 "dryrun": "dryrun.train.json", "manifest": "training.manifest.json",
                 "readback": "training.readback.json", "records": "optimizer.records.jsonl",
                 "telemetry": "telemetry.json"}
        files = {key: SOURCE.VISION._owned(SOURCE.VISION._single_regular(stage / name, f"preflight {key}"),
                                           self.runtime, f"preflight {key}")
                 for key, name in names.items()}
        telemetry, checkpoint = artifact["telemetry"], artifact["checkpoint"]
        if (not isinstance(telemetry, dict) or not isinstance(checkpoint, dict)
                or set(checkpoint) != {"path", "sha256"} or not isinstance(checkpoint.get("path"), str)
                or Path(str(telemetry.get("path", ""))).resolve() != files["telemetry"]
                or SOURCE.sha256(files["telemetry"]) != telemetry.get("sha256")
                or Path(str(artifact["manifest_path"])).resolve() != files["manifest"]
                or SOURCE.sha256(files["manifest"]) != artifact["manifest_sha256"]
                or SOURCE.read(files["manifest"]) != expected_manifest
                or SOURCE.read(files["readback"]) != expected_readback
                or files["records"].stat().st_size != 0):
            raise ValueError("preflight telemetry/manifest/readback 원문이 artifact와 다르다")
        checkpoint_path = SOURCE.VISION._single_regular(Path(checkpoint["path"]), "preflight checkpoint")
        if SOURCE.sha256(checkpoint_path) != checkpoint["sha256"]:
            raise ValueError("preflight checkpoint SHA가 다르다")
        execution = artifact["execution"]
        if not isinstance(execution, dict) or SOURCE.read(files["execution"]) != execution:
            raise ValueError("preflight execution 원문이 artifact와 다르다")
        self.verify_train_execution(execution)
        job = SOURCE.read(files["job"])
        expected_job = {"schema": "brev_recovery_worker_job_v1", "mode": "train", "stage": str(stage),
                        "arm": "NONE", "seed": seed, "run_name": request["run_name"],
                        "iterations": PREFLIGHT_ITERATIONS, "source_checkpoint_sha256": None,
                        "protocol_sha256_lf": self.source_sha, "source_commit_sha": self.anchor,
                        "manifest_path": str(files["manifest"]), "telemetry_path": str(files["telemetry"])}
        if any(job.get(key) != value for key, value in expected_job.items()):
            raise ValueError("preflight worker job이 NONE 20회 요청과 다르다")
        worker = SOURCE.read(files["worker"])
        expected_worker = {"schema": "brev_recovery_train_worker_v1", "completed": True, "arm": "NONE",
                           "seed": seed, "iterations": PREFLIGHT_ITERATIONS, "checkpoint": checkpoint,
                           "telemetry": telemetry, "manifest_path": artifact["manifest_path"],
                           "manifest_sha256": artifact["manifest_sha256"], "readback": expected_readback}
        if any(worker.get(key) != value for key, value in expected_worker.items()):
            raise ValueError("preflight worker 출력이 artifact와 다르다")
        dryrun = SOURCE.read(files["dryrun"])
        initial, final = dryrun.get("initial_learning_iteration"), dryrun.get("final_learning_iteration")
        expected_dryrun = {"completed": True, "mode": "train", "seed": seed, "run_name": request["run_name"],
                           "task": B.TASK, "observed_task": B.TASK, "num_envs": NUM_ENVS,
                           "observed_num_envs": NUM_ENVS, "requested_iterations": PREFLIGHT_ITERATIONS,
                           "observed_iterations": PREFLIGHT_ITERATIONS, "checkpoint_path": checkpoint["path"],
                           "checkpoint_sha256_raw": checkpoint["sha256"]}
        if (any(dryrun.get(key) != value for key, value in expected_dryrun.items())
                or type(initial) is not int or type(final) is not int
                or final - initial + 1 != PREFLIGHT_ITERATIONS
                or worker.get("learn_wall_s") != dryrun.get("learn_wall_s")
                or worker.get("wall_s") != dryrun.get("wall_s")):
            raise ValueError("preflight 관측 반복·env 수·checkpoint가 20회×4096 계약과 다르다")
        return dryrun

    def verify_train_execution(self, execution: Mapping[str, Any]) -> None:
        wall: Any = execution.get("wall_s")
        snapshot = execution.get("source_before")
        if (execution.get("schema") != "brev_recovery_posix_execution_v1" or execution.get("mode") != "train"
                or execution.get("completed") is not True or type(execution.get("exit_code")) is not int
                or execution["exit_code"] != 0 or execution.get("timed_out") is not False
                or execution.get("cleanup_verified") is not True
                or execution.get("gpu_before") != [] or execution.get("gpu_after") != []
                or not isinstance(execution.get("launch"), dict) or not isinstance(snapshot, dict)
                or snapshot != execution.get("source_after")
                or not _number(wall) or not 0 <= float(wall) <= STAGE_WALL_S[TRAIN_STAGE]):
            raise ValueError("preflight 학습 execution의 GPU·source·cleanup 경계가 유효하지 않다")
        protocol = snapshot.get("protocol")
        if (snapshot.get("source_anchor") != self.anchor or not isinstance(protocol, dict)
                or protocol.get("sha256_lf") != self.source_sha):
            raise ValueError("preflight 학습이 같은 source anchor/protocol runtime에서 실행되지 않았다")
        _verify_gpu_observations(execution, self.resource)

    def artifact_from_stage(self) -> dict[str, Any]:
        """B.Backend.train과 같은 키로 완료 stage 원문에서 artifact를 다시 조립한다."""
        stage = Path(self.train_request()["stage"])
        worker = SOURCE.read(SOURCE.VISION._single_regular(stage / "worker.json", "preflight worker"))
        execution = SOURCE.read(SOURCE.VISION._single_regular(stage / "execution.json", "preflight execution"))
        manifest_path = SOURCE.VISION._single_regular(Path(str(worker.get("manifest_path", ""))), "manifest")
        if SOURCE.sha256(manifest_path) != worker.get("manifest_sha256"):
            raise ValueError("preflight manifest SHA가 worker 출력과 다르다")
        return {
            "checkpoint": worker.get("checkpoint"), "telemetry": worker.get("telemetry"),
            "seed": self.training["seed"], "arm": "NONE", "original_faults": [], "applied_corrections": [],
            "readback": worker.get("readback"), "execution": execution,
            "manifest_path": worker.get("manifest_path"), "manifest_sha256": worker.get("manifest_sha256"),
            "manifest": SOURCE.read(manifest_path), "source_checkpoint_sha256": None,
        }

    def freshness_reference(self) -> dict[str, str]:
        path = self.folder / "freshness.json"
        freshness = SOURCE.read(SOURCE.VISION._single_regular(path, "preflight freshness"))
        if freshness.get("status") != "fresh" or freshness.get("seeds") != [self.training["seed"]]:
            raise ValueError("preflight seed freshness 소비 증거가 다르다")
        return self.reference(path, "preflight freshness")

    # smoke 단계
    def smoke_job(self, checkpoint: Mapping[str, Any], *, create: bool,
                  now: datetime | None = None) -> dict[str, Any]:
        """기존 render smoke와 같은 job/protocol/admission을 preflight checkpoint로 만든다."""
        inputs = self.smoke_inputs()
        protocol_path, admission_path = inputs / "protocol.json", inputs / "admission.json"
        if create:
            if inputs.exists():
                raise FileExistsError("intent 없는 기존 preflight smoke input은 재사용하지 않는다")
            inputs.mkdir(parents=True, exist_ok=False)
        elif not inputs.is_dir() or inputs.is_symlink():
            raise ValueError("preflight smoke input 폴더가 없다")
        resource = {"resource_id": self.resource["resource_id"], "gpu_uuid": self.resource["gpu_uuid"]}
        protocol_body = {
            "schema": RECORD.PROTOCOL_SCHEMA, "method": RECORD.METHOD, "mode": RECORD.RENDER_SYNC_SMOKE_MODE,
            "profile": RECORD.fixed_profile(), "source_commit": self.anchor,
            "source_files_lf": self.source["recording"]["source_files_lf"],
            "checkpoint_sha256": checkpoint["sha256"], "resource": resource,
        }
        if create:
            SOURCE.write_new(protocol_path, protocol_body)
        elif (not protocol_path.is_file() or protocol_path.is_symlink() or protocol_path.stat().st_nlink != 1
              or SOURCE.read(protocol_path) != protocol_body):
            raise ValueError("기존 preflight smoke protocol이 다르다")
        job = {
            "schema": RECORD.JOB_SCHEMA, "method": RECORD.METHOD, "mode": RECORD.RENDER_SYNC_SMOKE_MODE,
            "job_id": self.smoke["job_id"], "profile": RECORD.fixed_profile(),
            "source": {"repo_root": str(self.repo), "commit": self.anchor,
                       "files_lf": self.source["recording"]["source_files_lf"]},
            "protocol": {"path": str(protocol_path), "sha256": SOURCE.sha256(protocol_path)},
            "checkpoint": {"path": checkpoint["path"], "sha256": checkpoint["sha256"],
                           "logical_id": self.smoke["checkpoint_logical_id"]},
            "resource": resource,
            "remote_gpu_admission": {"resource_parent_sha256": self.resource_sha,
                                     "live_lease_proof_sha256": self.resource["raw_gpu_receipt_sha256"]},
            "output_directory": str(self.smoke_stage() / "output"),
        }
        existing = None
        if not create:
            if not admission_path.is_file() or admission_path.is_symlink() or admission_path.stat().st_nlink != 1:
                raise ValueError("기존 preflight smoke admission이 없다")
            existing = SOURCE.read(admission_path)
        if existing is not None:
            issued_at = existing.get("issued_at_utc")
        elif now is not None:
            issued_at = now.astimezone(SOURCE.UTC_TZ).isoformat()
        else:
            raise ValueError("새 smoke admission에는 현재 시각이 필요하다")
        if SOURCE._stamp(issued_at) + timedelta(seconds=STAGE_WALL_S[SMOKE_STAGE]) > min(self.expires, self.deadline):
            raise ValueError("preflight smoke admission 900초가 lease/deadline 범위를 벗어난다")
        admission = {
            "schema": RECORD.ADMISSION_SCHEMA, "status": "granted", "job_id": self.smoke["job_id"],
            "resource_kind": "remote_gpu", "lease_id": self.resource["lease_id"],
            "request_binding": RECORD.request_binding(job), "issued_at_utc": issued_at,
            "expires_at_utc": self.resource["expires_at_utc"],
            "resource_id": self.resource["resource_id"], "gpu_uuid": self.resource["gpu_uuid"],
            "resource_parent_sha256": self.resource_sha,
            "live_lease_proof_sha256": self.resource["raw_gpu_receipt_sha256"],
        }
        if existing is None:
            SOURCE.write_new(admission_path, admission)
        elif existing != admission:
            raise ValueError("기존 preflight smoke admission이 다르다")
        job["remote_gpu_admission"].update({"path": str(admission_path), "sha256": SOURCE.sha256(admission_path)})
        return job

    def smoke_payload(self, references: Mapping[str, Any]) -> dict[str, Any]:
        values = {}
        for name in ("result", "execution"):
            path, _ = RENDER._portable_reference(self.repo, references.get(name), f"preflight smoke {name}")
            values[name] = SOURCE.read(path)
        return {"status": "render_sync_smoke_complete", "model_use_ready": False,
                "result": values["result"], "execution": values["execution"]}

    def verify_smoke(self, job: Mapping[str, Any], smoke: Any, references: Any) -> None:
        """smoke 결과를 attested 원문과 기존 capture/smoke 검증기로 다시 확인한다."""
        required = {"job", "admission", "result", "execution", "attestation"}
        if not isinstance(references, dict) or not required <= set(references):
            raise ValueError("preflight smoke capture reference가 부족하다")
        for name, reference in references.items():
            RENDER._portable_reference(self.repo, reference, f"preflight smoke {name}")
        job_path, _ = RENDER._portable_reference(self.repo, references["job"], "preflight smoke job")
        admission_path, _ = RENDER._portable_reference(self.repo, references["admission"], "preflight admission")
        if (SOURCE.read(job_path) != dict(job) or job_path != self.smoke_stage().resolve() / "job.json"
                or admission_path != Path(job["remote_gpu_admission"]["path"]).resolve()):
            raise ValueError("preflight smoke capture가 등록 job/admission과 다르다")
        payload = self.smoke_payload(references)
        if smoke != payload or payload["result"].get("mode") != RECORD.RENDER_SYNC_SMOKE_MODE:
            raise ValueError("preflight smoke 결과가 attested cache와 다르다")
        _verify_capture_files(self.repo, references, self.source, self.source_path, self.source_sha,
                              self.resource, self.resource_sha)

    def evidence_window(self, index: int, raw: Mapping[str, Any]) -> tuple[datetime, datetime]:
        """backend가 남긴 실제 실행 시작·종료 시각. receipt 완료 시각은 이 종료 증거에 묶는다."""
        if index == 0:
            execution = raw["artifact"]["execution"]
            return SOURCE._stamp(execution.get("started_at")), SOURCE._stamp(execution.get("finished_at"))
        path, _ = RENDER._portable_reference(self.repo, raw["references"]["attestation"], "preflight attestation")
        attestation = SOURCE.read(path)
        started = SOURCE._stamp(attestation.get("execute_started_at_utc"))
        finished = SOURCE._stamp(attestation.get("execute_finished_at_utc"))
        attested = SOURCE._stamp(attestation.get("attested_at_utc"))
        if not started <= finished <= attested:
            raise ValueError("preflight smoke attestation 실행 시각 순서가 다르다")
        return started, attested

    # 완료 단계 재검증
    def verify_stage(self, index: int) -> dict[str, Any]:
        folder = self.stage_folder(index)
        paths = {name: folder / f"{name}.json" for name in ("intent", "raw", "receipt")}
        for name, path in paths.items():
            SOURCE.VISION._single_regular(path, f"preflight {name}")
        if (folder / "failure.json").exists() or (folder / "invalid.json").exists():
            raise ValueError("preflight 완료 receipt와 실패 기록이 함께 있다")
        intent, raw, receipt = (SOURCE.read(paths[name]) for name in ("intent", "raw", "receipt"))
        stage = STAGES[index]
        train_raw = self.verify_stage(0) if index == 1 else None
        if train_raw is None:
            binding = SOURCE.digest(self.train_request())
            job: dict[str, Any] | None = None
        else:
            job = self.smoke_job(train_raw["artifact"]["checkpoint"], create=False)
            binding = SOURCE.digest(job)
        if intent != self.intent_body(index, binding, intent.get("started_at_utc"), intent.get("cost_before_usd")):
            raise ValueError("preflight intent 결합이 다르다")
        if (set(receipt) != RECEIPT_KEYS or receipt.get("schema") != RECEIPT_SCHEMA
                or receipt.get("stage") != stage or receipt.get("job_id") != self.job_id(index)
                or receipt.get("intent_sha256") != SOURCE.sha256(paths["intent"])
                or receipt.get("raw_sha256") != SOURCE.sha256(paths["raw"])
                or receipt.get("started_at_utc") != intent["started_at_utc"]
                or receipt.get("cost_before_usd") != intent["cost_before_usd"]
                or not isinstance(receipt.get("recovered_from_cache"), bool)):
            raise ValueError("preflight receipt SHA 결합이 다르다")
        self.verify_window(index, receipt)
        if train_raw is not None:
            train_receipt = SOURCE.read(self.stage_folder(0) / "receipt.json")
            if SOURCE._stamp(train_receipt["finished_at_utc"]) > SOURCE._stamp(receipt["started_at_utc"]):
                raise ValueError("preflight smoke가 20회 학습 완료 전에 시작됐다")
        if job is None:
            expected_keys = {"schema", "stage", "job_id", "request", "artifact", "dryrun", "freshness"}
            if (set(raw) != expected_keys or raw.get("schema") != RAW_SCHEMA or raw.get("stage") != stage
                    or raw.get("job_id") != self.job_id(0) or raw.get("request") != self.train_request()
                    or raw.get("freshness") != self.freshness_reference()
                    or self.verify_train_artifact(raw.get("artifact")) != raw.get("dryrun")):
                raise ValueError("preflight 학습 raw가 원문과 다르다")
        else:
            expected_keys = {"schema", "stage", "job_id", "job_binding_sha256", "checkpoint", "smoke", "references"}
            if (set(raw) != expected_keys or raw.get("schema") != RAW_SCHEMA or raw.get("stage") != stage
                    or raw.get("job_id") != self.job_id(1) or raw.get("job_binding_sha256") != binding
                    or train_raw is None or raw.get("checkpoint") != train_raw["artifact"]["checkpoint"]):
                raise ValueError("preflight smoke raw 결합이 다르다")
            self.verify_smoke(job, raw.get("smoke"), raw.get("references"))
        started, finished = self.evidence_window(index, raw)
        if (not SOURCE._stamp(intent["started_at_utc"]) <= started <= finished
                or SOURCE._stamp(receipt["finished_at_utc"]) != finished):
            raise ValueError("preflight receipt 완료 시각이 backend 실행 종료 증거와 다르다")
        return raw

    def verify_window(self, index: int, receipt: Mapping[str, Any]) -> None:
        limit = self.preflight["limits"][STAGES[index]]
        started, finished = SOURCE._stamp(receipt["started_at_utc"]), SOURCE._stamp(receipt["finished_at_utc"])
        elapsed = (finished - started).total_seconds()
        before: Any = receipt.get("cost_before_usd")
        after: Any = receipt.get("cost_after_usd")
        if (not _number(before) or not _number(after) or not _number(receipt.get("elapsed_wall_s"))
                or abs(float(receipt["elapsed_wall_s"]) - elapsed) > TOLERANCE
                or not self.compute_started <= SOURCE._stamp(self.resource["issued_at_utc"]) <= started
                or not started <= finished <= min(self.expires, self.deadline)
                or elapsed > float(limit["max_wall_s"])
                or elapsed * self.quote / 3600.0 > float(limit["max_compute_usd"]) + TOLERANCE
                or abs(float(before) - self.lifecycle(started)) > TOLERANCE
                or abs(float(after) - self.lifecycle(finished)) > TOLERANCE
                or float(after) > self.cap + TOLERANCE):
            raise ValueError("preflight 단계 시간·비용·lease·deadline 결합이 다르다")

    def completion_body(self) -> dict[str, Any]:
        train_raw = self.verify_stage(0)
        self.verify_stage(1)
        smoke_receipt = SOURCE.read(self.stage_folder(1) / "receipt.json")
        stages = []
        for index, stage in enumerate(STAGES):
            folder = self.stage_folder(index)
            stages.append({"stage": stage, "job_id": self.job_id(index), "journal": {
                name: self.reference(folder / f"{name}.json", f"preflight {name}")
                for name in ("intent", "raw", "receipt")
            }})
        return {
            "schema": COMPLETION_SCHEMA, "protocol_id": self.preflight["protocol_id"], "scope": SCOPE,
            "model_use_ready": False, "source_anchor": self.anchor,
            "preflight_protocol": {"path": self.preflight_path.relative_to(self.repo).as_posix(),
                                   "sha256_lf": self.preflight_sha},
            "source_protocol": self.preflight["source_protocol"], "render_protocol": self.preflight["render_protocol"],
            "resource_sha256": self.resource_sha, "stages": stages,
            "train_checkpoint_sha256": train_raw["artifact"]["checkpoint"]["sha256"],
            "finished_at_utc": smoke_receipt["finished_at_utc"], "cost_after_usd": smoke_receipt["cost_after_usd"],
        }

    def completion(self, *, create: bool) -> dict[str, str]:
        path = self.folder / "completion.json"
        body = self.completion_body()
        if path.exists():
            if SOURCE.read(SOURCE.VISION._single_regular(path, "preflight completion")) != body:
                raise ValueError("preflight completion이 완료 원장과 다르다")
        elif create:
            SOURCE.write_new(path, body)
        else:
            raise ValueError("preflight completion이 아직 없다")
        return self.reference(path, "preflight completion")

    def gate_error(self, now: datetime) -> str | None:
        """현재 시각의 resource gate. 실패해도 완료 원장 조회·마감은 막지 않는다."""
        try:
            _resource_gate(self.source, self.source_sha, self.resource, now.astimezone(SOURCE.UTC_TZ))
        except ValueError as error:
            return str(error)
        return None

    def readiness(self, now: datetime) -> dict[str, Any]:
        """source 배치 시작 가능 여부. 모델 사용 준비는 항상 false다."""
        current = now.astimezone(SOURCE.UTC_TZ)
        gate_error = self.gate_error(current)
        reasons = []
        if [self.stage_state(index) for index in range(len(STAGES))] != ["complete", "complete"]:
            reasons.append("preflight_incomplete")
        if gate_error is not None:
            reasons.append("resource_gate_failed")
        if current + timedelta(seconds=SOURCE_FIRST_ACTION_RESERVE_S) > min(self.expires, self.deadline):
            reasons.append("source_first_action_window_insufficient")
        cost = self.lifecycle(current)
        if cost + self.source_reserve() > self.cap + TOLERANCE:
            reasons.append("source_budget_insufficient")
        return {"ready_for_source_batch": not reasons, "model_use_ready": False, "readiness_reasons": reasons,
                "resource_gate_error": gate_error, "observed_at_utc": current.isoformat(),
                "lifecycle_cost_observed_usd": cost, "source_reserve_usd": self.source_reserve(),
                "whole_batch_budget_usd": self.cap, "source_batch_started": self.source_started()}

    def status(self, now: datetime) -> dict[str, Any]:
        stages = []
        states = []
        for index, stage in enumerate(STAGES):
            state = self.stage_state(index)
            states.append(state)
            entry: dict[str, Any] = {"stage": stage, "state": state, "job_id": self.job_id(index)}
            folder = self.stage_folder(index)
            if state == "complete":
                self.verify_stage(index)
                entry["journal"] = {name: self.reference(folder / f"{name}.json", f"preflight {name}")
                                    for name in ("intent", "raw", "receipt")}
            elif state in {"failed", "invalid"}:
                name = "failure" if state == "failed" else "invalid"
                entry["record"] = self.reference(folder / f"{name}.json", f"preflight {name}")
            stages.append(entry)
        completion = None
        if states == ["complete", "complete"]:
            completion = self.completion(create=False) if (self.folder / "completion.json").exists() else None
            status = "preflight_complete" if completion is not None else "HOLD"
        elif any(state in {"failed", "invalid", "pending"} for state in states):
            status = "HOLD"
        else:
            status = "train_complete" if states[0] == "complete" else "not_started"
        result = {"status": status, "protocol_id": self.preflight["protocol_id"], "scope": SCOPE,
                  "stages": stages, "completion": completion, **self.readiness(now)}
        if status == "HOLD":
            result["reason"] = ("preflight_completion_missing" if states == ["complete", "complete"]
                                else "preflight_stage_unresolved")
        return result

    # backend 없이 쓰는 마감 기록
    @staticmethod
    def hold(reason: str, **extra: Any) -> dict[str, Any]:
        return {"status": "HOLD", "reason": reason, "model_use_ready": False,
                "ready_for_source_batch": False, **extra}

    def record(self, index: int, kind: str, reason: str, message: str, now: datetime,
               **extra: Any) -> dict[str, Any]:
        folder = self.stage_folder(index)
        observed = now.astimezone(SOURCE.UTC_TZ)
        body = {"schema": FAILURE_SCHEMA if kind == "failure" else INVALID_SCHEMA, "stage": STAGES[index],
                "job_id": self.job_id(index), "reason": reason, "message": message,
                "intent_sha256": SOURCE.sha256(folder / "intent.json"),
                "observed_at_utc": observed.isoformat(), "observed_cost_usd": self.lifecycle(observed),
                **extra}
        SOURCE.write_new(folder / f"{kind}.json", body)
        return self.hold("preflight_stage_failed" if kind == "failure" else "preflight_stage_invalid",
                         stage=STAGES[index], detail=reason)

    def close_train(self, artifact: Any, intent: Mapping[str, Any], now: datetime, *,
                    recovered: bool) -> dict[str, Any]:
        try:
            dryrun = self.verify_train_artifact(artifact)
            freshness = self.freshness_reference()
        except (OSError, TypeError, ValueError) as error:
            return self.record(0, "invalid", "train_evidence_invalid", str(error), now)
        raw = {"schema": RAW_SCHEMA, "stage": TRAIN_STAGE, "job_id": self.job_id(0),
               "request": self.train_request(), "artifact": artifact, "dryrun": dryrun, "freshness": freshness}
        return self.close(0, raw, intent, now, recovered=recovered)

    def close_smoke(self, job: Mapping[str, Any], smoke: Any, references: Any, intent: Mapping[str, Any],
                    now: datetime, *, recovered: bool) -> dict[str, Any]:
        try:
            self.verify_smoke(job, smoke, references)
        except (OSError, TypeError, ValueError) as error:
            return self.record(1, "invalid", "smoke_evidence_invalid", str(error), now)
        train_raw = self.verify_stage(0)
        raw = {"schema": RAW_SCHEMA, "stage": SMOKE_STAGE, "job_id": self.job_id(1),
               "job_binding_sha256": SOURCE.digest(dict(job)),
               "checkpoint": train_raw["artifact"]["checkpoint"], "smoke": smoke, "references": references}
        return self.close(1, raw, intent, now, recovered=recovered)

    def close(self, index: int, raw: Mapping[str, Any], intent: Mapping[str, Any], now: datetime, *,
              recovered: bool) -> dict[str, Any]:
        """receipt 완료 시각은 backend 종료 증거로 정하고, 현재 시각·현재 lease로 판정하지 않는다."""
        folder = self.stage_folder(index)
        raw_path = folder / "raw.json"
        if raw_path.exists():
            if SOURCE.read(raw_path) != dict(raw):
                raise ValueError("기존 preflight raw가 검증된 원문과 다르다")
        else:
            SOURCE.write_new(raw_path, raw)
        try:
            evidence_started, finished = self.evidence_window(index, raw)
        except (KeyError, OSError, TypeError, ValueError) as error:
            return self.record(index, "invalid", "execution_end_evidence_invalid", str(error), now,
                               raw_sha256=SOURCE.sha256(raw_path))
        started = SOURCE._stamp(intent["started_at_utc"])
        elapsed = (finished - started).total_seconds()
        receipt = {
            "schema": RECEIPT_SCHEMA, "stage": STAGES[index], "job_id": self.job_id(index),
            "intent_sha256": SOURCE.sha256(folder / "intent.json"), "raw_sha256": SOURCE.sha256(raw_path),
            "started_at_utc": intent["started_at_utc"], "finished_at_utc": finished.isoformat(),
            "elapsed_wall_s": elapsed, "cost_before_usd": intent["cost_before_usd"],
            "cost_after_usd": self.lifecycle(finished) if finished >= self.compute_started else -1.0,
            "recovered_from_cache": recovered,
        }
        try:
            if not started <= evidence_started <= finished:
                raise ValueError("backend 실행 시각이 intent 시작 전이거나 순서가 다르다")
            self.verify_window(index, receipt)
        except ValueError as error:
            return self.record(index, "invalid", "preflight_stage_limit_exceeded", str(error), now,
                               raw_sha256=receipt["raw_sha256"], finished_at_utc=receipt["finished_at_utc"],
                               cost_after_usd=receipt["cost_after_usd"])
        SOURCE.write_new(folder / "receipt.json", receipt)
        self.verify_stage(index)
        if index == 0:
            return {"status": "advanced", "stage": TRAIN_STAGE, "completed_stages": 1, "remaining_stages": 1,
                    "recovered_from_cache": recovered, "model_use_ready": False, "ready_for_source_batch": False}
        return self.complete_result(now, recovered=recovered)

    def complete_result(self, now: datetime, *, recovered: bool = False) -> dict[str, Any]:
        completion = self.completion(create=True)
        return {"status": "preflight_complete", "completed_stages": 2, "remaining_stages": 0,
                "recovered_from_cache": recovered, "scope": SCOPE, "completion": completion,
                **self.readiness(now)}

    def recover_pending(self, index: int, now: datetime) -> dict[str, Any]:
        """intent만 있는 단계를 backend·GPU·provider 호출 없이 원문으로만 마감하거나 보류한다."""
        folder = self.stage_folder(index)
        intent = SOURCE.read(folder / "intent.json")
        if index == 0:
            job: dict[str, Any] | None = None
            binding = SOURCE.digest(self.train_request())
            execution_path = Path(self.train_request()["stage"]) / "execution.json"
        else:
            train_raw = self.verify_stage(0)
            inputs = self.smoke_inputs()
            if not (inputs / "protocol.json").is_file() or not (inputs / "admission.json").is_file():
                raise ValueError("preflight smoke intent에 결합된 input 원문이 없다")
            job = self.smoke_job(train_raw["artifact"]["checkpoint"], create=False)
            binding = SOURCE.digest(job)
            execution_path = self.smoke_stage() / "execution.json"
        if intent != self.intent_body(index, binding, intent.get("started_at_utc"), intent.get("cost_before_usd")):
            raise ValueError("pending preflight intent 결합이 다르다")
        if not execution_path.is_file():
            waited = (now.astimezone(SOURCE.UTC_TZ) - SOURCE._stamp(intent["started_at_utc"])).total_seconds()
            if 0 <= waited <= float(self.preflight["limits"][STAGES[index]]["max_wall_s"]):
                return self.hold("preflight_stage_pending", stage=STAGES[index])
            return self.record(index, "invalid", "pending_unresolved_after_window",
                               "backend 실행 종료 증거 없이 단계 시간 한도가 지났다", now)
        execution = SOURCE.read(execution_path)
        if execution.get("completed") is not True:
            return self.record(index, "failure", str(execution.get("error_type", "InvalidExecutionEvidence")),
                               "backend execution 원문이 완료 상태가 아니다", now)
        if job is None:
            try:
                artifact = self.artifact_from_stage()
            except (OSError, TypeError, ValueError) as error:
                return self.record(0, "invalid", "train_cache_invalid", str(error), now)
            return self.close_train(artifact, intent, now, recovered=True)
        try:
            references = _capture_references_offline(self.repo, self.runtime, self.smoke["job_id"])
            smoke = self.smoke_payload(references)
        except (KeyError, OSError, TypeError, ValueError) as error:
            return self.record(1, "invalid", "smoke_cache_invalid", str(error), now)
        return self.close_smoke(job, smoke, references, intent, now, recovered=True)

    def settle(self, now: datetime) -> dict[str, Any] | None:
        """새 실행 전 offline 분기. 실패·pending·완료를 처리하고, 새 단계가 필요할 때만 None."""
        states = [self.stage_state(index) for index in range(len(STAGES))]
        for index, state in enumerate(states):
            if state in {"failed", "invalid"}:
                return self.hold(f"preflight_stage_{state}", stage=STAGES[index])
        for index, state in enumerate(states):
            if state == "pending":
                return self.recover_pending(index, now)
            if state == "complete":
                self.verify_stage(index)
        if states == ["complete", "complete"]:
            return self.complete_result(now)
        return None


class PreflightBatch(RENDER.RenderBatch):
    """RenderBatch의 source/resource/backend 결합을 재사용하고 preflight 두 단계만 실행한다."""

    def __init__(
        self, repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
        render_protocol_path: Path, render_protocol_sha256_lf: str, preflight_protocol_path: Path,
        preflight_protocol_sha256_lf: str, source_anchor: str, resource_path: Path, resource_sha256: str,
        lab: Path, python: Path, *,
        backend_factory: Callable[..., Any] = RENDER.RENDER.AttestedVisionBrevBackend.from_runtime,
        seed_guard: Callable[..., dict[str, Any]] = SOURCE.SEEDS.assert_fresh_seeds,
        now: Callable[[], datetime] = lambda: datetime.now(SOURCE.UTC_TZ),
    ):
        path = Path(preflight_protocol_path).resolve()
        if (not path.is_file() or path.is_symlink() or not isinstance(preflight_protocol_sha256_lf, str)
                or SOURCE.HEX64.fullmatch(preflight_protocol_sha256_lf) is None
                or SOURCE.sha256(path, lf=True) != preflight_protocol_sha256_lf):
            raise ValueError("preflight protocol 실제 LF SHA가 다르다")
        self.preflight_protocol_path, self.preflight_sha = path, preflight_protocol_sha256_lf
        super().__init__(
            repo, source_protocol_path, source_protocol_sha256_lf, render_protocol_path,
            render_protocol_sha256_lf, source_anchor, resource_path, resource_sha256, lab, python,
            backend_factory=backend_factory, seed_guard=seed_guard, now=now,
        )
        source_ref = {"path": self.protocol_path.relative_to(self.repo).as_posix(), "sha256_lf": self.protocol_sha}
        render_ref = {"path": self.render_protocol_path.relative_to(self.repo).as_posix(),
                      "sha256_lf": self.render_protocol_sha}
        self.preflight = validate_protocol(SOURCE.read(path), SOURCE.read(self.protocol_path), source_ref,
                                           self.render_protocol, render_ref)
        _verify_controller_binding(self.repo, self.anchor, path, self.preflight_sha,
                                   self.preflight["controller_sources_lf"])
        self.ledger = PreflightLedger(
            self.repo, self.runtime, self.protocol, self.protocol_path, self.protocol_sha,
            self.render_protocol_path, self.render_protocol_sha, self.preflight, path, self.preflight_sha,
            self.anchor, self.resource, self.resource_sha,
        )

    def _verify_bound_inputs(self) -> None:
        super()._verify_bound_inputs()
        if SOURCE.sha256(self.preflight_protocol_path, lf=True) != self.preflight_sha:
            raise ValueError("실행 중 preflight protocol LF SHA가 바뀌었다")

    def _verify_source(self) -> None:
        super()._verify_source()
        if hasattr(self, "preflight"):
            _verify_controller_binding(self.repo, self.anchor, self.preflight_protocol_path,
                                       self.preflight_sha, self.preflight["controller_sources_lf"])

    def _advance_admitted(self) -> dict[str, Any]:
        raise RuntimeError("preflight controller는 source/render action을 실행하지 않는다")

    def verify(self) -> dict[str, Any]:
        result = super().verify()
        result.update({"preflight_protocol_sha256_lf": self.preflight_sha, "preflight_stages": list(STAGES),
                       "scope": SCOPE, "model_use_ready": False})
        return result

    def advance(self) -> dict[str, Any]:
        """한 호출에 preflight 한 단계만 진행한다. pending·완료 처리는 resource gate·backend 전에 끝낸다."""
        self._verify_bound_inputs()
        self._verify_source()
        settled = self.ledger.settle(self.now())
        if settled is not None:
            return settled
        self._verify_resource(SOURCE.read(self.resource_path))
        if self.ledger.source_started():
            return self.ledger.hold("source_batch_already_started")
        backend = self._backend()
        with backend.action_admission():
            states = [self.ledger.stage_state(index) for index in range(len(STAGES))]
            index = 0 if states[0] != "complete" else 1
            if states[index] != "absent":
                return self.ledger.hold("preflight_state_changed", stage=STAGES[index])
            return self._start_stage(backend, index)

    def _start_stage(self, backend: Any, index: int) -> dict[str, Any]:
        current = self.now().astimezone(SOURCE.UTC_TZ)
        limit = self.preflight["limits"][STAGES[index]]
        if current + timedelta(seconds=limit["max_wall_s"]) > min(self.ledger.expires, self.ledger.deadline):
            return self.ledger.hold("preflight_window_insufficient", stage=STAGES[index])
        cost = self.ledger.lifecycle(current)
        if cost + self.ledger.remaining_reserve(index) > self.ledger.cap + TOLERANCE:
            return self.ledger.hold("preflight_budget_insufficient", stage=STAGES[index])
        self._verify_resource(SOURCE.read(self.resource_path))
        if index == 0:
            return self._start_train(backend, current, cost)
        return self._start_smoke(backend, current, cost)

    def _start_train(self, backend: Any, current: datetime, cost: float) -> dict[str, Any]:
        freshness_path = self.ledger.folder / "freshness.json"
        if freshness_path.exists():
            return self.ledger.hold("preflight_freshness_orphaned", stage=TRAIN_STAGE)
        seed = self.preflight["training"]["seed"]
        freshness = self.seed_guard([seed], self.lab, repo=self.repo)
        if freshness.get("status") != "fresh" or freshness.get("seeds") != [seed]:
            return self.ledger.hold("preflight_seed_not_fresh", stage=TRAIN_STAGE)
        SOURCE.write_new(freshness_path, freshness)
        request = self.ledger.train_request()
        intent = self.ledger.intent_body(0, SOURCE.digest(request), current.isoformat(), cost)
        SOURCE.write_new(self.ledger.stage_folder(0) / "intent.json", intent)
        try:
            artifact = backend.recovery.train(Path(request["stage"]), "NONE", seed, request["run_name"],
                                              iterations=PREFLIGHT_ITERATIONS)
        except Exception as error:  # noqa: BLE001 - 실패도 write-once 기록으로 닫는다.
            return self.ledger.record(0, "failure", _failure_kind(error), str(error), self.now())
        return self.ledger.close_train(artifact, intent, self.now(), recovered=False)

    def _start_smoke(self, backend: Any, current: datetime, cost: float) -> dict[str, Any]:
        train_raw = self.ledger.verify_stage(0)
        try:
            job = self.ledger.smoke_job(train_raw["artifact"]["checkpoint"], create=True, now=current)
        except FileExistsError:
            return self.ledger.hold("preflight_smoke_input_orphaned", stage=SMOKE_STAGE)
        intent = self.ledger.intent_body(1, SOURCE.digest(job), current.isoformat(), cost)
        SOURCE.write_new(self.ledger.stage_folder(1) / "intent.json", intent)
        try:
            smoke = backend.smoke(job)
            if smoke.get("status") == "HOLD":
                return self.ledger.hold("preflight_stage_pending", stage=SMOKE_STAGE,
                                        backend_reason=smoke.get("reason", "render_backend_hold"))
            references = backend.capture_references(job)
        except Exception as error:  # noqa: BLE001 - 실패도 write-once 기록으로 닫는다.
            return self.ledger.record(1, "failure", _failure_kind(error), str(error), self.now())
        return self.ledger.close_smoke(job, smoke, references, intent, self.now(), recovered=False)


def _offline_ledger(
    repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
    render_protocol_path: Path, render_protocol_sha256_lf: str, preflight_protocol_path: Path,
    preflight_protocol_sha256_lf: str, source_anchor: str, resource_path: Path, resource_sha256: str,
) -> PreflightLedger:
    """backend·GPU·provider 없이 protocol·anchor·resource 원문 결합만 확인해 원장을 연다."""
    repository = Path(repo).resolve(strict=True)
    paths = {name: Path(path).resolve(strict=True) for name, path in (
        ("source", source_protocol_path), ("render", render_protocol_path), ("preflight", preflight_protocol_path))}
    for name, expected in (("source", source_protocol_sha256_lf), ("render", render_protocol_sha256_lf),
                           ("preflight", preflight_protocol_sha256_lf)):
        if paths[name].is_symlink() or SOURCE.sha256(paths[name], lf=True) != expected:
            raise ValueError(f"offline {name} protocol LF SHA가 다르다")
    source = SOURCE.validate_protocol(SOURCE.read(paths["source"]))
    source_ref = {"path": paths["source"].relative_to(repository).as_posix(), "sha256_lf": source_protocol_sha256_lf}
    render_ref = {"path": paths["render"].relative_to(repository).as_posix(), "sha256_lf": render_protocol_sha256_lf}
    render = RENDER.validate_protocol(SOURCE.read(paths["render"]), source, source_ref)
    preflight = validate_protocol(SOURCE.read(paths["preflight"]), source, source_ref, render, render_ref)
    RENDER._verify_offline_binding(repository, paths["source"], source_protocol_sha256_lf, paths["render"],
                                   render_protocol_sha256_lf, source_anchor, source)
    _verify_controller_binding(repository, source_anchor, paths["preflight"], preflight_protocol_sha256_lf,
                               preflight["controller_sources_lf"])
    proof = Path(resource_path).resolve(strict=True)
    if proof.is_symlink() or proof.stat().st_nlink != 1 or SOURCE.sha256(proof) != resource_sha256:
        raise ValueError("offline resource proof 원본 SHA가 다르다")
    resource = SOURCE.read(proof)
    _resource_gate(source, source_protocol_sha256_lf, resource, SOURCE._stamp(resource.get("issued_at_utc")))
    runtime = repository / ".triage-runtime/vision-brev" / source["protocol_id"]
    return PreflightLedger(
        repository, runtime, source, paths["source"], source_protocol_sha256_lf, paths["render"],
        render_protocol_sha256_lf, preflight, paths["preflight"], preflight_protocol_sha256_lf,
        source_anchor, resource, resource_sha256,
    )


def inspect_preflight(
    repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
    render_protocol_path: Path, render_protocol_sha256_lf: str, preflight_protocol_path: Path,
    preflight_protocol_sha256_lf: str, source_anchor: str, resource_path: Path, resource_sha256: str,
    *, now: datetime | None = None,
) -> dict[str, Any]:
    """backend·GPU·provider 없이 완료 자료를 조회한다. lease 만료 뒤에도 원장은 읽는다."""
    ledger = _offline_ledger(repo, source_protocol_path, source_protocol_sha256_lf, render_protocol_path,
                             render_protocol_sha256_lf, preflight_protocol_path, preflight_protocol_sha256_lf,
                             source_anchor, resource_path, resource_sha256)
    return ledger.status((now or datetime.now(SOURCE.UTC_TZ)).astimezone(SOURCE.UTC_TZ))


def settle_preflight(
    repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
    render_protocol_path: Path, render_protocol_sha256_lf: str, preflight_protocol_path: Path,
    preflight_protocol_sha256_lf: str, source_anchor: str, resource_path: Path, resource_sha256: str,
    *, now: datetime | None = None,
) -> dict[str, Any] | None:
    """backend 생성 전 offline 분기. pending 마감·실패·완료를 처리하고 새 단계가 필요하면 None."""
    ledger = _offline_ledger(repo, source_protocol_path, source_protocol_sha256_lf, render_protocol_path,
                             render_protocol_sha256_lf, preflight_protocol_path, preflight_protocol_sha256_lf,
                             source_anchor, resource_path, resource_sha256)
    return ledger.settle((now or datetime.now(SOURCE.UTC_TZ)).astimezone(SOURCE.UTC_TZ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "advance", "status"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--render-protocol", required=True, type=Path)
    parser.add_argument("--render-protocol-sha256-lf", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight-protocol", type=Path)
    parser.add_argument("--preflight-protocol-sha256-lf")
    parser.add_argument("--source-anchor")
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve(strict=True)
    if args.command == "build":
        if args.seed is None:
            parser.error("build에는 --seed가 필요하다")
        source_path, render_path = Path(args.protocol).resolve(strict=True), Path(args.render_protocol).resolve(strict=True)
        if (SOURCE.sha256(source_path, lf=True) != args.protocol_sha256_lf
                or SOURCE.sha256(render_path, lf=True) != args.render_protocol_sha256_lf):
            raise ValueError("build 입력 protocol LF SHA가 다르다")
        result = build_protocol(
            SOURCE.read(source_path),
            {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": args.protocol_sha256_lf},
            SOURCE.read(render_path),
            {"path": render_path.relative_to(repo).as_posix(), "sha256_lf": args.render_protocol_sha256_lf},
            args.seed, controller_source_hashes(repo),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    required = (args.preflight_protocol, args.preflight_protocol_sha256_lf, args.source_anchor,
                args.resource_proof, args.resource_sha256)
    if any(value is None for value in required):
        parser.error("verify/advance/status에는 preflight protocol·source anchor·resource proof 인자가 필요하다")
    if args.command in {"verify", "advance"} and (args.isaaclab is None or args.python is None):
        parser.error("verify/advance에는 --isaaclab과 --python이 필요하다")
    offline_args = (repo, args.protocol, args.protocol_sha256_lf, args.render_protocol,
                    args.render_protocol_sha256_lf, args.preflight_protocol, args.preflight_protocol_sha256_lf,
                    args.source_anchor, args.resource_proof, args.resource_sha256)
    result: dict[str, Any] | None = None
    if args.command == "status":
        result = inspect_preflight(*offline_args)
    elif args.command == "advance":
        result = settle_preflight(*offline_args)
    if result is None:
        batch = PreflightBatch(*offline_args, args.isaaclab, args.python)
        result = batch.verify() if args.command == "verify" else batch.advance()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
