"""Writer-free native pull 녹화를 후속 영상 근거 입력으로 결합한다."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_pull_record_v2 as RUN  # pyright: ignore[reportMissingImports]
import vision_evidence_contract_v1 as EVIDENCE  # pyright: ignore[reportMissingImports]
import vision_pull_record_worker_v1 as WORKER  # pyright: ignore[reportMissingImports]
import vision_record_source_material_v1 as MATERIAL  # pyright: ignore[reportMissingImports]
import vision_record_source_readiness_v1 as LEGACY  # pyright: ignore[reportMissingImports]

SCHEMA = "vision_pull_record_source_readiness_v2"
BINDING_SCHEMA = "vision_pull_record_source_readiness_binding_v2"
INVENTORY_SCHEMA = "vision_pull_record_local_remap_inventory_v2"
TRANSLATION_SCHEMA = "vision_pull_record_spec_translation_v2"
PREPARED_SCHEMA = "vision_pull_record_contract_preparation_v2"
ROLE = {"normal": ("NONE", "healthy"), "registered-f1": ("F1", "unhealthy")}
ARTIFACT_KEYS = frozenset(
    {
        "video", "raw_trace", "encoder_receipt", "timing_receipt",
        "measurement_1", "spec_1", "measurement_2", "spec_2",
    }
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_TEXT_DENY = re.compile(
    r"(?i)(healthy|unhealthy|registered-f1|source-s\d+|checkpoint_sha256|"
    r"[a-z]:[\\/]|(?:^|\s)/(?:home|mnt|users)/|[0-9a-f]{64})"
)


def _read(path: Path) -> dict[str, Any]:
    return MATERIAL._read(path)


def _write_new(path: Path, value: Any) -> None:
    LEGACY._write_new(path, value)


def _ref(repo: Path, path: Path) -> dict[str, str]:
    return MATERIAL._ref(repo, path)


def _reference(repo: Path, value: Any, label: str) -> Path:
    return MATERIAL._reference(repo, value, label)


def _descriptor(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    return {"file": target.name, "sha256": MATERIAL._sha(target), "bytes": target.stat().st_size}


def _runtime_context(
    raw_job_path: Path,
    raw_job: Mapping[str, Any],
    checkpoint: Path,
    resource_proof: Path,
    result_path: Path,
) -> dict[str, Any]:
    """원 job 바이트를 바꾸지 않고 검증기에만 로컬 경로를 제공한다."""
    job = copy.deepcopy(dict(raw_job))
    checkpoint_row = MATERIAL._exact(
        job.get("checkpoint"), {"path", "sha256", "logical_id"}, "raw job checkpoint"
    )
    resource_row = MATERIAL._exact(
        job.get("resource_proof"), {"path", "sha256"}, "raw job resource proof"
    )
    checkpoint_row["path"] = str(checkpoint)
    resource_row["path"] = str(resource_proof)
    job["output_directory"] = str(result_path.parent)
    return {
        "job_path": raw_job_path,
        "job_sha256": MATERIAL._sha(raw_job_path),
        "job": job,
        "output_directory": result_path.parent,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": checkpoint_row["sha256"],
    }


def _execution(
    execution: Mapping[str, Any], validated: Mapping[str, Any],
    result_path: Path, resource_proof: Path, aaba_path: Path,
) -> None:
    phases = execution.get("phases")
    aaba = execution.get("aaba_receipt")
    if (
        execution.get("schema") != RUN.EXECUTION_SCHEMA
        or execution.get("completed") is not True
        or execution.get("technical_recording_complete") is not True
        or execution.get("exit_code") != 0
        or execution.get("timed_out") is not False
        or execution.get("cleanup_verified") is not True
        or execution.get("gpu_before") != []
        or execution.get("gpu_after") != []
        or execution.get("error_type") is not None
        or execution.get("error_message") is not None
        or execution.get("result_sha256") != MATERIAL._sha(result_path)
        or isinstance(execution.get("wall_s"), bool)
        or not isinstance(execution.get("wall_s"), (int, float))
        or execution.get("wall_s", 0) <= 0
        or execution.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or execution.get("resource_proof_sha256") != MATERIAL._sha(resource_proof)
        or execution.get("per_checkpoint_aaba_required") is not True
        or execution.get("full_recordings") != 1
        or execution.get("capture_approved") is not False
        or execution.get("model_ready") is not False
        or execution.get("used_for_decision") is not False
        or not isinstance(aaba, Mapping)
        or aaba.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or aaba.get("diagnostic_complete") is not True
        or aaba.get("capture_approved") is not False
        or aaba.get("used_for_decision") is not False
        or aaba.get("report") != _descriptor(aaba_path)
        or not isinstance(phases, list)
        or [row.get("name") for row in phases if isinstance(row, Mapping)] != ["aaba", "record"]
        or any(
            not isinstance(row, Mapping)
            or row.get("exit_code") != 0
            or row.get("cleanup_verified") is not True
            or row.get("gpu_before") != []
            or row.get("gpu_after") != []
            for row in phases
        )
    ):
        raise ValueError("native pull 부모 실행 증거가 다르다")


def _aaba_raw(repo: Path, aaba_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """AABA 네 arm의 NPZ와 JSON receipt를 바이트·배열 수준에서 재검증한다."""
    rows = report.get("raw_artifacts")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("AABA raw artifact가 네 건이 아니다")
    checked: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or raw.get("ordinal") != ordinal or raw.get("complete_capture") is not True:
            raise ValueError("AABA raw artifact 순서·완결성이 다르다")
        npz_descriptor = raw.get("npz")
        receipt_descriptor = raw.get("receipt")
        if not isinstance(npz_descriptor, Mapping) or not isinstance(receipt_descriptor, Mapping):
            raise TypeError("AABA raw descriptor가 없다")
        npz_path = aaba_path.parent / str(npz_descriptor.get("file", ""))
        receipt_path = aaba_path.parent / str(receipt_descriptor.get("file", ""))
        if _descriptor(npz_path) != dict(npz_descriptor) or _descriptor(receipt_path) != dict(receipt_descriptor):
            raise ValueError("AABA raw 원문 SHA가 descriptor와 다르다")
        receipt = _read(receipt_path)
        if (
            receipt.get("schema") != "vision_annotator_pull_raw_receipt_v1"
            or receipt.get("ordinal") != ordinal
            or receipt.get("complete_capture") is not True
            or receipt.get("npz") != dict(npz_descriptor)
            or receipt.get("atomicity_claimed") is not False
            or receipt.get("physics_before") != receipt.get("physics_after")
            or not isinstance(receipt.get("arrays"), Mapping)
            or not receipt["arrays"]
        ):
            raise ValueError("AABA raw receipt 내용이 다르다")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(receipt["arrays"]):
                raise ValueError("AABA NPZ 배열 목록이 receipt와 다르다")
            for name in archive.files:
                array = np.asarray(archive[name])
                metadata = receipt["arrays"].get(name)
                if (
                    not isinstance(metadata, Mapping)
                    or metadata.get("shape") != list(array.shape)
                    or metadata.get("dtype") != str(array.dtype)
                    or metadata.get("sha256")
                    != hashlib.sha256(array.copy(order="C").tobytes(order="C")).hexdigest()
                ):
                    raise ValueError("AABA NPZ 배열 metadata가 다르다")
        checked.append({"ordinal": ordinal, "npz": _ref(repo, npz_path), "receipt": _ref(repo, receipt_path)})
    return checked


def _continuation(
    continuation: Mapping[str, Any], validated: Mapping[str, Any],
    aaba_path: Path, gpu_observation: Path, resource: Mapping[str, Any],
) -> None:
    reference = continuation.get("gpu_observation")
    if (
        continuation.get("schema") != "vision_pull_record_phase_continuation_v2"
        or continuation.get("job_sha256") != validated["job_sha256"]
        or continuation.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or continuation.get("resource_proof_sha256")
        != validated["job"]["resource_proof"]["sha256"]
        or continuation.get("aaba_report") != _descriptor(aaba_path)
        or continuation.get("aaba_diagnostic_complete") is not True
        or continuation.get("capture_approved") is not False
        or continuation.get("used_for_decision") is not False
        or not isinstance(reference, Mapping)
        or reference.get("sha256") != MATERIAL._sha(gpu_observation)
    ):
        raise ValueError("AABA 단계 연속 receipt가 다르다")
    RUN.REMOTE.verify_observation_reference(
        {"path": str(gpu_observation), "sha256": reference["sha256"]},
        resource["resource_id"],
        resource["gpu_uuid"],
    )


def _training(
    repo: Path, reference: Any, seed: int, arm: str, checkpoint_sha: str,
    source_protocol: Mapping[str, Any],
) -> tuple[Path, Path]:
    path = _reference(repo, reference, "training binding")
    value = _read(path)
    allowed = {"schema", "source_seed", "arm", "checkpoint", "telemetry", "source_training_artifact"}
    fields = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or fields not in (allowed, allowed | {"training_action_raw"}):
        raise ValueError("training binding 필드가 다르다")
    checkpoint = _reference(repo, value["checkpoint"], "training checkpoint")
    _reference(repo, value["telemetry"], "training telemetry")
    artifact_path = _reference(repo, value["source_training_artifact"], "source training artifact")
    artifact = _read(artifact_path)
    if (
        value["schema"] != MATERIAL.TRAINING_SCHEMA
        or value["source_seed"] != seed
        or value["arm"] != arm
        or MATERIAL._sha(checkpoint) != checkpoint_sha
        or artifact.get("seed") != seed
        or artifact.get("arm") != arm
        or artifact.get("checkpoint", {}).get("sha256") != checkpoint_sha
        or not any(
            isinstance(row, Mapping) and row.get("seed") == seed and row.get("arm") == arm
            for row in source_protocol.get("training_plan", [])
        )
    ):
        raise ValueError("source 학습 계보가 역할·seed·checkpoint와 다르다")
    return path, artifact_path


def _fixed_eval(
    repo: Path, reference: Any, seed: int, arm: str, label: str,
    checkpoint_sha: str, source_protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = _reference(repo, reference, "fixed eval summary")
    value = _read(path)
    if value.get("schema") == "vision_source_prefix_public_result_v1":
        training = value.get("training")
        evaluations = value.get("evaluations")
        scope = value.get("scope")
        if (
            value.get("status") != "source_training_and_fixed_evaluation_complete"
            or not isinstance(training, list)
            or not isinstance(evaluations, list)
            or not isinstance(scope, Mapping)
            or scope.get("fixed_evaluation_is_behavior_label") is not True
        ):
            raise ValueError("source paired 고정 평가 원문이 다르다")
        matches = [
            row for row in training if isinstance(row, Mapping)
            and row.get("seed") == seed and row.get("arm") == arm
            and row.get("checkpoint_sha256") == checkpoint_sha
        ]
        normal = [
            row for row in training if isinstance(row, Mapping)
            and row.get("seed") == seed and row.get("arm") == "NONE"
        ]
        paired = [row for row in evaluations if isinstance(row, Mapping) and row.get("seed") == seed]
        evaluation_ref = source_protocol.get("evaluation")
        if not isinstance(evaluation_ref, Mapping) or set(evaluation_ref) != {"bands_path", "bands_sha256"}:
            raise ValueError("source protocol 고정 평가 bands 참조가 다르다")
        bands_path = MATERIAL._inside(repo, repo / str(evaluation_ref["bands_path"]))
        if MATERIAL._sha(bands_path) != evaluation_ref["bands_sha256"]:
            raise ValueError("source protocol 고정 평가 bands SHA가 다르다")
        bands_source = _read(bands_path)
        bands = bands_source.get("bands")
        verdict = paired[0].get("source_gate", {}).get("verdict", {}) if len(paired) == 1 else {}
        metrics = paired[0].get("metrics") if len(paired) == 1 else None
        if (
            len(matches) != 1 or len(normal) != 1 or len(paired) != 1
            or not isinstance(metrics, Mapping) or set(metrics) != {"F1", "NONE"}
            or verdict.get("label") != "unhealthy"
            or (label == "unhealthy" and arm != "F1")
            or (label == "healthy" and arm != "NONE")
            or not isinstance(bands, Mapping)
            or set(bands) != {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in bands.values())
            or not isinstance(bands_source.get("normal"), list) or not bands_source["normal"]
        ):
            raise ValueError("source paired actual label/reference/bands가 다르다")
        return path, {
            "run": matches[0].get("job_id"),
            "reference": normal[0].get("job_id"),
            "label": label,
            "bands": dict(bands),
        }
    bands = value.get("bands")
    runs = value.get("runs")
    if (
        value.get("kind") != "fixed_eval_summary"
        or not isinstance(bands, Mapping)
        or set(bands) != {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in bands.values())
        or not isinstance(runs, list)
    ):
        raise ValueError("고정 평가 summary/bands가 다르다")
    matches = [
        row for row in runs if isinstance(row, Mapping)
        and row.get("seed") == seed and row.get("checkpoint_sha256") == checkpoint_sha
    ]
    if len(matches) != 1 or matches[0].get("fixed_label") != label:
        raise ValueError("고정 평가 actual label이 역할·seed·checkpoint와 다르다")
    selected = matches[0]
    normal = [
        row for row in runs if isinstance(row, Mapping)
        and row.get("run") == selected.get("reference")
        and row.get("fixed_label") == "healthy"
        and row.get("is_reference") is True
    ]
    if len(normal) != 1 or not isinstance(value.get("normal"), list) or not value["normal"]:
        raise ValueError("고정 평가 정상 reference가 없다")
    return path, {"run": selected.get("run"), "reference": selected.get("reference"), "label": label, "bands": dict(bands)}


def _source_protocol(repo: Path, reference: Any) -> tuple[Path, dict[str, Any], str]:
    path = _reference(repo, reference, "source protocol")
    value = _read(path)
    digest = LEGACY._sha_lf(path)
    if (
        value.get("schema") != "vision_brev_source_protocol_v1"
        or not isinstance(value.get("protocol_id"), str)
        or not isinstance(value.get("training_plan"), list)
    ):
        raise ValueError("source protocol 원문이 다르다")
    return path, value, digest


def _build_body(repo: Path, binding_path: Path) -> dict[str, Any]:
    binding_path = MATERIAL._inside(repo, binding_path)
    binding = MATERIAL._exact(
        _read(binding_path),
        {
            "schema", "seed_group", "source_role", "source_seed", "raw_job",
            "result", "execution", "aaba_report", "aaba_continuation", "local_remap_inventory",
            "resource_bundle", "training_binding", "fixed_eval_summary", "source_protocol",
        },
        "native readiness binding",
    )
    role, seed = binding["source_role"], binding["source_seed"]
    if (
        binding["schema"] != BINDING_SCHEMA
        or not isinstance(binding["seed_group"], str) or not binding["seed_group"]
        or role not in ROLE or type(seed) is not int or seed <= 0
    ):
        raise ValueError("native readiness 역할·seed가 다르다")
    raw_job_path = _reference(repo, binding["raw_job"], "raw job")
    raw_job_before = raw_job_path.read_bytes()
    raw_job = _read(raw_job_path)
    result_path = _reference(repo, binding["result"], "native result")
    execution_path = _reference(repo, binding["execution"], "native execution")
    aaba_path = _reference(repo, binding["aaba_report"], "native AABA report")
    continuation_path = _reference(repo, binding["aaba_continuation"], "AABA continuation")
    inventory_path = _reference(repo, binding["local_remap_inventory"], "local remap inventory")
    inventory = MATERIAL._exact(
        _read(inventory_path),
        {"schema", "raw_job", "checkpoint", "result", "execution", "aaba_report", "aaba_continuation", "gpu_observation", "resource_proof", "native_manifest", "artifacts"},
        "local remap inventory",
    )
    if inventory["schema"] != INVENTORY_SCHEMA or inventory["raw_job"] != binding["raw_job"]:
        raise ValueError("local remap inventory가 raw job과 다르다")
    checkpoint = _reference(repo, inventory["checkpoint"], "local checkpoint")
    resource_proof = _reference(repo, inventory["resource_proof"], "local resource proof")
    gpu_observation = _reference(repo, inventory["gpu_observation"], "AABA GPU observation")
    manifest_path = _reference(repo, inventory["native_manifest"], "native manifest")
    stage = execution_path.parent
    if (
        raw_job.get("schema") != RUN.JOB.JOB_SCHEMA
        or raw_job.get("mode") != RUN.JOB.MODE
        or raw_job_path != stage / "job.json"
        or result_path != stage / "output" / "result.json"
        or execution_path != stage / "execution.json"
        or aaba_path != stage / "aaba-output" / RUN.AABA_REPORT_NAME
        or continuation_path != stage / RUN.AABA_CONTINUATION_NAME
        or manifest_path != result_path.parent / WORKER.MANIFEST_NAME
        or _reference(repo, inventory["result"], "inventory result") != result_path
        or _reference(repo, inventory["execution"], "inventory execution") != execution_path
        or _reference(repo, inventory["aaba_report"], "inventory AABA") != aaba_path
        or _reference(repo, inventory["aaba_continuation"], "inventory continuation") != continuation_path
        or result_path.parent.parent != execution_path.parent
        or aaba_path.parent.parent != execution_path.parent
    ):
        raise ValueError("local remap stage가 같은 실행 원문을 가리키지 않는다")
    artifacts_value = MATERIAL._exact(inventory["artifacts"], set(ARTIFACT_KEYS), "local artifacts")
    artifacts = {key: _reference(repo, artifacts_value[key], f"local artifact {key}") for key in ARTIFACT_KEYS}
    if any(result_path.parent not in path.parents for path in artifacts.values()):
        raise ValueError("local artifact가 native result output 밖의 다른 실행을 가리킨다")
    validated = _runtime_context(raw_job_path, raw_job, checkpoint, resource_proof, result_path)
    if MATERIAL._sha(checkpoint) != validated["checkpoint_sha256"]:
        raise ValueError("local checkpoint SHA가 raw job과 다르다")
    resource_paths, resource = LEGACY._resource(repo, binding["resource_bundle"], "record resource")
    validated["resource"] = resource
    terminal = RUN.V1.verify_terminal(validated)
    result = RUN.V1.verify_result(validated)
    aaba = RUN.verify_aaba(validated)
    embedded_cp = aaba.get("worker", {}).get("policy_probe", {}).get("checkpoint_sha256")
    if terminal != result or embedded_cp != validated["checkpoint_sha256"]:
        raise ValueError("native terminal 또는 AABA embedded checkpoint가 다르다")
    aaba_raw = _aaba_raw(repo, aaba_path, aaba)
    _continuation(_read(continuation_path), validated, aaba_path, gpu_observation, resource)
    worker = result.get("worker")
    descriptors = worker.get("artifacts") if isinstance(worker, Mapping) else None
    if not isinstance(descriptors, Mapping) or set(descriptors) != ARTIFACT_KEYS:
        raise ValueError("native result artifact 목록이 다르다")
    if any(dict(descriptors[key]) != _descriptor(artifacts[key]) for key in ARTIFACT_KEYS):
        raise ValueError("local artifact가 native result 원문과 다르다")
    rebuilt_manifest = WORKER._manifest(validated, artifacts)
    if _read(manifest_path) != rebuilt_manifest or worker.get("native_capture_manifest") != _descriptor(manifest_path):
        raise ValueError("625-frame native manifest가 현재 원문 재검증과 다르다")
    execution = _read(execution_path)
    _execution(execution, validated, result_path, resource_proof, aaba_path)
    if resource_paths["proof"] != resource_proof or raw_job.get("resource_proof", {}).get("sha256") != MATERIAL._sha(resource_proof):
        raise ValueError("resource proof가 raw job/local inventory와 다르다")
    source_path, source_protocol, source_sha = _source_protocol(repo, binding["source_protocol"])
    if resource.get("protocol_sha256") != source_sha:
        raise ValueError("resource proof와 source protocol이 다르다")
    arm, label = ROLE[role]
    expected_logical_id = f"source-s{seed}-{'none' if arm == 'NONE' else 'f1'}"
    source_profile_sha = source_protocol.get("record_profile_sha256")
    if (
        raw_job.get("checkpoint", {}).get("logical_id") != expected_logical_id
        or (
            source_profile_sha is not None
            and source_profile_sha != RUN.JOB.RECORD.digest(raw_job.get("profile"))
        )
    ):
        raise ValueError("raw job checkpoint 역할 또는 Play profile이 source protocol과 다르다")
    training_path, training_artifact_path = _training(
        repo, binding["training_binding"], seed, arm, validated["checkpoint_sha256"], source_protocol
    )
    fixed_path, fixed = _fixed_eval(
        repo, binding["fixed_eval_summary"], seed, arm, label,
        validated["checkpoint_sha256"], source_protocol,
    )
    if raw_job_path.read_bytes() != raw_job_before:
        raise ValueError("readiness 검증 중 raw job 원문이 바뀌었다")
    return {
        "schema": SCHEMA,
        "status": "READY",
        "outputs": 2,
        "model_use_ready": True,
        "capture_approved": False,
        "used_for_decision": False,
        "labels_inferred_from_recording": False,
        "seed_group": binding["seed_group"],
        "source_role": role,
        "source_seed": seed,
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "fixed_evaluation": fixed,
        "binding": _ref(repo, binding_path),
        "raw_job": _ref(repo, raw_job_path),
        "local_remap_inventory": _ref(repo, inventory_path),
        "result": _ref(repo, result_path),
        "execution": _ref(repo, execution_path),
        "aaba_report": _ref(repo, aaba_path),
        "aaba_continuation": _ref(repo, continuation_path),
        "aaba_raw_artifacts": aaba_raw,
        "native_manifest": _ref(repo, manifest_path),
        "source_protocol": _ref(repo, source_path),
        "source_protocol_sha256_lf": source_sha,
        "training_binding": _ref(repo, training_path),
        "source_training_artifact": _ref(repo, training_artifact_path),
        "fixed_eval_summary": _ref(repo, fixed_path),
        "record": {key: _ref(repo, path) for key, path in artifacts.items()},
        "resource_binding": {key: _ref(repo, path) for key, path in resource_paths.items()},
        "proof": {
            "frame_count": 625,
            "fps": 50,
            "native_pull_window_updates": 0,
            "clean_reserved_marker_count": 0,
            "writer_calls": {"create": 0, "attach": 0, "callback": 0, "detach": 0},
            "atomicity_claimed": False,
            "same_play_profile": copy.deepcopy(raw_job.get("profile")),
        },
    }


def build_readiness(repo_root: Path, binding_path: Path, output_path: Path) -> dict[str, Any]:
    """현재 원문이 없으면 HOLD를, 유효하면 재계산 가능한 readiness를 새로 쓴다."""
    repo = Path(repo_root).resolve(strict=True)
    output = Path(output_path).resolve()
    output.relative_to(repo)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    try:
        body = _build_body(repo, binding_path)
    except FileNotFoundError as error:
        body = {
            "schema": SCHEMA,
            "status": "HOLD",
            "outputs": 0,
            "model_use_ready": False,
            "capture_approved": False,
            "used_for_decision": False,
            "reason": "missing_inputs",
            "missing": Path(error.filename).name if error.filename else None,
        }
    _write_new(output, body)
    return body


def verify_readiness(repo_root: Path, readiness_path: Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    path = MATERIAL._inside(repo, readiness_path)
    value = _read(path)
    if value.get("schema") != SCHEMA:
        raise ValueError("native readiness schema가 다르다")
    if value.get("status") == "HOLD":
        if value.get("outputs") != 0 or value.get("model_use_ready") is not False:
            raise ValueError("HOLD readiness 경계가 다르다")
        return value
    binding = _reference(repo, value.get("binding"), "readiness binding")
    expected = _build_body(repo, binding)
    if value != expected:
        raise ValueError("native readiness가 현재 원문 재계산과 다르다")
    return value


def _translated_spec(
    original: Mapping[str, Any], video: Path, checkpoint: Path, measurement: Path,
) -> dict[str, Any]:
    expected = {"schema", "play_run_id", "source_lineage_id", "parent_source_lineage_id", "video", "checkpoint", "frames", "measurement", "play"}
    if set(original) != expected or original.get("schema") != EVIDENCE.PREPARE_SCHEMA:
        raise ValueError("원 V01 spec 필드가 다르다")
    result = copy.deepcopy(dict(original))
    for key, path in (("video", video), ("checkpoint", checkpoint), ("measurement", measurement)):
        row = result.get(key)
        if not isinstance(row, dict) or set(row) != {"path", "logical_id", "play_run_id"}:
            raise ValueError(f"원 V01 spec {key}가 다르다")
        row["path"] = str(path.resolve(strict=True))
    video_sha = MATERIAL._sha(video)
    if any(not isinstance(row, Mapping) or row.get("video_sha256") != video_sha for row in result["frames"]):
        raise ValueError("원 V01 spec frame이 local video SHA와 다르다")
    return result


def derive_local_specs(
    repo_root: Path, readiness_path: Path, output_dir: Path
) -> dict[str, Any]:
    """원 spec의 경로 세 곳만 로컬 strong-retrieval 원문으로 번역한다."""
    repo = Path(repo_root).resolve(strict=True)
    ready = verify_readiness(repo, readiness_path)
    if ready["status"] == "HOLD":
        return {"schema": TRANSLATION_SCHEMA, "status": "HOLD", "outputs": 0}
    output = Path(output_dir).resolve()
    output.relative_to(repo)
    output.mkdir(parents=True, exist_ok=False)
    record = ready["record"]
    video = _reference(repo, record["video"], "ready video")
    checkpoint = _reference(repo, _read(_reference(repo, ready["training_binding"], "ready training"))["checkpoint"], "ready checkpoint")
    rows = []
    for index in (1, 2):
        spec_path = _reference(repo, record[f"spec_{index}"], f"ready spec {index}")
        measurement = _reference(repo, record[f"measurement_{index}"], f"ready measurement {index}")
        translated = _translated_spec(_read(spec_path), video, checkpoint, measurement)
        target = output / f"window_{index}_local_spec.json"
        _write_new(target, translated)
        rows.append({
            "window": index,
            "original_spec": _ref(repo, spec_path),
            "derived_spec": _ref(repo, target),
            "replacements": {
                "video": _ref(repo, video),
                "checkpoint": _ref(repo, checkpoint),
                "measurement": _ref(repo, measurement),
            },
            "non_path_content_preserved": True,
        })
    inventory = {
        "schema": TRANSLATION_SCHEMA,
        "status": "READY",
        "outputs": 2,
        "readiness": _ref(repo, Path(readiness_path)),
        "path_fields_changed_only": ["video.path", "checkpoint.path", "measurement.path"],
        "windows": rows,
    }
    _write_new(output / "translation_inventory.json", inventory)
    return inventory


def _assert_public_view_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"source_role", "fixed_eval_label", "source_seed", "checkpoint_sha256", "filepath", "local_debug"}:
                raise ValueError("모델 입력에 비공개 판단 단서 key가 있다")
            _assert_public_view_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public_view_safe(child)
    elif isinstance(value, str) and not value.startswith("data:image/jpeg;base64,") and _MODEL_TEXT_DENY.search(value):
        raise ValueError("모델 입력 text에 정답·경로·checkpoint 단서가 있다")


def prepare_contracts(
    repo_root: Path, translation_inventory: Path, output_dir: Path
) -> dict[str, Any]:
    """두 local spec을 기존 V01 계약으로 준비하고 공개 모델 view를 점검한다."""
    repo = Path(repo_root).resolve(strict=True)
    inventory_path = MATERIAL._inside(repo, translation_inventory)
    inventory = _read(inventory_path)
    if inventory.get("schema") != TRANSLATION_SCHEMA or inventory.get("status") != "READY" or inventory.get("outputs") != 2:
        return {"schema": PREPARED_SCHEMA, "status": "HOLD", "outputs": 0}
    destination = Path(output_dir).resolve()
    destination.relative_to(repo)
    destination.mkdir(parents=True, exist_ok=False)
    contracts = []
    for row in inventory["windows"]:
        spec = _reference(repo, row["derived_spec"], "derived spec")
        contract = EVIDENCE.prepare(spec, destination / f"window_{row['window']}")
        view = EVIDENCE.public_model_view(contract)
        _assert_public_view_safe(view)
        contracts.append({"window": row["window"], "contract": _ref(repo, contract)})
    receipt = {
        "schema": PREPARED_SCHEMA,
        "status": "READY",
        "outputs": 2,
        "translation_inventory": _ref(repo, inventory_path),
        "contracts": contracts,
        "public_model_view_checked": True,
        "used_for_decision": False,
    }
    _write_new(destination / "preparation.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_readiness(args.repo, args.binding, args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0 if value["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
