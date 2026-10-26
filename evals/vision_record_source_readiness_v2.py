"""capture-first checkpoint smoke와 prepared record를 model readiness로 결합한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_checkpoint_smoke_v2 as SMOKE  # pyright: ignore[reportMissingImports]
import vision_record_source_readiness_v1 as V1  # pyright: ignore[reportMissingImports]

SCHEMA = "vision_record_source_readiness_v2"
BINDING_SCHEMA = "vision_record_source_readiness_binding_v2"
SOURCE = V1.SOURCE
VIEW = V1.VIEW
MATERIAL = V1.MATERIAL
RESOURCE_IDENTITY = V1.RESOURCE_IDENTITY
RESOURCE_BUNDLE_KEYS = V1.RESOURCE_BUNDLE_KEYS
RESOURCE_SHA_FIELDS = V1.RESOURCE_SHA_FIELDS
ROLE = V1.ROLE
_resource = V1._resource
_sha_lf = V1._sha_lf
_source_protocol = V1._source_protocol
_write_new = V1._write_new


def _smoke(
    repo: Path, result_ref: Any, execution_ref: Any
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """새 v2 terminal만 검증하고 v1 결과나 실패 atom은 승격하지 않는다."""
    result_path = MATERIAL._reference(repo, result_ref, "smoke result")
    execution_path = MATERIAL._reference(repo, execution_ref, "smoke execution")
    stage, output = result_path.parent.parent, result_path.parent
    if execution_path != stage / "execution.json":
        raise ValueError("smoke execution/result stage가 다르다")
    job_path = MATERIAL._inside(repo, stage / "job.json")
    job = MATERIAL._read(job_path)
    checkpoint = MATERIAL._exact(
        job.get("checkpoint"), {"path", "sha256", "logical_id"}, "smoke checkpoint"
    )
    validated = {
        "job_path": job_path,
        "job_sha256": MATERIAL._sha(job_path),
        "job": job,
        "output_directory": output,
        "checkpoint_path": Path(checkpoint["path"]),
        "checkpoint_sha256": checkpoint["sha256"],
    }
    result = SMOKE.verify_terminal(validated)
    execution = MATERIAL._read(execution_path)
    if (
        execution.get("schema") != SMOKE.EXECUTION_SCHEMA
        or execution.get("completed") is not True
        or execution.get("cleanup_verified") is not True
        or execution.get("gpu_before") != []
        or execution.get("gpu_after") != []
        or execution.get("failure_atom") is not None
        or execution.get("result_sha256") != MATERIAL._sha(result_path)
        or execution.get("checkpoint_sha256") != checkpoint["sha256"]
        or execution.get("resource_proof_sha256")
        != job.get("resource_proof", {}).get("sha256")
        or execution.get("resource_freshness_validated_at_start") is not True
    ):
        raise ValueError("checkpoint smoke v2 부모 실행 증거가 다르다")
    return job, result, execution_path


def _build_body(
    repo: Path,
    binding_path: Path,
    *,
    expected_source_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binding_path = MATERIAL._inside(repo, binding_path)
    binding = MATERIAL._exact(
        MATERIAL._read(binding_path),
        {
            "schema",
            "seed_group",
            "source_role",
            "source_seed",
            "record_adapter",
            "smoke_result",
            "smoke_execution",
            "training_binding",
            "fixed_eval_binding",
            "snapshot",
            "record_resource",
            "smoke_resource",
            "source_protocol",
        },
        "readiness binding",
    )
    group, role, seed = (
        binding["seed_group"],
        binding["source_role"],
        binding["source_seed"],
    )
    if (
        binding["schema"] != BINDING_SCHEMA
        or group not in {"source-01", "source-02"}
        or role not in ROLE
        or type(seed) is not int
        or seed <= 0
    ):
        raise ValueError("readiness binding 역할·seed가 다르다")
    source_protocol_path, source_protocol, source_protocol_sha = _source_protocol(
        repo, binding["source_protocol"]
    )
    if expected_source_protocol is not None:
        expected_sha = expected_source_protocol.get("_verified_sha256_lf")
        if expected_sha != source_protocol_sha or {
            key: value
            for key, value in expected_source_protocol.items()
            if key != "_verified_sha256_lf"
        } != {
            key: value
            for key, value in source_protocol.items()
            if key != "_verified_sha256_lf"
        }:
            raise ValueError(
                "호출자가 고정한 source protocol과 readiness binding이 다르다"
            )
    adapter_path = MATERIAL._reference(
        repo, binding["record_adapter"], "record adapter"
    )
    adapter = MATERIAL._read(adapter_path)
    if (
        adapter.get("schema") != MATERIAL.TECHNICAL_SCHEMA
        or adapter.get("record_completed") is not True
        or adapter.get("model_comparison_ready") is not False
        or adapter.get("labels_inferred_from_recording") is not False
        or not isinstance(adapter.get("record"), dict)
    ):
        raise ValueError("record adapter가 technical-only 완료 원문이 아니다")
    record_job, record_result, record_job_path, _execution_path = (
        MATERIAL._verify_record_terminal(repo, adapter)
    )
    smoke_job, smoke_result, smoke_execution_path = _smoke(
        repo, binding["smoke_result"], binding["smoke_execution"]
    )
    record_checkpoint = record_result["checkpoint_sha256"]
    if (
        smoke_result["checkpoint_sha256"] != record_checkpoint
        or smoke_job["checkpoint"] != record_job["checkpoint"]
        or smoke_job.get("profile") != record_job.get("profile")
        or smoke_job.get("source") != record_job.get("source")
    ):
        raise ValueError("record/smoke checkpoint·profile·source가 다르다")
    record_resource_paths, record_resource = _resource(
        repo, binding["record_resource"], "record resource"
    )
    smoke_resource_paths, smoke_resource = _resource(
        repo, binding["smoke_resource"], "smoke resource"
    )
    if any(record_resource[key] != smoke_resource[key] for key in RESOURCE_IDENTITY):
        raise ValueError("record/smoke VM·GPU·lease·start·protocol identity가 다르다")
    if (
        record_job.get("resource_proof", {}).get("sha256")
        != MATERIAL._sha(record_resource_paths["proof"])
        or smoke_job.get("resource_proof", {}).get("sha256")
        != MATERIAL._sha(smoke_resource_paths["proof"])
        or smoke_resource.get("protocol_sha256") != source_protocol_sha
        or record_resource.get("protocol_sha256") != source_protocol_sha
    ):
        raise ValueError("record/smoke resource proof가 job·source protocol과 다르다")
    training_path = MATERIAL._reference(
        repo, binding["training_binding"], "training binding"
    )
    training = MATERIAL._read(training_path)
    required_training = {
        "schema",
        "source_seed",
        "arm",
        "checkpoint",
        "telemetry",
        "source_training_artifact",
    }
    if not isinstance(training, dict) or frozenset(training) not in {
        frozenset(required_training),
        frozenset(required_training | {"training_action_raw"}),
    }:
        raise ValueError("training binding 필드가 다르다")
    expected_arm, expected_label = ROLE[role]
    checkpoint_path = MATERIAL._reference(
        repo, training["checkpoint"], "training checkpoint"
    )
    MATERIAL._reference(repo, training["telemetry"], "training telemetry")
    source_training_artifact = MATERIAL._reference(
        repo, training["source_training_artifact"], "source training artifact"
    )
    training_action_raw = (
        MATERIAL._reference(
            repo, training["training_action_raw"], "training action raw"
        )
        if "training_action_raw" in training
        else None
    )
    if (
        training["schema"] != MATERIAL.TRAINING_SCHEMA
        or training["source_seed"] != seed
        or training["arm"] != expected_arm
        or MATERIAL._sha(checkpoint_path) != record_checkpoint
    ):
        raise ValueError("training binding이 역할·seed·checkpoint와 다르다")
    training_artifact = MATERIAL._read(source_training_artifact)
    if (
        training_artifact.get("seed") != seed
        or training_artifact.get("arm") != expected_arm
        or training_artifact.get("checkpoint", {}).get("sha256") != record_checkpoint
        or not any(
            row.get("seed") == seed and row.get("arm") == expected_arm
            for row in source_protocol.get("training_plan", [])
            if isinstance(row, dict)
        )
    ):
        raise ValueError(
            "source training artifact가 source protocol·checkpoint와 다르다"
        )
    fixed_path = MATERIAL._reference(
        repo, binding["fixed_eval_binding"], "fixed eval binding"
    )
    fixed = MATERIAL._exact(
        MATERIAL._read(fixed_path),
        {"schema", "checkpoint_sha256", "label", "fixed_eval_artifact"},
        "fixed eval binding",
    )
    MATERIAL._reference(repo, fixed["fixed_eval_artifact"], "fixed eval artifact")
    if (
        fixed["schema"] != MATERIAL.FIXED_EVAL_SCHEMA
        or fixed["checkpoint_sha256"] != record_checkpoint
        or fixed["label"] != expected_label
    ):
        raise ValueError("고정 평가 label/checkpoint가 역할과 다르다")
    snapshot = MATERIAL._reference(repo, binding["snapshot"], "selection snapshot")
    snapshot_body = VIEW.checked(MATERIAL._read(snapshot))
    case_binding = (
        snapshot_body.get("inputs", {})
        .get("packet", {})
        .get("input_bindings", {})
        .get("case")
    )
    expected_case_binding = {
        "checkpoint_sha256": training_artifact["checkpoint"]["sha256"],
        "telemetry_sha256": training_artifact["telemetry"]["sha256"],
    }
    if case_binding != expected_case_binding:
        raise ValueError("selection snapshot case binding이 training artifact와 다르다")
    record = adapter["record"]
    paths = {
        key: MATERIAL._reference(repo, record[key], f"record {key}")
        for key in (
            "video",
            "trace",
            "measurement_1",
            "measurement_2",
            "spec_1",
            "spec_2",
        )
    }
    body = {
        "schema": SCHEMA,
        "scope": "checkpoint_policy_record_and_smoke",
        "model_use_ready": True,
        "seed_group": group,
        "source_role": role,
        "source_seed": seed,
        "checkpoint_sha256": record_checkpoint,
        "fixed_eval_label": fixed["label"],
        "labels_inferred_from_recording": False,
        "binding": MATERIAL._ref(repo, binding_path),
        "source_protocol": MATERIAL._ref(repo, source_protocol_path),
        "source_protocol_sha256_lf": source_protocol_sha,
        "source_training_artifact": MATERIAL._ref(repo, source_training_artifact),
        "training_action_raw": (
            MATERIAL._ref(repo, training_action_raw)
            if training_action_raw is not None
            else None
        ),
        "record_job": MATERIAL._ref(repo, record_job_path),
        "training_binding": MATERIAL._ref(repo, training_path),
        "fixed_eval_binding": MATERIAL._ref(repo, fixed_path),
        "snapshot": MATERIAL._ref(repo, snapshot),
        "record": {key: MATERIAL._ref(repo, path) for key, path in paths.items()},
        "checkpoint_smoke": {
            "result": binding["smoke_result"],
            "execution": MATERIAL._ref(repo, smoke_execution_path),
            "policy_environment_steps": 1,
            "capture_phase_physics_steps": 0,
            "universal_zero_lag_claim": False,
        },
        "resource_binding": {
            "record": {
                key: MATERIAL._ref(repo, path)
                for key, path in record_resource_paths.items()
            },
            "smoke": {
                key: MATERIAL._ref(repo, path)
                for key, path in smoke_resource_paths.items()
            },
            "identity": {key: record_resource[key] for key in RESOURCE_IDENTITY},
            "raw_proofs_independently_hashed": record_resource_paths["proof"]
            != smoke_resource_paths["proof"],
        },
        "proof": {
            "trace_sha256": MATERIAL._sha(paths["trace"]),
            "checkpoint_sha256": record_checkpoint,
            "video_sha256": MATERIAL._sha(paths["video"]),
            "source_seed": seed,
            "source_role": role,
            "measurement_sha256": {
                "window-01": MATERIAL._sha(paths["measurement_1"]),
                "window-02": MATERIAL._sha(paths["measurement_2"]),
            },
        },
    }
    return body


def build_readiness(
    repo_root: Path, binding_path: Path, output_path: Path
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    output_path = Path(output_path).resolve()
    output_path.relative_to(repo)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    body = _build_body(repo, binding_path)
    _write_new(output_path, body)
    return body


def verify_model_readiness(
    repo: Path, path: Path, source_protocol: dict[str, Any]
) -> dict[str, Any]:
    readiness_path = MATERIAL._inside(repo, path)
    value = MATERIAL._read(readiness_path)
    if (
        value.get("schema") != SCHEMA
        or value.get("model_use_ready") is not True
        or value.get("labels_inferred_from_recording") is not False
        or not isinstance(value.get("proof"), dict)
    ):
        raise ValueError("first-class record source readiness가 다르다")
    binding = value.get("binding")
    binding_path = MATERIAL._reference(repo, binding, "readiness binding")
    recomputed = _build_body(
        Path(repo).resolve(strict=True),
        binding_path,
        expected_source_protocol=source_protocol,
    )
    if value != recomputed:
        raise ValueError("first-class readiness가 현재 원문 재계산과 다르다")
    return dict(value["proof"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_readiness(args.repo, args.binding, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
