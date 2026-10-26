"""네 source 정책 기록에서 checkpoint별 smoke 요구를 고정한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import run_vision_prepared_record_v1 as RUN_V1  # pyright: ignore[reportMissingImports]
import run_vision_prepared_record_v2 as RUN_V2  # pyright: ignore[reportMissingImports]

SCHEMA = "vision_record_source_smoke_plan_v1"
REGISTRATION_SCHEMA = "vision_record_source_readiness_registration_v1"
TECHNICAL_SCHEMA = "vision_record_to_capsule_ready_v1"
TRAINING_SCHEMA = "vision_record_training_binding_v1"
FIXED_EVAL_SCHEMA = "vision_record_fixed_eval_binding_v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
POLICY_ORDER = (
    ("source-01", "normal"),
    ("source-01", "registered-f1"),
    ("source-02", "normal"),
    ("source-02", "registered-f1"),
)
ROLE = {"normal": ("NONE", "healthy"), "registered-f1": ("F1", "unhealthy")}
SMOKE_ARTIFACTS = (
    "render_sync_smoke",
    "audit_rgb_1",
    "audit_rgb_2",
    "audit_rgb_3",
)
CAPTURE_REFERENCES = (
    "job",
    "result",
    "execution",
    "sdk_report",
    "resource_proof",
    "gpu_observation_before",
    "gpu_observation_after",
    *SMOKE_ARTIFACTS,
)
READINESS_RECEIPT_FIELDS = (
    "schema",
    "scope",
    "model_use_ready",
    "binding",
    "checkpoint_sha256",
    "fixed_eval_label",
    "labels_inferred_from_recording",
    "source_training_artifact",
    "training_action_raw",
    "record_job",
    "training_binding",
    "fixed_eval_binding",
    "snapshot",
    "record",
    "checkpoint_smoke",
    "resource_binding",
    "source_protocol",
    "source_protocol_sha256_lf",
    "proof",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"중복 JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 수: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} 필드가 다르다")
    return value


def _inside(repo: Path, path: Path) -> Path:
    target = Path(path).resolve(strict=True)
    target.relative_to(repo)
    if target.is_symlink() or not target.is_file():
        raise ValueError("readiness 입력은 저장소 안 regular file이어야 한다")
    return target


def _reference(repo: Path, value: Any, label: str) -> Path:
    row = _exact(value, {"path", "sha256"}, label)
    if (
        not isinstance(row["path"], str)
        or not row["path"]
        or "\\" in row["path"]
        or Path(row["path"]).is_absolute()
        or not isinstance(row["sha256"], str)
        or HEX64.fullmatch(row["sha256"]) is None
    ):
        raise ValueError(f"{label} reference가 다르다")
    path = _inside(repo, repo / row["path"])
    if _sha(path) != row["sha256"]:
        raise ValueError(f"{label} SHA가 다르다")
    return path


def _ref(repo: Path, path: Path) -> dict[str, str]:
    target = _inside(repo, path)
    return {"path": target.relative_to(repo).as_posix(), "sha256": _sha(target)}


def _verify_record_terminal(
    repo: Path, adapter: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    result_path = _reference(repo, adapter.get("record_result"), "record result")
    execution_path = _reference(
        repo, adapter.get("record_execution"), "record execution"
    )
    output = result_path.parent
    job_path = _inside(repo, output.parent / "job.json")
    if execution_path != output.parent / "execution.json":
        raise ValueError("record execution이 result와 같은 stage가 아니다")
    result, execution, job = _read(result_path), _read(execution_path), _read(job_path)
    checkpoint = _exact(
        job.get("checkpoint"), {"path", "sha256", "logical_id"}, "record checkpoint"
    )
    validated = {
        "job_path": job_path,
        "job_sha256": _sha(job_path),
        "output_directory": output,
        "checkpoint_sha256": checkpoint["sha256"],
        "job": job,
    }
    schema = execution.get("schema")
    if schema == RUN_V1.EXECUTION_SCHEMA:
        verified = RUN_V1.verify_result(validated)
    elif schema == RUN_V2.EXECUTION_SCHEMA:
        verified = RUN_V2.verify_terminal(validated)
        if execution.get("failure_atom") is not None:
            raise ValueError("실패 atom이 있는 v2 record는 smoke 대상으로 쓰지 않는다")
    else:
        raise ValueError("지원하지 않는 prepared record execution schema다")
    if (
        verified != result
        or result.get("completed") is not True
        or result.get("record_completed") is not True
        or result.get("checkpoint_sha256") != checkpoint["sha256"]
        or execution.get("completed") is not True
        or execution.get("cleanup_verified") is not True
        or execution.get("gpu_after") != []
        or execution.get("result_sha256") != _sha(result_path)
    ):
        raise ValueError("prepared record 완료·checkpoint·cleanup 결합이 다르다")
    record_refs = adapter.get("record")
    artifacts = result.get("sdk", {}).get("artifacts")
    expected_keys = {
        "video": "video",
        "trace": "raw_trace",
        "encoder_receipt": "encoder_receipt",
        "timing_receipt": "timing_receipt",
        "measurement_1": "measurement_1",
        "spec_1": "spec_1",
        "measurement_2": "measurement_2",
        "spec_2": "spec_2",
    }
    if not isinstance(record_refs, dict) or set(record_refs) != set(expected_keys):
        raise ValueError("record adapter artifact reference가 다르다")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_keys.values()):
        raise ValueError("record terminal artifact descriptor가 다르다")
    for adapter_key, artifact_key in expected_keys.items():
        path = _reference(repo, record_refs[adapter_key], f"record {adapter_key}")
        descriptor = artifacts[artifact_key]
        matches = list(output.rglob(str(descriptor.get("file", ""))))
        if (
            len(matches) != 1
            or path != matches[0].resolve()
            or descriptor.get("sha256") != _sha(path)
            or descriptor.get("bytes") != path.stat().st_size
        ):
            raise ValueError(
                f"record adapter와 terminal artifact가 다르다: {adapter_key}"
            )
    return job, result, job_path, execution_path


def _policy(
    repo: Path, row: dict[str, Any], expected: tuple[str, str]
) -> dict[str, Any]:
    row = _exact(
        row,
        {
            "seed_group",
            "source_role",
            "source_seed",
            "record_adapter",
            "training_binding",
            "fixed_eval_binding",
            "snapshot",
        },
        "source readiness policy",
    )
    group, role = expected
    if (
        (row["seed_group"], row["source_role"]) != expected
        or type(row["source_seed"]) is not int
        or row["source_seed"] <= 0
    ):
        raise ValueError("source policy 순서·seed가 등록 design과 다르다")
    adapter_path = _reference(repo, row["record_adapter"], "record adapter")
    adapter = _read(adapter_path)
    if (
        adapter.get("schema") != TECHNICAL_SCHEMA
        or adapter.get("status") != "technical_only"
        or adapter.get("record_completed") is not True
        or adapter.get("model_comparison_ready") is not False
        or adapter.get("policy_row") is not None
        or adapter.get("labels_inferred_from_recording") is not False
    ):
        raise ValueError("record adapter가 technical-only 완료 상태가 아니다")
    job, result, job_path, execution_path = _verify_record_terminal(repo, adapter)
    checkpoint_sha = result["checkpoint_sha256"]
    training_path = _reference(repo, row["training_binding"], "training binding")
    training = _read(training_path)
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
    arm, label = ROLE[role]
    training_checkpoint = _reference(
        repo, training["checkpoint"], "training checkpoint"
    )
    _reference(repo, training["telemetry"], "training telemetry")
    _reference(repo, training["source_training_artifact"], "source training artifact")
    if "training_action_raw" in training:
        _reference(repo, training["training_action_raw"], "training action raw")
    if (
        training["schema"] != TRAINING_SCHEMA
        or training["source_seed"] != row["source_seed"]
        or training["arm"] != arm
        or _sha(training_checkpoint) != checkpoint_sha
    ):
        raise ValueError("training binding의 seed·arm·checkpoint가 record와 다르다")
    fixed_path = _reference(repo, row["fixed_eval_binding"], "fixed eval binding")
    fixed = _exact(
        _read(fixed_path),
        {"schema", "checkpoint_sha256", "label", "fixed_eval_artifact"},
        "fixed eval binding",
    )
    _reference(repo, fixed["fixed_eval_artifact"], "fixed eval artifact")
    if (
        fixed["schema"] != FIXED_EVAL_SCHEMA
        or fixed["checkpoint_sha256"] != checkpoint_sha
        or fixed["label"] != label
    ):
        raise ValueError("fixed evaluation label·checkpoint가 선언 역할과 다르다")
    snapshot = _reference(repo, row["snapshot"], "selection snapshot")
    source = job.get("source")
    resource = job.get("resource_proof")
    profile = job.get("profile")
    if (
        not isinstance(source, dict)
        or not isinstance(resource, dict)
        or profile != RECORD.fixed_profile()
    ):
        raise ValueError("record job의 source·resource·profile이 smoke 결합에 부족하다")
    return {
        "policy_id": f"{group}-{role}",
        "seed_group": group,
        "source_role": role,
        "source_seed": row["source_seed"],
        "record_adapter": _ref(repo, adapter_path),
        "training_binding": _ref(repo, training_path),
        "fixed_eval_binding": _ref(repo, fixed_path),
        "snapshot": _ref(repo, snapshot),
        "checkpoint": _ref(repo, training_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "record_job": _ref(repo, job_path),
        "record_result": _ref(
            repo, _reference(repo, adapter["record_result"], "record result")
        ),
        "record_execution": _ref(repo, execution_path),
        "smoke": {
            "status": "pending",
            "job_id": f"smoke-{job['job_id']}",
            "mode": "prepared_checkpoint_smoke_only",
            "checkpoint_sha256": checkpoint_sha,
            "must_match_record_job_fields": [
                "checkpoint.path",
                "checkpoint.sha256",
                "checkpoint.logical_id",
                "source",
                "profile",
            ],
            "required_artifacts": list(SMOKE_ARTIFACTS),
            "required_capture_references": list(CAPTURE_REFERENCES),
            "required_readiness_receipt_fields": list(READINESS_RECEIPT_FIELDS),
            "fixed_evaluation_used_for_role_only": True,
            "recording_used_to_infer_role": False,
        },
    }


def build_plan(
    repo_root: Path, registration_path: Path, output_path: Path
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    registration_path = _inside(repo, registration_path)
    registration = _exact(
        _read(registration_path),
        {"schema", "policies"},
        "source readiness registration",
    )
    rows = registration["policies"]
    if (
        registration["schema"] != REGISTRATION_SCHEMA
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise ValueError("source readiness registration schema 또는 policy 수가 다르다")
    policies = [
        _policy(repo, row, expected)
        for row, expected in zip(rows, POLICY_ORDER, strict=True)
    ]
    seeds = {
        group: {row["source_seed"] for row in policies if row["seed_group"] == group}
        for group, _role in POLICY_ORDER[::2]
    }
    if (
        any(len(values) != 1 for values in seeds.values())
        or len({next(iter(values)) for values in seeds.values()}) != 2
    ):
        raise ValueError("source seed group이 두 독립 seed와 다르다")
    checkpoint_shas = [row["checkpoint_sha256"] for row in policies]
    if len(set(checkpoint_shas)) != 4:
        raise ValueError("네 source 정책은 서로 다른 checkpoint여야 한다")
    body = {
        "schema": SCHEMA,
        "status": "checkpoint_smoke_pending",
        "model_comparison_ready": False,
        "technical_prepare_smoke_reusable": False,
        "required_policy_count": 4,
        "pending_checkpoint_smoke_count": 4,
        "policies": policies,
        "labels_inferred_from_recording": False,
        "next_gate": "all_four_checkpoint_smokes_and_readiness_receipts_verified",
    }
    output = Path(output_path).resolve(strict=False)
    output.relative_to(repo)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            body, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_plan(args.repo, args.registration, args.output),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
