"""first-class record readiness를 기존 selection E2E 상태 기계에 연결한다."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import vision_record_selection_execution_v1 as RECORD_EXECUTION
import vision_record_selection_worker_v1 as RECORD_WORKER
import vision_record_source_readiness_v1 as READINESS
import vision_selection_e2e_v1 as BASE

SCHEMA = BASE.SCHEMA
API_CAP = BASE.API_CAP
TARGETS = BASE.TARGETS
APPROVAL_FIELDS = BASE.APPROVAL_FIELDS
PROBE_RECEIPT_FIELDS = BASE.PROBE_RECEIPT_FIELDS

approval_candidate = BASE.approval_candidate
commands = BASE.commands
_row = BASE._row


def _training_jobs(
    repo: Path,
    location: Mapping[str, Any],
    source: Mapping[str, Any],
    row: Mapping[str, Any],
    expected_manifest_sha: str,
) -> dict[str, str]:
    """새 readiness의 원문 ref로 source·record·corrected job을 결합한다."""
    manifest_path, manifest = BASE._artifact(
        repo, location.get("input_manifest"), "input manifest", frozen=True
    )
    if BASE._sha(manifest_path) != expected_manifest_sha:
        raise ValueError("production verifier의 input manifest SHA가 다르다")
    readiness_path, readiness = BASE._artifact(
        repo, manifest.get("model_readiness"), "model readiness"
    )
    proof = READINESS.verify_model_readiness(repo, readiness_path, dict(source))
    if (
        readiness.get("schema") != READINESS.SCHEMA
        or readiness.get("model_use_ready") is not True
        or readiness.get("labels_inferred_from_recording") is not False
        or not isinstance(proof, dict)
        or proof.get("checkpoint_sha256") != row.get("checkpoint_sha256")
        or proof.get("source_seed") != row.get("source_seed")
        or proof.get("source_role") != row.get("source_role")
    ):
        raise ValueError("first-class readiness가 selection row와 다르다")
    training_ref = readiness.get("training_action_raw")
    record_ref = readiness.get("record_job")
    if not isinstance(training_ref, Mapping) or not isinstance(record_ref, Mapping):
        raise TypeError("readiness의 training_action_raw/record_job 원문 위치가 없다")
    _, training_raw = BASE._artifact(repo, training_ref, "training action raw")
    _, record_job = BASE._artifact(repo, record_ref, "record job")
    source_job = training_raw.get("job_id")
    record_job_id = record_job.get("job_id")
    artifact = training_raw.get("artifact")
    plan = source.get("training_plan")
    if (
        training_raw.get("kind") != "train"
        or not isinstance(source_job, str)
        or not isinstance(record_job_id, str)
        or not isinstance(artifact, Mapping)
        or artifact.get("checkpoint", {}).get("sha256") != row.get("checkpoint_sha256")
        or not isinstance(plan, list)
    ):
        raise TypeError("readiness의 source/record job 또는 checkpoint 결합이 없다")
    source_matches = [
        item
        for item in plan
        if isinstance(item, dict)
        and item.get("job_id") == source_job
        and item.get("seed") == row.get("source_seed")
        and item.get("arm") == "F1"
    ]
    corrected = [
        item
        for item in plan
        if isinstance(item, dict)
        and item.get("seed") == row.get("source_seed")
        and item.get("arm") == "CORRECTED"
        and item.get("source_job_id") == source_job
    ]
    if (
        len(source_matches) != 1
        or len(corrected) != 1
        or not isinstance(corrected[0].get("job_id"), str)
    ):
        raise ValueError("등록 source/CORRECTED training job이 유일하지 않다")
    return {
        "source_training_job_id": source_job,
        "record_job_id": record_job_id,
        "corrected_job_id": corrected[0]["job_id"],
    }


def prepare_plan(
    repo: Path,
    protocol_path: Path,
    input_locations_path: Path,
    output_path: Path,
    *,
    execution_id: str,
    source_anchor: str,
    protocol_sha256: str,
    production_prepare: Callable[..., dict[str, Any]] = RECORD_WORKER.prepare,
    initialize: Callable[..., dict[str, str]] = BASE.CALL.initialize,
) -> dict[str, Any]:
    """first-class worker 검증 뒤 기존 결정·교정 상태 기계용 계획을 봉인한다."""
    repository = Path(repo).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    locations_path = Path(input_locations_path).resolve(strict=True)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    prepared = production_prepare(
        protocol_path, protocol_sha256, source_anchor, locations_path, root=repository
    )
    if not isinstance(prepared, dict) or set(prepared) != {"protocol", "source", "inputs", "plan"}:
        raise ValueError("first-class production worker 준비 결과 형식이 다르다")
    protocol = prepared["protocol"]
    if not isinstance(protocol, dict):
        raise TypeError("production protocol이 객체가 아니다")
    comparison_repo, comparison_stage, comparison_manifest = BASE.CALL._plan(prepared["plan"])
    if (
        comparison_repo != repository
        or comparison_manifest.get("phase") != "comparison"
        or comparison_manifest.get("execution_id") != protocol.get("execution_id")
    ):
        raise ValueError("comparison execution/root 결합이 다르다")
    comparison_output = comparison_stage / "comparison-output"
    result_paths = (
        comparison_output / "choice_results.json",
        comparison_output / "comparison_summary.json",
    )
    if tuple(comparison_stage.glob("decisions/*/terminal.json")) or any(
        path.exists() or path.is_symlink() for path in result_paths
    ):
        raise ValueError("comparison 결과를 본 뒤 e2e 입력을 고정할 수 없다")
    targets = BASE._target_rows(protocol)
    verified_inputs = prepared["inputs"]
    input_rows = verified_inputs.get("rows") if isinstance(verified_inputs, dict) else None
    if not isinstance(input_rows, list) or len(input_rows) != 24:
        raise ValueError("production verifier의 input 24행이 필요하다")
    by_call = {item.get("call_id"): item for item in input_rows if isinstance(item, dict)}
    if len(by_call) != 24:
        raise ValueError("production verifier call_id가 중복됐다")
    locations = BASE._read(locations_path)
    location_rows = locations.get("rows")
    if (
        BASE._sha(locations_path) != verified_inputs.get("input_locations_sha256")
        or not isinstance(location_rows, list)
        or len(location_rows) != 24
    ):
        raise ValueError("production verifier 뒤 input locations가 변경됐다")
    by_location = {item.get("call_id"): item for item in location_rows if isinstance(item, dict)}
    manifest_hashes = verified_inputs.get("input_manifest_sha256")
    if len(by_location) != 24 or not isinstance(manifest_hashes, dict):
        raise ValueError("production verifier의 input manifest map이 다르다")
    source_ref = protocol.get("source_protocol")
    if not isinstance(source_ref, dict):
        raise TypeError("source protocol reference가 없다")
    _, source_protocol = BASE._artifact(
        repository,
        {"path": source_ref.get("path"), "sha256": source_ref.get("sha256_lf")},
        "source protocol",
        lf=True,
    )
    source_protocol["_verified_sha256_lf"] = source_ref.get("sha256_lf")
    calls: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    for target, row in zip(TARGETS, targets, strict=True):
        verified = by_call.get(row.get("call_id"))
        if not isinstance(verified, dict) or set(verified) != {"call_id", "run_id", "capsule"}:
            raise ValueError("production verifier 대상 capsule이 없다")
        capsule = BASE._capsule_ref(repository, verified["capsule"], row.get("capsule_sha256"))
        location = by_location.get(row.get("call_id"))
        expected_manifest_sha = manifest_hashes.get(row.get("call_id"))
        if not isinstance(location, dict) or not isinstance(expected_manifest_sha, str):
            raise TypeError("대상 input manifest 결합이 없다")
        jobs = _training_jobs(repository, location, source_protocol, row, expected_manifest_sha)
        frozen = BASE.CALL.prepared(capsule)
        if (
            frozen.get("view_kind") != "text_and_video"
            or frozen.get("checkpoint_sha256") != row.get("checkpoint_sha256")
        ):
            raise ValueError("대상 capsule 내용이 등록 row와 다르다")
        eligible = frozen.get("bundle", {}).get("selection_view", {}).get("eligible_probes")
        if not isinstance(eligible, list) or not eligible or any(
            not isinstance(probe, str) or not probe for probe in eligible
        ):
            raise ValueError("대상 capsule 등록 probe 목록이 없다")
        call_id = f"e2e-{target['seed_group']}"
        references = [
            candidate
            for candidate in protocol["rows"]
            if candidate.get("seed_group") == target["seed_group"]
            and candidate.get("source_role") == "normal"
            and candidate.get("window") == "window-01"
            and candidate.get("view_kind") == "text_and_video"
        ]
        if len(references) != 1:
            raise ValueError("NONE reference row가 유일하지 않다")
        calls.append(
            {
                "call_id": call_id,
                "run_id": f"vision-selection-{execution_id}-{call_id}",
                "capsule": capsule,
            }
        )
        frozen_rows.append(
            {
                "target": dict(target),
                "call_id": call_id,
                "source_call_id": row["call_id"],
                "source_seed": row["source_seed"],
                "source_job_id": jobs["source_training_job_id"],
                "record_job_id": jobs["record_job_id"],
                "source_lineage_id": row["source_lineage_id"],
                "corrected_job_id": jobs["corrected_job_id"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "reference_checkpoint_sha256": references[0]["checkpoint_sha256"],
                "capsule_sha256": row["capsule_sha256"],
                "eligible_probes": list(eligible),
                "source_protocol_sha256_lf": row["source_protocol_sha256"],
            }
        )
    reference = initialize(
        repository, execution_id, "e2e", source_anchor, protocol_sha256, calls
    )
    body = {
        "schema": SCHEMA,
        "api_cap": API_CAP,
        "execution_id": execution_id,
        "source_anchor": source_anchor,
        "protocol_sha256": protocol_sha256,
        "input_locations_sha256": BASE._sha(locations_path),
        "selection_plan": reference,
        "comparison": {
            "selection_plan": prepared["plan"],
            "execution_root": comparison_stage.relative_to(repository).as_posix(),
            "output_root": comparison_output.relative_to(repository).as_posix(),
            "worker": "vision_record_selection_worker_v1",
            "execution": "vision_record_selection_execution_v1",
        },
        "rows": frozen_rows,
        "correction": {"id": "restore_body_frame_reward", "recovery_mode": "restart"},
        "fixed_eval_decides_recovery": True,
        "post_observation_used_for_decision": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return body


def execution_module() -> Any:
    """호출자가 사용할 first-class 실행 모듈을 명시한다."""
    return RECORD_EXECUTION
