"""렌더 context가 record/smoke 실행보다 먼저 수집됐음을 write-once로 봉인한다."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import vision_render_context_v1 as CONTEXT

SCHEMA = "vision_render_attestation_v1"
FAILURE_SCHEMA = "vision_render_attestation_failure_v1"
WRAPPER_RELATIVE = "bench/vision_render_attestation_v1.py"
ATTESTATION_FILE = "attestation.json"
FAILURE_FILE = "failure.json"
PHASE_ORDER = ["source_before", "context", "execute", "source_after", "seal"]
RECEIPT_KEYS = frozenset({
    "schema", "completed", "job_sha256", "wrapper_sha256_lf", "context", "execution",
    "result", "source_snapshot_sha256", "execute_started_at_utc", "execute_finished_at_utc",
    "attested_at_utc", "phase_order",
})
REFERENCE_KEYS = frozenset({"path", "sha256"})
EXECUTION_KEYS = frozenset({
    "schema", "completed", "job_sha256", "exit_code", "timed_out", "cleanup_verified",
    "gpu_before", "gpu_after", "gpu_before_observation", "gpu_after_observation",
    "source_before", "source_after", "result_sha256", "launch", "wall_s",
})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Brev Python 3.10


def _clock_value(clock: Callable[[], datetime], anchor: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{anchor}는 aware UTC datetime이어야 한다")
    return value.astimezone(timezone.utc)  # noqa: UP017 - Brev Python 3.10


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    data = CONTEXT._canonical(dict(value)) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _reference(value: Any, anchor: str) -> dict[str, str]:
    row = CONTEXT._exact(value, REFERENCE_KEYS, anchor)
    path = CONTEXT._portable_absolute_path(row["path"], f"{anchor} path")
    digest = CONTEXT._hash(row["sha256"], anchor)
    return {"path": str(path), "sha256": digest}


def _fixed_target_provenance(job: Mapping[str, Any]) -> tuple[str, str]:
    output = CONTEXT._portable_absolute_path(job.get("output_directory"), "record output_directory")
    return str(output.parent / "execution.json"), str(output / "result.json")


def _live_fixed_targets(job: Mapping[str, Any], *, require_absent: bool) -> tuple[Path, Path]:
    execution_raw, result_raw = _fixed_target_provenance(job)
    targets = (Path(execution_raw), Path(result_raw))
    for target, anchor in zip(targets, ("record execution target", "record result target"), strict=True):
        if not target.is_absolute() or target.resolve() != target:
            raise ValueError(f"{anchor} 경로는 정규화된 절대경로여야 한다")
        CONTEXT._reject_symlink_ancestors(target, anchor)
        if require_absent and target.exists():
            raise FileExistsError(f"{anchor}가 execute 전에 이미 존재한다")
    return targets


def _actual_reference(
    reference: Mapping[str, Any], actual_path: Path, anchor: str,
) -> tuple[dict[str, str], Path]:
    row = _reference(reference, anchor)
    actual = CONTEXT._canonical_regular(Path(actual_path), anchor)
    if CONTEXT._sha256(actual) != row["sha256"]:
        raise ValueError(f"{anchor} 실제 bytes SHA가 reference와 다르다")
    return row, actual


def _wrapper_binding(job: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    wrapper_sha = CONTEXT._sha256(Path(__file__).resolve(), lf=True)
    job_files = job["source"]["files_lf"]
    source_files = snapshot["source_sha256_lf"]
    if job_files.get(WRAPPER_RELATIVE) != wrapper_sha or source_files.get(WRAPPER_RELATIVE) != wrapper_sha:
        raise ValueError("attestation wrapper LF SHA가 job/source snapshot과 다르다")
    return wrapper_sha


def _verify_record_protocol(
    job: Mapping[str, Any], snapshot: Mapping[str, Any], protocol_path: Path,
) -> None:
    descriptor = CONTEXT._exact(job.get("protocol"), {"path", "sha256"}, "record protocol")
    CONTEXT._portable_absolute_path(descriptor["path"], "record protocol provenance path")
    protocol_path = CONTEXT._canonical_regular(Path(protocol_path), "record protocol local original")
    expected = CONTEXT._hash(descriptor["sha256"], "record protocol")
    if CONTEXT._sha256(protocol_path) != expected:
        raise ValueError("record protocol bytes SHA가 job과 다르다")
    body = CONTEXT._read_object(protocol_path, "record protocol")
    files = CONTEXT._relative_hashes(body.get("source_files_lf"), "record protocol source_files_lf")
    if body.get("source_commit") != snapshot["source_anchor"] or files != job["source"]["files_lf"]:
        raise ValueError("record protocol source binding이 job/source snapshot과 다르다")
    _wrapper_binding(job, snapshot)


def _execution_body(
    path: Path, expected_sha: str, job_sha: str, source_snapshot: Mapping[str, Any],
    result_sha: str,
) -> dict[str, Any]:
    if CONTEXT._sha256(path) != expected_sha:
        raise ValueError("record execution bytes SHA가 reference와 다르다")
    value = CONTEXT._exact(
        CONTEXT._read_object(path, "record execution"), EXECUTION_KEYS, "record execution",
    )
    if (
        value["schema"] != "vision_brev_record_execution_v1"
        or value["completed"] is not True
        or value["job_sha256"] != job_sha
        or value["exit_code"] != 0
        or value["timed_out"] is not False
        or value["cleanup_verified"] is not True
        or value["gpu_before"] != []
        or value["gpu_after"] != []
        or not isinstance(value["launch"], dict)
        or not value["launch"]
        or value["result_sha256"] != result_sha
        or CONTEXT._canonical(value["source_before"]) != CONTEXT._canonical(source_snapshot)
        or CONTEXT._canonical(value["source_after"]) != CONTEXT._canonical(source_snapshot)
    ):
        raise ValueError("record execution 완료·source·result 결합이 다르다")
    return value


def _result_body(path: Path, expected_sha: str, job_sha: str) -> dict[str, Any]:
    if CONTEXT._sha256(path) != expected_sha:
        raise ValueError("record result bytes SHA가 reference와 다르다")
    value = CONTEXT._read_object(path, "record result")
    if (
        value.get("schema") != "vision_play_record_result_v1"
        or value.get("completed") is not True
        or value.get("job_sha256") != job_sha
    ):
        raise ValueError("record result schema·완료·job 결합이 다르다")
    return value


def _verify_times(receipt: Mapping[str, Any], context_path: Path) -> None:
    context_body = CONTEXT._read_object(context_path, "render context receipt")
    context_finished = CONTEXT._utc_stamp(context_body.get("finished_at_utc"), "context finished_at_utc")
    execute_started = CONTEXT._utc_stamp(receipt["execute_started_at_utc"], "execute_started_at_utc")
    execute_finished = CONTEXT._utc_stamp(receipt["execute_finished_at_utc"], "execute_finished_at_utc")
    attested = CONTEXT._utc_stamp(receipt["attested_at_utc"], "attested_at_utc")
    if not context_finished <= execute_started <= execute_finished <= attested:
        raise ValueError("context→execute→attestation 시각 순서가 다르다")


def verify_attestation(
    attestation_path: Path,
    job: dict[str, Any],
    job_sha256: str,
    source_snapshot: dict[str, Any],
    context_path: Path,
    execution_path: Path,
    result_path: Path,
    *,
    protocol_path: Path,
) -> dict[str, Any]:
    """Attestation과 별도 보관된 context/execution/result 실제 bytes를 재검증한다."""
    attestation_path = CONTEXT._canonical_regular(Path(attestation_path), "render attestation")
    if attestation_path.name != ATTESTATION_FILE:
        raise ValueError("render attestation 파일명이 다르다")
    snapshot = CONTEXT._validate_snapshot(source_snapshot, portable=True)
    checked_job = CONTEXT._validate_job(job, snapshot, verify_files=False)
    expected_job_sha = CONTEXT._hash(job_sha256, "job")
    wrapper_sha = _wrapper_binding(checked_job, snapshot)
    _verify_record_protocol(checked_job, snapshot, protocol_path)
    receipt = CONTEXT._exact(
        CONTEXT._read_object(attestation_path, "render attestation"), RECEIPT_KEYS,
        "render attestation",
    )
    if (
        receipt["schema"] != SCHEMA
        or receipt["completed"] is not True
        or receipt["job_sha256"] != expected_job_sha
        or receipt["wrapper_sha256_lf"] != wrapper_sha
        or receipt["source_snapshot_sha256"] != CONTEXT._digest(snapshot)
        or receipt["phase_order"] != PHASE_ORDER
    ):
        raise ValueError("render attestation 고정 결합이 다르다")

    context_ref = CONTEXT._exact(receipt["context"], REFERENCE_KEYS, "context reference")
    if context_ref["path"] != f"context/{CONTEXT.RECEIPT_FILE}":
        raise ValueError("context reference 상대경로가 다르다")
    context_actual = CONTEXT._canonical_regular(Path(context_path), "render context receipt")
    context_sha = CONTEXT._hash(context_ref["sha256"], "context reference")
    if CONTEXT._sha256(context_actual) != context_sha:
        raise ValueError("render context receipt bytes SHA가 다르다")
    CONTEXT.read_context(context_actual, checked_job, expected_job_sha, snapshot)

    execution_ref, execution_actual = _actual_reference(
        receipt["execution"], execution_path, "record execution",
    )
    result_ref, result_actual = _actual_reference(receipt["result"], result_path, "record result")
    expected_execution, expected_result = _fixed_target_provenance(checked_job)
    if execution_ref["path"] != expected_execution or result_ref["path"] != expected_result:
        raise ValueError("record execution/result provenance가 job 고정 target과 다르다")
    _result_body(result_actual, result_ref["sha256"], expected_job_sha)
    _execution_body(
        execution_actual, execution_ref["sha256"], expected_job_sha, snapshot,
        result_ref["sha256"],
    )
    _verify_times(receipt, context_actual)
    return receipt


def _failure(output_dir: Path, stage: str, error: BaseException, job_sha: str) -> None:
    _write_json_new(
        output_dir / FAILURE_FILE,
        {
            "schema": FAILURE_SCHEMA, "completed": False, "stage": stage,
            "error_class": type(error).__name__, "job_sha256": job_sha,
        },
    )


def attest_capture(
    job_path: Path,
    source_snapshot_provider: Callable[[], dict[str, Any]],
    python: Path,
    output_dir: Path,
    execute: Callable[[], dict[str, Any]],
    *,
    clock: Callable[[], datetime] = _utc_now,
    context_capture: Callable[..., Path] = CONTEXT.capture_context,
) -> Path:
    """source→context→execute→source 순서를 한 번만 수행해 attestation을 만든다."""
    job_path = CONTEXT._canonical_regular(Path(job_path), "job")
    job_sha = CONTEXT._sha256(job_path)
    job = CONTEXT._read_object(job_path, "job")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute() or output_dir.resolve() != output_dir:
        raise ValueError("attestation output 경로는 정규화된 절대경로여야 한다")
    CONTEXT._canonical_directory(output_dir.parent, "attestation output parent")
    output_dir.mkdir(exist_ok=False)
    CONTEXT._reject_symlink_ancestors(output_dir, "attestation output")

    stage = "source_before"
    try:
        source_before = source_snapshot_provider()
        snapshot = CONTEXT._validate_snapshot(source_before, python, portable=False)
        checked_job = CONTEXT._validate_job(job, snapshot, verify_files=True)
        wrapper_sha = _wrapper_binding(checked_job, snapshot)
        record_protocol_path = Path(checked_job["protocol"]["path"])
        _verify_record_protocol(checked_job, snapshot, record_protocol_path)
        expected_execution, expected_result = _live_fixed_targets(checked_job, require_absent=True)

        stage = "context"
        context_path = context_capture(job_path, snapshot, python, output_dir / "context")
        expected_context = (output_dir / "context" / CONTEXT.RECEIPT_FILE).resolve()
        context_path = CONTEXT._canonical_regular(Path(context_path), "render context receipt")
        if context_path != expected_context:
            raise ValueError("context capture가 소유 output/context receipt를 반환하지 않았다")
        CONTEXT.read_context(context_path, checked_job, job_sha, snapshot)
        context_finished = CONTEXT._utc_stamp(
            CONTEXT._read_object(context_path, "render context receipt")["finished_at_utc"],
            "context finished_at_utc",
        )

        stage = "execute"
        execute_started = _clock_value(clock, "execute 시작 시각")
        if execute_started < context_finished:
            raise ValueError("execute가 context 수집 완료보다 먼저 시작됐다")
        _live_fixed_targets(checked_job, require_absent=True)
        returned = execute()
        execute_finished = _clock_value(clock, "execute 종료 시각")
        if execute_finished < execute_started:
            raise ValueError("execute 종료 시각이 시작 시각보다 빠르다")
        result = CONTEXT._exact(returned, {"execution", "result"}, "execute 반환값")
        execution_ref = _reference(result["execution"], "record execution")
        result_ref = _reference(result["result"], "record result")
        execution_path = CONTEXT._canonical_regular(
            Path(execution_ref["path"]), "record execution",
        )
        result_path = CONTEXT._canonical_regular(Path(result_ref["path"]), "record result")
        if execution_path != expected_execution or result_path != expected_result:
            raise ValueError("execute 반환 경로가 job 고정 target과 다르다")

        stage = "source_after"
        source_after = source_snapshot_provider()
        if CONTEXT._canonical(source_after) != CONTEXT._canonical(snapshot):
            raise ValueError("execute 전후 source snapshot이 다르다")
        CONTEXT._validate_snapshot(source_after, python, portable=False)
        _result_body(result_path, result_ref["sha256"], job_sha)
        _execution_body(
            execution_path, execution_ref["sha256"], job_sha, snapshot, result_ref["sha256"],
        )

        stage = "seal"
        attested = _clock_value(clock, "attestation 시각")
        if attested < execute_finished:
            raise ValueError("attestation 시각이 execute 종료보다 빠르다")
        receipt = {
            "schema": SCHEMA,
            "completed": True,
            "job_sha256": job_sha,
            "wrapper_sha256_lf": wrapper_sha,
            "context": {
                "path": f"context/{CONTEXT.RECEIPT_FILE}",
                "sha256": CONTEXT._sha256(context_path),
            },
            "execution": execution_ref,
            "result": result_ref,
            "source_snapshot_sha256": CONTEXT._digest(snapshot),
            "execute_started_at_utc": execute_started.isoformat(),
            "execute_finished_at_utc": execute_finished.isoformat(),
            "attested_at_utc": attested.isoformat(),
            "phase_order": PHASE_ORDER,
        }
        attestation_path = output_dir / ATTESTATION_FILE
        _write_json_new(attestation_path, receipt)
        verify_attestation(
            attestation_path, checked_job, job_sha, snapshot, context_path,
            execution_path, result_path, protocol_path=record_protocol_path,
        )
        return attestation_path
    except BaseException as error:
        _failure(output_dir, stage, error, job_sha)
        raise
