"""Atomic Writer 원문까지 결합한 Play 실행 증거 manifest를 만든다."""

from __future__ import annotations

import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_atomic_smoke_v1 as SMOKE
import vision_execution_evidence_v1 as V1
import vision_frame_sync_v1 as FRAME
import vision_shutdown_policy_v1 as SHUTDOWN
import vision_writer_payload_v1 as PAYLOAD

EXECUTION_SCHEMA = V1.EXECUTION_SCHEMA
MANIFEST_SCHEMA = "vision_play_execution_evidence_v2"
MANIFEST_NAME = V1.MANIFEST_NAME
def _atomic_paths(stage: Path, job: Path, count: int) -> list[tuple[Path, Path]]:
    expected = set()
    rows = []
    for ordinal in range(count):
        prefix = f"{job.stem}.atomic-{ordinal:06d}"
        json_path, npz_path = stage / f"{prefix}.json", stage / f"{prefix}.npz"
        rows.append((json_path, npz_path))
        expected.update({json_path.name, npz_path.name})
    actual = {path.name for path in stage.glob(f"{job.stem}.atomic-*")}
    if actual != expected:
        raise ValueError("atomic sidecar exact set이 다르다")
    return rows


def _stage_leaf(path: Path, stage: Path, label: str) -> dict[str, Any]:
    target = V1._regular(path, stage, label)
    relative = target.relative_to(stage).as_posix()
    return {"file": relative, "sha256": R.sha256_file(target), "bytes": target.stat().st_size}


def _stage_leaf_path(stage: Path, value: Any, label: str) -> Path:
    if (
        not isinstance(value, dict)
        or set(value) != {"file", "sha256", "bytes"}
        or not isinstance(value.get("file"), str)
        or Path(value["file"]).is_absolute()
        or ".." in Path(value["file"]).parts
        or not isinstance(value.get("sha256"), str)
        or R._SHA256.fullmatch(value["sha256"]) is None
        or type(value.get("bytes")) is not int
        or value["bytes"] <= 0
    ):
        raise ValueError(f"{label} leaf 형식이 다르다")
    target = V1._regular(stage / Path(value["file"]), stage, label)
    metadata = target.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != value["bytes"]
        or R.sha256_file(target) != value["sha256"]
    ):
        raise ValueError(f"{label} leaf SHA 또는 bytes가 다르다")
    return target


def _result_evidence(
    result: Path, stage: Path, mode: str
) -> tuple[dict[str, Any], Path, list[Path]]:
    if mode == R.RENDER_SYNC_SMOKE_MODE:
        primary = V1._artifact(result, "render_sync_smoke", stage)
        audits = [V1._artifact(result, f"audit_rgb_{index}", stage) for index in (1, 2, 3)]
    else:
        primary = V1._artifact(result, "trace", stage)
        audits = []
    return (
        {
            "primary": _stage_leaf(primary, stage, "result primary evidence"),
            "audits": [_stage_leaf(path, stage, "result audit evidence") for path in audits],
        },
        primary,
        audits,
    )


def _trace_rows(path: Path, count: int) -> list[dict[str, Any]]:
    trace = R.read_object(path)
    rows = trace.get("records")
    if (
        trace.get("schema") != R.TRACE_SCHEMA
        or trace.get("frame_count") != count
        or not isinstance(rows, list)
        or len(rows) != count
    ):
        raise ValueError("record trace capture가 다르다")
    return rows


def _build(job_path: Path, result_path: Path, sdk_child_exit_code: int) -> dict[str, Any]:
    stage, job, result = V1._layout(job_path, result_path)
    if sdk_child_exit_code != 0:
        raise ValueError("SDK child exit code가 0이 아니다")
    job_sha = R.sha256_file(job)
    mode, count = V1._mode(job)
    smoke = mode == R.RENDER_SYNC_SMOKE_MODE
    result_evidence, primary, audits = _result_evidence(result, stage, mode)
    readiness, policy, captures = V1._paths(stage, job, count, smoke)
    V1._reject_extra_capture_sidecars(stage, job, captures)
    atomic_paths = _atomic_paths(stage, job, count)
    readiness_path = V1._regular(readiness, stage, "readiness")
    require_fxaa = V1.FXAA.required(R.read_object(job))
    V1._verify_readiness(readiness_path, require_fxaa=require_fxaa)
    SHUTDOWN.verify(job, {"job_sha256": job_sha}, result)

    if smoke:
        report = SMOKE.verify_report(
            primary,
            job,
            {f"audit_rgb_{index}": path for index, path in zip((1, 2, 3), audits, strict=True)},
        )
        source_rows = report.get("captures")
    else:
        source_rows = _trace_rows(primary, count)
    if not isinstance(source_rows, list):
        raise TypeError("result capture rows가 없다")

    manifest_rows = []
    bundles: list[dict[str, Any]] = []
    raw_rows: list[tuple[bytes, bytes]] = []
    for ordinal, ((reset, timing, rgb), (atomic_json, atomic_npz)) in enumerate(
        zip(captures, atomic_paths, strict=True)
    ):
        reset_path = V1._regular(reset, stage, "reset")
        timing_path = V1._regular(timing, stage, "timing")
        rgb_path = V1._regular(rgb, stage, "RGB") if rgb is not None else None
        atomic_json_path = V1._regular(atomic_json, stage, "atomic journal")
        atomic_npz_path = V1._regular(atomic_npz, stage, "atomic raw")
        V1._verify_reset(reset_path, ordinal, require_fxaa=require_fxaa)
        journal, bundle, raw = PAYLOAD.load_capture_journal(job, ordinal)
        PAYLOAD.verify_writer_bundle(journal, bundle, mode, ordinal)
        source_row = source_rows[ordinal]
        if not isinstance(source_row, Mapping):
            raise TypeError("result capture row가 mapping이 아니다")
        physics = bundle.get("physics")
        if not isinstance(physics, Mapping):
            raise TypeError("atomic physics가 없다")
        if smoke:
            if source_row.get("atomic_bundle") != bundle:
                raise ValueError("smoke report bundle이 atomic journal과 다르다")
            simulation_time = source_row.get("simulation_time_before_s")
            if (
                source_row.get("simulation_time_after_s") != simulation_time
                or physics.get("simulation_time_before_s") != simulation_time
                or physics.get("simulation_time_after_s") != simulation_time
            ):
                raise ValueError("smoke report physics time이 atomic bundle과 다르다")
        else:
            state = source_row.get("state")
            simulation_time = state.get("simulation_time_s") if isinstance(state, Mapping) else None
            if (
                source_row.get("capture_ordinal") != ordinal
                or physics.get("simulation_time_before_s") != simulation_time
                or physics.get("simulation_time_after_s") != simulation_time
            ):
                raise ValueError("record trace physics time이 atomic bundle과 다르다")
            if (
                source_row.get("render_timing") != bundle.get("timing")
                or source_row.get("rgb_sha256") != bundle.get("rgb", {}).get("sha256")
            ):
                raise ValueError("record trace timing/RGB가 atomic bundle과 다르다")
        if not isinstance(simulation_time, (int, float)) or isinstance(simulation_time, bool):
            raise TypeError("result capture simulation time이 없다")
        normalized, rgb_sha, artifact = V1._timing_value(
            timing_path, ordinal, float(simulation_time)
        )
        if normalized != bundle.get("timing") or rgb_sha != bundle.get("rgb", {}).get("sha256"):
            raise ValueError("timing raw 값 또는 RGB SHA가 atomic bundle과 다르다")
        if smoke:
            assert rgb_path is not None
            expected_rgb = {
                "file": rgb_path.name,
                "sha256": R.sha256_file(rgb_path),
                "bytes": rgb_path.stat().st_size,
            }
            if artifact != expected_rgb or rgb_sha != expected_rgb["sha256"]:
                raise ValueError("smoke timing raw RGB artifact 결합이 다르다")
        elif artifact is not None:
            raise ValueError("record timing에는 RGB 원문 파일을 보존하지 않는다")
        bundles.append(bundle)
        raw_rows.append(raw)
        manifest_rows.append(
            {
                "ordinal": ordinal,
                "reset": V1._leaf(reset_path, stage, "reset"),
                "timing": V1._leaf(timing_path, stage, "timing"),
                "rgb": V1._leaf(rgb_path, stage, "RGB") if rgb_path is not None else None,
                "atomic_journal": V1._leaf(atomic_json_path, stage, "atomic journal"),
                "atomic_raw": V1._leaf(atomic_npz_path, stage, "atomic raw"),
            }
        )
    FRAME.verify_sequence(bundles, raw_rows)
    return {
        "schema": MANIFEST_SCHEMA,
        "completed": True,
        "execution_schema": EXECUTION_SCHEMA,
        "sdk_child_exit_code": 0,
        "supervisor_verification_complete": True,
        "job_sha256": job_sha,
        "result_sha256": R.sha256_file(result),
        "mode": mode,
        "capture_count": count,
        "readiness": V1._leaf(readiness_path, stage, "readiness"),
        "shutdown_policy": V1._leaf(policy, stage, "shutdown policy"),
        "result_evidence": result_evidence,
        "captures": manifest_rows,
    }


def write_manifest(
    job_path: Path, result_path: Path, *, sdk_child_exit_code: int = 0
) -> Path:
    if sdk_child_exit_code != 0:
        raise ValueError("SDK child exit code가 0이 아니다")
    stage, job, result = V1._layout(job_path, result_path)
    path = stage / MANIFEST_NAME
    if path.exists() or path.is_symlink():
        verify_manifest(path, job, result)
        return path.resolve(strict=True)
    R.write_new(path, _build(job, result, sdk_child_exit_code))
    return path.resolve(strict=True)


def verify_manifest(
    manifest_path: Path, job_path: Path, result_path: Path
) -> dict[str, Any]:
    stage, job, result = V1._layout(job_path, result_path)
    manifest = V1._regular(manifest_path, stage, "execution evidence manifest")
    if manifest != stage / MANIFEST_NAME:
        raise ValueError("execution evidence manifest 이름 또는 위치가 다르다")
    value = R.read_object(manifest)
    expected = {
        "schema",
        "execution_schema",
        "completed",
        "sdk_child_exit_code",
        "supervisor_verification_complete",
        "job_sha256",
        "result_sha256",
        "mode",
        "capture_count",
        "readiness",
        "shutdown_policy",
        "result_evidence",
        "captures",
    }
    mode, count = V1._mode(job)
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("execution_schema") != EXECUTION_SCHEMA
        or value.get("completed") is not True
        or value.get("sdk_child_exit_code") != 0
        or value.get("supervisor_verification_complete") is not True
        or value.get("job_sha256") != R.sha256_file(job)
        or value.get("result_sha256") != R.sha256_file(result)
        or value.get("mode") != mode
        or value.get("capture_count") != count
        or not isinstance(value.get("captures"), list)
        or len(value["captures"]) != count
    ):
        raise ValueError("execution evidence manifest schema 또는 결합이 다르다")
    V1._leaf_path(stage, value["readiness"], "readiness")
    policy = V1._leaf_path(stage, value["shutdown_policy"], "shutdown policy")
    if policy != stage / SHUTDOWN.SIDECAR:
        raise ValueError("shutdown policy 위치가 다르다")
    result_evidence = value.get("result_evidence")
    if (
        not isinstance(result_evidence, dict)
        or set(result_evidence) != {"primary", "audits"}
        or not isinstance(result_evidence.get("audits"), list)
        or len(result_evidence["audits"]) != (3 if mode == R.RENDER_SYNC_SMOKE_MODE else 0)
    ):
        raise ValueError("result evidence manifest 형식이 다르다")
    _stage_leaf_path(stage, result_evidence["primary"], "result primary evidence")
    for audit in result_evidence["audits"]:
        _stage_leaf_path(stage, audit, "result audit evidence")
    for ordinal, row in enumerate(value["captures"]):
        if (
            not isinstance(row, dict)
            or set(row)
            != {"ordinal", "reset", "timing", "rgb", "atomic_journal", "atomic_raw"}
            or row.get("ordinal") != ordinal
        ):
            raise ValueError("capture manifest ordinal 또는 필드가 다르다")
        for field in ("reset", "timing", "atomic_journal", "atomic_raw"):
            V1._leaf_path(stage, row[field], field)
        if row["rgb"] is not None:
            V1._leaf_path(stage, row["rgb"], "RGB")
    if value != _build(job, result, 0):
        raise ValueError("execution evidence manifest가 현재 원문 증거와 다르다")
    return value
