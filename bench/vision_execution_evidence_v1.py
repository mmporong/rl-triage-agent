"""Play 실행의 readiness·capture·shutdown 증거를 이동 가능한 manifest로 봉인한다."""

from __future__ import annotations

import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_capture_readiness_v1 as READINESS
import vision_fxaa_policy_v1 as FXAA
import vision_shutdown_policy_v1 as SHUTDOWN
import vision_timing_adapter_v1 as TIMING

EXECUTION_SCHEMA = "vision_brev_record_execution_v2"
MANIFEST_SCHEMA = "vision_play_execution_evidence_v1"
MANIFEST_NAME = "job.execution-evidence.json"
RESET_SCHEMA = "vision_capture_renderer_reset_v1"
RESET_METHOD = "omni.usd.UsdContext.reset_renderer_accumulation"
COUNT_KEYS = frozenset({"loading", "loaded", "aborted", "synthetic_loading"})


def _regular(path: Path, stage: Path, label: str) -> Path:
    target = Path(path)
    if (not target.is_file() or target.is_symlink()
            or any(parent.is_symlink() for parent in target.parents)):
        raise ValueError(f"{label}가 regular file이 아니다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label}가 단일 regular file이 아니다")
    resolved = target.resolve(strict=True)
    try:
        resolved.relative_to(stage)
    except ValueError as error:
        raise ValueError(f"{label}가 stage 밖을 가리킨다") from error
    return resolved


def _layout(job_path: Path, result_path: Path) -> tuple[Path, Path, Path]:
    raw_job = Path(job_path)
    stage = raw_job.parent.resolve(strict=True)
    job = _regular(raw_job, stage, "job")
    result = _regular(Path(result_path), stage, "result")
    if job != stage / "job.json" or result != stage / "output" / "result.json":
        raise ValueError("execution evidence layout이 다르다")
    return stage, job, result


def _leaf(path: Path, stage: Path, label: str) -> dict[str, Any]:
    target = _regular(path, stage, label)
    return {"file": target.name, "sha256": R.sha256_file(target),
            "bytes": target.stat().st_size}


def _leaf_path(stage: Path, value: Any, label: str) -> Path:
    if (not isinstance(value, dict) or set(value) != {"file", "sha256", "bytes"}
            or not isinstance(value.get("file"), str)
            or Path(value["file"]).name != value["file"]
            or not isinstance(value.get("sha256"), str)
            or R._SHA256.fullmatch(value["sha256"]) is None
            or type(value.get("bytes")) is not int or value["bytes"] <= 0):
        raise ValueError(f"{label} leaf 형식이 다르다")
    target = _regular(stage / value["file"], stage, label)
    if target.stat().st_size != value["bytes"] or R.sha256_file(target) != value["sha256"]:
        raise ValueError(f"{label} leaf SHA 또는 bytes가 다르다")
    return target


def _mode(job: Path) -> tuple[str, int]:
    mode = R.read_object(job).get("mode")
    if mode == R.RENDER_SYNC_SMOKE_MODE:
        return mode, 4
    if mode == R.RECORD_MODE:
        return mode, R.FRAME_COUNT
    raise ValueError("job mode가 record/render_sync_smoke가 아니다")


def _event_counts(value: Any) -> dict[str, int]:
    if (not isinstance(value, dict) or set(value) != COUNT_KEYS
            or any(type(item) is not int or item < 0 for item in value.values())
            or value["synthetic_loading"] > value["loading"]
            or value["loaded"] + value["aborted"] > value["loading"]):
        raise ValueError("readiness asset event count 형식이 다르다")
    return value


def _verify_readiness(path: Path, *, require_fxaa: bool = False) -> None:
    value = R.read_object(path)
    bounds = value.get("bounds")
    counts = _event_counts(value.get("event_counts"))
    history = value.get("loading_history")
    renderer = value.get("renderer")
    camera = R.fixed_profile()["camera"]
    capture_profile = {key: camera[key] for key in
                       ("capture_primitive", "rt_subframes", "pause_timeline", "delta_time_s")}
    elapsed = value.get("elapsed_s")
    updates = value.get("updates")
    if (set(value) != {"schema", "status", "reason", "bounds", "updates", "elapsed_s",
                       "event_counts", "event_history", "loading_history", "before_loading",
                       "final_loading", "physics_before", "physics_after", "renderer"}
            or value.get("schema") != READINESS.SCHEMA or value.get("status") != "ready"
            or value.get("reason") != "consecutive_loading_streaming_idle"
            or bounds != {"max_updates": READINESS.MAX_UPDATES,
                          "polling_elapsed_budget_s": READINESS.TIMEOUT_S,
                          "polling_elapsed_budget_scope": "checked_between_synchronous_app_updates_not_a_hard_watchdog",
                          "required_consecutive_idle_updates": READINESS.REQUIRED_IDLE_UPDATES}
            or type(updates) is not int
            or not READINESS.REQUIRED_IDLE_UPDATES <= updates <= READINESS.MAX_UPDATES
            or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed < 0
            or counts.get("aborted") != 0
            or counts.get("loading") != counts.get("loaded")
            or not isinstance(value.get("physics_before"), dict) or not value["physics_before"]
            or value.get("physics_before") != value.get("physics_after")):
        raise ValueError("capture readiness 성공 불변식이 다르다")
    final = value.get("final_loading")
    if (not isinstance(final, dict) or final.get("files_loaded") != 0
            or final.get("files_total") != 0 or final.get("streaming_busy") is not False
            or not isinstance(history, list) or len(history) != updates):
        raise ValueError("capture readiness 최종 idle 증거가 없다")
    previous = {key: 0 for key in COUNT_KEYS}
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict) or row.get("update") != index:
            raise ValueError("capture readiness update 순서가 다르다")
        observed = _event_counts(row.get("event_counts"))
        control = row.get("simulation_update_control")
        if (not isinstance(control, dict) or set(control) != {"setting", "before", "during", "after"}
                or control["setting"] != READINESS.PLAYER_SETTING
                or type(control["before"]) is not bool or control["during"] is not False
                or control["after"] is not control["before"]):
            raise ValueError("capture readiness simulation update 제어가 다르다")
        pending = observed["loading"] - observed["loaded"] - observed["aborted"]
        if (type(row.get("pending_asset_batches")) is not int
                or row["pending_asset_batches"] != pending
                or any(observed[key] < previous[key] for key in COUNT_KEYS)):
            raise ValueError("capture readiness history count 결합이 다르다")
        previous = observed
    if previous != counts:
        raise ValueError("capture readiness 최종 event count가 history와 다르다")
    for row in history[-READINESS.REQUIRED_IDLE_UPDATES:]:
        loading = row.get("loading") if isinstance(row, dict) else None
        if (not isinstance(loading, dict) or row.get("pending_asset_batches") != 0
                or loading.get("files_loaded") != 0 or loading.get("files_total") != 0
                or loading.get("streaming_busy") is not False):
            raise ValueError("capture readiness 연속 idle 증거가 다르다")
    if (not isinstance(renderer, dict)
            or renderer.get("expected_capture_profile") != capture_profile
            or renderer.get("renderer_accumulation_reset") != RESET_METHOD
            or not isinstance(renderer.get("attached_hydra_engine_names"), list)
            or not renderer["attached_hydra_engine_names"]
            or any(not isinstance(name, str) or not name for name in renderer["attached_hydra_engine_names"])
            or not isinstance(renderer.get("carb_settings"), dict)
            or set(renderer["carb_settings"]) != set(READINESS.CARB_SETTING_KEYS)):
        raise ValueError("capture readiness live renderer readback이 다르다")
    if require_fxaa:
        FXAA.verify_receipt(renderer.get("render_policy"))
        if (renderer["carb_settings"]["/rtx/post/aa/op"] != 2
                or renderer["carb_settings"]["/rtx/rendermode"]
                != renderer["render_policy"]["after"]["/rtx/rendermode"]):
            raise ValueError("FXAA 정책과 readiness renderer 원문이 다르다")


def _verify_reset(path: Path, ordinal: int, *, require_fxaa: bool = False) -> None:
    value = R.read_object(path)
    if require_fxaa:
        FXAA.verify_readback(value.pop("render_policy_readback", None))
    if value != {"schema": RESET_SCHEMA, "capture_ordinal": ordinal,
                 "completed": True, "method": RESET_METHOD}:
        raise ValueError("renderer reset receipt가 ordinal과 다르다")


def _typed_mapping(value: Any, label: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != {"type", "fields", "non_string_key_types"}
            or not isinstance(value.get("type"), str) or not value["type"]
            or not isinstance(value.get("fields"), dict)
            or value.get("non_string_key_types") != []):
        raise ValueError(f"{label} typed mapping 형식이 다르다")
    decoded = {}
    for key, scalar in value["fields"].items():
        if (not isinstance(key, str) or not isinstance(scalar, dict)
                or set(scalar) != {"type", "encoding", "value"}
                or not isinstance(scalar.get("type"), str)
                or not isinstance(scalar.get("encoding"), str)
                or not scalar["encoding"].endswith("json_native")):
            raise ValueError(f"{label} scalar를 실제 값으로 복원할 수 없다")
        decoded[key] = scalar["value"]
    return decoded


def _timing_value(path: Path, ordinal: int, simulation_time: float) -> tuple[dict[str, Any], str, Any]:
    value = R.read_object(path)
    if (set(value) != {"schema", "capture_ordinal", "observed_read_times_key_count", "read_times",
                       "reference_time", "execution_port", "rgb_sha256", "rgb_artifact"}
            or value.get("schema") != TIMING.SCHEMA or value.get("capture_ordinal") != ordinal):
        raise ValueError("timing raw receipt schema 또는 ordinal이 다르다")
    raw_times = _typed_mapping(value.get("read_times"), "IsaacReadTimes")
    reference = _typed_mapping(value.get("reference_time"), "ReferenceTime")
    if value.get("observed_read_times_key_count") != len(raw_times):
        raise ValueError("timing raw key count가 원문과 다르다")
    execution_port = value.get("execution_port")
    present = TIMING.EXECUTION_PORT in raw_times
    if (not isinstance(execution_port, dict)
            or set(execution_port) != {"field", "present", "used_as_timing", "raw"}
            or execution_port["field"] != TIMING.EXECUTION_PORT
            or execution_port["present"] is not present
            or execution_port["used_as_timing"] is not False
            or execution_port["raw"] != value["read_times"]["fields"].get(TIMING.EXECUTION_PORT)):
        raise ValueError("timing execution port 원문이 read times와 다르다")
    normalized = TIMING.normalize_render_timing(raw_times, reference, simulation_time)
    rgb_sha = value.get("rgb_sha256")
    if not isinstance(rgb_sha, str) or R._SHA256.fullmatch(rgb_sha) is None:
        raise ValueError("timing raw RGB SHA가 없다")
    return normalized, rgb_sha, value.get("rgb_artifact")


def _artifact(result: Path, logical_id: str, stage: Path) -> Path:
    artifacts = R.read_object(result).get("artifacts")
    descriptor = artifacts.get(logical_id) if isinstance(artifacts, dict) else None
    if (not isinstance(descriptor, dict) or set(descriptor) != {"file", "sha256", "bytes"}
            or not isinstance(descriptor.get("file"), str)
            or Path(descriptor["file"]).name != descriptor["file"]):
        raise ValueError(f"result {logical_id} artifact가 없다")
    matches = [path for path in result.parent.rglob(descriptor["file"]) if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"result {logical_id} artifact가 유일하지 않다")
    target = _regular(matches[0], stage, f"result {logical_id}")
    if target.stat().st_size != descriptor.get("bytes") or R.sha256_file(target) != descriptor.get("sha256"):
        raise ValueError(f"result {logical_id} artifact 결합이 다르다")
    return target


def _result_rows(result: Path, mode: str, count: int, stage: Path) -> list[dict[str, Any]]:
    if mode == R.RENDER_SYNC_SMOKE_MODE:
        report = R.read_object(_artifact(result, "render_sync_smoke", stage))
        rows = report.get("captures")
        if (report.get("schema") != "vision_render_sync_smoke_v1"
                or report.get("status") != "pass" or not isinstance(rows, list)
                or len(rows) != count):
            raise ValueError("render smoke report capture가 다르다")
        return rows
    trace = R.read_object(_artifact(result, "trace", stage))
    rows = trace.get("records")
    if (trace.get("schema") != R.TRACE_SCHEMA or trace.get("frame_count") != count
            or not isinstance(rows, list) or len(rows) != count):
        raise ValueError("record trace capture가 다르다")
    return rows


def _paths(stage: Path, job: Path, count: int, smoke: bool) -> tuple[Path, Path, list[tuple[Path, Path, Path | None]]]:
    captures = []
    for ordinal in range(count):
        prefix = f"{job.stem}.capture-{ordinal:06d}"
        captures.append((stage / f"{prefix}.renderer-reset.json",
                         stage / f"{prefix}.timing-raw.json",
                         stage / f"{prefix}.rgb" if smoke else None))
    return job.with_suffix(READINESS.SIDECAR_SUFFIX), stage / SHUTDOWN.SIDECAR, captures


def _reject_extra_capture_sidecars(stage: Path, job: Path, captures: list[tuple[Path, Path, Path | None]]) -> None:
    expected = {path.name for row in captures for path in row if path is not None}
    actual = {
        path.name for path in stage.glob(f"{job.stem}.capture-*")
        if path.name != job.with_suffix(READINESS.SIDECAR_SUFFIX).name
    }
    if actual != expected:
        raise ValueError("capture sidecar exact set이 다르다")


def _build(job_path: Path, result_path: Path, sdk_child_exit_code: int) -> dict[str, Any]:
    stage, job, result = _layout(job_path, result_path)
    if sdk_child_exit_code != 0:
        raise ValueError("SDK child exit code가 0이 아니다")
    job_sha = R.sha256_file(job)
    mode, count = _mode(job)
    smoke = mode == R.RENDER_SYNC_SMOKE_MODE
    report_rows = _result_rows(result, mode, count, stage)
    readiness, policy, captures = _paths(stage, job, count, smoke)
    _reject_extra_capture_sidecars(stage, job, captures)
    readiness_path = _regular(readiness, stage, "readiness")
    require_fxaa = FXAA.required(R.read_object(job))
    _verify_readiness(readiness_path, require_fxaa=require_fxaa)
    SHUTDOWN.verify(job, {"job_sha256": job_sha}, result)
    rows = []
    for ordinal, (reset, timing, rgb) in enumerate(captures):
        reset_path = _regular(reset, stage, "reset")
        timing_path = _regular(timing, stage, "timing")
        rgb_path = _regular(rgb, stage, "RGB") if rgb is not None else None
        _verify_reset(reset_path, ordinal, require_fxaa=require_fxaa)
        report_row = report_rows[ordinal]
        if not isinstance(report_row, Mapping):
            raise TypeError("result capture row가 mapping이 아니다")
        if smoke:
            simulation_time = report_row.get("simulation_time_s")
            report_timing = {"read_times": report_row.get("read_times"),
                             "reference_time": report_row.get("reference_time")}
        else:
            if report_row.get("capture_ordinal") != ordinal:
                raise ValueError("record trace capture ordinal이 다르다")
            state = report_row.get("state")
            simulation_time = state.get("simulation_time_s") if isinstance(state, Mapping) else None
            report_timing = report_row.get("render_timing")
        if not isinstance(simulation_time, (int, float)) or isinstance(simulation_time, bool):
            raise TypeError("result capture simulation time이 없다")
        normalized, rgb_sha, artifact = _timing_value(timing_path, ordinal, float(simulation_time))
        if normalized != report_timing or rgb_sha != report_row.get("rgb_sha256"):
            raise ValueError("timing raw 값 또는 RGB SHA가 result capture와 다르다")
        if smoke:
            assert rgb_path is not None
            expected_rgb = {"file": rgb_path.name, "sha256": R.sha256_file(rgb_path),
                            "bytes": rgb_path.stat().st_size}
            if artifact != expected_rgb or rgb_sha != expected_rgb["sha256"]:
                raise ValueError("smoke timing raw RGB artifact 결합이 다르다")
        elif artifact is not None:
            raise ValueError("record timing에는 RGB 원문 파일을 보존하지 않는다")
        rows.append({"ordinal": ordinal, "reset": _leaf(reset_path, stage, "reset"),
                     "timing": _leaf(timing_path, stage, "timing"),
                     "rgb": _leaf(rgb_path, stage, "RGB") if rgb_path is not None else None})
    return {"schema": MANIFEST_SCHEMA, "completed": True,
            "execution_schema": EXECUTION_SCHEMA, "sdk_child_exit_code": 0,
            "supervisor_verification_complete": True, "job_sha256": job_sha,
            "result_sha256": R.sha256_file(result), "mode": mode,
            "capture_count": count, "readiness": _leaf(readiness_path, stage, "readiness"),
            "shutdown_policy": _leaf(policy, stage, "shutdown policy"), "captures": rows}


def write_manifest(job_path: Path, result_path: Path, *, sdk_child_exit_code: int) -> Path:
    if sdk_child_exit_code != 0:
        raise ValueError("SDK child exit code가 0이 아니다")
    stage, job, result = _layout(job_path, result_path)
    path = stage / MANIFEST_NAME
    if path.exists() or path.is_symlink():
        verify_manifest(path, job, result)
        return path.resolve(strict=True)
    R.write_new(path, _build(job, result, sdk_child_exit_code))
    return path.resolve(strict=True)


def verify_manifest(manifest_path: Path, job_path: Path, result_path: Path) -> dict[str, Any]:
    stage, job, result = _layout(job_path, result_path)
    manifest = _regular(manifest_path, stage, "execution evidence manifest")
    if manifest != stage / MANIFEST_NAME:
        raise ValueError("execution evidence manifest 이름 또는 위치가 다르다")
    value = R.read_object(manifest)
    if value.get("schema") == "vision_play_execution_evidence_v2":
        import vision_execution_evidence_v2 as V2

        return V2.verify_manifest(manifest, job, result)
    expected = {"schema", "execution_schema", "completed", "sdk_child_exit_code",
                "supervisor_verification_complete", "job_sha256", "result_sha256", "mode",
                "capture_count", "readiness", "shutdown_policy", "captures"}
    mode, count = _mode(job)
    if (set(value) != expected or value.get("schema") != MANIFEST_SCHEMA
            or value.get("execution_schema") != EXECUTION_SCHEMA or value.get("completed") is not True
            or value.get("sdk_child_exit_code") != 0
            or value.get("supervisor_verification_complete") is not True
            or value.get("job_sha256") != R.sha256_file(job)
            or value.get("result_sha256") != R.sha256_file(result)
            or value.get("mode") != mode or value.get("capture_count") != count
            or not isinstance(value.get("captures"), list) or len(value["captures"]) != count):
        raise ValueError("execution evidence manifest schema 또는 결합이 다르다")
    _leaf_path(stage, value["readiness"], "readiness")
    policy = _leaf_path(stage, value["shutdown_policy"], "shutdown policy")
    if policy != stage / SHUTDOWN.SIDECAR:
        raise ValueError("shutdown policy 위치가 다르다")
    for ordinal, row in enumerate(value["captures"]):
        if (not isinstance(row, dict) or set(row) != {"ordinal", "reset", "timing", "rgb"}
                or row["ordinal"] != ordinal):
            raise ValueError("capture manifest ordinal이 다르다")
        _leaf_path(stage, row["reset"], "reset")
        _leaf_path(stage, row["timing"], "timing")
        if row["rgb"] is not None:
            _leaf_path(stage, row["rgb"], "RGB")
    if value != _build(job, result, 0):
        raise ValueError("execution evidence manifest가 현재 원문 증거와 다르다")
    return value
