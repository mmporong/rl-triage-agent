from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

R = importlib.import_module("record_vision_evidence_v1")
READINESS = importlib.import_module("vision_capture_readiness_v1")
E = importlib.import_module("vision_execution_evidence_v1")
SHUTDOWN = importlib.import_module("vision_shutdown_policy_v1")
TIMING = importlib.import_module("vision_timing_adapter_v1")


class _Settings:
    def __init__(self) -> None:
        self.value = False

    def get_as_bool(self, _key: str) -> bool:
        return self.value

    def set_bool(self, _key: str, value: bool) -> None:
        self.value = value


def _rewrite(path: Path, value: dict[str, Any]) -> None:
    path.unlink()
    R.write_new(path, value)


def _read_times(simulation_time: float) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for field in R.ISAAC_READ_TIMES_FIELDS:
        value[field] = 1 if field in R.ISAAC_READ_TIMES_INTEGER_FIELDS else 1.0
    value["simulationTime"] = simulation_time
    value["rationalTimeOfSimDenominator"] = 1
    return value


def _readiness(stage: Path) -> Path:
    loading = {"message": "", "files_loaded": 0, "files_total": 0,
               "streaming_busy": False}
    physics = {"simulation_time_s": 1.0, "common_step_counter": 1}
    camera = R.fixed_profile()["camera"]
    body = {
        "schema": READINESS.SCHEMA,
        "status": "ready",
        "reason": "consecutive_loading_streaming_idle",
        "bounds": {
            "max_updates": READINESS.MAX_UPDATES,
            "polling_elapsed_budget_s": READINESS.TIMEOUT_S,
            "polling_elapsed_budget_scope": (
                "checked_between_synchronous_app_updates_not_a_hard_watchdog"
            ),
            "required_consecutive_idle_updates": READINESS.REQUIRED_IDLE_UPDATES,
        },
        "updates": 3,
        "elapsed_s": 1.0,
        "event_counts": {"loading": 0, "loaded": 0, "aborted": 0,
                         "synthetic_loading": 0},
        "event_history": [],
        "loading_history": [
            {"update": index, "loading": loading,
             "event_counts": {"loading": 0, "loaded": 0, "aborted": 0, "synthetic_loading": 0},
             "pending_asset_batches": 0,
             "simulation_update_control": {"setting": READINESS.PLAYER_SETTING,
                                           "before": True, "during": False, "after": True}}
            for index in range(1, 4)
        ],
        "before_loading": loading,
        "final_loading": loading,
        "physics_before": physics,
        "physics_after": physics,
        "renderer": {
            "attached_hydra_engine_names": ["rtx"],
            "carb_settings": {key: None for key in READINESS.CARB_SETTING_KEYS},
            "expected_capture_profile": {
                key: camera[key]
                for key in ("capture_primitive", "rt_subframes", "pause_timeline", "delta_time_s")
            },
            "renderer_accumulation_reset": E.RESET_METHOD,
        },
    }
    path = stage / "job.capture-readiness.json"
    R.write_new(path, body)
    return path


def _stage(tmp_path: Path, *, mode: str = R.RENDER_SYNC_SMOKE_MODE) -> tuple[Path, Path, Path]:
    stage = tmp_path / "stage"
    output = stage / "output"
    output.mkdir(parents=True)
    job = stage / "job.json"
    R.write_new(job, {"mode": mode})
    _readiness(stage)
    count = 4 if mode == R.RENDER_SYNC_SMOKE_MODE else R.FRAME_COUNT
    result_rows = []
    for ordinal in range(count):
        simulation_time = float(ordinal + 1)
        raw_times = _read_times(simulation_time)
        reference = {"referenceTimeNumerator": ordinal + 1,
                     "referenceTimeDenominator": 1}
        normalized = TIMING.normalize_render_timing(raw_times, reference, simulation_time)
        rgb = f"rgb-{ordinal}".encode()
        rgb_sha = hashlib.sha256(rgb).hexdigest()
        TIMING.write_capture_journal(
            job, ordinal, raw_times, reference, rgb_sha,
            rgb_bytes=rgb if mode == R.RENDER_SYNC_SMOKE_MODE else None,
        )
        R.write_new(
            stage / f"job.capture-{ordinal:06d}.renderer-reset.json",
            {"schema": E.RESET_SCHEMA, "capture_ordinal": ordinal,
             "completed": True, "method": E.RESET_METHOD},
        )
        if mode == R.RENDER_SYNC_SMOKE_MODE:
            result_rows.append({"label": ("warmup", "A", "B", "A")[ordinal],
                                "rgb_sha256": rgb_sha, "simulation_time_s": simulation_time,
                                **normalized, "camera": {}})
        else:
            result_rows.append({"frame_index": ordinal, "capture_ordinal": ordinal,
                                "state": {"simulation_time_s": simulation_time},
                                "render_timing": normalized, "rgb_sha256": rgb_sha})
    if mode == R.RENDER_SYNC_SMOKE_MODE:
        artifact = output / "render_sync_smoke.json"
        R.write_new(artifact, {"schema": "vision_render_sync_smoke_v1", "status": "pass",
                               "captures": result_rows})
        logical_id = "render_sync_smoke"
    else:
        artifact = output / "raw_trace.json"
        R.write_new(artifact, {"schema": R.TRACE_SCHEMA, "frame_count": count,
                               "records": result_rows})
        logical_id = "trace"
    result = output / "result.json"
    R.write_new(result, {"artifacts": {logical_id: {
        "file": artifact.name, "sha256": R.sha256_file(artifact),
        "bytes": artifact.stat().st_size,
    }}})
    SHUTDOWN.apply(
        {"job_path": job, "job_sha256": R.sha256_file(job), "output_directory": output},
        _Settings(), startup_fast_shutdown=False,
    )
    return stage, job, result


def test_write_verify_reuse_and_relocation(tmp_path: Path) -> None:
    stage, job, result = _stage(tmp_path)
    manifest = E.write_manifest(job, result, sdk_child_exit_code=0)
    first = manifest.read_bytes()
    value = E.verify_manifest(manifest, job, result)
    assert value["capture_count"] == 4
    assert value["sdk_child_exit_code"] == 0
    assert E.write_manifest(job, result, sdk_child_exit_code=0).read_bytes() == first

    relocated = tmp_path / "relocated"
    shutil.copytree(stage, relocated)
    moved = E.verify_manifest(
        relocated / E.MANIFEST_NAME, relocated / "job.json", relocated / "output/result.json"
    )
    assert moved == value


def test_record_count_and_timing_trace_binding(tmp_path: Path) -> None:
    _stage_dir, job, result = _stage(tmp_path, mode=R.RECORD_MODE)
    value = E.verify_manifest(
        E.write_manifest(job, result, sdk_child_exit_code=0), job, result
    )
    assert value["capture_count"] == R.FRAME_COUNT
    assert all(row["rgb"] is None for row in value["captures"])


def test_manifest_tamper_count_and_path_escape_are_rejected(tmp_path: Path) -> None:
    _stage_dir, job, result = _stage(tmp_path)
    manifest = E.write_manifest(job, result, sdk_child_exit_code=0)
    body = R.read_object(manifest)
    body["capture_count"] = 3
    _rewrite(manifest, body)
    with pytest.raises(ValueError):
        E.verify_manifest(manifest, job, result)

    body["capture_count"] = 4
    body["readiness"]["file"] = "../escape.json"
    _rewrite(manifest, body)
    with pytest.raises(ValueError):
        E.verify_manifest(manifest, job, result)


@pytest.mark.parametrize("mutation", ["missing", "reset_ordinal", "abort", "physics", "policy"])
def test_source_failures_are_rejected(tmp_path: Path, mutation: str) -> None:
    stage, job, result = _stage(tmp_path)
    if mutation == "missing":
        (stage / "job.capture-000003.timing-raw.json").unlink()
    elif mutation == "reset_ordinal":
        path = stage / "job.capture-000002.renderer-reset.json"
        body = R.read_object(path)
        body["capture_ordinal"] = 1
        _rewrite(path, body)
    elif mutation == "abort":
        path = stage / "job.capture-readiness.json"
        body = R.read_object(path)
        body["event_counts"]["aborted"] = 1
        _rewrite(path, body)
    elif mutation == "physics":
        path = stage / "job.capture-readiness.json"
        body = R.read_object(path)
        body["physics_after"] = {"simulation_time_s": 2.0}
        _rewrite(path, body)
    else:
        (stage / SHUTDOWN.SIDECAR).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        E.write_manifest(job, result, sdk_child_exit_code=0)


def test_duplicate_ordinal_and_timing_report_mismatch_are_rejected(tmp_path: Path) -> None:
    stage, job, result = _stage(tmp_path)
    shutil.copyfile(
        stage / "job.capture-000000.timing-raw.json",
        stage / "job.capture-000004.timing-raw.json",
    )
    with pytest.raises(ValueError, match="exact set"):
        E.write_manifest(job, result, sdk_child_exit_code=0)
    (stage / "job.capture-000004.timing-raw.json").unlink()

    report = stage / "output/render_sync_smoke.json"
    body = R.read_object(report)
    body["captures"][0]["read_times"]["simulationTime"] = 99.0
    _rewrite(report, body)
    result_body = R.read_object(result)
    result_body["artifacts"]["render_sync_smoke"] = {
        "file": report.name, "sha256": R.sha256_file(report), "bytes": report.stat().st_size,
    }
    _rewrite(result, result_body)
    policy = stage / SHUTDOWN.SIDECAR
    policy_body = R.read_object(policy)
    policy_body["terminal_evidence"]["sha256"] = R.sha256_file(result)
    _rewrite(policy, policy_body)
    with pytest.raises(ValueError, match="timing raw"):
        E.write_manifest(job, result, sdk_child_exit_code=0)


def test_wrong_mode_and_nonzero_child_are_rejected(tmp_path: Path) -> None:
    _stage_dir, job, result = _stage(tmp_path)
    with pytest.raises(ValueError, match="exit code"):
        E.write_manifest(job, result, sdk_child_exit_code=1)
    _rewrite(job, {"mode": "unknown"})
    with pytest.raises(ValueError, match="job mode"):
        E.write_manifest(job, result, sdk_child_exit_code=0)


@pytest.mark.parametrize("mutation", ["execution_port", "count_type", "history", "engine", "update_control", "extra_file", "extra_directory"])
def test_auxiliary_contradictions_are_rejected(tmp_path: Path, mutation: str) -> None:
    stage, job, result = _stage(tmp_path)
    if mutation == "execution_port":
        path = stage / "job.capture-000000.timing-raw.json"
        value = R.read_object(path)
        value["execution_port"] = {"field": "wrong", "present": False,
                                   "used_as_timing": True, "raw": "contradiction"}
        _rewrite(path, value)
    elif mutation in {"count_type", "history", "engine", "update_control"}:
        path = stage / "job.capture-readiness.json"
        value = R.read_object(path)
        if mutation == "count_type":
            value["event_counts"].update(loading="same", loaded="same")
        elif mutation == "history":
            value["loading_history"][0]["event_counts"]["loading"] = 1
        elif mutation == "engine":
            value["renderer"]["attached_hydra_engine_names"] = [42]
        else:
            value["loading_history"][0]["simulation_update_control"]["during"] = True
        _rewrite(path, value)
    elif mutation == "extra_file":
        shutil.copyfile(stage / "job.capture-000000.timing-raw.json",
                        stage / "job.capture-0000000.timing-raw.json")
    else:
        (stage / "job.capture-000000.extra").mkdir()
    with pytest.raises(ValueError):
        E.write_manifest(job, result, sdk_child_exit_code=0)


def test_execution_port_raw_matches_present_read_times(tmp_path: Path) -> None:
    stage, job, result = _stage(tmp_path)
    path = stage / "job.capture-000000.timing-raw.json"
    value = R.read_object(path)
    scalar = {"type": "builtins.int", "encoding": "json_native", "value": 1}
    value["read_times"]["fields"][TIMING.EXECUTION_PORT] = scalar
    value["observed_read_times_key_count"] += 1
    value["execution_port"].update(present=True, raw=scalar)
    _rewrite(path, value)
    manifest = E.write_manifest(job, result, sdk_child_exit_code=0)
    assert E.verify_manifest(manifest, job, result)["completed"] is True
    value["execution_port"]["raw"] = {**scalar, "value": 2}
    _rewrite(path, value)
    manifest.unlink()
    with pytest.raises(ValueError, match="execution port"):
        E.write_manifest(job, result, sdk_child_exit_code=0)


def test_hardlink_and_symlink_are_rejected(tmp_path: Path) -> None:
    stage, job, result = _stage(tmp_path)
    manifest = E.write_manifest(job, result, sdk_child_exit_code=0)
    readiness = stage / "job.capture-readiness.json"
    try:
        os.link(readiness, stage / "readiness-hardlink.json")
    except OSError:
        pytest.skip("현재 파일시스템은 hardlink 생성을 지원하지 않는다")
    with pytest.raises(ValueError, match="단일 regular"):
        E.verify_manifest(manifest, job, result)

    (stage / "readiness-hardlink.json").unlink()
    manifest.unlink()
    try:
        manifest.symlink_to(stage / "job.json")
    except OSError:
        pytest.skip("현재 파일시스템은 symlink 생성을 지원하지 않는다")
    with pytest.raises(ValueError, match="regular"):
        E.verify_manifest(manifest, job, result)
