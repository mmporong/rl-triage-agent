from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

R = importlib.import_module("record_vision_evidence_v1")
READINESS = importlib.import_module("vision_capture_readiness_v1")
E1 = importlib.import_module("vision_execution_evidence_v1")
E2 = importlib.import_module("vision_execution_evidence_v2")
FRAME = importlib.import_module("vision_frame_sync_v1")
PAYLOAD = importlib.import_module("vision_writer_payload_v1")
SHUTDOWN = importlib.import_module("vision_shutdown_policy_v1")
SMOKE = importlib.import_module("vision_atomic_smoke_v1")
TIMING = importlib.import_module("vision_timing_adapter_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 5)
    monkeypatch.setattr(R, "HEIGHT", 1)


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


def _descriptor(path: Path) -> dict[str, Any]:
    return {"file": path.name, "sha256": R.sha256_file(path), "bytes": path.stat().st_size}


def _readiness(stage: Path) -> None:
    loading = {"message": "", "files_loaded": 0, "files_total": 0, "streaming_busy": False}
    physics = {"simulation_time_s": 2.0, "common_step_counter": 10}
    camera = R.fixed_profile()["camera"]
    counts = {"loading": 0, "loaded": 0, "aborted": 0, "synthetic_loading": 0}
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
        "event_counts": counts,
        "event_history": [],
        "loading_history": [
            {
                "update": index,
                "loading": loading,
                "event_counts": counts,
                "pending_asset_batches": 0,
                "simulation_update_control": {
                    "setting": READINESS.PLAYER_SETTING,
                    "before": True,
                    "during": False,
                    "after": True,
                },
            }
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
            "renderer_accumulation_reset": E1.RESET_METHOD,
        },
    }
    R.write_new(stage / "job.capture-readiness.json", body)


def _read_times(ordinal: int, simulation_time: float, mode: str) -> tuple[dict[str, Any], dict[str, int]]:
    frame = (55 if mode == R.RENDER_SYNC_SMOKE_MODE else 1) + ordinal
    denominator = 30
    value: dict[str, Any] = {}
    for field in R.ISAAC_READ_TIMES_FIELDS:
        value[field] = 0 if field in R.ISAAC_READ_TIMES_INTEGER_FIELDS else 0.0
    value.update(
        frameNumber=frame,
        swhFrameNumber=0,
        simulationTime=simulation_time,
        rationalTimeOfSimNumerator=frame,
        rationalTimeOfSimDenominator=denominator,
    )
    return value, {
        "referenceTimeNumerator": frame,
        "referenceTimeDenominator": denominator,
    }


def _smoke_pixels(code: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    labels = sorted(FRAME.marker_labels(code))
    mask = np.arange(1, 6, dtype=np.uint32).reshape(1, 5)
    rgb = np.zeros((1, 5, 3), dtype=np.uint8)
    for index, label in enumerate(labels):
        role = "sentinel" if label == FRAME.SENTINEL else label.rsplit("/", 1)[1]
        rgb[0, index] = {
            "sentinel": [10, 40, 30],
            "active": [40, 10, 30],
            "inactive": [10, 30, 40],
        }[role]
    return rgb, mask, {str(index + 1): label for index, label in enumerate(labels)}


def _decoded(
    ordinal: int, mode: str, simulation_time: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], np.ndarray]:
    smoke = mode == R.RENDER_SYNC_SMOKE_MODE
    if smoke:
        rgb, mask, labels = _smoke_pixels(ordinal)
    else:
        rgb = np.tile(
            np.array([[[ordinal % 251, 7, 9]]], dtype=np.uint8), (1, 5, 1)
        )
        mask = np.zeros((1, 5), dtype=np.uint32)
        labels = {}
    read_times, reference = _read_times(ordinal, simulation_time, mode)
    view = np.eye(4, dtype=np.float64)
    if smoke and ordinal == 2:
        view[0, 3] = 1.0
    projection = np.eye(4, dtype=np.float64)
    callback_ref = f"callback-{mode}-{ordinal}"
    camera_raw = {
        "cameraViewTransform": view.reshape(-1),
        "cameraProjection": projection.reshape(-1),
        "renderProductResolution": np.array([rgb.shape[1], rgb.shape[0]], dtype=np.int64),
    }
    decoded = {
        "callback_ref": callback_ref,
        "render_product_ref": "render-product-0",
        "rgb": rgb,
        "semantic": {"mask": mask, "idToLabels": labels},
        "camera": {
            "view_matrix": view.tolist(),
            "projection": projection.tolist(),
            "resolution": [rgb.shape[1], rgb.shape[0]],
        },
        "camera_params_raw": camera_raw,
        "camera_path": R.fixed_profile()["camera"]["camera_prim_path"],
        "render_product_resolution": [rgb.shape[1], rgb.shape[0]],
        "writer_swh_frame_number": 0,
        "read_times": read_times,
        "read_times_raw": TIMING._typed_mapping(read_times, "IsaacReadTimes"),
        "reference_time": reference,
        "reference_time_raw": TIMING._typed_mapping(reference, "ReferenceTime"),
        "writer_reference_time": [reference["referenceTimeNumerator"], reference["referenceTimeDenominator"]],
    }
    return decoded, read_times, reference, rgb


def _reset_and_timing(
    job: Path,
    ordinal: int,
    read_times: dict[str, Any],
    reference: dict[str, int],
    rgb: np.ndarray,
    mode: str,
) -> None:
    rgb_bytes = rgb.tobytes(order="C")
    rgb_sha = hashlib.sha256(rgb_bytes).hexdigest()
    TIMING.write_capture_journal(
        job,
        ordinal,
        read_times,
        reference,
        rgb_sha,
        rgb_bytes=rgb_bytes if mode == R.RENDER_SYNC_SMOKE_MODE else None,
    )
    R.write_new(
        job.with_name(f"job.capture-{ordinal:06d}.renderer-reset.json"),
        {
            "schema": E1.RESET_SCHEMA,
            "capture_ordinal": ordinal,
            "completed": True,
            "method": E1.RESET_METHOD,
        },
    )


def _stage(tmp_path: Path, mode: str) -> tuple[Path, Path, Path]:
    stage = tmp_path / "stage"
    output = stage / "output"
    output.mkdir(parents=True)
    job = stage / "job.json"
    R.write_new(job, {"mode": mode})
    _readiness(stage)
    count = 4 if mode == R.RENDER_SYNC_SMOKE_MODE else R.FRAME_COUNT
    writer = FRAME.AtomicFrameWriter(mode, count)
    smoke_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    smoke_state = {
        "command": [1.0, 0.0, 0.0],
        "motion": [0.0, 0.0, 0.0],
        "pose": {"base_height_m": 0.3, "up_axis_z": 1.0},
        "root_position_world_m": [0.0, 0.0, 0.0],
        "common_step_counter": 10,
    }
    audit_bytes: dict[int, bytes] = {}
    for ordinal in range(count):
        simulation_time = 2.0 if mode == R.RENDER_SYNC_SMOKE_MODE else float(ordinal + 1)
        state = smoke_state if mode == R.RENDER_SYNC_SMOKE_MODE else {"step": ordinal}
        decoded, read_times, reference, rgb = _decoded(ordinal, mode, simulation_time)
        writer.arm(ordinal, ordinal if mode == R.RENDER_SYNC_SMOKE_MODE else None, state, simulation_time)
        writer.before_step(ordinal)
        writer.callback(PAYLOAD.frame_payload(decoded))
        bundle = writer.close(ordinal, state, simulation_time)
        PAYLOAD.write_atomic_journal(job, ordinal, mode, decoded, bundle)
        _reset_and_timing(job, ordinal, read_times, reference, rgb, mode)
        if mode == R.RENDER_SYNC_SMOKE_MODE:
            offset = [-3.0, 3.0, 2.0] if ordinal == 2 else [3.0, 3.0, 2.0]
            smoke_rows.append(
                {
                    "label": SMOKE.ORDER[ordinal],
                    "state_before": smoke_state,
                    "state_after": smoke_state,
                    "simulation_time_before_s": simulation_time,
                    "simulation_time_after_s": simulation_time,
                    "camera": {"eye_world_m": offset, "lookat_world_m": [0.0, 0.0, 0.35]},
                    "atomic_bundle": bundle,
                }
            )
            audit_bytes[ordinal] = rgb.tobytes(order="C")
        else:
            trace_rows.append(
                {
                    "frame_index": ordinal,
                    "capture_ordinal": ordinal,
                    "state": {"simulation_time_s": simulation_time},
                    "render_timing": bundle["timing"],
                    "rgb_sha256": bundle["rgb"]["sha256"],
                }
            )
    writer.finish()
    artifacts: dict[str, dict[str, Any]] = {}
    if mode == R.RENDER_SYNC_SMOKE_MODE:
        primary = output / "render_sync_smoke.json"
        R.write_new(primary, SMOKE._assess(smoke_rows, job))
        artifacts["render_sync_smoke"] = _descriptor(primary)
        for ordinal in (1, 2, 3):
            audit = output / f"audit_{ordinal}_{SMOKE.ORDER[ordinal]}.rgb"
            audit.write_bytes(audit_bytes[ordinal])
            artifacts[f"audit_rgb_{ordinal}"] = _descriptor(audit)
    else:
        primary = output / "raw_trace.json"
        R.write_new(
            primary,
            {"schema": R.TRACE_SCHEMA, "frame_count": count, "records": trace_rows},
        )
        artifacts["trace"] = _descriptor(primary)
    result = output / "result.json"
    R.write_new(result, {"artifacts": artifacts})
    SHUTDOWN.apply(
        {"job_path": job, "job_sha256": R.sha256_file(job), "output_directory": output},
        _Settings(),
        startup_fast_shutdown=False,
    )
    return stage, job, result


def _refresh_result(stage: Path, result: Path, logical_id: str, artifact: Path) -> None:
    result_body = R.read_object(result)
    result_body["artifacts"][logical_id] = _descriptor(artifact)
    _rewrite(result, result_body)
    policy = stage / SHUTDOWN.SIDECAR
    policy_body = R.read_object(policy)
    policy_body["terminal_evidence"]["sha256"] = R.sha256_file(result)
    _rewrite(policy, policy_body)


def test_smoke_manifest_seals_atomic_sources_reuses_and_relocates(tmp_path: Path) -> None:
    stage, job, result = _stage(tmp_path, R.RENDER_SYNC_SMOKE_MODE)
    manifest = E2.write_manifest(job, result, sdk_child_exit_code=0)
    original = manifest.read_bytes()
    value = E2.verify_manifest(manifest, job, result)
    assert value["schema"] == E2.MANIFEST_SCHEMA
    assert len(value["captures"]) == 4
    assert len(value["result_evidence"]["audits"]) == 3
    assert E2.write_manifest(job, result, sdk_child_exit_code=0).read_bytes() == original

    relocated = tmp_path / "relocated"
    shutil.copytree(stage, relocated)
    assert E2.verify_manifest(
        relocated / E2.MANIFEST_NAME,
        relocated / "job.json",
        relocated / "output/result.json",
    ) == value


def test_record_manifest_seals_all_625_atomic_frames(tmp_path: Path) -> None:
    _stage_dir, job, result = _stage(tmp_path, R.RECORD_MODE)
    value = E2.verify_manifest(
        E2.write_manifest(job, result, sdk_child_exit_code=0), job, result
    )
    assert value["capture_count"] == R.FRAME_COUNT
    assert len(value["captures"]) == R.FRAME_COUNT
    assert all(row["rgb"] is None for row in value["captures"])


@pytest.mark.parametrize(
    "mutation",
    ["npz", "journal_writer", "atomic_extra", "readiness", "reset", "timing", "shutdown"],
)
def test_atomic_and_v1_source_corruption_is_rejected(tmp_path: Path, mutation: str) -> None:
    stage, job, result = _stage(tmp_path, R.RENDER_SYNC_SMOKE_MODE)
    if mutation == "npz":
        with (stage / "job.atomic-000001.npz").open("ab") as stream:
            stream.write(b"tamper")
    elif mutation == "journal_writer":
        path = stage / "job.atomic-000001.json"
        body = R.read_object(path)
        body["writer_payload"]["callback_ref"] = "other-callback"
        _rewrite(path, body)
    elif mutation == "atomic_extra":
        (stage / "job.atomic-999999.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "readiness":
        path = stage / "job.capture-readiness.json"
        body = R.read_object(path)
        body["event_counts"]["aborted"] = 1
        _rewrite(path, body)
    elif mutation == "reset":
        path = stage / "job.capture-000001.renderer-reset.json"
        body = R.read_object(path)
        body["capture_ordinal"] = 2
        _rewrite(path, body)
    elif mutation == "timing":
        path = stage / "job.capture-000001.timing-raw.json"
        body = R.read_object(path)
        body["rgb_sha256"] = "0" * 64
        _rewrite(path, body)
    else:
        (stage / SHUTDOWN.SIDECAR).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        E2.write_manifest(job, result, sdk_child_exit_code=0)


@pytest.mark.parametrize("mutation", ["report_bundle", "audit", "record_time"])
def test_result_cross_binding_corruption_is_rejected(tmp_path: Path, mutation: str) -> None:
    mode = R.RECORD_MODE if mutation == "record_time" else R.RENDER_SYNC_SMOKE_MODE
    stage, job, result = _stage(tmp_path, mode)
    if mutation == "report_bundle":
        report = stage / "output/render_sync_smoke.json"
        body = R.read_object(report)
        body["captures"][1]["atomic_bundle"]["adapter_callback_ref"] = "replayed"
        _rewrite(report, body)
        _refresh_result(stage, result, "render_sync_smoke", report)
    elif mutation == "audit":
        audit = stage / "output/audit_1_A.rgb"
        audit.write_bytes(b"tamper")
        _refresh_result(stage, result, "audit_rgb_1", audit)
    else:
        trace = stage / "output/raw_trace.json"
        body = R.read_object(trace)
        body["records"][0]["state"]["simulation_time_s"] = 99.0
        _rewrite(trace, body)
        _refresh_result(stage, result, "trace", trace)
    with pytest.raises(ValueError):
        E2.write_manifest(job, result, sdk_child_exit_code=0)


def test_manifest_tamper_path_escape_nonzero_and_hardlink_are_rejected(tmp_path: Path) -> None:
    stage, job, result = _stage(tmp_path, R.RENDER_SYNC_SMOKE_MODE)
    with pytest.raises(ValueError, match="exit code"):
        E2.write_manifest(job, result, sdk_child_exit_code=1)
    manifest = E2.write_manifest(job, result, sdk_child_exit_code=0)
    body = R.read_object(manifest)
    body["captures"][0]["atomic_raw"]["file"] = "../escape.npz"
    _rewrite(manifest, body)
    with pytest.raises(ValueError):
        E2.verify_manifest(manifest, job, result)

    _rewrite(manifest, E2._build(job, result, 0))
    atomic = stage / "job.atomic-000000.npz"
    try:
        os.link(atomic, stage / "atomic-hardlink.npz")
    except OSError:
        pytest.skip("현재 파일시스템은 hardlink 생성을 지원하지 않는다")
    with pytest.raises(ValueError, match="단일 regular|hardlink"):
        E2.verify_manifest(manifest, job, result)
