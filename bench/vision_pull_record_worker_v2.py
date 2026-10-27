"""raw 12키 timing과 정규화된 Play trace를 엄격히 결합한다."""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import vision_pull_record_capture_v2 as CAPTURE
import vision_pull_record_worker_v1 as PREVIOUS
import vision_timing_adapter_v1 as TIMING

RECORD = PREVIOUS.RECORD
RECORD_VERIFY = PREVIOUS.RECORD_VERIFY
PLAY = PREVIOUS.PLAY
JOB = PREVIOUS.JOB
SCHEMA = "vision_pull_record_worker_v2"
MANIFEST_SCHEMA = "vision_pull_record_native_capture_manifest_v2"
MANIFEST_NAME = PREVIOUS.MANIFEST_NAME
REPORT_NAME = PREVIOUS.REPORT_NAME
PullRecordFailure = PREVIOUS.PullRecordFailure
verify_startup = PREVIOUS.verify_startup
prepare_record_app = PREVIOUS.prepare_record_app


def _normalized_trace_timing(
    timing: Any, raw: Mapping[str, Any], simulation_time_s: float
) -> dict[str, Any]:
    if not isinstance(timing, Mapping):
        raise TypeError("Play trace render timing이 mapping이 아니다")
    read_times = raw.get("read_times")
    reference_time = raw.get("reference_time")
    if not isinstance(read_times, Mapping) or not isinstance(reference_time, Mapping):
        raise TypeError("raw timing receipt가 mapping이 아니다")
    expected = TIMING.normalize_render_timing(
        read_times, reference_time, simulation_time_s
    )
    observed = RECORD.validate_render_timing(timing, simulation_time_s)
    if observed != expected:
        raise ValueError("정규화된 Play trace와 Isaac timing 원문이 다르다")
    return observed


def _manifest(
    validated: Mapping[str, Any], artifacts: Mapping[str, Path]
) -> dict[str, Any]:
    stage = Path(validated["job_path"]).parent
    trace = RECORD.read_object(artifacts["raw_trace"])
    records = trace.get("records")
    if not isinstance(records, list) or len(records) != RECORD.FRAME_COUNT:
        raise ValueError("native pull record trace frame 수가 다르다")
    rows: list[dict[str, Any]] = []
    previous_frame: int | None = None
    previous_fraction: tuple[int, int] | None = None
    for ordinal, record in enumerate(records):
        npz_path, raw_path, receipt_path = PREVIOUS._receipt_paths(stage, ordinal)
        raw = RECORD.read_object(JOB._regular(raw_path, "raw receipt"))
        receipt = RECORD.read_object(JOB._regular(receipt_path, "pull receipt"))
        if (
            receipt.get("schema") != CAPTURE.SCHEMA
            or receipt.get("capture_ordinal") != ordinal
            or receipt.get("annotator_order") != list(CAPTURE.ANNOTATORS)
            or receipt.get("complete_capture") is not True
            or receipt.get("reserved_marker_count") != 0
            or receipt.get("atomicity_claimed") is not False
            or receipt.get("capture_approved") is not False
            or receipt.get("used_for_decision") is not False
            or receipt.get("raw_npz") != PREVIOUS._descriptor(npz_path)
            or receipt.get("raw_receipt") != PREVIOUS._descriptor(raw_path)
            or raw.get("schema") != CAPTURE.RAW_SCHEMA
            or raw.get("capture_ordinal") != ordinal
            or raw.get("npz") != PREVIOUS._descriptor(npz_path)
            or raw.get("raw_fields") != list(CAPTURE.ANNOTATORS)
            or raw.get("reserved_marker_count") != 0
        ):
            raise ValueError(f"native pull receipt[{ordinal}] 결합이 다르다")
        counter = receipt.get("app_update_counter")
        step = receipt.get("step")
        fraction = receipt.get("rational_time")
        frame = receipt.get("frame_number")
        if (
            not isinstance(counter, Mapping)
            or counter.get("source") != "omni.kit.app.IApp.get_update_number"
            or counter.get("fail_closed") is not True
            or counter.get("pull_window_updates") != 0
            or not isinstance(counter.get("render_window_updates"), int)
            or counter["render_window_updates"] <= 0
            or not isinstance(counter.get("after_each_pull"), list)
            or len(counter["after_each_pull"]) != 5
            or len(set(counter["after_each_pull"])) != 1
            or not isinstance(step, Mapping)
            or step.get("call_count") != 1
            or step.get("delta_time_s") != 0.0
            or receipt.get("physics_before") != receipt.get("physics_after")
            or type(frame) is not int
            or not isinstance(fraction, list)
            or len(fraction) != 2
            or any(type(value) is not int for value in fraction)
            or fraction[1] <= 0
        ):
            raise ValueError(f"native pull receipt[{ordinal}] 불변식이 다르다")
        current_fraction = (fraction[0], fraction[1])
        if previous_frame is not None and frame <= previous_frame:
            raise ValueError("native pull frame이 증가하지 않는다")
        if previous_fraction is not None and (
            current_fraction[0] * previous_fraction[1]
            <= previous_fraction[0] * current_fraction[1]
        ):
            raise ValueError("native pull rational time이 증가하지 않는다")
        if not isinstance(record, Mapping) or record.get("capture_ordinal") != ordinal:
            raise ValueError("native pull manifest와 Play trace ordinal이 다르다")
        state = record.get("state")
        simulation_time = (
            state.get("simulation_time_s") if isinstance(state, Mapping) else None
        )
        if (
            isinstance(simulation_time, bool)
            or not isinstance(simulation_time, (int, float))
            or not math.isfinite(float(simulation_time))
        ):
            raise ValueError("Play trace physics time이 수치가 아니다")
        _normalized_trace_timing(record.get("render_timing"), raw, float(simulation_time))
        rows.append({
            "ordinal": ordinal,
            "frame_number": frame,
            "rational_time": fraction,
            "npz": PREVIOUS._descriptor(npz_path),
            "raw_receipt": PREVIOUS._descriptor(raw_path),
            "pull_receipt": PREVIOUS._descriptor(receipt_path),
        })
        previous_frame, previous_fraction = frame, current_fraction
    return {
        "schema": MANIFEST_SCHEMA,
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "frame_count": RECORD.FRAME_COUNT,
        "fps": RECORD.FPS,
        "annotators": list(CAPTURE.ANNOTATORS),
        "raw_file_count": RECORD.FRAME_COUNT * 2,
        "receipt_count": RECORD.FRAME_COUNT,
        "orchestrator_step_count": RECORD.FRAME_COUNT,
        "native_pull_window_updates": 0,
        "physics_state_time_unchanged": True,
        "reference_time_is_actual_annotator": True,
        "clean_reserved_marker_count": 0,
        "atomicity_claimed": False,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
        "frames": rows,
    }


def _report(
    validated: Mapping[str, Any], artifacts: Mapping[str, Path], manifest_path: Path
) -> dict[str, Any]:
    value = PREVIOUS._report(validated, artifacts, manifest_path)
    value["schema"] = SCHEMA
    return value


def run_policy_record(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    run_real: Callable[..., tuple[dict[str, Path], Path]] = PLAY.run_real,
    capture_factory: Callable[..., PLAY.CameraCaptureFactory] = CAPTURE.make_capture_factory,
    prepare_app: Callable[[Any], dict[str, Any]] = prepare_record_app,
) -> dict[str, Any]:
    """v1 Play·종료 경계를 유지하고 timing consumer와 manifest만 v2로 결합한다."""
    if not callable(seal_report):
        raise TypeError("pull record seal_report가 callable이 아니다")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "completed": False,
        "technical_recording_complete": False,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
    }
    try:
        startup = verify_startup(prepare_app(app))
        report["startup"] = startup
        factory = capture_factory(validated, app)
        artifacts, video = run_real(
            validated, app, camera_capture_factory=factory
        )
        if artifacts.get("video") != video:
            raise ValueError("pull record primary video와 artifact가 다르다")
        manifest = _manifest(validated, artifacts)
        manifest_path = Path(validated["output_directory"]) / MANIFEST_NAME
        JOB.write_new(manifest_path, manifest)
        report = _report(validated, artifacts, manifest_path)
        report["startup"] = startup
        seal_report(report)
        return report
    except BaseException as error:
        report["error_type"] = f"{type(error).__module__}.{type(error).__qualname__}"
        report["error_message"] = str(error)
        try:
            seal_report(report)
        except Exception as seal_error:  # noqa: BLE001
            traceback.print_exception(seal_error)
        raise PullRecordFailure(report, error) from error
