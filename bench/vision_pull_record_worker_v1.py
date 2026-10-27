"""Writer-free native pull factory를 고정 Play 녹화와 결합한다."""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as RECORD
import run_vision_prepared_record_v1 as RECORD_VERIFY
import vision_play_worker_v1 as PLAY
import vision_pull_record_capture_v1 as CAPTURE
import vision_pull_record_job_v1 as JOB
import vision_replicator_startup_v1 as STARTUP
import vision_semantic_sdk_v1 as SDK

SCHEMA = "vision_pull_record_worker_v1"
MANIFEST_SCHEMA = "vision_pull_record_native_capture_manifest_v1"
MANIFEST_NAME = "native_capture_manifest.json"
REPORT_NAME = "native-policy-record.report.json"


def verify_startup(value: Any) -> dict[str, Any]:
    checked = SDK._startup_readback(value)
    after = checked["after"]
    if after.get("orchestrator_enabled") is not True or after.get("capture_on_play") is not False:
        raise ValueError("녹화 앱 startup readback이 고정 설정과 다르다")
    return checked


def prepare_record_app(app: Any, *, dependencies: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """별도 녹화 앱의 설정을 환경 생성 전에 고정하고 readback을 보존한다."""
    deps = {
        "enable_extension": SDK._sdk_enable_extension,
        "settings": STARTUP._load_settings,
        "replicator": STARTUP._load_replicator,
    }
    if dependencies is not None:
        deps.update(dependencies)
    if deps["enable_extension"](SDK.EXTENSION) is not True:
        raise RuntimeError("녹화 앱 isaacsim.core.nodes 활성화 실패")
    app.update()
    return verify_startup(STARTUP.configure_replicator_startup(
        settings=deps["settings"](), replicator=deps["replicator"](),
    ))


class PullRecordFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(f"native pull record failed: {type(primary).__name__}: {primary}")
        self.report = report
        self.primary = primary


def _descriptor(path: Path) -> dict[str, Any]:
    target = JOB._regular(path.absolute(), "pull record artifact")
    return {
        "file": target.name,
        "sha256": JOB._sha(target),
        "bytes": target.stat().st_size,
    }


def _artifacts(validated: Mapping[str, Any], artifacts: Mapping[str, Path]) -> dict[str, Any]:
    if set(artifacts) != RECORD_VERIFY.ARTIFACT_KEYS:
        raise ValueError("native pull record Play artifact 목록이 다르다")
    if not RECORD_VERIFY._record_artifacts(validated, artifacts):
        raise ValueError("native pull record Play trace/PTS/CP 결합이 다르다")
    return {key: _descriptor(path) for key, path in artifacts.items()}


def _receipt_paths(stage: Path, ordinal: int) -> tuple[Path, Path, Path]:
    stem = f"job.pull-record-{ordinal:06d}"
    return stage / f"{stem}.npz", stage / f"{stem}.raw.json", stage / f"{stem}.json"


def _manifest(validated: Mapping[str, Any], artifacts: Mapping[str, Path]) -> dict[str, Any]:
    stage = Path(validated["job_path"]).parent
    trace = RECORD.read_object(artifacts["raw_trace"])
    records = trace.get("records")
    if not isinstance(records, list) or len(records) != RECORD.FRAME_COUNT:
        raise ValueError("native pull record trace frame 수가 다르다")
    rows: list[dict[str, Any]] = []
    previous_frame: int | None = None
    previous_fraction: tuple[int, int] | None = None
    for ordinal, record in enumerate(records):
        npz_path, raw_path, receipt_path = _receipt_paths(stage, ordinal)
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
            or receipt.get("raw_npz") != _descriptor(npz_path)
            or receipt.get("raw_receipt") != _descriptor(raw_path)
            or raw.get("schema") != CAPTURE.RAW_SCHEMA
            or raw.get("capture_ordinal") != ordinal
            or raw.get("npz") != _descriptor(npz_path)
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
        timing = record.get("render_timing")
        if (
            not isinstance(timing, Mapping)
            or timing.get("read_times") != raw.get("read_times")
            or timing.get("reference_time") != raw.get("reference_time")
        ):
            raise ValueError("native pull actual timing 원문과 Play trace가 다르다")
        state = record.get("state")
        simulation_time = state.get("simulation_time_s") if isinstance(state, Mapping) else None
        read_times = raw.get("read_times")
        if (
            not isinstance(simulation_time, (int, float))
            or not math.isfinite(float(simulation_time))
            or not isinstance(read_times, Mapping)
            or read_times.get("simulationTime") != simulation_time
        ):
            raise ValueError("native pull IsaacReadTimes와 physics time이 다르다")
        rows.append({
            "ordinal": ordinal,
            "frame_number": frame,
            "rational_time": fraction,
            "npz": _descriptor(npz_path),
            "raw_receipt": _descriptor(raw_path),
            "pull_receipt": _descriptor(receipt_path),
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
    return {
        "schema": SCHEMA,
        "completed": True,
        "technical_recording_complete": True,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "checkpoint_logical_id": validated["job"]["checkpoint"]["logical_id"],
        "frame_count": RECORD.FRAME_COUNT,
        "fps": RECORD.FPS,
        "artifacts": _artifacts(validated, artifacts),
        "native_capture_manifest": _descriptor(manifest_path),
        "writer_calls": {"create": 0, "attach": 0, "callback": 0, "detach": 0},
        "atomicity_claimed": False,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
        "training_actions": 0,
        "model_api_requests": 0,
    }


def run_policy_record(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    run_real: Callable[..., tuple[dict[str, Path], Path]] = PLAY.run_real,
    capture_factory: Callable[..., PLAY.CameraCaptureFactory] = CAPTURE.make_capture_factory,
    prepare_app: Callable[[Any], dict[str, Any]] = prepare_record_app,
) -> dict[str, Any]:
    """625-frame Play를 native pull capture로 실행하고 close 전 보고서를 봉인한다."""
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
        except Exception as seal_error:  # noqa: BLE001 - 원래 capture 실패를 유지한다.
            traceback.print_exception(seal_error)
        raise PullRecordFailure(report, error) from error
