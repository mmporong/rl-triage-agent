"""build trace 5키와 별도 primary video를 검증된 8키 artifact로 바꾼다."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import vision_pull_record_worker_v2 as PREVIOUS

CAPTURE = PREVIOUS.CAPTURE
RECORD = PREVIOUS.RECORD
PLAY = PREVIOUS.PLAY
JOB = PREVIOUS.JOB
SCHEMA = "vision_pull_record_worker_v3"
MANIFEST_SCHEMA = PREVIOUS.MANIFEST_SCHEMA
MANIFEST_NAME = PREVIOUS.MANIFEST_NAME
REPORT_NAME = PREVIOUS.REPORT_NAME
PullRecordFailure = PREVIOUS.PullRecordFailure
verify_startup = PREVIOUS.verify_startup
prepare_record_app = PREVIOUS.prepare_record_app
_manifest = PREVIOUS._manifest


PRODUCED_KEYS = {
    "trace", "measurement_1", "spec_1", "measurement_2", "spec_2"
}


def _normalized_artifacts(
    validated: Mapping[str, Any], produced: Mapping[str, Path], video: Path
) -> dict[str, Path]:
    """build_trace의 5개 출력과 primary video를 동결 8개 계약으로 바꾼다."""
    if not isinstance(produced, Mapping) or set(produced) != PRODUCED_KEYS:
        raise ValueError("pull record build trace artifact 목록이 다르다")
    output = JOB.COMMON._directory(
        Path(validated["output_directory"]).absolute(), "pull record output"
    )
    primary = JOB._regular(Path(video).absolute(), "pull record primary video")
    selected = {
        key: JOB._regular(Path(path).absolute(), f"pull record {key}")
        for key, path in produced.items()
    }
    if any(output not in path.parents for path in [primary, *selected.values()]):
        raise ValueError("pull record artifact가 job output 경계를 벗어났다")
    trace = RECORD.read_object(selected["trace"])
    encoder = trace.get("encoder_receipt")
    timing = trace.get("timing_receipt")
    video_sha = JOB._sha(primary)
    if (
        not isinstance(encoder, Mapping)
        or not isinstance(timing, Mapping)
        or trace.get("video_sha256") != video_sha
        or timing.get("source_sha256") != video_sha
    ):
        raise ValueError("pull record trace receipt와 primary video 결합이 다르다")
    encoder_path = selected["trace"].parent / "encoder_receipt.json"
    timing_path = selected["trace"].parent / "timing_receipt.json"
    if any(path.exists() or path.is_symlink() for path in (encoder_path, timing_path)):
        raise FileExistsError("pull record 파생 receipt가 이미 있다")
    encoder_created = False
    timing_created = False
    try:
        JOB.write_new(encoder_path, dict(encoder))
        encoder_created = True
        JOB.write_new(timing_path, dict(timing))
        timing_created = True
        artifacts = {
            "video": primary,
            "raw_trace": selected["trace"],
            "encoder_receipt": encoder_path,
            "timing_receipt": timing_path,
            "measurement_1": selected["measurement_1"],
            "spec_1": selected["spec_1"],
            "measurement_2": selected["measurement_2"],
            "spec_2": selected["spec_2"],
        }
        if not PREVIOUS.RECORD_VERIFY._record_artifacts(validated, artifacts):
            raise ValueError("pull record 8개 artifact의 trace/PTS/receipt 결합이 다르다")
        return artifacts
    except BaseException:
        if timing_created:
            timing_path.unlink(missing_ok=True)
        if encoder_created:
            encoder_path.unlink(missing_ok=True)
        raise


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
    """v2 timing·manifest를 유지하고 복사된 video의 내용 결합을 검증한다."""
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
        produced, video = run_real(
            validated, app, camera_capture_factory=factory
        )
        artifacts = _normalized_artifacts(validated, produced, video)
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
