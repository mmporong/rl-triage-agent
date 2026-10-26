"""준비 callback 뒤 625 정책 frame의 소유 실행·종료 증거를 검증한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import run_vision_prepared_capture_v1 as BASE  # pyright: ignore[reportMissingImports]
import vision_brev_backend_v1 as OWNED  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v2 as GATE  # pyright: ignore[reportMissingImports]
import vision_prepared_record_v1 as PREPARED  # pyright: ignore[reportMissingImports]
import vision_record_job_v1 as JOB  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]
import vision_shutdown_policy_v1 as SHUTDOWN  # pyright: ignore[reportMissingImports]

REPORT_NAME = "prepared-record.report.json"
EXECUTION_SCHEMA = "vision_prepared_record_execution_v1"
POSTFLIGHT_RESERVE_S = 200.0
OWNED_CLEANUP_WORST_CASE_S = 40.0
ARTIFACT_KEYS = frozenset(
    {
        "video",
        "raw_trace",
        "encoder_receipt",
        "timing_receipt",
        "measurement_1",
        "spec_1",
        "measurement_2",
        "spec_2",
    }
)
ARTIFACT_LEAVES = {
    "video": "same_play.mp4",
    "raw_trace": "raw_trace.json",
    "encoder_receipt": "encoder_receipt.json",
    "timing_receipt": "timing_receipt.json",
    "measurement_1": "window_1_measurements.json",
    "spec_1": "window_1_v01_spec.json",
    "measurement_2": "window_2_measurements.json",
    "spec_2": "window_2_v01_spec.json",
}
LIFECYCLE_FLAGS = frozenset(
    {
        "startup_completed",
        "run_real_started",
        "run_real_returned",
        "writer_attach_attempted",
        "writer_attached",
        "prepare_completed",
        "arms_completed",
        "writer_detach_attempted",
        "writer_detached",
        "render_product_destroyed",
        "environment_closed",
        "mux_resolved",
        "mux_sealed",
        "envelope_sealed",
    }
)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return JOB._read(path)


def _regular(path: Path) -> Path:
    return JOB._regular(path.absolute(), "prepared record 증거")


def _descriptor(path: Path) -> dict[str, Any]:
    target = _regular(path)
    return {"file": target.name, "sha256": _sha(target), "bytes": target.stat().st_size}


def _artifact_paths(validated: Mapping[str, Any], value: Any) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != ARTIFACT_KEYS:
        raise ValueError("prepared record artifact 목록이 다르다")
    output = Path(validated["output_directory"])
    resolved: dict[str, Path] = {}
    for key in sorted(ARTIFACT_KEYS):
        descriptor = value[key]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"file", "sha256", "bytes"}
            or not isinstance(descriptor["file"], str)
            or not descriptor["file"]
            or Path(descriptor["file"]).name != descriptor["file"]
            or descriptor["file"] != ARTIFACT_LEAVES[key]
            or not isinstance(descriptor["sha256"], str)
            or len(descriptor["sha256"]) != 64
            or type(descriptor["bytes"]) is not int
            or descriptor["bytes"] <= 0
        ):
            raise ValueError(f"prepared record {key} descriptor가 다르다")
        matches = [
            path
            for path in output.rglob(descriptor["file"])
            if path.is_file() and not path.is_symlink()
        ]
        if len(matches) != 1:
            raise ValueError(f"prepared record {key} 파일이 하나가 아니다")
        path = _regular(matches[0])
        if output.resolve() not in path.parents:
            raise ValueError("prepared record artifact가 output 밖에 있다")
        if (
            _sha(path) != descriptor["sha256"]
            or path.stat().st_size != descriptor["bytes"]
        ):
            raise ValueError(f"prepared record {key} 원문이 descriptor와 다르다")
        resolved[key] = path
    return resolved


def _prepared_structure(value: Any) -> bool:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != PREPARED.SCHEMA
        or value.get("approved") is not True
        or value.get("state") != "sealed"
        or value.get("expected_count") != RECORD.FRAME_COUNT
        or value.get("committed_count") != RECORD.FRAME_COUNT
        or value.get("failure") is not None
        or value.get("callback_rows_dropped") != 0
    ):
        return False
    raw_count = value.get("raw_callback_count")
    callbacks = value.get("callbacks")
    manifests = value.get("manifests")
    preparation = value.get("preparation")
    if (
        type(raw_count) is not int
        or raw_count not in {RECORD.FRAME_COUNT + 1, RECORD.FRAME_COUNT + 2}
        or value.get("raw_callback_limit") != RECORD.FRAME_COUNT + 2
        or not isinstance(callbacks, list)
        or len(callbacks) != raw_count
        or not isinstance(manifests, list)
        or len(manifests) != RECORD.FRAME_COUNT
        or not isinstance(preparation, Mapping)
        or type(preparation.get("bootstrap_empty_observed")) is not bool
    ):
        return False
    frames: list[int] = []
    for ordinal, row in enumerate(manifests):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != ordinal
            or type(row.get("frame_number")) is not int
            or row.get("callback_ref") is None
            or not isinstance(row.get("commit_receipt"), Mapping)
        ):
            return False
        frames.append(row["frame_number"])
    return all(right > left for left, right in pairwise(frames))


def _record_artifacts(
    validated: Mapping[str, Any], artifacts: Mapping[str, Path]
) -> bool:
    trace = _read(artifacts["raw_trace"])
    encoder = _read(artifacts["encoder_receipt"])
    timing_artifact = _read(artifacts["timing_receipt"])
    video_sha = _sha(artifacts["video"])
    records = trace.get("records")
    timing = trace.get("timing_receipt")
    if (
        trace.get("schema") != RECORD.TRACE_SCHEMA
        or trace.get("profile") != RECORD.fixed_profile()
        or trace.get("profile_sha256") != RECORD.digest(RECORD.fixed_profile())
        or trace.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or trace.get("video_sha256") != video_sha
        or trace.get("frame_count") != RECORD.FRAME_COUNT
        or not isinstance(records, list)
        or len(records) != RECORD.FRAME_COUNT
        or not isinstance(timing, Mapping)
        or timing.get("source_sha256") != video_sha
        or timing.get("frame_count") != RECORD.FRAME_COUNT
        or encoder != trace.get("encoder_receipt")
        or timing_artifact != timing
    ):
        return False
    pts = timing.get("relative_actual_pts_s")
    if not isinstance(pts, list) or len(pts) != RECORD.FRAME_COUNT:
        return False
    first_step: int | None = None
    for ordinal, (record, timestamp) in enumerate(zip(records, pts, strict=True)):
        if not isinstance(record, Mapping) or not isinstance(timestamp, (int, float)):
            return False
        step = record.get("simulation_step")
        if type(step) is not int:
            return False
        if first_step is None:
            first_step = step
        expected_time = ordinal / RECORD.FPS
        if (
            record.get("frame_index") != ordinal
            or record.get("capture_ordinal") != ordinal
            or step != first_step + ordinal
            or not math.isclose(
                float(timestamp), expected_time, abs_tol=RECORD.PTS_TOLERANCE_S
            )
            or record.get("actual_timestamp_s") != timestamp
        ):
            return False
    by_timestamp = {row["actual_timestamp_s"]: row for row in records}
    for window, timestamps in enumerate(RECORD.WINDOWS, 1):
        measurement = _read(artifacts[f"measurement_{window}"])
        spec = _read(artifacts[f"spec_{window}"])
        samples = measurement.get("samples")
        if (
            measurement.get("schema") != RECORD.MEASUREMENT_SCHEMA
            or measurement.get("video_sha256") != video_sha
            or measurement.get("checkpoint_sha256") != validated["checkpoint_sha256"]
            or not isinstance(samples, list)
            or [sample.get("timestamp_s") for sample in samples] != list(timestamps)
            or any(timestamp not in by_timestamp for timestamp in timestamps)
            or spec.get("schema") != RECORD.PREPARE_SCHEMA
            or spec.get("source_lineage_id")
            != f"{validated['job']['job_id']}-same-play"
            or spec.get("parent_source_lineage_id") is not None
            or spec.get("video", {}).get("play_run_id")
            != measurement.get("play_run_id")
            or spec.get("checkpoint", {}).get("play_run_id")
            != measurement.get("play_run_id")
            or spec.get("measurement", {}).get("play_run_id")
            != measurement.get("play_run_id")
            or [frame.get("timestamp_s") for frame in spec.get("frames", [])]
            != list(timestamps)
        ):
            return False
        lowered = json.dumps(spec, ensure_ascii=False).lower()
        if any(
            token in lowered
            for token in ('"behavior_label"', '"fixed_eval_label"', '"fault_label"')
        ):
            return False
    return True


def _successful_sdk_envelope(
    validated: Mapping[str, Any], value: Mapping[str, Any], *, sealed: bool
) -> bool:
    try:
        import vision_prepared_record_sdk_v1 as SDK

        if set(value) != {
            "schema",
            "prepared_record",
            "artifacts",
            "lifecycle",
            "renderer",
            "setup_evidence",
            "record_result",
            "approved",
        }:
            return False
        if (
            value.get("schema") != SDK.SDK_REPORT_SCHEMA
            or value.get("approved") is not True
        ):
            return False
        lifecycle = value.get("lifecycle")
        if (
            not isinstance(lifecycle, Mapping)
            or set(lifecycle) != LIFECYCLE_FLAGS | {"errors"}
            or lifecycle.get("errors") != []
            or any(
                lifecycle.get(flag) is not True
                for flag in LIFECYCLE_FLAGS - {"arms_completed", "envelope_sealed"}
            )
            or lifecycle.get("arms_completed") != RECORD.FRAME_COUNT
            or lifecycle.get("envelope_sealed") is not sealed
            or not _prepared_structure(value.get("prepared_record"))
            or not isinstance(value.get("renderer"), Mapping)
            or not value["renderer"]
            or not isinstance(value.get("setup_evidence"), Mapping)
            or not value["setup_evidence"]
        ):
            return False
        paths = _artifact_paths(validated, value["artifacts"])
        record_result = value.get("record_result")
        if (
            not isinstance(record_result, Mapping)
            or set(record_result) != {"frame_count", "fps", "video"}
            or record_result.get("frame_count") != RECORD.FRAME_COUNT
            or record_result.get("fps") != RECORD.FPS
            or record_result.get("video") != value["artifacts"]["video"]
        ):
            return False
        return _record_artifacts(validated, paths)
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _terminal_body(
    validated: Mapping[str, Any], envelope: Mapping[str, Any], report: Path
) -> dict[str, Any]:
    return {
        "schema": JOB.RESULT_SCHEMA,
        "completed": True,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "record_completed": True,
        "frame_count": RECORD.FRAME_COUNT,
        "fps": RECORD.FPS,
        "report": _descriptor(report),
        "sdk": dict(envelope),
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 1,
    }


def verify_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    for name in ("failure.json", "shutdown_failure.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise ValueError("prepared record 성공과 실패 증거가 함께 있다")
    report_path = _regular(output / REPORT_NAME)
    result = _read(_regular(output / "result.json"))
    envelope = result.get("sdk")
    if not isinstance(envelope, Mapping) or not _successful_sdk_envelope(
        validated, envelope, sealed=True
    ):
        raise ValueError(
            "prepared record SDK lifecycle 또는 625 frame 검증이 완료되지 않았다"
        )
    sealed = _read(report_path)
    if not _successful_sdk_envelope(validated, sealed, sealed=False):
        raise ValueError("prepared record 봉인 보고서가 cleanup 후 경계와 다르다")
    expected = dict(envelope)
    expected["lifecycle"] = dict(envelope["lifecycle"])
    expected["lifecycle"]["envelope_sealed"] = False
    if sealed != expected or result != _terminal_body(validated, envelope, report_path):
        raise ValueError("prepared record terminal·봉인 원문 결합이 다르다")
    SHUTDOWN.verify(Path(validated["job_path"]), validated, output / "result.json")
    return result


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """SDK close 전에 cleanup 완료 원문과 terminal 증거를 쓴다."""
    validated = JOB.validate_job(job_path)
    output = Path(validated["output_directory"])
    app = None
    primary: BaseException | None = None
    report_path = output / REPORT_NAME
    try:
        if app_factory is None:
            import vision_play_worker_v3 as V3

            app_factory = V3._sdk_app_factory
        if run_sdk is None:
            import vision_prepared_record_sdk_v1 as SDK

            run_sdk = SDK.run_prepared_record
        app = app_factory(validated)
        envelope = run_sdk(
            validated,
            app,
            seal_report=lambda value: JOB.write_new(report_path, value),
        )
        if not _successful_sdk_envelope(validated, envelope, sealed=True):
            raise ValueError(
                "prepared record SDK 수집·cleanup·원문 검증 경계가 실패했다"
            )
        JOB.write_new(
            output / "result.json", _terminal_body(validated, envelope, report_path)
        )
    except BaseException as error:  # noqa: BLE001 - close 전에 원래 실패를 봉인한다.
        primary = error
        output.mkdir(parents=True, exist_ok=True)
        partial = None
        try:
            if report_path.exists() or report_path.is_symlink():
                partial = _descriptor(report_path)
        except Exception as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
        try:
            JOB.write_new(
                output / "failure.json",
                {
                    "schema": JOB.FAILURE_SCHEMA,
                    "completed": False,
                    "mode": JOB.MODE,
                    "job_sha256": validated["job_sha256"],
                    "checkpoint_sha256": validated["checkpoint_sha256"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "partial_report": partial,
                    "sdk": getattr(
                        error,
                        "prepared_record_sdk_envelope",
                        getattr(error, "envelope", getattr(error, "report", None)),
                    ),
                },
            )
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:  # noqa: BLE001
                try:
                    JOB.write_new(
                        output / "shutdown_failure.json",
                        {
                            "schema": "vision_prepared_record_shutdown_failure_v1",
                            "completed": False,
                            "job_sha256": validated["job_sha256"],
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    )
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
                if primary is None:
                    primary = error
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = OWNED._process_identity,
    cleanup: Callable[..., bool] = OWNED._safe_cleanup,
) -> dict[str, Any]:
    validated = JOB.validate_job(job_path)
    stage = Path(validated["job_path"]).parent
    if (stage / "execution.json").exists() or (stage / "launch.json").exists():
        raise FileExistsError("실행 이력이 있는 prepared record job을 반복하지 않는다")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError(
            "실제 prepared record GPU 실행은 소유 Linux process group에서 수행한다"
        )
    started = time.monotonic()
    max_wall_s = float(JOB.LIMITS["max_wall_s"])
    deadline = started + max_wall_s
    if gpu_reader is None:
        resource = validated["resource"]
        gpu_reader = REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=BASE._gpu_query_run(deadline),
        )
    reader = gpu_reader
    if reader is None:
        raise TypeError("prepared record GPU 관찰 함수가 없다")
    before_bytes = Path(job_path).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "completed": False,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "exit_code": None,
        "timed_out": False,
        "cleanup_verified": False,
        "launch": None,
        "gpu_before": None,
        "gpu_after": None,
        "gpu_before_observation": None,
        "gpu_after_observation": None,
        "result_sha256": None,
        "error_type": None,
        "error_message": None,
        "wall_s": None,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 1,
        "timing": {
            "max_wall_s": max_wall_s,
            "child_max_s": max_wall_s - POSTFLIGHT_RESERVE_S,
            "postflight_reserve_s": POSTFLIGHT_RESERVE_S,
            "cleanup_worst_case_s": OWNED_CLEANUP_WORST_CASE_S,
            "child_wait_timeout_s": None,
            "cleanup_wall_s": None,
            "postflight_wall_s": None,
            "deadline_overrun_s": 0.0,
        },
    }
    process = None
    primary: BaseException | None = None
    cleanup_attempted = False
    postflight_started: float | None = None

    def clean_owned() -> bool:
        nonlocal cleanup_attempted
        cleanup_attempted = True
        begin = time.monotonic()
        try:
            return cleanup(receipt["launch"], process)
        finally:
            receipt["timing"]["cleanup_wall_s"] = time.monotonic() - begin

    command = [
        str(validated["job"]["python"]),
        "-I",
        "-S",
        "-c",
        GATE.OWNED_WORKER_GATE,
        str(validated["job"]["python"]),
        "-B",
        str(Path(__file__).resolve()),
        "--sdk-child",
        "--job",
        str(Path(job_path).absolute()),
    ]
    try:
        receipt["gpu_before"] = reader()
        if receipt["gpu_before"]:
            raise RuntimeError("등록 GPU에 다른 compute process가 있다")
        receipt["gpu_before_observation"] = dict(BASE._gpu_reference(reader, validated))
        BASE._remaining(deadline, reserve=POSTFLIGHT_RESERVE_S)
        JOB.write_new(
            stage / "launch.json",
            {"command": command, "cwd": validated["job"]["isaaclab"]},
        )
        with (
            (stage / "stdout.log").open("xb") as stdout,
            (stage / "stderr.log").open("xb") as stderr,
        ):
            process = popen(
                command,
                cwd=validated["job"]["isaaclab"],
                env={
                    **os.environ,
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.PIPE,
                start_new_session=True,
                shell=False,
            )
            try:
                receipt["launch"] = dict(identify(process.pid))
            except BaseException:
                receipt["cleanup_verified"] = GATE._cleanup_waiting_gate(process)
                process = None
                raise
            if process.stdin is None:
                raise RuntimeError("소유 prepared record gate에 stdin이 없다")
            remaining = min(
                max_wall_s - POSTFLIGHT_RESERVE_S,
                BASE._remaining(deadline, reserve=POSTFLIGHT_RESERVE_S),
            )
            receipt["timing"]["child_wait_timeout_s"] = remaining
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            try:
                receipt["exit_code"] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                raise
            finally:
                postflight_started = time.monotonic()
                receipt["cleanup_verified"] = clean_owned()
        BASE._remaining(deadline)
        receipt["gpu_after"] = reader()
        receipt["gpu_after_observation"] = dict(BASE._gpu_reference(reader, validated))
        if Path(job_path).read_bytes() != before_bytes:
            raise ValueError("prepared record SDK child 실행 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        if (
            receipt["exit_code"] != 0
            or receipt["cleanup_verified"] is not True
            or receipt["gpu_after"] != []
        ):
            raise RuntimeError(
                "prepared record child 종료·소유 group 정리·GPU idle 증거가 실패했다"
            )
        verify_result(validated)
        if receipt["timing"]["cleanup_wall_s"] > OWNED_CLEANUP_WORST_CASE_S:
            raise TimeoutError("소유 process group 정리가 예약한 시간을 넘었다")
        receipt["result_sha256"] = _sha(
            Path(validated["output_directory"]) / "result.json"
        )
        receipt["completed"] = True
    except BaseException as error:  # noqa: BLE001
        primary = error
        receipt["error_type"], receipt["error_message"] = (
            type(error).__name__,
            str(error),
        )
    finally:
        if process is not None and not cleanup_attempted:
            try:
                postflight_started = time.monotonic()
                receipt["cleanup_verified"] = clean_owned()
                receipt["exit_code"] = process.returncode
            except BaseException as error:  # noqa: BLE001
                receipt["cleanup_verified"] = False
                if primary is None:
                    primary = error
                    receipt["error_type"], receipt["error_message"] = (
                        type(error).__name__,
                        str(error),
                    )
        if receipt["gpu_after"] is None:
            try:
                BASE._remaining(deadline)
                receipt["gpu_after"] = reader()
                receipt["gpu_after_observation"] = getattr(reader, "last_receipt", None)
            except BaseException as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                    receipt["error_type"], receipt["error_message"] = (
                        type(error).__name__,
                        str(error),
                    )
        receipt["wall_s"] = time.monotonic() - started
        receipt["timing"]["postflight_wall_s"] = (
            time.monotonic() - postflight_started
            if postflight_started is not None
            else None
        )
        receipt["timing"]["deadline_overrun_s"] = max(
            0.0, receipt["wall_s"] - max_wall_s
        )
        if receipt["wall_s"] >= max_wall_s:
            receipt["completed"] = False
            if primary is None:
                receipt["error_type"] = "TimeoutError"
                receipt["error_message"] = "prepared record 전체 시간 상한이 끝났다"
        if primary is not None:
            receipt["completed"] = False
        JOB.write_new(stage / "execution.json", receipt)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return receipt


def prepare_job(
    protocol_path: Path,
    resource_path: Path,
    resource_sha: str,
    isaaclab: Path,
    python: Path,
    stage: Path,
    checkpoint_logical_id: str,
) -> Path:
    return JOB.create_job(
        protocol_path,
        resource_path,
        resource_sha,
        isaaclab,
        python,
        stage,
        checkpoint_logical_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--checkpoint-logical-id")
    args = parser.parse_args(argv)
    if args.sdk_child:
        if args.job is None:
            parser.error("--sdk-child에는 --job이 필요하다")
        return sdk_child(args.job)
    if args.job is None:
        values = (
            args.protocol,
            args.resource_proof,
            args.resource_sha256,
            args.isaaclab,
            args.python,
            args.stage,
            args.checkpoint_logical_id,
        )
        if any(value is None for value in values):
            parser.error("job 또는 prepare 인자가 모두 필요하다")
        args.job = prepare_job(
            args.protocol,
            args.resource_proof,
            args.resource_sha256,
            args.isaaclab,
            args.python,
            args.stage,
            args.checkpoint_logical_id,
        )
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
