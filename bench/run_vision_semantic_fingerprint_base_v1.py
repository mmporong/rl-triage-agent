"""진단 전용 Isaac child의 출처·종료·원문 보존을 검증한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import vision_brev_backend_v1 as OWNED  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v2 as GATE  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]
import vision_semantic_fingerprint_job_v1 as JOB
import vision_shutdown_policy_v1 as SHUTDOWN

REPORT_NAME = "one-view.semantic-diagnostic.json"
EXECUTION_SCHEMA = "vision_semantic_sdk_execution_v1"
POSTFLIGHT_RESERVE_S = 200.0
OWNED_CLEANUP_WORST_CASE_S = 40.0
LIFECYCLE_FLAGS = frozenset(
    {
        "setup_completed",
        "writer_attach_attempted",
        "writer_attached",
        "writer_detach_attempted",
        "writer_detached",
        "report_sealed",
        "render_product_destroyed",
        "environment_closed",
    }
)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return JOB._read(path)


def _regular(path: Path) -> Path:
    return JOB._regular(path.absolute(), "진단 증거")


def _descriptor(path: Path) -> dict[str, Any]:
    target = _regular(path)
    return {"file": target.name, "sha256": _sha(target), "bytes": target.stat().st_size}


def _diagnostic_structure(diagnostic: Mapping[str, Any]) -> bool:
    import vision_semantic_diagnostic_v1 as D

    callback_keys = {
        "ordinal",
        "window",
        "descriptor",
        "raw_frame_time",
        "classification",
        "classification_error_type",
        "strict_error_type",
        "counterfactual_schema_check",
    }
    events, callbacks = diagnostic["events"], diagnostic["callbacks"]
    for index, row in enumerate(callbacks):
        if (
            not isinstance(row, dict)
            or set(row) != callback_keys
            or row["ordinal"] != index
        ):
            return False
        if row["window"] not in {
            "after_attach",
            "after_writer_attach_before_collection",
            "after_writer_detach_return_before_render_product_cleanup",
            "attaching",
            "detaching",
            "collection_finished",
            "event_0_running",
            "event_1_running",
            "after_event_0_return",
            "after_event_1_return",
        }:
            return False
        descriptor, timing, counterfactual = (
            row["descriptor"],
            row["raw_frame_time"],
            row["counterfactual_schema_check"],
        )
        if (
            row["classification"] not in {"valid", "invalid"}
            or not isinstance(descriptor, dict)
            or descriptor.get("schema") != "vision_writer_payload_descriptor_v1"
            or not isinstance(descriptor.get("payload_type"), str)
            or not isinstance(descriptor.get("top_level_keys"), list)
            or not isinstance(descriptor.get("annotators"), dict)
            or not isinstance(timing, dict)
            or set(timing)
            != {
                "swhFrameNumber",
                "reference_time",
                "isaac_read_times",
                "unexpected_isaac_read_times_fields",
            }
            or not isinstance(timing["isaac_read_times"], dict)
            or not isinstance(timing["unexpected_isaac_read_times_fields"], list)
            or not isinstance(counterfactual, dict)
            or set(counterfactual) != {"attempted", "passed", "reason", "error_type"}
            or type(counterfactual["attempted"]) is not bool
            or type(counterfactual["passed"]) is not bool
            or (counterfactual["passed"] and not counterfactual["attempted"])
        ):
            return False
        if row["classification"] == "valid" and (
            row["classification_error_type"] is not None
            or row["strict_error_type"] is not None
            or counterfactual
            != {"attempted": False, "passed": False, "reason": None, "error_type": None}
        ):
            return False
    previous_count = 0
    for row in events:
        if not isinstance(row, Mapping):
            return False
        before, after = row.get("physics_before"), row.get("physics_after")
        pre, post = row.get("pre_callback_count"), row.get("post_callback_count")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or before != after
            or D._sha(before) != row.get("physics_before_sha256")
            or D._sha(after) != row.get("physics_after_sha256")
            or type(pre) is not int
            or type(post) is not int
            or not previous_count <= pre <= post <= len(callbacks)
            or any(
                row.get(key) is not None
                for key in (
                    "run_event_error_type",
                    "physics_after_error_type",
                    "diagnostic_error_phase",
                )
            )
        ):
            return False
        previous_count = post
    observer = D.DiagnosticObserver()
    observer.total_callbacks, observer.callbacks = len(callbacks), callbacks
    expected = D._build_report(observer, events, 2)
    expected["collection_scope"] = diagnostic.get("collection_scope")
    return dict(diagnostic) == expected


def _setup_structure(setup: Any, diagnostic: Mapping[str, Any]) -> bool:
    required = {
        "phase_order",
        "setup_physics_before_after",
        "extension",
        "startup",
        "registrations",
        "callback_count_before_detach",
        "callback_count_after_detach",
        "observation_window",
    }
    if not isinstance(setup, Mapping) or set(setup) != required:
        return False
    physics, extension, startup = (
        setup["setup_physics_before_after"],
        setup["extension"],
        setup["startup"],
    )
    if (
        not isinstance(physics, Mapping)
        or set(physics) != {"before", "after", "equal"}
        or not isinstance(physics["before"], Mapping)
        or physics["before"] != physics["after"]
        or physics["equal"] is not True
        or extension
        != {
            "name": "isaacsim.core.nodes",
            "enable_result": True,
            "app_update_completed": True,
        }
        or not isinstance(startup, Mapping)
        or set(startup) != {"before", "after"}
        or not isinstance(startup["before"], Mapping)
        or not isinstance(startup["after"], Mapping)
        or setup["registrations"]
        != dict.fromkeys(
            ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes"), True
        )
    ):
        return False
    before, after = (
        setup["callback_count_before_detach"],
        setup["callback_count_after_detach"],
    )
    if (
        type(before) is not int
        or type(after) is not int
        or not 0 <= before <= after == diagnostic["callbacks_observed"]
    ):
        return False
    order = setup["phase_order"]
    essential = (
        "extension_enable",
        "extension_app_update",
        "replicator_startup",
        "annotator_registrations",
        "marker_create",
        "capture_readiness",
        "marker_bind",
        "writer_attach",
        "event_0",
        "event_1",
        "writer_detach",
    )
    if (
        not isinstance(order, list)
        or any(order.count(name) != 1 for name in essential)
        or [order.index(name) for name in essential]
        != sorted(order.index(name) for name in essential)
    ):
        return False
    window = {
        "starts": "before_writer_attach",
        "ends": "after_writer_detach_return_before_render_product_cleanup",
        "detach_callbacks_included": True,
        "complete_drain_proven": False,
    }
    scope = {
        "starts": window["starts"],
        "ends": window["ends"],
        "callbacks_before_detach_preserved": True,
        "callbacks_during_detach_included": True,
        "complete_drain_proven": False,
        "all_future_callbacks_collected": False,
    }
    return (
        setup["observation_window"] == window
        and diagnostic.get("collection_scope") == scope
    )


def _successful_sdk_envelope(value: Mapping[str, Any]) -> bool:
    import vision_semantic_sdk_v1 as SDK

    if (
        set(value)
        != {
            "schema",
            "diagnostic",
            "lifecycle",
            "renderer",
            "no_capture_approved",
            "setup_evidence",
        }
        or value.get("schema") != SDK.SDK_REPORT_SCHEMA
        or value.get("no_capture_approved") is not True
        or not isinstance(value.get("renderer"), Mapping)
    ):
        return False
    lifecycle, diagnostic = value.get("lifecycle"), value.get("diagnostic")
    if (
        not isinstance(lifecycle, Mapping)
        or set(lifecycle) != LIFECYCLE_FLAGS | {"errors"}
        or any(lifecycle.get(flag) is not True for flag in LIFECYCLE_FLAGS)
        or lifecycle.get("errors") != []
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("schema") != "vision_semantic_diagnostic_v1"
        or diagnostic.get("no_capture_approved") is not True
        or diagnostic.get("step_return_is_complete_drain") is not False
        or diagnostic.get("callback_event_causality_proven") is not False
        or diagnostic.get("bounds") != {"max_events": 2, "max_callbacks": 16}
        or diagnostic.get("events_requested") != 2
        or diagnostic.get("events_attempted") != 2
        or diagnostic.get("events_completed") != 2
    ):
        return False
    observed, preserved = (
        diagnostic.get("callbacks_observed"),
        diagnostic.get("callbacks_preserved"),
    )
    callbacks, events, observations = (
        diagnostic.get("callbacks"),
        diagnostic.get("events"),
        diagnostic.get("observations"),
    )
    if (
        type(observed) is not int
        or type(preserved) is not int
        or not 0 <= observed == preserved <= 16
        or not isinstance(callbacks, list)
        or len(callbacks) != preserved
        or not isinstance(events, list)
        or len(events) != 2
        or not isinstance(observations, Mapping)
        or any(
            observations.get(key) is not False
            for key in ("physics_changed", "overflow", "source_error")
        )
    ):
        return False
    return (
        _diagnostic_structure(diagnostic)
        and _setup_structure(value.get("setup_evidence"), diagnostic)
        and all(
            isinstance(row, Mapping)
            and row.get("event_index") == index
            and row.get("diagnostic_error_type") is None
            and row.get("physics_equal") is True
            for index, row in enumerate(events)
        )
        and all(
            isinstance(row, Mapping) and row.get("ordinal") == index
            for index, row in enumerate(callbacks)
        )
    )


def _terminal_body(
    validated: Mapping[str, Any], envelope: Mapping[str, Any], report: Path
) -> dict[str, Any]:
    return {
        "schema": JOB.RESULT_SCHEMA,
        "completed": True,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "diagnostic_collection_completed": True,
        "no_capture_approved": True,
        "report": _descriptor(report),
        "sdk": dict(envelope),
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
    }


def verify_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    for name in ("failure.json", "shutdown_failure.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise ValueError("진단 성공과 실패 증거가 함께 있다")
    report_path = _regular(output / REPORT_NAME)
    result = _read(_regular(output / "result.json"))
    envelope = result.get("sdk")
    if not isinstance(envelope, Mapping) or not _successful_sdk_envelope(envelope):
        raise ValueError("진단 SDK lifecycle 또는 관찰 경계가 완료되지 않았다")
    sealed = _read(report_path)
    if set(sealed) != set(envelope) or any(
        sealed.get(key) != envelope.get(key)
        for key in (
            "schema",
            "diagnostic",
            "renderer",
            "no_capture_approved",
            "setup_evidence",
        )
    ):
        raise ValueError("봉인 관찰 원문이 최종 SDK 결과와 다르다")
    sealed_lifecycle = sealed.get("lifecycle")
    expected = dict(envelope["lifecycle"])
    for key in ("report_sealed", "render_product_destroyed", "environment_closed"):
        expected[key] = False
    if sealed_lifecycle != expected:
        raise ValueError("관찰 원문 봉인이 detach 뒤 cleanup 전 경계가 아니다")
    if result != _terminal_body(validated, envelope, report_path):
        raise ValueError("진단 terminal result의 job·원문·모드 결합이 다르다")
    SHUTDOWN.verify(Path(validated["job_path"]), validated, output / "result.json")
    return result


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """SDK close 전에 원문과 terminal 증거를 쓴다. child exit은 부모가 검증한다."""
    validated = JOB.validate_job(job_path)
    output = Path(validated["output_directory"])
    output.mkdir(exist_ok=False)
    app = None
    primary: BaseException | None = None
    report_path = output / REPORT_NAME
    try:
        if app_factory is None:
            import vision_play_worker_v3 as V3

            app_factory = V3._sdk_app_factory
        if run_sdk is None:
            import vision_semantic_sdk_v1 as SDK

            run_sdk = SDK.run_one_view
        app = app_factory(validated)

        def seal_report(value: Mapping[str, Any]) -> None:
            JOB.write_new(report_path, value)

        envelope = run_sdk(validated, app, seal_report=seal_report)
        if not _successful_sdk_envelope(envelope):
            raise ValueError("진단 수집의 source·physics·cleanup 경계가 실패했다")
        JOB.write_new(
            output / "result.json", _terminal_body(validated, envelope, report_path)
        )
    except BaseException as error:  # noqa: BLE001 - close 전에 실패 원문을 보존한다.
        primary = error
        partial = None
        try:
            if report_path.exists() or report_path.is_symlink():
                partial = _descriptor(report_path)
        except Exception as evidence_error:  # noqa: BLE001 - descriptor 실패보다 primary를 보존한다.
            traceback.print_exception(evidence_error)
        failure = {
            "schema": JOB.FAILURE_SCHEMA,
            "completed": False,
            "mode": JOB.MODE,
            "job_sha256": validated["job_sha256"],
            "no_capture_approved": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "partial_report": partial,
            "sdk": getattr(
                error, "envelope", getattr(error, "semantic_sdk_envelope", None)
            ),
        }
        try:
            JOB.write_new(output / "failure.json", failure)
        except BaseException as evidence_error:  # noqa: BLE001 - 원 primary를 보존한다.
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:  # noqa: BLE001 - 부모가 실패를 식별한다.
                try:
                    JOB.write_new(
                        output / "shutdown_failure.json",
                        {
                            "schema": "vision_semantic_shutdown_failure_v1",
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


def _remaining(deadline: float, *, reserve: float = 0.0) -> float:
    value = deadline - time.monotonic() - reserve
    if value <= 0:
        raise TimeoutError("진단 전체 시간 또는 후속 검증 예약이 소진됐다")
    return value


def _gpu_query_run(
    deadline: float, run: Callable[..., Any] = subprocess.run
) -> Callable[..., Any]:
    def bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs["timeout"] = min(
            30.0, float(kwargs.get("timeout", 30.0)), _remaining(deadline)
        )
        result = run(*args, **kwargs)
        _remaining(deadline)
        return result

    return bounded


def _gpu_reference(reader: Any, validated: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = getattr(reader, "last_receipt", None)
    if not isinstance(reference, Mapping):
        raise TypeError("진단 GPU 관찰 원문 참조가 없다")
    REMOTE.verify_observation_reference(
        reference,
        validated["resource"]["resource_id"],
        validated["resource"]["gpu_uuid"],
    )
    return reference


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
        raise FileExistsError("실행 이력이 있는 진단 job을 반복하지 않는다")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("실제 진단 GPU 실행은 소유 Linux process group에서 수행한다")
    started = time.monotonic()
    max_wall_s = float(JOB.LIMITS["max_wall_s"])
    deadline = started + max_wall_s
    if gpu_reader is None:
        resource = validated["resource"]
        gpu_reader = REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=_gpu_query_run(deadline),
        )
    if gpu_reader is None:
        raise TypeError("진단 GPU 관찰 함수가 없다")
    before_bytes = Path(job_path).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "completed": False,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
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
        "no_capture_approved": True,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
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
        receipt["gpu_before"] = gpu_reader()
        if receipt["gpu_before"]:
            raise RuntimeError("등록 GPU에 다른 compute process가 있다")
        receipt["gpu_before_observation"] = dict(_gpu_reference(gpu_reader, validated))
        _remaining(deadline, reserve=POSTFLIGHT_RESERVE_S)
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
                raise RuntimeError("소유 진단 gate에 stdin이 없다")
            remaining = min(
                max_wall_s - POSTFLIGHT_RESERVE_S,
                _remaining(deadline, reserve=POSTFLIGHT_RESERVE_S),
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
        _remaining(deadline)
        receipt["gpu_after"] = gpu_reader()
        receipt["gpu_after_observation"] = dict(_gpu_reference(gpu_reader, validated))
        _remaining(deadline)
        if Path(job_path).read_bytes() != before_bytes:
            raise ValueError("SDK child 실행 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        _remaining(deadline)
        if (
            receipt["exit_code"] != 0
            or receipt["cleanup_verified"] is not True
            or receipt["gpu_after"] != []
        ):
            raise RuntimeError(
                "진단 child 종료·소유 group 정리·GPU idle 증거가 실패했다"
            )
        verify_result(validated)
        _remaining(deadline)
        if receipt["timing"]["cleanup_wall_s"] > OWNED_CLEANUP_WORST_CASE_S:
            raise TimeoutError("소유 process group 정리가 예약한 시간을 넘었다")
        receipt["result_sha256"] = _sha(
            Path(validated["output_directory"]) / "result.json"
        )
        receipt["completed"] = True
    except BaseException as error:  # noqa: BLE001 - 실행·취소 증거를 정리 뒤 남긴다.
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
                _remaining(deadline)
                receipt["gpu_after"] = gpu_reader()
                receipt["gpu_after_observation"] = getattr(
                    gpu_reader, "last_receipt", None
                )
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
                receipt["error_message"] = "진단 전체 시간 상한이 끝났다"
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
) -> Path:
    protocol = JOB.validate_protocol(protocol_path)
    _regular(resource_path)
    if _sha(resource_path) != resource_sha:
        raise ValueError("진단 resource proof SHA가 다르다")
    stage = stage.absolute()
    JOB._check_ancestors(stage, "진단 stage")
    if stage.exists():
        raise FileExistsError("진단 stage는 symlink가 없는 새 폴더여야 한다")
    stage.mkdir(parents=True, exist_ok=False)
    path = stage / "job.json"
    JOB.write_new(
        path,
        {
            "schema": JOB.JOB_SCHEMA,
            "mode": JOB.MODE,
            "job_id": protocol["protocol_id"],
            "source": {
                "repo_root": str(ROOT),
                "commit": protocol["source_commit"],
                "files_lf": protocol["source_sha256_lf"],
            },
            "protocol": {
                "path": str(protocol_path.absolute()),
                "sha256_lf": _sha(protocol_path, lf=True),
            },
            "resource_proof": {
                "path": str(resource_path.absolute()),
                "sha256": resource_sha,
            },
            "isaaclab": str(isaaclab.absolute()),
            "python": str(python.absolute()),
            "output_directory": str(stage / "output"),
        },
    )
    JOB.validate_job(path)
    return path


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
    args = parser.parse_args(argv)
    if args.sdk_child:
        if args.job is None:
            parser.error("--sdk-child에는 --job이 필요하다")
        return sdk_child(args.job)
    if args.job is None:
        if any(
            value is None
            for value in (
                args.protocol,
                args.resource_proof,
                args.resource_sha256,
                args.isaaclab,
                args.python,
                args.stage,
            )
        ):
            parser.error(
                "새 실행에는 protocol/resource/isaaclab/python/stage가 필요하다"
            )
        args.job = prepare_job(
            args.protocol,
            args.resource_proof,
            args.resource_sha256,
            args.isaaclab,
            args.python,
            args.stage,
        )
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["completed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
