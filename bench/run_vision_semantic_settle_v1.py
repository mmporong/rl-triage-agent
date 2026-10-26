"""checkpoint callback 경계 진단 한 변형을 소유 Linux child로 실행한다."""

from __future__ import annotations

import argparse
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

import run_vision_checkpoint_smoke_v2 as PARENT
import run_vision_prepared_record_v1 as RECORD_PARENT
import vision_semantic_settle_checkpoint_v1 as SDK
import vision_semantic_settle_job_v1 as JOB

REPORT_NAME = "semantic-settle-report.json"
PRIMARY_ERROR_NAME = "primary-error.json"
EXECUTION_SCHEMA = "vision_semantic_settle_execution_v1"
FAILURE_ATOM_SCHEMA = "vision_semantic_settle_failure_atom_v1"
POSTFLIGHT_RESERVE_S = 200.0
CHILD_MAX_S = 400.0


class ChildFailure(RuntimeError):
    def __init__(self, failure: dict[str, Any], first: dict[str, Any]) -> None:
        super().__init__(str(first["error_message"]))
        self.failure, self.first = failure, first


def _descriptor(path: Path) -> dict[str, Any]:
    return RECORD_PARENT._descriptor(path)


def _diagnostic(validated: Mapping[str, Any], value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != SDK.TOP_KEYS:
        return False
    boundary = value.get("boundary")
    if not isinstance(boundary, Mapping) or set(boundary) != SDK.BOUNDARY.TOP_KEYS:
        return False
    callbacks = boundary.get("callbacks")
    operations = boundary.get("operations")
    capture = boundary.get("capture_on_play")
    marker_states = value.get("marker_states")
    return (
        value.get("schema") == SDK.SCHEMA
        and value.get("variant") == validated["job"]["callback_variant"]
        and boundary.get("schema") == SDK.BOUNDARY.SCHEMA
        and boundary.get("variant") == "manual_schedule"
        and isinstance(boundary.get("underlying"), Mapping)
        and isinstance(callbacks, list)
        and bool(callbacks)
        and all(SDK.BOUNDARY._callback_complete(row) for row in callbacks)
        and isinstance(operations, Mapping)
        and set(operations) == {"timeline", "counts"}
        and isinstance(operations["timeline"], list)
        and isinstance(operations["counts"], Mapping)
        and isinstance(capture, Mapping)
        and capture.get("after_startup") is False
        and capture.get("restored") == capture.get("before")
        and boundary.get("diagnostic_complete") is True
        and boundary.get("primary_error") is None
        and boundary.get("used_for_decision") is False
        and boundary.get("approved") is False
        and SDK._settles_complete(value.get("settles"))
        and SDK._marker_states_complete(marker_states)
        and value.get("diagnostic_complete") is True
        and value.get("used_for_decision") is False
        and value.get("approved") is False
        and value.get("primary_error") is None
    )


def _result_body(
    validated: Mapping[str, Any], sdk: Mapping[str, Any], report: Path
) -> dict[str, Any]:
    return {
        "schema": JOB.RESULT_SCHEMA,
        "completed": True,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "callback_variant": validated["job"]["callback_variant"],
        "diagnostic_complete": True,
        "capture_approved": False,
        "used_for_decision": False,
        "report": _descriptor(report),
        "sdk": dict(sdk),
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
    }


def verify_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    if (output / "failure.json").exists():
        raise ValueError("callback diagnostic 성공과 실패가 함께 있다")
    result = PARENT.BASE._read(PARENT.BASE._regular(output / "result.json"))
    report_path = PARENT.BASE._regular(output / REPORT_NAME)
    sdk = result.get("sdk")
    report = PARENT.BASE._read(report_path)
    if (
        not isinstance(sdk, Mapping)
        or not _diagnostic(validated, sdk)
        or report != sdk
        or result != _result_body(validated, sdk, report_path)
    ):
        raise ValueError("callback diagnostic terminal 결합이 다르다")
    return result


def _first_error(error: BaseException) -> dict[str, Any]:
    primary = getattr(error, "primary", error)
    return {
        "schema": "vision_semantic_settle_primary_error_v1",
        "error_type": f"{type(primary).__module__}.{type(primary).__qualname__}",
        "error_message": str(primary),
        "traceback": "".join(
            traceback.format_exception(type(primary), primary, primary.__traceback__)
        ),
    }


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] | None = None,
) -> int:
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
        app = app_factory(validated)
        run_sdk = run_sdk or SDK.run_semantic_settle_diagnostic
        envelope = run_sdk(
            validated, app, seal_report=lambda value: JOB.write_new(report_path, value)
        )
        if not _diagnostic(validated, envelope):
            raise ValueError("semantic settle SDK 검증 경계가 실패했다")
        JOB.write_new(
            output / "result.json", _result_body(validated, envelope, report_path)
        )
    except BaseException as error:  # noqa: BLE001
        primary = error
        first_path = output / PRIMARY_ERROR_NAME
        try:
            JOB.write_new(first_path, _first_error(error))
            JOB.write_new(
                output / "failure.json",
                {
                    "schema": JOB.FAILURE_SCHEMA,
                    "completed": False,
                    "mode": JOB.MODE,
                    "job_sha256": validated["job_sha256"],
                    "checkpoint_sha256": validated["checkpoint_sha256"],
                    "callback_variant": validated["job"]["callback_variant"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "first_cause": _descriptor(first_path),
                    "partial_report": _descriptor(report_path)
                    if report_path.exists()
                    else None,
                    "sdk": getattr(error, "envelope", getattr(error, "report", None)),
                },
            )
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:  # noqa: BLE001
                primary = primary or error
                try:
                    JOB.write_new(
                        output / "shutdown_failure.json",
                        {
                            "schema": "vision_semantic_settle_shutdown_failure_v1",
                            "completed": False,
                            "job_sha256": validated["job_sha256"],
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    )
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def _verify_failure(validated: Mapping[str, Any]) -> ChildFailure:
    output = Path(validated["output_directory"])
    failure = PARENT.BASE._read(PARENT.BASE._regular(output / "failure.json"))
    if (
        failure.get("schema") != JOB.FAILURE_SCHEMA
        or failure.get("completed") is not False
        or failure.get("job_sha256") != validated["job_sha256"]
        or failure.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or failure.get("callback_variant") != validated["job"]["callback_variant"]
    ):
        raise ValueError("callback diagnostic failure terminal이 다르다")
    desc = failure.get("first_cause")
    path = output / str(desc.get("file", "")) if isinstance(desc, Mapping) else output
    if (
        not isinstance(desc, Mapping)
        or path.name != PRIMARY_ERROR_NAME
        or _descriptor(path) != dict(desc)
    ):
        raise ValueError("callback diagnostic first cause descriptor가 다르다")
    first = PARENT.BASE._read(path)
    if first.get("schema") != "vision_semantic_settle_primary_error_v1" or not all(
        isinstance(first.get(key), str) and first[key]
        for key in ("error_type", "error_message", "traceback")
    ):
        raise ValueError("callback diagnostic first cause 원문이 다르다")
    return ChildFailure(failure, first)


def verify_terminal(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    success, failure = (
        (output / "result.json").exists(),
        (output / "failure.json").exists(),
    )
    if success and failure:
        raise ValueError("callback diagnostic 성공·실패 terminal이 함께 있다")
    if failure:
        raise _verify_failure(validated)
    if not success:
        raise FileNotFoundError("callback diagnostic terminal이 없다")
    return verify_result(validated)


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[
        [int], Mapping[str, Any]
    ] = RECORD_PARENT.OWNED._process_identity,
    cleanup: Callable[..., bool] = RECORD_PARENT.OWNED._safe_cleanup,
) -> dict[str, Any]:
    validated = JOB.validate_job(job_path)
    stage = Path(validated["job_path"]).parent
    if (stage / "execution.json").exists() or (stage / "launch.json").exists():
        raise FileExistsError("callback diagnostic job을 반복 실행하지 않는다")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError(
            "callback diagnostic 실제 실행은 소유 Linux group에서 수행한다"
        )
    started, deadline = (
        time.monotonic(),
        time.monotonic() + JOB.LIMITS["max_wall_s_per_variant"],
    )
    if gpu_reader is None:
        resource = validated["resource"]
        gpu_reader = RECORD_PARENT.REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in RECORD_PARENT.REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=RECORD_PARENT.BASE._gpu_query_run(deadline),
        )
    if gpu_reader is None:
        raise TypeError("callback diagnostic GPU reader가 없다")
    reader = gpu_reader
    before = Path(job_path).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "completed": False,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "callback_variant": validated["job"]["callback_variant"],
        "exit_code": None,
        "timed_out": False,
        "cleanup_verified": False,
        "launch": None,
        "gpu_before": None,
        "gpu_after": None,
        "gpu_before_observation": None,
        "gpu_after_observation": None,
        "result_sha256": None,
        "failure_atom": None,
        "error_type": None,
        "error_message": None,
        "wall_s": None,
        "diagnostic_complete": False,
        "capture_approved": False,
        "used_for_decision": False,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
        "resource_proof_sha256": validated["job"]["resource_proof"]["sha256"],
        "resource_identity": {
            key: validated["resource"][key]
            for key in (
                "resource_id",
                "gpu_uuid",
                "lease_id",
                "compute_started_at_utc",
                "protocol_sha256",
            )
        },
        "resource_freshness_validated_at_start": True,
    }
    process = None
    primary: BaseException | None = None
    command = [
        str(validated["job"]["python"]),
        "-I",
        "-S",
        "-c",
        RECORD_PARENT.GATE.OWNED_WORKER_GATE,
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
        receipt["gpu_before_observation"] = dict(
            RECORD_PARENT.BASE._gpu_reference(reader, validated)
        )
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
                receipt["cleanup_verified"] = RECORD_PARENT.GATE._cleanup_waiting_gate(
                    process
                )
                process = None
                raise
            if process.stdin is None:
                raise RuntimeError("callback diagnostic gate stdin이 없다")
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            try:
                receipt["exit_code"] = process.wait(timeout=CHILD_MAX_S)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                raise
            finally:
                receipt["cleanup_verified"] = cleanup(receipt["launch"], process)
        receipt["gpu_after"] = reader()
        receipt["gpu_after_observation"] = dict(
            RECORD_PARENT.BASE._gpu_reference(reader, validated)
        )
        if Path(job_path).read_bytes() != before:
            raise ValueError("callback diagnostic 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        if receipt["cleanup_verified"] is not True or receipt["gpu_after"] != []:
            raise RuntimeError("callback diagnostic cleanup/GPU idle 증거가 실패했다")
        terminal = verify_terminal(validated)
        if receipt["exit_code"] != 0:
            raise RuntimeError(
                "callback diagnostic 성공 terminal과 실패 exit가 함께 있다"
            )
        receipt["result_sha256"] = PARENT.BASE._sha(
            Path(validated["output_directory"]) / "result.json"
        )
        receipt["completed"] = terminal["completed"] is True
        receipt["diagnostic_complete"] = terminal["diagnostic_complete"] is True
    except ChildFailure as error:
        primary = error
        receipt["failure_atom"] = {
            "schema": FAILURE_ATOM_SCHEMA,
            "status": "failed",
            "child_exit_code": receipt["exit_code"],
            "child_exit_accepted": False,
            "failure": _descriptor(
                Path(validated["output_directory"]) / "failure.json"
            ),
            "first_cause": error.failure["first_cause"],
            "first_cause_type": error.first["error_type"],
            "first_cause_message": error.first["error_message"],
        }
        receipt["error_type"], receipt["error_message"] = (
            error.first["error_type"],
            error.first["error_message"],
        )
    except BaseException as error:  # noqa: BLE001
        primary = error
        receipt["error_type"], receipt["error_message"] = (
            type(error).__name__,
            str(error),
        )
    finally:
        if process is not None and receipt["cleanup_verified"] is not True:
            try:
                receipt["cleanup_verified"] = cleanup(receipt["launch"], process)
            except BaseException as cleanup_error:  # noqa: BLE001 - 정리 후 최초 오류를 보존한다.
                primary = primary or cleanup_error
        if receipt["gpu_after"] is None:
            try:
                receipt["gpu_after"] = reader()
                receipt["gpu_after_observation"] = getattr(reader, "last_receipt", None)
            except BaseException as gpu_error:  # noqa: BLE001 - 정리 뒤 GPU 관측 오류를 보존한다.
                primary = primary or gpu_error
        receipt["wall_s"] = time.monotonic() - started
        if receipt["wall_s"] >= JOB.LIMITS["max_wall_s_per_variant"]:
            primary = primary or TimeoutError("callback diagnostic 600초 상한 초과")
        if primary is not None:
            receipt["completed"] = False
            receipt["diagnostic_complete"] = False
        JOB.write_new(stage / "execution.json", receipt)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return receipt


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
    parser.add_argument("--callback-variant", choices=JOB.VARIANTS)
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
            args.callback_variant,
        )
        if any(value is None for value in values):
            parser.error("job 또는 prepare 인자가 모두 필요하다")
        args.job = JOB.create_job(*values)
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
