"""준비 녹화의 첫 실패 원문을 보존하고 성공·실패 terminal을 분리한다."""

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

import run_vision_prepared_record_v1 as V1  # pyright: ignore[reportMissingImports]
import vision_record_job_v1 as JOB  # pyright: ignore[reportMissingImports]

REPORT_NAME = V1.REPORT_NAME
PRIMARY_ERROR_NAME = "primary-error.json"
RUNTIME_PREFLIGHT_NAME = "record-runtime-preflight.json"
EXECUTION_SCHEMA = "vision_prepared_record_execution_v2"
PRIMARY_ERROR_SCHEMA = "vision_prepared_record_primary_error_v2"
FAILURE_ATOM_SCHEMA = "vision_prepared_record_failure_atom_v2"
SHUTDOWN_FAILURE_SCHEMA = "vision_prepared_record_shutdown_failure_v2"


class ChildTerminalFailure(RuntimeError):
    """검증된 실패 terminal과 첫 원인을 부모 실행기에 전달한다."""

    def __init__(self, failure: dict[str, Any], first_cause: dict[str, Any]) -> None:
        message = str(first_cause.get("error_message") or failure["error_message"])
        super().__init__(message)
        self.failure = failure
        self.first_cause = first_cause


def _type_name(value: BaseException) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _first_exception(error: BaseException) -> BaseException:
    candidate = getattr(error, "primary", None)
    if isinstance(candidate, BaseException):
        return candidate
    if isinstance(error.__cause__, BaseException):
        return error.__cause__
    return error


def _primary_error_body(error: BaseException) -> dict[str, Any]:
    first = _first_exception(error)
    return {
        "schema": PRIMARY_ERROR_SCHEMA,
        "error_type": _type_name(first),
        "error_message": str(first),
        "traceback": "".join(
            traceback.format_exception(type(first), first, first.__traceback__)
        ),
    }


def _primary_error_path(validated: Mapping[str, Any], error: BaseException) -> Path:
    path = Path(validated["job_path"]).parent / PRIMARY_ERROR_NAME
    if not path.exists() and not path.is_symlink():
        JOB.write_new(path, _primary_error_body(error))
    return V1._regular(path)


def _runtime_preflight_descriptor(
    validated: Mapping[str, Any], *, required: bool
) -> dict[str, Any] | None:
    path = Path(validated["job_path"]).parent / RUNTIME_PREFLIGHT_NAME
    if not path.exists() and not path.is_symlink():
        if required:
            raise FileNotFoundError("prepared record runtime preflight 증거가 없다")
        return None
    receipt = V1._read(V1._regular(path))
    if (
        receipt.get("schema") != "vision_prepared_record_runtime_preflight_v2"
        or receipt.get("status") not in {"complete", "failed"}
        or not isinstance(receipt.get("tools"), list)
    ):
        raise ValueError("prepared record runtime preflight 원문이 다르다")
    return V1._descriptor(path)


def _failure_body(
    validated: Mapping[str, Any],
    error: BaseException,
    report_path: Path,
    primary_path: Path,
) -> dict[str, Any]:
    partial = None
    if report_path.exists() or report_path.is_symlink():
        partial = V1._descriptor(report_path)
    sdk = getattr(
        error,
        "prepared_record_sdk_envelope",
        getattr(error, "envelope", getattr(error, "report", None)),
    )
    return {
        "schema": JOB.FAILURE_SCHEMA,
        "completed": False,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "first_cause": V1._descriptor(primary_path),
        "runtime_preflight": _runtime_preflight_descriptor(validated, required=False),
        "partial_report": partial,
        "sdk": sdk,
    }


def _read_first_cause(output: Path, failure: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = failure.get("first_cause")
    if not isinstance(descriptor, Mapping):
        raise TypeError("prepared record 실패에 first cause descriptor가 없다")
    stage = output.parent
    path = stage / str(descriptor.get("file", ""))
    if (
        path.name != PRIMARY_ERROR_NAME
        or set(descriptor) != {"file", "sha256", "bytes"}
        or V1._descriptor(path) != dict(descriptor)
    ):
        raise ValueError("prepared record first cause descriptor가 원문과 다르다")
    first = V1._read(path)
    schema = first.get("schema")
    if schema == PRIMARY_ERROR_SCHEMA and isinstance(first.get("exception"), Mapping):
        exception = first["exception"]
        frames = exception.get("traceback_frames")
        if (
            first.get("phase") not in {"runtime_preflight", "run_real"}
            or not isinstance(exception.get("type"), str)
            or not exception["type"]
            or not isinstance(exception.get("message"), str)
            or not isinstance(frames, list)
            or not all(isinstance(row, Mapping) for row in frames)
        ):
            raise ValueError("prepared record SDK first cause 원문이 불완전하다")
        preflight = first.get("preflight")
        preflight_path = stage / RUNTIME_PREFLIGHT_NAME
        if not isinstance(preflight, Mapping) or preflight != {
            "path": RUNTIME_PREFLIGHT_NAME,
            "sha256": V1._sha(V1._regular(preflight_path)),
        }:
            raise ValueError("prepared record first cause와 preflight 결합이 다르다")
        return {
            "error_type": exception["type"],
            "error_message": exception["message"],
            "traceback_frames": frames,
            "phase": first["phase"],
        }
    if (
        not isinstance(schema, str)
        or not schema.startswith("vision_prepared_record_primary_error_v")
        or not isinstance(first.get("error_type"), str)
        or not first["error_type"]
        or not isinstance(first.get("error_message"), str)
        or not isinstance(first.get("traceback"), str)
        or not first["traceback"]
        or first["error_type"].rsplit(".", 1)[-1] not in first["traceback"]
    ):
        raise ValueError("prepared record first cause 원문이 불완전하다")
    return {
        "error_type": first["error_type"],
        "error_message": first["error_message"],
        "traceback": first["traceback"],
        "phase": "parent",
    }


def _verify_failure(
    validated: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(validated["output_directory"])
    failure_path = V1._regular(output / "failure.json")
    failure = V1._read(failure_path)
    if (
        failure.get("schema") != JOB.FAILURE_SCHEMA
        or failure.get("completed") is not False
        or failure.get("mode") != JOB.MODE
        or failure.get("job_sha256") != validated["job_sha256"]
        or failure.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or not isinstance(failure.get("error_type"), str)
        or not isinstance(failure.get("error_message"), str)
    ):
        raise ValueError("prepared record failure terminal 결합이 다르다")
    first = _read_first_cause(output, failure)
    if failure.get("runtime_preflight") != _runtime_preflight_descriptor(
        validated, required=True
    ):
        raise ValueError("prepared record 실패와 runtime preflight 결합이 다르다")
    report = V1._regular(output / REPORT_NAME)
    if failure.get("partial_report") != V1._descriptor(report):
        raise ValueError("prepared record 부분 보고서 descriptor가 다르다")
    sealed = V1._read(report)
    sdk = failure.get("sdk")
    if not isinstance(sdk, Mapping) or not isinstance(sealed, Mapping):
        raise TypeError("prepared record 실패 SDK 보고서가 없다")
    expected = dict(sdk)
    lifecycle = sdk.get("lifecycle")
    if (
        sdk.get("approved") is not False
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("envelope_sealed") is not True
        or not isinstance(lifecycle.get("errors"), list)
        or not lifecycle["errors"]
    ):
        raise ValueError("prepared record 실패 SDK lifecycle가 불완전하다")
    expected["lifecycle"] = dict(lifecycle)
    expected["lifecycle"]["envelope_sealed"] = False
    if sealed != expected:
        raise ValueError("prepared record 실패 SDK와 부분 보고서가 다르다")
    return failure, first


def verify_terminal(validated: Mapping[str, Any]) -> dict[str, Any]:
    """성공 terminal만 반환하고, 유효한 실패 terminal은 첫 원인으로 구분한다."""
    output = Path(validated["output_directory"])
    has_result = (output / "result.json").exists() or (
        output / "result.json"
    ).is_symlink()
    has_failure = (output / "failure.json").exists() or (
        output / "failure.json"
    ).is_symlink()
    if has_result and has_failure:
        raise ValueError("prepared record 성공과 실패 terminal이 함께 있다")
    if has_failure:
        failure, first = _verify_failure(validated)
        raise ChildTerminalFailure(failure, first)
    if not has_result:
        raise FileNotFoundError("prepared record terminal 증거가 없다")
    return V1.verify_result(validated)


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """app.close 전에 첫 원인·trace와 성공 또는 실패 terminal을 봉인한다."""
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
            import vision_prepared_record_sdk_v2 as SDK

            run_sdk = SDK.run_prepared_record
        app = app_factory(validated)
        envelope = run_sdk(
            validated,
            app,
            seal_report=lambda value: JOB.write_new(report_path, value),
        )
        if not V1._successful_sdk_envelope(validated, envelope, sealed=True):
            raise ValueError(
                "prepared record SDK 수집·cleanup·원문 검증 경계가 실패했다"
            )
        JOB.write_new(
            output / "result.json", V1._terminal_body(validated, envelope, report_path)
        )
    except BaseException as error:  # noqa: BLE001 - close 전에 원래 실패를 봉인한다.
        primary = error
        output.mkdir(parents=True, exist_ok=True)
        try:
            primary_path = _primary_error_path(validated, error)
            JOB.write_new(
                output / "failure.json",
                _failure_body(validated, error, report_path, primary_path),
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
                            "schema": SHUTDOWN_FAILURE_SCHEMA,
                            "completed": False,
                            "job_sha256": validated["job_sha256"],
                            "error_type": _type_name(error),
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


def _failure_atom(
    validated: Mapping[str, Any], error: ChildTerminalFailure, exit_code: int | None
) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    return {
        "schema": FAILURE_ATOM_SCHEMA,
        "status": "failed",
        "child_exit_code": exit_code,
        "child_exit_accepted": False,
        "failure": V1._descriptor(output / "failure.json"),
        "partial_report": error.failure["partial_report"],
        "first_cause": error.failure["first_cause"],
        "first_cause_type": error.first_cause["error_type"],
        "first_cause_message": error.first_cause["error_message"],
    }


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = V1.OWNED._process_identity,
    cleanup: Callable[..., bool] = V1.OWNED._safe_cleanup,
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
        gpu_reader = V1.REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in V1.REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=V1.BASE._gpu_query_run(deadline),
        )
    if gpu_reader is None:
        raise TypeError("prepared record GPU 관찰 함수가 없다")
    reader = gpu_reader
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
        "runtime_preflight": None,
        "failure_atom": None,
        "error_type": None,
        "error_message": None,
        "wall_s": None,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 1,
        "timing": {
            "max_wall_s": max_wall_s,
            "child_max_s": max_wall_s - V1.POSTFLIGHT_RESERVE_S,
            "postflight_reserve_s": V1.POSTFLIGHT_RESERVE_S,
            "cleanup_worst_case_s": V1.OWNED_CLEANUP_WORST_CASE_S,
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
        V1.GATE.OWNED_WORKER_GATE,
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
            V1.BASE._gpu_reference(reader, validated)
        )
        V1.BASE._remaining(deadline, reserve=V1.POSTFLIGHT_RESERVE_S)
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
                receipt["cleanup_verified"] = V1.GATE._cleanup_waiting_gate(process)
                process = None
                raise
            if process.stdin is None:
                raise RuntimeError("소유 prepared record gate에 stdin이 없다")
            remaining = min(
                max_wall_s - V1.POSTFLIGHT_RESERVE_S,
                V1.BASE._remaining(deadline, reserve=V1.POSTFLIGHT_RESERVE_S),
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
        V1.BASE._remaining(deadline)
        receipt["gpu_after"] = reader()
        receipt["gpu_after_observation"] = dict(
            V1.BASE._gpu_reference(reader, validated)
        )
        if Path(job_path).read_bytes() != before_bytes:
            raise ValueError("prepared record SDK child 실행 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        receipt["runtime_preflight"] = _runtime_preflight_descriptor(
            validated, required=True
        )
        if receipt["cleanup_verified"] is not True or receipt["gpu_after"] != []:
            raise RuntimeError(
                "prepared record 소유 group 정리·GPU idle 증거가 실패했다"
            )
        terminal = verify_terminal(validated)
        if receipt["exit_code"] != 0:
            raise RuntimeError(
                "prepared record 성공 terminal과 child 실패 exit가 함께 있다"
            )
        if receipt["timing"]["cleanup_wall_s"] > V1.OWNED_CLEANUP_WORST_CASE_S:
            raise TimeoutError("소유 process group 정리가 예약한 시간을 넘었다")
        receipt["result_sha256"] = V1._sha(
            Path(validated["output_directory"]) / "result.json"
        )
        receipt["completed"] = terminal.get("completed") is True
    except ChildTerminalFailure as error:
        primary = error
        receipt["failure_atom"] = _failure_atom(validated, error, receipt["exit_code"])
        receipt["error_type"] = error.first_cause["error_type"]
        receipt["error_message"] = error.first_cause["error_message"]
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
                V1.BASE._remaining(deadline)
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
