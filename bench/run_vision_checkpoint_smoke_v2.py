"""A-A-B-A smoke 뒤 checkpoint 정책 probe를 소유 실행한다."""

from __future__ import annotations

import argparse
import json
import os
import re
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

import run_vision_prepared_capture_v1 as BASE  # pyright: ignore[reportMissingImports]
import run_vision_prepared_record_v1 as RECORD_PARENT  # pyright: ignore[reportMissingImports]
import vision_checkpoint_smoke_job_v2 as JOB  # pyright: ignore[reportMissingImports]
import vision_checkpoint_smoke_sdk_v2 as SDK  # pyright: ignore[reportMissingImports]

REPORT_NAME = "checkpoint-smoke-v2-sdk-report.json"
PRIMARY_ERROR_NAME = "primary-error.json"
EXECUTION_SCHEMA = "vision_checkpoint_smoke_execution_v2"
FAILURE_ATOM_SCHEMA = "vision_checkpoint_smoke_failure_atom_v2"
POSTFLIGHT_RESERVE_S = 200.0
CHILD_MAX_S = 400.0
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ChildFailure(RuntimeError):
    def __init__(self, failure: dict[str, Any], first: dict[str, Any]) -> None:
        super().__init__(str(first["error_message"]))
        self.failure = failure
        self.first = first


def _adapted(validated: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(validated)
    value["job"] = {**dict(validated["job"]), "mode": BASE.JOB.MODE}
    return value


def _probe(value: Any, checkpoint_sha: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    tensor_keys = {"shape", "dtype", "sha256"}
    before = value.get("step_counter_before")
    return (
        set(value)
        == {
            "schema",
            "checkpoint_sha256",
            "command",
            "observation",
            "action",
            "environment_steps",
            "step_counter_before",
            "step_counter_after",
            "capture_completed_before_policy_step",
            "writer_detached_before_policy_step",
            "actual_environment_observation_used",
            "pure_checkpoint_read_only",
        }
        and value.get("schema") == "vision_checkpoint_policy_probe_v2"
        and value.get("checkpoint_sha256") == checkpoint_sha
        and value.get("command") == [1.0, 0.0, 0.0]
        and value.get("environment_steps") == 1
        and type(before) is int
        and value.get("step_counter_after") == before + 1
        and value.get("capture_completed_before_policy_step") is True
        and value.get("writer_detached_before_policy_step") is True
        and value.get("actual_environment_observation_used") is True
        and value.get("pure_checkpoint_read_only") is False
        and all(
            isinstance(value.get(name), Mapping)
            and set(value[name]) == tensor_keys
            and isinstance(value[name]["shape"], list)
            and bool(value[name]["shape"])
            and all(type(item) is int and item > 0 for item in value[name]["shape"])
            and isinstance(value[name]["dtype"], str)
            and HEX64.fullmatch(str(value[name]["sha256"])) is not None
            for name in ("observation", "action")
        )
    )


def _successful_sdk(validated: Mapping[str, Any], value: Any, *, sealed: bool) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "prepared_smoke",
        "policy_probe",
        "approved",
    }:
        return False
    prepared = value.get("prepared_smoke")
    return (
        value.get("schema") == SDK.SCHEMA
        and value.get("approved") is True
        and _probe(value.get("policy_probe"), str(validated["checkpoint_sha256"]))
        and isinstance(prepared, Mapping)
        and BASE._successful_sdk_envelope(_adapted(validated), prepared, sealed=sealed)
        and prepared.get("capture_report", {}).get("policy_steps") == 0
        and prepared.get("capture_report", {}).get("capture_phase_physics_steps") == 0
    )


def _descriptor(path: Path) -> dict[str, Any]:
    return RECORD_PARENT._descriptor(path)


def _result_body(
    validated: Mapping[str, Any], sdk: Mapping[str, Any], report: Path
) -> dict[str, Any]:
    return {
        "schema": JOB.RESULT_SCHEMA,
        "completed": True,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "capture_smoke_completed": True,
        "checkpoint_policy_probe_completed": True,
        "renderer_smoke_only": False,
        "report": _descriptor(report),
        "sdk": dict(sdk),
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
    }


def verify_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    if (output / "failure.json").exists():
        raise ValueError("checkpoint smoke 성공과 실패가 함께 있다")
    result = BASE._read(BASE._regular(output / "result.json"))
    report_path = BASE._regular(output / REPORT_NAME)
    sdk = result.get("sdk")
    if not isinstance(sdk, Mapping) or not _successful_sdk(validated, sdk, sealed=True):
        raise ValueError("checkpoint smoke SDK 성공 envelope가 다르다")
    sealed = BASE._read(report_path)
    if not _successful_sdk(validated, sealed, sealed=False):
        raise ValueError("checkpoint smoke 봉인 원문이 다르다")
    prepared = sdk["prepared_smoke"]
    if not isinstance(prepared, Mapping) or not isinstance(
        prepared.get("lifecycle"), Mapping
    ):
        raise TypeError("checkpoint smoke prepared lifecycle가 없다")
    expected: dict[str, Any] = dict(sdk)
    expected["prepared_smoke"] = dict(prepared)
    expected["prepared_smoke"]["lifecycle"] = dict(prepared["lifecycle"])
    expected["prepared_smoke"]["lifecycle"]["envelope_sealed"] = False
    if sealed != expected or result != _result_body(validated, sdk, report_path):
        raise ValueError("checkpoint smoke terminal과 봉인 원문 결합이 다르다")
    return result


def _first_error(error: BaseException) -> dict[str, Any]:
    primary = getattr(error, "primary", error)
    return {
        "schema": "vision_checkpoint_smoke_primary_error_v2",
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
        run_sdk = run_sdk or SDK.run_checkpoint_smoke
        envelope = run_sdk(
            validated, app, seal_report=lambda value: JOB.write_new(report_path, value)
        )
        if not _successful_sdk(validated, envelope, sealed=True):
            raise ValueError("checkpoint smoke SDK 검증 경계가 실패했다")
        JOB.write_new(
            output / "result.json", _result_body(validated, envelope, report_path)
        )
    except BaseException as error:  # noqa: BLE001
        primary = error
        first_path = output / PRIMARY_ERROR_NAME
        try:
            JOB.write_new(first_path, _first_error(error))
            partial = _descriptor(report_path) if report_path.exists() else None
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
                    "first_cause": _descriptor(first_path),
                    "partial_report": partial,
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
                try:
                    JOB.write_new(
                        output / "shutdown_failure.json",
                        {
                            "schema": "vision_checkpoint_smoke_shutdown_failure_v2",
                            "completed": False,
                            "job_sha256": validated["job_sha256"],
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    )
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
                primary = primary or error
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def _verify_failure(validated: Mapping[str, Any]) -> ChildFailure:
    output = Path(validated["output_directory"])
    failure = BASE._read(BASE._regular(output / "failure.json"))
    if (
        failure.get("schema") != JOB.FAILURE_SCHEMA
        or failure.get("completed") is not False
        or failure.get("job_sha256") != validated["job_sha256"]
        or failure.get("checkpoint_sha256") != validated["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint smoke failure terminal이 다르다")
    desc = failure.get("first_cause")
    path = output / str(desc.get("file", "")) if isinstance(desc, Mapping) else output
    if (
        not isinstance(desc, Mapping)
        or path.name != PRIMARY_ERROR_NAME
        or _descriptor(path) != dict(desc)
    ):
        raise ValueError("checkpoint smoke first cause descriptor가 다르다")
    first = BASE._read(path)
    if (
        first.get("schema") != "vision_checkpoint_smoke_primary_error_v2"
        or not isinstance(first.get("error_type"), str)
        or not isinstance(first.get("error_message"), str)
        or not isinstance(first.get("traceback"), str)
        or not first["traceback"]
    ):
        raise ValueError("checkpoint smoke first cause 원문이 다르다")
    return ChildFailure(failure, first)


def verify_terminal(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    success, failure = (
        (output / "result.json").exists(),
        (output / "failure.json").exists(),
    )
    if success and failure:
        raise ValueError("checkpoint smoke 성공·실패 terminal이 함께 있다")
    if failure:
        raise _verify_failure(validated)
    if not success:
        raise FileNotFoundError("checkpoint smoke terminal이 없다")
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
        raise FileExistsError("checkpoint smoke job을 반복 실행하지 않는다")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("checkpoint smoke 실제 실행은 소유 Linux group에서 수행한다")
    started, deadline = time.monotonic(), time.monotonic() + JOB.LIMITS["max_wall_s"]
    if gpu_reader is None:
        resource = validated["resource"]
        gpu_reader = RECORD_PARENT.REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in RECORD_PARENT.REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=RECORD_PARENT.BASE._gpu_query_run(deadline),
        )
    reader = gpu_reader
    if reader is None:
        raise TypeError("checkpoint smoke GPU reader가 없다")
    before = Path(job_path).read_bytes()
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
        "failure_atom": None,
        "error_type": None,
        "error_message": None,
        "wall_s": None,
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
                raise RuntimeError("checkpoint smoke gate stdin이 없다")
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
            raise ValueError("checkpoint smoke 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        if receipt["cleanup_verified"] is not True or receipt["gpu_after"] != []:
            raise RuntimeError("checkpoint smoke cleanup/GPU idle 증거가 실패했다")
        terminal = verify_terminal(validated)
        if receipt["exit_code"] != 0:
            raise RuntimeError("checkpoint smoke 성공 terminal과 실패 exit가 함께 있다")
        receipt["result_sha256"] = BASE._sha(
            Path(validated["output_directory"]) / "result.json"
        )
        receipt["completed"] = terminal["completed"] is True
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
            except BaseException as cleanup_error:  # noqa: BLE001
                primary = primary or cleanup_error
        if receipt["gpu_after"] is None:
            try:
                receipt["gpu_after"] = reader()
                receipt["gpu_after_observation"] = getattr(reader, "last_receipt", None)
            except BaseException as gpu_error:  # noqa: BLE001
                primary = primary or gpu_error
        receipt["wall_s"] = time.monotonic() - started
        if receipt["wall_s"] >= JOB.LIMITS["max_wall_s"]:
            primary = primary or TimeoutError("checkpoint smoke 600초 상한 초과")
        if primary is not None:
            receipt["completed"] = False
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
        args.job = JOB.create_job(*values)
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
