"""AABA 전용 shutdown receipt까지 결합한 native 정책 녹화 실행기."""

from __future__ import annotations

import argparse
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

import run_vision_annotator_pull_v2 as AABA
import run_vision_pull_record_v1 as V1
import vision_aaba_shutdown_v1 as AABA_SHUTDOWN
import vision_pull_record_checkpoint_v3 as AABA_CHECKPOINT
import vision_pull_record_job_v3 as JOB
import vision_pull_record_worker_v1 as WORKER

BASE = V1.BASE
OWNED = V1.OWNED
GATE = V1.GATE
REMOTE = V1.REMOTE
REPORT_NAME = WORKER.REPORT_NAME
EXECUTION_SCHEMA = "vision_pull_record_execution_v3"
PRIMARY_ERROR_SCHEMA = "vision_pull_record_primary_error_v2"
FAILURE_ATOM_SCHEMA = "vision_pull_record_failure_atom_v2"
AABA_REPORT_NAME = "native-aaba-preflight.report.json"
AABA_FAILURE_NAME = "native-aaba-preflight.failure.json"
AABA_PRIMARY_NAME = "native-aaba-preflight.primary-error.json"
AABA_CONTINUATION_NAME = "native-aaba-continuation.json"
POSTFLIGHT_RESERVE_S = float(JOB.LIMITS["postflight_reserve_s_per_policy"])
FAILURE_POLL_S = 0.25


class ChildFailure(RuntimeError):
    def __init__(self, failure: Mapping[str, Any], first: Mapping[str, Any]) -> None:
        super().__init__(str(first["error_message"]))
        self.failure = dict(failure)
        self.first = dict(first)


class ShutdownPolicyFailure(RuntimeError):
    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__(str(receipt["error_message"]))
        self.receipt = dict(receipt)


def _descriptor(path: Path) -> dict[str, Any]:
    target = JOB._regular(path.absolute(), "pull record evidence")
    return {
        "file": target.name,
        "sha256": JOB._sha(target),
        "bytes": target.stat().st_size,
    }


def _primary_error(error: BaseException) -> dict[str, Any]:
    primary = getattr(error, "primary", error)
    serialization = getattr(error, "serialization_error", None)
    return {
        "schema": PRIMARY_ERROR_SCHEMA,
        "error_type": f"{type(primary).__module__}.{type(primary).__qualname__}",
        "error_message": str(primary),
        "traceback": "".join(
            traceback.format_exception(type(primary), primary, primary.__traceback__)
        ),
        "serialization_error": (
            None
            if serialization is None
            else {
                "error_type": f"{type(serialization).__module__}.{type(serialization).__qualname__}",
                "error_message": str(serialization),
            }
        ),
    }


def _aaba_validated(validated: Mapping[str, Any]) -> dict[str, Any]:
    adapted = dict(validated)
    stage = Path(validated["job_path"]).parent
    adapted["output_directory"] = stage / "aaba-output"
    adapted["job"] = {
        **dict(validated["job"]),
        "mode": AABA_CHECKPOINT.MODE,
        "callback_variant": AABA_CHECKPOINT.VARIANT,
        "output_directory": str(stage / "aaba-output"),
    }
    return adapted


def _aaba_failure(stage: Path) -> ChildFailure:
    output = stage / "aaba-output"
    failure = BASE._read(BASE._regular(output / AABA_FAILURE_NAME))
    descriptor = failure.get("first_cause")
    if not isinstance(descriptor, Mapping):
        raise TypeError("AABA first cause descriptor가 없다")
    first_path = output / str(descriptor.get("file", ""))
    if _descriptor(first_path) != dict(descriptor):
        raise ValueError("AABA first cause 원문이 descriptor와 다르다")
    first = BASE._read(first_path)
    if (
        failure.get("schema") != "vision_pull_record_aaba_failure_v2"
        or failure.get("completed") is not False
        or first.get("schema") != PRIMARY_ERROR_SCHEMA
        or not isinstance(first.get("traceback"), str)
    ):
        raise ValueError("AABA failure terminal이 다르다")
    return ChildFailure(failure, first)


def aaba_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_diagnostic: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """독립 aaba-output을 만들고 app.close 전에 첫 오류를 봉인한다."""
    validated = JOB.validate_job(job_path)
    adapted = _aaba_validated(validated)
    output = Path(adapted["output_directory"])
    report_path = output / AABA_REPORT_NAME
    app = None
    primary: BaseException | None = None
    try:
        if app_factory is None:
            app_factory = AABA_SHUTDOWN.sdk_app_factory
        app = app_factory(adapted)
        selected = run_diagnostic or AABA_CHECKPOINT.run_annotator_pull_diagnostic
        report = selected(
            adapted, app, seal_report=lambda value: JOB.write_new(report_path, value)
        )
        if not AABA._diagnostic(adapted, report):
            raise ValueError("checkpoint별 native AABA 진단이 유효하지 않다")
    except BaseException as error:  # noqa: BLE001 - close 전에 최초 원인을 봉인한다.
        primary = error
        output.mkdir(parents=False, exist_ok=True)
        try:
            first_path = output / AABA_PRIMARY_NAME
            JOB.write_new(first_path, _primary_error(error))
            JOB.write_new(
                output / AABA_FAILURE_NAME,
                {
                    "schema": "vision_pull_record_aaba_failure_v2",
                    "completed": False,
                    "job_sha256": validated["job_sha256"],
                    "checkpoint_sha256": validated["checkpoint_sha256"],
                    "first_cause": _descriptor(first_path),
                    "partial_report": _descriptor(report_path) if report_path.exists() else None,
                    "capture_approved": False,
                    "used_for_decision": False,
                },
            )
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as close_error:  # noqa: BLE001
                primary = primary or close_error
                try:
                    JOB.write_new(
                        output / "aaba-shutdown-failure.json",
                        {
                            "schema": "vision_pull_record_aaba_shutdown_failure_v2",
                            "error_type": type(close_error).__name__,
                            "error_message": str(close_error),
                            "first_error_already_sealed": (output / AABA_FAILURE_NAME).exists(),
                        },
                    )
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def _verify_aaba_terminal(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["job_path"]).parent / "aaba-output"
    if (output / AABA_FAILURE_NAME).exists():
        raise _aaba_failure(Path(validated["job_path"]).parent)
    report = BASE._read(BASE._regular(output / AABA_REPORT_NAME))
    adapted = _aaba_validated(validated)
    if (
        not AABA._diagnostic(adapted, report)
        or report.get("worker", {}).get("policy_probe", {}).get("checkpoint_sha256")
        != validated["checkpoint_sha256"]
        or report.get("worker", {}).get("policy_probe", {}).get("environment_steps") != 1
    ):
        raise ValueError("checkpoint별 native AABA receipt가 다르다")
    AABA_SHUTDOWN.verify(adapted)
    return report


def verify_aaba(
    validated: Mapping[str, Any], gpu_observation: Mapping[str, Any]
) -> dict[str, Any]:
    report = _verify_aaba_terminal(validated)
    adapted = _aaba_validated(validated)
    REMOTE.verify_observation_reference(
        gpu_observation,
        validated["resource"]["resource_id"],
        validated["resource"]["gpu_uuid"],
    )
    AABA_SHUTDOWN.verify_process_exit(adapted, gpu_observation)
    return report


def _continuation_body(
    validated: Mapping[str, Any], aaba_report: Mapping[str, Any], gpu_observation: Mapping[str, Any]
) -> dict[str, Any]:
    stage = Path(validated["job_path"]).parent
    return {
        "schema": "vision_pull_record_phase_continuation_v2",
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "resource_proof_sha256": validated["job"]["resource_proof"]["sha256"],
        "aaba_report": _descriptor(stage / "aaba-output" / AABA_REPORT_NAME),
        "aaba_process_exit": _descriptor(stage / AABA_SHUTDOWN.PROCESS_EXIT_NAME),
        "aaba_diagnostic_complete": aaba_report.get("diagnostic_complete") is True,
        "gpu_observation": dict(gpu_observation),
        "capture_approved": False,
        "used_for_decision": False,
    }


def _verify_continuation(validated: Mapping[str, Any]) -> dict[str, Any]:
    stage = Path(validated["job_path"]).parent
    value = BASE._read(BASE._regular(stage / AABA_CONTINUATION_NAME))
    expected_report = _descriptor(stage / "aaba-output" / AABA_REPORT_NAME)
    expected_process_exit = _descriptor(stage / AABA_SHUTDOWN.PROCESS_EXIT_NAME)
    if (
        value.get("schema") != "vision_pull_record_phase_continuation_v2"
        or value.get("job_sha256") != validated["job_sha256"]
        or value.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or value.get("resource_proof_sha256")
        != validated["job"]["resource_proof"]["sha256"]
        or value.get("aaba_report") != expected_report
        or value.get("aaba_process_exit") != expected_process_exit
        or value.get("aaba_diagnostic_complete") is not True
        or value.get("capture_approved") is not False
        or value.get("used_for_decision") is not False
    ):
        raise ValueError("AABA 단계 연속 receipt가 다르다")
    observation = value.get("gpu_observation")
    if not isinstance(observation, Mapping):
        raise TypeError("AABA 이후 GPU 관찰 원문 참조가 없다")
    REMOTE.verify_observation_reference(
        observation,
        validated["resource"]["resource_id"],
        validated["resource"]["gpu_uuid"],
    )
    AABA_SHUTDOWN.verify_process_exit(_aaba_validated(validated), observation)
    return value


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_worker: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """초기 freshness 대신 결합된 AABA 연속 receipt로 두 번째 phase를 승인한다."""
    validated = JOB.validate_job(job_path, require_fresh=False)
    _verify_continuation(validated)
    output = Path(validated["output_directory"])
    app = None
    primary: BaseException | None = None
    report_path = output / REPORT_NAME
    try:
        if app_factory is None:
            import vision_play_worker_v3 as V3

            app_factory = V3._sdk_app_factory
        app = app_factory(validated)
        selected = run_worker or WORKER.run_policy_record
        report = selected(
            validated, app, seal_report=lambda value: JOB.write_new(report_path, value)
        )
        if report.get("technical_recording_complete") is not True:
            raise ValueError("pull record worker가 기술 완료를 증명하지 않았다")
        JOB.write_new(output / "result.json", V1._result_body(validated, report, report_path))
    except BaseException as error:  # noqa: BLE001 - close 전에 첫 실패를 봉인한다.
        primary = error
        output.mkdir(parents=True, exist_ok=True)
        try:
            first_path = output / V1.PRIMARY_ERROR_NAME
            JOB.write_new(first_path, V1._primary_error(error))
            JOB.write_new(
                output / "failure.json",
                {
                    "schema": JOB.FAILURE_SCHEMA,
                    "completed": False,
                    "technical_recording_complete": False,
                    "mode": JOB.MODE,
                    "job_sha256": validated["job_sha256"],
                    "checkpoint_sha256": validated["checkpoint_sha256"],
                    "checkpoint_logical_id": validated["job"]["checkpoint"]["logical_id"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "first_cause": _descriptor(first_path),
                    "partial_report": _descriptor(report_path) if report_path.exists() else None,
                    "capture_approved": False,
                    "model_ready": False,
                    "used_for_decision": False,
                },
            )
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as close_error:  # noqa: BLE001
                primary = primary or close_error
                try:
                    JOB.write_new(
                        output / "shutdown_failure.json",
                        {
                            "schema": "vision_pull_record_shutdown_failure_v2",
                            "completed": False,
                            "job_sha256": validated["job_sha256"],
                            "error_type": type(close_error).__name__,
                            "error_message": str(close_error),
                        },
                    )
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def _aaba_failure_if_present(stage: Path) -> ChildFailure | None:
    path = stage / "aaba-output" / AABA_FAILURE_NAME
    return _aaba_failure(stage) if path.exists() else None


def _wait_for_child(
    process: Any,
    *,
    timeout_s: float,
    phase_name: str,
    stage: Path,
    poll_s: float = FAILURE_POLL_S,
    validated: Mapping[str, Any] | None = None,
) -> int:
    """짧게 poll하며 AABA failure atom을 wall timeout보다 먼저 반환한다."""
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("child wait 시간은 양수여야 한다")
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(getattr(process, "args", phase_name), timeout_s)
        try:
            return int(process.wait(timeout=min(poll_s, remaining)))
        except subprocess.TimeoutExpired:
            if phase_name == "aaba":
                failure = _aaba_failure_if_present(stage)
                if failure is not None:
                    raise failure
                if validated is None:
                    raise ValueError("AABA shutdown failure 검증에 입장 정보가 없다")
                policy_failure = AABA_SHUTDOWN.policy_failure(
                    _aaba_validated(validated)
                )
                if policy_failure is not None:
                    raise ShutdownPolicyFailure(policy_failure)
            elif phase_name == "record" and (stage / "output" / "failure.json").exists():
                if validated is None:
                    raise ValueError("record failure 검증에 입장 시 결합 정보가 없다")
                raise V1._verify_failure(validated)


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = OWNED._process_identity,
    cleanup: Callable[..., bool] = OWNED._safe_cleanup,
) -> dict[str, Any]:
    """AABA와 record를 한 deadline에서 실행하고 첫 실패를 우선 보존한다."""
    validated = JOB.validate_job(job_path)
    stage = Path(validated["job_path"]).parent
    if (stage / "execution.json").exists() or any(
        (stage / f"{name}-launch.json").exists() for name in ("aaba", "record")
    ):
        raise FileExistsError("pull record job을 반복 실행하지 않는다")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("pull record 실제 실행은 소유 Linux group에서 수행한다")
    started = time.monotonic()
    deadline = started + JOB.LIMITS["max_wall_s_per_policy"]
    if gpu_reader is None:
        resource = validated["resource"]
        gpu_reader = REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=BASE._gpu_query_run(deadline),
        )
    reader = gpu_reader
    if not callable(reader):
        raise TypeError("pull record GPU reader가 없다")
    before = Path(job_path).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "completed": False,
        "technical_recording_complete": False,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "checkpoint_logical_id": validated["job"]["checkpoint"]["logical_id"],
        "exit_code": None,
        "timed_out": False,
        "cleanup_verified": False,
        "gpu_before": None,
        "gpu_after": None,
        "error_type": None,
        "error_message": None,
        "result_sha256": None,
        "failure_atom": None,
        "wall_s": None,
        "full_recordings": 0,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
        "resource_proof_sha256": validated["job"]["resource_proof"]["sha256"],
        "per_checkpoint_aaba_required": True,
        "aaba_receipt": None,
        "phases": [],
    }
    process = None
    active_launch: Mapping[str, Any] | None = None
    primary: BaseException | None = None

    def launch_phase(name: str, flag: str) -> dict[str, Any]:
        nonlocal active_launch, process
        phase: dict[str, Any] = {
            "name": name,
            "exit_code": None,
            "cleanup_verified": False,
            "gpu_before": reader(),
            "gpu_after": None,
            "launch": None,
        }
        if phase["gpu_before"]:
            raise RuntimeError(f"{name} 시작 전 등록 GPU에 compute process가 있다")
        command = [
            str(validated["job"]["python"]), "-I", "-S", "-c", GATE.OWNED_WORKER_GATE,
            str(validated["job"]["python"]), "-B", str(Path(__file__).resolve()),
            flag, "--job", str(Path(job_path).absolute()),
        ]
        JOB.write_new(stage / f"{name}-launch.json", {"command": command, "cwd": validated["job"]["isaaclab"]})
        receipt["phases"].append(phase)
        with (stage / f"{name}-stdout.log").open("xb") as stdout, (stage / f"{name}-stderr.log").open("xb") as stderr:
            process = popen(
                command,
                cwd=validated["job"]["isaaclab"],
                env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"},
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.PIPE,
                start_new_session=True,
                shell=False,
            )
            phase["launch"] = dict(identify(process.pid))
            active_launch = phase["launch"]
            if process.stdin is None:
                raise RuntimeError(f"{name} gate stdin이 없다")
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            remaining = deadline - time.monotonic() - POSTFLIGHT_RESERVE_S
            if remaining <= 0:
                raise TimeoutError("pull record child 시작 전 postflight 예산이 없다")
            try:
                phase["exit_code"] = _wait_for_child(
                    process,
                    timeout_s=remaining,
                    phase_name=name,
                    stage=stage,
                    validated=validated,
                )
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
            finally:
                phase["cleanup_verified"] = cleanup(phase["launch"], process)
                receipt["cleanup_verified"] = phase["cleanup_verified"]
                process = None
                active_launch = None
        phase["gpu_after"] = reader()
        if phase["cleanup_verified"] is not True or phase["gpu_after"] != []:
            raise RuntimeError(f"{name} cleanup/GPU idle 증거가 실패했다")
        return phase

    try:
        receipt["gpu_before"] = reader()
        if receipt["gpu_before"]:
            raise RuntimeError("등록 GPU에 다른 compute process가 있다")
        aaba_phase = launch_phase("aaba", "--aaba-child")
        if aaba_phase["exit_code"] != 0:
            failure = _aaba_failure_if_present(stage)
            if failure is not None:
                raise failure
            if receipt["timed_out"]:
                raise TimeoutError("AABA child가 제한 시간 안에 종료되지 않았다")
            raise RuntimeError("checkpoint별 native AABA child가 실패했다")
        adapted = _aaba_validated(validated)
        _verify_aaba_terminal(validated)
        observation = dict(BASE._gpu_reference(reader, validated))
        AABA_SHUTDOWN.write_process_exit(
            adapted,
            exit_code=int(aaba_phase["exit_code"]),
            cleanup_verified=bool(aaba_phase["cleanup_verified"]),
            gpu_observation=observation,
        )
        aaba_report = verify_aaba(validated, observation)
        continuation = _continuation_body(validated, aaba_report, observation)
        JOB.write_new(stage / AABA_CONTINUATION_NAME, continuation)
        record_output = Path(validated["output_directory"])
        if record_output.exists() or record_output.is_symlink():
            raise FileExistsError("record output이 AABA 단계 전에 생성됐다")
        record_output.mkdir()
        receipt["aaba_receipt"] = {
            "report": continuation["aaba_report"],
            "shutdown_policy": _descriptor(stage / AABA_SHUTDOWN.POLICY_NAME),
            "close_entered": _descriptor(stage / AABA_SHUTDOWN.CLOSE_ENTERED_NAME),
            "close_returned": (
                _descriptor(stage / AABA_SHUTDOWN.CLOSE_RETURNED_NAME)
                if (stage / AABA_SHUTDOWN.CLOSE_RETURNED_NAME).exists()
                else None
            ),
            "process_exit": _descriptor(stage / AABA_SHUTDOWN.PROCESS_EXIT_NAME),
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "diagnostic_complete": True,
            "capture_approved": False,
            "used_for_decision": False,
        }
        if Path(job_path).read_bytes() != before:
            raise ValueError("AABA preflight가 원 job을 바꿨다")
        record_phase = launch_phase("record", "--sdk-child")
        receipt["exit_code"] = record_phase["exit_code"]
        receipt["cleanup_verified"] = record_phase["cleanup_verified"]
        receipt["gpu_after"] = record_phase["gpu_after"]
        if Path(job_path).read_bytes() != before:
            raise ValueError("pull record 실행 중 job 원문이 바뀌었다")
        terminal = V1.verify_terminal(validated)
        if receipt["exit_code"] != 0:
            raise RuntimeError("pull record 성공 terminal과 nonzero exit가 함께 있다")
        receipt["result_sha256"] = JOB._sha(Path(validated["output_directory"]) / "result.json")
        receipt["completed"] = terminal["completed"] is True
        receipt["technical_recording_complete"] = terminal["technical_recording_complete"] is True
        receipt["full_recordings"] = 1
    except (ChildFailure, V1.ChildFailure) as error:
        primary = error
        failure_output = stage / ("aaba-output" if isinstance(error, ChildFailure) else "output")
        failure_name = AABA_FAILURE_NAME if isinstance(error, ChildFailure) else "failure.json"
        receipt["failure_atom"] = {
            "schema": FAILURE_ATOM_SCHEMA,
            "status": "failed",
            "failure": _descriptor(failure_output / failure_name),
            "first_cause": error.failure["first_cause"],
            "first_cause_type": error.first["error_type"],
            "first_cause_message": error.first["error_message"],
        }
        receipt["error_type"] = error.first["error_type"]
        receipt["error_message"] = error.first["error_message"]
    except ShutdownPolicyFailure as error:
        primary = error
        policy_failure_descriptor = _descriptor(
            stage / AABA_SHUTDOWN.POLICY_FAILURE_NAME
        )
        receipt["failure_atom"] = {
            "schema": FAILURE_ATOM_SCHEMA,
            "status": "failed",
            "failure": policy_failure_descriptor,
            "first_cause": policy_failure_descriptor,
            "first_cause_type": error.receipt["error_type"],
            "first_cause_message": error.receipt["error_message"],
        }
        receipt["error_type"] = error.receipt["error_type"]
        receipt["error_message"] = error.receipt["error_message"]
    except BaseException as error:  # noqa: BLE001
        primary = error
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)
    finally:
        if process is not None:
            try:
                receipt["cleanup_verified"] = cleanup(active_launch, process)
            except BaseException as cleanup_error:  # noqa: BLE001
                primary = primary or cleanup_error
        if receipt["gpu_after"] is None:
            try:
                receipt["gpu_after"] = reader()
                if receipt["phases"] and receipt["phases"][-1]["gpu_after"] is None:
                    receipt["phases"][-1]["gpu_after"] = receipt["gpu_after"]
            except BaseException as gpu_error:  # noqa: BLE001
                primary = primary or gpu_error
        receipt["wall_s"] = time.monotonic() - started
        if primary is not None:
            receipt["completed"] = False
            receipt["technical_recording_complete"] = False
            receipt["full_recordings"] = 0
        JOB.write_new(stage / "execution.json", receipt)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--aaba-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--checkpoint-logical-id", choices=JOB.CHECKPOINT_IDS)
    args = parser.parse_args(argv)
    if args.sdk_child or args.aaba_child:
        if args.job is None:
            parser.error("child 실행에는 --job이 필요하다")
        return aaba_child(args.job) if args.aaba_child else sdk_child(args.job)
    if args.job is None:
        required = (args.protocol, args.resource_proof, args.resource_sha256, args.isaaclab, args.python, args.stage, args.checkpoint_logical_id)
        if any(value is None for value in required):
            parser.error("job 생성 인자가 모두 필요하다")
        args.job = JOB.create_job(
            args.protocol, args.resource_proof, args.resource_sha256,
            args.isaaclab, args.python, args.stage, args.checkpoint_logical_id,
        )
    result = run_job(args.job)
    return 0 if result.get("completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
