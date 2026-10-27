"""Native pull 정책 녹화를 소유 SDK child와 부모 deadline 경계로 실행한다."""

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

import run_vision_annotator_pull_v2 as AABA
import run_vision_prepared_record_v2 as PARENT
import vision_annotator_pull_checkpoint_v2 as AABA_CHECKPOINT
import vision_pull_record_job_v1 as JOB
import vision_pull_record_worker_v1 as WORKER

BASE = PARENT.V1.BASE
OWNED = PARENT.V1.OWNED
GATE = PARENT.V1.GATE
REMOTE = PARENT.V1.REMOTE
REPORT_NAME = WORKER.REPORT_NAME
PRIMARY_ERROR_NAME = "primary-error.json"
EXECUTION_SCHEMA = "vision_pull_record_execution_v1"
PRIMARY_ERROR_SCHEMA = "vision_pull_record_primary_error_v1"
FAILURE_ATOM_SCHEMA = "vision_pull_record_failure_atom_v1"
POSTFLIGHT_RESERVE_S = 200.0
AABA_REPORT_NAME = "native-aaba-preflight.report.json"
AABA_FAILURE_NAME = "native-aaba-preflight.failure.json"


class ChildFailure(RuntimeError):
    def __init__(self, failure: dict[str, Any], first: dict[str, Any]) -> None:
        super().__init__(str(first["error_message"]))
        self.failure, self.first = failure, first


def _descriptor(path: Path) -> dict[str, Any]:
    target = JOB._regular(path.absolute(), "pull record evidence")
    return {"file": target.name, "sha256": JOB._sha(target), "bytes": target.stat().st_size}


def _result_body(
    validated: Mapping[str, Any], report: Mapping[str, Any], report_path: Path
) -> dict[str, Any]:
    return {
        "schema": JOB.RESULT_SCHEMA,
        "completed": True,
        "technical_recording_complete": True,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "checkpoint_logical_id": validated["job"]["checkpoint"]["logical_id"],
        "report": _descriptor(report_path),
        "worker": dict(report),
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 1,
    }


def verify_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    if (output / "failure.json").exists():
        raise ValueError("pull record 성공과 실패 terminal이 함께 있다")
    result = BASE._read(BASE._regular(output / "result.json"))
    report_path = BASE._regular(output / REPORT_NAME)
    report = BASE._read(report_path)
    WORKER.verify_startup(report.get("startup"))
    manifest_path = BASE._regular(output / WORKER.MANIFEST_NAME)
    manifest = BASE._read(manifest_path)
    artifacts = report.get("artifacts")
    artifact_paths: dict[str, Path] = {}
    if not isinstance(artifacts, Mapping):
        raise TypeError("pull record Play artifact descriptor가 없다")
    for key, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise TypeError("pull record artifact descriptor가 mapping이 아니다")
        matches = [
            path for path in output.rglob(str(descriptor.get("file", "")))
            if path.is_file() and not path.is_symlink()
        ]
        if len(matches) != 1 or _descriptor(matches[0]) != dict(descriptor):
            raise ValueError("pull record artifact 원문과 descriptor가 다르다")
        artifact_paths[str(key)] = matches[0]
    rebuilt = WORKER._manifest(validated, artifact_paths)
    if (
        report.get("schema") != WORKER.SCHEMA
        or report.get("completed") is not True
        or report.get("technical_recording_complete") is not True
        or report.get("frame_count") != 625
        or report.get("fps") != 50
        or report.get("writer_calls") != {"create": 0, "attach": 0, "callback": 0, "detach": 0}
        or report.get("atomicity_claimed") is not False
        or report.get("capture_approved") is not False
        or report.get("model_ready") is not False
        or report.get("used_for_decision") is not False
        or report.get("native_capture_manifest") != _descriptor(manifest_path)
        or manifest != rebuilt
        or result != _result_body(validated, report, report_path)
    ):
        raise ValueError("pull record terminal/manifest 결합이 다르다")
    return result


def _primary_error(error: BaseException) -> dict[str, Any]:
    primary = getattr(error, "primary", error)
    return {
        "schema": PRIMARY_ERROR_SCHEMA,
        "error_type": f"{type(primary).__module__}.{type(primary).__qualname__}",
        "error_message": str(primary),
        "traceback": "".join(traceback.format_exception(type(primary), primary, primary.__traceback__)),
    }


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_worker: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """app.close 전에 성공 또는 첫 실패 terminal을 봉인한다."""
    validated = JOB.validate_job(job_path)
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
        JOB.write_new(output / "result.json", _result_body(validated, report, report_path))
    except BaseException as error:  # noqa: BLE001 - SDK close 전 첫 실패를 봉인한다.
        primary = error
        output.mkdir(parents=True, exist_ok=True)
        try:
            first_path = output / PRIMARY_ERROR_NAME
            JOB.write_new(first_path, _primary_error(error))
            JOB.write_new(output / "failure.json", {
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
            })
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:  # noqa: BLE001
                primary = primary or error
                try:
                    JOB.write_new(output / "shutdown_failure.json", {
                        "schema": "vision_pull_record_shutdown_failure_v1",
                        "completed": False,
                        "job_sha256": validated["job_sha256"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    })
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def _aaba_validated(validated: Mapping[str, Any]) -> dict[str, Any]:
    adapted = dict(validated)
    adapted["job"] = {
        **dict(validated["job"]),
        "mode": AABA_CHECKPOINT.MODE,
        "callback_variant": AABA_CHECKPOINT.VARIANT,
    }
    return adapted


def aaba_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_diagnostic: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """같은 checkpoint를 별도 앱에서 AABA 진단하고 원 job은 바꾸지 않는다."""
    validated = JOB.validate_job(job_path)
    adapted = _aaba_validated(validated)
    stage = Path(validated["job_path"]).parent
    report_path = stage / AABA_REPORT_NAME
    app = None
    primary: BaseException | None = None
    try:
        if app_factory is None:
            import vision_play_worker_v3 as V3

            app_factory = V3._sdk_app_factory
        app = app_factory(adapted)
        selected = run_diagnostic or AABA_CHECKPOINT.run_annotator_pull_diagnostic
        report = selected(
            adapted, app, seal_report=lambda value: JOB.write_new(report_path, value)
        )
        if not AABA._diagnostic(adapted, report):
            raise ValueError("checkpoint별 native AABA 진단이 유효하지 않다")
    except BaseException as error:  # noqa: BLE001
        primary = error
        try:
            JOB.write_new(stage / AABA_FAILURE_NAME, {
                "schema": "vision_pull_record_aaba_failure_v1",
                "completed": False,
                "job_sha256": validated["job_sha256"],
                "checkpoint_sha256": validated["checkpoint_sha256"],
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error_message": str(error),
                "partial_report": _descriptor(report_path) if report_path.exists() else None,
            })
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:  # noqa: BLE001
                primary = primary or error
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def verify_aaba(validated: Mapping[str, Any]) -> dict[str, Any]:
    stage = Path(validated["job_path"]).parent
    if (stage / AABA_FAILURE_NAME).exists():
        raise RuntimeError("checkpoint별 native AABA 진단이 실패했다")
    report = BASE._read(BASE._regular(stage / AABA_REPORT_NAME))
    adapted = _aaba_validated(validated)
    if (
        not AABA._diagnostic(adapted, report)
        or report.get("worker", {}).get("policy_probe", {}).get("checkpoint_sha256")
        != validated["checkpoint_sha256"]
        or report.get("worker", {}).get("policy_probe", {}).get("environment_steps") != 1
        or report.get("worker", {}).get("lifecycle", {}).get("annotators_detached") is not True
        or report.get("worker", {}).get("lifecycle", {}).get("environment_closed") is not True
    ):
        raise ValueError("checkpoint별 native AABA receipt가 다르다")
    return report


def _verify_failure(validated: Mapping[str, Any]) -> ChildFailure:
    output = Path(validated["output_directory"])
    failure = BASE._read(BASE._regular(output / "failure.json"))
    if (
        failure.get("schema") != JOB.FAILURE_SCHEMA
        or failure.get("completed") is not False
        or failure.get("technical_recording_complete") is not False
        or failure.get("job_sha256") != validated["job_sha256"]
        or failure.get("checkpoint_sha256") != validated["checkpoint_sha256"]
    ):
        raise ValueError("pull record failure terminal이 다르다")
    descriptor = failure.get("first_cause")
    path = output / str(descriptor.get("file", "")) if isinstance(descriptor, Mapping) else output
    if not isinstance(descriptor, Mapping) or _descriptor(path) != dict(descriptor):
        raise ValueError("pull record first cause descriptor가 다르다")
    first = BASE._read(path)
    if (
        first.get("schema") != PRIMARY_ERROR_SCHEMA
        or not isinstance(first.get("error_type"), str)
        or not isinstance(first.get("error_message"), str)
        or not isinstance(first.get("traceback"), str)
    ):
        raise ValueError("pull record first cause 원문이 다르다")
    return ChildFailure(failure, first)


def verify_terminal(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    success, failure = (output / "result.json").exists(), (output / "failure.json").exists()
    if success and failure:
        raise ValueError("pull record 성공·실패 terminal이 함께 있다")
    if failure:
        raise _verify_failure(validated)
    if not success:
        raise FileNotFoundError("pull record terminal이 없다")
    return verify_result(validated)


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
            stage / "gpu-observations", run=BASE._gpu_query_run(deadline),
        )
    if gpu_reader is None:
        raise TypeError("pull record GPU reader가 없다")
    reader = gpu_reader
    before = Path(job_path).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA, "completed": False,
        "technical_recording_complete": False, "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "checkpoint_logical_id": validated["job"]["checkpoint"]["logical_id"],
        "exit_code": None, "timed_out": False, "cleanup_verified": False,
        "launch": None, "gpu_before": None, "gpu_after": None,
        "gpu_before_observation": None, "gpu_after_observation": None,
        "result_sha256": None, "failure_atom": None,
        "error_type": None, "error_message": None, "wall_s": None,
        "capture_approved": False, "model_ready": False, "used_for_decision": False,
        "training_actions": 0, "model_api_requests": 0, "full_recordings": 0,
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
            "name": name, "exit_code": None, "cleanup_verified": False,
            "gpu_before": None, "gpu_after": None, "launch": None,
        }
        command = [
            str(validated["job"]["python"]), "-I", "-S", "-c", GATE.OWNED_WORKER_GATE,
            str(validated["job"]["python"]), "-B", str(Path(__file__).resolve()),
            flag, "--job", str(Path(job_path).absolute()),
        ]
        phase["gpu_before"] = reader()
        if phase["gpu_before"]:
            raise RuntimeError(f"{name} 시작 전 등록 GPU에 compute process가 있다")
        JOB.write_new(
            stage / f"{name}-launch.json",
            {"command": command, "cwd": validated["job"]["isaaclab"]},
        )
        with (stage / f"{name}-stdout.log").open("xb") as stdout, (
            stage / f"{name}-stderr.log"
        ).open("xb") as stderr:
            process = popen(
                command, cwd=validated["job"]["isaaclab"],
                env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"},
                stdout=stdout, stderr=stderr, stdin=subprocess.PIPE,
                start_new_session=True, shell=False,
            )
            try:
                phase["launch"] = dict(identify(process.pid))
                active_launch = phase["launch"]
            except BaseException:
                phase["cleanup_verified"] = GATE._cleanup_waiting_gate(process)
                process = None
                raise
            if process.stdin is None:
                raise RuntimeError(f"{name} gate stdin이 없다")
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            remaining = deadline - time.monotonic() - POSTFLIGHT_RESERVE_S
            if remaining <= 0:
                raise TimeoutError("pull record child 시작 전 postflight 예산이 없다")
            try:
                phase["exit_code"] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                raise
            finally:
                try:
                    phase["cleanup_verified"] = cleanup(phase["launch"], process)
                finally:
                    process = None
                    active_launch = None
        phase["gpu_after"] = reader()
        if phase["cleanup_verified"] is not True or phase["gpu_after"] != []:
            raise RuntimeError(f"{name} cleanup/GPU idle 증거가 실패했다")
        receipt["phases"].append(phase)
        return phase

    try:
        receipt["gpu_before"] = reader()
        if receipt["gpu_before"]:
            raise RuntimeError("등록 GPU에 다른 compute process가 있다")
        receipt["gpu_before_observation"] = dict(BASE._gpu_reference(reader, validated))
        aaba_phase = launch_phase("aaba", "--aaba-child")
        receipt["launch"] = aaba_phase["launch"]
        receipt["cleanup_verified"] = aaba_phase["cleanup_verified"]
        if aaba_phase["exit_code"] != 0:
            raise RuntimeError("checkpoint별 native AABA child가 실패했다")
        aaba_report = verify_aaba(validated)
        receipt["aaba_receipt"] = {
            "report": _descriptor(stage / AABA_REPORT_NAME),
            "checkpoint_sha256": aaba_report["worker"]["policy_probe"]["checkpoint_sha256"],
            "diagnostic_complete": aaba_report["diagnostic_complete"],
            "capture_approved": False,
            "used_for_decision": False,
        }
        if Path(job_path).read_bytes() != before:
            raise ValueError("AABA preflight가 원 job을 바꿨다")
        record_phase = launch_phase("record", "--sdk-child")
        receipt["exit_code"] = record_phase["exit_code"]
        receipt["launch"] = record_phase["launch"]
        receipt["cleanup_verified"] = record_phase["cleanup_verified"]
        receipt["gpu_after"] = reader()
        receipt["gpu_after_observation"] = dict(BASE._gpu_reference(reader, validated))
        if Path(job_path).read_bytes() != before:
            raise ValueError("pull record 실행 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        if receipt["cleanup_verified"] is not True or receipt["gpu_after"] != []:
            raise RuntimeError("pull record cleanup/GPU idle 증거가 실패했다")
        terminal = verify_terminal(validated)
        if receipt["exit_code"] != 0:
            raise RuntimeError("pull record 성공 terminal과 nonzero exit가 함께 있다")
        receipt["result_sha256"] = JOB._sha(Path(validated["output_directory"]) / "result.json")
        receipt["completed"] = terminal["completed"] is True
        receipt["technical_recording_complete"] = terminal["technical_recording_complete"] is True
        receipt["full_recordings"] = 1
    except ChildFailure as error:
        primary = error
        receipt["failure_atom"] = {
            "schema": FAILURE_ATOM_SCHEMA, "status": "failed",
            "child_exit_code": receipt["exit_code"], "child_exit_accepted": False,
            "failure": _descriptor(Path(validated["output_directory"]) / "failure.json"),
            "first_cause": error.failure["first_cause"],
            "first_cause_type": error.first["error_type"],
            "first_cause_message": error.first["error_message"],
        }
        receipt["error_type"] = error.first["error_type"]
        receipt["error_message"] = error.first["error_message"]
    except BaseException as error:  # noqa: BLE001
        primary = error
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)
    finally:
        if process is not None and receipt["cleanup_verified"] is not True:
            try:
                receipt["cleanup_verified"] = cleanup(active_launch, process)
            except BaseException as cleanup_error:  # noqa: BLE001
                primary = primary or cleanup_error
        if receipt["gpu_after"] is None:
            try:
                receipt["gpu_after"] = reader()
                receipt["gpu_after_observation"] = getattr(reader, "last_receipt", None)
            except BaseException as gpu_error:  # noqa: BLE001
                primary = primary or gpu_error
        receipt["wall_s"] = time.monotonic() - started
        if receipt["wall_s"] >= JOB.LIMITS["max_wall_s_per_policy"]:
            primary = primary or TimeoutError("pull record 600초 상한 초과")
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
        values = (
            args.protocol, args.resource_proof, args.resource_sha256, args.isaaclab,
            args.python, args.stage, args.checkpoint_logical_id,
        )
        if any(value is None for value in values):
            parser.error("job 또는 prepare 인자가 모두 필요하다")
        args.job = JOB.create_job(*values)
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
