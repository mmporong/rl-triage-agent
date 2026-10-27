"""AABA terminal을 결합한 phase 전용 Isaac 앱 종료 정책."""

from __future__ import annotations

import json
import stat
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_vision_annotator_pull_v2 as AABA
import vision_pull_record_job_v3 as JOB

SCHEMA = "vision_aaba_shutdown_policy_v1"
SETTING = "/app/fastShutdown"
REPORT_NAME = "native-aaba-preflight.report.json"
FAILURE_NAME = "native-aaba-preflight.failure.json"
POLICY_NAME = "aaba-shutdown-policy.json"
CLOSE_ENTERED_NAME = "aaba-close-entered.json"
CLOSE_RETURNED_NAME = "aaba-close-returned.json"
POLICY_FAILURE_NAME = "aaba-close-policy-failure.json"
PROCESS_EXIT_NAME = "aaba-process-exit.json"


def _regular(path: Path, label: str) -> Path:
    target = Path(path)
    if (
        not target.is_file()
        or target.is_symlink()
        or any(parent.is_symlink() for parent in target.parents)
    ):
        raise ValueError(f"{label}가 regular file이 아니다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label}가 단일 regular file이 아니다")
    return target


def _descriptor(path: Path) -> dict[str, Any]:
    target = _regular(path, "AABA 종료 증거")
    return {"file": target.name, "sha256": JOB._sha(target), "bytes": target.stat().st_size}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(_regular(path, "AABA 종료 JSON").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("AABA 종료 JSON이 object가 아니다")
    return value


def _success_terminal(validated: Mapping[str, Any], output: Path) -> dict[str, Any]:
    report_path = _regular(output / REPORT_NAME, "AABA report")
    report = _read(report_path)
    if (
        not AABA._diagnostic(validated, report)
        or report.get("primary_error") is not None
        or report.get("diagnostic_complete") is not True
    ):
        raise ValueError("AABA 성공 report가 유효하지 않다")
    worker = report.get("worker")
    probe = worker.get("policy_probe") if isinstance(worker, Mapping) else None
    lifecycle = worker.get("lifecycle") if isinstance(worker, Mapping) else None
    if (
        not isinstance(probe, Mapping)
        or probe.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or probe.get("environment_steps") != 1
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("environment_closed") is not True
    ):
        raise ValueError("AABA report의 checkpoint/step/environment close 결합이 다르다")
    artifacts = report.get("raw_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise ValueError("AABA raw artifact가 네 개가 아니다")
    for ordinal, row in enumerate(artifacts):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != ordinal
            or row.get("complete_capture") is not True
        ):
            raise ValueError("AABA raw artifact 순서가 다르다")
        for key in ("npz", "receipt"):
            descriptor = row.get(key)
            if not isinstance(descriptor, Mapping):
                raise TypeError("AABA raw artifact descriptor가 없다")
            artifact = output / str(descriptor.get("file", ""))
            if _descriptor(artifact) != dict(descriptor):
                raise ValueError("AABA raw artifact SHA/bytes가 report와 다르다")
    return {"kind": "success_report", **_descriptor(report_path)}


def _failure_terminal(validated: Mapping[str, Any], output: Path) -> dict[str, Any]:
    failure_path = _regular(output / FAILURE_NAME, "AABA failure")
    failure = _read(failure_path)
    first = failure.get("first_cause")
    if (
        failure.get("schema") != "vision_pull_record_aaba_failure_v2"
        or failure.get("completed") is not False
        or not isinstance(first, Mapping)
    ):
        raise ValueError("AABA failure terminal이 유효하지 않다")
    first_path = output / str(first.get("file", ""))
    if _descriptor(first_path) != dict(first):
        raise ValueError("AABA first cause descriptor가 원문과 다르다")
    first_value = _read(first_path)
    if (
        first_value.get("schema") != "vision_pull_record_primary_error_v2"
        or not isinstance(first_value.get("traceback"), str)
        or failure.get("job_sha256") != validated["job_sha256"]
        or failure.get("checkpoint_sha256") != validated["checkpoint_sha256"]
    ):
        raise ValueError("AABA first cause/job/checkpoint 결합이 다르다")
    partial = failure.get("partial_report")
    report_path = output / REPORT_NAME
    if report_path.exists() or report_path.is_symlink():
        if not isinstance(partial, Mapping) or _descriptor(report_path) != dict(partial):
            raise ValueError("AABA failure partial report가 원문과 다르다")
    elif partial is not None:
        raise ValueError("AABA failure가 없는 partial report를 가리킨다")
    return {"kind": "failure", **_descriptor(failure_path)}


def terminal(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    present = [
        name
        for name in (REPORT_NAME, FAILURE_NAME)
        if (output / name).exists() or (output / name).is_symlink()
    ]
    if FAILURE_NAME in present:
        return _failure_terminal(validated, output)
    if present == [REPORT_NAME]:
        return _success_terminal(validated, output)
    raise ValueError("AABA success/failure terminal이 정확히 하나가 아니다")


def _body(validated: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "completed": True,
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "phase": "after_aaba_terminal_before_sdk_close",
        "setting": SETTING,
        "startup_fast_shutdown": False,
        "before": False,
        "after": True,
        "terminal_evidence": dict(evidence),
        "wait_for_replicator": False,
    }


def apply(
    validated: Mapping[str, Any], settings: Any, *, startup_fast_shutdown: bool
) -> dict[str, Any]:
    if startup_fast_shutdown is not False:
        raise ValueError("AABA app 생성 직후 fastShutdown이 false가 아니다")
    evidence = terminal(validated)
    before = settings.get_as_bool(SETTING)
    if before is not False:
        raise ValueError("AABA terminal 뒤 fastShutdown 시작값이 false가 아니다")
    settings.set_bool(SETTING, True)
    after = settings.get_as_bool(SETTING)
    if after is not True:
        raise RuntimeError("AABA fastShutdown true 전환 readback이 실패했다")
    value = _body(validated, evidence)
    JOB.write_new(Path(validated["job_path"]).parent / POLICY_NAME, value)
    return value


def verify(validated: Mapping[str, Any]) -> dict[str, Any]:
    stage = Path(validated["job_path"]).parent
    expected = _body(validated, terminal(validated))
    policy = _read(stage / POLICY_NAME)
    if policy != expected:
        raise ValueError("AABA shutdown policy가 terminal과 다르다")
    entered = _read(stage / CLOSE_ENTERED_NAME)
    if entered != {
        "schema": "vision_aaba_close_entered_v1",
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "policy_sha256": JOB._sha(stage / POLICY_NAME),
        "wait_for_replicator": False,
    }:
        raise ValueError("AABA close entered receipt가 다르다")
    returned_path = stage / CLOSE_RETURNED_NAME
    if returned_path.exists() or returned_path.is_symlink():
        returned = _read(returned_path)
        if returned != {
            "schema": "vision_aaba_close_returned_v1",
            "job_sha256": validated["job_sha256"],
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "returned": True,
            "error_type": None,
            "error_message": None,
        }:
            raise ValueError("AABA close returned receipt가 다르다")
    if (stage / POLICY_FAILURE_NAME).exists() or (stage / POLICY_FAILURE_NAME).is_symlink():
        raise ValueError("AABA 성공과 close policy failure가 함께 있다")
    return policy


def close_returned_observed(validated: Mapping[str, Any]) -> bool:
    verify(validated)
    path = Path(validated["job_path"]).parent / CLOSE_RETURNED_NAME
    return path.exists() and not path.is_symlink()


def process_exit_body(
    validated: Mapping[str, Any],
    *,
    exit_code: int,
    cleanup_verified: bool,
    gpu_observation: Mapping[str, Any],
) -> dict[str, Any]:
    if exit_code != 0 or cleanup_verified is not True:
        raise ValueError("AABA process exit 승인에는 exit 0과 owned cleanup이 필요하다")
    stage = Path(validated["job_path"]).parent
    verify(validated)
    if not isinstance(gpu_observation, Mapping) or not gpu_observation:
        raise TypeError("AABA process exit GPU 원문 참조가 없다")
    return {
        "schema": "vision_aaba_process_exit_v1",
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "policy_sha256": JOB._sha(stage / POLICY_NAME),
        "close_entered_sha256": JOB._sha(stage / CLOSE_ENTERED_NAME),
        "close_returned_observed": close_returned_observed(validated),
        "exit_code": 0,
        "cleanup_verified": True,
        "gpu_after": [],
        "gpu_observation": dict(gpu_observation),
    }


def write_process_exit(
    validated: Mapping[str, Any],
    *,
    exit_code: int,
    cleanup_verified: bool,
    gpu_observation: Mapping[str, Any],
) -> dict[str, Any]:
    value = process_exit_body(
        validated,
        exit_code=exit_code,
        cleanup_verified=cleanup_verified,
        gpu_observation=gpu_observation,
    )
    JOB.write_new(Path(validated["job_path"]).parent / PROCESS_EXIT_NAME, value)
    return value


def verify_process_exit(
    validated: Mapping[str, Any], gpu_observation: Mapping[str, Any]
) -> dict[str, Any]:
    expected = process_exit_body(
        validated,
        exit_code=0,
        cleanup_verified=True,
        gpu_observation=gpu_observation,
    )
    value = _read(Path(validated["job_path"]).parent / PROCESS_EXIT_NAME)
    if value != expected:
        raise ValueError("AABA process exit receipt가 부모 관찰과 다르다")
    return value


def policy_failure(validated: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(validated["job_path"]).parent / POLICY_FAILURE_NAME
    if not (path.exists() or path.is_symlink()):
        return None
    value = _read(path)
    if (
        value.get("schema") != "vision_aaba_close_policy_failure_v1"
        or value.get("job_sha256") != validated["job_sha256"]
        or value.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or not isinstance(value.get("error_type"), str)
        or not isinstance(value.get("error_message"), str)
    ):
        raise ValueError("AABA close policy failure receipt가 다르다")
    return value


class AppProxy:
    """AABA terminal 뒤에만 fast shutdown을 켜고 SDK app을 닫는다."""

    def __init__(self, app: Any, validated: Mapping[str, Any], settings: Any) -> None:
        self._app = app
        self._validated = validated
        self._settings = settings
        self._startup_fast_shutdown = settings.get_as_bool(SETTING)
        if self._startup_fast_shutdown is not False:
            raise ValueError("AABA app 생성 직후 fastShutdown이 false가 아니다")

    def update(self) -> Any:
        return self._app.update()

    def close(self) -> Any:
        stage = Path(self._validated["job_path"]).parent
        policy_error: BaseException | None = None
        try:
            apply(
                self._validated,
                self._settings,
                startup_fast_shutdown=self._startup_fast_shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - raw app close는 항상 시도한다.
            policy_error = error
            JOB.write_new(
                stage / POLICY_FAILURE_NAME,
                {
                    "schema": "vision_aaba_close_policy_failure_v1",
                    "job_sha256": self._validated["job_sha256"],
                    "checkpoint_sha256": self._validated["checkpoint_sha256"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        JOB.write_new(
            stage / CLOSE_ENTERED_NAME,
            {
                "schema": "vision_aaba_close_entered_v1",
                "job_sha256": self._validated["job_sha256"],
                "checkpoint_sha256": self._validated["checkpoint_sha256"],
                "policy_sha256": (
                    JOB._sha(stage / POLICY_NAME) if policy_error is None else None
                ),
                "wait_for_replicator": False,
            },
        )
        close_error: BaseException | None = None
        result: Any = None
        try:
            result = self._app.close(wait_for_replicator=False)
        except BaseException as error:  # noqa: BLE001 - 두 오류를 분리해 기록한다.
            close_error = error
        JOB.write_new(
            stage / CLOSE_RETURNED_NAME,
            {
                "schema": "vision_aaba_close_returned_v1",
                "job_sha256": self._validated["job_sha256"],
                "checkpoint_sha256": self._validated["checkpoint_sha256"],
                "returned": close_error is None,
                "error_type": None if close_error is None else type(close_error).__name__,
                "error_message": None if close_error is None else str(close_error),
            },
        )
        if policy_error is not None:
            if close_error is not None:
                traceback.print_exception(close_error)
            raise policy_error
        if close_error is not None:
            raise close_error
        return result


def sdk_app_factory(validated: Mapping[str, Any]) -> AppProxy:
    import carb.settings  # pyright: ignore[reportMissingImports]
    import record_vision_evidence_v1 as RECORD
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = AppLauncher(**RECORD.fixed_profile()["launcher"]).app
    try:
        return AppProxy(app, validated, carb.settings.get_settings())
    except BaseException:
        app.close(wait_for_replicator=False)
        raise
