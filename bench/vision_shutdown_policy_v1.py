"""Isaac terminal evidence 뒤 fast shutdown 전환을 봉인하는 종료 정책."""
from __future__ import annotations

import stat
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R

SCHEMA = "vision_shutdown_policy_v1"
SETTING = "/app/fastShutdown"
SIDECAR = "shutdown-policy.json"


def _single_regular(path: Path, label: str) -> Path:
    target = Path(path)
    if (not target.is_file() or target.is_symlink()
            or any(parent.is_symlink() for parent in target.parents)):
        raise ValueError(f"{label}가 regular file이 아니다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label}가 단일 regular file이 아니다")
    return target


def _terminal(validated: Mapping[str, Any]) -> Path:
    output = Path(validated["output_directory"])
    candidates = [output / "result.json", output / "failure.json"]
    present = [path for path in candidates if path.exists() or path.is_symlink()]
    if len(present) != 1:
        raise ValueError("SDK close 전 terminal evidence가 정확히 하나여야 한다")
    return _single_regular(present[0], "terminal evidence")


def apply(validated: Mapping[str, Any], settings: Any, *, startup_fast_shutdown: bool) -> dict[str, Any]:
    """terminal evidence를 확인한 뒤 fastShutdown 전이를 write-once로 기록한다."""
    if startup_fast_shutdown is not False:
        raise ValueError("launcher 직후 fastShutdown이 false가 아니다")
    terminal = _terminal(validated)
    before = settings.get_as_bool(SETTING)
    if before is not False:
        raise ValueError("terminal evidence 뒤 fastShutdown 시작값이 false가 아니다")
    settings.set_bool(SETTING, True)
    after = settings.get_as_bool(SETTING)
    if after is not True:
        raise RuntimeError("fastShutdown true 전환 readback이 실패했다")
    value = {
        "schema": SCHEMA, "completed": True, "job_sha256": validated["job_sha256"],
        "phase": "after_terminal_evidence_before_sdk_close", "setting": SETTING,
        "startup_fast_shutdown": False, "before": False, "after": True,
        "terminal_evidence": {"file": terminal.name, "sha256": R.sha256_file(terminal)},
        "wait_for_replicator": False,
    }
    R.write_new(Path(validated["job_path"]).parent / SIDECAR, value)
    return value


def verify(job_path: Path, validated: Mapping[str, Any], terminal: Path) -> dict[str, Any]:
    sidecar = _single_regular(Path(job_path).parent / SIDECAR, "shutdown policy")
    terminal_file = _single_regular(terminal, "terminal evidence")
    expected = {
        "schema": SCHEMA, "completed": True, "job_sha256": validated["job_sha256"],
        "phase": "after_terminal_evidence_before_sdk_close", "setting": SETTING,
        "startup_fast_shutdown": False, "before": False, "after": True,
        "terminal_evidence": {"file": terminal_file.name, "sha256": R.sha256_file(terminal_file)},
        "wait_for_replicator": False,
    }
    value = R.read_object(sidecar)
    if value != expected:
        raise ValueError("shutdown policy 영수증이 terminal/job과 다르다")
    return value


class AppProxy:
    """정책 실패에도 SDK close를 시도하고 원래 정책 실패를 보존한다."""

    def __init__(self, app: Any, validated: Mapping[str, Any], settings: Any) -> None:
        self._app = app
        self._validated = validated
        self._settings = settings
        self._startup_fast_shutdown = settings.get_as_bool(SETTING)
        if self._startup_fast_shutdown is not False:
            raise ValueError("App 생성 직후 fastShutdown이 false가 아니다")

    def update(self) -> Any:
        return self._app.update()

    def close(self) -> Any:
        policy_error: BaseException | None = None
        try:
            apply(self._validated, self._settings,
                  startup_fast_shutdown=self._startup_fast_shutdown)
        except BaseException as error:  # noqa: BLE001 - close를 항상 이어간다.
            policy_error = error
        try:
            result = self._app.close(wait_for_replicator=False)
        except BaseException as close_error:
            if policy_error is not None:
                traceback.print_exception(close_error)
                raise policy_error
            raise
        if policy_error is not None:
            raise policy_error
        return result
