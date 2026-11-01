from __future__ import annotations

import errno
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from evals import lease_metadata_reader_v1 as M


def permission(*, winerror: int | None = 32, err: int = errno.EACCES) -> PermissionError:
    error = PermissionError(err, "lease metadata is busy", "gpu-lease.json")
    if winerror is not None:
        error.winerror = winerror
    return error


class Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_permission_errors_retry_then_parse(winerror: int,
                                                              tmp_path: Path) -> None:
    metadata = {"owner": "registered", "state": "running"}
    values: list[object] = [permission(winerror=winerror), json.dumps(metadata).encode()]
    clock = Clock()

    def reader(_path: Path) -> bytes:
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    assert M.read_lease_metadata(tmp_path / "gpu-lease.json", monotonic=clock,
                                 sleep=clock.sleep, reader=reader) == metadata
    assert clock.sleeps == [M.RETRY_INTERVAL_S]


def test_errno_eacces_without_winerror_is_transient(tmp_path: Path) -> None:
    values: list[object] = [permission(winerror=None), b'{"leaseId":"fresh"}']
    clock = Clock()

    def reader(_path: Path) -> bytes:
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    assert M.read_lease_metadata(tmp_path / "gpu-lease.json", monotonic=clock,
                                 sleep=clock.sleep, reader=reader)["leaseId"] == "fresh"


def test_invalid_json_and_non_object_are_not_retried(tmp_path: Path) -> None:
    for raw, error_type in ((b"{", json.JSONDecodeError), (b"[]", ValueError)):
        calls = 0
        sleeps: list[float] = []

        def reader(_path: Path) -> bytes:
            nonlocal calls
            calls += 1
            return raw

        with pytest.raises(error_type):
            M.read_lease_metadata(tmp_path / "gpu-lease.json", reader=reader,
                                  sleep=sleeps.append)
        assert calls == 1 and sleeps == []


def test_non_transient_permission_and_other_io_errors_are_not_retried(tmp_path: Path) -> None:
    failures = [permission(winerror=123, err=errno.EPERM), FileNotFoundError("missing")]
    for failure in failures:
        calls = 0
        sleeps: list[float] = []

        def reader(_path: Path) -> bytes:
            nonlocal calls
            calls += 1
            raise failure

        with pytest.raises(type(failure)) as caught:
            M.read_lease_metadata(tmp_path / "gpu-lease.json", reader=reader,
                                  sleep=sleeps.append)
        assert caught.value is failure and calls == 1 and sleeps == []


def test_attempt_limit_preserves_last_error_and_bounds_sleep(tmp_path: Path) -> None:
    errors: list[PermissionError] = []
    clock = Clock()

    def reader(_path: Path) -> bytes:
        error = permission(winerror=32)
        errors.append(error)
        raise error

    with pytest.raises(PermissionError) as caught:
        M.read_lease_metadata(tmp_path / "gpu-lease.json", max_attempts=4, max_wait_s=10,
                              retry_interval_s=.05, monotonic=clock,
                              sleep=clock.sleep, reader=reader)
    assert len(errors) == 4 and caught.value is errors[-1]
    assert clock.value == pytest.approx(.15) and len(clock.sleeps) == 3


def test_deadline_after_sleep_stops_before_next_read(tmp_path: Path) -> None:
    error = permission(winerror=33)
    calls = 0
    clock = Clock()

    def reader(_path: Path) -> bytes:
        nonlocal calls
        calls += 1
        raise error

    def reach_deadline(_seconds: float) -> None:
        clock.value = .2

    with pytest.raises(PermissionError) as caught:
        M.read_lease_metadata(tmp_path / "gpu-lease.json", max_attempts=20,
                              max_wait_s=.2, retry_interval_s=.1,
                              monotonic=clock, sleep=reach_deadline, reader=reader)
    assert caught.value is error and calls == 1


def _locked_file_process(path: Path, ready: Path, release: Path) -> subprocess.Popen[str]:
    script = (
        "$p=$env:LEASE_TEST_PATH;$ready=$env:LEASE_TEST_READY;$release=$env:LEASE_TEST_RELEASE;"
        "$s=[System.IO.File]::Open($p,[System.IO.FileMode]::Open,[System.IO.FileAccess]::ReadWrite,"
        "[System.IO.FileShare]::None);[System.IO.File]::WriteAllText($ready,'ready');"
        "try{while(-not [System.IO.File]::Exists($release)){Start-Sleep -Milliseconds 5}}finally{$s.Dispose()}"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    environment = dict(os.environ)
    environment.update(LEASE_TEST_PATH=str(path), LEASE_TEST_READY=str(ready),
                       LEASE_TEST_RELEASE=str(release))
    return subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags,
        env=environment)


def _release_process(process: subprocess.Popen[str], release: Path) -> None:
    release.write_text("release", encoding="utf-8")
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def _wait_ready(process: subprocess.Popen[str], ready: Path) -> None:
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"lock subprocess exited: {stdout} {stderr}")
        time.sleep(.01)
    assert ready.exists(), "lock subprocess readiness timeout"


@pytest.mark.skipif(os.name != "nt", reason="Windows FileShare.None 회귀")
def test_real_fileshare_none_first_failure_then_release_success(tmp_path: Path) -> None:
    path, ready, release = (tmp_path / name for name in ("gpu-lease.json", "ready", "release"))
    path.write_text('{"leaseId":"fresh-after-release"}', encoding="utf-8")
    process = _locked_file_process(path, ready, release)
    try:
        _wait_ready(process, ready)
        calls = 0

        def reader(target: Path) -> bytes:
            nonlocal calls
            calls += 1
            return target.read_bytes()

        timer = threading.Timer(.1, lambda: release.write_text("release", encoding="utf-8"))
        timer.start()
        try:
            value = M.read_lease_metadata(path, max_wait_s=1, retry_interval_s=.02,
                                          reader=reader)
        finally:
            timer.join(timeout=1)
        assert value == {"leaseId": "fresh-after-release"} and calls >= 2
    finally:
        _release_process(process, release)


@pytest.mark.skipif(os.name != "nt", reason="Windows FileShare.None 회귀")
def test_real_fileshare_none_persistent_lock_hits_deadline(tmp_path: Path) -> None:
    path, ready, release = (tmp_path / name for name in ("gpu-lease.json", "ready", "release"))
    path.write_text('{"leaseId":"never-read-while-locked"}', encoding="utf-8")
    process = _locked_file_process(path, ready, release)
    try:
        _wait_ready(process, ready)
        started = time.monotonic()
        with pytest.raises(PermissionError):
            M.read_lease_metadata(path, max_attempts=100, max_wait_s=.15,
                                  retry_interval_s=.02)
        elapsed = time.monotonic() - started
        assert .10 <= elapsed < 1.0
    finally:
        _release_process(process, release)
