from __future__ import annotations

import ctypes
import errno
import json
import os
from ctypes import wintypes
from pathlib import Path

import pytest

from evals import lease_read_retry_v1 as subject


def test_success_returns_original_validated_object_without_retry(tmp_path: Path) -> None:
    value = {"leaseId": "validated-live-lease"}
    assert subject.read_live_lease(lambda: value, sleep=lambda _: pytest.fail("unexpected retry")) is value


def test_exact_metadata_permission_failure_reloads_and_revalidates(tmp_path: Path) -> None:
    path = tmp_path / "gpu-lease.json"
    calls, delays = [], []

    def read():
        calls.append("read-and-validate")
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, "sharing failure", str(path))
        return {"leaseId": "fresh-after-sharing-failure"}

    result = subject.read_live_lease(read, lease_path=path, sleep=delays.append)
    assert result["leaseId"] == "fresh-after-sharing-failure"
    assert calls == ["read-and-validate", "read-and-validate"]
    assert delays == [0.01]


@pytest.mark.parametrize("failure", [ValueError("lease changed"), json.JSONDecodeError("bad JSON", "", 0),
                                   FileNotFoundError("metadata missing")])
def test_authority_or_integrity_error_is_not_retried(failure: Exception, tmp_path: Path) -> None:
    path = tmp_path / "gpu-lease.json"
    calls = []

    def read():
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, "sharing failure", str(path))
        raise failure

    with pytest.raises(type(failure)):
        subject.read_live_lease(read, lease_path=path, sleep=lambda _: None)
    assert len(calls) == 2


def test_persistent_permission_failure_remains_bounded_and_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "gpu-lease.json"
    calls, delays = [], []

    def read():
        calls.append(1)
        raise PermissionError(errno.EACCES, "still inaccessible", str(path))

    with pytest.raises(PermissionError):
        subject.read_live_lease(read, lease_path=path, sleep=delays.append)
    assert len(calls) == 6 and tuple(delays) == subject.DELAYS_S
    assert sum(delays) == pytest.approx(0.31)


@pytest.mark.parametrize("filename", [None, "unrelated.json"])
def test_unrelated_permission_error_is_not_retried(filename: str | None, tmp_path: Path) -> None:
    def read():
        raise PermissionError(errno.EACCES, "unrelated access denied", filename)

    with pytest.raises(PermissionError):
        subject.read_live_lease(read, lease_path=tmp_path / "gpu-lease.json",
                                sleep=lambda _: pytest.fail("unrelated error retried"))


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_real_windows_exclusive_handle_then_fresh_validated_read(tmp_path: Path) -> None:
    path = tmp_path / "gpu-lease.json"
    path.write_text('{"leaseId":"fresh-lease"}', encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    delays, validations = [], []
    released = False

    def read():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["leaseId"] != "fresh-lease":
            raise ValueError("lease changed")
        validations.append(value["leaseId"])
        return value

    def release(_delay: float):
        nonlocal released
        delays.append(_delay)
        assert kernel.CloseHandle(handle)
        released = True

    try:
        result = subject.read_live_lease(read, lease_path=path, sleep=release)
    finally:
        if not released:
            kernel.CloseHandle(handle)
    assert result == {"leaseId": "fresh-lease"}
    assert delays == [0.01] and validations == ["fresh-lease"]
