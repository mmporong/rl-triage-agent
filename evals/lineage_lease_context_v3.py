"""Windows 공유 교체를 허용해 canonical GPU lease metadata를 읽는다."""
from __future__ import annotations

import ctypes
import errno
import json
import os
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from threading import RLock

import lineage_lease_context_v2 as V2

SOURCE_FILES = V2.SOURCE_FILES + (
    "evals/lineage_lease_context_v3.py",
    "tests/test_lineage_lease_context_v3.py",
)
GENERIC_READ = 0x80000000
# FILE_SHARE_DELETE는 열린 동안 rename/delete 요청을 허용한다.
# https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-createfilew
FILE_SHARE_READ_WRITE_DELETE = 7
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_ORIGINAL_READ = V2._READ
_LOCK = RLock()


def _kernel32():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    kernel.GetFileSizeEx.restype = wintypes.BOOL
    kernel.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _winerror(path: Path, code: int) -> OSError:
    if code in {5, 32}:
        message = "Access is denied" if code == 5 else "File is in use"
        error = PermissionError(errno.EACCES, message, str(path))
        error.winerror = code
        return error
    return OSError(code, "Windows GPU lease metadata read failed", str(path))


def _read_windows_json(path: Path, *, kernel=None) -> dict:
    """한 handle에서 UTF-8 JSON 전체를 읽고 handle을 항상 닫는다."""
    api = _kernel32() if kernel is None else kernel
    handle = api.CreateFileW(
        str(path), GENERIC_READ, FILE_SHARE_READ_WRITE_DELETE, None, OPEN_EXISTING, 0, None
    )
    if handle == INVALID_HANDLE_VALUE:
        raise _winerror(path, ctypes.get_last_error())
    try:
        size = ctypes.c_longlong()
        if not api.GetFileSizeEx(handle, ctypes.byref(size)):
            raise _winerror(path, ctypes.get_last_error())
        if size.value < 0 or size.value > 16 * 1024 * 1024:
            raise ValueError("GPU lease metadata 크기가 허용 범위를 벗어났다")
        buffer = ctypes.create_string_buffer(max(1, size.value))
        read = wintypes.DWORD()
        if size.value and not api.ReadFile(
            handle, buffer, size.value, ctypes.byref(read), None
        ):
            raise _winerror(path, ctypes.get_last_error())
        if read.value != size.value:
            raise OSError("GPU lease metadata를 한 handle에서 전부 읽지 못했다")
        value = json.loads(buffer.raw[: read.value].decode("utf-8-sig"))
        if type(value) is not dict:
            raise TypeError("GPU lease metadata는 JSON 객체여야 한다")
        return value
    finally:
        api.CloseHandle(handle)


def shared_read(path: Path) -> dict:
    """canonical metadata만 share 7로 읽고 다른 파일은 기존 reader에 위임한다."""
    target = Path(path)
    if os.name != "nt" or target.resolve() != V2.canonical_path():
        return _ORIGINAL_READ(target)
    return _read_windows_json(target)


@contextmanager
def reader_scope():
    """v2 retry reader가 소비하는 단일 read callback만 잠시 교체한다."""
    with _LOCK, V2._LOCK:
        if V2._READ is not _ORIGINAL_READ:
            raise ValueError("v2 lease reader가 등록 원본이 아니다")
        V2._READ = shared_read
        try:
            yield
        finally:
            V2._READ = _ORIGINAL_READ


def lease_context(evidence_dir: Path | None = None) -> dict:
    """v2의 소유자·PID·heartbeat·배타 lock 검증을 그대로 실행한다."""
    with reader_scope():
        return V2.lease_context(evidence_dir)
