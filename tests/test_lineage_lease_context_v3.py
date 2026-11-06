# pyright: reportMissingImports=false
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("lineage_lease_context_v3")


class Kernel:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.share = None
        self.disposition = None
        self.closed = []

    def CreateFileW(self, _path, access, share, _security, disposition, _flags, _template):
        assert access == M.GENERIC_READ
        self.share = share
        self.disposition = disposition
        return 41

    def GetFileSizeEx(self, _handle, pointer):
        pointer._obj.value = len(self.raw)
        return True

    def ReadFile(self, _handle, buffer, size, pointer, _overlapped):
        assert size == len(self.raw)
        buffer[:size] = self.raw
        pointer._obj.value = size
        return True

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return True


def test_windows_reader_requests_share_7_and_closes_handle(tmp_path):
    kernel = Kernel(b'{"state":"running"}')
    value = M._read_windows_json(tmp_path / "gpu-lease.json", kernel=kernel)
    assert value == {"state": "running"}
    assert kernel.share == M.FILE_SHARE_READ_WRITE_DELETE == 7
    assert kernel.disposition == M.OPEN_EXISTING
    assert kernel.closed == [41]


@pytest.mark.parametrize("winerror", [5, 32])
def test_windows_busy_errors_preserve_v2_retry_fields(tmp_path, winerror):
    path = (tmp_path / "gpu-lease.json").resolve()
    error = M._winerror(path, winerror)
    assert isinstance(error, PermissionError)
    assert error.errno == 13
    assert error.winerror == winerror
    assert Path(error.filename).resolve() == path


def test_noncanonical_path_delegates_to_original_reader(tmp_path):
    target = tmp_path / "other.json"
    marker = {"source": "original"}
    with patch.object(M, "_ORIGINAL_READ", return_value=marker) as reader:
        assert M.shared_read(target) is marker
    reader.assert_called_once_with(target)


def test_reader_scope_restores_after_failure(tmp_path):
    failure = RuntimeError("validation failed")
    with patch.object(M.V2, "lease_context", side_effect=failure) as base, \
            pytest.raises(RuntimeError) as caught:
        M.lease_context(tmp_path)
    assert caught.value is failure
    base.assert_called_once_with(tmp_path)
    assert M.V2._READ is M._ORIGINAL_READ


def test_reader_scope_holds_v2_lock_against_other_threads():
    entered = threading.Event()

    def use_v2_scope():
        with M.V2.read_scope():
            entered.set()

    with M.reader_scope():
        thread = threading.Thread(target=use_v2_scope)
        thread.start()
        assert not entered.wait(timeout=0.05)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert entered.is_set()


def test_v2_validation_entrypoint_consumes_shared_reader(tmp_path):
    canonical = M.V2.canonical_path()
    marker = {"owner": "validated"}

    def base_context(evidence_dir):
        assert evidence_dir == tmp_path
        assert M.V2._READ is M.shared_read
        return M.V2._READ(canonical)

    with patch.object(M, "_read_windows_json", return_value=marker) as reader, \
            patch.object(M.V2, "lease_context", side_effect=base_context):
        assert M.lease_context(tmp_path) is marker
    reader.assert_called_once_with(canonical)
    assert M.V2._READ is M._ORIGINAL_READ


@pytest.mark.skipif(os.name != "nt", reason="Windows atomic replace 공유 모드 회귀")
def test_actual_windows_atomic_replace_can_coexist_with_fresh_reads(tmp_path):
    path = tmp_path / "gpu-lease.json"
    path.write_text('{"revision":0}', encoding="utf-8")
    start = threading.Event()
    done = threading.Event()
    failures = []

    def writer():
        start.wait()
        try:
            for revision in range(1, 101):
                replacement = tmp_path / f"lease-{revision}.tmp"
                replacement.write_text(json.dumps({"revision": revision}), encoding="utf-8")
                for attempt in range(20):
                    try:
                        os.replace(replacement, path)
                        break
                    except PermissionError:
                        if attempt == 19:
                            raise
                        time.sleep(0.005)
        except OSError as error:  # pragma: no cover - asserted below
            failures.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=writer)
    thread.start()
    start.set()
    observations = []
    while not done.is_set():
        observations.append(M._read_windows_json(path)["revision"])
    thread.join(timeout=5)
    observations.append(M._read_windows_json(path)["revision"])
    assert not thread.is_alive()
    assert failures == []
    assert observations
    assert all(type(value) is int and 0 <= value <= 100 for value in observations)


def test_existing_v2_validation_and_lock_are_not_bypassed(tmp_path):
    canonical = M.V2.canonical_path()
    metadata = {"owner": "raw"}
    validated = {"owner": "validated"}

    def context():
        current = M.V2.B.read(canonical)
        result = M.V2.B.validate_lease(current, [], 123, None)
        M.V2.B.assert_exclusive_lock(canonical.with_name("gpu-lease.lock"))
        return result

    with patch.object(M, "_read_windows_json", return_value=metadata), \
            patch.object(M.V2, "_LEASE_CONTEXT", side_effect=context), \
            patch.object(M.V2.B, "validate_lease", return_value=validated) as validate, \
            patch.object(M.V2.B, "assert_exclusive_lock") as lock:
        assert M.lease_context(tmp_path) is validated
    validate.assert_called_once_with(metadata, [], 123, None)
    lock.assert_called_once_with(canonical.with_name("gpu-lease.lock"))
