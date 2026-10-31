from __future__ import annotations

import errno
import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("lineage_lease_context_v2")


def permission(path: Path) -> PermissionError:
    error = PermissionError(errno.EACCES, "Access is denied", str(path))
    error.winerror = 5
    return error


def call_read(path: Path):
    return M.B.read(path)


def test_actual_base_lease_context_retries_fresh_read_then_revalidates(tmp_path):
    canonical = M.canonical_path()
    fresh = {"childPid": 456, "state": "running", "marker": "fresh-metadata"}
    reads = []
    values = iter([permission(canonical), fresh])

    def reader(path):
        reads.append(Path(path).resolve())
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    validated = []

    def validate(metadata, ancestors, caller, now):
        validated.append(metadata)
        return {**metadata, "validated": True}

    response = SimpleNamespace(stdout="[]")
    with patch.object(M, "_READ", side_effect=reader), \
            patch.object(M, "_SLEEP"), patch.object(M.B.subprocess, "run", return_value=response), \
            patch.object(M.B, "validate_lease", side_effect=validate), \
            patch.object(M.B, "assert_exclusive_lock"):
        result = M.lease_context(tmp_path / "evidence")

    assert result == {**fresh, "validated": True}
    assert reads == [canonical, canonical]
    assert validated == [fresh]
    evidence = json.loads(
        (tmp_path / "evidence" / M.EVIDENCE_NAME).read_text(encoding="utf-8")
    )
    assert [row["status"] for row in evidence["attempts"]] == [
        "permission_error", "fresh_success"
    ]
    assert evidence["permission_error_attempts"] == 1
    assert evidence["fresh_read_success"] is True
    assert M.B.read is M._REGISTERED_READ


def test_noncanonical_permission_error_is_not_retried(tmp_path):
    other = (tmp_path / "other.json").resolve()
    error = permission(other)
    reads = []

    def reader(path):
        reads.append(Path(path).resolve())
        raise error

    with patch.object(M, "_READ", side_effect=reader), \
            patch.object(M, "_LEASE_CONTEXT", side_effect=lambda: call_read(other)), \
            patch.object(M, "_SLEEP") as sleep, pytest.raises(PermissionError) as caught:
        M.lease_context(tmp_path / "evidence")
    assert caught.value is error and reads == [other]
    sleep.assert_not_called()
    evidence = json.loads(
        (tmp_path / "evidence" / M.EVIDENCE_NAME).read_text(encoding="utf-8")
    )
    assert evidence["permission_error_attempts"] == 1
    assert evidence["fresh_read_success"] is False
    assert evidence["failure"]["filename"] == str(other)


def test_json_decode_error_is_not_retried(tmp_path):
    canonical = M.canonical_path()
    error = json.JSONDecodeError("bad", "{", 1)
    with patch.object(M, "_READ", side_effect=error) as reader, \
            patch.object(M, "_LEASE_CONTEXT",
                         side_effect=lambda: call_read(canonical)), \
            patch.object(M, "_SLEEP") as sleep, pytest.raises(json.JSONDecodeError) as caught:
        M.lease_context(tmp_path / "evidence")
    assert caught.value is error and reader.call_count == 1
    sleep.assert_not_called()
    evidence = json.loads(
        (tmp_path / "evidence" / M.EVIDENCE_NAME).read_text(encoding="utf-8")
    )
    assert evidence["attempts"] == []
    assert evidence["failure"]["type"] == "JSONDecodeError"


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def test_persistent_canonical_permission_error_stops_at_attempt_limit(tmp_path):
    canonical = M.canonical_path()
    errors = []

    def reader(_path):
        error = permission(canonical)
        errors.append(error)
        raise error

    clock = Clock()
    with patch.object(M, "_READ", side_effect=reader), \
            patch.object(M, "_LEASE_CONTEXT",
                         side_effect=lambda: call_read(canonical)), \
            patch.object(M, "_CLOCK", clock), patch.object(M, "_SLEEP", clock.sleep), \
            pytest.raises(PermissionError) as caught:
        M.lease_context(tmp_path / "evidence")
    assert len(errors) == M.MAX_ATTEMPTS == 20
    assert caught.value is errors[-1]
    assert 0 <= clock.value <= M.MAX_WAIT_S
    evidence = json.loads(
        (tmp_path / "evidence" / M.EVIDENCE_NAME).read_text(encoding="utf-8")
    )
    assert len(evidence["attempts"]) == 20
    assert evidence["permission_error_attempts"] == 20
    assert evidence["status"] == "failed"


def test_deadline_after_sleep_stops_before_another_read_and_preserves_error():
    canonical = M.canonical_path()
    error = permission(canonical)
    clock = Clock()

    def reach_deadline(_seconds):
        clock.value = M.MAX_WAIT_S

    with patch.object(M, "_READ", side_effect=error) as reader, \
            patch.object(M, "_LEASE_CONTEXT", side_effect=lambda: call_read(canonical)), \
            patch.object(M, "_CLOCK", clock), patch.object(M, "_SLEEP", reach_deadline), \
            pytest.raises(PermissionError) as caught:
        M.lease_context()
    assert caught.value is error
    assert reader.call_count == 1


def test_original_validation_failure_is_preserved_without_retry(tmp_path):
    error = ValueError("현재 controller와 공용 GPU lease 소유자가 다르다")
    with patch.object(M, "_LEASE_CONTEXT", side_effect=error), \
            patch.object(M, "_READ") as reader, pytest.raises(ValueError) as caught:
        M.lease_context(tmp_path / "evidence")
    assert caught.value is error
    reader.assert_not_called()
    assert M.B.read is M._REGISTERED_READ
    evidence = json.loads(
        (tmp_path / "evidence" / M.EVIDENCE_NAME).read_text(encoding="utf-8")
    )
    assert evidence["failure"]["type"] == "ValueError"
    assert evidence["attempts"] == []


@pytest.mark.parametrize("failure", ["owner", "lease_id", "heartbeat", "lock"])
def test_base_lease_validation_and_lock_fail_closed_without_retry(tmp_path, failure):
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 회귀도 실행한다.
    caller = os.getpid()
    wrapper = caller + 100000
    metadata = {
        "owner": M.B.OWNER,
        "state": "running",
        "completedUtc": None,
        "wrapperPid": wrapper,
        "childPid": caller,
        "acquiredUtc": now.isoformat(),
        "heartbeatUtc": now.isoformat(),
        "leaseId": "11111111-1111-1111-1111-111111111111",
    }
    if failure == "owner":
        metadata["owner"] = "other-owner"
    elif failure == "lease_id":
        metadata["leaseId"] = "bad"
    elif failure == "heartbeat":
        old = now - timedelta(seconds=120)
        metadata["acquiredUtc"] = old.isoformat()
        metadata["heartbeatUtc"] = old.isoformat()
    ancestors = [
        {"pid": caller, "parent": wrapper, "name": "python.exe"},
        {"pid": wrapper, "parent": 4, "name": "powershell.exe"},
    ]
    response = SimpleNamespace(stdout=json.dumps(ancestors))
    lock_error = ValueError("공용 GPU lock의 배타 보유를 확인할 수 없다")
    lock = patch.object(
        M.B, "assert_exclusive_lock", side_effect=lock_error if failure == "lock" else None
    )
    with patch.object(M, "_READ", return_value=metadata) as reader, \
            patch.object(M.B.subprocess, "run", return_value=response), lock, \
            pytest.raises(ValueError):
        M.lease_context(tmp_path / failure)
    assert reader.call_count == 1
    evidence = json.loads(
        (tmp_path / failure / M.EVIDENCE_NAME).read_text(encoding="utf-8")
    )
    assert evidence["permission_error_attempts"] == 0
    assert evidence["fresh_read_success"] is True
    assert evidence["status"] == "failed"


def test_without_evidence_dir_reraises_same_permission_error():
    canonical = M.canonical_path()
    error = permission(canonical)
    with patch.object(M, "_READ", side_effect=error), \
            patch.object(M, "_LEASE_CONTEXT", side_effect=lambda: call_read(canonical)), \
            patch.object(M, "MAX_ATTEMPTS", 1), pytest.raises(PermissionError) as caught:
        M.lease_context()
    assert caught.value is error
    assert M.B.read is M._REGISTERED_READ


def test_read_scope_restores_and_rejects_external_reader():
    with M.read_scope():
        assert M.B.read is M._retry_read
        with M.read_scope():
            assert M.B.read is M._retry_read
    assert M.B.read is M._REGISTERED_READ
    replacement = lambda _path: {}
    with patch.object(M.B, "read", replacement), \
            pytest.raises(ValueError, match="등록 원본"), M.read_scope():
        pass


def test_evidence_is_write_once_before_lease_read(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    evidence = evidence_dir / M.EVIDENCE_NAME
    evidence.write_text("preserved\n", encoding="utf-8")
    with patch.object(M, "_LEASE_CONTEXT") as context, \
            pytest.raises(FileExistsError, match="이미 존재"):
        M.lease_context(evidence_dir)
    context.assert_not_called()
    assert evidence.read_text(encoding="utf-8") == "preserved\n"
