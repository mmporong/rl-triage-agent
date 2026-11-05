from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from evals import registered_gpu_admission_journal_v1 as J


@pytest.fixture
def journal_root(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "ROOT", tmp_path)
    monkeypatch.setattr(J, "PRIVATE", tmp_path / "bench/private")
    return tmp_path / "bench/private/own-run/journal"


def test_same_result_object_and_query_count(journal_root, monkeypatch):
    expected = {"busy": True, "fixture_callbacks_used": True}
    calls = []

    def query():
        calls.append(1)
        return expected

    monkeypatch.setattr(J.gpu, "query_registered_desktop_gpu_retry", query)
    with J.admission_journal_scope(journal_root):
        assert J.gpu.query_registered_desktop_gpu_retry() is expected
        assert J.gpu.query_registered_desktop_gpu_retry() is expected
    assert J.gpu.query_registered_desktop_gpu_retry is query
    assert calls == [1, 1]
    row = json.loads((journal_root / "000001.result.json").read_bytes())
    assert row["snapshot"] == expected
    assert row["SDK"] == row["API"] == 0
    assert len(list(journal_root.iterdir())) == 4


def test_original_error_identity_and_receipt_preserved(journal_root, monkeypatch):
    cause = ValueError("unregistered desktop process")
    error = J.gpu.RetryAdmissionError("denied", receipt={"decision": "rejected", "busy": True})
    error.__cause__ = cause

    def query():
        raise error

    monkeypatch.setattr(J.gpu, "query_registered_desktop_gpu_retry", query)
    with J.admission_journal_scope(journal_root), pytest.raises(J.gpu.RetryAdmissionError) as raised:
        J.gpu.query_registered_desktop_gpu_retry()
    assert raised.value is error
    row = json.loads((journal_root / "000001.result.json").read_bytes())
    assert [r["error_type"] for r in row["exception_chain"]] == ["RetryAdmissionError", "ValueError"]
    assert row["error_receipt"] == error.receipt
    assert J.gpu.query_registered_desktop_gpu_retry is query


def test_nested_same_directory_restores_after_outer(journal_root, monkeypatch):
    original = lambda: {"busy": False}
    monkeypatch.setattr(J.gpu, "query_registered_desktop_gpu_retry", original)
    with J.admission_journal_scope(journal_root):
        wrapped = J.gpu.query_registered_desktop_gpu_retry
        with J.admission_journal_scope(journal_root):
            assert J.gpu.query_registered_desktop_gpu_retry is wrapped
        with pytest.raises(J.AdmissionJournalError, match="nested"), J.admission_journal_scope(journal_root / "other"):
            pass
        assert J.gpu.query_registered_desktop_gpu_retry is wrapped
    assert J.gpu.query_registered_desktop_gpu_retry is original


def test_other_thread_cannot_query(journal_root):
    errors = []

    def run():
        try:
            J.gpu.query_registered_desktop_gpu_retry()
        except J.AdmissionJournalError as error:
            errors.append(error)

    with J.admission_journal_scope(journal_root):
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], J.AdmissionJournalError)
    assert not list(journal_root.iterdir())


def test_forbidden_is_rejected_before_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(J.os, "lstat", lambda *a: pytest.fail("filesystem touched"))
    with pytest.raises(J.AdmissionJournalError):
        J._folder(tmp_path / J.FORBIDDEN / "new")


@pytest.mark.parametrize("path", ["..", "child/../other"])
def test_parent_traversal_rejected(journal_root, path):
    with pytest.raises(J.AdmissionJournalError):
        J._folder(journal_root / path)


def test_existing_folder_is_not_overwritten(journal_root):
    journal_root.mkdir(parents=True)
    sentinel = journal_root / "original"
    sentinel.write_bytes(b"old")
    with pytest.raises(FileExistsError), J.admission_journal_scope(journal_root):
        pass
    assert sentinel.read_bytes() == b"old"


def test_intent_write_failure_never_calls_query(journal_root, monkeypatch):
    monkeypatch.setattr(J, "_write", lambda *a: (_ for _ in ()).throw(OSError("storage unavailable")))
    monkeypatch.setattr(J.gpu, "query_registered_desktop_gpu_retry", lambda: pytest.fail("query called"))
    with J.admission_journal_scope(journal_root), pytest.raises(OSError):
        J.gpu.query_registered_desktop_gpu_retry()


def test_result_write_failure_rejects_success(journal_root, monkeypatch):
    write = J._write

    def failing(path: Path, body):
        if body.get("status") == "returned":
            raise OSError("storage unavailable")
        write(path, body)

    monkeypatch.setattr(J, "_write", failing)
    monkeypatch.setattr(J.gpu, "query_registered_desktop_gpu_retry", lambda: {"busy": False})
    with J.admission_journal_scope(journal_root), pytest.raises(OSError):
        J.gpu.query_registered_desktop_gpu_retry()
