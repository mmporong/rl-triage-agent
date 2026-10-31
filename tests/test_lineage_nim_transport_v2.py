from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import lineage_nim_transport_v2 as T


def test_atomic_writer_never_exposes_empty_or_partial_target(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "response.json"
    entered, release = threading.Event(), threading.Event()
    actual_link = os.link

    def blocked_link(source: Path, destination: Path) -> None:
        assert Path(source).read_bytes().endswith(b"\n")
        json.loads(Path(source).read_text(encoding="utf-8"))
        entered.set()
        assert release.wait(2)
        actual_link(source, destination)

    monkeypatch.setattr(T.os, "link", blocked_link)
    thread = threading.Thread(target=T._atomic_write_new, args=(target, {"complete": True}))
    thread.start()
    assert entered.wait(2)
    assert not target.exists()
    assert len(list(tmp_path.glob(".*.tmp"))) == 1
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_two_writers_publish_one_complete_value_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "response.json"
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def writer(value: int) -> None:
        barrier.wait()
        try:
            T._atomic_write_new(target, {"writer": value})
        except FileExistsError:
            outcomes.append("lost")
        else:
            outcomes.append("won")

    threads = [threading.Thread(target=writer, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert sorted(outcomes) == ["lost", "won"]
    assert json.loads(target.read_text(encoding="utf-8"))["writer"] in {1, 2}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "response.json"
    target.write_text('{"old":true}\n', encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        T._atomic_write_new(target, {"new": True})
    assert target.read_bytes() == before
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publish_failure_removes_private_temp_and_keeps_target_absent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "response.json"

    def fail(*_args, **_kwargs):
        raise OSError("publication failed")

    monkeypatch.setattr(T.os, "link", fail)
    monkeypatch.setattr(T.os, "rename", fail)
    with pytest.raises(OSError, match="publication failed"):
        T._atomic_write_new(target, {"complete": True})
    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_writer_scope_restores_t1_and_supports_same_scope_reentry(tmp_path: Path) -> None:
    original = T.T1._write_new
    with T.atomic_writer_scope():
        assert T.T1._write_new is T._atomic_write_new
        with T.atomic_writer_scope():
            T.T1._write_new(tmp_path / "inside.json", {"ok": True})
    assert T.T1._write_new is original
    assert json.loads((tmp_path / "inside.json").read_text(encoding="utf-8")) == {"ok": True}


def test_prepare_protocol_keeps_v1_model_plan_and_adds_only_atomic_sources() -> None:
    v1 = ROOT / "bench/protocols/lineage_compound_nim_20261031_v1.json"
    protocol = T.prepare_protocol(v1)
    original = T._read(v1)
    assert protocol["model"] == original["model"]
    assert protocol["plan"] == original["plan"]
    assert protocol["params"] == original["params"]
    assert protocol["publication"]["partial_target_visible"] is False
    assert protocol["publication"]["existing_target_overwrite"] is False
    assert protocol["scope"] == {"new_queue_only": True,
                                  "existing_v1_queue_recovery": False,
                                  "provider_retry": False, "duplicate_send": False}
    assert set(protocol["source_sha256_lf"]) == set(T.SOURCE_FILES)


def test_serve_one_and_worker_run_all_t1_writes_inside_atomic_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_binding", lambda _root: ({}, {}))
    writes: list[Path] = []

    def serve(_root: Path, _ref: dict, *, complete=None):
        assert T.T1._write_new is T._atomic_write_new
        path = tmp_path / "serve.json"
        T.T1._write_new(path, {"complete": complete is not None})
        writes.append(path)
        return {"ok": True}

    def worker(_root: Path, **_kwargs):
        assert T.T1._write_new is T._atomic_write_new
        path = tmp_path / "worker.json"
        T.T1._write_new(path, {"complete": True})
        writes.append(path)
        return {"status": "closed"}

    monkeypatch.setattr(T.T1, "serve_one", serve)
    monkeypatch.setattr(T.T1, "run_worker", worker)
    assert T.serve_one(tmp_path, {"request_id": "r"}, complete=lambda *_a: {}) == {"ok": True}
    assert T.run_worker(tmp_path, max_wall_s=1, complete=lambda *_a: {}) == {"status": "closed"}
    assert [json.loads(path.read_text(encoding="utf-8"))["complete"] for path in writes] == [True, True]


def test_file_completion_uses_atomic_writer_at_callback_time(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_binding", lambda _root: ({}, {}))

    def factory(*_args, **_kwargs):
        def complete(_messages: list[dict], _params: dict) -> dict:
            assert T.T1._write_new is T._atomic_write_new
            T.T1._write_new(tmp_path / "completion.json", {"valid": True})
            return {"content": "ok"}
        return complete

    monkeypatch.setattr(T.T1, "file_completion", factory)
    callback = T.file_completion(tmp_path, "slot", "a" * 64, "b" * 64, 5)
    assert callback([], {}) == {"content": "ok"}
    assert json.loads((tmp_path / "completion.json").read_text(encoding="utf-8"))["valid"] is True


def test_existing_dispatch_rejection_is_propagated_without_second_send(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_binding", lambda _root: ({}, {}))
    calls = 0

    def reject(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("기존 dispatch claim/failure가 있어 request를 재전송하지 않는다")

    monkeypatch.setattr(T.T1, "serve_one", reject)
    with pytest.raises(RuntimeError, match="재전송하지 않는다"):
        T.serve_one(tmp_path, {"request_id": "r"}, complete=lambda *_a: {})
    assert calls == 1


def test_queue_without_v2_binding_is_rejected_before_t1_use(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        T.status(tmp_path)


def test_cli_is_new_queue_v2_surface() -> None:
    source = (ROOT / "evals/lineage_nim_transport_v2.py").read_text(encoding="utf-8")
    assert '"initialize", "serve-next", "once", "close", "status", "worker"' in source
    assert "existing_v1_queue_recovery" in source
