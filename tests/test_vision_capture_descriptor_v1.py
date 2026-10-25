from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

F = importlib.import_module("test_vision_atomic_capture_v1")
C = F.C


@pytest.fixture(autouse=True)
def _resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C.R, "WIDTH", 10)
    monkeypatch.setattr(C.R, "HEIGHT", 10)


def _descriptor(job: Path, ordinal: int) -> Path:
    return job.with_name(f"{job.stem}.writer-descriptor-{ordinal:06d}.json")


def _capture(capture: Any, base: Any, rep: Any, ordinal: int) -> Any:
    return capture(
        base,
        rep,
        [],
        F.PROFILE,
        [3.0, 3.0, 2.0] if ordinal != 2 else [-3.0, 3.0, 2.0],
        [0.0, 0.0, 0.35],
        ordinal,
    )


def test_actual_writer_callback_error_keeps_descriptor_and_detaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, base, rep, _events = F._setup(
        tmp_path, monkeypatch, mode=C.R.RENDER_SYNC_SMOKE_MODE
    )
    original = F._payload

    def unavailable_rgb(ordinal: int, *, smoke: bool) -> dict[str, Any]:
        payload = original(ordinal, smoke=smoke)
        payload["annotators"]["rgb"]["RenderProduct_0"]["data"] = None
        return payload

    monkeypatch.setattr(F, "_payload", unavailable_rgb)
    capture, _cleanup = setup["factory"](base, rep, "/Render/RenderProduct_0", F.PROFILE)
    with pytest.raises(TypeError, match="ndarray"):
        _capture(capture, base, rep, 0)
    job = setup["validated"]["job_path"]
    body = json.loads(_descriptor(job, 0).read_text(encoding="utf-8"))
    assert body["schema"] == C.DESCRIPTOR_SCHEMA
    assert body["job_sha256"] == hashlib.sha256(job.read_bytes()).hexdigest()
    assert body["callback_count"] == 1
    assert body["callback_error_type"] == "TypeError"
    assert body["descriptor"]
    assert not rep.writers
    assert not list(tmp_path.glob("job.atomic-*"))
    # 구조 진단이 픽셀·의미 라벨 본문을 남기지 않는다.
    assert "robot/body" not in _descriptor(job, 0).read_text(encoding="utf-8")


def test_success_records_each_smoke_writer_but_only_first_record_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, mode, count, expected_files in (
        ("smoke", C.R.RENDER_SYNC_SMOKE_MODE, 4, 4),
        ("record", C.R.RECORD_MODE, 2, 1),
    ):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.setattr(C.R, "FRAME_COUNT", count)
        setup, base, rep, _events = F._setup(folder, monkeypatch, mode=mode)
        capture, cleanup = setup["factory"](
            base, rep, "/Render/RenderProduct_0", F.PROFILE
        )
        for ordinal in range(count):
            _capture(capture, base, rep, ordinal)
        cleanup()
        paths = sorted(folder.glob("*.writer-descriptor-*.json"))
        assert len(paths) == expected_files
        assert all(
            json.loads(path.read_text(encoding="utf-8"))["callback_error_type"] is None
            for path in paths
        )
        assert len(list(folder.glob("job.atomic-*.json"))) == count


def test_existing_descriptor_is_preserved_and_prevents_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, base, rep, _events = F._setup(
        tmp_path, monkeypatch, mode=C.R.RENDER_SYNC_SMOKE_MODE
    )
    job = setup["validated"]["job_path"]
    path = _descriptor(job, 0)
    original = b'{"preserved":true}\n'
    path.write_bytes(original)
    capture, _cleanup = setup["factory"](base, rep, "/Render/RenderProduct_0", F.PROFILE)
    with pytest.raises(FileExistsError):
        _capture(capture, base, rep, 0)
    assert path.read_bytes() == original
    assert not rep.writers
    assert not list(tmp_path.glob("job.atomic-*"))


def test_descriptor_keeps_existing_capture_evidence_exact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = importlib.import_module("test_vision_execution_evidence_v2")
    monkeypatch.setattr(C.R, "WIDTH", 5)
    monkeypatch.setattr(C.R, "HEIGHT", 1)
    stage, job, result = evidence._stage(tmp_path, C.R.RENDER_SYNC_SMOKE_MODE)
    writer = SimpleNamespace(
        last_payload_descriptor={"schema": "structure_only"},
        callback_count=1,
        callback_error=None,
    )
    path = C._write_writer_descriptor({"job_path": job}, 0, writer)
    assert path is not None and path.parent == stage
    manifest = evidence.E2.write_manifest(job, result, sdk_child_exit_code=0)
    assert evidence.E2.verify_manifest(manifest, job, result)["completed"]


def test_descriptor_storage_error_does_not_hide_primary_callback_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, base, rep, _events = F._setup(
        tmp_path, monkeypatch, mode=C.R.RENDER_SYNC_SMOKE_MODE
    )
    original = F._payload

    def wrong_rgb(ordinal: int, *, smoke: bool) -> dict[str, Any]:
        payload = original(ordinal, smoke=smoke)
        payload["annotators"]["rgb"]["RenderProduct_0"]["data"] = np.zeros(
            (10, 10, 4), dtype=np.float32
        )
        return payload

    monkeypatch.setattr(F, "_payload", wrong_rgb)

    def unavailable_storage(*_args: Any) -> None:
        raise OSError("descriptor storage unavailable")

    monkeypatch.setattr(C, "_write_writer_descriptor", unavailable_storage)
    capture, _cleanup = setup["factory"](base, rep, "/Render/RenderProduct_0", F.PROFILE)
    with pytest.raises(ValueError, match="uint8"):
        _capture(capture, base, rep, 0)
    assert not rep.writers
    assert not list(tmp_path.glob("job.atomic-*"))
