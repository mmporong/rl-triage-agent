"""준비된 실제 protocol과 backend의 필수 소스 집합을 CPU에서 함께 대조한다."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
B = importlib.import_module("vision_brev_render_backend_v4")


def _protocol(version: int) -> dict:
    path = ROOT / f"bench/protocols/vision_brev_source_20261025_v{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _registered_source_tree(tmp_path: Path, version: int, protocol: dict) -> Path:
    """등록 protocol과 같은 commit의 소스 blob만 임시 트리에 복원한다."""
    relative_protocol = f"bench/protocols/vision_brev_source_20261025_v{version}.json"
    history = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--format=%H", "--", relative_protocol],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.splitlines()
    registration = None
    for commit in history:
        raw = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative_protocol}"],
            check=True, capture_output=True,
        ).stdout
        if json.loads(raw) == protocol:
            registration = commit
            break
    assert registration is not None, "등록된 v10 protocol commit을 찾지 못했다"
    for relative in B.SOURCE_FILES:
        raw = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{registration}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return tmp_path


def test_preserved_v8_fails_before_worker_launch() -> None:
    with pytest.raises(ValueError, match="source 결합"):
        B._verify_source_closure(_protocol(8))


def test_registered_v10_factory_accepts_the_actual_prepared_source_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol(10)
    registered_tree = _registered_source_tree(tmp_path, 10, protocol)
    calls = []
    sentinel = object()

    def parent(cls, *args, **kwargs):
        calls.append((cls, args, kwargs))
        return sentinel

    verify_source_closure = B._verify_source_closure
    monkeypatch.setattr(
        B, "_verify_source_closure",
        lambda value: verify_source_closure(value, registered_tree),
    )
    monkeypatch.setattr(B.V2.AttestedVisionBrevBackend, "from_runtime", classmethod(parent))
    result = B.SupervisedVisionBrevBackend.from_runtime(
        Path("/unused/lab"), Path("/unused/python"), protocol,
        "a" * 40, Path("/unused/runtime"), [], popen=lambda *_a, **_k: pytest.fail("GPU 실행 금지"),
    )
    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][1][2] == protocol
    assert isinstance(calls[0][2]["popen"], B.PinnedVisionWorkerPopen)


def test_recording_and_runtime_source_sets_are_identical(tmp_path: Path) -> None:
    protocol = _protocol(10)
    assert protocol["recording"]["source_files_lf"] == protocol["source_sha256_lf"]
    B._verify_source_closure(protocol, _registered_source_tree(tmp_path, 10, protocol))
