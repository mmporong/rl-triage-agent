from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import run_agentic_brev_probe_parent_v3 as P  # pyright: ignore[reportMissingImports]


def backend_with(expected: list[str], stage: Path):
    bounded = P.V2.V1.BoundedPopen(lambda: None, 60, expected)
    backend = object.__new__(P.V2.V1.U.OWNED.SafeRecoveryBackend)
    backend.python = Path(expected[0])
    backend.lab = stage.parent
    backend.read_gpu = list
    backend.popen = bounded
    backend._source_snapshot = dict
    return backend, bounded


def test_actual_backend_reproduces_worker2_vs_registered_worker1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "without-scope"
    expected = [
        sys.executable, "-B", str(P.WORKER1), "--job", str(stage / "job.json"),
        "--output", str(stage / "worker.json"),
    ]
    called: list[list[str]] = []
    monkeypatch.setattr(P.V2.V1.U, "_owned_popen", lambda args, **_kwargs: called.append(list(args)))
    backend, _bounded = backend_with(expected, stage)
    with pytest.raises(P.BACKEND.BackendError):
        backend._execute(stage, "probe", {})
    assert called == []
    assert P.BACKEND.WORKER == P.WORKER2


def test_registered_worker_scope_accepts_worker1_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "with-scope"
    expected = [
        sys.executable, "-B", str(P.WORKER1), "--job", str(stage / "job.json"),
        "--output", str(stage / "worker.json"),
    ]
    called: list[list[str]] = []

    def boundary(args, **_kwargs):
        called.append(list(args))
        raise RuntimeError("after-bounded-validation")

    monkeypatch.setattr(P.V2.V1.U, "_owned_popen", boundary)
    backend, _bounded = backend_with(expected, stage)
    with P.registered_worker_scope(), pytest.raises(P.BACKEND.BackendError):
        assert P.BACKEND.WORKER == P.WORKER1
        backend._execute(stage, "probe", {})
    assert called == [expected]
    assert P.BACKEND.WORKER == P.WORKER2


def test_unapproved_argv_still_rejected() -> None:
    expected = ["python", "-B", str(P.WORKER1), "--job", "j", "--output", "o"]
    bounded = P.V2.V1.BoundedPopen(lambda: None, 60, expected)
    with pytest.raises(ValueError, match="unapproved child command"):
        bounded([*expected[:-1], "different"], start_new_session=True)


def test_worker_probe_ast_is_equal() -> None:
    P.verify_probe_equivalence()
