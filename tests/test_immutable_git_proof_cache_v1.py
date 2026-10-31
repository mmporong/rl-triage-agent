# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import immutable_git_proof_cache_v1 as C


def install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    function: object,
) -> Any:
    module = C.RESIDUAL
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_git", function)
    return module


def completed(args: tuple[object, ...], *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["git", *map(str, args)], returncode, "blob", "")


def test_only_immutable_commit_blob_is_memoized_and_deepcopied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake(*args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append((args, check))
        return completed(args)

    module = install(monkeypatch, tmp_path, fake)
    spec = f"{'a' * 40}:evals/source.py"
    with C.proof_scope((module,), repo=tmp_path) as stats:
        first = module._git("show", spec)
        first.stdout = "mutated"
        second = module._git("show", spec)
    assert second.stdout == "blob"
    assert len(calls) == 1
    assert stats["hits"] == 1 and stats["git_show_calls"] == 1
    assert module._git is fake


def test_head_index_other_commands_and_failed_results_are_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake(*args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append((args, check))
        failed = args == ("show", f"{'b' * 40}:missing.py")
        return completed(args, returncode=1 if failed else 0)

    module = install(monkeypatch, tmp_path, fake)
    with C.proof_scope((module,), repo=tmp_path) as stats:
        for _ in range(2):
            module._git("show", "HEAD:evals/source.py")
            module._git("status", "--porcelain")
            module._git("show", f"{'b' * 40}:missing.py", check=False)
    assert len(calls) == 6
    assert stats["mutable_calls"] == 4 and stats["git_show_calls"] == 2
    assert stats["hits"] == 0


def test_cached_blob_does_not_hide_current_file_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "evals/source.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"registered\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    def fake(*args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["git", *map(str, args)], 0, "registered\n", "")

    module = install(monkeypatch, tmp_path, fake)

    def original_validation() -> None:
        blob = module._git("show", f"{'c' * 40}:evals/source.py").stdout.encode()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("commit blob mismatch")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("current source mismatch")

    with C.proof_scope((module,), repo=tmp_path) as stats:
        original_validation()
        source.write_bytes(b"dirty\n")
        with pytest.raises(ValueError, match="current source"):
            original_validation()
    assert stats["hits"] == 1


def test_wrong_repo_unallowed_module_external_change_and_nested_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(*args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return completed(args)

    module = install(monkeypatch, tmp_path, fake)
    with (pytest.raises(ValueError, match="repo identity"),
          C.proof_scope((module,), repo=tmp_path / "other")):
        pass
    with (pytest.raises(ValueError, match="허용되지 않은"),
          C.proof_scope((sys,), repo=tmp_path)):
        pass

    with C.proof_scope((module,), repo=tmp_path):
        outer = module._git
        with C.proof_scope((module,), repo=tmp_path):
            assert module._git is not outer
        assert module._git is outer
    assert module._git is fake

    replacement = lambda *_args, **_kwargs: completed(())
    with (pytest.raises(RuntimeError, match="외부에서 바뀌었다"),
          C.proof_scope((module,), repo=tmp_path)):
        module._git = replacement
    assert module._git is fake
