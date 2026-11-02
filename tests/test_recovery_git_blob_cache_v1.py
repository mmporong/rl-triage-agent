# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import recovery_git_blob_cache_v1 as C


def module_fixture(tmp_path: Path, relative: str = "bench/lineage_same_play_capture_v1.py") -> ModuleType:
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# fixture\n", encoding="utf-8")
    module = ModuleType("isolated_fixture")
    module.__file__ = str(source)
    module.ROOT = tmp_path
    module.subprocess = subprocess
    return module


def completed(argv: object, stdout: bytes = b"blob\n", returncode: int = 0):
    return subprocess.CompletedProcess(argv, returncode, stdout, b"")


def test_immutable_success_is_reused_deepcopied_and_options_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = module_fixture(tmp_path)
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed(argv)

    monkeypatch.setattr(subprocess, "run", fake)
    spec = f"{'a' * 40}:evals/source.py"
    with C.proof_scope((module,), repo=tmp_path) as stats:
        one = module.subprocess.run(["git", "-C", str(tmp_path), "show", spec],
                                   capture_output=True, check=True)
        one.stdout = b"changed"
        two = module.subprocess.run(["git", "-C", str(tmp_path), "show", spec],
                                   capture_output=True, check=True)
        three = module.subprocess.run(["git", "-C", str(tmp_path), "show", spec],
                                     capture_output=True, check=False)
    assert two.stdout == b"blob\n" and three.stdout == b"blob\n"
    assert len(calls) == 2
    assert stats["hits"] == 1 and stats["misses"] == 2
    assert stats["cached_entries"] == 2
    assert module.subprocess is subprocess


def test_failed_show_head_mutable_and_unsafe_options_are_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = module_fixture(tmp_path)
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        failed = argv[-1].endswith("missing.py")
        return completed(argv, returncode=1 if failed else 0)

    monkeypatch.setattr(subprocess, "run", fake)
    failed = f"{'b' * 40}:missing.py"
    safe = f"{'c' * 40}:evals/source.py"
    unsafe_specs = (
        f"{'d' * 40}:evals//source.py",
        f"{'d' * 40}:evals/./source.py",
        f"{'d' * 40}:evals/../source.py",
    )
    with C.scoped_cache((module,), repo=tmp_path) as stats:
        for _ in range(2):
            module.subprocess.run(["git", "show", failed], cwd=tmp_path,
                                  capture_output=True, check=False)
            module.subprocess.run(["git", "show", "HEAD:evals/source.py"], cwd=tmp_path)
            module.subprocess.run(["git", "status", "--porcelain"], cwd=tmp_path)
            module.subprocess.run(["git", "show", safe],
                                  cwd=tmp_path, env={})
            module.subprocess.run(["git", "show", safe], cwd=tmp_path,
                                  executable="other-git")
            module.subprocess.run(["git", "show", safe], cwd=tmp_path,
                                  stdout=object())
            module.subprocess.run(["git", "show", safe], cwd=tmp_path,
                                  unknown_option=True)
            module.subprocess.run(["git", "show", safe], cwd=tmp_path / "other")
            for unsafe in unsafe_specs:
                module.subprocess.run(["git", "show", unsafe], cwd=tmp_path)
    assert len(calls) == 22
    assert stats["git_show_calls"] == 2 and stats["hits"] == 0
    assert stats["uncached_calls"] == 20


def test_only_primitive_registered_kwargs_are_cache_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = module_fixture(tmp_path)
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed(argv)

    monkeypatch.setattr(subprocess, "run", fake)
    spec = f"{'e' * 40}:evals/source.py"
    options = {
        "cwd": tmp_path,
        "check": True,
        "capture_output": True,
        "text": False,
        "encoding": None,
        "errors": None,
        "universal_newlines": False,
        "timeout": 3.0,
    }
    with C.proof_scope((module,), repo=tmp_path) as stats:
        module.subprocess.run(["git", "show", spec], **options)
        module.subprocess.run(["git", "show", spec], **options)
        module.subprocess.run(["git", "show", spec], cwd=tmp_path, check=1)
        module.subprocess.run(["git", "show", spec], cwd=tmp_path, timeout=True)
        module.subprocess.run(["git", "show", spec], cwd=tmp_path, encoding=object())
    assert len(calls) == 4
    assert stats["hits"] == 1 and stats["uncached_calls"] == 3


def test_current_file_check_is_not_hidden_by_blob_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = module_fixture(tmp_path)
    current = tmp_path / "evals/source.py"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"registered\n")
    expected = hashlib.sha256(current.read_bytes()).hexdigest()

    def fake(argv, **kwargs):
        return completed(argv, b"registered\n")

    monkeypatch.setattr(subprocess, "run", fake)
    spec = f"{'d' * 40}:evals/source.py"

    def validate() -> None:
        blob = module.subprocess.run(["git", "show", spec], cwd=tmp_path,
                                    capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("blob")
        if hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            raise ValueError("current")

    with C.scoped_cache((module,), repo=tmp_path):
        validate()
        current.write_bytes(b"dirty\n")
        with pytest.raises(ValueError, match="current"):
            validate()


def test_scope_restores_on_error_rejects_wrong_module_and_detects_external_change(
    tmp_path: Path,
) -> None:
    module = module_fixture(tmp_path)
    with (pytest.raises(RuntimeError, match="body"),
          C.scoped_cache((module,), repo=tmp_path)):
        raise RuntimeError("body")
    assert module.subprocess is subprocess

    outside = module_fixture(tmp_path, "evals/not_allowed.py")
    with (pytest.raises(ValueError, match="허용되지 않은"),
          C.scoped_cache((outside,), repo=tmp_path)):
        pass

    with (pytest.raises(RuntimeError, match="외부에서 바뀌었다"),
          C.scoped_cache((module,), repo=tmp_path)):
        module.subprocess = object()
    assert module.subprocess is subprocess


def test_memoize_false_and_real_scratch_git_boundary(tmp_path: Path) -> None:
    repo = tmp_path.resolve()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "cache@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Cache Test"], cwd=repo, check=True)
    module = module_fixture(repo)
    tracked = repo / "evidence.txt"
    tracked.write_text("frozen\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "evidence.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    anchor = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    spec = f"{anchor}:evidence.txt"
    with C.scoped_cache((module,), repo=repo, memoize=False) as stats:
        first = module.subprocess.run(["git", "show", spec], cwd=repo,
                                     capture_output=True, check=True).stdout
        second = module.subprocess.run(["git", "show", spec], cwd=repo,
                                      capture_output=True, check=True).stdout
    assert first == second == b"frozen\n"
    assert stats["hits"] == 0 and stats["git_show_calls"] == 2
    assert stats["cached_entries"] == 0
