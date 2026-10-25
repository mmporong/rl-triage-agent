from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import vision_brev_render_backend_v4 as B  # pyright: ignore[reportMissingImports]


def test_pinned_worker_translates_only_exact_gate_and_keeps_training() -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    delegate = lambda args, **kwargs: calls.append((args, kwargs)) or "process"
    popen = B.PinnedVisionWorkerPopen(delegate, Path("/opt/isaac/python"),
                                      Path("/repo/worker1.py"), Path("/repo/worker3.py"))
    owned = [popen.python, "-I", "-S", "-c", B.V2.OWNED_WORKER_GATE,
             popen.python, "-B", popen.worker_v1, "--job", "/runtime/job.json"]
    assert popen(owned, cwd="/lab") == "process"
    assert calls[-1] == ([*owned[:7], popen.worker_v3, *owned[8:]], {"cwd": "/lab"})
    training = [popen.python, "-u", "/repo/train.py"]
    assert popen(training, stdout=subprocess.PIPE) == "process"
    assert calls[-1] == (training, {"stdout": subprocess.PIPE})


@pytest.mark.parametrize("case", ["v1", "v3", "gate"])
def test_pinned_worker_rejects_direct_or_extra_argv(case: str) -> None:
    popen = B.PinnedVisionWorkerPopen(lambda *_a, **_k: pytest.fail("delegate"), Path("PY"),
                                      Path("W1"), Path("W3"))
    normalized = {
        "v1": [popen.python, "-B", popen.worker_v1, "--job", "j"],
        "v3": [popen.python, "-B", popen.worker_v3, "--job", "j"],
        "gate": [popen.python, "-I", "-S", "-c", B.V2.OWNED_WORKER_GATE, "extra"],
    }[case]
    with pytest.raises(ValueError, match="owned-gate"):
        popen(normalized)


def test_source_closure_requires_every_file_and_current_lf_bytes(tmp_path: Path) -> None:
    source: dict[str, str] = {}
    for relative in B.SOURCE_FILES:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
        source[relative] = B._sha256_lf(path)
    B._verify_source_closure({"source_sha256_lf": source}, tmp_path)
    changed = next(iter(B.SOURCE_FILES)); (tmp_path / changed).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LF SHA"):
        B._verify_source_closure({"source_sha256_lf": source}, tmp_path)
    del source[next(iter(source))]
    with pytest.raises(ValueError, match="source 결합"):
        B._verify_source_closure({"source_sha256_lf": source}, tmp_path)


def test_factory_wraps_delegate_after_source_validation(monkeypatch) -> None:
    monkeypatch.setattr(B, "_verify_source_closure", lambda _protocol: None)
    captured: dict[str, Any] = {}
    sentinel = object()

    def parent(cls: type[Any], *args: Any, **kwargs: Any) -> object:
        captured.update({"cls": cls, "popen": kwargs["popen"]})
        return sentinel

    monkeypatch.setattr(B.V2.AttestedVisionBrevBackend, "from_runtime", classmethod(parent))
    delegate = object()
    result = B.SupervisedVisionBrevBackend.from_runtime(
        Path("/lab"), Path("/python"), {}, "a" * 40, Path("/runtime"), [], popen=delegate)
    assert result is sentinel and captured["cls"] is B.SupervisedVisionBrevBackend
    assert isinstance(captured["popen"], B.PinnedVisionWorkerPopen)
    assert captured["popen"].delegate is delegate
