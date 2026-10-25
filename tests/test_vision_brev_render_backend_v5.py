from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import test_vision_brev_render_backend_v4 as PRIOR

B = importlib.import_module("vision_brev_render_backend_v5")


def test_atomic_backend_refuses_old_source_closure_before_dispatch(monkeypatch):
    monkeypatch.setattr(B.V2.AttestedVisionBrevBackend, "from_runtime",
                        classmethod(lambda *_a, **_k: pytest.fail("dispatched incomplete source")))
    with pytest.raises(ValueError, match="source 결합"):
        B.AtomicVisionBrevBackend.from_runtime(Path("lab"), Path("python"),
                                              {"source_sha256_lf": {}}, "a" * 40, Path("runtime"), [])


def test_atomic_backend_pins_new_worker_and_preserves_owned_gate(monkeypatch):
    monkeypatch.setattr(B, "_verify_source_closure", lambda _protocol: None)
    observed = {}

    def parent(cls, *_args, **kwargs):
        observed.update(cls=cls, popen=kwargs["popen"])
        return observed

    monkeypatch.setattr(B.V2.AttestedVisionBrevBackend, "from_runtime", classmethod(parent))
    calls = []
    B.AtomicVisionBrevBackend.from_runtime(
        Path("lab"), Path("python"), {}, "a" * 40, Path("runtime"), [],
        popen=lambda args, **kwargs: calls.append((args, kwargs)))
    pinned = observed["popen"]
    assert observed["cls"] is B.AtomicVisionBrevBackend
    assert pinned.worker_v3 == str(B.WORKER_V4.absolute())
    argv = [pinned.python, "-I", "-S", "-c", B.V2.OWNED_WORKER_GATE,
            pinned.python, "-B", pinned.worker_v1, "--job", "job.json"]
    pinned(argv, cwd="lab")
    assert calls == [([*argv[:7], pinned.worker_v3, *argv[8:]], {"cwd": "lab"})]
    with pytest.raises(ValueError, match="owned-gate"):
        pinned([pinned.python, "-B", pinned.worker_v3, "--job", "job.json"])


def test_atomic_source_set_rechecks_all_new_file_bytes(tmp_path):
    refs = {}
    for relative in B.SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        refs[relative] = PRIOR.B._sha256_lf(path)
    B._verify_source_closure({"source_sha256_lf": refs}, tmp_path)
    (tmp_path / "bench/vision_atomic_capture_v1.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="LF SHA"):
        B._verify_source_closure({"source_sha256_lf": refs}, tmp_path)
