from __future__ import annotations

import importlib

import pytest
import test_run_vision_brev_annotated_v2 as PRIOR

C = importlib.import_module("run_vision_brev_atomic_v1")


@pytest.mark.parametrize("phase", ["preflight", "render"])
def test_atomic_entry_injects_new_backend_with_existing_admission(
    tmp_path, monkeypatch, capsys, phase
):
    observed = {}

    class Batch:
        def __init__(self, *_args, **kwargs):
            observed["factory"] = kwargs["backend_factory"]

        def verify(self):
            return {"status": "verified"}

    monkeypatch.setattr(C.ENTRY.PREFLIGHT, "PreflightBatch", Batch)
    monkeypatch.setattr(C.ENTRY.RENDER, "RenderBatch", Batch)
    assert (
        C.main(
            [
                *PRIOR._args(tmp_path, phase, "verify"),
                "--isaaclab",
                "lab",
                "--python",
                "python",
            ]
        )
        == 0
    )
    assert observed["factory"].__self__ is C.BACKEND.AtomicVisionBrevBackend


def test_atomic_entry_does_not_admit_gpu_after_offline_terminal_receipt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        C.ENTRY.PREFLIGHT, "settle_preflight", lambda *_a: {"status": "HOLD"}
    )
    monkeypatch.setattr(
        C.ENTRY.PREFLIGHT,
        "PreflightBatch",
        lambda *_a, **_k: pytest.fail("GPU backend opened"),
    )
    assert C.main(PRIOR._args(tmp_path, "preflight", "advance")) == 2
