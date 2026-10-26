from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

B = importlib.import_module("vision_brev_training_backend_v2")
V1 = importlib.import_module("brev_recovery_backend_v4")


def test_worker_popen_rewrites_only_the_frozen_worker_slot() -> None:
    captured: dict[str, Any] = {}

    def fake_popen(argv, *args, **kwargs):
        captured.update(argv=list(argv), args=args, kwargs=kwargs)
        return "process"

    launch = B.worker_popen(fake_popen)
    result = launch(
        ["python", "-B", str(V1.WORKER), "--job", "/stage/job.json", "--output", "/stage/worker.json"],
        cwd="/lab",
    )
    assert result == "process"
    assert captured["argv"][2] == str(B.WORKER)
    assert captured["argv"][:2] == ["python", "-B"]
    assert captured["argv"][3:] == ["--job", "/stage/job.json", "--output", "/stage/worker.json"]
    assert captured["kwargs"] == {"cwd": "/lab"}


def test_worker_popen_rejects_an_unexpected_child_command() -> None:
    launch = B.worker_popen(lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="argv 결합"):
        launch(["python", "-B", "/tmp/other.py"])
