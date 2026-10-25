from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_vision_brev_annotated_v2 as C  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v4 as B  # pyright: ignore[reportMissingImports]


def _args(tmp_path: Path, phase: str, command: str) -> list[str]:
    return ["--phase", phase, "--command", command, "--repo", str(tmp_path),
            "--protocol", "source.json", "--protocol-sha256-lf", "1" * 64,
            "--render-protocol", "render.json", "--render-protocol-sha256-lf", "2" * 64,
            "--preflight-protocol", "preflight.json", "--preflight-protocol-sha256-lf", "3" * 64,
            "--source-anchor", "a" * 40, "--resource-proof", "resource.json",
            "--resource-sha256", "4" * 64]


@pytest.mark.parametrize("phase", ["preflight", "render"])
def test_active_phase_uses_supervised_backend(tmp_path: Path, monkeypatch, capsys, phase: str) -> None:
    captured: dict[str, Any] = {}
    class Batch:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["factory"] = kwargs["backend_factory"]
        def verify(self) -> dict[str, str]: return {"status": "verified"}
    monkeypatch.setattr(C.PREFLIGHT, "PreflightBatch", Batch)
    monkeypatch.setattr(C.RENDER, "RenderBatch", Batch)
    assert C.main([*_args(tmp_path, phase, "verify"), "--isaaclab", "lab", "--python", "python"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "verified"}
    assert captured["factory"].__func__ is B.SupervisedVisionBrevBackend.from_runtime.__func__


def test_preflight_advance_settles_offline_before_backend(tmp_path: Path, monkeypatch, capsys) -> None:
    expected = {"status": "HOLD", "reason": "offline_settled"}
    monkeypatch.setattr(C.PREFLIGHT, "settle_preflight", lambda *_args: expected)
    monkeypatch.setattr(C.PREFLIGHT, "PreflightBatch",
                        lambda *_a, **_k: pytest.fail("offline settle 뒤 backend 생성"))
    assert C.main(_args(tmp_path, "preflight", "advance")) == 2
    assert json.loads(capsys.readouterr().out) == expected
