"""D4 v2 worker가 v2 protocol·script에 결합되는지 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import selection_comparison_demo_v2 as D  # type: ignore[import-not-found]  # noqa: E402
import selection_comparison_worker_v2 as W  # type: ignore[import-not-found]  # noqa: E402


def test_worker_uses_v2_demo_protocol_and_script():
    assert W.DEMO is D
    assert W.SCRIPT == "evals/selection_comparison_worker_v2.py"
    assert W.DEMO.PROTOCOL == ROOT / "bench/protocols/selection_comparison_demo_v2.json"


def test_worker_args_bind_v2_script_and_protocol_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(W.MC, "wsl_path", lambda path: path.as_posix())
    root = tmp_path / "model"
    receipt = {"ordinal": 1, "capsule_sha256": "a" * 64, "slot_id": "slot"}
    args = W._worker_args(root, receipt, "b" * 40, "c" * 64)
    assert args[0].endswith("/evals/selection_comparison_worker_v2.py")
    assert args[args.index("--protocol-sha") + 1] == "c" * 64
    assert "selection_comparison_worker_v1.py" not in " ".join(args)
