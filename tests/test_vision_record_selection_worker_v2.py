from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "tests", ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import test_vision_selection_worker_v1 as FIXTURE  # pyright: ignore[reportMissingImports]
import vision_record_selection_worker_v2 as W  # pyright: ignore[reportMissingImports]


def test_worker_v2_binds_only_inputs_v2() -> None:
    assert W.INPUTS.__name__ == "vision_record_selection_inputs_v2"


@pytest.fixture
def worker_case(tmp_path: Path) -> dict[str, Any]:
    return cast(Any, FIXTURE.case).__wrapped__(tmp_path)


def test_prepare_routes_existing_worker_through_first_class_input_verifier(
    worker_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(W.INPUTS, "verify_inputs", worker_case["verifier"])
    prepared = W.prepare(
        worker_case["protocol"],
        worker_case["sha"],
        worker_case["anchor"],
        worker_case["locations"],
        root=worker_case["repo"],
    )
    assert prepared["plan"]["status"] == "planned"
    assert len(prepared["protocol"]["rows"]) == 24
    assert worker_case["verifier_calls"][-1][2] == worker_case["locations"]


def test_advance_injects_verifier_and_preserves_worker_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def advance(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], int]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "blocked_pending", "provider_calls": 0}, 4

    preflight = lambda: None
    monkeypatch.setattr(W.WORKER, "advance", advance)
    result, code = W.advance(
        tmp_path / "protocol.json",
        "a" * 64,
        "b" * 40,
        tmp_path / "locations.json",
        "choice-01",
        root=tmp_path,
        provider_preflight=preflight,
    )
    assert code == 4 and result["provider_calls"] == 0
    assert captured["kwargs"]["input_verifier"] is W.INPUTS.verify_inputs
    assert captured["kwargs"]["provider_preflight"] is preflight


def test_cli_preserves_existing_verify_and_advance_surface() -> None:
    with pytest.raises(SystemExit) as captured:
        W.main(["--help"])
    assert captured.value.code == 0
