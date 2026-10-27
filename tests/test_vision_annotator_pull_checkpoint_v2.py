from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_vision_annotator_pull_checkpoint_v1 as V1_FIXTURE  # pyright: ignore[reportMissingImports]
import vision_annotator_pull_checkpoint_v2 as D  # pyright: ignore[reportMissingImports]


def _worker() -> dict[str, Any]:
    worker = V1_FIXTURE._worker()
    worker["schema"] = D.WORKER.SCHEMA
    worker["setup"] = {
        "app_update_counter": {
            "source": "omni.kit.app.IApp.get_update_number",
            "method": "app.get_update_number()",
            "initial_value": 10,
            "after_extension_update": 11,
            "fail_closed": True,
        }
    }
    return worker


def test_checkpoint_v2_uses_worker_v2_and_seals_native_counter_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = V1_FIXTURE._validated(tmp_path)
    sealed: list[dict[str, Any]] = []
    calls: list[str] = []

    def worker(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("worker_v2")
        return _worker()

    monkeypatch.setattr(D.WORKER, "run_worker", worker)
    result = D.run_annotator_pull_diagnostic(
        validated,
        object(),
        seal_report=lambda value: sealed.append(dict(value)),
    )

    assert calls == ["worker_v2"]
    assert result["schema"] == D.SCHEMA
    assert D._native_counter_receipt(result)["initial_value"] == 10
    assert sealed == [result]


def test_checkpoint_v2_rejects_success_without_native_counter_receipt(
    tmp_path: Path,
) -> None:
    validated = V1_FIXTURE._validated(tmp_path)
    worker = _worker()
    worker.pop("setup")

    with pytest.raises(ValueError, match="native app update counter receipt"):
        D.run_annotator_pull_diagnostic(
            validated,
            object(),
            seal_report=lambda _value: None,
            run_worker=lambda *_args, **_kwargs: worker,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source", "explicit_app_updates"),
        ("fail_closed", False),
        ("after_extension_update", 10),
    ],
)
def test_native_counter_receipt_is_strict(key: str, value: Any) -> None:
    report = {
        "worker": {
            "setup": {
                "app_update_counter": {
                    "source": "omni.kit.app.IApp.get_update_number",
                    "method": "app.get_update_number()",
                    "initial_value": 10,
                    "after_extension_update": 11,
                    "fail_closed": True,
                    key: value,
                }
            }
        }
    }
    with pytest.raises(ValueError, match="native app update counter receipt"):
        D._native_counter_receipt(report)
