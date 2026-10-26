from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_vision_semantic_settle_v1 as R  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = _write(
        tmp_path / "job.json",
        {"mode": R.JOB.MODE, "callback_variant": R.SDK.VARIANT},
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    return {
        "job": {"mode": R.JOB.MODE, "callback_variant": R.SDK.VARIANT},
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": tmp_path / "output",
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def _callback() -> dict[str, Any]:
    return {
        "ordinal": 0,
        "monotonic_ns": 1,
        "thread_id": 1,
        "window": "arming0",
        "marker_code": 0,
        "reference_time": [1, 30],
        "frame_number": 1,
        "classification": "valid",
        "classification_error_type": None,
        "raw_callback_count_before": 0,
        "raw_payload_sha256": "0" * 64,
        "components": {
            key: {"sha256": "1" * 64, "metadata": {}}
            for key in R.SDK.BOUNDARY.COMPONENT_KEYS
        },
        "trigger_outputs": {"sha256": "1" * 64, "metadata": {}},
        "og_node_metadata": {},
        "operation_counts_before": {},
        "instrumentation_error": None,
        "raw_callback_count_after": 1,
        "mux_failure": None,
        "operation_counts_after": {},
    }


def _marker_states() -> list[dict[str, Any]]:
    rows = []
    for index, phase in enumerate(
        ["bound", "prepare", "arm_0", "arm_1", "arm_2", "arm_3"]
    ):
        rows.append(
            {
                "phase": phase,
                "code": None if index == 0 else 3 if index == 1 else index - 2,
                "path_count": 9,
                "visible_count": 0 if index == 0 else 5,
                "visibility_sha256": f"{index + 1:x}" * 64,
                "semantic_attributes_sha256": "a" * 64,
                "semantic_attribute_count": 9,
            }
        )
    return rows


def _settles() -> list[dict[str, Any]]:
    rows = []
    states = _marker_states()[2:]
    for ordinal, marker in enumerate(states):
        rows.append(
            {
                "ordinal": ordinal,
                "code": ordinal,
                "status": "complete",
                "monotonic_ns": ordinal + 1,
                "thread_id": 1,
                "callbacks_before": ordinal,
                "callbacks_after": ordinal,
                "callback_delta": 0,
                "physics_before_sha256": "b" * 64,
                "physics_after_sha256": "b" * 64,
                "simulation_time_before_s": 2.0,
                "simulation_time_after_s": 2.0,
                "physics_equal": True,
                "render_contract": {
                    "delta_time_s": 0.0,
                    "rt_subframes": 1,
                    "pause_timeline": False,
                },
                "marker_before": {
                    key: value for key, value in marker.items() if key != "phase"
                },
                "marker_after": {
                    key: value for key, value in marker.items() if key != "phase"
                },
                "error_type": None,
                "error_message": None,
            }
        )
    return rows


def _envelope() -> dict[str, Any]:
    boundary = {
        "schema": R.SDK.BOUNDARY.SCHEMA,
        "variant": "manual_schedule",
        "underlying": {"schema": "checkpoint", "approved": True},
        "callbacks": [_callback()],
        "operations": {"timeline": [{"name": "orchestrator_step"}], "counts": {}},
        "capture_on_play": {
            "before": True,
            "after_startup": False,
            "restored": True,
        },
        "primary_error": None,
        "diagnostic_complete": True,
        "used_for_decision": False,
        "approved": False,
    }
    return {
        "schema": R.SDK.SCHEMA,
        "variant": R.SDK.VARIANT,
        "boundary": boundary,
        "settles": _settles(),
        "marker_states": _marker_states(),
        "primary_error": None,
        "diagnostic_complete": True,
        "used_for_decision": False,
        "approved": False,
    }


def test_terminal_requires_complete_non_approving_settle_evidence(
    tmp_path: Path,
) -> None:
    item = _validated(tmp_path)
    value = _envelope()
    assert R._diagnostic(item, value)
    value["settles"][0]["callback_delta"] = 1
    assert not R._diagnostic(item, value)


def test_sdk_child_seals_terminal_before_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _validated(tmp_path)
    calls: list[str] = []

    class App:
        def close(self) -> None:
            calls.append("close")

    def run(_validated: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
        calls.append("run")
        value = _envelope()
        seal_report(value)
        return value

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: item)
    assert R.sdk_child(item["job_path"], app_factory=lambda _: App(), run_sdk=run) == 0
    assert calls == ["run", "close"]
    result = R.verify_result(item)
    assert result["diagnostic_complete"] is True
    assert result["capture_approved"] is False
    assert result["report"]["file"] == R.REPORT_NAME


def test_sdk_child_preserves_first_failure_before_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _validated(tmp_path)
    calls: list[str] = []

    class App:
        def close(self) -> None:
            calls.append("close")

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("run")
        raise ValueError("settle evidence missing")

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: item)
    assert R.sdk_child(item["job_path"], app_factory=lambda _: App(), run_sdk=fail) == 1
    assert calls == ["run", "close"]
    with pytest.raises(R.ChildFailure, match="settle evidence missing"):
        R.verify_terminal(item)
