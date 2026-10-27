from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_vision_annotator_pull_v1 as R  # pyright: ignore[reportMissingImports]


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
        "job": {
            "mode": R.JOB.MODE,
            "callback_variant": R.SDK.VARIANT,
        },
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": tmp_path / "output",
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _envelope(validated: dict[str, Any], *, write_raw: bool) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    artifacts = []
    for ordinal in range(4):
        npz = output / f"raw-{ordinal}.npz"
        receipt = output / f"raw-{ordinal}.json"
        if write_raw:
            npz.write_bytes(b"npz" + bytes([ordinal]))
            receipt.write_text(json.dumps({"ordinal": ordinal}), encoding="utf-8")
        artifacts.append(
            {
                "ordinal": ordinal,
                "complete_capture": True,
                "npz": _descriptor(npz) if write_raw else {},
                "receipt": _descriptor(receipt) if write_raw else {},
            }
        )
    worker = {
        "collection": {
            "status": "collected_unvalidated",
            "capture_count": 4,
            "atomicity_claimed": False,
        },
        "validation": {
            "approved": True,
            "camera_sequence": ["A", "A", "B", "A"],
            "marker_codes": [0, 1, 2, 3],
            "atomicity_claimed": False,
        },
        "policy_probe": {
            "environment_steps": 1,
            "annotators_detached_before_policy_step": True,
        },
        "lifecycle": {"annotators_attached": True, "annotators_detached": True},
        "writer_calls": {"create": 0, "attach": 0, "callback": 0, "detach": 0},
        "diagnostic_complete": True,
    }
    return {
        "schema": R.SDK.SCHEMA,
        "variant": R.SDK.VARIANT,
        "worker": worker,
        "raw_artifacts": artifacts,
        "primary_error": None,
        "diagnostic_complete": True,
        "used_for_decision": False,
        "capture_approved": False,
        "approved": False,
        "atomicity_claimed": False,
    }


def test_terminal_requires_writer_zero_and_non_approving_boundary(tmp_path: Path) -> None:
    item = _validated(tmp_path)
    value = _envelope(item, write_raw=False)
    assert R._diagnostic(item, value)
    value["worker"]["writer_calls"]["callback"] = 1
    assert not R._diagnostic(item, value)


def test_sdk_child_seals_and_verifies_four_raw_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _validated(tmp_path)
    calls: list[str] = []

    class App:
        def close(self) -> None:
            calls.append("close")

    def run(_validated: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
        calls.append("run")
        value = _envelope(item, write_raw=True)
        seal_report(value)
        return value

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: item)
    assert R.sdk_child(item["job_path"], app_factory=lambda _: App(), run_sdk=run) == 0
    assert calls == ["run", "close"]
    result = R.verify_result(item)
    assert result["diagnostic_complete"] is True
    assert result["capture_approved"] is False
    assert result["used_for_decision"] is False


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
        raise ValueError("annotator pull evidence missing")

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: item)
    assert R.sdk_child(item["job_path"], app_factory=lambda _: App(), run_sdk=fail) == 1
    assert calls == ["run", "close"]
    with pytest.raises(R.ChildFailure, match="annotator pull evidence missing"):
        R.verify_terminal(item)
