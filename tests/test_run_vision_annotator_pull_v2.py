from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import run_vision_annotator_pull_v2 as R  # pyright: ignore[reportMissingImports]
import test_vision_annotator_pull_checkpoint_v2 as SDK_FIXTURE  # pyright: ignore[reportMissingImports]


def _validated(tmp_path: Path) -> dict[str, Any]:
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "job": {
            "mode": R.JOB.MODE,
            "callback_variant": R.SDK.VARIANT,
            "profile": RECORD.fixed_profile(),
        },
        "job_path": job_path,
        "job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
        "output_directory": tmp_path / "output",
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def test_sdk_child_default_chain_reaches_checkpoint_v2_and_worker_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    calls: list[str] = []

    class App:
        def close(self) -> None:
            calls.append("close")

    def worker(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("worker_v2")
        return SDK_FIXTURE._worker()

    monkeypatch.setattr(R.JOB, "validate_job", lambda _path: validated)
    monkeypatch.setattr(R.SDK.WORKER, "run_worker", worker)

    assert R.sdk_child(validated["job_path"], app_factory=lambda _value: App()) == 0
    assert calls == ["worker_v2", "close"]
    result = R.verify_result(validated)
    assert result["schema"] == R.JOB.RESULT_SCHEMA
    assert result["sdk"]["schema"] == R.SDK.SCHEMA
    assert R.SDK._native_counter_receipt(result["sdk"])["fail_closed"] is True


def test_terminal_rejects_explicit_counter_substitute(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    envelope = {
        "schema": R.SDK.SCHEMA,
        "variant": R.SDK.VARIANT,
        "worker": SDK_FIXTURE._worker(),
        "raw_artifacts": [{"complete_capture": True}] * 4,
        "primary_error": None,
        "diagnostic_complete": True,
        "used_for_decision": False,
        "capture_approved": False,
        "approved": False,
        "atomicity_claimed": False,
    }
    envelope["worker"]["setup"]["app_update_counter"]["source"] = (
        "explicit_app_updates"
    )

    assert R._diagnostic(validated, envelope) is False
