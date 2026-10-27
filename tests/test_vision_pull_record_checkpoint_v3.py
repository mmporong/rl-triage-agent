from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import vision_pull_record_checkpoint_v3 as CHECKPOINT


def _validated(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    return {
        "job_path": job,
        "output_directory": tmp_path / "aaba-output",
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": "a" * 64,
        "job": {
            "mode": CHECKPOINT.MODE,
            "callback_variant": CHECKPOINT.VARIANT,
            "profile": {
                "camera": {
                    "rt_subframes": 1,
                    "pause_timeline": False,
                    "delta_time_s": 0.0,
                }
            },
        },
    }


def _worker(validated: dict[str, Any], _app: object, **_kwargs: Any) -> dict[str, Any]:
    captures = []
    for ordinal in range(4):
        captures.append(
            {
                "schema": "capture",
                "ordinal": ordinal,
                "raw": {
                    "rgb": np.full((2, 2, 4), ordinal, dtype=np.uint8),
                    "semantic_segmentation": {
                        "data": np.full((2, 2), ordinal, dtype=np.uint32),
                        "info": {"idToLabels": {"1": {"class": "robot"}}},
                    },
                    "CameraParams": {},
                    "IsaacReadTimes": {"frameNumber": ordinal + 1},
                },
            }
        )
    return {
        "diagnostic_complete": True,
        "collection": {"captures": captures, "partial_capture": None},
        "policy_probe": {
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "environment_steps": 1,
        },
        "setup": {
            "app_update_counter": {
                "source": "omni.kit.app.IApp.get_update_number",
                "method": "app.get_update_number()",
                "initial_value": 1,
                "after_extension_update": 2,
                "fail_closed": True,
            }
        },
    }


def test_actual_checkpoint_writes_npz_before_json_seal(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    report_path = tmp_path / "report.json"

    result = CHECKPOINT.run_annotator_pull_diagnostic(
        validated,
        object(),
        run_worker=_worker,
        seal_report=lambda value: report_path.write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        ),
    )

    assert result["diagnostic_complete"] is True
    assert len(list((tmp_path / "aaba-output").glob("*.npz"))) == 4
    assert '"artifact"' in report_path.read_text(encoding="utf-8")
    assert CHECKPOINT._json_problem(result) is None


def test_missing_output_is_created_and_original_worker_failure_is_preserved(
    tmp_path: Path,
) -> None:
    validated = _validated(tmp_path)
    sealed: list[dict[str, Any]] = []

    class WorkerFailure(RuntimeError):
        def __init__(self) -> None:
            super().__init__("first-cause")
            self.primary = ValueError("first-cause")
            self.envelope = _worker(validated, object())

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise WorkerFailure()

    with pytest.raises(CHECKPOINT.PullRecordCheckpointFailure) as caught:
        CHECKPOINT.run_annotator_pull_diagnostic(
            validated, object(), run_worker=fail, seal_report=lambda value: sealed.append(dict(value))
        )

    assert isinstance(caught.value.primary, ValueError)
    assert str(caught.value.primary) == "first-cause"
    assert caught.value.serialization_error is None
    assert sealed[0]["primary_error"]["error_message"] == "first-cause"
    assert CHECKPOINT._json_problem(sealed[0]) is None
    assert len(list((tmp_path / "aaba-output").glob("*.npz"))) == 4


def test_raw_numpy_serialization_problem_reports_field_path() -> None:
    with pytest.raises(TypeError, match=r"\$\.worker\.collection\[0\]\.raw"):
        CHECKPOINT._json_safe(
            {"worker": {"collection": [{"raw": np.zeros((1,), dtype=np.uint8)}]}}
        )


def test_checkpoint_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    validated = _validated(tmp_path)

    def wrong(value: dict[str, Any], app: object, **kwargs: Any) -> dict[str, Any]:
        result = _worker(value, app, **kwargs)
        result["policy_probe"]["checkpoint_sha256"] = "b" * 64
        return result

    with pytest.raises(CHECKPOINT.PullRecordCheckpointFailure) as caught:
        CHECKPOINT.run_annotator_pull_diagnostic(
            validated, object(), run_worker=wrong, seal_report=lambda _value: None
        )
    assert isinstance(caught.value.primary, ValueError)
