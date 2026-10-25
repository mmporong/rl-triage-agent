from __future__ import annotations

import hashlib
import importlib
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
A = importlib.import_module("vision_timing_adapter_v1")
R = importlib.import_module("record_vision_evidence_v1")


class ExecutionState(Enum):
    ENABLED = 1


class PybindLikeExecutionState:
    def __int__(self) -> int:
        return 1

    def __repr__(self) -> str:
        raise AssertionError("unsupported SDK 값에 repr을 호출하면 안 된다")

    def __str__(self) -> str:
        raise AssertionError("unsupported SDK 값에 str을 호출하면 안 된다")


def _read_times(simulation_time: Any = 1.0) -> dict[str, Any]:
    return {
        "durationDenominator": 50,
        "durationNumerator": 1,
        "externalTimeOfSimNs": 1_000_000_000,
        "frameNumber": 50,
        "rationalTimeOfSimDenominator": 50,
        "rationalTimeOfSimNumerator": 50,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": simulation_time,
        "simulationTimeMonotonic": 1.0,
        "swhFrameNumber": 50,
        "systemTime": 1_001.0,
    }


def _reference() -> dict[str, Any]:
    return {"referenceTimeNumerator": 1, "referenceTimeDenominator": 1}


def test_normalize_accepts_only_registered_11_or_execout_12_without_mutation() -> None:
    eleven = _read_times()
    eleven_before = dict(eleven)
    normalized = A.normalize_render_timing(eleven, _reference(), 1.0)
    assert list(normalized["read_times"]) == list(R.ISAAC_READ_TIMES_FIELDS)
    assert normalized["read_times"] == eleven_before
    assert eleven == eleven_before

    twelve = {**eleven, "execOut": ExecutionState.ENABLED}
    twelve_before = dict(twelve)
    normalized = A.normalize_render_timing(twelve, _reference(), 1.0)
    assert normalized["read_times"] == eleven
    assert "execOut" not in normalized["read_times"]
    assert twelve == twelve_before


@pytest.mark.parametrize("change", ["missing", "unknown", "execout_plus_unknown"])
def test_normalize_rejects_missing_or_unknown_fields(change: str) -> None:
    read_times = _read_times()
    if change == "missing":
        read_times.pop("frameNumber")
    elif change == "unknown":
        read_times["unknown"] = 1
    else:
        read_times.update({"execOut": ExecutionState.ENABLED, "unknown": 1})
    with pytest.raises(ValueError, match="11필드|12필드"):
        A.normalize_render_timing(read_times, _reference(), 1.0)


@pytest.mark.parametrize(
    ("field", "value", "simulation_time", "message"),
    [
        ("simulationTime", 1.1, 1.0, "시각"),
        ("rationalTimeOfSimDenominator", 0, 1.0, "분모"),
        ("systemTime", float("inf"), 1.0, "유한"),
    ],
)
def test_existing_timing_invariants_remain_authoritative(
    field: str, value: Any, simulation_time: float, message: str
) -> None:
    read_times = _read_times()
    read_times[field] = value
    with pytest.raises(ValueError, match=message):
        A.normalize_render_timing(read_times, _reference(), simulation_time)


def test_journal_preserves_sdk_types_and_values_without_copying_rgb(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    read_times = _read_times(np.float64(1.0))
    read_times["frameNumber"] = np.int64(50)
    read_times["execOut"] = ExecutionState.ENABLED
    reference = {
        "referenceTimeNumerator": np.int64(1),
        "referenceTimeDenominator": np.int64(1),
    }
    rgb = b"rgb bytes stay in the existing capture artifact"
    rgb_sha = hashlib.sha256(rgb).hexdigest()

    path = A.write_capture_journal(
        job, 3, read_times, reference, rgb_sha, rgb_bytes=rgb
    )
    body = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "job.capture-000003.timing-raw.json"
    assert body["observed_read_times_key_count"] == 12
    assert body["read_times"]["fields"]["frameNumber"] == {
        "type": "numpy.int64",
        "encoding": "numpy_item:json_native",
        "value": 50,
    }
    assert body["read_times"]["fields"]["simulationTime"] == {
        "type": "numpy.float64",
        "encoding": "numpy_item:json_native",
        "value": 1.0,
    }
    assert body["execution_port"] == {
        "field": "execOut",
        "present": True,
        "used_as_timing": False,
        "raw": {
            "type": f"{__name__}.ExecutionState",
            "encoding": "enum_value:json_native",
            "value": 1,
        },
    }
    assert body["rgb_sha256"] == rgb_sha
    assert body["rgb_artifact"] == {
        "file": "job.capture-000003.rgb",
        "sha256": rgb_sha,
        "bytes": len(rgb),
    }
    assert (tmp_path / "job.capture-000003.rgb").read_bytes() == rgb
    assert "rgb bytes" not in path.read_text(encoding="utf-8")
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        A.write_capture_journal(job, 3, read_times, reference, rgb_sha, rgb_bytes=rgb)


def test_record_capture_journal_does_not_duplicate_rgb(tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    path = A.write_capture_journal(job, 7, _read_times(), _reference(), "2" * 64)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["rgb_artifact"] is None
    assert not (tmp_path / "job.capture-000007.rgb").exists()


def test_journal_preserves_unsupported_and_nonfinite_without_repr(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    read_times = _read_times()
    read_times["execOut"] = PybindLikeExecutionState()
    read_times["unexpectedNonfinite"] = float("nan")
    read_times["unexpectedVector"] = np.array([1, 2])
    path = A.write_capture_journal(job, 0, read_times, _reference(), "1" * 64)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["read_times"]["fields"]["execOut"] == {
        "type": f"{__name__}.PybindLikeExecutionState",
        "encoding": "unsupported",
        "value": None,
    }
    assert body["read_times"]["fields"]["unexpectedNonfinite"] == {
        "type": "builtins.float",
        "encoding": "json_nonfinite_marker",
        "value": {"nonfinite": "nan"},
    }
    assert body["read_times"]["fields"]["unexpectedVector"] == {
        "type": "numpy.ndarray",
        "encoding": "unsupported",
        "value": None,
    }


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_journal_precedes_shape_rejection_and_preserves_observed_keys(
    tmp_path: Path, change: str
) -> None:
    job = tmp_path / f"{change}.json"
    job.write_text("{}\n", encoding="utf-8")
    read_times = _read_times()
    if change == "missing":
        read_times.pop("frameNumber")
    else:
        read_times["unknown"] = object()
    path = A.write_capture_journal(job, 0, read_times, _reference(), "3" * 64)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert set(body["read_times"]["fields"]) == set(read_times)
    with pytest.raises(ValueError, match="11필드|12필드"):
        A.normalize_render_timing(read_times, _reference(), 1.0)
