from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_pull_record_worker_v2 as W  # pyright: ignore[reportMissingImports]

SIM_TIME = 0.009999999776482582
RAW_TIMES = {
    "durationDenominator": 18446744073709551615,
    "durationNumerator": 0,
    "execOut": 1,
    "externalTimeOfSimNs": -1,
    "frameNumber": 46,
    "rationalTimeOfSimDenominator": 30,
    "rationalTimeOfSimNumerator": 46,
    "sampleTimeOffsetInSimFrames": 18446744073709551615,
    "simulationTime": SIM_TIME,
    "simulationTimeMonotonic": SIM_TIME,
    "swhFrameNumber": 0,
    "systemTime": 1793090838.3596811,
}
REFERENCE = {"referenceTimeDenominator": 30, "referenceTimeNumerator": 46}


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, trace_mutation: Any = None, raw_mutation: Any = None) -> tuple[dict[str, Any], dict[str, Path]]:
    job = _write(tmp_path / "job.json", {})
    output = tmp_path / "output"
    output.mkdir()
    validated = {"job_path": job, "checkpoint_sha256": "c" * 64}
    npz, raw_path, receipt_path = W.PREVIOUS._receipt_paths(tmp_path, 0)
    npz.write_bytes(b"npz")
    raw_times = dict(RAW_TIMES)
    if raw_mutation:
        raw_mutation(raw_times)
    raw = {
        "schema": W.CAPTURE.RAW_SCHEMA,
        "capture_ordinal": 0,
        "npz": W.PREVIOUS._descriptor(npz),
        "raw_fields": list(W.CAPTURE.ANNOTATORS),
        "read_times": raw_times,
        "reference_time": REFERENCE,
        "reserved_marker_count": 0,
    }
    _write(raw_path, raw)
    _write(receipt_path, {
        "schema": W.CAPTURE.SCHEMA,
        "capture_ordinal": 0,
        "annotator_order": list(W.CAPTURE.ANNOTATORS),
        "complete_capture": True,
        "reserved_marker_count": 0,
        "atomicity_claimed": False,
        "capture_approved": False,
        "used_for_decision": False,
        "raw_npz": W.PREVIOUS._descriptor(npz),
        "raw_receipt": W.PREVIOUS._descriptor(raw_path),
        "app_update_counter": {
            "source": "omni.kit.app.IApp.get_update_number",
            "fail_closed": True,
            "pull_window_updates": 0,
            "render_window_updates": 1,
            "after_each_pull": [47] * 5,
        },
        "step": {"call_count": 1, "delta_time_s": 0.0},
        "physics_before": {"simulation_time_s": SIM_TIME},
        "physics_after": {"simulation_time_s": SIM_TIME},
        "frame_number": 46,
        "rational_time": [46, 30],
    })
    timing = W.TIMING.normalize_render_timing(RAW_TIMES, REFERENCE, SIM_TIME)
    record = {
        "capture_ordinal": 0,
        "state": {"simulation_time_s": SIM_TIME},
        "render_timing": timing,
    }
    if trace_mutation:
        trace_mutation(record)
    trace = _write(output / "raw_trace.json", {"records": [record]})
    return validated, {"raw_trace": trace}


def test_actual_raw_receipt_normalized_trace_and_manifest_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(W.RECORD, "FRAME_COUNT", 1)
    validated, artifacts = _fixture(tmp_path)
    manifest = W._manifest(validated, artifacts)
    assert manifest["schema"] == W.MANIFEST_SCHEMA
    assert manifest["frame_count"] == 1
    assert manifest["frames"][0]["frame_number"] == 46
    raw = json.loads((tmp_path / "job.pull-record-000000.raw.json").read_text())
    assert raw["read_times"]["execOut"] == 1


@pytest.mark.parametrize("kind", ["unknown", "physics"])
def test_manifest_rejects_unknown_trace_field_and_physics_time_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    monkeypatch.setattr(W.RECORD, "FRAME_COUNT", 1)

    def mutate(record: dict[str, Any]) -> None:
        if kind == "unknown":
            record["render_timing"]["read_times"]["unknown"] = 1
        else:
            record["state"]["simulation_time_s"] = 1.0

    validated, artifacts = _fixture(tmp_path, trace_mutation=mutate)
    with pytest.raises(ValueError):
        W._manifest(validated, artifacts)


def test_manifest_rejects_unknown_raw_timing_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(W.RECORD, "FRAME_COUNT", 1)
    validated, artifacts = _fixture(
        tmp_path, raw_mutation=lambda timing: timing.update({"unknown": 1})
    )
    with pytest.raises(ValueError, match="11필드 또는"):
        W._manifest(validated, artifacts)
