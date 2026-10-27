from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_pull_record_capture_v2 as C  # pyright: ignore[reportMissingImports]

SIM_TIME = 0.009999999776482582
READ_TIMES = {
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
REFERENCE_TIME = {
    "referenceTimeDenominator": 30,
    "referenceTimeNumerator": 46,
}


def _capture_result() -> dict[str, Any]:
    return {
        "rgb": b"rgb",
        "read_times": dict(READ_TIMES),
        "reference_time": dict(REFERENCE_TIME),
        "pull_receipt": Path("pull.json"),
        "pull_raw_receipt": Path("raw.json"),
        "pull_npz": Path("raw.npz"),
    }


def test_actual_twelve_key_fixture_normalizes_for_atomic_consumer() -> None:
    raw = _capture_result()
    result = C.normalize_capture_result(raw, simulation_time_s=SIM_TIME)
    assert set(result["read_times"]) == set(C.TIMING.R.ISAAC_READ_TIMES_FIELDS)
    assert "execOut" not in result["read_times"]
    assert raw["read_times"]["execOut"] == 1
    assert C.TIMING.R.validate_render_timing(
        {"read_times": result["read_times"], "reference_time": result["reference_time"]},
        SIM_TIME,
    ) == {"read_times": result["read_times"], "reference_time": result["reference_time"]}


def test_factory_preserves_v1_raw_writer_and_only_normalizes_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _capture_result()
    cleanup = object()

    def previous_factory(*_args: Any, **_kwargs: Any) -> Any:
        return (lambda *_capture_args: raw), cleanup

    monkeypatch.setattr(C.PREVIOUS, "make_capture_factory", lambda *_a, **_k: previous_factory)
    factory = C.make_capture_factory({}, object())
    capture, observed_cleanup = factory(object(), object(), object(), {})
    base = SimpleNamespace(sim=SimpleNamespace(current_time=SIM_TIME))
    result = capture(base, object(), [], {}, [], [], 0)
    assert observed_cleanup is cleanup
    assert "execOut" not in result["read_times"]
    assert raw["read_times"]["execOut"] == 1


@pytest.mark.parametrize("mutation, message", [
    (lambda row: row["read_times"].update({"unknown": 1}), "11필드 또는"),
    (lambda row: row["read_times"].update({"simulationTime": 1.0}), "simulation state"),
])
def test_normalizer_rejects_unknown_shape_and_time_mismatch(
    mutation: Any, message: str,
) -> None:
    raw = _capture_result()
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        C.normalize_capture_result(raw, simulation_time_s=SIM_TIME)
