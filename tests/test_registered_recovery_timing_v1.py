import json
from pathlib import Path

import pytest

from evals import registered_recovery_timing_v1 as timing
from rl_triage import investigation_session as io


def _write(path: Path, value: dict) -> None:
    io.write_new(path, value)


def _begin(sequence: int = 1) -> dict:
    binding = {"sequence": sequence, "case_id": "case-05", "method": "fixed_sequence",
               "started_utc": "2026-11-04T00:00:00+00:00", "started_perf_ns": 1_000_000_000}
    return {"schema": timing.BEGIN_SCHEMA, **binding, "transition_id": io.digest(binding)}


def test_transition_records_active_wall_and_window_span_without_double_counting(tmp_path: Path) -> None:
    with timing.transition(tmp_path, case_id="case-05", method="fixed_sequence"):
        pass
    with timing.transition(tmp_path, case_id="case-05", method="fixed_sequence"):
        pass
    result = timing.read_timing(tmp_path)
    assert result["transition_count"] == 2 and result["unknown_transition_count"] == 0
    assert result["active_command_wall_s"] == result["known_active_command_wall_s"]
    assert result["window_span_s"] >= result["active_command_wall_s"]
    assert result["per_arm"][0]["completed_transitions"] == 2
    assert len(result["private_refs"]) == 4


def test_read_timing_is_read_only(tmp_path: Path) -> None:
    with timing.transition(tmp_path, case_id="case-05", method="static_rule"):
        pass
    files = sorted((tmp_path / "timing/transitions").iterdir())
    before = [(path.name, path.stat().st_mtime_ns, io.file_sha(path)) for path in files]
    timing.read_timing(tmp_path)
    after = [(path.name, path.stat().st_mtime_ns, io.file_sha(path)) for path in files]
    assert after == before


def test_begin_only_crash_stays_unknown_and_blocks_next_transition(tmp_path: Path) -> None:
    _write(tmp_path / "timing/transitions/000001-begin.json", _begin())
    result = timing.read_timing(tmp_path)
    assert result["transition_count"] == 1 and result["unknown_transition_count"] == 1
    assert result["active_command_wall_s"] is None and result["window_span_s"] is None
    assert result["transitions"][0]["status"] == "unknown_crash"
    with (pytest.raises(timing.TimingError, match="incomplete"),
          timing.transition(tmp_path, case_id="case-05", method="fixed_sequence")):
        pass


def test_clock_mismatch_is_unknown_not_zero(tmp_path: Path) -> None:
    begin = _begin()
    begin_path = tmp_path / "timing/transitions/000001-begin.json"
    _write(begin_path, begin)
    _write(tmp_path / "timing/transitions/000001-end.json", {
        "schema": timing.END_SCHEMA, "sequence": 1, "case_id": "case-05",
        "method": "fixed_sequence", "transition_id": begin["transition_id"],
        "begin_sha256": io.file_sha(begin_path), "ended_utc": "2026-11-04T00:00:30+00:00",
        "ended_perf_ns": 2_000_000_000})
    result = timing.read_timing(tmp_path)
    assert result["unknown_transition_count"] == 1
    assert result["known_active_command_wall_s"] is None
    assert result["active_command_wall_s"] is None


def test_duplicate_unexpected_or_mutated_timing_artifacts_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "timing/transitions"
    _write(root / "000001-begin.json", _begin())
    (root / "notes.json").write_text(json.dumps({"caller": "trusted"}), encoding="utf-8")
    with pytest.raises(timing.TimingError, match="unexpected"):
        timing.read_timing(tmp_path)
    (root / "notes.json").unlink()
    begin = json.loads((root / "000001-begin.json").read_text(encoding="utf-8"))
    begin["case_id"] = "case05"
    (root / "000001-begin.json").write_text(json.dumps(begin), encoding="utf-8")
    with pytest.raises(timing.TimingError, match="begin binding"):
        timing.read_timing(tmp_path)


def test_duplicated_begin_identity_under_new_sequence_is_rejected(tmp_path: Path) -> None:
    first = _begin(1)
    second = {**_begin(2), "started_utc": "2026-11-04T00:00:01+00:00",
              "started_perf_ns": 2_000_000_000, "transition_id": first["transition_id"]}
    _write(tmp_path / "timing/transitions/000001-begin.json", first)
    _write(tmp_path / "timing/transitions/000002-begin.json", second)
    with pytest.raises(timing.TimingError, match="identity differs"):
        timing.read_timing(tmp_path)
