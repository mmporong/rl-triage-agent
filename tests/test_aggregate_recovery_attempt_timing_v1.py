from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import evals.aggregate_recovery_attempt_timing_v1 as A
from evals import registered_recovery_timing_v1 as T
from rl_triage import investigation_session as io


def write_attempt(
    tmp_path: Path,
    name: str,
    *,
    started: datetime,
    duration_s: float = 2.0,
    case_id: str = "case-05",
    method: str = "fixed_sequence",
    incomplete: bool = False,
    manifest_nonce: str = "",
) -> Path:
    root = tmp_path / name
    transitions = root / "timing" / "transitions"
    transitions.mkdir(parents=True)
    plan = tmp_path / f"{name}-plan.json"
    plan.write_text(json.dumps({"schema": "fixture-plan", "name": name}), encoding="utf-8")
    manifest = {
        "schema": A.MANIFEST_SCHEMA,
        "plan": {"path": str(plan.resolve()), "sha256": io.file_sha(plan)},
        "nonce": manifest_nonce or name,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    anchor = {"schema": A.ANCHOR_SCHEMA, "manifest_sha256": io.file_sha(manifest_path)}
    (root / "anchor.json").write_text(json.dumps(anchor), encoding="utf-8")

    started_perf_ns = 1_000_000_000
    binding = {
        "sequence": 1,
        "case_id": case_id,
        "method": method,
        "started_utc": started.isoformat(),
        "started_perf_ns": started_perf_ns,
    }
    transition_id = io.digest(binding)
    begin = {
        "schema": T.BEGIN_SCHEMA,
        **binding,
        "transition_id": transition_id,
    }
    begin_path = transitions / "000001-begin.json"
    begin_path.write_text(json.dumps(begin), encoding="utf-8")
    if not incomplete:
        end = {
            "schema": T.END_SCHEMA,
            "sequence": 1,
            "case_id": case_id,
            "method": method,
            "transition_id": transition_id,
            "begin_sha256": io.file_sha(begin_path),
            "ended_utc": (started + timedelta(seconds=duration_s)).isoformat(),
            "ended_perf_ns": started_perf_ns + int(duration_s * 1_000_000_000),
        }
        (transitions / "000001-end.json").write_text(json.dumps(end), encoding="utf-8")
    return root


def test_aggregate_separates_active_commands_from_all_attempt_wall_span(tmp_path: Path) -> None:
    first = write_attempt(
        tmp_path,
        "v3",
        started=datetime(2026, 11, 4, 0, 0, tzinfo=UTC),
        duration_s=2.0,
    )
    second = write_attempt(
        tmp_path,
        "v4",
        started=datetime(2026, 11, 4, 0, 10, tzinfo=UTC),
        duration_s=3.0,
        method="static_rule",
    )
    result = A.aggregate([first, second])
    assert result["status"] == "complete"
    assert result["transition_count"] == 2
    assert result["active_command_wall_s"] == 5.0
    assert result["all_attempt_wall_span_s"] == 603.0
    assert result["all_attempt_wall_span_is_algorithm_cost"] is False
    assert result["all_attempt_wall_span_is_gpu_owned_time"] is False
    assert len(result["per_arm"]) == 16
    unseen = next(
        row
        for row in result["per_arm"]
        if row["case_id"] == "case-08" and row["method"] == "iterative_agent"
    )
    assert unseen["observation_status"] == "not_observed"
    assert unseen["active_command_wall_s"] == 0.0


def test_begin_only_keeps_complete_totals_null_and_known_subset_separate(tmp_path: Path) -> None:
    complete = write_attempt(
        tmp_path,
        "complete",
        started=datetime(2026, 11, 4, 0, 0, tzinfo=UTC),
        duration_s=2.0,
    )
    incomplete = write_attempt(
        tmp_path,
        "incomplete",
        started=datetime(2026, 11, 4, 0, 5, tzinfo=UTC),
        incomplete=True,
        method="static_rule",
    )
    result = A.aggregate([complete, incomplete])
    assert result["status"] == "incomplete"
    assert result["unknown_transition_count"] == 1
    assert result["known_active_command_wall_s"] == 2.0
    assert result["active_command_wall_s"] is None
    assert result["known_completed_transition_span_s"] == 2.0
    assert result["all_attempt_wall_span_s"] is None


def test_duplicate_root_and_transition_identity_are_rejected(tmp_path: Path) -> None:
    started = datetime(2026, 11, 4, 0, 0, tzinfo=UTC)
    first = write_attempt(tmp_path, "first", started=started)
    with pytest.raises(A.AttemptTimingError, match="prepared root"):
        A.aggregate([first, first])
    duplicate = write_attempt(
        tmp_path,
        "duplicate",
        started=started,
        manifest_nonce="different-source",
    )
    with pytest.raises(A.AttemptTimingError, match="transition ID"):
        A.aggregate([first, duplicate])


def test_manifest_anchor_and_plan_reference_are_reverified(tmp_path: Path) -> None:
    root = write_attempt(
        tmp_path,
        "attempt",
        started=datetime(2026, 11, 4, 0, 0, tzinfo=UTC),
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    Path(manifest["plan"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(A.AttemptTimingError, match="plan reference"):
        A.aggregate([root])


def test_cli_is_write_once(tmp_path: Path) -> None:
    root = write_attempt(
        tmp_path,
        "attempt",
        started=datetime(2026, 11, 4, 0, 0, tzinfo=UTC),
    )
    out = tmp_path / "summary.json"
    argv = ["--prepared", str(root), "--out", str(out)]
    assert A.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    with pytest.raises(FileExistsError):
        A.main(argv)
