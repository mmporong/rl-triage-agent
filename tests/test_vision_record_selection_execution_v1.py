from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "tests", ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import test_vision_record_source_readiness_v1 as READY_FIXTURE  # pyright: ignore[reportMissingImports]
import test_vision_selection_execution_v1 as LEGACY_FIXTURE  # pyright: ignore[reportMissingImports]
import vision_record_selection_execution_v1 as E  # pyright: ignore[reportMissingImports]

UTC = timezone.utc  # noqa: UP017 - Python 3.10


def test_load_context_uses_record_worker_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.json"
    locations = tmp_path / "locations.json"
    protocol.write_text("{}", encoding="utf-8")
    locations.write_text("{}", encoding="utf-8")
    plan = {"path": str(tmp_path / "plan.json")}
    prepared = {"plan": plan, "protocol": {"rows": []}}
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        E.WORKER,
        "prepare",
        lambda *args: calls.append(args) or prepared,
    )
    monkeypatch.setattr(
        E.CALL,
        "_plan",
        lambda value: (tmp_path, tmp_path / "stage", {"rows": [], "api_cap": 72}),
    )
    context = E.load_context(protocol, "a" * 64, "b" * 40, locations)
    assert calls == [(protocol, "a" * 64, "b" * 40, locations)]
    assert context["prepared"] is prepared


def test_legacy_twenty_four_slot_state_emits_new_worker_command_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = LEGACY_FIXTURE._context(tmp_path)
    monkeypatch.setattr(
        E.CALL,
        "_row",
        lambda repo, manifest, call_id: (
            0,
            next(row for row in manifest["rows"] if row["call_id"] == call_id),
            {"messages": []},
        ),
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = E.remaining_commands(context, datetime(2026, 10, 25, tzinfo=UTC))
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    command = result["commands"][0]
    assert result["status"] == "runnable" and result["next_call_id"] == "choice-01"
    assert command[0] == "setsid"
    assert Path(command[2]).name == "vision_record_selection_worker_v1.py"
    assert Path(command[2]).parent.name == "evals"
    assert command[-2:] == ["--call-id", "choice-01"]
    assert before == after


def test_first_class_readiness_fixture_recomputes_bound_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    binding, job, smoke_job, result, execution = cast(
        Any, READY_FIXTURE._case(repo)
    )
    READY_FIXTURE._patch_terminals(
        monkeypatch, job, smoke_job, result, execution
    )
    monkeypatch.setattr(E.READINESS.SOURCE, "validate_protocol", lambda value: value)
    output = repo / "evidence/readiness.json"
    readiness = E.READINESS.build_readiness(repo, binding, output)
    protocol = READY_FIXTURE._source_protocol(repo, readiness)
    assert E.READINESS.verify_model_readiness(repo, output, protocol) == readiness["proof"]
    video = repo / readiness["record"]["video"]["path"]
    video.write_bytes(video.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA"):
        E.READINESS.verify_model_readiness(repo, output, protocol)


def test_evaluate_evidence_uses_new_readiness_and_preserves_receipt_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, context, source, rows, evaluation, _folder = (
        LEGACY_FIXTURE._evaluation_fixture(tmp_path)
    )
    calls: list[Path] = []

    def verify(_root: Path, path: Path, _protocol: dict[str, Any]) -> dict[str, Any]:
        calls.append(path)
        role = "registered-f1" if "registered-f1" in path.as_posix() else "normal"
        return {
            "checkpoint_sha256": "b" * 64 if role == "registered-f1" else "a" * 64,
            "source_seed": 4101,
            "source_role": role,
        }

    monkeypatch.setattr(E.READINESS, "verify_model_readiness", verify)
    reports, artifacts = E._evaluate_evidence(
        root, context, "source-01", evaluation, source, rows
    )
    assert set(reports) == {"NONE", "F1"}
    assert artifacts["NONE"]["checkpoint"]["sha256"] == "a" * 64
    assert len(calls) == 6
    assert {"normal", "registered-f1"} == {
        "registered-f1" if "registered-f1" in path.as_posix() else "normal"
        for path in calls
    }

    receipt_path = root / evaluation["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["raw_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    evaluation["receipt"] = LEGACY_FIXTURE._ref(root, receipt_path)
    with pytest.raises(ValueError, match="receipt SHA"):
        E._evaluate_evidence(root, context, "source-01", evaluation, source, rows)


def test_summarize_reuses_existing_projection_budget_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = LEGACY_FIXTURE._context(tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    state = {
        "status": "complete",
        "api": {"remaining": 30},
        "counts": {"terminal": 24},
        "next_call_id": None,
    }
    monkeypatch.setattr(E, "inspect_run", lambda *_: state)
    monkeypatch.setattr(E, "project_results", lambda *_: [])
    monkeypatch.setattr(E, "_evidence", lambda *_: ({}, {}, "c" * 64))
    monkeypatch.setattr(
        E, "_usage", lambda *_: {"requests": 0, "tokens": {"known": 0, "missing": 0, "sum": 0}, "provider_wall_s_sum": 0.0}
    )
    monkeypatch.setattr(
        E.COMPARISON,
        "summarize",
        lambda *args, **kwargs: {"schema": "summary", "execution_status": "complete"},
    )
    output = tmp_path / "out"
    first = E.summarize(context, evidence, output, True)
    second = E.summarize(context, evidence, output, True)
    assert first == second and first["finalized"] is True
