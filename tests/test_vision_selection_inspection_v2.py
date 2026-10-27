from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "evals"), str(ROOT / "src")]

import model_decision_journal_v1 as JOURNAL
import test_vision_selection_contract_v1 as FIXTURE
import vision_selection_call_v1 as CALL
import vision_selection_inspection_v2 as INSPECT


def _provider(content: str) -> dict:
    return {
        "provider_model": "private/nvidia/nemotron-3.5-super-120b-a12b",
        "finish_reason": "stop", "content": content,
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "wall_s": 0.01,
    }


def _case(tmp_path: Path, count: int = 3) -> tuple[dict, dict, dict]:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path / "case")
    reference = CALL.freeze(snapshot, contract, binding, "text_only", tmp_path / "frozen")
    execution = "inspection-v2-fixture"
    rows = [{"call_id": f"choice-{index:02d}",
             "run_id": f"vision-selection-{execution}-choice-{index:02d}",
             "capsule": reference} for index in range(1, count + 1)]
    plan = CALL.initialize(tmp_path, execution, "comparison", "a" * 40, "b" * 64, rows)
    repository, stage, manifest = CALL._plan(plan)
    protocol = tmp_path / "protocol.json"
    locations = tmp_path / "locations.json"
    protocol.write_text("{}\n", encoding="utf-8")
    locations.write_text("{}\n", encoding="utf-8")
    context = {"repository": repository, "stage": stage, "manifest": manifest,
               "prepared": {"plan": plan}, "protocol_path": protocol,
               "protocol_sha256_lf": "b" * 64, "source_anchor": "a" * 40,
               "input_locations": locations}
    frozen = CALL.prepared(reference)
    fact = next(iter(frozen["bundle"]["selection_view"]["evidence_facts"]))
    content = json.dumps({"action": "hold", "probe": None, "fact_ids": [fact]})
    return context, plan, {"reference": reference, "content": content}


def _tree(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _complete(plan: dict, call_id: str, content: str) -> None:
    result = CALL.run(plan, call_id, lambda *_: _provider(content))
    assert result["journal"]["status"] == "terminal_hold"


def _journal(context: dict, call_id: str):
    row = next(row for row in context["manifest"]["rows"] if row["call_id"] == call_id)
    _, checked, frozen = CALL._row(context["repository"], context["manifest"], call_id)
    assert checked == row
    return CALL._journal(context["stage"], context["manifest"], checked, frozen), frozen


def test_initial_inspection_is_read_only_and_runnable(tmp_path):
    context, _plan, _data = _case(tmp_path)
    before = _tree(tmp_path)
    state = INSPECT.inspect_run(context)
    assert state["status"] == "runnable" and state["next_call_id"] == "choice-01"
    assert state["api"] == {"cap": 72, "reserved": 0, "pending": 0, "remaining": 72}
    assert _tree(tmp_path) == before
    assert not (context["stage"] / "api").exists()
    assert not (context["stage"] / "decisions").exists()
    command = INSPECT.remaining_commands(context)
    assert command["status"] == "runnable"
    assert command["commands"][0][-2:] == ["--call-id", "choice-01"]
    assert _tree(tmp_path) == before


def test_first_and_second_terminal_leave_next_unstarted_runnable_without_writes(tmp_path):
    context, plan, data = _case(tmp_path)
    _complete(plan, "choice-01", data["content"])
    before = _tree(tmp_path)
    first = INSPECT.inspect_run(context)
    assert first["status"] == "runnable" and first["next_call_id"] == "choice-02"
    assert first["counts"]["terminal"] == 1 and first["counts"]["unstarted"] == 2
    assert _tree(tmp_path) == before
    assert not (context["stage"] / "decisions/choice-02").exists()

    _complete(plan, "choice-02", data["content"])
    before = _tree(tmp_path)
    second = INSPECT.inspect_run(context)
    assert second["status"] == "runnable" and second["next_call_id"] == "choice-03"
    assert second["api"]["reserved"] == 2
    assert _tree(tmp_path) == before


def test_precreated_empty_registered_manifests_are_read_only_and_runnable(tmp_path):
    context, _plan, _data = _case(tmp_path)
    for row in context["manifest"]["rows"]:
        _journal(context, row["call_id"])
    before = _tree(tmp_path)
    state = INSPECT.inspect_run(context)
    assert state["status"] == "runnable" and state["next_call_id"] == "choice-01"
    assert state["api"]["pending"] == 0
    assert _tree(tmp_path) == before


def test_unknown_pending_is_hold_and_never_requests_again(tmp_path):
    context, _plan, _data = _case(tmp_path)
    journal, frozen = _journal(context, "choice-01")
    JOURNAL._write_new(context["stage"] / "api/reservations/0001.json", {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": journal.manifest_sha256, "binding_sha256": journal.binding_sha256,
        "run_id": journal.binding["run_id"], "ordinal": 1,
        "messages_sha256": JOURNAL.digest(frozen["messages"]),
        "params_sha256": journal.params_sha256, "reserved_at": "2026-10-27T00:00:00+00:00",
    })
    before = _tree(tmp_path)
    state = INSPECT.inspect_run(context)
    assert state["status"] == "blocked_pending" and state["decision"] == "HOLD"
    assert state["api"]["pending"] == 1
    assert INSPECT.remaining_commands(context)["commands"] == []
    assert _tree(tmp_path) == before


def test_genuine_partial_decision_without_manifest_is_rejected(tmp_path):
    context, _plan, _data = _case(tmp_path)
    partial = context["stage"] / "decisions/choice-01/events"
    partial.mkdir(parents=True)
    (partial / "0001.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(JOURNAL.JournalIntegrityError, match="manifest가 없다"):
        INSPECT.inspect_run(context)


def test_terminal_order_gap_is_rejected(tmp_path):
    context, plan, data = _case(tmp_path)
    _complete(plan, "choice-01", data["content"])
    _complete(plan, "choice-02", data["content"])
    (context["stage"] / "decisions/choice-01/terminal.json").unlink()

    with pytest.raises(JOURNAL.JournalIntegrityError, match="순서에 gap"):
        INSPECT.inspect_run(context)


def test_missing_owner_and_foreign_reservations_fail_closed(tmp_path):
    context, _plan, _data = _case(tmp_path)
    first, frozen = _journal(context, "choice-01")
    missing, _ = _journal(context, "choice-02")
    (context["stage"] / "decisions/choice-02/manifest.json").unlink()
    (context["stage"] / "decisions/choice-02").rmdir()
    reservation = {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": missing.manifest_sha256, "binding_sha256": missing.binding_sha256,
        "run_id": missing.binding["run_id"], "ordinal": 1,
        "messages_sha256": JOURNAL.digest(frozen["messages"]),
        "params_sha256": first.params_sha256, "reserved_at": "2026-10-27T00:00:00+00:00",
    }
    JOURNAL._write_new(context["stage"] / "api/reservations/0001.json", reservation)
    with pytest.raises(JOURNAL.JournalIntegrityError, match="소유 decision manifest"):
        INSPECT.inspect_run(context)

    reservation["manifest_sha256"] = "f" * 64
    path = context["stage"] / "api/reservations/0001.json"
    path.write_text(json.dumps(reservation) + "\n", encoding="utf-8")
    with pytest.raises(JOURNAL.JournalIntegrityError, match="등록 밖 decision"):
        INSPECT.inspect_run(context)


def test_tampered_api_plan_and_capsule_are_rejected(tmp_path):
    context, _plan, _data = _case(tmp_path)
    _journal(context, "choice-01")
    api_manifest = context["stage"] / "api/manifest.json"
    api_manifest.write_text(json.dumps({"schema": JOURNAL.API_SCHEMA, "api_cap": 71}) + "\n",
                            encoding="utf-8")
    with pytest.raises(JOURNAL.JournalIntegrityError, match="manifest"):
        INSPECT.inspect_run(context)

    context2, _plan2, _data2 = _case(tmp_path / "plan")
    plan_manifest = context2["stage"] / "manifest.json"
    body = json.loads(plan_manifest.read_text(encoding="utf-8"))
    body["source_anchor"] = "c" * 40
    plan_manifest.write_bytes(CALL._canonical(body))
    with pytest.raises((ValueError, JOURNAL.JournalIntegrityError), match="reference SHA|seal"):
        INSPECT.inspect_run(context2)

    context3, _plan3, data3 = _case(tmp_path / "capsule")
    bundle = Path(data3["reference"]["path"]).parent / "bundle.json"
    changed = json.loads(bundle.read_text(encoding="utf-8"))
    changed["tampered"] = True
    bundle.write_bytes(CALL._canonical(changed))
    with pytest.raises(ValueError, match="SHA"):
        INSPECT.inspect_run(context3)


def test_tampered_decision_manifest_is_rejected(tmp_path):
    context, _plan, _data = _case(tmp_path)
    _journal(context, "choice-01")
    decision_manifest = context["stage"] / "decisions/choice-01/manifest.json"
    body = json.loads(decision_manifest.read_text(encoding="utf-8"))
    body["retry_delay_s"] = 31.0
    decision_manifest.write_bytes(CALL._canonical(body))

    with pytest.raises(JOURNAL.JournalIntegrityError, match="decision choice-01 manifest"):
        INSPECT.inspect_run(context)
