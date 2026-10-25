"""동결 영상 선택 caller의 CPU 경계 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "evals"), str(ROOT / "src")]
import model_decision_journal_v1 as J
import test_vision_selection_contract_v1 as FIXTURE
import vision_selection_call_v1 as C


def _provider(content: str) -> dict:
    return {
        "provider_model": "private/nvidia/nemotron-3.5-super-120b-a12b",
        "finish_reason": "stop", "content": content,
        "input_tokens": 100, "output_tokens": 40, "total_tokens": 140, "wall_s": 0.1,
    }


def _case(tmp_path: Path, kind: str = "text_and_video") -> tuple[Path, dict, dict, dict]:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    reference = C.freeze(snapshot, contract, binding, kind, tmp_path / "frozen")
    return contract, binding, snapshot, reference


def _stage(tmp_path: Path, reference: dict, *, phase: str = "comparison",
           calls: tuple[str, ...] = ("call-01",),
           execution: str = "fixture-selection-v1") -> dict:
    rows = [{
        "call_id": call_id,
        "run_id": f"vision-selection-{execution}-{call_id}",
        "capsule": reference,
    } for call_id in calls]
    return C.initialize(tmp_path, execution, phase, "a" * 40, "b" * 64, rows)
def _valid(reference: dict, *, hold: bool = False) -> str:
    frozen = C.prepared(reference)
    fact = next(iter(frozen["bundle"]["selection_view"]["evidence_facts"]))
    return json.dumps({
        "action": "hold" if hold else "continue",
        "probe": None if hold else frozen["bundle"]["selection_view"]["eligible_probes"][0],
        "fact_ids": [fact],
    })


def test_freeze_contains_real_images_and_prepared_never_reads_live_inputs(tmp_path: Path) -> None:
    contract, binding, snapshot, reference = _case(tmp_path)
    frozen = C.prepared(reference)
    encoded = json.dumps(frozen["messages"])
    assert encoded.count("data:image/jpeg;base64,") == 6
    changed_snapshot = deepcopy(snapshot)
    changed_snapshot["view_digest"] = "f" * 64
    assert C.prepared(reference) == frozen

    stage = _stage(tmp_path, reference)
    C.run(stage, "call-01", lambda *_: _provider(_valid(reference)))
    replayed = C.run(
        stage, "call-01",
        lambda *_: (_ for _ in ()).throw(AssertionError("frozen terminal cache provider 호출")),
    )
    assert replayed["journal"]["api"] == {"reserved": 1, "pending": 0, "remaining": 71}
    activated = C.activate(
        stage, "call-01", snapshot, contract, binding, tmp_path / "valid.bridge.json",
    )
    assert activated["host"]["status"] == "request_ready"
    assert activated["host"]["approval_granted"] is False
    with pytest.raises(ValueError, match="snapshot|selection|상태|digest|다르다"):
        C.activate(stage, "call-01", changed_snapshot, contract, binding, tmp_path / "stale.bridge.json")
    assert not (tmp_path / "stale.bridge.json").exists()


@pytest.mark.parametrize("name", ["bundle.json", "messages.json", "capsule.json"])
def test_frozen_file_cross_tampering_is_rejected(tmp_path: Path, name: str) -> None:
    _, _, _, reference = _case(tmp_path)
    path = Path(reference["path"]).parent / name
    value = json.loads(path.read_text(encoding="utf-8"))
    value["tampered"] = True
    path.write_bytes(C._canonical(value))
    with pytest.raises(ValueError, match="SHA|형식|reference"):
        C.prepared(reference)


def test_plan_rejects_unregistered_order_runid_and_manifest_change(tmp_path: Path) -> None:
    _, _, _, reference = _case(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        C.initialize(tmp_path, "bad-plan", "comparison", "a" * 40, "b" * 64, [{
            "call_id": "call-01", "run_id": "different", "capsule": reference,
        }])
    stage = _stage(tmp_path, reference, calls=("call-01", "call-02"))
    assert _stage(tmp_path, reference, calls=("call-01", "call-02")) == stage
    with pytest.raises(J.JournalIntegrityError, match="plan"):
        _stage(tmp_path, reference, calls=("call-01",))
    called = []
    with pytest.raises(ValueError, match="앞 selection"):
        C.run(stage, "call-02", lambda *_: called.append(1) or _provider(_valid(reference)))
    assert called == []
    with pytest.raises(ValueError, match="등록되지 않은"):
        C.run(stage, "outside", lambda *_: called.append(1) or _provider(_valid(reference)))
    assert called == []

    manifest_path = Path(stage["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["api_cap"] = 73
    manifest_path.write_bytes(C._canonical(manifest))
    with pytest.raises((ValueError, J.JournalIntegrityError), match="reference SHA|seal|phase/API"):
        C.run(stage, "call-01", lambda *_: _provider(_valid(reference)))


def test_engine_allows_one_feedback_and_one_infra_retry_without_fallback(tmp_path: Path) -> None:
    _, _, _, reference = _case(tmp_path, "video_only")
    frozen = C.prepared(reference)
    responses = iter([
        {"status_code": 500, "infra_error": "InternalServerError"},
        _provider('{"action":"continue"}'),
        _provider(_valid(reference)),
    ])
    result = C.engine(frozen, lambda *_: next(responses))
    assert result["accepted"] is True and result["model_status"] == "valid"
    assert len(result["calls"]) == 3
    assert sum(row["transport"] == "infra_error" for row in result["calls"]) == 1
    assert sum("validation" in row and row["validation"]["status"] == "invalid"
               for row in result["calls"]) == 1
    assert "projected_choice" not in result and "request_spec" not in result


def test_global_cap_pending_and_terminal_cache_are_preserved(tmp_path: Path) -> None:
    _, _, _, reference = _case(tmp_path)
    calls = tuple(f"call-{index:02d}" for index in range(1, 6))
    stage = _stage(tmp_path, reference, phase="heldout", calls=calls)
    invalid = _provider('{"action":"continue"}')
    for call_id in calls[:4]:
        result = C.run(stage, call_id, lambda *_: invalid)
        assert result["journal"]["status"] == "terminal_hold"
    fifth = C.run(stage, calls[4], lambda *_: invalid)
    assert fifth["journal"]["status"] == "terminal_hold"
    assert fifth["journal"]["api"] == {"reserved": 10, "pending": 0, "remaining": 0}
    replay = C.run(
        stage, calls[0],
        lambda *_: (_ for _ in ()).throw(AssertionError("terminal cache provider 호출")),
    )
    assert replay["journal"]["api"]["reserved"] == 10

    pending_stage = _stage(
        tmp_path, reference, phase="heldout", calls=("pending-01",),
        execution="fixture-pending-v1",
    )
    _, pending_root, manifest = C._plan(pending_stage)
    _, row, frozen = C._row(tmp_path, manifest, "pending-01")
    journal = C._journal(pending_root, manifest, row, frozen)
    J._write_new(pending_root / "api/reservations/0001.json", {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": journal.manifest_sha256, "binding_sha256": journal.binding_sha256,
        "run_id": row["run_id"], "ordinal": 1,
        "messages_sha256": J.digest(frozen["messages"]),
        "params_sha256": journal.params_sha256, "reserved_at": "2026-10-25T00:00:00+00:00",
    })
    called = []
    blocked = C.run(pending_stage, "pending-01", lambda *_: called.append(1) or invalid)
    assert blocked["journal"]["status"] == "blocked_pending" and called == []


def test_activate_valid_hold_and_invalid_never_fall_back(tmp_path: Path) -> None:
    contract, binding, snapshot, reference = _case(tmp_path)
    for suffix, content, expected_reason in (
        ("hold", _valid(reference, hold=True), "model_hold"),
        ("invalid", '{"action":"continue"}', "model_invalid"),
    ):
        execution_root = tmp_path / suffix
        execution_root.mkdir()
        frozen_folder = execution_root / "frozen"
        source = Path(reference["path"]).parent
        for item in source.iterdir():
            frozen_folder.mkdir(exist_ok=True)
            (frozen_folder / item.name).write_bytes(item.read_bytes())
        capsule = frozen_folder / "capsule.json"
        copied = {"path": str(capsule.resolve()), "sha256": J.file_sha(capsule), "status": "frozen"}
        stage = _stage(execution_root, copied)
        C.run(stage, "call-01", lambda *_, _content=content: _provider(_content))
        activated = C.activate(
            stage, "call-01", snapshot, contract, binding, execution_root / "bridge.json",
        )
        assert activated["host"]["status"] == "HOLD"
        assert activated["host"]["reason"] == expected_reason
        assert activated["approval_granted"] is False
        assert activated["rule_fallback_used"] is False
        assert not (tmp_path / "case/probes.ledger.jsonl").exists()
