from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_recovery_candidate_v1 as C2


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def test_typed_backend_scope_restores_candidate_v1_backend() -> None:
    original = C2.CANDIDATE1.P
    with C2.typed_backend_scope():
        assert C2.CANDIDATE1.P is C2.POST_BACKEND
    assert C2.CANDIDATE1.P is original


@pytest.mark.parametrize(
    ("evidence_scope", "end_state", "hold_reason"),
    [("cpu_fixture", "confirmed", None),
     ("actual_backend", "HOLD", "probe_outcome_unknown")],
)
def test_candidate_rejects_fixture_or_unknown_epoch2(
        tmp_path: Path, evidence_scope: str, end_state: str, hold_reason: str | None) -> None:
    result = {
        "schema": C2.E2.SCHEMA, "status": "complete", "evidence_scope": evidence_scope,
        "epoch": 2, "epoch2_registered": True, "end_state": end_state,
        "hold_reason": hold_reason, "repair_supported": True,
        "correction_executed": False, "training_executed": False,
    }
    path = tmp_path / "result.json"
    _write(path, result)
    with pytest.raises(ValueError):
        C2.prepare_candidate(
            path, post_input=tmp_path / "post.json", reference_input=tmp_path / "none.json",
            gate_path=tmp_path / "gate.json", gate_sha256="a" * 64, case_id="case-07",
        )


def test_completed_ledger_rejects_pending_or_tampered_sequence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLedger:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def events(self) -> list[dict]:
            return [{"event": "proposed", "request_id": "r1"}]

        def requests(self) -> dict:
            return {"r1": {"state": "approved"}}

    monkeypatch.setattr(C2.C.S.M, "RegistryLedger", FakeLedger)
    with pytest.raises(ValueError, match="pending/unknown"):
        C2._completed_ledger(tmp_path / "ledger.jsonl", [{"request_id": "r1"}])


def test_second_registered_correction_must_be_distinct(
        monkeypatch: pytest.MonkeyPatch) -> None:
    applied = ["restore_body_frame_reward"]
    monkeypatch.setattr(
        C2.C, "registered_correction",
        lambda _final, applied: ("reward", applied[0]),
    )
    with pytest.raises(ValueError, match="달라야"):
        C2._registered_second_correction({}, applied, "reward")

    monkeypatch.setattr(
        C2.C, "registered_correction",
        lambda _final, applied: ("optimizer_update_count", "restore_configured_ppo_epochs"),
    )
    assert C2._registered_second_correction({}, applied, "optimizer_update_count") == (
        "optimizer_update_count", "restore_configured_ppo_epochs")
