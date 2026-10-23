"""D4 조건부 caller의 20-slot·60-request·portable capsule 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import selection_comparison_conditional_call_v1 as C  # noqa: E402

VIEW_SHA = C.V.file_digest(C.V.PROTOCOL)
FACTS = {"config:action_scale": 0.25}
BINDINGS = {"case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
            "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64}}


def packet(epoch: int = 1) -> dict:
    return C.V.S.build_packet(epoch=epoch, applied_corrections=[], prior_receipt=None,
                              facts=FACTS, input_bindings=deepcopy(BINDINGS), execution_budget_s=1200)


def snapshot(root: Path, policy: str = "model_adaptive", epoch: int = 1) -> dict:
    ledger = root / policy / f"epoch{epoch}" / "probes.ledger.jsonl"
    return C.V.build_snapshot(packet(epoch), f"conditional_{policy}", policy, VIEW_SHA, ledger)


def slot(plan: dict, slot_id: str) -> dict:
    return next(row for row in plan["slots"] if row["slot_id"] == slot_id)


def response(value: dict, *, valid: bool = True) -> dict:
    view = value["view"]
    body = ({"action": "continue", "probe": view["eligible_probes"][0],
             "fact_ids": [next(iter(view["evidence_facts"]))]} if valid else {"bad": True})
    return {"provider_model": C.MODEL, "finish_reason": "stop", "content": json.dumps(body),
            "input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "wall_s": 0.01}


def add_observation(value: dict, outcome: str = "normal") -> dict:
    inputs = value["inputs"]
    ledger = C.V.S.M.RegistryLedger(inputs["ledger_path"], C.V.S.C1.registry())
    probe = value["view"]["eligible_probes"][0]
    request = ledger.propose(value["binding"], probe, 1200)
    ledger.approve(request["request_id"], "cpu")
    ledger.consume(request["request_id"], value["binding"], probe)
    measurement = ({"timeout_ratio": 0.1 if outcome == "abnormal" else 1.0}
                   if probe == "P_episode" else {"status": outcome})
    ledger.receipt(request["request_id"], outcome, measurement, 0.0, 0)
    return C.V.build_snapshot(inputs["packet"], inputs["run_id"], inputs["policy"],
                              inputs["protocol_sha256"], inputs["ledger_path"], inputs["parent"])


def setup_executed(tmp_path: Path, *, slot_id: str = "model_adaptive:e1:open0") -> tuple[dict, Path, dict, dict]:
    plan = C.plan_value()
    root = tmp_path / "caller"
    C.initialize(root, plan)
    value = snapshot(tmp_path / "source")
    materialized = C.materialize_slot(slot(plan, slot_id), value)
    receipt = C.record_slot(root, plan, materialized)
    return plan, root, value, receipt


def test_static_plan_has_twenty_slots_and_sixty_request_bound() -> None:
    plan = C.plan_value()
    assert plan["slot_count"] == 20
    assert len(plan["slots"]) == 20
    assert sum(row["max_requests"] for row in plan["slots"]) == 60
    assert sum(row["policy"] == "model_first_then_rule" for row in plan["slots"]) == 2
    assert sum(row["policy"] == "model_adaptive" and row["condition"] == "production"
               for row in plan["slots"]) == 14
    assert sum(row["condition"] != "production" for row in plan["slots"]) == 4
    assert all(set(row) == C.SLOT_FIELDS and row["plan_sha256"] == plan["plan_sha256"]
               for row in plan["slots"])

    changed = deepcopy(plan)
    changed["slots"][0]["max_requests"] = 2
    with pytest.raises(ValueError, match="plan drift"):
        C.validate_plan(changed)


def test_portable_capsule_removes_windows_source_path_and_rebuilds_from_frozen_history(tmp_path: Path) -> None:
    plan = C.plan_value()
    source_root = tmp_path / "windows-source"
    value = snapshot(source_root)
    ledger = Path(value["inputs"]["ledger_path"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")
    root = tmp_path / "wsl-root"
    C.initialize(root, plan)
    materialized = C.materialize_slot(slot(plan, "model_adaptive:e1:open0"), value)
    serialized = json.dumps(materialized, ensure_ascii=False)
    assert str(ledger) not in serialized
    assert "ledger_path" not in serialized
    receipt = C.record_slot(root, plan, materialized)
    ledger.unlink()

    rebuilt, inputs = C._rebuild_snapshot(root, slot(plan, receipt["slot_id"]), receipt)
    assert C._digest(C._portable_snapshot(rebuilt)) == receipt["snapshot_digest"]
    assert inputs["messages_sha256"]


def test_materialization_closes_absent_terminal_and_no_model_slots(tmp_path: Path) -> None:
    plan = C.plan_value()
    assert C.materialize_slot(slot(plan, "model_adaptive:e2:open0"), None)["disposition"] == \
        "skipped_epoch_absent"

    first = snapshot(tmp_path / "first", "model_first_then_rule")
    observed = add_observation(first, "normal")
    assert observed["view"]["status"] == "open"
    assert C.materialize_slot(slot(plan, "model_first_then_rule:e1:first"), observed)["disposition"] == \
        "no_model_required"

    adaptive = snapshot(tmp_path / "exposure")
    adaptive = add_observation(adaptive, "normal")
    assert C.materialize_slot(slot(plan, "model_adaptive:e1:exposure_full"), adaptive)["disposition"] == \
        "executed"
    assert C.materialize_slot(slot(plan, "model_adaptive:e1:exposure_masked"), adaptive)["disposition"] == \
        "executed"

    terminal = snapshot(tmp_path / "terminal")
    terminal = add_observation(terminal, "abnormal")
    assert terminal["view"]["status"] != "open"
    assert C.materialize_slot(slot(plan, "model_adaptive:e1:open6"), terminal)["disposition"] == \
        "skipped_terminal"
    assert C.materialize_slot(slot(plan, "model_adaptive:e1:exposure_full"), terminal)["disposition"] == \
        "executed"

    open_value = snapshot(tmp_path / "hold")
    skipped = C.materialize_skipped_slot(slot(plan, "model_adaptive:e1:open1"), open_value,
                                         "model_hold", "a" * 64)
    assert skipped["disposition"] == "skipped_terminal"
    assert skipped["terminal_reason"] == "model_hold"
    assert skipped["terminal_receipt_sha256"] == "a" * 64
    assert skipped["capsule"]["input_digest"] is None

    # 첫 모델 판단이 HOLD면 관측이 없으므로 exposure query를 만들 수 없다. 남은
    # exposure slot도 실행 입력 없이 같은 terminal receipt에 묶어 닫는다.
    skipped_exposure = C.materialize_skipped_slot(
        slot(plan, "model_adaptive:e1:exposure_masked"), open_value, "model_hold", "a" * 64)
    assert skipped_exposure["disposition"] == "skipped_terminal"
    assert skipped_exposure["capsule"]["input_digest"] is None
    with pytest.raises(ValueError, match="terminal receipt SHA"):
        C.materialize_skipped_slot(slot(plan, "model_adaptive:e1:open1"), open_value,
                                   "model_hold", "bad")


def test_three_attempt_dialogue_replays_exactly_and_exception_keeps_class_only(tmp_path: Path) -> None:
    plan, root, value, receipt = setup_executed(tmp_path)
    provider = C.FixtureProvider([TimeoutError("secret detail"), response(value, valid=False), response(value)])
    record = C.run_dialogue(root, plan, receipt, provider)
    assert record["engine"]["requests"] == 3
    assert record["engine"]["selection"]["model_status"] == "valid"
    assert record["safe_results"][0]["error_type"] == "TimeoutError"
    assert "secret detail" not in json.dumps(record, ensure_ascii=False)
    audit = C.audit_api(root, plan)
    assert (audit["reserved"], audit["completed"], audit["pending"], audit["timeout"]) == (3, 3, 0, 1)
    assert C.review_dialogue(root, plan, receipt) == record
    for other in plan["slots"]:
        if other["slot_id"] != receipt["slot_id"]:
            C.record_slot(root, plan, C.materialize_slot(other, None))
    finalized = C.finalize(root, plan)
    assert finalized["status"] == "complete" and finalized["slots_closed"] == 20
    assert finalized["executed"] == 1 and finalized["api"]["reserved"] == 3


def test_invalid_model_output_holds_without_rule_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, root, value, receipt = setup_executed(tmp_path)
    monkeypatch.setattr(C.V, "select", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback")))
    monkeypatch.setattr(C.V, "rule_probe", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback")))
    record = C.run_dialogue(root, plan, receipt,
                            C.FixtureProvider([response(value, valid=False), response(value, valid=False)]))
    choice = record["engine"]["selection"]
    assert choice["action"] == "hold" and choice["probe"] is None and choice["model_status"] == "invalid"


def test_pending_is_unknown_consumed_blocks_new_request_and_invalidates_batch(tmp_path: Path) -> None:
    plan, root, _value, receipt = setup_executed(tmp_path)
    slot_value, stored = C._load_receipt(root, plan, receipt["slot_id"])
    _snapshot, inputs = C._rebuild_snapshot(root, slot_value, stored)
    C._reserve(root, plan, stored, inputs["messages"], 1)
    audit = C.audit_api(root, plan)
    assert audit["pending"] == audit["unknown_consumed"] == 1
    with pytest.raises(C.MC.BudgetStop, match="unknown"):
        C._reserve(root, plan, stored, inputs["messages"], 2)
    for other in plan["slots"]:
        if other["slot_id"] != receipt["slot_id"]:
            C.record_slot(root, plan, C.materialize_slot(other, None))
    result = C.finalize(root, plan)
    assert result["status"] == "invalid"
    assert result["costs"] == {"input_tokens": None, "output_tokens": None,
                                "total_tokens": None, "wall_s": None}


def test_duplicate_capsule_plan_and_dialogue_tampering_are_rejected(tmp_path: Path) -> None:
    plan, root, value, receipt = setup_executed(tmp_path)
    with pytest.raises(FileExistsError):
        C.record_slot(root, plan, C.materialize_slot(slot(plan, receipt["slot_id"]), value))

    capsule = root / "capsules" / f"{receipt['capsule_sha256']}.json"
    capsule.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="capsule"):
        C.run_dialogue(root, plan, receipt, C.FixtureProvider([response(value)]))


def test_finalize_requires_all_slots_and_allows_twenty_explicit_skips(tmp_path: Path) -> None:
    plan = C.plan_value()
    root = tmp_path / "all-skipped"
    C.initialize(root, plan)
    C.record_slot(root, plan, C.materialize_slot(plan["slots"][0], None))
    with pytest.raises(ValueError, match="20 slot"):
        C.finalize(root, plan)
    for row in plan["slots"][1:]:
        C.record_slot(root, plan, C.materialize_slot(row, None))
    result = C.finalize(root, plan)
    assert result["status"] == "complete" and result["slots_closed"] == 20
    assert result["executed"] == result["api"]["reserved"] == 0
    assert result["dispositions"]["skipped_epoch_absent"] == 20


def test_worker_binding_and_cli_plan_are_source_free(tmp_path: Path) -> None:
    plan = C.plan_value()
    assert C.__file__ is not None
    module_path = Path(C.__file__)
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"schema":"fixture"}\n', encoding="utf-8", newline="\n")
    C.verify_worker_binding(plan, plan["plan_sha256"], C.V.file_digest(module_path),
                            protocol, C.V.file_digest(protocol))
    with pytest.raises(ValueError, match="source SHA"):
        C.verify_worker_binding(plan, plan["plan_sha256"], "0" * 64,
                                protocol, C.V.file_digest(protocol))

    output = tmp_path / "plan.json"
    completed = subprocess.run(
        [sys.executable, str(module_path), "plan", "--output", str(output)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    assert completed.returncode == 0 and json.loads(output.read_text(encoding="utf-8")) == plan
    assert "C:\\Users" not in output.read_text(encoding="utf-8")
