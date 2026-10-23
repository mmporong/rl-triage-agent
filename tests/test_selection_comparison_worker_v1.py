"""D4 worker의 원래 snapshot 재검산·터미널 slot·소유 종료 경계."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src", "tests")]
import selection_comparison_worker_v1 as W  # noqa: E402
import test_selection_comparison_controller_v1 as FIX  # noqa: E402


class Budget:
    def remaining(self):
        return {"overall_elapsed_s": 123.0}


def fixture_launcher(root, plan, receipt, protocol, anchor, timeout_s):
    assert timeout_s == 123.0
    snapshot, inputs = W.API._rebuild_snapshot(root, W.API._slot(plan, receipt["slot_id"]), receipt)
    provider = W.API.FixtureProvider([FIX.response(inputs["view"], hold=True)])
    record = W.API.run_dialogue(root, plan, receipt, provider)
    W.DEMO.write_new(W._paths(root, receipt)["transport"], {
        "slot_id": receipt["slot_id"], "status": "complete", "cleanup": "natural_exit_pid_released"})
    return record


def test_model_hold_closes_entered_epoch_as_terminal_and_future_epoch_as_absent(tmp_path):
    root = tmp_path / "api"
    caller = W.ModelCaller(root, W.DEMO.fixed_protocol(), "a" * 40, Budget(), launcher=fixture_launcher)
    snapshot = FIX.snapshot(tmp_path / "case", "model_adaptive", "model_adaptive")
    dialogue = caller(snapshot, "production", "model_adaptive", 1, 0)
    assert W.CONTROL.choose(snapshot, "model_adaptive", dialogue)["action"] == "hold"
    caller.finish_epoch("model_adaptive", 1, snapshot, "model_hold")
    summary = caller.finalize(status="complete", arms={"model_adaptive": {"outcome": "HOLD"}})
    assert summary["slots_closed"] == 20 and summary["api"]["reserved"] == 1
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "slots").glob("*.json")]
    assert sum(row["terminal_reason"] == "model_hold" for row in receipts) == 8
    assert sum(row["disposition"] == "skipped_epoch_absent" for row in receipts) == 11
    with pytest.raises(ValueError, match="중복"):
        caller.finalize(status="complete", arms={})


def test_worker_record_must_match_original_snapshot_and_has_no_rule_fallback(tmp_path, monkeypatch):
    def bad_launcher(*args):
        record = fixture_launcher(*args)
        record["engine"]["selection"]["action"] = "continue"
        return record

    caller = W.ModelCaller(tmp_path / "api", W.DEMO.fixed_protocol(), "a" * 40, Budget(), launcher=bad_launcher)
    snapshot = FIX.snapshot(tmp_path / "case", "model_adaptive", "model_adaptive")
    monkeypatch.setattr(W.CONTROL.V, "rule_probe", lambda *args: pytest.fail("fallback"))
    with pytest.raises(ValueError, match="반환과 저장"):
        caller(snapshot, "production", "model_adaptive", 1, 0)


def test_absent_source_closes_all_slots_without_requests(tmp_path):
    caller = W.ModelCaller(tmp_path / "api", W.DEMO.fixed_protocol(), "a" * 40, Budget(),
                           launcher=lambda *args: pytest.fail("model not requested"))
    value = caller.finalize(status="source_failure_absent", arms={})
    assert value["slots_closed"] == 20 and value["api"]["reserved"] == 0
    assert value["dispositions"]["skipped_epoch_absent"] == 20


def test_changed_terminal_receipt_cannot_close_remaining_slots(tmp_path):
    root = tmp_path / "api"
    caller = W.ModelCaller(root, W.DEMO.fixed_protocol(), "a" * 40, Budget(), launcher=fixture_launcher)
    snapshot = FIX.snapshot(tmp_path / "case", "model_adaptive", "model_adaptive")
    caller.finish_epoch("model_adaptive", 1, snapshot, "model_hold")
    path = root / "terminals/model_adaptive_e1.json"
    changed = W.DEMO.read(path)
    changed["reason"] = "batch_invalid"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="종료 증거 변경"):
        caller.finalize(status="complete", arms={})
    assert not list((root / "slots").glob("*.json"))


def test_transport_timeout_preserves_failure_and_invokes_owned_cleanup(tmp_path, monkeypatch):
    snapshot = FIX.snapshot(tmp_path / "case", "model_adaptive", "model_adaptive")
    root, plan = tmp_path / "api", W.API.plan_value()
    W.API.initialize(root, plan)
    slot = next(row for row in plan["slots"] if row["slot_id"] == "model_adaptive:e1:open0")
    receipt = W.API.record_slot(root, plan, W.API.materialize_slot(slot, snapshot))
    protocol = W.DEMO.fixed_protocol()
    monkeypatch.setattr(W.DEMO, "verify_repository", lambda *args: None)
    monkeypatch.setattr(W.DEMO, "sha", lambda *args: "a" * 64)
    cleaned = []
    paths = W._paths(root, receipt)

    def timeout(*args, **kwargs):
        W.DEMO.write_new(paths["pid"], {"pid": 123, "start_ticks": 456})
        raise subprocess.TimeoutExpired("fixture", 1)

    monkeypatch.setattr(W.subprocess, "run", timeout)
    monkeypatch.setattr(W, "cleanup_worker", lambda *args: cleaned.append(args) or "verified")
    with pytest.raises(subprocess.TimeoutExpired):
        W.launch(root, plan, receipt, protocol, "a" * 40, 1.0)
    assert cleaned == [(paths["pid"], paths["capsule"])]
    assert W.DEMO.read(paths["transport"])["cleanup"] == "verified"


def test_worker_requires_matching_root_before_provider_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(W.sys, "platform", "linux")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(W.DEMO, "read", lambda *args: {})
    monkeypatch.setattr(W.DEMO, "verify_repository", lambda *args: None)
    monkeypatch.setattr(W.API, "NvidiaProvider", lambda: pytest.fail("provider created"))
    with pytest.raises(ValueError, match="root"):
        W.worker(SimpleNamespace(root=tmp_path, approve_model_call=True, anchor="a" * 40,
                                 protocol_sha="b" * 64, root_digest="c" * 64))
