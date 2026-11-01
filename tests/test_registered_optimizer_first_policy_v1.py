# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import registered_optimizer_first_policy_v1 as P

C = P.C
VIEW_SHA = C.V.file_digest(C.V.PROTOCOL)
BINDINGS = {
    "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
    "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
}


def _snapshot(root: Path, *, budget: int = 7) -> dict:
    packet = C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings=deepcopy(BINDINGS),
        budget_left=budget, execution_budget_s=1200)
    return C.V.build_snapshot(packet, "optimizer_first_test", "registry_next", VIEW_SHA,
                              root / "probes.ledger.jsonl")


def _receipt(root: Path, snapshot: dict, outcome: str) -> None:
    choice = P.choose(snapshot)
    spec = P.request_spec(snapshot, choice)
    ledger = C.S.M.RegistryLedger(root / "probes.ledger.jsonl", C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], "cpu-test")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    ledger.receipt(request["request_id"], outcome, {}, 1.0, 0)


def test_explicit_optimizer_then_reward_then_registry_order(tmp_path: Path) -> None:
    first = _snapshot(tmp_path)
    assert P.choose(first)["probe"] == "P_optimizer_updates"
    _receipt(tmp_path, first, "normal")
    second = _snapshot(tmp_path)
    assert P.choose(second)["probe"] == "P_reward"
    _receipt(tmp_path, second, "normal")
    third = _snapshot(tmp_path)
    selected = P.choose(third)
    assert selected["probe"] == C.V.rule_probe(third["view"], "registry_next")
    assert selected["probe"] not in {"P_optimizer_updates", "P_reward"}


def test_unknown_and_abnormal_stop_without_fallback(tmp_path: Path) -> None:
    for outcome in ("unknown", "abnormal"):
        root = tmp_path / outcome
        current = _snapshot(root)
        _receipt(root, current, outcome)
        stopped = P.choose(_snapshot(root))
        assert stopped["action"] == "hold" and stopped["probe"] is None


def test_duplicate_probe_and_exhausted_budget_are_not_requested(tmp_path: Path) -> None:
    current = _snapshot(tmp_path)
    _receipt(tmp_path, current, "normal")
    after = _snapshot(tmp_path)
    assert P.choose(after)["probe"] == "P_reward"
    assert P.choose(after)["probe"] not in {
        row["probe"] for row in after["view"]["observations"]}

    exhausted = _snapshot(tmp_path / "empty", budget=0)
    assert P.choose(exhausted)["action"] == "hold"


def test_confirmed_state_holds_and_request_spec_is_host_generated(tmp_path: Path) -> None:
    current = _snapshot(tmp_path)
    choice = P.choose(current)
    spec = P.request_spec(current, choice)
    assert spec["probe"] == "P_optimizer_updates"
    assert spec["args"] == C.RC.registry()["probes"]["P_optimizer_updates"]["args"]
    assert spec["approval_granted"] is False
    assert P.LEGACY_RECORDED_PROBE not in current["view"]["eligible_probes"]

    _receipt(tmp_path, current, "abnormal")
    confirmed = _snapshot(tmp_path)
    assert confirmed["view"]["status"] == "confirmed"
    assert P.choose(confirmed)["action"] == "hold"
    with pytest.raises(ValueError, match="현재 optimizer-first"):
        P.request_spec(confirmed, choice)


def test_tampered_choice_and_ineligible_optimizer_fail_closed(tmp_path: Path) -> None:
    current = _snapshot(tmp_path)
    tampered = P.choose(current)
    tampered["probe"] = "P_reward"
    with pytest.raises(ValueError, match="현재 optimizer-first"):
        P.request_spec(current, tampered)

    snapshot = deepcopy(current)
    snapshot["view"]["eligible_probes"].remove("P_optimizer_updates")
    snapshot["view_digest"] = C.L.digest(snapshot["view"])
    with pytest.raises(ValueError, match="오래된 snapshot"):
        P.choose(snapshot)
