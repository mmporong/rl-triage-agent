# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import runtime_observation_diagnosis_v1 as DIAG
import runtime_observation_recovery_v1 as C


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Ledger:
    instances: ClassVar[list[Ledger]] = []

    def __init__(self, path: Path, registry: dict):
        self.path, self.registry = path, registry
        self.events: list[tuple] = []
        self.__class__.instances.append(self)

    def propose(self, _state, component, _receipts, correction, _mode, _case, _budget, evidence=None):
        request = {"request_id": f"request-{len(self.events) + 1}"}
        self.events.append(("proposed", component, correction, evidence))
        return request

    def approve(self, request_id, approver):
        self.events.append(("approved", request_id, approver))

    def consume(self, request_id, _state):
        self.events.append(("consumed", request_id))

    def receipt(self, request_id, _tag, verdict, _sha, _wall, _exit):
        self.events.append(("receipt", request_id, verdict))


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline_valid: bool = True,
    source_failure: bool = True,
    diagnosis_states: list[str] | None = None,
    repair_outcomes: list[str] | None = None,
    snapshot_tamper: bool = False,
) -> SimpleNamespace:
    calls = SimpleNamespace(train=[], fixed=[], diagnosis=[], close=[], repair_verify=[], contexts=0)
    protocol = {"components": {"probe": {"path": "probe.json"}}}
    monkeypatch.setattr(C, "verify_protocol", lambda *_args: protocol)
    monkeypatch.setattr(C.SEEDS, "assert_fresh_seeds", lambda seeds, lab: {"seeds": seeds})
    monkeypatch.setattr(C.FIRST.LEASE, "lease_context", lambda _path: {"owner": C.OWNER})
    monkeypatch.setattr(C, "MAX_TOTAL_S", 10**9)
    monkeypatch.setattr(C.C.D1.V.R1.U, "walks", lambda _report: baseline_valid)
    monkeypatch.setattr(C.C.V, "checked", lambda _path: {"snapshot": True})
    monkeypatch.setattr(C.C.L, "digest", lambda value: "digest-" + str(bool(value)))
    monkeypatch.setattr(C.C.RC, "correction_registry", lambda: {
        "reward": {"correction_id": "restore_body_frame_reward", "protocol_case": "case_01",
                   "recovery_modes": ["restart"]},
        "optimizer_update_count": {"correction_id": "restore_configured_ppo_epochs",
                                   "protocol_case": "optimizer_epoch_count",
                                   "recovery_modes": ["restart"]},
    })
    Ledger.instances = []
    monkeypatch.setattr(C, "CorrectionLedger", Ledger)
    monkeypatch.setattr(C, "_training_checked", lambda training: ({}, {}))

    @contextmanager
    def view_scope(*_args):
        yield

    monkeypatch.setattr(C.RV, "view_scope", view_scope)
    monkeypatch.setattr(C.T, "close", lambda queue, reason: calls.close.append((queue, reason)))
    monkeypatch.setattr(C.TRAIN, "build_manifest", lambda *_args, **kwargs: {
        "label": Path(kwargs["stage"]).parent.name, "mode": kwargs["mode"],
        "source_checkpoint_sha256": kwargs["source_checkpoint_sha256"]})

    def train(stage, fixed, **_kwargs):
        label = Path(stage).parent.name
        calls.train.append((label, fixed["source_checkpoint_sha256"]))
        cp_sha = (label.encode().hex() + "0" * 64)[:64]
        evidence_path = Path(stage) / "training/training_evidence.json"
        _json(evidence_path, {"checkpoint": {"path": str(Path(stage) / "model.pt"), "sha256": cp_sha}})
        for name in ("manifest.json", "owned-result.json", "collected.json", "case-facts.json",
                     "reference-facts.json"):
            _json(Path(stage) / name, {"label": label, "name": name})
        return {
            "manifest": C.D.ref(Path(stage) / "manifest.json"),
            "evidence": C.D.ref(evidence_path),
            "owned_result": C.D.ref(Path(stage) / "owned-result.json"),
            "checkpoint": {"path": str(Path(stage) / "model.pt"), "sha256": cp_sha},
            "collected": C.D.ref(Path(stage) / "collected.json") if fixed["mode"] == "train" else None,
            "runtime_facts": {
                "case": C.D.ref(Path(stage) / "case-facts.json"),
                "reference": C.D.ref(Path(stage) / "reference-facts.json"),
            } if fixed["mode"] == "train" else None,
        }

    monkeypatch.setattr(C.TRAIN, "run_training", train)

    def fixed(stage, evidence, *, name, **_kwargs):
        label = Path(stage).parent.name
        calls.fixed.append(label)
        report = {"label": label, "name": name, "checkpoint": evidence["checkpoint"]}
        _json(Path(stage) / "corrected.json", report)
        return report, {"exit_code": 0}

    monkeypatch.setattr(C.TRAIN, "run_fixed", fixed)

    repair_values = iter(repair_outcomes or [])

    def recovery(report, _reference, _bands):
        label = report["label"]
        if label == "none":
            return {"outcome": "recovered", "verdict": {"label": "healthy"}}
        if label == "compound":
            state = "unhealthy" if source_failure else "healthy"
            return {"outcome": "not_recovered" if source_failure else "recovered",
                    "verdict": {"label": state}}
        outcome = next(repair_values)
        return {"outcome": outcome,
                "verdict": {"label": "healthy" if outcome == "recovered" else "unhealthy"}}

    monkeypatch.setattr(C.C.D1, "recovery", recovery)
    states = iter(diagnosis_states or [])

    def diagnose(out, *_args, **kwargs):
        state = next(states)
        calls.diagnosis.append({"state": state, "applied": kwargs["applied_corrections"],
                                "prior": kwargs["prior_receipt"], "parent": kwargs["parent"]})
        _json(Path(out) / "result.json", {"state": state})
        snapshot_path = Path(out) / "snapshot.json"
        _json(snapshot_path, {"snapshot": True})
        snapshot_ref = C.D.ref(snapshot_path)
        if snapshot_tamper:
            _json(snapshot_path, {"snapshot": "tampered"})
        result = {"end_state": state,
                  "hold_reason": "model_invalid" if state != "confirmed" else None,
                  "final_snapshot": snapshot_ref, "steps": []}
        _json(Path(out) / "result.json", result)
        return result

    monkeypatch.setattr(DIAG, "run_epoch", diagnose)
    components = iter([
        ("reward", "restore_body_frame_reward"),
        ("optimizer_update_count", "restore_configured_ppo_epochs"),
    ])
    monkeypatch.setattr(C, "verify_diagnosis", lambda result, applied: next(components))

    def verify_repair(path, *_args):
        calls.repair_verify.append(Path(path))
        return C.D.read(path)

    monkeypatch.setattr(C, "verify_repair", verify_repair)
    bands = tmp_path / "bands.json"
    _json(bands, {"bands": {}})
    monkeypatch.setattr(C.OLD, "BANDS", bands)
    return calls


def _run(tmp_path: Path) -> dict:
    return C.run(
        tmp_path / "protocol.json", "a" * 40, tmp_path / "queue", tmp_path / "out",
        lab=tmp_path / "lab", isaac_python=Path(sys.executable),
        smoke_checkpoint_sha256="c" * 64)


def test_invalid_none_stops_before_compound_model_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(tmp_path, monkeypatch, baseline_valid=False)
    result = _run(tmp_path)
    assert result["reason"] == "baseline_invalid"
    assert [row[0] for row in calls.train] == ["smoke", "none"]
    assert calls.fixed == ["none"] and calls.diagnosis == [] and Ledger.instances == []
    assert len(calls.close) == 1


def test_source_failure_absent_stops_before_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(tmp_path, monkeypatch, source_failure=False)
    result = _run(tmp_path)
    assert result["reason"] == "source_failure_absent_or_unknown"
    assert [row[0] for row in calls.train] == ["smoke", "none", "compound"]
    assert calls.diagnosis == [] and Ledger.instances == []


@pytest.mark.parametrize("state", ["HOLD", "unknown", "model_invalid"])
def test_unconfirmed_diagnosis_never_approves_or_trains_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str,
) -> None:
    calls = _install(tmp_path, monkeypatch, diagnosis_states=[state])
    result = _run(tmp_path)
    assert result["reason"] == "model_invalid"
    assert [row[0] for row in calls.train] == ["smoke", "none", "compound"]
    assert len(calls.diagnosis) == 1 and Ledger.instances == []


def test_first_recovered_stops_without_epoch2_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(
        tmp_path, monkeypatch, diagnosis_states=["confirmed"], repair_outcomes=["recovered"])
    result = _run(tmp_path)
    assert result["outcome"] == "recovered"
    assert [row[0] for row in calls.train] == ["smoke", "none", "compound", "repair1"]
    assert len(calls.diagnosis) == 1 and len(calls.repair_verify) == 1
    events = Ledger.instances[0].events
    assert [row[0] for row in events] == ["proposed", "approved", "consumed", "receipt"]


def test_valid_not_recovered_parent_allows_distinct_second_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(tmp_path, monkeypatch, diagnosis_states=["confirmed", "confirmed"],
                     repair_outcomes=["not_recovered", "recovered"])
    result = _run(tmp_path)
    assert result["outcome"] == "recovered"
    assert [row[0] for row in calls.train] == [
        "smoke", "none", "compound", "repair1", "repair2"]
    assert calls.diagnosis[0]["prior"] is None and calls.diagnosis[0]["parent"] is None
    assert calls.diagnosis[1]["prior"]["verdict"] == "not_recovered"
    assert calls.diagnosis[1]["parent"]["epoch"] == 1
    proposed = [row for ledger in Ledger.instances for row in ledger.events if row[0] == "proposed"]
    assert [(row[1], row[2]) for row in proposed] == [
        ("reward", "restore_body_frame_reward"),
        ("optimizer_update_count", "restore_configured_ppo_epochs"),
    ]
    assert len(calls.repair_verify) >= 2


def test_failure_is_preserved_and_transport_closes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(tmp_path, monkeypatch)
    original = C.TRAIN.run_training
    count = 0

    def fail_after_smoke(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("source training stopped")
        return original(*args, **kwargs)

    monkeypatch.setattr(C.TRAIN, "run_training", fail_after_smoke)
    with pytest.raises(RuntimeError, match="source training stopped"):
        _run(tmp_path)
    failure = C.D.read(tmp_path / "out/failure.json")
    assert failure["status"] == "invalid" and failure["error_type"] == "RuntimeError"
    assert len(calls.close) == 1


def test_swapped_final_snapshot_stops_before_repair_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(
        tmp_path, monkeypatch, diagnosis_states=["confirmed"],
        repair_outcomes=["recovered"], snapshot_tamper=True)
    with pytest.raises(ValueError, match="final snapshot 원문 SHA"):
        _run(tmp_path)
    assert [row[0] for row in calls.train] == ["smoke", "none", "compound"]
    assert Ledger.instances == [] and calls.repair_verify == []
    assert len(calls.close) == 1
