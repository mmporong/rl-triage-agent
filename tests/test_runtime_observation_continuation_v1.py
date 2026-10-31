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

import runtime_observation_continuation_v1 as C


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@contextmanager
def _scope(*_args, **_kwargs):
    yield {}


def test_prepare_protocol_preserves_origin_contract_and_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = {
        "schema": C.R.SCHEMA, "seed": C.R.SEED, "model": "registered-model",
        "params": {"temperature": 0}, "conditions": ["NONE", "COMPOUND", "FIRST", "BOTH"],
        "training": {"updates": 300}, "predictions": {"two": "registered"},
        "criteria": {"epoch2": "valid receipt only"},
        "wording": {"recovered": "r", "not_recovered": "n", "HOLD": "h"},
        "runtime_source_sha256_lf": {f"runtime-{index}": "a" * 64 for index in range(9)},
    }
    transport = {"schema": C.TRANSPORT.SCHEMA, "model": origin["model"],
                 "params": origin["params"], "plan": {
                     "case_ids": ["continuation-s2311"], "global_request_cap": 9,
                     "per_slot_max": 3, "logical_slots": [
                         {"case_id": "continuation-s2311", "slot_id": f"epoch2.decision{index}"}
                         for index in (1, 2, 3)]}}
    origin_path, probe_path, transport_path = (tmp_path / name for name in
        ("origin.json", "probe.json", "transport.json"))
    _write(origin_path, origin)
    _write(probe_path, {})
    _write(transport_path, transport)
    view_path, fixed_path = tmp_path / "view.json", tmp_path / "fixed.json"
    _write(view_path, {})
    _write(fixed_path, {})
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "SOURCE_FILES", ())
    monkeypatch.setattr(C.RV, "PROTOCOL", view_path)
    monkeypatch.setattr(C.R.OLD, "FIXED_PROTOCOL", fixed_path)
    value = C.prepare_protocol(origin_path, probe_path, transport_path)
    assert value["inherited"] == {
        "conditions": origin["conditions"], "training": origin["training"],
        "predictions": origin["predictions"], "criteria": origin["criteria"]}
    assert value["bounds"]["source_training"] == 0
    assert value["bounds"]["first_repair_training"] == 0
    assert value["bounds"]["second_repair_training_max"] == 1
    assert value["bounds"]["epoch2_model_requests_max"] == 9
    assert value["bounds"]["combined_provider_requests_max"] == 24


def test_closed_origin_queue_rejects_any_epoch2_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue"
    _write(queue / "requests/one.json", {"slot_id": "epoch2.decision1"})
    monkeypatch.setattr(C.TRANSPORT, "status", lambda _root: {
        "closed": {"schema": "lineage_nim_transport_closed_v1"},
        "requests": 1, "dispatches": 1, "responses": 1,
        "pending": 0, "unknown": 0, "failures": 0})
    with pytest.raises(ValueError, match="epoch1 밖"):
        C._closed_origin_queue(queue, 1)


class FakeLedger:
    instances: ClassVar[list[FakeLedger]] = []

    def __init__(self, path: Path, _registry: dict):
        self.path = Path(path)
        self.rows: list[tuple] = []
        self.__class__.instances.append(self)

    def propose(self, _state, component, _receipts, correction, _mode, _case, _budget,
                evidence=None):
        self.rows.append(("proposed", component, correction, evidence))
        return {"request_id": "request-2"}

    def approve(self, request_id, approver):
        self.rows.append(("approved", request_id, approver))

    def consume(self, request_id, _state):
        self.rows.append(("consumed", request_id))

    def receipt(self, request_id, _tag, verdict, _sha, _wall, _exit):
        self.rows.append(("receipt", request_id, verdict))


def _installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, state: str,
               subtype: str = "optimizer_update_count") -> SimpleNamespace:
    calls = SimpleNamespace(train=[], fixed=[], close=[], diagnosis=[])
    ledger = tmp_path / "origin-ledger.jsonl"
    ledger.write_text("origin-ledger\n", encoding="utf-8")
    first_collected, first_case, first_ref = (tmp_path / name for name in
        ("first-collected.json", "first-case.json", "first-ref.json"))
    reference_collected = tmp_path / "reference-collected.json"
    for path in (first_collected, first_case, first_ref, reference_collected):
        _write(path, {"name": path.name})
    first = {"checkpoint": {"path": str(tmp_path / "first.pt"), "sha256": "1" * 64},
             "collected": {"path": str(first_collected)},
             "runtime_facts": {"case": {"path": str(first_case)},
                               "reference": {"path": str(first_ref)}}}
    reference = {"checkpoint": {"path": str(tmp_path / "none.pt"), "sha256": "0" * 64},
                 "collected": {"path": str(reference_collected)},
                 "runtime_facts": {"reference": {"path": str(first_ref)}}}
    parent = tmp_path / "origin-receipt.json"
    reference_report = tmp_path / "none.json"
    _write(reference_report, {"name": "none"})
    _write(parent, {"reference_report": C.ref(reference_report)})
    inputs_path = tmp_path / "inputs.json"
    protocol_path = tmp_path / "protocol.json"
    inputs = {
        "continuation_protocol": {"path": str(protocol_path.resolve())},
        "origin_anchor": "a" * 40,
        "origin": {"ledger": C.ref(ledger), "first": C.ref(_json_file(tmp_path / "first.json", first)),
                   "reference": C.ref(_json_file(tmp_path / "reference.json", reference)),
                   "receipt": C.ref(parent),
                   "prior": {"verdict": "not_recovered",
                             "correction_id": "restore_body_frame_reward",
                             "result_sha256": C.sha(parent),
                             "checkpoint_sha256": "1" * 64}},
        "budget": {"origin_wall_s": 10.0, "prepare_gap_s": 1.0,
                   "prepared_time_ns": C.time.time_ns(),
                   "combined_wall_s_max": C.R.MAX_TOTAL_S},
    }
    _write(inputs_path, inputs)
    protocol = {"purpose": "continue_verified_first_not_recovered_epoch_only",
                "components": {"probe": {"path": "probe.json"}}}
    monkeypatch.setattr(C, "verify_origin", lambda _path: inputs)
    monkeypatch.setattr(C, "verify_protocol", lambda *_args: protocol)
    monkeypatch.setattr(C.CACHE, "proof_scope", _scope)
    monkeypatch.setattr(C.RV, "view_scope", _scope)
    monkeypatch.setattr(C.R, "verify_repair", lambda *_args: {})
    monkeypatch.setattr(C.R, "verify_diagnosis",
                        lambda _result, _applied: ("optimizer_update_count",
                                                  "restore_configured_ppo_epochs"))
    monkeypatch.setattr(C.R, "load_verified_snapshot", lambda _result: {"snapshot": True})
    monkeypatch.setattr(C.R.C.L, "digest", lambda _value: "digest")
    monkeypatch.setattr(C.R.C.RC, "correction_registry", lambda: {
        "optimizer_update_count": {"correction_id": "restore_configured_ppo_epochs",
                                   "protocol_case": "optimizer_epoch_count",
                                   "recovery_modes": ["restart"]}})
    monkeypatch.setattr(C, "CorrectionLedger", FakeLedger)
    FakeLedger.instances = []
    bands = tmp_path / "bands.json"
    _write(bands, {"bands": {}})
    monkeypatch.setattr(C.R.OLD, "BANDS", bands)
    monkeypatch.setattr(C.R.FIRST.LEASE, "lease_context", lambda _path: {"owner": C.R.OWNER})
    monkeypatch.setattr(C.TRANSPORT, "close",
                        lambda queue, reason: calls.close.append((queue, reason)))
    monkeypatch.setattr(C.TRANSPORT, "status", lambda _queue: {
        "closed": None, "requests": 0, "dispatches": 0, "responses": 0,
        "pending": 0, "unknown": 0, "failures": 0})

    def diagnose(out: Path, *_args, **kwargs):
        kwargs["verify_context"]()
        calls.diagnosis.append(kwargs)
        result = {"end_state": state, "hold_reason": "model_invalid" if state != "confirmed" else None,
                  "confirmed_subtype": subtype if state == "confirmed" else None,
                  "steps": [], "final_snapshot": {"path": str(out / "snapshot.json"),
                                                     "sha256": "2" * 64}}
        _write(out / "result.json", result)
        return result

    monkeypatch.setattr(C.TRAIN, "build_manifest", lambda *_args, **kwargs: kwargs)

    def train(stage: Path, _fixed: dict, **_kwargs):
        calls.train.append(Path(stage))
        evidence = Path(stage) / "training/training_evidence.json"
        _write(evidence, {"checkpoint": {"path": str(Path(stage) / "model.pt"),
                                         "sha256": "3" * 64}})
        output = {"checkpoint": {"path": str(Path(stage) / "model.pt"), "sha256": "3" * 64},
                  "evidence": C.ref(evidence)}
        return output

    monkeypatch.setattr(C.TRAIN, "run_training", train)

    def fixed(stage: Path, _evidence: dict, *, name: str, **_kwargs):
        calls.fixed.append(Path(stage))
        report = {"name": name}
        _write(Path(stage) / "corrected.json", report)
        return report, {"exit_code": 0}

    monkeypatch.setattr(C.TRAIN, "run_fixed", fixed)
    monkeypatch.setattr(C.R.C.D1, "recovery",
                        lambda *_args: {"outcome": "recovered", "verdict": {"label": "healthy"}})
    return SimpleNamespace(calls=calls, inputs=inputs_path, protocol=protocol_path, diagnose=diagnose)


def _json_file(path: Path, value: object) -> Path:
    _write(path, value)
    return path


def test_hold_does_not_train_or_approve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _installed(tmp_path, monkeypatch, state="HOLD")
    result = C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                       tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                       runner=lambda *_args: None, wait=lambda *_args: None, diagnose=setup.diagnose)
    assert result["outcome"] == "HOLD" and result["reason"] == "model_invalid"
    assert setup.calls.train == [] and FakeLedger.instances == []
    assert len(setup.calls.close) == 1


def test_confirmed_runs_one_distinct_second_repair(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed")
    result = C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                       tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                       runner=lambda *_args: None, wait=lambda *_args: None, diagnose=setup.diagnose)
    assert result["outcome"] == "recovered" and result["second_repair_training"] == 1
    assert len(setup.calls.train) == len(setup.calls.fixed) == 1
    assert setup.calls.diagnosis[0]["applied_corrections"] == ("restore_body_frame_reward",)
    assert setup.calls.diagnosis[0]["parent"]["epoch"] == 1
    assert [row[0] for row in FakeLedger.instances[0].rows] == [
        "proposed", "approved", "consumed", "receipt"]
    assert FakeLedger.instances[0].rows[0][2] == "restore_configured_ppo_epochs"


def test_repeated_second_correction_stops_before_approval_and_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed", subtype="reward")
    registry = C.R.C.RC.correction_registry()
    registry["reward"] = {"correction_id": "restore_body_frame_reward",
                          "protocol_case": "case_01", "recovery_modes": ["restart"]}
    monkeypatch.setattr(C.R.C.RC, "correction_registry", lambda: registry)
    result = C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                       tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                       runner=lambda *_args: None, wait=lambda *_args: None,
                       diagnose=setup.diagnose)
    assert result["reason"] == "repeated_correction"
    assert setup.calls.train == [] and FakeLedger.instances == []
    assert (tmp_path / "out/result.json").is_file() and len(setup.calls.close) == 1


def test_cumulative_wall_budget_stops_before_diagnosis_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed")
    setup_inputs = C.read(setup.inputs)
    setup_inputs["budget"]["combined_wall_s_max"] = 0
    _write(setup.inputs, setup_inputs)
    monkeypatch.setattr(C, "verify_origin", lambda _path: setup_inputs)
    with pytest.raises(ValueError, match="prepare 이후"):
        C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                  tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                  runner=lambda *_args: None, wait=lambda *_args: None, diagnose=setup.diagnose)
    assert setup.calls.diagnosis == [] and setup.calls.train == []
    assert len(setup.calls.close) == 1


def test_all_confirmed_phases_remain_inside_cache_and_view_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed")
    active = {"cache": 0, "view": 0}

    @contextmanager
    def cache_scope(*_args, **_kwargs):
        active["cache"] += 1
        try:
            yield {}
        finally:
            active["cache"] -= 1

    @contextmanager
    def view_scope(*_args, **_kwargs):
        active["view"] += 1
        try:
            yield {}
        finally:
            active["view"] -= 1

    monkeypatch.setattr(C.CACHE, "proof_scope", cache_scope)
    monkeypatch.setattr(C.RV, "view_scope", view_scope)

    def checked_snapshot(_result):
        assert active["cache"] > 0 and active["view"] > 0
        return {"snapshot": True}

    def checked_repair(*_args):
        assert active["cache"] > 0 and active["view"] > 0
        return {}

    monkeypatch.setattr(C.R, "load_verified_snapshot", checked_snapshot)
    monkeypatch.setattr(C.R, "verify_repair", checked_repair)
    result = C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                       tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                       runner=lambda *_args: None, wait=lambda *_args: None,
                       diagnose=setup.diagnose)
    assert result["outcome"] == "recovered" and active == {"cache": 0, "view": 0}


def test_invalid_lease_stops_before_model_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed")
    monkeypatch.setattr(C.R.FIRST.LEASE, "lease_context", lambda _path: {"owner": "other"})
    with pytest.raises(ValueError, match="lease 소유자"):
        C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                  tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                  runner=lambda *_args: None, wait=lambda *_args: None,
                  diagnose=setup.diagnose)
    assert setup.calls.diagnosis == [] and setup.calls.train == []


def test_origin_queue_cap_and_training_metadata_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue"
    _write(queue / "requests/one.json", {"slot_id": "epoch1.decision1"})
    monkeypatch.setattr(C.TRANSPORT, "status", lambda _root: {
        "closed": {"schema": "lineage_nim_transport_closed_v1"},
        "requests": 10, "dispatches": 10, "responses": 10,
        "pending": 0, "unknown": 0, "failures": 0})
    with pytest.raises(ValueError, match="완료된 epoch1"):
        C._closed_origin_queue(queue, 10)
    monkeypatch.setattr(C.R, "_training_checked", lambda _training: ({}, {
        "private_training": {"seed": C.R.SEED, "arm": "COMPOUND",
                             "original_faults": [], "applied_corrections": []}}))
    with pytest.raises(ValueError, match="origin NONE"):
        C._check_origin_training({}, arm="NONE", faults=[], corrections=[])


def test_confirmed_unsupported_subtype_holds_without_raw_repair_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed", subtype="unsupported_physics")

    def unexpected(*_args):
        raise AssertionError("unsupported subtype는 registered raw repair 검증으로 넘기지 않는다")

    monkeypatch.setattr(C.R, "verify_diagnosis", unexpected)
    result = C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                       tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                       runner=lambda *_args: None, wait=lambda *_args: None,
                       diagnose=setup.diagnose)
    assert result["reason"] == "repair_unsupported"
    assert setup.calls.train == [] and FakeLedger.instances == []


def test_supported_subtype_raw_tamper_is_not_hidden_as_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _installed(tmp_path, monkeypatch, state="confirmed")
    monkeypatch.setattr(C.R, "verify_diagnosis",
                        lambda *_args: (_ for _ in ()).throw(ValueError("raw receipt tamper")))
    with pytest.raises(ValueError, match="raw receipt tamper"):
        C.execute(setup.inputs, setup.protocol, "b" * 40, tmp_path / "queue",
                  tmp_path / "out", lab=tmp_path / "lab", isaac_python=Path(sys.executable),
                  runner=lambda *_args: None, wait=lambda *_args: None,
                  diagnose=setup.diagnose)
    assert setup.calls.train == [] and FakeLedger.instances == []
    assert C.read(tmp_path / "out/failure.json")["status"] == "invalid"
