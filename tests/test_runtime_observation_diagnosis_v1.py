from __future__ import annotations

# pyright: reportMissingImports=false
import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import runtime_observation_diagnosis_v1 as D

RUNTIME_FACT = "runtime:case:optimizer_steps_min"


def packet(epoch: int = 1) -> dict:
    prior = None if epoch == 1 else {
        "verdict": "not_recovered", "correction_id": D.C.RC.REWARD,
        "result_sha256": "e" * 64, "checkpoint_sha256": "a" * 64}
    facts = {"telemetry:case:Train/mean_reward": 1.0,
             "telemetry:reference:Train/mean_reward": 2.0,
             **{f"runtime:{role}:{name}": (
                 True if name == "available" else 3 if name == "updates_observed" else 15)
                for role in ("case", "reference") for name in D.RV.FACT.FACT_NAMES}}
    return D.C.S.build_packet(
        epoch=epoch, applied_corrections=[] if epoch == 1 else [D.C.RC.REWARD],
        prior_receipt=prior, facts=facts,
        input_bindings={
            "case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "b" * 64},
            "reference": {"checkpoint_sha256": "c" * 64, "telemetry_sha256": "d" * 64}},
        budget_left=3, execution_budget_s=1200)


def envelope(value: dict) -> dict:
    return {"schema": D.INPUT.PACKET_SCHEMA, "packet": deepcopy(value),
            "private_metadata": {"fixture": True},
            "scope": {"activity": "cpu_packet_materialization_only",
                      "approval": False, "execution_allowed": False}}


def measurement(probe: str, abnormal: bool) -> dict:
    if probe == "P_value":
        return {"critic_change": 0.01 if abnormal else 1.0, "actor_change": 0.5}
    if probe == "P_episode":
        return {"timeout_ratio": 0.1 if abnormal else 1.0}
    if probe == "P_physics":
        return {"mismatch_count": 1 if abnormal else 0}
    if probe == "P_noise":
        return {"noise_ratio": 4.0 if abnormal else 1.0}
    if probe == "P_reward":
        value = 1.0 if abnormal else 0.0
        return {"track_lin_vel_xy_exp_rel_error": value,
                "track_ang_vel_z_exp_rel_error": value}
    if probe == "P_torque":
        return {"low_speed_saturation": 1.0 if abnormal else 0.0}
    return {"status": "abnormal" if abnormal else "normal"}


def install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, epoch: int = 1,
            replies: str | None = None, outcomes: list[str] | None = None) -> tuple[dict, dict]:
    state = {"packet": packet(epoch), "context_calls": 0, "transport_calls": []}
    monkeypatch.setattr(D.INPUT, "make_packet", lambda *_args, **_kwargs: envelope(state["packet"]))

    @contextmanager
    def scope(_path, _anchor, *, fixture=False):
        old = (D.C.V.BASE_FACT_IDS, D.C.V.PROTOCOL, D.C.V.verify_protocol)
        D.C.V.BASE_FACT_IDS, D.C.V.PROTOCOL = D.RV.BASE_FACT_IDS, tmp_path / "view.json"
        D.C.V.verify_protocol = lambda _sha: {}
        try:
            yield {}
        finally:
            D.C.V.BASE_FACT_IDS, D.C.V.PROTOCOL, D.C.V.verify_protocol = old

    monkeypatch.setattr(D.RV, "view_scope", scope)
    monkeypatch.setattr(D.RV, "file_digest", lambda _path: "f" * 64)

    class Probe:
        AUTH_SCHEMA = "lineage_probe_authorization_v1"

        @staticmethod
        def prepare_probe(_source, probe, *, stage, role, **_kwargs):
            stage = Path(stage)
            stage.mkdir(parents=True, exist_ok=False)
            value = {"stage": str(stage), "probe": probe, "role": role}
            (stage / "plan.json").write_text(json.dumps(value) + "\n", encoding="utf-8")
            return value

    monkeypatch.setattr(D, "_probe_module", lambda: Probe)
    sequence = iter(outcomes or ["abnormal"])
    current: dict[str, str | None] = {"outcome": None}

    def executor(plan_path: Path, _authorization: Path) -> dict:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        if plan["role"] == "reference":
            current["outcome"] = next(sequence)
        outcome = current["outcome"]
        if outcome == "unknown":
            return {"status": "unknown", "role": plan["role"], "measurement": None,
                    "execution": {}, "error_type": "FixtureUnknown"}
        return {"status": "complete", "role": plan["role"],
                "measurement": measurement(
                    plan["probe"], outcome == "abnormal" and plan["role"] == "case"),
                "execution": {"wall_s": 0.1}, "error_type": None}

    def factory(_root, slot, _context, _view, timeout):
        state["transport_calls"].append((slot, timeout))

        def complete(messages, _params):
            view = json.loads(messages[1]["content"])
            if replies == "invalid":
                content = "{}"
            elif replies == "hold":
                content = json.dumps({"action": "hold", "probe": None,
                                      "fact_ids": [RUNTIME_FACT]})
            else:
                content = json.dumps({"action": "continue",
                                      "probe": view["eligible_probes"][0],
                                      "fact_ids": [RUNTIME_FACT]})
            return {"provider_model": D.C.CALL.MODEL, "finish_reason": "stop",
                    "content": content, "input_tokens": 3, "output_tokens": 2,
                    "total_tokens": 5, "wall_s": 0.01}
        return complete

    return state, {"probe_executor": executor, "complete_factory": factory}


def call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, epoch: int = 1,
         replies: str | None = None, outcomes: list[str] | None = None, verify_context=None):
    state, injected = install(monkeypatch, tmp_path, epoch=epoch, replies=replies, outcomes=outcomes)
    parent, prior, applied = None, None, ()
    if epoch == 2:
        prior = deepcopy(state["packet"]["prior_receipt"])
        applied = (D.C.RC.REWARD,)
        parent = {"run_id": "fixture_run", "policy": D.POLICY, "epoch": 1, **prior}

    def verified():
        state["context_calls"] += 1
        if verify_context is not None:
            verify_context(state, tmp_path)

    result = D.run_epoch(
        tmp_path / "out", tmp_path / "case.json", tmp_path / "reference.json",
        case_runtime_path=tmp_path / "case-runtime.json", reference_runtime_path=None,
        run_id="fixture_run", view_protocol=tmp_path / "view.json",
        probe_protocol=tmp_path / "probe.json", anchor="a" * 40, lab=tmp_path / "lab",
        transport_root=tmp_path / "transport", slot_ids=("slot-1", "slot-2", "slot-3"),
        verify_context=verified, applied_corrections=applied, prior_receipt=prior,
        parent=parent, view_fixture=True, **injected)
    return result, state


def test_one_abnormal_probe_confirms_and_seals_every_artifact(tmp_path, monkeypatch):
    result, state = call(tmp_path, monkeypatch, outcomes=["abnormal"])
    assert result["end_state"] == "confirmed" and result["confirmed_subtype"] is not None
    assert result["decision_count"] == result["request_total"] == 1
    assert result["runtime_fact_citations"] == [
        {"decision": 1, "fact_ids": [RUNTIME_FACT], "count": 1}]
    step = result["steps"][0]
    for key in ("snapshot", "choice", "dialogue", "request_spec", "authorization",
                "ledger_receipt"):
        assert D._sha(Path(step[key]["path"])) == step[key]["sha256"]
    assert set(step["plans"]) == set(step["receipts"]) == {"case", "reference"}
    assert state["transport_calls"] == [("slot-1", D.MODEL_TIMEOUT_S)]
    assert not result["correction_executed"] and not result["training_executed"]


def test_three_normal_probes_end_as_budget_hold_not_open_assertion(tmp_path, monkeypatch):
    result, _state = call(tmp_path, monkeypatch, outcomes=["normal", "normal", "normal"])
    assert result["decision_count"] == result["request_total"] == 3
    assert result["final_status"] == "unidentifiable"
    assert result["hold_reason"] == "budget_exhausted" and result["end_state"] == "HOLD"


def test_unknown_probe_holds_once_without_retry(tmp_path, monkeypatch):
    result, state = call(tmp_path, monkeypatch, outcomes=["unknown"])
    assert result["decision_count"] == 1 and result["hold_reason"] == "probe_outcome_unknown"
    assert result["steps"][0]["outcome"] == "unknown"
    assert state["transport_calls"] == [("slot-1", D.MODEL_TIMEOUT_S)]


@pytest.mark.parametrize("reply,reason", [("invalid", "model_invalid_hold"),
                                           ("hold", "model_valid")])
def test_model_invalid_or_explicit_hold_executes_no_probe(tmp_path, monkeypatch, reply, reason):
    result, _state = call(tmp_path, monkeypatch, replies=reply)
    assert result["decision_count"] == 0 and result["hold_reason"] == reason
    assert result["steps"][0]["request_spec"] is None
    assert not (tmp_path / "out/probes.ledger.jsonl").exists()


def test_epoch_two_requires_explicit_parent_and_preserves_binding(tmp_path, monkeypatch):
    result, _state = call(tmp_path, monkeypatch, epoch=2, outcomes=["abnormal"])
    assert result["epoch"] == 2 and result["parent"]["checkpoint_sha256"] == "a" * 64
    state, injected = install(monkeypatch, tmp_path / "missing", epoch=2)
    with pytest.raises(ValueError, match="parent"):
        D.run_epoch(
            tmp_path / "missing/out", tmp_path / "case", tmp_path / "reference",
            case_runtime_path=tmp_path / "runtime", reference_runtime_path=None,
            run_id="fixture_run", view_protocol=tmp_path / "view",
            probe_protocol=tmp_path / "probe", anchor="a" * 40, lab=tmp_path / "lab",
            transport_root=tmp_path / "transport", slot_ids=("a", "b", "c"),
            verify_context=lambda: None, applied_corrections=(D.C.RC.REWARD,),
            prior_receipt=state["packet"]["prior_receipt"], parent=None,
            view_fixture=True, **injected)


def test_changed_full_input_before_approval_is_rejected_and_scope_restored(tmp_path, monkeypatch):
    old = (D.C.V.BASE_FACT_IDS, D.C.V.PROTOCOL, D.C.V.verify_protocol)

    def change_after_plans(state, root):
        if list((root / "out/steps/step_01").glob("*/plan.json")):
            state["packet"]["facts"][RUNTIME_FACT] = 99

    with pytest.raises(ValueError, match="full input"):
        call(tmp_path, monkeypatch, outcomes=["normal"], verify_context=change_after_plans)
    assert not (tmp_path / "out/probes.ledger.jsonl").exists()
    assert (D.C.V.BASE_FACT_IDS, D.C.V.PROTOCOL, D.C.V.verify_protocol) == old
    assert (tmp_path / "out.context-preflight.json").is_file()
