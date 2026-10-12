"""명시적 mechanism registry 고리와 digest 결합 원장 회귀."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_triage import mechanism_registry as M  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402


PREREG = {"case": "h04_s2027", "hypotheses": ["actuator", "optimizer"],
          "checkpoint_sha256": "a" * 64}


def _outcomes(truth: str) -> dict[str, str]:
    return {probe: (spec["expect"][truth] or "normal") for probe, spec in L.PROBES.items()}


@pytest.mark.parametrize("truth", L.MECHANISMS)
@pytest.mark.parametrize("strategy", ["exhaustive", "fixed", "random"])
def test_legacy_registry_matches_frozen_simulate(truth, strategy):
    outcomes = _outcomes(truth)
    old = L.simulate(strategy, outcomes, seed=3)
    new = M.simulate(M.legacy_registry(), strategy, outcomes, seed=3)
    for key in ("status", "conclusion", "probes_used", "steps"):
        assert new[key] == old[key]


@pytest.mark.parametrize("relative", [
    "evals/results/p1a_loop_compare_20261009/loop_compare.jsonl",
    "evals/results/t1_loop_compare_20261009/loop_compare.jsonl",
])
def test_archived_non_model_rows_replay_with_legacy_registry(relative):
    registry = M.legacy_registry()
    rows = [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()]
    checked = 0
    for row in rows:
        label = row["strategy"]
        if label not in ("fixed", "exhaustive") and not label.startswith("random"):
            continue
        strategy = "random" if label.startswith("random") else label
        seed = int(label.removeprefix("random")) if strategy == "random" else 0
        outcomes = {step["probe"]: step["outcome"] for step in row["steps"]}
        old = L.simulate(strategy, outcomes, seed=seed)
        new = M.simulate(registry, strategy, outcomes, seed=seed)
        for key in ("status", "conclusion", "probes_used", "steps"):
            assert new[key] == old[key]
            assert new[key] == row[key]
        checked += 1
    assert checked > 0


def test_unknown_observation_preserves_hypotheses():
    registry = M.legacy_registry()
    assert M.update(registry, ["reward", "physics"], "P_reward", "unknown") == (
        ["reward", "physics"], [])
    result = M.simulate(registry, "fixed", {probe: "unknown" for probe in registry["probes"]})
    assert result["status"] == "unidentifiable"
    assert all(step["dropped"] == [] for step in result["steps"])


@pytest.mark.parametrize("mutation", ["fields", "mechanisms", "cost", "expect", "order", "nan_args"])
def test_registry_normalization_rejects_invalid_definitions(mutation):
    registry = M.legacy_registry()
    if mutation == "fields":
        registry["extra"] = True
    elif mutation == "mechanisms":
        registry["mechanisms"].append("reward")
    elif mutation == "cost":
        registry["probes"]["P_reward"]["cost_gpu_s"] = True
    elif mutation == "expect":
        registry["probes"]["P_reward"]["expect"].pop("reward")
    elif mutation == "order":
        registry["fixed_order"].append(registry["fixed_order"][0])
    else:
        registry["probes"]["P_reward"]["args"]["bad"] = float("nan")
    with pytest.raises(ValueError):
        M.normalize(registry)


def test_registry_copy_is_independent_and_accepts_external_verified_check_definition():
    source = M.legacy_registry()
    normalized = M.normalize(source)
    source["probes"]["P_reward"]["args"]["rollout_steps"] = 1
    assert normalized["probes"]["P_reward"]["args"]["rollout_steps"] == 200
    custom = deepcopy(normalized)
    custom["probes"]["P_verified_external"] = {
        "measures": "caller-validated categorical readback", "cost_gpu_s": 0,
        "args": {"contract": "external_v1"},
        "expect": {name: None for name in custom["mechanisms"]},
    }
    custom["fixed_order"].append("P_verified_external")
    assert "P_verified_external" in M.normalize(custom)["probes"]
    assert "P_actuator_envelope" not in M.legacy_registry()["probes"]


def test_torque_neutral_factory_cannot_confirm_actuator_or_reject_optimizer():
    registry = M.without_torque_signal()
    assert set(registry["probes"]["P_torque"]["expect"].values()) == {None}
    assert "P_torque" not in registry["fixed_order"]
    keep, dropped = M.update(registry, ["actuator", "optimizer"], "P_torque", "abnormal")
    assert keep == ["actuator", "optimizer"] and dropped == []
    h04_like = {probe: "normal" for probe in registry["probes"]}
    h04_like.update(P_torque="abnormal", P_value="abnormal")
    result = M.simulate(registry, "fixed", h04_like)
    assert result["status"] == "confirmed" and result["conclusion"] == "optimizer"
    assert not any(step["probe"] == "P_torque" and "optimizer" in step["dropped"] for step in result["steps"])


def test_registry_ledger_binds_approval_args_registry_and_single_use(tmp_path):
    registry = M.legacy_registry()
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", registry)
    proposal = ledger.propose(PREREG, "P_value", 20)
    request_id = proposal["request_id"]
    assert proposal["registry_digest"] == M.registry_digest(registry)
    assert proposal["args_digest"] == L.digest(registry["probes"]["P_value"]["args"])
    assert proposal["effect_key"] == ledger.effect_key(
        proposal["prereg_digest"], proposal["probe"], proposal["args_digest"])
    approved = ledger.approve(request_id, "human")
    assert approved["registry_digest"] == proposal["registry_digest"]

    changed = deepcopy(registry)
    changed["probes"]["P_value"]["args"]["to"] = "model_299.pt"
    restarted_changed = M.RegistryLedger(ledger.path, changed)
    with pytest.raises(L.LedgerError, match="registry changed"):
        restarted_changed.consume(request_id, PREREG, "P_value")

    restarted = M.RegistryLedger(ledger.path, registry)
    consumed = restarted.consume(request_id, PREREG, "P_value")
    assert consumed["registry_digest"] == proposal["registry_digest"]
    with pytest.raises(L.LedgerError, match="not approved"):
        restarted.consume(request_id, PREREG, "P_value")
    receipt = restarted.receipt(request_id, "abnormal", {"critic_change": 0.0}, 5.0, 0)
    assert receipt["registry_digest"] == proposal["registry_digest"]
    assert receipt["effect_key"] == proposal["effect_key"]
    assert M.RegistryLedger(ledger.path, registry).requests()[request_id]["state"] == "done"


def test_pending_request_cannot_be_approved_after_registry_change(tmp_path):
    original = M.legacy_registry()
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", original)
    request_id = ledger.propose(PREREG, "P_value", 20)["request_id"]
    changed = M.without_torque_signal(original)
    with pytest.raises(L.LedgerError, match="registry changed"):
        M.RegistryLedger(ledger.path, changed).approve(request_id, "human")


@pytest.mark.parametrize("operation", ["receipt", "recover"])
def test_running_request_cannot_be_finalized_after_registry_change(tmp_path, operation):
    original = M.legacy_registry()
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", original)
    proposal = ledger.propose(PREREG, "P_value", 20)
    request_id = proposal["request_id"]
    ledger.approve(request_id, "human")
    ledger.consume(request_id, PREREG, "P_value")
    changed = M.RegistryLedger(ledger.path, M.without_torque_signal(original))
    with pytest.raises(L.LedgerError, match="registry changed"):
        if operation == "receipt":
            changed.receipt(request_id, "abnormal", {}, 1.0, 0)
        else:
            changed.recover(request_id, None)


@pytest.mark.parametrize("tamper", ["missing", "probe", "args_digest", "registry_digest", "effect_key"])
def test_artifact_recovery_requires_exact_registry_effect_binding(tmp_path, tamper):
    registry = M.legacy_registry()
    ledger = M.RegistryLedger(tmp_path / f"{tamper}.jsonl", registry)
    proposal = ledger.propose(PREREG, "P_value", 20)
    request_id = proposal["request_id"]
    ledger.approve(request_id, "human")
    ledger.consume(request_id, PREREG, "P_value")
    artifact = {"outcome": "abnormal", "measurement": {"critic_change": 0.0}, "gpu_s": 5.0,
                "exit_code": 0, "checkpoint_sha256": PREREG["checkpoint_sha256"], "source": "result.json",
                "probe": "P_value", "args_digest": proposal["args_digest"],
                "registry_digest": proposal["registry_digest"], "effect_key": proposal["effect_key"]}
    if tamper == "missing":
        artifact.pop("effect_key")
    else:
        artifact[tamper] = "wrong"
    with pytest.raises(L.LedgerError, match="binding differs"):
        ledger.recover(request_id, artifact, PREREG)
    assert ledger.requests()[request_id]["state"] == "running"


def test_artifact_recovery_preserves_all_registry_bindings(tmp_path):
    registry = M.legacy_registry()
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", registry)
    proposal = ledger.propose(PREREG, "P_value", 20)
    request_id = proposal["request_id"]
    ledger.approve(request_id, "human")
    ledger.consume(request_id, PREREG, "P_value")
    artifact = {"outcome": "abnormal", "measurement": {"critic_change": 0.0}, "gpu_s": 5.0,
                "exit_code": 0, "checkpoint_sha256": PREREG["checkpoint_sha256"], "source": "result.json",
                "probe": proposal["probe"], "args_digest": proposal["args_digest"],
                "registry_digest": proposal["registry_digest"], "effect_key": proposal["effect_key"]}
    receipt = ledger.recover(request_id, artifact, PREREG)
    for key in ("args_digest", "registry_digest", "effect_key"):
        assert receipt[key] == proposal[key]
    assert receipt["recovered_from"] == "result.json"


def test_inherited_orphan_recovery_and_legacy_event_replay(tmp_path):
    registry = M.legacy_registry()
    ledger = M.RegistryLedger(tmp_path / "registry.jsonl", registry)
    request_id = ledger.propose(PREREG, "P_value", 20)["request_id"]
    ledger.approve(request_id, "human")
    ledger.consume(request_id, PREREG, "P_value")
    restarted = M.RegistryLedger(ledger.path, registry)
    assert restarted.orphans() == [request_id]
    receipt = restarted.recover(request_id, None)
    assert receipt["outcome"] == "unknown" and restarted.orphans() == []

    legacy_path = tmp_path / "legacy.jsonl"
    old = L.Ledger(legacy_path)
    old_id = old.propose(PREREG, "P_value", 20)["request_id"]
    old.approve(old_id, "human")
    old.consume(old_id, PREREG, "P_value")
    old.receipt(old_id, "normal", {}, 1.0, 0)
    assert M.RegistryLedger(legacy_path, registry).requests()[old_id]["state"] == "done"


@pytest.mark.parametrize("gpu_s", [float("nan"), float("inf"), -1.0, True, None])
def test_invalid_duration_cannot_finalize_approval(tmp_path, gpu_s):
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", M.legacy_registry())
    request_id = ledger.propose(PREREG, "P_value", 20)["request_id"]
    ledger.approve(request_id, "human")
    ledger.consume(request_id, PREREG, "P_value")
    with pytest.raises(L.LedgerError, match="duration"):
        ledger.receipt(request_id, "abnormal", {}, gpu_s, 0)
    assert ledger.requests()[request_id]["state"] == "running"


@pytest.mark.parametrize("missing", ["gpu_s", "exit_code", "measurement", "source"])
def test_recovery_cannot_default_missing_execution_evidence_to_success(tmp_path, missing):
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", M.legacy_registry())
    proposal = ledger.propose(PREREG, "P_value", 20)
    request_id = proposal["request_id"]
    ledger.approve(request_id, "human")
    ledger.consume(request_id, PREREG, "P_value")
    artifact = {"outcome": "abnormal", "measurement": {}, "gpu_s": 5.0,
                "exit_code": 0, "source": "result.json",
                "checkpoint_sha256": PREREG["checkpoint_sha256"],
                **{key: proposal[key] for key in ("probe", "args_digest", "registry_digest", "effect_key")}}
    artifact.pop(missing)
    with pytest.raises(L.LedgerError, match="execution fields"):
        ledger.recover(request_id, artifact, PREREG)
    assert ledger.requests()[request_id]["state"] == "running"


@pytest.mark.parametrize("budget", [-1, True, float("nan")])
def test_invalid_budget_cannot_create_approval(tmp_path, budget):
    ledger = M.RegistryLedger(tmp_path / "ledger.jsonl", M.legacy_registry())
    with pytest.raises(L.LedgerError, match="budget"):
        ledger.propose(PREREG, "P_value", budget)
    assert ledger.events() == []


def test_finite_individual_costs_cannot_overflow_total_evidence_cost():
    registry = M.legacy_registry()
    for spec in registry["probes"].values():
        spec["cost_gpu_s"] = 1e308
    with pytest.raises(ValueError, match="total probe cost"):
        M.simulate(registry, "exhaustive", {}, budget=2)
