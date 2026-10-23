"""D4 v2의 짧은 probe 실행 이름과 새 seed 경계 회귀."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests")]
import selection_comparison_flow_v2 as F  # type: ignore[import-not-found]  # noqa: E402
import test_selection_comparison_controller_v1 as FIX  # type: ignore[import-not-found]  # noqa: E402

ACTUAL_TAG = "selection_comparison_s2171_20261023"
MAX_SUFFIX_TAG = ACTUAL_TAG + "abcdefghijkl"


def test_probe_run_tokens_cover_registry_once_without_collision():
    registered = {*F.CONTROL.L.PROBES, F.CONTROL.RC.NEW_PROBE}
    assert set(F.PROBE_RUN_TOKENS) == registered
    assert len(set(F.PROBE_RUN_TOKENS.values())) == len(registered) == 7


def test_every_preregistered_probe_name_passes_common_contract_and_is_unique():
    names = {}
    for policy in F.CONTROL.POLICIES:
        for epoch in (1, 2):
            for ordinal in range(1, 8):
                for probe in F.PROBE_RUN_TOKENS:
                    value = F._probe_run_name(ACTUAL_TAG, policy, epoch, ordinal, probe)
                    assert F.CONTROL.BACKEND.V1._name(value) == value  # noqa: SLF001
                    assert len(value) <= 80
                    key = policy, epoch, ordinal, probe
                    assert value not in names, (key, names.get(value))
                    names[value] = key
    assert len(names) == 4 * 2 * 7 * 7


def test_maximum_allowed_tag_suffix_keeps_all_backend_names_within_contract():
    names = {
        F._run_name(MAX_SUFFIX_TAG, f"s{F.SEED}", arm)
        for arm in ("none", "compound", "source", "reference_legacy", "reference_optimizer")
    }
    for policy in F.CONTROL.POLICIES:
        for epoch in (1, 2):
            names.add(F._run_name(MAX_SUFFIX_TAG, policy, f"e{epoch}", "repair"))
            names.add(F._run_name(MAX_SUFFIX_TAG, policy, f"e{epoch}", "evaluation"))
            for ordinal in range(1, 8):
                for probe in F.PROBE_RUN_TOKENS:
                    names.add(F._probe_run_name(MAX_SUFFIX_TAG, policy, epoch, ordinal, probe))
    assert all(F.CONTROL.BACKEND.V1._name(name) == name for name in names)  # noqa: SLF001
    assert len(names) == 5 + 4 * 2 * 2 + 4 * 2 * 7 * 7
    assert max(map(len, names)) <= 80


def test_longest_actual_policy_epoch_ordinal_optimizer_name_is_shortened():
    old = f"{ACTUAL_TAG}_model_first_then_rule_e2_p7_p_optimizer_updates"
    assert len(old) == 83
    with pytest.raises(ValueError, match="run_name"):
        F.CONTROL.BACKEND.V1._name(old)  # noqa: SLF001
    fixed = F._probe_run_name(
        ACTUAL_TAG, "model_first_then_rule", 2, 7, "P_optimizer_updates")
    assert fixed == "s2171_20261023_model_first_then_rule_e2_p7_p_opt_updates"
    assert len(fixed) == 56


def test_real_authorize_and_validation_accept_optimizer_execution_plan(tmp_path):
    arms = tmp_path / "arms"
    arm = policy = "model_first_then_rule"
    for _ in range(4):
        snapshot = FIX.snapshot(arms, arm, policy)
        dialogue = FIX.model_dialogue(snapshot) if F.CONTROL.dialogue_input(snapshot)["needs_model"] else None
        choice = F.CONTROL.choose(snapshot, policy, dialogue)
        spec = F.CONTROL.request_spec(snapshot, choice)
        ledger = F.CONTROL.S.M.RegistryLedger(
            arms / arm / "epoch1/probes.ledger.jsonl", F.CONTROL.RC.registry())
        request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
        ledger.approve(request["request_id"], "cpu")
        ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
        ledger.receipt(request["request_id"], "normal", {}, 1.0, 0)

    snapshot = FIX.snapshot(arms, arm, policy)
    choice = F.CONTROL.choose(snapshot, policy, None)
    assert choice["probe"] == "P_optimizer_updates"
    spec = F.CONTROL.request_spec(snapshot, choice)
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    plan = {
        "stage": str((tmp_path / "probe_stage").resolve()),
        "run_name": F._probe_run_name(ACTUAL_TAG, policy, 1, 5, choice["probe"]),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "seed": F.SEED,
        "args": spec["args"],
        "gpu_contract": F.GPU_CONTRACT,
        "reference": {
            "evidence_path": str(reference.resolve()),
            "evidence_sha256": F.CONTROL.digest_file(reference),
            "checkpoint_sha256": spec["reference_checkpoint_sha256"],
            "stage": str(reference.parent.resolve()),
            "run_name": F._run_name(ACTUAL_TAG, "reference"),
            "seed": F.SEED,
            "gpu_contract": F.GPU_CONTRACT,
        },
    }
    authorization = F.CONTROL.authorize_probe(
        arms, arm, 1, snapshot, choice, "cpu", evidence_kind=F.CONTROL.ACTUAL_EVIDENCE,
        execution_plan=plan)
    verified = F.CONTROL.validate_probe_authorization(authorization, snapshot)
    assert verified["execution_plan"] == plan
    assert verified["request"]["state"] == "running"


def test_unregistered_probe_and_token_registry_drift_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="등록되지 않은"):
        F._probe_run_name(ACTUAL_TAG, "registry_next", 1, 1, "P_unknown")
    monkeypatch.setitem(F.PROBE_RUN_TOKENS, "P_reward", "p_value")
    with pytest.raises(ValueError, match="충돌"):
        F._probe_run_name(ACTUAL_TAG, "registry_next", 1, 1, "P_reward")


def test_execute_uses_fresh_seed_before_creating_new_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(F.DEMO, "PRIVATE", tmp_path)
    monkeypatch.setattr(F.DEMO, "verify", lambda *_: ({}, "a" * 40))
    monkeypatch.setattr(F.DEMO, "sha", lambda *a, **k: "b" * 64)
    calls = []

    def guard(seeds, isaaclab, *, repo):
        calls.append((tuple(seeds), isaaclab, repo))
        return {"fresh": True}

    monkeypatch.setattr(F, "run", lambda *a, **k: {
        "schema": "selection_comparison_result_v2", "status": "complete"})
    result = F.execute(ACTUAL_TAG, tmp_path, "operator", True, backend=object(),
                       dialogue_caller=object(), seed_guard=guard)
    assert result["status"] == "complete" and calls[0][0] == (2171,)
    assert (tmp_path / ACTUAL_TAG / "result.json").is_file()
