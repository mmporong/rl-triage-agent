from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

torch = pytest.importorskip("torch")

import recovery_reset_interventions as I  # noqa: E402
import run_recovery_reset as H  # noqa: E402


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
        self.critic = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 1))
        self.std = torch.nn.Parameter(torch.ones(3))


def _fixture():
    torch.manual_seed(41)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    observation = torch.randn(11, 5)
    loss = policy.actor(observation).square().mean() + policy.critic(observation).square().mean() + policy.std.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert len(optimizer.state) == len(list(policy.parameters()))
    return policy, optimizer


def _tensors(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _same(left, right, names=None):
    names = left if names is None else names
    return all(torch.equal(left[name], right[name]) for name in names)


def test_optimizer_only_clears_all_adam_state_and_preserves_policy_groups_and_rng():
    policy, optimizer = _fixture()
    policy_before = _tensors(policy)
    groups_before = copy.deepcopy(optimizer.state_dict()["param_groups"])
    rng_before = torch.random.get_rng_state().clone()

    row = I.apply_intervention(policy, optimizer, "optimizer_only", 120260, torch)

    assert _same(policy_before, _tensors(policy))
    assert optimizer.state == {}
    assert optimizer.state_dict()["param_groups"] == groups_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert row["target_parameter_count"] == len(list(policy.parameters()))
    assert row["policy_state_before"] == row["policy_state_after"]
    assert row["optimizer_state_after"] == row["optimizer_state_expected_after"]


def test_critic_reset_changes_only_critic_and_clears_only_its_adam_state():
    policy, optimizer = _fixture()
    before = _tensors(policy)
    actor_parameters = set(policy.actor.parameters()) | {policy.std}
    critic_parameters = set(policy.critic.parameters())
    actor_moments = {parameter: copy.deepcopy(optimizer.state[parameter]) for parameter in actor_parameters}
    rng_before = torch.random.get_rng_state().clone()

    row = I.apply_intervention(policy, optimizer, "critic_reset", 120261, torch)
    after = _tensors(policy)

    assert _same(before, after, [name for name in before if not name.startswith("critic.")])
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("critic."))
    assert all(parameter not in optimizer.state for parameter in critic_parameters)
    for parameter in actor_parameters:
        assert optimizer.state[parameter].keys() == actor_moments[parameter].keys()
        assert all(torch.equal(optimizer.state[parameter][key], value)
                   for key, value in actor_moments[parameter].items())
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert row["target_modules"] == ["critic.0", "critic.2"]


def test_actor_last_layer_reset_preserves_actor_prefix_critic_std_and_parameter_identity():
    policy, optimizer = _fixture()
    before = _tensors(policy)
    identities = {name: id(parameter) for name, parameter in policy.named_parameters()}
    target_parameters = set(policy.actor[2].parameters())
    other_states = {parameter: copy.deepcopy(state) for parameter, state in optimizer.state.items()
                    if parameter not in target_parameters}

    row = I.apply_intervention(policy, optimizer, "actor_last_layer_reset", 120262, torch)
    after = _tensors(policy)

    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed == {"actor.2.weight", "actor.2.bias"}
    assert {name: id(parameter) for name, parameter in policy.named_parameters()} == identities
    assert all(parameter not in optimizer.state for parameter in target_parameters)
    for parameter, state in other_states.items():
        assert all(torch.equal(optimizer.state[parameter][key], value) for key, value in state.items())
    assert row["target_modules"] == ["actor.2"]
    assert row["parameter_identity_preserved"] is True


def test_reset_is_deterministic_for_registered_seed_and_fail_closed_for_invalid_targets():
    first_policy, first_optimizer = _fixture()
    second_policy, second_optimizer = _fixture()
    I.apply_intervention(first_policy, first_optimizer, "critic_reset", 120261, torch)
    I.apply_intervention(second_policy, second_optimizer, "critic_reset", 120261, torch)
    assert _same(_tensors(first_policy), _tensors(second_policy))

    with pytest.raises(ValueError, match="unregistered"):
        I.apply_intervention(first_policy, first_optimizer, "actor_all", 1, torch)
    first_policy.critic = torch.nn.Tanh()
    with pytest.raises(ValueError, match="no target Linear"):
        I.apply_intervention(first_policy, first_optimizer, "critic_reset", 1, torch)


def test_moment_indices_follow_optimizer_order_when_target_parameter_order_differs():
    torch.manual_seed(7)
    policy = Policy()
    reversed_parameters = list(reversed(list(policy.parameters())))
    optimizer = torch.optim.Adam(reversed_parameters, lr=3e-4)
    for index, parameter in enumerate(reversed_parameters):
        if parameter in set(policy.critic.parameters()) and index % 2 == 0:
            optimizer.state[parameter] = {"step": torch.tensor(1.0),
                                          "exp_avg": torch.ones_like(parameter),
                                          "exp_avg_sq": torch.ones_like(parameter)}
    flat = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = sorted(index for index, parameter in enumerate(flat)
                      if parameter in optimizer.state and parameter in set(policy.critic.parameters()))
    row = I.apply_intervention(policy, optimizer, "critic_reset", 120261, torch)
    assert row["optimizer_moment_indices_present_before"] == expected


def test_protocol_contract_registers_three_arms_and_fixed_result_wording():
    assert H.ARMS == I.INTERVENTIONS
    assert set(H.FIXED["predictions"].values()) == {None}
    assert H.FIXED["design_disclosure"]
    assert H.FIXED["mechanism_claim_limit"]
    assert H.FIXED["literature_scope_limit"]
    assert H.reset_seed(2026, "optimizer_only") != H.reset_seed(2026, "critic_reset")
    with pytest.raises(ValueError):
        H.intervention_for_arm("restart")


def test_preregistered_protocol_binds_current_sources_and_seen_v4_references():
    protocol = H.read(H.PROTOCOL)
    assert set(protocol) == set(H.FIXED) | H.DYNAMIC
    assert all(protocol[key] == value for key, value in H.FIXED.items())
    assert protocol["source_files"] == {name: H.sha(ROOT / name, True) for name in sorted(H.CODE)}
    assert protocol["v4_reference_hashes"] == H.reference_hashes()


def test_intervention_readback_rejects_fabricated_state_binding():
    policy, optimizer = _fixture()
    row = I.apply_intervention(policy, optimizer, "critic_reset", H.reset_seed(2026, "critic_reset"), torch)
    source = {"policy_state_sha256": row["policy_state_before"],
              "optimizer_state_sha256": row["optimizer_state_before"]}
    initial = {"observed": {"environment_seed": 2026},
               "policy_state": {"sha256": row["policy_state_after"]},
               "optimizer_state": {"sha256": row["optimizer_state_after"]}}
    H.validate_intervention(row, "critic_reset", source, initial)
    with pytest.raises(ValueError, match="not bound"):
        H.validate_intervention({**row, "policy_state_after": "0" * 64}, "critic_reset", source, initial)


@pytest.mark.parametrize("status", ["partial", "gpu_busy_no_launch", "invalid"])
def test_execute_cli_returns_nonzero_for_every_incomplete_status(monkeypatch, status):
    monkeypatch.setattr(H, "execute", lambda tag, seed, isaaclab: {"status": status})
    assert H.main(["execute", "--tag", "exit_contract", "--seed", "2026"]) == 1


def test_aggregate_rejects_unbound_result_dictionary_file(tmp_path):
    forged = tmp_path / "result.json"
    forged.write_text(json.dumps({"tag": "forged", "seed": 2026, "status": "complete"}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its registered tag"):
        H.aggregate([forged])


def test_reference_validation_rejects_bad_fixed_report(monkeypatch):
    report_path = H._reference_paths(2026)["report"]
    original_read = H.read
    bad = copy.deepcopy(original_read(report_path))
    bad["task"] = "Other-Task-v0"

    def fake_read(path):
        return bad if Path(path).resolve() == report_path.resolve() else original_read(path)

    monkeypatch.setattr(H, "read", fake_read)
    with pytest.raises(ValueError, match="task"):
        H._validated_reference(2026)
