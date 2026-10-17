from pathlib import Path
import sys
from types import SimpleNamespace

from bench import combination_semantics_v1 as S


def _synthetic_isaaclab(tmp_path: Path) -> Path:
    root = tmp_path / "isaaclab"
    velocity = root / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
    rough = root / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py"
    flat = root / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/flat_env_cfg.py"
    rewards = root / S.FAMILY.SOURCE
    for path in (velocity, rough, flat, rewards):
        path.parent.mkdir(parents=True, exist_ok=True)
    velocity.write_text("""
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
""", encoding="utf-8")
    rough.write_text("class AnymalCRoughEnvCfg(LocomotionVelocityRoughEnvCfg):\n    pass\n", encoding="utf-8")
    flat.write_text("class AnymalCFlatEnvCfg(AnymalCRoughEnvCfg):\n    pass\n", encoding="utf-8")
    rewards.write_text("""
def track_lin_vel_xy_exp(env, std, command_name, asset_cfg=None): return 1
def track_ang_vel_z_exp(env, std, command_name, asset_cfg=None): return 2
def lin_vel_z_l2(env, asset_cfg=None): return 3
def ang_vel_xy_l2(env, asset_cfg=None): return 4
""", encoding="utf-8")
    return root


def test_registered_weights_cover_all_u1_targets():
    assert set(S.EXPECTED_WEIGHTS) == set(S.FAMILY.FUNCTIONS)
    assert S.EXPECTED_WEIGHTS == {
        "track_lin_vel_xy_exp": 1.0,
        "track_ang_vel_z_exp": 0.5,
        "lin_vel_z_l2": -2.0,
        "ang_vel_xy_l2": -0.05,
    }


def test_task_binding_reads_anymal_inheritance_sources(tmp_path):
    isaaclab = _synthetic_isaaclab(tmp_path)
    terms, weights, binding = S._task_terms(isaaclab)
    assert weights == S.EXPECTED_WEIGHTS
    assert all(terms[name]["function_name"] == name for name in S.FAMILY.FUNCTIONS)
    assert not any(binding["registered_targets_overridden_in_anymal"].values())
    assert binding["inheritance"] == {
        "AnymalCRoughEnvCfg": ["LocomotionVelocityRoughEnvCfg"],
        "AnymalCFlatEnvCfg": ["AnymalCRoughEnvCfg"],
    }
    assert all(len(row["sha256_lf"]) == 64 for row in binding["sources"].values())


def test_isolated_source_functions_match_registered_fingerprints(tmp_path):
    isaaclab = _synthetic_isaaclab(tmp_path)
    contract = S.FAMILY.source_contract(isaaclab)
    module, _ = S._isolated_reward_module(isaaclab, contract)
    actual = {name: S.FAMILY.BASE.fingerprint(getattr(module, name)) for name in S.FAMILY.FUNCTIONS}
    assert actual == contract["functions"]


def test_cpu_result_uses_registered_family_apply_for_restore(monkeypatch, tmp_path):
    isaaclab = _synthetic_isaaclab(tmp_path)
    monkeypatch.chdir(isaaclab)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    result = S.inspect(isaaclab)
    row = result["restoration_readback"]
    assert row["selected_correction"] == S.FAMILY.CORRECTION_ID
    assert row["restored_targets"] == list(S.FAMILY.FUNCTIONS)
    assert row["unrelated_changed_after_correction"] == []
