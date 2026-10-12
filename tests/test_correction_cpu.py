from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    bench = str(ROOT / "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    spec = importlib.util.spec_from_file_location("check_correction_cpu_under_test", ROOT / "bench" / "check_correction_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_sources(root: Path) -> None:
    rows = {
        "source/isaaclab/isaaclab/envs/mdp/rewards.py": "def track_lin_vel_xy_exp(env, std, command_name, asset_cfg=None):\n return 1\n",
        "source/isaaclab/isaaclab/actuators/actuator_pd.py": "class DCMotor:\n def _clip_effort(self, effort):\n  return effort\n",
        "source/isaaclab/isaaclab/envs/mdp/terminations.py": "def time_out(env):\n return False\n",
    }
    for relative, text in rows.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")


def _fake_go2_source(root: Path, *, effort=23.5) -> None:
    path = root / "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "UNITREE_GO2_CFG = ArticulationCfg(actuators={'base_legs': DCMotorCfg("
        + f"effort_limit={effort}, saturation_effort=23.5, velocity_limit=30.0)"
        + "})\n",
        encoding="utf-8", newline="",
    )


def test_inputs_preserve_requested_values_and_are_not_performance_claims():
    module = _module()
    assert module.INPUTS["F1"] == {
        "command_xy": [1.0, 0.0], "body_linear_velocity_xy": [1.0, 0.0],
        "world_linear_velocity_xy": [0.0, 1.0], "std": 0.5,
    }
    assert module.INPUTS["F2"]["cases"][0] == {
        "name": "positive_speed", "effort": 23.5, "joint_velocity": 6.0,
    }
    assert module.INPUTS["F2"]["velocity_limit"] == 30.0
    assert module.INPUTS["F2"]["saturation_effort"] == 23.5
    assert module.INPUTS["F6"] == {"episode_length_step": 50, "maximum_episode_length": 1000}


def test_contract_only_mode_writes_once_without_importing_torch(tmp_path):
    module = _module()
    _fake_sources(tmp_path)
    output = tmp_path / "private" / "contract.json"
    before = set(sys.modules)
    assert module.main(["--isaaclab", str(tmp_path), "--contract-output", str(output)]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["schema"] == "correction_source_contract_v1"
    assert set(value["targets"]) == set(module.A.TARGETS)
    assert "torch" not in set(sys.modules) - before
    with pytest.raises(FileExistsError):
        module.main(["--isaaclab", str(tmp_path), "--contract-output", str(output)])


@pytest.mark.parametrize("missing", ["fault", "correction", "output"])
def test_scenario_arguments_are_all_required(tmp_path, missing):
    module = _module()
    values = {
        "fault": ["--fault", "F1"],
        "correction": ["--correction", "restore_body_frame_reward"],
        "output": ["--output", str(tmp_path / "result.json")],
    }
    argv = ["--isaaclab", str(tmp_path)]
    for key, pair in values.items():
        if key != missing:
            argv.extend(pair)
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2


def test_contract_mode_rejects_scenario_arguments(tmp_path):
    module = _module()
    with pytest.raises(SystemExit) as error:
        module.main([
            "--isaaclab", str(tmp_path), "--contract-output", str(tmp_path / "contract.json"),
            "--fault", "F1",
        ])
    assert error.value.code == 2


def test_go2_motor_inputs_are_bound_to_asset_source(tmp_path):
    module = _module()
    _fake_go2_source(tmp_path)
    binding = module._go2_motor_binding(tmp_path)
    assert binding["source"] == module.GO2_SOURCE
    assert binding["values"] == {
        "effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0,
    }
    _fake_go2_source(tmp_path, effort=23.7)
    with pytest.raises(ValueError, match="differ"):
        module._go2_motor_binding(tmp_path)


def test_verification_result_fails_when_any_check_is_false():
    checks = {
        "clean_differs_from_fault": True,
        "matching_restore_returns_clean": True,
        "cuda_remained_uninitialized": False,
    }
    assert all(checks.values()) is False
