"""D3 probe 진입점: probe 프로세스에 부분집합을 걸 때 코드 교체·env 설정 적용·확인 줄이 맞는지 가짜 모듈로 본다."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    sys.path.insert(0, str(directory))

import d3_changes as D  # noqa: E402
import d3_isaac as I  # noqa: E402
import run_d3_experiment as X  # noqa: E402
from test_d3_changes import CFG_VALUES, _env_cfg, _fake_isaac, _lines  # noqa: E402

OLD_VALUES = {key: value for key, value in CFG_VALUES.items()}
OLD_VALUES.update({"rewards.track_lin_vel_xy_exp.weight": 1.5, "scene.robot.actuators.base_legs.effort_limit": 23.5,
                   "scene.robot.actuators.base_legs.saturation_effort": 23.5,
                   "observations.policy.joint_vel.noise.n_min": -1.5, "observations.policy.joint_vel.noise.n_max": 1.5,
                   "terminations.base_contact.params.threshold": 1.0, "rewards.dof_acc_l2.weight": -2.5e-07})


@pytest.mark.parametrize("spec", [",".join(D.HARMLESS) + ",U1", ",".join(D.HARMLESS) + ",F2", "S1,S3,F1", "NONE"])
def test_probe_arming_swaps_code_and_writes_the_env_overrides_into_the_cfg_before_the_env_exists(monkeypatch, capsys, spec):
    gymnasium = _fake_isaac(monkeypatch)
    cfg = _env_cfg(OLD_VALUES)  # probes.py는 hydra 없이 기본 설정으로 env를 만든다
    ids = I.arm_for_probes(spec)
    gymnasium.make("task", cfg=cfg)
    gymnasium.make("task", cfg=cfg)
    (line,) = _lines(capsys)
    assert line["ok"] is True and line["changes"] == list(ids) and X.readback_ok(line, ids)
    for key, (_old, new) in D.expected_params(ids).items():
        if key.startswith("env.yaml:"):
            assert D._resolve(cfg, key[len("env.yaml:"):]) == new
    if spec == "NONE":
        assert ids == () and D._resolve(cfg, "rewards.track_lin_vel_xy_exp.weight") == 1.5


def test_probe_arming_needs_the_cfg_keyword_and_a_known_subset(monkeypatch, capsys):
    gymnasium = _fake_isaac(monkeypatch)
    I.arm_for_probes("H2")
    with pytest.raises(SystemExit, match="cfg"):
        gymnasium.make("task")
    with pytest.raises(SystemExit):
        I.arm_for_probes("U1,F1")


def test_a_setting_the_probe_env_did_not_receive_fails_the_readback(monkeypatch, capsys):
    gymnasium = _fake_isaac(monkeypatch)
    monkeypatch.setattr(I, "apply_env_overrides", lambda cfg, ids: None)  # 적용을 빠뜨린 경우
    I.arm_for_probes("H5")
    gymnasium.make("task", cfg=_env_cfg(OLD_VALUES))
    (line,) = _lines(capsys)
    assert line["ok"] is False and not X.readback_ok(line, ("H5",))


def test_main_runs_the_frozen_probes_script_with_the_arguments_after_the_separator(monkeypatch):
    calls = []
    monkeypatch.setattr(I, "arm_for_probes", lambda spec: calls.append(("arm", spec)))
    monkeypatch.setattr(I.runpy, "run_path", lambda path, run_name: calls.append(("run", Path(path).name, list(sys.argv))))
    monkeypatch.setenv(D.ENV_VAR, "H1")
    monkeypatch.setattr(sys, "argv", ["d3_isaac.py"])
    assert I.main(["--", "--jobs", "j.json", "--headless"]) == 0
    assert calls == [("arm", "H1"), ("run", "probes.py", [str(I.PROBES_SCRIPT), "--jobs", "j.json", "--headless"])]
