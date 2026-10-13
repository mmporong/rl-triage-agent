"""Isaac 쪽 공용 U1 진입점(가짜 보상 모듈·gymnasium, GPU 없음): 주입 1회와 확인 줄, 보조 지문, 교정 순서, shim 계약."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import train_with_recovery_mode as R  # noqa: E402
import upstream_faults as U  # noqa: E402
import upstream_isaac as I  # noqa: E402

SIGNATURES = {"lin_vel_z_l2": "env, asset_cfg=None", "ang_vel_xy_l2": "env, asset_cfg=None",
              "track_lin_vel_xy_exp": "env, std, command_name, asset_cfg=None",
              "track_ang_vel_z_exp": "env, std, command_name, asset_cfg=None", "is_alive": "env"}
TERMS = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "is_alive", "feet_air_time"]


@pytest.fixture
def fake_isaac(monkeypatch):
    """보상 모듈(4함수 + is_alive)과 그 함수에 묶인 보상 항을 가진 환경을 만드는 gymnasium."""
    rewards = ModuleType(U.REWARDS_MODULE)
    exec("\n".join(f"def {name}({args}):\n    return '{name}'" for name, args in SIGNATURES.items()),  # noqa: S102
         rewards.__dict__)
    for name in SIGNATURES:
        getattr(rewards, name).__module__ = U.REWARDS_MODULE
    monkeypatch.setitem(sys.modules, U.REWARDS_MODULE, rewards)
    other = ModuleType("isaaclab_tasks.velocity_mdp")
    exec("def feet_air_time(env):\n    return 0", other.__dict__)  # noqa: S102

    def make(*_args, **_kwargs):
        funcs = {name: getattr(rewards, name, None) or other.feet_air_time for name in TERMS}
        manager = SimpleNamespace(active_terms=TERMS, get_term_cfg=lambda name: SimpleNamespace(func=funcs[name]))
        return SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=manager))
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    return SimpleNamespace(rewards=rewards, gymnasium=gymnasium)


def _lines(capsys, tag: str) -> list[dict]:
    return [json.loads(line[len(tag) + 1:]) for line in capsys.readouterr().out.splitlines() if line.startswith(tag + " ")]


def test_arm_injects_once_and_records_the_aux_fingerprints(fake_isaac, capsys):
    injected = []
    I.arm("U1", inject=lambda fault: injected.append(fault) or U.before_env(fault))
    fake_isaac.gymnasium.make("task")
    fake_isaac.gymnasium.make("task")
    out = capsys.readouterr().out.splitlines()
    readbacks = [json.loads(line[len("[upstream_fault] "):]) for line in out if line.startswith("[upstream_fault] ")]
    aux = [json.loads(line[len("[upstream_aux] "):]) for line in out if line.startswith("[upstream_aux] ")]
    assert injected == ["U1"] and len(readbacks) == len(aux) == 1 and readbacks[0]["ok"] is True
    row = aux[0]
    assert row["used_for_verdict"] is False and "판정에 쓰지 않는다" in row["note"]
    assert row["differs_from_clean"] == sorted(U.SWAPS["U1"]) and row["clean_count"] == 1
    assert row["active_reward_terms"] == {t: t for t in TERMS if t != "feet_air_time"}
    assert row["differing_active_terms"] == sorted(U.SWAPS["U1"])
    with pytest.raises(SystemExit):
        I.arm("U9")


def test_shim_keeps_capture_inject_apply_order_and_records_the_remaining_functions(fake_isaac, capsys):
    """train_with_recovery_mode.execute처럼 _before_env를 바꾼 뒤 arm을 부르면 교정 순서가 그 함수 안에서 지켜진다."""
    shim = I.hidden_faults_shim("U1")
    original_before = shim._before_env
    events = []
    clean_code = fake_isaac.rewards.track_lin_vel_xy_exp.__code__

    def corrected_before(fault):  # execute.corrected_before와 같은 모양: capture → 원래 _before_env → apply
        events.append("capture")
        original_before(fault)
        events.append("inject_done")
        fake_isaac.rewards.track_lin_vel_xy_exp.__code__ = clean_code  # 등록 교정 restore_body_frame_reward
        events.append("apply")
    shim._before_env = corrected_before
    shim.arm("U1")
    fake_isaac.gymnasium.make("task")
    out = capsys.readouterr().out.splitlines()
    readback = next(json.loads(line[17:]) for line in out if line.startswith("[upstream_fault] "))
    aux = next(json.loads(line[15:]) for line in out if line.startswith("[upstream_aux] "))
    assert events == ["capture", "inject_done", "apply"]
    assert readback["swapped"] == {"track_lin_vel_xy_exp": False, "track_ang_vel_z_exp": True,
                                   "lin_vel_z_l2": True, "ang_vel_xy_l2": True} and readback["ok"] is False
    assert aux["differs_from_clean"] == ["ang_vel_xy_l2", "lin_vel_z_l2", "track_ang_vel_z_exp"]
    with pytest.raises(RuntimeError):
        original_before("F1")


def test_recovery_trainer_only_uses_the_two_shim_attributes():
    """shim이 채우는 계약(_before_env·arm)이 train_with_recovery_mode.execute에서 쓰는 hidden_faults 속성의 전부다."""
    tree = ast.parse(inspect.getsource(R.execute))
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "hidden_faults"}
    assert used == {"_before_env", "arm"}
    assert "import hidden_faults" in inspect.getsource(R.execute)  # 호출 시점 import라 sys.modules의 shim이 쓰인다


def test_correction_inject_captures_before_the_fault_and_restores_one_target(fake_isaac, monkeypatch, tmp_path):
    order = []
    fake_snapshot = object()
    monkeypatch.setattr(I.A, "capture", lambda root, contract: order.append(("capture", root, contract)) or fake_snapshot)
    monkeypatch.setattr(I.U, "before_env", lambda fault: order.append(("inject", fault)))
    readback = {"schema": "correction_readback_v1", "before_fault": {"restore_body_frame_reward": "a"},
                "after_fault": {"restore_body_frame_reward": "b"}, "after_correction": {"restore_body_frame_reward": "a"}}
    monkeypatch.setattr(I.A, "apply", lambda snap, cid: order.append(("apply", snap is fake_snapshot, cid)) or readback)
    out = tmp_path / "fix/correction.json"
    I.correction_inject("U1", "restore_body_frame_reward", tmp_path, {"c": 1}, out)("U1")
    assert [o[0] for o in order] == ["capture", "inject", "apply"] and order[2][1:] == (True, "restore_body_frame_reward")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["registered_target_state"] == "restored_to_clean"
    assert body["registered_target"] == "isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp"
    assert "그 밖 함수 상태는 보조 지문 기록" in body["scope_note"] and body["registered_readback"] == readback
    with pytest.raises(FileExistsError):  # 교정 readback은 한 번만 쓴다
        I.correction_inject("U1", "restore_body_frame_reward", tmp_path, {}, out)("U1")
    with pytest.raises(SystemExit):
        I.correction_inject("U1", "restore_everything", tmp_path, {}, out)


def test_main_routes_modes(fake_isaac, monkeypatch, tmp_path):
    calls = []
    metered = ModuleType("metered_probes")
    metered.main = lambda argv: calls.append(("probes", argv)) or 0
    monkeypatch.setitem(sys.modules, "metered_probes", metered)
    monkeypatch.setattr(I, "_run_train", lambda rest: calls.append(("train", rest)))
    monkeypatch.setattr(I, "arm", lambda fault, inject=None, on_env=None: calls.append(("arm", fault, inject is None)))
    assert I.main(["probes", "--fault", "U1", "--", "--jobs", "j.json", "--metrics", "m.json", "--headless"]) == 0
    I.main(["train", "--fault", "U1", "--", "--task", "T", "--seed", "2040"])
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    I.main(["correct", "--fault", "U1", "--correction", "restore_body_frame_reward", "--contract", str(contract),
            "--readback", str(tmp_path / "c.json"), "--", "--seed", "3040"])
    assert calls == [("arm", "U1", True), ("probes", ["--jobs", "j.json", "--metrics", "m.json", "--headless"]),
                     ("arm", "U1", True), ("train", ["--task", "T", "--seed", "2040"]),
                     ("arm", "U1", False), ("train", ["--seed", "3040"])]
    with pytest.raises(SystemExit):
        I.main(["correct", "--fault", "U1", "--", "--seed", "1"])
