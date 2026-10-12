from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _adapter():
    spec = importlib.util.spec_from_file_location("correction_adapter_under_test", ROOT / "bench" / "correction_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sources(root: Path) -> None:
    files = {
        "source/isaaclab/isaaclab/envs/mdp/rewards.py": "def track_lin_vel_xy_exp(env, std, command_name, asset_cfg=None):\n    return ('body', env, std)\n",
        "source/isaaclab/isaaclab/actuators/actuator_pd.py": "class DCMotor:\n    def _clip_effort(self, effort):\n        return ('motor', effort)\n",
        "source/isaaclab/isaaclab/envs/mdp/terminations.py": "def time_out(env):\n    return env.episode_length_buf >= env.max_episode_length\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")


def _load_fake_modules(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, types.ModuleType]:
    modules = {}
    for name, relative in {
        "isaaclab.envs.mdp.rewards": "source/isaaclab/isaaclab/envs/mdp/rewards.py",
        "isaaclab.actuators.actuator_pd": "source/isaaclab/isaaclab/actuators/actuator_pd.py",
        "isaaclab.envs.mdp.terminations": "source/isaaclab/isaaclab/envs/mdp/terminations.py",
    }.items():
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    return modules


@pytest.fixture
def setup(tmp_path, monkeypatch):
    adapter = _adapter()
    _write_sources(tmp_path)
    modules = _load_fake_modules(tmp_path, monkeypatch)
    contract = adapter.source_contract(tmp_path)
    return adapter, tmp_path, modules, contract


def _fault_code(function, value):
    namespace = {}
    exec(f"def replacement(*args, **kwargs):\n    return {value!r}\n", namespace)
    function.__code__ = namespace["replacement"].__code__


@pytest.mark.parametrize(
    ("correction_id", "module_name", "attribute"),
    [
        ("restore_body_frame_reward", "isaaclab.envs.mdp.rewards", "track_lin_vel_xy_exp"),
        ("restore_configured_timeout", "isaaclab.envs.mdp.terminations", "time_out"),
    ],
)
def test_code_mode_restores_in_place(setup, correction_id, module_name, attribute):
    adapter, root, modules, contract = setup
    snapshot = adapter.capture(root, contract)
    target = getattr(modules[module_name], attribute)
    identity = id(target)
    _fault_code(target, "fault")

    result = adapter.apply(snapshot, correction_id)

    assert id(getattr(modules[module_name], attribute)) == identity
    assert result["changed_targets"] == [correction_id]
    assert result["restored_targets"] == [correction_id]
    assert result["after_correction"][correction_id] == result["before_fault"][correction_id]


def test_attribute_mode_restores_raw_class_method_with_fault_closure(setup):
    adapter, root, modules, contract = setup
    snapshot = adapter.capture(root, contract)
    motor = modules["isaaclab.actuators.actuator_pd"].DCMotor
    clean = motor.__dict__["_clip_effort"]
    marker = "torch"

    def faulty(self, effort):
        return marker, effort

    motor._clip_effort = faulty
    assert adapter.fingerprint(faulty) != adapter.fingerprint(clean)
    result = adapter.apply(snapshot, "restore_motor_torque_speed")

    assert motor.__dict__["_clip_effort"] is clean
    assert result["restored_targets"] == ["restore_motor_torque_speed"]


def test_wrong_target_keeps_actual_fault_and_changes_nothing(setup):
    adapter, root, modules, contract = setup
    snapshot = adapter.capture(root, contract)
    timeout = modules["isaaclab.envs.mdp.terminations"].time_out
    _fault_code(timeout, "still faulty")
    faulty = adapter.fingerprint(timeout)

    result = adapter.apply(snapshot, "restore_body_frame_reward")

    assert result["changed_targets"] == []
    assert result["restored_targets"] == []
    assert result["after_correction"]["restore_configured_timeout"] == faulty


def test_none_on_clean_snapshot_is_an_audited_noop(setup):
    adapter, root, _, contract = setup
    snapshot = adapter.capture(root, contract)
    result = adapter.apply(snapshot, None)
    assert result["selected_correction"] is None
    assert result["changed_targets"] == []
    assert result["before_fault"] == result["after_fault"] == result["after_correction"]


def test_unknown_id_and_snapshot_reuse_are_rejected(setup):
    adapter, root, _, contract = setup
    unknown = adapter.capture(root, contract)
    with pytest.raises(ValueError, match="unknown correction"):
        adapter.apply(unknown, "restore_everything")
    adapter.apply(unknown, None)
    with pytest.raises(RuntimeError, match="already applied"):
        adapter.apply(unknown, None)


def test_capture_rejects_prepatched_live_target(setup):
    adapter, root, modules, contract = setup
    _fault_code(modules["isaaclab.envs.mdp.rewards"].track_lin_vel_xy_exp, "prepatched")
    with pytest.raises(ValueError, match="not the clean implementation"):
        adapter.capture(root, contract)


def test_capture_rejects_source_changed_after_contract(setup):
    adapter, root, _, contract = setup
    path = root / adapter.TARGETS["restore_configured_timeout"]["source"]
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the current"):
        adapter.capture(root, contract)


def test_contract_is_runtime_bound_and_snapshot_owns_a_deep_copy(setup):
    adapter, root, _, contract = setup
    assert contract["runtime"]["cache_tag"] == sys.implementation.cache_tag
    assert contract["runtime"]["version"] == list(sys.version_info[:3])
    assert all("expected_source_path" not in row for row in contract["targets"].values())
    snapshot = adapter.capture(root, contract)
    contract["targets"]["restore_body_frame_reward"]["source_sha256"] = "0" * 64
    assert snapshot.contract["targets"]["restore_body_frame_reward"]["source_sha256"] != "0" * 64


def test_capture_rejects_contract_from_another_python_runtime(setup):
    adapter, root, _, contract = setup
    contract["runtime"]["cache_tag"] = "cpython-310"
    with pytest.raises(ValueError, match="runtime differs"):
        adapter.capture(root, contract)


def test_capture_recompiles_source_and_rejects_tampered_clean_fingerprint(setup):
    adapter, root, _, contract = setup
    contract["targets"]["restore_body_frame_reward"]["clean_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the bound source"):
        adapter.capture(root, contract)


def test_apply_rejects_code_mode_slot_replacement(setup):
    adapter, root, modules, contract = setup
    snapshot = adapter.capture(root, contract)

    def replacement(*args, **kwargs):
        return "wrong replacement mode"

    modules["isaaclab.envs.mdp.rewards"].track_lin_vel_xy_exp = replacement
    with pytest.raises(ValueError, match="replaced instead of mutated"):
        adapter.apply(snapshot, "restore_body_frame_reward")


def test_apply_rejects_class_owner_replacement(setup):
    adapter, root, modules, contract = setup
    snapshot = adapter.capture(root, contract)
    old_class = modules["isaaclab.actuators.actuator_pd"].DCMotor

    class DCMotor:
        _clip_effort = old_class._clip_effort

    modules["isaaclab.actuators.actuator_pd"].DCMotor = DCMotor
    with pytest.raises(ValueError, match="owner changed"):
        adapter.apply(snapshot, "restore_motor_torque_speed")


def test_source_contract_uses_lf_hash_and_position_independent_fingerprint(setup):
    adapter, root, _, contract = setup
    row = contract["targets"]["restore_configured_timeout"]
    raw = (root / row["source"]).read_bytes()
    assert row["source_sha256"] == adapter._sha256(raw.replace(b"\r\n", b"\n"))
    one = compile("def f(x):\n return x + 1\n", "a.py", "exec").co_consts[0]
    two = compile("\n\n\ndef f(x):\n return x + 1\n", "b.py", "exec").co_consts[0]
    assert adapter.fingerprint(one) == adapter.fingerprint(two)


def test_python311_exception_table_participates_in_fingerprint():
    adapter = _adapter()
    if not hasattr((lambda: None).__code__, "co_exceptiontable"):
        pytest.skip("separate exception table starts in Python 3.11")
    plain = compile("def f():\n return 1\n", "a.py", "exec").co_consts[0]
    guarded = compile("def f():\n try:\n  return 1\n except Exception:\n  return 1\n", "b.py", "exec").co_consts[0]
    assert adapter.fingerprint(plain) != adapter.fingerprint(guarded)
