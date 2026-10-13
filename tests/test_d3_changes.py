"""D3 변경 카탈로그·주입기·diff 렌더러·canary를 Isaac Sim 없이 고정한다."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import d3_changes as D  # noqa: E402
import d3_diff as R  # noqa: E402
import hidden_faults as HF  # noqa: E402
import run_d3_harmless_smoke as H  # noqa: E402
import upstream_faults as U  # noqa: E402

ISAACLAB = Path.home() / "IsaacLab"
needs_isaaclab = pytest.mark.skipif(not (ISAACLAB / D.REWARDS[1]).is_file(), reason="로컬 Isaac Lab v2.1.1 소스가 없다")


def _clip(x, min=None, max=None):  # torch.clip과 같은 키워드 이름
    return np.clip(x, min, max)


FAKE_TORCH = SimpleNamespace(square=np.square, exp=np.exp, zeros_like=np.zeros_like, clip=_clip, Tensor=object,
                             sum=lambda x, dim: np.sum(x, axis=dim), norm=lambda x, dim: np.linalg.norm(x, axis=dim),
                             clamp=lambda x, max=None: np.minimum(x, max))


# --- 카탈로그 ---
def test_the_catalog_has_seven_harmless_changes_across_categories_and_three_harmful_candidates():
    assert D.HARMLESS_V1 == ("H1", "H2", "H3", "H4", "H5", "H6", "H7")  # smoke v1(fail)의 목록은 그대로 남긴다
    assert D.HARMLESS == D.HARMLESS_V2 == ("H1", "H3", "H4", "H5", "H6", "H7", "S4") and D.RETIRED == ("H2",)
    assert list(D.HARMLESS) == [cid for cid in D.CHANGES if cid in D.HARMLESS]  # 확인 줄·부분집합 id와 같은 순서
    assert D.RESERVE == ("S1", "S2", "S3") and D.HARMFUL == ("U1", "F1", "F2")
    kinds = {cid: (D.CHANGES[cid]["kind"], D.CHANGES[cid]["category"]) for cid in D.HARMLESS}
    assert sorted(kinds.values()) == sorted([("code", "reward"), ("code", "reward"), ("code", "reward"),
                                             ("config", "actuator"), ("config", "observation"), ("code", "logging"),
                                             ("config", "optimizer")])
    assert [D.CHANGES[cid]["category"] for cid in D.HARMFUL] == ["reward", "reward", "actuator"]
    # 범주 규칙만으로 범인을 고를 수 없게 해로운 후보와 같은 범주의 무해 변경이 있다
    assert {D.CHANGES[cid]["category"] for cid in D.HARMFUL} <= {kinds[cid][1] for cid in D.HARMLESS}
    assert all(D.CHANGES[cid]["sources"] and D.CHANGES[cid]["definition"] for cid in D.CHANGES)


def test_new_config_values_lie_in_documented_ranges_or_are_marked_as_designer_choices():
    for cid, change in D.CHANGES.items():
        if change["kind"] != "config":
            continue
        documented = [abs(v) for v in change["range"]["values"].values()]
        news = [abs(new) for _old, new in change["overrides"].values()]
        inside = all(min(documented) <= v <= max(documented) for v in news)
        assert inside is change["range"]["inside"], cid
        if not inside:
            assert any("설계자 선택" in s for s in change["sources"]), cid


def test_the_catalog_refuses_a_harmless_change_on_a_harmful_target_or_a_shared_override(monkeypatch):
    D.check_catalog()  # U1·F1은 같은 함수를 바꾸지만 해로운 후보끼리라 허용된다
    harmless_on_u1 = {**D.CHANGES["H1"], "targets": [{**D.code_targets("H1")[0], "qualname": "track_ang_vel_z_exp"}]}
    monkeypatch.setitem(D.CHANGES, "H1", harmless_on_u1)
    with pytest.raises(ValueError, match="같은 대상"):
        D.check_catalog()
    monkeypatch.setitem(D.CHANGES, "H1", {**D.CHANGES["H2"]})  # H2와 같은 override 키
    with pytest.raises(ValueError, match="같은 대상"):
        D.check_catalog()


def test_amendments_from_the_design_draft_keep_their_reasons():
    assert [a["id"] for a in D.AMENDMENTS] == ["H1", "H4", "H6", "H2", "reserve"]
    assert all(a["draft"] and a["now"] and a["reason"] for a in D.AMENDMENTS)
    assert "track_ang_vel_z_exp" in U.SWAPS["U1"]  # H1을 옮긴 이유가 지금도 성립한다
    assert all(t["qualname"] not in U.SWAPS["U1"] for t in D.code_targets("H1"))


def test_subsets_are_parsed_in_catalog_order_with_at_most_one_harmful_change():
    assert D.parse_subset("U1, H7,H1") == ("H1", "H7", "U1")
    assert D.parse_subset("NONE") == ()
    for bad in ("", " ", "H1,H1", "H9", "U1,F1", "h1"):
        with pytest.raises(ValueError):
            D.parse_subset(bad)
    with pytest.raises(SystemExit):
        D.arm("U1,F2")


def test_hydra_overrides_and_params_keys_come_from_the_catalog():
    v2 = D.hydra_overrides(D.HARMLESS)
    assert D.hydra_overrides(D.HARMLESS_V1) == ["env.rewards.track_lin_vel_xy_exp.weight=1.4", *v2]  # v2는 H2만 뺐다
    assert v2 == [
        "env.scene.robot.actuators.base_legs.effort_limit=23.7",
        "env.scene.robot.actuators.base_legs.saturation_effort=23.7",
        "env.observations.policy.joint_vel.noise.n_min=-1.2",
        "env.observations.policy.joint_vel.noise.n_max=1.2",
        "agent.algorithm.entropy_coef=0.008"]
    assert D.hydra_overrides(("H1", "H3", "H6", "U1")) == []
    assert D.expected_params(("H7", "S2")) == {"agent.yaml:algorithm.entropy_coef": (0.01, 0.008),
                                               "agent.yaml:algorithm.learning_rate": (0.001, 0.0009)}


@needs_isaaclab
def test_replacement_bodies_keep_the_original_parameter_names_and_have_no_closure():
    for cid in D.CHANGES:
        for target in D.code_targets(cid):
            original = R._node(ast.parse(R.read_source(ISAACLAB, target["file"])), target["qualname"])
            names = tuple(a.arg for a in [*original.args.posonlyargs, *original.args.args, *original.args.kwonlyargs])
            assert U.parameters(target["replacement"].__code__) == names, (cid, target["qualname"])
            assert not target["replacement"].__code__.co_freevars


# --- 주입기(가짜 Isaac 모듈) ---
CFG_VALUES = {"rewards.track_lin_vel_xy_exp.weight": 1.4, "rewards.dof_acc_l2.weight": -3.0e-07,
              "scene.robot.actuators.base_legs.effort_limit": 23.7,
              "scene.robot.actuators.base_legs.saturation_effort": 23.7,
              "observations.policy.joint_vel.noise.n_min": -1.2, "observations.policy.joint_vel.noise.n_max": 1.2,
              "terminations.base_contact.params.threshold": 1.1}


def _env_cfg(values: dict):
    """점 경로 값 → 설정 객체. Isaac Lab처럼 actuators·params 아래는 dict, 나머지는 속성이다."""
    tree: dict = {}
    for dotted, value in values.items():
        *parents, leaf = dotted.split(".")
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value

    def build(node, key=None):
        if not isinstance(node, dict):
            return node
        children = {k: build(v, k) for k, v in node.items()}
        return children if key in ("actuators", "params") else SimpleNamespace(**children)
    return build(tree)


def _fake_isaac(monkeypatch, *, bind=None, cfg_values=None, override_actuator=False, broken_env=False):
    """대상 모듈 4개와 gymnasium을 가짜로 둔다. make는 보상 항·액추에이터·env cfg를 가진 env를 돌려준다."""
    rewards, velocity = ModuleType(D.REWARDS[0]), ModuleType(D.VELOCITY_REWARDS[0])
    signatures = {"flat_orientation_l2": "env, asset_cfg=None", "action_rate_l2": "env",
                  "track_lin_vel_xy_exp": "env, std, command_name, asset_cfg=None",
                  "track_ang_vel_z_exp": "env, std, command_name, asset_cfg=None",
                  "lin_vel_z_l2": "env, asset_cfg=None", "ang_vel_xy_l2": "env, asset_cfg=None"}
    exec("\n".join(f"def {n}({a}):\n    return 'original'" for n, a in signatures.items()), rewards.__dict__)  # noqa: S102
    exec("def feet_air_time(env, command_name, sensor_cfg, threshold):\n    return 'original'", velocity.__dict__)  # noqa: S102

    class RewardManager:
        def __init__(self, funcs):
            self._funcs, self.active_terms = funcs, list(funcs)

        def get_term_cfg(self, name):
            return SimpleNamespace(func=self._funcs[name])

        def __str__(self):
            return "original"

    class DCMotor:
        def _clip_effort(self, effort):
            return effort

    class OtherMotor(DCMotor):
        def _clip_effort(self, effort):
            return effort

    manager_module, actuator_module = ModuleType(D.REWARD_MANAGER[0]), ModuleType(D.ACTUATOR_PD[0])
    manager_module.RewardManager, actuator_module.DCMotor = RewardManager, DCMotor
    for module in (rewards, velocity, manager_module, actuator_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    env_cfg = _env_cfg({**CFG_VALUES, **(cfg_values or {})})

    def make(*_args, **_kwargs):
        if broken_env:
            return SimpleNamespace(unwrapped=SimpleNamespace())  # 보상 관리자가 없는 env
        funcs = {name: getattr(rewards, name) for name in signatures}
        funcs["feet_air_time"] = velocity.feet_air_time
        funcs.update(bind or {})
        robot = SimpleNamespace(actuators={"base_legs": (OtherMotor if override_actuator else DCMotor)()})
        return SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=RewardManager(funcs), scene={"robot": robot},
                                                         cfg=_kwargs.get("cfg", env_cfg)))
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    return gymnasium


def _lines(capsys) -> list[dict]:
    return [json.loads(line[len(D.READBACK_PREFIX):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(D.READBACK_PREFIX)]


@pytest.mark.parametrize("spec", [",".join(D.HARMLESS), ",".join(D.HARMLESS) + ",U1", ",".join(D.HARMLESS) + ",F2",
                                  "S1,S2,S3,S4,F1"])
def test_arm_swaps_once_before_the_first_env_and_prints_one_ok_readback(monkeypatch, capsys, spec):
    gymnasium = _fake_isaac(monkeypatch)
    calls = []
    original = D.before_env
    monkeypatch.setattr(D, "before_env", lambda ids: calls.append(ids) or original(ids))
    assert D.arm(spec) == D.hydra_overrides(D.parse_subset(spec))
    gymnasium.make("task")
    gymnasium.make("task")
    assert calls == [D.parse_subset(spec)]
    (line,) = _lines(capsys)
    assert line["ok"] is True and line["changes"] == list(D.parse_subset(spec)), line
    for cid in D.parse_subset(spec):
        for target in D.code_targets(cid):
            function = D._resolve(sys.modules[target["module"]], target["qualname"])
            assert function.__code__ is target["replacement"].__code__
    if "H7" in spec:
        assert line["config"]["H7"] == {"agent.algorithm.entropy_coef": "params"}
    assert H.readback_ok(line) is (spec == ",".join(D.HARMLESS))  # smoke는 무해 7개만 정확히 받아들인다


@pytest.mark.parametrize("case", ["other_function_bound", "old_env_value", "actuator_overrides_method", "broken_env"])
def test_readback_is_not_ok_when_the_change_did_not_reach_the_env(monkeypatch, capsys, case):
    spec = {"other_function_bound": "H1", "old_env_value": "H5", "actuator_overrides_method": "F2", "broken_env": "H1"}[case]
    gymnasium = _fake_isaac(monkeypatch, bind={"flat_orientation_l2": lambda env, asset_cfg=None: 0}
                            if case == "other_function_bound" else None,
                            cfg_values={"observations.policy.joint_vel.noise.n_max": 1.5} if case == "old_env_value" else None,
                            override_actuator=case == "actuator_overrides_method", broken_env=case == "broken_env")
    D.arm(spec)
    gymnasium.make("task")  # 확인이 실패해도 학습은 멈추지 않고 ok=false 줄을 남긴다
    captured = capsys.readouterr()
    (line,) = [json.loads(text[len(D.READBACK_PREFIX):]) for text in captured.out.splitlines()
               if text.startswith(D.READBACK_PREFIX)]
    assert ("[d3_changes_error]" in captured.err) is (case == "broken_env")
    assert line["ok"] is False
    if case == "other_function_bound":
        assert line["code"]["H1"]["flat_orientation_l2"] == {"swapped": True, "bound": False}
    elif case == "old_env_value":
        assert line["config"]["H5"]["env.observations.policy.joint_vel.noise.n_max"] is False
    elif case == "actuator_overrides_method":
        assert line["code"]["F2"]["DCMotor._clip_effort"] == {"swapped": True, "bound": False}
    else:
        assert line["error"] == "AttributeError"  # 공개 결과에 들어가므로 예외 문구(경로가 있을 수 있음)는 남기지 않는다


# --- 교체 본문의 의미 ---
def _bound(function, namespace):
    return FunctionType(function.__code__, namespace, function.__name__)


def test_the_d3_f2_body_equals_the_hidden_faults_f2_body(monkeypatch):
    """hidden_faults F2는 함수 안에서 정의돼 그대로 쓰지 못한다. 같은 입력에서 값이 같아야 한다."""
    actuator_module = ModuleType(D.ACTUATOR_PD[0])
    actuator_module.DCMotor = type("DCMotor", (), {})
    monkeypatch.setitem(sys.modules, D.ACTUATOR_PD[0], actuator_module)
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    HF._before_env("F2")
    reference = actuator_module.DCMotor._clip_effort
    ours = _bound(D._clip_effort_scaled, {"torch": FAKE_TORCH})
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = {"_joint_vel": rng.normal(0, 20, size=(8, 12)), "_vel_at_effort_lim": 60.0, "_saturation_effort": 23.5,
                 "velocity_limit": 30.0, "effort_limit": 23.5}
        effort = rng.normal(0, 40, size=(8, 12))
        a, b = SimpleNamespace(**{k: np.copy(v) for k, v in state.items()}), SimpleNamespace(**{k: np.copy(v) for k, v in state.items()})
        assert np.array_equal(reference(a, effort), ours(b, effort)) and np.array_equal(a._joint_vel, b._joint_vel)


@needs_isaaclab
def test_harmless_code_bodies_keep_values_except_the_registered_guard():
    namespace = {"torch": FAKE_TORCH, "SceneEntityCfg": lambda name: SimpleNamespace(name=name), "ManagerBasedRLEnv": object,
                 "RigidObject": object, "ContactSensor": object}
    originals = {}
    for cid in ("H1", "H3", "S4"):
        target = D.code_targets(cid)[0]
        node = R._node(ast.parse(R.read_source(ISAACLAB, target["file"])), target["qualname"])
        exec(compile(ast.Module(body=[node], type_ignores=[]), target["file"], "exec"), namespace)  # noqa: S102
        originals[cid] = (namespace[node.name], _bound(target["replacement"], {"torch": FAKE_TORCH}))
    rng = np.random.default_rng(1)
    gravity = rng.normal(size=(16, 3))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))},
                          action_manager=SimpleNamespace(action=rng.normal(size=(16, 12)), prev_action=rng.normal(size=(16, 12))))
    for cid in ("H1", "S4"):
        original, ours = originals[cid]
        args = (env, SimpleNamespace(name="robot")) if cid == "H1" else (env,)
        assert np.array_equal(original(*args), ours(*args)), cid

    def air_env(air):
        sensor = SimpleNamespace(compute_first_contact=lambda dt: np.ones_like(air),
                                 data=SimpleNamespace(last_air_time=air))
        return SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}), step_dt=0.02,
                               command_manager=SimpleNamespace(get_command=lambda name: np.ones((air.shape[0], 3))))
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=slice(None))
    original, ours = originals["H3"]
    walking = rng.uniform(0.0, 4.9, size=(16, 4))
    assert np.array_equal(original(air_env(walking), "base_velocity", sensor_cfg, 0.5),
                          ours(air_env(walking), "base_velocity", sensor_cfg, 0.5))
    long = walking.copy()
    long[0, 0] = 9.0
    assert ours(air_env(long), "base_velocity", sensor_cfg, 0.5)[0] < original(air_env(long), "base_velocity", sensor_cfg, 0.5)[0]


def test_the_reward_table_change_only_formats_the_weight_column(monkeypatch):
    class Table:
        def __init__(self):
            self.rows, self.align = [], {}

        def add_row(self, row):
            self.rows.append(row)

        def get_string(self):
            return repr(self.rows)
    manager = SimpleNamespace(_term_names=["a", "b"], _term_cfgs=[SimpleNamespace(weight=-2.5e-7), SimpleNamespace(weight=1.5)])
    text = _bound(D._reward_table, {"PrettyTable": Table})(manager)
    assert text == "<RewardManager> contains 2 active terms.\n" + repr([[0, "a", "-2.5e-07"], [1, "b", "1.5"]]) + "\n"


# --- diff 렌더러·canary ---
@needs_isaaclab
@pytest.mark.parametrize("cid", list(D.CHANGES))
def test_every_change_renders_a_diff_that_matches_the_executed_change_without_clues(cid):
    diff = R.render(cid, ISAACLAB)
    assert R.canary(diff) == [], cid
    removed = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    edits = [e for t in D.code_targets(cid) for e in t["edits"]] or D.CHANGES[cid]["edits"]
    assert len(removed) == len(edits) and len(added) == sum(new.count("\n") + 1 for _old, new in edits)
    files = {t["file"] for t in D.code_targets(cid)} or {D.CHANGES[cid]["file"]}
    assert {line[len("--- a/"):] for line in diff.splitlines() if line.startswith("--- a/")} == files


@needs_isaaclab
def test_a_rendered_edit_that_differs_from_the_executed_body_is_refused(monkeypatch):
    target = {**D.code_targets("H3")[0], "edits": [(D.code_targets("H3")[0]["edits"][0][0],
                                                     "last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]")]}
    monkeypatch.setitem(D.CHANGES, "H3", {**D.CHANGES["H3"], "targets": [target]})
    with pytest.raises(ValueError, match="실제 교체 본문"):
        R.render("H3", ISAACLAB)


@needs_isaaclab
def test_a_signature_edit_is_refused_because_code_swaps_keep_the_original_signature(monkeypatch):
    target = D.code_targets("H3")[0]
    edits = [*target["edits"], ("threshold: float\n)", "threshold: float = 9.0\n)")]  # 기본값은 __defaults__라 실행되지 않는다
    monkeypatch.setitem(D.CHANGES, "H3", {**D.CHANGES["H3"], "targets": [{**target, "edits": edits}]})
    with pytest.raises(ValueError, match="서명"):
        R.render("H3", ISAACLAB)


def _private_table(self):
    return self.__rows  # 클래스 밖이라 _RewardManager__rows로 바뀌지 않는다


def test_name_mangled_identifiers_are_found_for_method_replacements():
    method = ast.parse("def f(self):\n    return self.__p + self.__init__() + __q\n").body[0]
    assert R.mangled_names(method) == ["__p", "__q"]
    assert all(not R.mangled_names(R.replacement_ast(t["replacement"])) for cid in D.CHANGES for t in D.code_targets(cid))


@needs_isaaclab
def test_a_method_replacement_with_a_mangled_name_is_refused(monkeypatch):
    target = {**D.code_targets("H6")[0], "replacement": _private_table}
    monkeypatch.setitem(D.CHANGES, "H6", {**D.CHANGES["H6"], "targets": [target]})
    with pytest.raises(ValueError, match="이름이 바뀌는"):
        R.render("H6", ISAACLAB)


@needs_isaaclab
def test_a_config_edit_that_differs_from_its_override_is_refused(monkeypatch):
    monkeypatch.setitem(D.CHANGES, "H7", {**D.CHANGES["H7"], "overrides": {"agent.algorithm.entropy_coef": (0.01, 0.009)}})
    with pytest.raises(ValueError, match="override"):
        R.render("H7", ISAACLAB)
    same_numbers_other_fields = {"env.scene.robot.actuators.base_legs.velocity_limit": (23.5, 23.7),
                                 "env.scene.robot.actuators.base_legs.stiffness": (23.5, 23.7)}
    monkeypatch.setitem(D.CHANGES, "H4", {**D.CHANGES["H4"], "overrides": same_numbers_other_fields})
    with pytest.raises(ValueError, match="필드"):
        R.render("H4", ISAACLAB)


@needs_isaaclab
def test_a_config_edit_that_changes_a_number_no_override_covers_is_refused(monkeypatch):
    only_n_min = {"env.observations.policy.joint_vel.noise.n_min": (-1.5, -1.2)}  # 편집은 n_max도 바꾼다
    monkeypatch.setitem(D.CHANGES, "H5", {**D.CHANGES["H5"], "overrides": only_n_min})
    with pytest.raises(ValueError, match="숫자 변화"):
        R.render("H5", ISAACLAB)


def test_field_values_are_read_next_to_their_names():
    assert R.field_value("noise=Unoise(n_min=-1.5, n_max=1.5)", "n_max") == 1.5
    assert R.field_value('"threshold": 1.0},', "threshold") == 1.0
    assert R.field_value("self.rewards.dof_acc_l2.weight = -2.5e-7", "weight") == -2.5e-7
    assert R.field_value("saturation_effort=23.5,", "effort_limit") is None


@needs_isaaclab
@pytest.mark.parametrize("old", ["entropy_coef=0.02,", "=", "        "])
def test_an_edit_whose_original_text_is_missing_or_ambiguous_is_refused(monkeypatch, old):
    monkeypatch.setitem(D.CHANGES, "H7", {**D.CHANGES["H7"], "edits": [(old, "entropy_coef=0.008,")]})
    with pytest.raises(ValueError, match="정확히 한 번"):
        R.render("H7", ISAACLAB)


def test_the_canary_catches_planted_clues_and_ignores_the_change_itself():
    clean = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-    a = data.root_lin_vel_b\n+    a = torch.zeros_like(data.root_lin_vel_w)\n"
    assert R.canary(clean) == []
    planted = {"+    x = 1  # tidy\n": "added_comment", "+    H3 = 1\n": "id:H3", "+    fault = 1\n": "word:fault",
               "+    y = _feet_air_time\n": "name:_feet_air_time", "+    z = '결함'\n": "word:결함",
               " context mentions U1 here\n": "id:U1", "+    harmless = 1\n": "word:harm"}
    for line, label in planted.items():
        assert label in R.canary(clean + line), line
    assert R.canary(clean + "+    debug_value = 1\n") == []  # 단어 앞이 영문자면 다른 단어다


def test_number_extraction_reads_signs_but_not_identifier_digits():
    assert R.numbers("noise=Unoise(n_min=-1.5, n_max=1.5)") == [-1.5, 1.5]
    assert R.numbers("self.rewards.dof_acc_l2.weight = -2.5e-7") == [-2.5e-7]
    assert R.numbers("(1.0 - 5.0 * self._joint_vel / self.velocity_limit)") == [1.0, 5.0]
    assert R.numbers("x[0] - 2") == [0.0, 2.0]
    assert R.numbers('params={"threshold": 1.0},') == [1.0]
