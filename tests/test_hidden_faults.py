"""P0-C 숨은 결함: Isaac 없이 확인할 수 있는 부분(본문 교체가 함수 정체성을 지키는지, 범주·이름 규칙)."""
import importlib.util

import pytest

from rl_triage import triage_tools as T
from test_offline_replay import ROOT


def _module():
    spec = importlib.util.spec_from_file_location("hidden_faults_under_test", ROOT / "bench" / "hidden_faults.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_fault_maps_to_one_agent_mechanism():
    H = _module()
    cats = [f["category"] for k, f in H.FAULTS.items() if k != "NONE"]
    assert sorted(cats) == sorted(T.MECHANISMS)  # 범주마다 하나
    assert H.FAULTS["NONE"]["category"] is None


def test_code_swap_keeps_identity_name_and_defaults():
    """설정은 함수 객체를 참조하고 params dump는 모듈·이름을 쓴다. 본문만 바뀌어야 diff에 드러나지 않는다."""
    H = _module()
    ns = {}
    exec("def time_out(env):\n    return env.episode_length_buf >= env.max_episode_length\n", ns)
    target = ns["time_out"]
    ident, name, module = id(target), target.__name__, target.__module__

    class Env:
        episode_length_buf, max_episode_length = 150, 1000

    assert target(Env()) is False
    H._swap_code(target, H._time_out_early)
    assert (id(target), target.__name__, target.__module__) == (ident, name, module)
    assert target(Env()) is True  # 150 >= 50 (최대 길이의 5%)
    Env.episode_length_buf = 49
    assert target(Env()) is False


def test_unknown_fault_is_rejected_before_any_patch():
    H = _module()
    with pytest.raises(SystemExit, match="모르는 TRIAGE_FAULT"):
        H.arm("F9")


def _bench_module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "bench" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_p0c_key_is_a_secret_bijection_and_plan_has_baselines():
    C = _bench_module("p0c_cases")
    a, b = C.make_key(123), C.make_key(123)
    assert a == b and sorted(a["cases"]) == [f"h{i:02d}" for i in range(1, 7)]
    assert sorted(v["fault"] for v in a["cases"].values()) == [f"F{i}" for i in range(1, 7)]
    runs = C.plan()
    assert len(runs) == 21 and sum(r["case_id"] == "baseline_p0c" for r in runs) == 3
    assert all("fault" not in r for r in runs)  # 실행 목록에는 결함 이름이 없다
