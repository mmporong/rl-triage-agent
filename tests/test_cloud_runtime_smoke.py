"""CPU lifecycle regressions; actual cloud runtime is checked separately."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("step_error", [False, True])
@pytest.mark.parametrize("close_exits", [False, True])
def test_cloud_smoke_unsubscribes_context_before_app_close(monkeypatch, tmp_path, capsys, step_error, close_exits):
    spec = importlib.util.spec_from_file_location(
        "cloud_runtime_smoke_test", Path(__file__).parents[1] / "bench/cloud_runtime_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    events = []

    class SimulationCfg:
        def __init__(self, *, dt):
            assert dt == 0.01

    class SimulationContext:
        instance_active = False
        callbacks_active = False

        def __init__(self, _cfg):
            type(self).instance_active = type(self).callbacks_active = True
            events.append("context")

        def reset(self):
            events.append("reset")

        def step(self):
            events.append("step")
            if step_error:
                raise RuntimeError("injected step failure")

        def clear_all_callbacks(self):
            type(self).callbacks_active = False
            events.append("callbacks_cleared")

        @classmethod
        def clear_instance(cls):
            assert not cls.callbacks_active
            cls.instance_active = False
            events.append("instance_cleared")

    class App:
        def close(self):
            # The SDK's live STOP subscription can re-enter rendering during stage close.
            assert not SimulationContext.instance_active
            assert not SimulationContext.callbacks_active
            events.append("app_closed")
            if close_exits:
                raise SystemExit(0)

    class AppLauncher:
        def __init__(self, *, headless):
            assert headless
            self.app = App()

    app_module, sim_module = ModuleType("isaaclab.app"), ModuleType("isaaclab.sim")
    setattr(app_module, "AppLauncher", AppLauncher)
    setattr(sim_module, "SimulationCfg", SimulationCfg)
    setattr(sim_module, "SimulationContext", SimulationContext)
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)
    monkeypatch.setattr(smoke, "version", lambda _name: "test-only")
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["cloud_runtime_smoke.py", "--out", str(report)])
    if close_exits:
        with pytest.raises(SystemExit) as error:
            smoke.main()
        assert error.value.code == 0
    elif step_error:
        with pytest.raises(RuntimeError, match="injected step failure"):
            smoke.main()
    else:
        smoke.main()
    if step_error:
        assert not report.exists()
        assert "RL_TRIAGE_CLOUD_SMOKE_ERROR RuntimeError" in capsys.readouterr().out
    else:
        assert json.loads(report.read_text(encoding="utf-8"))["completed"] is True
        assert "before_app_close" in json.loads(report.read_text(encoding="utf-8"))["cpu_scope"]
    assert events == ["context", "reset", "step", "callbacks_cleared", "instance_cleared", "app_closed"]
