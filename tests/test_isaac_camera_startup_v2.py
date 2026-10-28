import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "bench/probe_isaac_camera_startup_v2.py"
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(SOURCE.parent), str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("probe2", SOURCE)
assert spec is not None and spec.loader is not None
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class Settings:
    def __init__(self): self.value = False
    def get_as_bool(self, _): return self.value
    def set_bool(self, _, value): self.value = value


class App:
    def __init__(self):
        self.updated = 0
        self.close_arg = None
    def update(self): self.updated += 1
    def close(self, *, wait_for_replicator): self.close_arg = wait_for_replicator


def test_terminal_policy_and_close_order_are_sealed(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    P.V1.write_new(output / "camera-job.json", {"x": 1})
    app, settings = App(), Settings()
    result = P.run_sdk(output, P.V1.LAUNCHER, app_factory=lambda **_: app, settings=settings)
    assert result["close_returned"] is True and settings.value is True and app.close_arg is False
    names = [json.loads(path.read_text())["event"] for path in sorted((output / "markers").glob("*.json"),
             key=lambda path: json.loads(path.read_text())["sequence"])]
    assert names == list(P.REQUIRED_MARKERS) + [P.OPTIONAL_MARKER]
    policy = json.loads((output / P.POLICY.SIDECAR).read_text())
    assert policy["before"] is False and policy["after"] is True and policy["wait_for_replicator"] is False


def test_close_return_marker_is_not_created_when_close_does_not_return(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    P.V1.write_new(output / "camera-job.json", {"x": 1})
    class ExitApp(App):
        def close(self, *, wait_for_replicator): raise SystemExit(0)
    try:
        P.run_sdk(output, P.V1.LAUNCHER, app_factory=lambda **_: ExitApp(), settings=Settings())
    except SystemExit as error:
        assert error.code == 0
    assert not (output / "markers/close-returned.json").exists()
    assert (output / "markers/close-entered.json").is_file()
    assert (output / P.POLICY.SIDECAR).is_file()


def test_settings_factory_runs_only_after_app_creation(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    P.V1.write_new(output / "camera-job.json", {"x": 1})
    events=[]
    app=App()
    settings=Settings()
    def app_factory(**_):
        events.append("app-created")
        return app
    def settings_factory():
        events.append("settings-acquired")
        return settings
    P.run_sdk(output,P.V1.LAUNCHER,app_factory=app_factory,settings_factory=settings_factory)
    assert events == ["app-created","settings-acquired"]
