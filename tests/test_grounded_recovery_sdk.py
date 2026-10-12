"""Grounded recovery SDK 자식의 CPU 계측 계약."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys

import pytest
import yaml


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SPEC = importlib.util.spec_from_file_location("grounded_recovery_sdk_test", ROOT / "bench/grounded_recovery_sdk.py")
assert SPEC and SPEC.loader
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


class FakeBase:
    def __init__(self):
        self.common_step_counter = 0
        self._sim_step_counter = 0
        self.reset_seen = []

    def reset(self, *, seed=None):
        self.reset_seen.append(seed)
        return "reset"


class FakeEnv:
    def __init__(self, base):
        self.unwrapped = base


class FakeWrapper:
    def __init__(self, base):
        self.unwrapped = base


def fake_step(wrapper, actions):
    wrapper.unwrapped.common_step_counter += 1
    wrapper.unwrapped._sim_step_counter += 4
    return actions


def test_fake_reset_and_step_instrumentation_exact_contract():
    base = FakeBase()
    tracker = G.FixedEvalInstrumentation(horizon=3, decimation=4)
    tracker.environment_created(FakeEnv(base))
    base.reset()
    base.reset(seed=2026)
    wrapper = FakeWrapper(base)
    assert [tracker.measured_step(fake_step, wrapper, n) for n in range(3)] == [0, 1, 2]
    clocks = tracker.verify()
    assert clocks == {"rows": _rows(), "decimation": 4, "reset_calls": 2}
    assert G.validate_clock_trace(clocks["rows"], reset_calls=clocks["reset_calls"],
                                  env_create_count=1, horizon=3) == clocks
    assert base.reset_seen == [None, 2026]


def test_instrumentation_rejects_extra_reset_environment_and_step():
    base = FakeBase()
    tracker = G.FixedEvalInstrumentation(horizon=1)
    tracker.environment_created(FakeEnv(base))
    with pytest.raises(RuntimeError, match="environment"):
        tracker.environment_created(FakeEnv(FakeBase()))
    base.reset()
    base.reset()
    with pytest.raises(RuntimeError, match="extra"):
        base.reset()
    wrapper = FakeWrapper(base)
    tracker.measured_step(fake_step, wrapper, None)
    with pytest.raises(RuntimeError, match="horizon"):
        tracker.measured_step(fake_step, wrapper, None)


def _rows(n=3):
    return [{"control_before": i, "control_after": i + 1,
             "physics_before": i * 4, "physics_after": (i + 1) * 4} for i in range(n)]


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows[1].update(control_before=0),
    lambda rows: rows[1].update(physics_after=9),
    lambda rows: rows[1].update(control_after=True),
    lambda rows: rows[1].update(extra=1),
])
def test_clock_trace_rejects_size_discontinuity_bool_and_schema(change):
    rows = _rows()
    change(rows)
    with pytest.raises(ValueError):
        G.validate_clock_trace(rows, reset_calls=2, env_create_count=1, horizon=3, decimation=4)


def test_clock_trace_rejects_wrong_reset_and_environment_counts():
    with pytest.raises(ValueError, match="reset"):
        G.validate_clock_trace(_rows(), reset_calls=1, env_create_count=1, horizon=3)
    with pytest.raises(ValueError, match="environment"):
        G.validate_clock_trace(_rows(), reset_calls=2, env_create_count=2, horizon=3)


def test_event_readback_accepts_only_upstream_reset_ranges():
    resolved = {"mode": "reset", "params": {"com_range": {
        "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
    }}}
    G.validate_event_readback(deepcopy(G.B.EXPECTED_INJECTION_READBACK), resolved)
    bad = deepcopy(resolved)
    bad["params"]["com_range"]["x"] = (-0.06, 0.06)
    with pytest.raises(ValueError, match="x range"):
        G.validate_event_readback(deepcopy(G.B.EXPECTED_INJECTION_READBACK), bad)
    with pytest.raises(ValueError, match="injection"):
        G.validate_event_readback({}, resolved)


def test_sdk_tuple_readback_is_equivalent_to_saved_json_without_mutating_values():
    raw = deepcopy(G.B.EXPECTED_INJECTION_READBACK)
    ranges = raw["changed"]["events.base_com"]["com_range"]
    for key, value in ranges.items():
        ranges[key] = tuple(value)
    original = deepcopy(raw)
    assert raw != G.B.EXPECTED_INJECTION_READBACK
    assert G.normalize_injection_readback(raw) == G.B.EXPECTED_INJECTION_READBACK
    assert raw == original


@pytest.mark.parametrize("change", [
    lambda raw: raw["checks"].update(base_com_in_reset=1),
    lambda raw: raw["changed"]["events.base_com"]["com_range"].update(x=(-0.06, 0.06)),
    lambda raw: raw["changed"]["events.base_com"]["com_range"].update(z=(float("nan"), 0.01)),
    lambda raw: raw["changed"]["events.base_com"].update(mode="startup"),
])
def test_tuple_normalization_preserves_rejection_of_changed_semantics(change):
    raw = deepcopy(G.B.EXPECTED_INJECTION_READBACK)
    change(raw)
    with pytest.raises(ValueError):
        G.normalize_injection_readback(raw)


def test_target_body_binding_joins_original_resolved_and_runtime_names():
    original = {"name": "robot", "body_names": "base", "body_ids": slice(None)}
    resolved = {"name": "robot", "body_names": ["base"], "body_ids": [1]}
    assert G.validate_target_body_binding(["trunk", "base", "foot"], original, resolved) == {
        "source": "base.scene['robot'].body_names", "asset_name": "robot",
        "body_names": ["trunk", "base", "foot"], "base_body_index": 1,
    }


@pytest.mark.parametrize("body_names,resolved_ids", [
    (["trunk", "foot"], [1]),
    (["base", "base"], [0]),
    (["trunk", "base"], []),
    (["trunk", "base"], [1, 0]),
    (["trunk", "base"], [True]),
    (["trunk", "base"], [0]),
])
def test_target_body_binding_rejects_missing_duplicate_invalid_or_wrong_id(body_names, resolved_ids):
    original = {"name": "robot", "body_names": "base", "body_ids": slice(None)}
    resolved = {"name": "robot", "body_names": ["base"], "body_ids": resolved_ids}
    with pytest.raises(ValueError):
        G.validate_target_body_binding(body_names, original, resolved)


@pytest.mark.parametrize("change", [
    {"name": "other"}, {"body_names": ["trunk"]}, {"body_names": "base"},
])
def test_target_body_binding_rejects_wrong_resolved_asset_identity(change):
    original = {"name": "robot", "body_names": "base", "body_ids": slice(None)}
    resolved = {"name": "robot", "body_names": ["base"], "body_ids": [0], **change}
    with pytest.raises(ValueError):
        G.validate_target_body_binding(["base", "foot"], original, resolved)


def test_observation_configuration_sha_uses_stored_yaml_roundtrip(tmp_path):
    path = tmp_path / "configuration.yaml"
    original = {"tuple_value": (1, 2), "slice_value": slice(None, 4, 2),
                "nested": {"enabled": True, "scale": 0.25}}
    path.write_text(yaml.dump({"observation_configuration": original}, sort_keys=True), encoding="utf-8")
    loaded = G.B._load_configuration_yaml(path)["observation_configuration"]
    assert loaded["tuple_value"] == [1, 2] and loaded["slice_value"] == [None, 4, 2]
    assert G._observation_configuration_sha(path) == G.S._canonical_sha(loaded)


def _validated(folder):
    return {
        "manifest": {"arm": "fault_before", "callback": "X1", "checkpoint_path": str(folder / "model.pt"),
                     "checkpoint_sha256": "a" * 64, "action_scale": 0.25},
        "paths": {"report_path": folder / "report.json", "body_path": folder / "body.json",
                  "configuration_path": folder / "configuration.yaml"},
        "protocol": {"device": "cuda:0", "fixed_protocol_path": G.FIXED_PROTOCOL,
                     "isaacsim_version": G.ISAACSIM_VERSION},
        "source_receipt": {"source": "ok"}, "project_commit_sha": "b" * 40,
        "protocol_sha256": "c" * 64, "manifest_sha256": "d" * 64,
    }


def test_child_started_precedes_sdk_and_success_body_is_written(tmp_path, monkeypatch):
    validated = _validated(tmp_path)
    writes = []
    fake_host = SimpleNamespace(validate_manifest=lambda path: validated,
                                write_new=lambda path, value: writes.append((Path(path).name, deepcopy(value))))
    monkeypatch.setitem(sys.modules, "run_grounded_recovery", fake_host)
    monkeypatch.setattr(G.S, "process_identity", lambda: {"process_id": 12, "process_creation_time_filetime": 34})
    app = SimpleNamespace(close=lambda: writes.append(("app_closed", {})))

    def run_sdk(_validated):
        assert writes[0][0] == "child_started.json"
        return ({"checkpoint_sha256": "a" * 64, "configuration_sha256": "e" * 64,
                 "isaacsim_version": G.ISAACSIM_VERSION,
                 "observation_configuration_sha256": "f" * 64,
                 "callback_code_identity_verified": True,
                 "injection_readback": deepcopy(G.B.EXPECTED_INJECTION_READBACK),
                 "resolved_event_readback": {"base_com": {}},
                 "target_body_readback": {"source": "base.scene['robot'].body_names", "asset_name": "robot",
                                          "body_names": ["base"], "base_body_index": 0},
                 "clocks": {"rows": _rows(1000), "decimation": 4, "reset_calls": 2},
                 "env_create_count": 1, "report_sha256": "1" * 64}, app)

    monkeypatch.setattr(G, "run_sdk", run_sdk)
    assert G.child_main(tmp_path / "manifest.json") == 0
    assert [name for name, _ in writes] == ["child_started.json", "body.json", "app_closed"]
    body = writes[1][1]
    assert body["schema"] == G.BODY_SCHEMA and body["completed"] is True
    assert body["isaacsim_version"] == G.ISAACSIM_VERSION
    assert body["cleanup_confirmed"] is False and len(body["clocks"]["rows"]) == 1000
    assert G.validate_clock_trace(body["clocks"]["rows"], reset_calls=body["clocks"]["reset_calls"],
                                  env_create_count=body["env_create_count"])["decimation"] == 4
    assert body["process_identity"] == {"process_id": 12, "process_creation_time_filetime": 34}


def test_child_failure_records_type_before_app_close_without_body(tmp_path, monkeypatch):
    validated = _validated(tmp_path)
    writes = []
    fake_host = SimpleNamespace(validate_manifest=lambda path: validated,
                                write_new=lambda path, value: writes.append((Path(path).name, deepcopy(value))))
    monkeypatch.setitem(sys.modules, "run_grounded_recovery", fake_host)
    monkeypatch.setattr(G.S, "process_identity", lambda: {"process_id": 12, "process_creation_time_filetime": 34})
    app = SimpleNamespace(close=lambda: writes.append(("app_closed", {})))
    monkeypatch.setattr(G, "run_sdk", lambda validated: (_ for _ in ()).throw(G.SdkExecutionError(ValueError("bad"), app)))
    with pytest.raises(ValueError, match="bad"):
        G.child_main(tmp_path / "manifest.json")
    assert [name for name, _ in writes] == ["child_started.json", "child_failure.json", "app_closed"]
    assert writes[1][1]["error_type"] == "ValueError"


def test_child_rejects_existing_artifact_before_started_receipt(tmp_path, monkeypatch):
    validated = _validated(tmp_path)
    (tmp_path / "jobs.json").write_text("{}", encoding="utf-8")
    fake_host = SimpleNamespace(validate_manifest=lambda path: validated,
                                write_new=lambda path, value: pytest.fail("must not write"))
    monkeypatch.setitem(sys.modules, "run_grounded_recovery", fake_host)
    with pytest.raises(FileExistsError, match="never resumed"):
        G.child_main(tmp_path / "manifest.json")


@pytest.mark.parametrize("name", ["fixed_eval.pending.tmp", "child_failure.json"])
def test_child_rejects_orphan_artifact_before_app_and_preserves_it(tmp_path, monkeypatch, name):
    validated = _validated(tmp_path)
    orphan = tmp_path / name
    orphan.write_bytes(b"preserve-orphan")
    calls = []
    fake_host = SimpleNamespace(validate_manifest=lambda path: validated,
                                write_new=lambda path, value: calls.append(Path(path).name))
    monkeypatch.setitem(sys.modules, "run_grounded_recovery", fake_host)
    monkeypatch.setattr(G, "run_sdk", lambda *_args: pytest.fail("orphan artifact launched App/GPU"))
    monkeypatch.setattr(G.S, "process_identity", lambda: pytest.fail("orphan artifact read process identity"))
    with pytest.raises(FileExistsError, match="never resumed"):
        G.child_main(tmp_path / "manifest.json")
    assert orphan.read_bytes() == b"preserve-orphan" and calls == []


def test_wrong_loaded_isaacsim_version_stops_before_fixed_gpu_evaluation(tmp_path, monkeypatch):
    validated = _validated(tmp_path)
    calls = []
    app = SimpleNamespace(close=lambda: calls.append("close"))
    fake_app_module = SimpleNamespace(AppLauncher=lambda **kwargs: SimpleNamespace(app=app))
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app_module)
    monkeypatch.setattr(G, "_loaded_isaacsim_version", lambda: "4.6.0")
    monkeypatch.setattr(G, "_execute_fixed_eval", lambda *_args: pytest.fail("version mismatch entered GPU evaluation"))
    with pytest.raises(G.SdkExecutionError, match="loaded Isaac Sim") as caught:
        G.run_sdk(validated)
    assert isinstance(caught.value.cause, ValueError) and calls == []
    caught.value.app.close()
    assert calls == ["close"]


def test_loaded_isaacsim_version_uses_runtime_components(monkeypatch):
    isaacsim = ModuleType("isaacsim")
    core = ModuleType("isaacsim.core")
    version = ModuleType("isaacsim.core.version")
    version.get_version = lambda: ("unused", "unused", 4, 5, 0, "build")
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.version", version)
    assert G._loaded_isaacsim_version() == "4.5.0"
