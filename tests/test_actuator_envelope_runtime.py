from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
for directory in (ROOT / "src", ROOT / "bench"):
    sys.path.insert(0, str(directory))
SPEC = importlib.util.spec_from_file_location(
    "actuator_envelope_runtime_test", ROOT / "bench/actuator_envelope_runtime.py",
)
assert SPEC and SPEC.loader
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def curve(matches=True):
    rows = []
    for velocity in B.E.VELOCITY_GRID:
        minimum, maximum = B.E._documented_limits(velocity, CONFIG)
        rows.append({
            "velocity_rad_s": velocity,
            "expected_min_effort": minimum, "expected_max_effort": maximum,
            "measured_min_effort": [minimum] * 12,
            "measured_max_effort": [maximum if matches else maximum - 1.0] * 12,
            "matches_expected": matches,
        })
    return {
        "schema": "actuator_envelope_readback_v1", "status": "valid",
        "curve_matches_expected": matches, "invalid_reason": None,
        "requested_efforts": {"positive": 23500.0, "negative": -23500.0},
        "measurements": rows, "config_provenance": {
            "expected_config": CONFIG, "runtime_config_readback": CONFIG,
            "velocity_grid_rad_s": list(B.E.VELOCITY_GRID), "source_binding_verified": False,
            "caller_asserted_source_sha256": "a" * 64,
        }, "limitations": dict(B.E.LIMITATIONS),
    }


CONFIG = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_live_asset_hash_matches_ast_binding_with_platform_line_endings(tmp_path, newline):
    import check_correction_cpu as cpu

    path = tmp_path / B.GO2_SOURCE
    path.parent.mkdir(parents=True)
    source = b"# Go2 motor\nUNITREE_GO2_CFG = DCMotorCfg(\n effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0\n)\n"
    path.write_bytes(source.replace(b"\n", newline))
    expected = cpu._go2_motor_binding(tmp_path)["source_sha256"]
    assert B._asset_source_sha(path) == expected
    path.write_bytes(path.read_bytes().replace(b"# Go2 motor", b"# altered source"))
    assert B._asset_source_sha(path) != expected


@pytest.mark.parametrize("matches", [True, False])
def test_curve_contract_accepts_complete_signed_13_by_12_grid(matches):
    assert B.validate_curve(curve(matches), CONFIG) == []


@pytest.mark.parametrize("mutation, reason", [
    (lambda value: value["measurements"].pop(), "measurement_row_count_mismatch"),
    (lambda value: value["measurements"][2]["measured_min_effort"].pop(),
     "measurement_2_measured_min_effort_invalid"),
    (lambda value: value["measurements"][1]["measured_max_effort"].__setitem__(0, float("nan")),
     "measurement_1_measured_max_effort_invalid"),
    (lambda value: value["requested_efforts"].update(positive=-23500.0),
     "requested_effort_sign_or_magnitude_mismatch"),
])
def test_curve_contract_fails_closed_on_partial_nonfinite_or_wrong_sign(mutation, reason):
    value = curve()
    mutation(value)
    assert reason in B.validate_curve(value, CONFIG)


def test_helper_invalid_stays_distinct_from_curve_difference():
    invalid = {
        "schema": "actuator_envelope_readback_v1", "status": "invalid",
        "curve_matches_expected": None, "invalid_reason": "runtime_config_mismatch",
        "requested_efforts": None, "measurements": [], "config_provenance": {},
        "limitations": dict(B.E.LIMITATIONS),
    }
    assert B.validate_curve(invalid, CONFIG) == []
    assert invalid["curve_matches_expected"] is None
    assert curve(False)["status"] == "valid"
    assert curve(False)["curve_matches_expected"] is False


def test_restoration_requires_identity_contents_and_both_counters():
    before = {"schema": "actuator_envelope_state_v1",
              "buffers": {name: {"identity": index + 1, "shape": [1, 12], "values": [0.0] * 12}
                          for index, name in enumerate(B.STATE_BUFFER_FIELDS)},
              "common_step_counter": 4, "sim_step_counter": 8}
    assert B._restoration(before, deepcopy(before))["all_restored"]
    after = deepcopy(before)
    after["buffers"]["_joint_vel"]["identity"] = 2
    assert not B._restoration(before, after)["all_restored"]
    after = deepcopy(before)
    after["buffers"]["_joint_vel"]["values"][0] = 1.0
    assert not B._restoration(before, after)["all_restored"]
    after = deepcopy(before)
    after["sim_step_counter"] += 1
    assert not B._restoration(before, after)["all_restored"]


def test_runtime_config_rejects_nonuniform_or_nonfinite_live_values():
    class Scalar:
        def __init__(self, values): self.values = values
        def detach(self): return self
        def cpu(self): return self
        def reshape(self, *_): return self
        def tolist(self): return self.values

    actuator = type("A", (), {})()
    actuator.effort_limit = Scalar([23.5] * 12)
    actuator._saturation_effort = Scalar([23.5] * 12)
    actuator.velocity_limit = Scalar([30.0] * 12)
    assert B._runtime_config(actuator) == CONFIG
    actuator.velocity_limit = Scalar([30.0] * 11 + [-30.0])
    with pytest.raises(ValueError, match="finite and uniform"):
        B._runtime_config(actuator)


def test_source_ref_binds_file_and_code_not_caller_assertion(tmp_path):
    module_path = tmp_path / "bound.py"
    module_path.write_text("def target(value):\n    return value + 1\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("bound_runtime_method", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = B._source_ref(module.target)
    assert receipt["path"] == str(module_path.resolve())
    assert len(receipt["sha256"]) == 64 and len(receipt["code_sha256"]) == 64


def test_recursive_code_fingerprint_is_stable_without_repr_addresses():
    source = "def outer():\n    def inner(value):\n        return lambda: value + 1\n    return inner\n"
    first = compile(source, "first.py", "exec", dont_inherit=True)
    second = compile(source, "second.py", "exec", dont_inherit=True)
    assert B._code_sha(first) == B._code_sha(second)
    canonical = B._canonical_code(first)
    assert "0x" not in str(canonical)


@pytest.mark.parametrize("mutate, expected", [
    (lambda state: state.clear(), "state_schema_keys_mismatch"),
    (lambda state: state["buffers"].clear(), "state_buffer_set_mismatch"),
    (lambda state: state["buffers"].pop("applied_effort"), "state_buffer_set_mismatch"),
    (lambda state: state["buffers"]["_joint_vel"].update(shape=[12]), "state__joint_vel_tensor_invalid"),
    (lambda state: state.update(common_step_counter=True), "state_common_step_counter_invalid"),
    (lambda state: state["buffers"]["computed_effort"]["values"].__setitem__(0, float("inf")),
     "state_computed_effort_tensor_invalid"),
])
def test_state_schema_fails_closed_for_empty_missing_shape_bool_and_nonfinite(mutate, expected):
    state = {"schema": "actuator_envelope_state_v1",
             "buffers": {name: {"identity": index + 1, "shape": [1, 12], "values": [0.0] * 12}
                         for index, name in enumerate(B.STATE_BUFFER_FIELDS)},
             "common_step_counter": 0, "sim_step_counter": 0}
    mutate(state)
    assert expected in B.validate_state_snapshot(state)


def test_curve_rejects_finite_garbage_bool_and_forged_aggregate():
    garbage = curve(True)
    for row in garbage["measurements"]:
        row["measured_min_effort"] = [999.0] * 12
        row["measured_max_effort"] = [999.0] * 12
        row["matches_expected"] = True
    assert any("matches_expected_mismatch" in reason for reason in B.validate_curve(garbage, CONFIG))

    aggregate = curve(True)
    aggregate["curve_matches_expected"] = False
    assert "curve_matches_expected_aggregate_mismatch" in B.validate_curve(aggregate, CONFIG)

    disguised = curve(True)
    disguised["measurements"][0]["measured_min_effort"][0] = True
    assert "measurement_0_measured_min_effort_invalid" in B.validate_curve(disguised, CONFIG)


def test_curve_rejects_expected_limit_row_and_helper_contract_tamper():
    value = curve(True)
    value["measurements"][0]["expected_max_effort"] += 1.0
    value["limitations"] = {**B.E.LIMITATIONS, "scope": "forged"}
    value["extra"] = True
    reasons = B.validate_curve(value, CONFIG)
    assert "measurement_0_expected_limit_mismatch" in reasons
    assert "helper_limitations_mismatch" in reasons
    assert "helper_keyset_mismatch" in reasons


def test_method_transition_rejects_each_partial_or_cross_signal_change(tmp_path):
    original = tmp_path / "original.py"
    patched = tmp_path / "patched.py"
    original.write_text("original\n", encoding="utf-8")
    patched.write_text("patched\n", encoding="utf-8")
    original_sha, patched_sha = B.S._sha256_file(original), B.S._sha256_file(patched)
    contract = {key: {"before": {"path": str(original), "source_sha256": original_sha,
                                  "code_sha256": "0" * 64},
                      "patched": {"path": str(patched), "source_sha256": patched_sha,
                                  "code_sha256": "1" * 64}}
                for key in B.TARGET_METHODS}
    before = {key: {"identity": index + 1, "path": str(original), "sha256": original_sha,
                    "code_sha256": "0" * 64} for index, key in enumerate(B.TARGET_METHODS)}
    after = deepcopy(before)
    after["F2"] = {"identity": 100, "path": str(patched), "sha256": patched_sha,
                   "code_sha256": "1" * 64}
    assert B.validate_method_transition(before, after, "F2", contract) == []

    identity_only = deepcopy(before)
    identity_only["F2"]["identity"] = 100
    assert "method_F2_after_binding_mismatch" in B.validate_method_transition(
        before, identity_only, "F2", contract)
    code_only = deepcopy(before)
    code_only["F2"]["code_sha256"] = "1" * 64
    assert "method_F2_patch_transition_mismatch" in B.validate_method_transition(
        before, code_only, "F2", contract)
    cross_signal = deepcopy(after)
    cross_signal["F3"]["identity"] = 200
    assert "method_F3_unexpected_cross_signal" in B.validate_method_transition(
        before, cross_signal, "F2", contract)


def test_success_publish_occurs_only_after_both_closes(monkeypatch, tmp_path):
    events = []

    class Resource:
        def __init__(self, name, error=None): self.name, self.error = name, error
        def close(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    monkeypatch.setattr(B.S, "_publish_bytes_no_overwrite", lambda *_: events.append("publish"))
    monkeypatch.setattr(B.gc, "collect", lambda: events.append("gc"))
    B._publish_success(tmp_path / "body.json", {"ok": True}, {"env": Resource("env")}, Resource("app"))
    assert events == ["env", "gc", "app", "publish"]

    events.clear()
    with pytest.raises(RuntimeError, match="close failed"):
        B._publish_success(tmp_path / "body.json", {"ok": True},
                           {"env": Resource("env", RuntimeError("close failed"))}, Resource("app"))
    assert events == ["env", "gc", "app"]

    events.clear()
    with pytest.raises(RuntimeError, match="SystemExit: 0"):
        B._publish_success(tmp_path / "body.json", {"ok": True}, {"env": Resource("env")},
                           Resource("app", SystemExit(0)))
    assert events == ["env", "gc", "app"]


@pytest.mark.parametrize("outcome", ["readback_error", "nan", "object", "success"])
def test_child_releases_sdk_and_environment_before_plugin_shutdown(monkeypatch, tmp_path, outcome):
    from types import ModuleType
    import weakref
    import run_actuator_envelope_runtime as parent

    events, references = [], []
    target = tmp_path / "body.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    validated = {"paths": {"output_path": target}, "protocol": {"scope": "test"},
                 "manifest": {"condition": "NONE"}, "protocol_sha256": "a" * 64,
                 "manifest_path": manifest, "project_commit_sha": "b" * 40,
                 "source_receipt": {}, "request_lineage": {}}
    monkeypatch.setattr(parent, "validate_manifest", lambda _: validated)
    monkeypatch.setattr(B.S, "process_identity", lambda: {})
    sdk = ModuleType("isaaclab.app")
    sdk.AppLauncher = object
    monkeypatch.setitem(sys.modules, "isaaclab.app", sdk)

    class RetainedResource:
        pass

    class Env:
        def __init__(self):
            self.cycle = self
            references.append(weakref.ref(self))
        def close(self):
            events.append("env")

    class App:
        def close(self):
            assert all(reference() is None for reference in references)
            events.append("app")

    def fail_after_environment_creation(validated, lifecycle):
        resource = RetainedResource()
        resource.cycle = resource
        references.append(weakref.ref(resource))
        lifecycle["env"] = Env()
        if outcome == "readback_error":
            raise ValueError("SDK readback failed")
        result = {"measurement_status": "valid", "protocol_status": "expected"}
        if outcome in {"nan", "object"}:
            result["bad"] = float("nan") if outcome == "nan" else object()
        return result, lifecycle["env"]

    monkeypatch.setattr(B, "_launch_app", lambda _: App())
    monkeypatch.setattr(B, "run_sdk", fail_after_environment_creation)
    if outcome == "success":
        assert B.child_main(manifest) == 0
    else:
        with pytest.raises(RuntimeError, match="actuator inspection failed"):
            B.child_main(manifest)
    assert events == ["env", "app"]
    assert target.exists() is (outcome == "success")


@pytest.mark.parametrize("report", [{"ok": True}, {"bad": float("nan")}, {"bad": object()}])
def test_success_or_serialization_failure_releases_environment_before_app_close(tmp_path, report):
    import weakref

    events, references = [], []
    class Env:
        def __init__(self):
            self.cycle = self
            references.append(weakref.ref(self))
        def close(self):
            events.append("env")
    class App:
        def close(self):
            assert references[0]() is None
            events.append("app")
    holder = {"env": Env()}
    target = tmp_path / "body.json"
    if "bad" in report:
        with pytest.raises((TypeError, ValueError)):
            B._publish_success(target, report, holder, App())
        assert not target.exists()
    else:
        B._publish_success(target, report, holder, App())
        assert target.exists()
    assert holder == {} and events == ["env", "app"]


def test_app_launcher_disables_fast_shutdown():
    observed = {}

    class Launcher:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.app = object()

    assert B._launch_app(Launcher) is not None
    assert observed == {"headless": True, "fast_shutdown": False}
