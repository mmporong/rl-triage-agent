from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import anymal_lstm_motor_semantics_v2 as S

ISAACLAB = Path.home() / "IsaacLab"
HAS_SOURCES = (ISAACLAB / "source/isaaclab/isaaclab/actuators/actuator_net.py").is_file()


def parsed_function(statement: str) -> ast.FunctionDef:
    return ast.parse(f"def f(self, value):\n    {statement}\n").body[0]


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("self._joint_vel = value", True),
        ("self._joint_vel[:] = value", True),
        ("self._joint_vel[0] += value", True),
        ("self._joint_vel.copy_(value)", True),
        ("self._joint_vel.fill_(0)", True),
        ("self._joint_vel.zero_()", True),
        ("value = self._joint_vel", False),
        ("self._joint_vel_history[:] = value", False),
    ],
)
def test_write_detector_covers_mutation_shapes(statement: str, expected: bool):
    assert S.self_attribute_written(parsed_function(statement), "_joint_vel") is expected


def test_v2_discloses_post_result_validator_revision():
    protocol = json.loads(S.PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered_revalidation_after_validator_defect"
    assert protocol["predictions"]["semantic_gate"] == "not_applicable"
    assert protocol["validator_revision"]["evidence_class"].startswith("v1 결과를 본 뒤")
    assert "독립 재현" in protocol["claims_not_supported"]
    assert protocol["api_calls"] == 0 and protocol["gpu_calls"] == 0


def valid_structural() -> dict:
    return {
        "inherits_dc_motor": True,
        "calls_inherited_clip": True,
        "dc_init_zeros_clip_joint_velocity": True,
        "dc_compute_writes_clip_joint_velocity": True,
        "lstm_compute_writes_clip_joint_velocity": False,
        "network_input_uses_joint_velocity": True,
        "anymal_c_legs_use_anydrive_3_lstm": True,
        "sdk_nonself_joint_velocity_access_count": 0,
    }


def valid_grid() -> dict:
    return {
        "zero_velocity_equal": True,
        "nonzero_difference_count": 1,
        "restore_equal": True,
        "max_abs_difference": 1.0,
    }


def test_decision_accepts_only_confirmed_missing_lstm_update():
    assert S.decide_outcome(valid_structural(), valid_grid()) == "not_applicable"
    structural = valid_structural()
    structural["lstm_compute_writes_clip_joint_velocity"] = True
    assert S.decide_outcome(structural, valid_grid()) == "applicable"


@pytest.mark.parametrize(
    ("kind", "key", "value"),
    [
        ("structural", "inherits_dc_motor", False),
        ("structural", "calls_inherited_clip", False),
        ("structural", "dc_init_zeros_clip_joint_velocity", False),
        ("structural", "dc_compute_writes_clip_joint_velocity", False),
        ("structural", "sdk_nonself_joint_velocity_access_count", 1),
        ("grid", "zero_velocity_equal", False),
        ("grid", "nonzero_difference_count", 0),
        ("grid", "restore_equal", False),
        ("grid", "max_abs_difference", 0.0),
    ],
)
def test_unknown_or_broken_contract_is_invalid_not_not_applicable(kind: str, key: str, value):
    structural = valid_structural()
    grid = valid_grid()
    (structural if kind == "structural" else grid)[key] = value
    with pytest.raises(ValueError):
        S.decide_outcome(structural, grid)


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_real_dc_compute_is_positive_and_lstm_compute_is_negative_control():
    paths = S.source_paths(ISAACLAB)
    pd_tree = ast.parse(paths["actuator_pd"].read_text(encoding="utf-8"))
    net_tree = ast.parse(paths["actuator_net"].read_text(encoding="utf-8"))
    dc = S.class_node(pd_tree, "DCMotor")
    lstm = S.class_node(net_tree, "ActuatorNetLSTM")
    assert S.self_attribute_initialized_with_zeros(S.function_node(dc.body, "__init__"), "_joint_vel")
    assert S.self_attribute_written(S.function_node(dc.body, "compute"), "_joint_vel")
    assert not S.self_attribute_written(S.function_node(lstm.body, "compute"), "_joint_vel")


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_anymal_c_asset_binds_legs_to_anydrive_lstm():
    tree = ast.parse(S.source_paths(ISAACLAB)["anymal_asset"].read_text(encoding="utf-8"))
    assert S.anymal_c_uses_lstm(tree)
    assert S.literal_config(tree) == {
        "saturation_effort": 120.0,
        "effort_limit": 80.0,
        "velocity_limit": 7.5,
    }


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_sdk_scope_is_bound_and_has_no_nonself_joint_velocity_access():
    protocol = json.loads(S.PROTOCOL.read_text(encoding="utf-8"))
    files = S.sdk_python_files(ISAACLAB)
    assert len(files) == protocol["sdk_scope"]["python_file_count"]
    assert S.sdk_manifest_sha256(ISAACLAB) == protocol["sdk_scope"]["python_manifest_sha256"]
    accesses = S.sdk_joint_vel_accesses(ISAACLAB)
    assert any(row["class"] == "DCMotor" and row["function"] == "compute" for row in accesses)
    assert not [row for row in accesses if row["base"] != "self"]


def test_public_v2_files_have_no_user_path_or_forbidden_word():
    windows_user_root = "C:" + "\\Users\\"
    for path in (S.PROTOCOL, ROOT / "bench/anymal_lstm_motor_semantics_v2.py"):
        text = path.read_text(encoding="utf-8")
        assert windows_user_root not in text
        assert "직" + "접" not in text
