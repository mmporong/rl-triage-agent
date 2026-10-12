"""정책 없는 DC motor 토크-속도 곡선 readback 회귀."""
from __future__ import annotations

import math
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_triage import actuator_envelope as E


CONFIG = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}
SOURCE_SHA = "a" * 64


class FakeTensor:
    def __init__(self, shape, values):
        self.shape = tuple(shape)
        size = math.prod(shape)
        self.values = [float(values)] * size if isinstance(values, (int, float)) else list(values)
        assert len(self.values) == size

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, size):
        assert size == -1
        return self

    def tolist(self):
        return list(self.values)

    def clone(self):
        return FakeTensor(self.shape, self.values)

    def fill_(self, value):
        self.values[:] = [float(value)] * len(self.values)
        return self


class FakeTensorModule:
    @staticmethod
    def full_like(tensor, value):
        return FakeTensor(tensor.shape, value)


TENSORS = FakeTensorModule()


class FakeMotor:
    def __init__(self, *, factor=1.0, shape=(2, 3)):
        self.effort_limit = FakeTensor(shape, CONFIG["effort_limit"])
        self.velocity_limit = FakeTensor(shape, CONFIG["velocity_limit"])
        self._saturation_effort = CONFIG["saturation_effort"]
        self._vel_at_effort_lim = 60.0
        self._joint_vel = FakeTensor(shape, 0.0)
        self.factor = factor

    def _clip_effort(self, effort):
        output = []
        for velocity, requested, effort_limit, velocity_limit in zip(
                self._joint_vel.values, effort.values, self.effort_limit.values,
                self.velocity_limit.values, strict=True):
            velocity = max(-self._vel_at_effort_lim, min(self._vel_at_effort_lim, velocity))
            maximum = min(self._saturation_effort * (1.0 - self.factor * velocity / velocity_limit), effort_limit)
            minimum = max(self._saturation_effort * (-1.0 - self.factor * velocity / velocity_limit), -effort_limit)
            output.append(maximum if minimum > maximum else max(minimum, min(maximum, requested)))
        return FakeTensor(self._joint_vel.shape, output)


def test_stock_motor_matches_registered_documented_curve_without_gpu():
    motor = FakeMotor()
    original = motor._joint_vel
    result = E.inspect_actuator_envelope(
        motor, CONFIG, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "valid" and result["curve_matches_expected"] is True
    assert [row["velocity_rad_s"] for row in result["measurements"]] == list(E.VELOCITY_GRID)
    assert all(row["matches_expected"] for row in result["measurements"])
    assert motor._joint_vel is original and original.values == [0.0] * 6
    assert result["config_provenance"] == {
        "caller_asserted_source_sha256": SOURCE_SHA, "source_binding_verified": False,
        "expected_config": CONFIG,
        "velocity_grid_rad_s": list(E.VELOCITY_GRID),
        "runtime_config_readback": CONFIG,
    }
    assert result["requested_efforts"] == {"positive": 23_500.0, "negative": -23_500.0}
    assert result["limitations"]["policy_rollout_used"] is False
    assert result["limitations"]["cause_confirmed"] is False
    assert result["limitations"]["llm_superiority_claimed"] is False
    assert result["limitations"]["agreement_scope"] == "registered_input_grid_only"
    assert result["limitations"]["scope"] == "device_agnostic_policy_free_bounded_curve_readback"
    assert "torch" not in sys.modules


def test_f2_shaped_curve_is_measured_as_a_difference_without_cause_claim():
    result = E.inspect_actuator_envelope(
        FakeMotor(factor=5.0), CONFIG, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "valid" and result["curve_matches_expected"] is False
    rows = {row["velocity_rad_s"]: row for row in result["measurements"]}
    assert rows[3.0]["measured_max_effort"] == pytest.approx([11.75] * 6)
    assert rows[3.0]["expected_max_effort"] == pytest.approx(21.15)
    assert rows[6.0]["measured_max_effort"] == pytest.approx([0.0] * 6)
    assert rows[18.0]["measured_min_effort"] == pytest.approx([-47.0] * 6)
    assert result["limitations"]["cause_confirmed"] is False


def test_large_saturation_requests_expose_over_limit_mutant():
    class OverLimitMotor(FakeMotor):
        def _clip_effort(self, effort):
            mutant_limit = 2.0 * CONFIG["effort_limit"]
            return FakeTensor(effort.shape, [max(-mutant_limit, min(mutant_limit, value))
                                             for value in effort.values])

    result = E.inspect_actuator_envelope(
        OverLimitMotor(), CONFIG, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    zero = result["measurements"][0]
    assert result["status"] == "valid" and result["curve_matches_expected"] is False
    assert zero["measured_max_effort"] == pytest.approx([47.0] * 6)
    assert zero["measured_min_effort"] == pytest.approx([-47.0] * 6)
    assert result["requested_efforts"]["positive"] > CONFIG["effort_limit"]


def test_measurement_exception_restores_original_velocity_object_and_values():
    class RaisingMotor(FakeMotor):
        def _clip_effort(self, effort):
            self._joint_vel.fill_(9.0)
            raise RuntimeError("measurement failed")

    motor = RaisingMotor()
    original = motor._joint_vel
    original.fill_(1.25)
    saved = original.clone()
    result = E.inspect_actuator_envelope(
        motor, CONFIG, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "invalid"
    assert result["curve_matches_expected"] is None
    assert result["invalid_reason"] == "measurement_error:RuntimeError"
    assert motor._joint_vel is original and original.values == saved.values


@pytest.mark.parametrize(("motor", "config", "source_sha", "reason"), [
    (None, CONFIG, SOURCE_SHA, "missing_actuator_interface"),
    (FakeMotor(), {**CONFIG, "velocity_limit": float("nan")}, SOURCE_SHA, "invalid_expected_config"),
    (FakeMotor(), CONFIG, "bad", "invalid_config_source_sha256"),
])
def test_missing_or_nonfinite_inputs_are_invalid_not_curve_findings(motor, config, source_sha, reason):
    result = E.inspect_actuator_envelope(
        motor, config, config_source_sha256=source_sha, tensor_module=TENSORS)
    assert result["status"] == "invalid"
    assert result["curve_matches_expected"] is None
    assert result["invalid_reason"] == reason


def test_runtime_config_mismatch_is_invalid_not_a_curve_difference():
    motor = FakeMotor()
    motor._saturation_effort = 20.0
    result = E.inspect_actuator_envelope(
        motor, CONFIG, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "invalid"
    assert result["curve_matches_expected"] is None
    assert result["invalid_reason"] == "runtime_config_mismatch"


def test_request_effort_overflow_is_invalid():
    config = {"effort_limit": 1.0e308, "saturation_effort": 1.0e308, "velocity_limit": 30.0}
    motor = FakeMotor()
    motor.effort_limit = FakeTensor((2, 3), config["effort_limit"])
    motor._saturation_effort = config["saturation_effort"]
    result = E.inspect_actuator_envelope(
        motor, config, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "invalid"
    assert result["invalid_reason"] == "nonfinite_request_effort"
    assert result["requested_efforts"] is None


@pytest.mark.parametrize("bad_config", [
    {**CONFIG, "effort_limit": math.nan},
    {**CONFIG, "saturation_effort": math.inf},
    {**CONFIG, "velocity_limit": object()},
])
def test_invalid_config_metadata_is_canonical_json(bad_config):
    result = E.inspect_actuator_envelope(
        FakeMotor(), bad_config, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "invalid"
    assert result["config_provenance"]["expected_config"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_invalid_clip_outputs_are_not_reported_as_curve_differences(failure):
    class BadOutputMotor(FakeMotor):
        def _clip_effort(self, effort):
            if failure == "shape":
                return FakeTensor((1,), 0.0)
            return FakeTensor(effort.shape, math.nan)

    result = E.inspect_actuator_envelope(
        BadOutputMotor(), CONFIG, config_source_sha256=SOURCE_SHA, tensor_module=TENSORS)
    assert result["status"] == "invalid"
    assert result["curve_matches_expected"] is None
    assert result["invalid_reason"] == (
        "clip_output_shape_mismatch" if failure == "shape" else "nonfinite_clip_output")
