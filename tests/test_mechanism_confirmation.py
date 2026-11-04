from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from rl_triage import mechanism_confirmation as C

ROOT = Path(__file__).parents[1]
for directory in (ROOT / "src", ROOT / "bench"):
    sys.path.insert(0, str(directory))
SPEC = importlib.util.spec_from_file_location(
    "mechanism_confirmation_actuator_runtime", ROOT / "bench/actuator_envelope_runtime.py",
)
assert SPEC and SPEC.loader
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)

CONFIG = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}
SOURCE = {"kind": "cpu_development_fixture", "identifier": "unit", "sha256": "a" * 64}


def curve(matches: bool) -> dict:
    rows = []
    for velocity in B.E.VELOCITY_GRID:
        minimum, maximum = B.E._documented_limits(velocity, CONFIG)
        measured_maximum = maximum if matches else maximum - 1.0
        rows.append({
            "velocity_rad_s": velocity,
            "expected_min_effort": minimum,
            "expected_max_effort": maximum,
            "measured_min_effort": [minimum] * 12,
            "measured_max_effort": [measured_maximum] * 12,
            "matches_expected": matches,
        })
    return {
        "schema": "actuator_envelope_readback_v1",
        "status": "valid",
        "curve_matches_expected": matches,
        "invalid_reason": None,
        "requested_efforts": {"positive": 23500.0, "negative": -23500.0},
        "measurements": rows,
        "config_provenance": {
            "expected_config": CONFIG,
            "runtime_config_readback": CONFIG,
            "velocity_grid_rad_s": list(B.E.VELOCITY_GRID),
            "source_binding_verified": False,
            "caller_asserted_source_sha256": "a" * 64,
        },
        "limitations": dict(B.E.LIMITATIONS),
    }


def checked(probe: str, raw: dict, verifier=lambda _raw, _source: ()):
    return C.verify_probe_result(probe, raw, verifier=verifier, source=SOURCE)


def all_hypotheses() -> list[str]:
    return C.mechanism_registry()["mechanisms"]


def test_registry_makes_torque_symptom_only_and_registers_separate_causes() -> None:
    registry = C.mechanism_registry()
    assert set(registry["probes"][C.TORQUE_PROBE]["expect"].values()) == {None}
    assert registry["probes"][C.ACTUATOR_PROBE]["expect"]["actuator"] == "abnormal"
    assert registry["probes"][C.COUNT_PROBE]["expect"][C.COUNT_CAUSE] == "abnormal"
    assert registry["probes"][C.GRADIENT_PROBE]["expect"][C.GRADIENT_CAUSE] == "abnormal"
    assert C.selector_probe_metadata()[C.TORQUE_PROBE]["role"] == "symptom_only"


def test_runtime_profile_exposes_only_supported_tools_and_keeps_unsupported_causes() -> None:
    full = C.mechanism_registry()
    assert full == C.mechanism_registry(profile="full")
    runtime = C.mechanism_registry(profile="runtime_confirmation_v1")
    expected = [C.COUNT_PROBE, C.GRADIENT_PROBE, C.ACTUATOR_PROBE]
    assert set(runtime["probes"]) == set(expected)
    assert runtime["fixed_order"] == expected
    assert runtime["mechanisms"] == full["mechanisms"]
    remaining = list(runtime["mechanisms"])
    for probe in expected:
        remaining, _ = C.registry_tools.update(runtime, remaining, probe, "normal")
    assert remaining and "actuator" not in remaining
    assert C.COUNT_CAUSE not in remaining and C.GRADIENT_CAUSE not in remaining
    assert C.registry_tools._status(runtime, remaining, dict.fromkeys(expected, "normal"), 0) == "unidentifiable"
    with pytest.raises(ValueError, match="profile"):
        C.mechanism_registry(profile="arbitrary")


@pytest.mark.parametrize(("matches", "expected"), [(True, None), (False, "actuator")])
def test_real_actuator_curve_validator_separates_normal_mapping_and_f2_curve(
    matches: bool, expected: str | None,
) -> None:
    result = checked(
        C.ACTUATOR_PROBE,
        curve(matches),
        lambda raw, source: (
            B.validate_curve(raw, CONFIG)
            if source["sha256"] == raw["config_provenance"]["caller_asserted_source_sha256"]
            else ["source_binding_mismatch"]
        ),
    )
    decision = C.confirm_registered_cause(C.mechanism_registry(), all_hypotheses(), [result])
    assert decision["confirmed_cause"] == expected
    assert decision["observations"][0]["outcome"] == ("normal" if matches else "abnormal")
    if matches:
        assert "actuator" not in decision["remaining_hypotheses"]
        assert decision["status"] == "unresolved"
        assert "모든 actuator" in decision["observations"][0]["claim_limit"]


def test_ppo_inverse_ratio_with_normal_mapping_confirms_surrogate_not_actuator() -> None:
    mapping = checked(
        C.ACTUATOR_PROBE, curve(True),
        lambda raw, _source: B.validate_curve(raw, CONFIG),
    )
    count = checked(C.COUNT_PROBE, {"count_status": "normal"})
    gradient = checked(C.GRADIENT_PROBE, {"gradient_status": "abnormal"})
    decision = C.confirm_registered_cause(
        C.mechanism_registry(), all_hypotheses(), [mapping, count, gradient],
    )
    assert decision["status"] == "confirmed"
    assert decision["confirmed_cause"] == C.GRADIENT_CAUSE
    assert decision["remaining_hypotheses"] == [C.GRADIENT_CAUSE]


def test_count_fault_confirms_existing_update_count_hypothesis() -> None:
    decision = C.confirm_registered_cause(
        C.mechanism_registry(), all_hypotheses(),
        [checked(C.COUNT_PROBE, {"count_status": "abnormal"})],
    )
    assert decision["confirmed_cause"] == C.COUNT_CAUSE


def test_unknown_and_torque_abnormal_never_confirm_actuator() -> None:
    invalid = checked(
        C.ACTUATOR_PROBE, curve(True),
        lambda _raw, _source: ("source_binding_mismatch",),
    )
    torque = checked(C.TORQUE_PROBE, {"outcome": "abnormal"})
    decision = C.confirm_registered_cause(
        C.mechanism_registry(), all_hypotheses(), [invalid, torque],
    )
    assert decision["status"] == "unresolved"
    assert decision["confirmed_cause"] is None
    assert decision["observations"][0]["outcome"] == "unknown"
    assert decision["remaining_hypotheses"] == all_hypotheses()


def test_multiple_abnormal_optimizer_symptoms_do_not_force_single_cause() -> None:
    count = checked(C.COUNT_PROBE, {"count_status": "abnormal"})
    gradient = checked(C.GRADIENT_PROBE, {"gradient_status": "abnormal"})
    decision = C.confirm_registered_cause(C.mechanism_registry(), all_hypotheses(), [count, gradient])
    assert decision["status"] == "none_supported"
    assert decision["confirmed_cause"] is None


def test_reference_mismatch_and_tampering_are_true_unknown_boundaries() -> None:
    raw = {"count_status": "abnormal", "reference_id": "wrong"}
    result = checked(
        C.COUNT_PROBE,
        raw,
        lambda value, source: (
            () if value.get("reference_id") == source.get("reference_id") else ("reference_mismatch",)
        ),
    )
    assert C.normalize_observation(result)["outcome"] == "unknown"
    with pytest.raises(TypeError, match="issued"):
        C.normalize_observation({"probe": C.COUNT_PROBE, "count_status": "abnormal"})  # type: ignore[arg-type]

    result.raw["count_status"] = "normal"
    with pytest.raises(ValueError, match="changed"):
        C.normalize_observation(result)

    fresh = checked(C.COUNT_PROBE, {"count_status": "normal"})
    fresh.source["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="changed"):
        C.normalize_observation(fresh)


def test_verification_preserves_source_without_accepting_caller_verified_flag() -> None:
    source = {**SOURCE, "verified": True, "lineage": {"receipt_sha256": "b" * 64}}
    result = C.verify_probe_result(
        C.GRADIENT_PROBE,
        {"gradient_status": "abnormal"},
        verifier=lambda _raw, _source: (),
        source=source,
    )
    observation = C.normalize_observation(result)
    assert observation["source"] == source
    assert observation["source_sha256"]
    assert observation["outcome"] == "abnormal"
    copied = deepcopy(source)
    copied["lineage"]["receipt_sha256"] = "c" * 64
    assert observation["source"] != copied


def test_truthy_verifier_return_cannot_be_used_as_proof() -> None:
    with pytest.raises(TypeError, match="sequence"):
        C.verify_probe_result(
            C.COUNT_PROBE,
            {"count_status": "abnormal"},
            verifier=lambda _raw, _source: True,  # type: ignore[arg-type,return-value]
            source=SOURCE,
        )
