# pyright: reportMissingImports=false
"""D4 processed GPU receipt의 clear flag 위조를 CPU fixture로 거부한다."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import selection_comparison_gpu_v1 as G  # noqa: E402

PID = 1320
IDENTITY = {"ProcessId": PID, "CreationDate": "2026-10-23T00:00:00.0000000Z", "SessionId": 1}


def clear(*, processes=None, engines=None, compute=0):
    return {
        "busy": False,
        "busy_reasons": [],
        "admission_policy": G.EXECUTION_V3.GPU_PROFILE,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "utilization_percent": 0,
        "memory_used_mib": 100,
        "compute_process_count": compute,
        "processes": processes or [],
        "active_engines": engines or [],
    }


def process(pid, name, *, process_type="C+G", known=False):
    return {"pid": pid, "name": name, "type": process_type, "known_workload": known}


def engine(pid, name, *, engine_type="3D", utilization=1):
    return {
        "name": f"pid_{pid}_luid_0x0_phys_0_eng_0_engtype_{engine_type}",
        "utilization_percent": utilization,
        "process_name": name,
    }


def transient():
    reasons = [f"unapproved_active_engine:{PID}:3D", f"unverified_active_engine_process:{PID}"]
    return {
        **clear(engines=[engine(PID, "csrss.exe")]),
        "busy": True,
        "busy_reasons": reasons,
    }


def receipt(samples, identities=None, *, reason=None, accepted=True):
    identities = identities or []
    reason = reason or ("initial_clear" if len(samples) == 1 else "two_consecutive_strict_clear")
    bounded = {
        "attempted": len(samples) > 1,
        "accepted": accepted,
        "reason": reason,
        "interval_s": 2,
        "max_samples": 3,
        "sample_count": len(samples),
        "required_consecutive_clear": 2,
        "identities": identities,
    }
    return {**samples[-1], "bounded_resample": bounded, "samples": samples}


def test_valid_empty_and_allowlisted_display_initial_clear():
    assert G.verify_gpu_receipt(receipt([clear()])) == "initial_clear"
    chrome = clear(
        processes=[process(5, "chrome.exe")],
        engines=[engine(5, "chrome.exe", engine_type="Copy")],
    )
    assert G.verify_gpu_receipt(receipt([chrome])) == "initial_clear"


@pytest.mark.parametrize("name", ["kit.exe", "isaac-sim.exe", "cuda-worker.exe", "ollama.exe"])
def test_initial_clear_flag_cannot_hide_registered_workload_names(name):
    forged = clear(processes=[process(5, name)], engines=[engine(5, name)])
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(receipt([forged]))


def test_followup_clear_flag_cannot_hide_kit_payload():
    forged = clear(processes=[process(5, "kit.exe")], engines=[engine(5, "kit.exe")])
    value = receipt([transient(), forged, clear()], [IDENTITY, IDENTITY, IDENTITY])
    assert G.FROZEN_EXPORT._validate_gpu_receipt(value) == "resampled_clear"
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(value)


def test_valid_csrss_three_sample_receipt():
    value = receipt([transient(), clear(), clear()], [IDENTITY, IDENTITY, IDENTITY])
    assert G.verify_gpu_receipt(value) == "resampled_clear"


def test_csrss_exception_does_not_hide_other_first_sample_kit_process():
    first = transient()
    first["processes"] = [process(5, "kit.exe")]
    value = receipt([first, clear(), clear()], [IDENTITY, IDENTITY, IDENTITY])
    assert G.FROZEN_EXPORT._validate_gpu_receipt(value) == "resampled_clear"
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(value)


def test_pid_mismatch_between_engine_and_process_is_rejected():
    forged = clear(processes=[process(5, "chrome.exe")], engines=[engine(6, "chrome.exe")])
    with pytest.raises(ValueError, match="PID"):
        G.verify_gpu_receipt(receipt([forged]))


def test_known_workload_true_with_clear_flag_is_rejected():
    forged = clear(processes=[process(5, "chrome.exe", known=True)], compute=1)
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(receipt([forged]))


def test_compute_count_nonzero_with_clear_flag_is_rejected():
    with pytest.raises(ValueError, match="compute process"):
        G.verify_gpu_receipt(receipt([clear(compute=1)]))


def test_disallowed_type_and_engine_type_are_rejected():
    wrong_process = clear(processes=[process(5, "chrome.exe", process_type="C")])
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(receipt([wrong_process]))
    wrong_engine = clear(
        processes=[process(5, "chrome.exe")],
        engines=[engine(5, "chrome.exe", engine_type="Compute")],
    )
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(receipt([wrong_engine]))


def test_frozen_bounded_structure_and_identity_validation_remain_required():
    value = receipt([transient(), clear(), clear()], [IDENTITY, {**IDENTITY, "SessionId": 2}, IDENTITY])
    with pytest.raises(ValueError):
        G.verify_gpu_receipt(value)
