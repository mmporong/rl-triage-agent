from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]
W = importlib.import_module("lineage_gpu_wait_v1")


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def lease(identifier="11111111-1111-1111-1111-111111111111"):
    return {"owner": W.B.OWNER, "state": "running", "leaseId": identifier}


def sample(*, busy=False, reason="unapproved_gpu_process:3084"):
    return {"admission_policy": W.GPU.GPU_PROFILE, "busy": busy,
            "busy_reasons": [reason] if busy else [], "compute_process_count": 0}


def invoke(stage, values, **kwargs):
    clock = Clock()
    rows = iter(values)
    return W.await_clear(stage, query=lambda: next(rows), lease_check=lease,
                         clock=clock, sleep=clock.sleep, **kwargs)


def test_busy_then_two_clear_observations_preserves_original_policy_and_raw_samples(tmp_path):
    original = W.GPU.policy()
    result = invoke(tmp_path, [sample(busy=True), sample(), sample()])
    assert result["status"] == "clear" and result["clear_streak"] == 2
    assert result["elapsed_s"] == 4 and result["gpu_jobs_started"] == 0
    assert result["training_retry"] is False and len(result["samples"]) == 3
    root = tmp_path / "gpu-wait"
    assert json.loads((root / result["samples"][0]["path"]).read_text())["busy"] is True
    assert json.loads((root / "intent.json").read_text())["admission"] == original
    assert W.GPU.policy() == original


def test_intervening_busy_resets_clear_streak(tmp_path):
    result = invoke(tmp_path, [sample(), sample(busy=True), sample(), sample()])
    assert result["status"] == "clear" and len(result["samples"]) == 4


def test_persistent_busy_times_out_without_sdk_or_training_retry(tmp_path):
    with pytest.raises(W.GpuWaitError, match="상한"):
        invoke(tmp_path, [sample(busy=True)] * 3, max_wait_s=6)
    result = json.loads((tmp_path / "gpu-wait/result.json").read_text())
    assert result["status"] == "timeout" and result["elapsed_s"] == 6
    assert result["gpu_jobs_started"] == 0 and len(result["samples"]) == 3
    assert result["training_retry"] is False


def test_second_clear_arriving_after_deadline_is_not_accepted(tmp_path):
    clock = Clock()
    calls = 0

    def query():
        nonlocal calls
        calls += 1
        clock.value += 4
        return sample()

    with pytest.raises(W.GpuWaitError, match="상한"):
        W.await_clear(tmp_path, max_wait_s=6, interval_s=1, query=query,
                      lease_check=lease, clock=clock, sleep=clock.sleep)
    result = json.loads((tmp_path / "gpu-wait/result.json").read_text())
    assert calls == 2 and result["status"] == "timeout" and result["clear_streak"] == 2


def test_lease_change_during_query_blocks_admission_and_preserves_error(tmp_path):
    clock = Clock()
    checks = 0

    def changing_lease():
        nonlocal checks
        checks += 1
        return lease() if checks < 3 else lease("22222222-2222-2222-2222-222222222222")

    with pytest.raises(W.GpuWaitError, match="lease"):
        W.await_clear(tmp_path, query=lambda: sample(), lease_check=changing_lease,
                      clock=clock, sleep=clock.sleep)
    result = json.loads((tmp_path / "gpu-wait/result.json").read_text())
    assert result["status"] == "error" and result["gpu_jobs_started"] == 0


@pytest.mark.parametrize("bad", [
    {**sample(), "compute_process_count": 1},
    {**sample(), "admission_policy": "other-profile"},
    {**sample(), "busy_reasons": ["still blocked"]},
    {**sample(), "busy": None},
])
def test_unverified_or_contradictory_clear_is_never_accepted(tmp_path, bad):
    with pytest.raises(W.GpuWaitError):
        invoke(tmp_path, [bad])
    assert json.loads((tmp_path / "gpu-wait/result.json").read_text())["status"] == "error"


@pytest.mark.parametrize("bounds", [
    {"max_wait_s": 601}, {"max_wait_s": float("inf")}, {"max_wait_s": True},
    {"interval_s": 0}, {"interval_s": 31}, {"consecutive_clear": 1},
])
def test_bounds_fail_before_new_output_or_lease(tmp_path, bounds):
    with pytest.raises(ValueError):
        invoke(tmp_path, [], **bounds)
    assert not (tmp_path / "gpu-wait").exists()


def test_existing_wait_result_is_not_overwritten(tmp_path):
    invoke(tmp_path, [sample(), sample()])
    before = (tmp_path / "gpu-wait/result.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(tmp_path, [sample(), sample()])
    assert (tmp_path / "gpu-wait/result.json").read_bytes() == before
