from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_triage.gpu_resource_sampler import GpuSamplingError, PeriodicDeviceSampler


def lease() -> dict:
    return {"leaseId": "cpu-test-lease"}


def sample() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - SDK Python 3.10 호환
    return [{"device_uuid": "fixture-device", "sample_timestamp": now,
             "power_draw_w": 30.0, "memory_used_mib": None,
             "utilization": {"gpu_percent": None, "memory_percent": None}}]


def read_ref(ref: dict) -> dict:
    path = Path(ref["path"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    return json.loads(raw)


def test_periodic_collection_stops_and_seals_raw_refs_with_energy_scope(tmp_path: Path) -> None:
    sampled_again = threading.Event()
    calls = 0

    def reader() -> list[dict]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            sampled_again.set()
        return sample()

    sampler = PeriodicDeviceSampler(tmp_path, interval_s=0.01, max_gap_s=2,
                                    sample=reader, lease_check=lease)
    start = sampler.start()
    try:
        assert sampled_again.wait(timeout=2)
    finally:
        stop = sampler.stop()
    assert sampler._thread is not None and not sampler._thread.is_alive()
    assert read_ref(start["receipt_ref"])["reader"] == "injected_reader"
    report = read_ref(stop["receipt_ref"])
    assert report["sample_count"] >= 3 and report["lease_matches"] is True
    assert report["errors"] == [] and report["scope"] == "device_wide"
    device = report["energy"]["devices"]["fixture-device"]
    assert device["energy_wh"] > 0 and 0 < device["coverage_fraction"] <= 1
    assert device["process_attribution"] is None
    metrics = report["device_summary"]["fixture-device"]["metrics"]
    assert metrics["memory_used_mib_peak"]["value"] is None
    assert metrics["gpu_util_percent_sample_mean"]["value"] is None
    assert metrics["power_draw_w_peak"]["value"] == 30.0
    assert report["process_gpu_seconds"] is None and report["provider_gpu_seconds"] is None
    assert sampler.stop() == stop
    assert read_ref(stop["receipt_ref"]) == report


def test_reader_failure_remains_unavailable_instead_of_zero_energy(tmp_path: Path) -> None:
    def unavailable() -> list[dict]:
        raise GpuSamplingError("unsupported")

    sampler = PeriodicDeviceSampler(tmp_path, sample=unavailable, lease_check=lease)
    sampler.start()
    report = read_ref(sampler.stop()["receipt_ref"])
    assert report["sample_count"] == 0 and len(report["errors"]) >= 2
    assert report["energy"]["devices"] == {}
    assert report["process_gpu_seconds"] is None


def test_external_lease_change_is_preserved_without_discarding_samples(tmp_path: Path) -> None:
    checks = iter([{"leaseId": "first"}, {"leaseId": "changed"}])
    sampler = PeriodicDeviceSampler(tmp_path, sample=sample, lease_check=lambda: next(checks))
    sampler.start()
    report = read_ref(sampler.stop()["receipt_ref"])
    assert report["lease_matches"] is False and report["sample_count"] >= 2
    assert report["release"]["lease_id"] == "changed"


def test_restart_allocates_new_collection_and_preserves_old_bytes(tmp_path: Path) -> None:
    sampler = PeriodicDeviceSampler(tmp_path, sample=sample, lease_check=lease)
    with pytest.raises(GpuSamplingError, match="not started"):
        sampler.stop()
    sampler.start()
    try:
        with pytest.raises(GpuSamplingError, match="already active"):
            sampler.start()
    finally:
        first = sampler.stop()
    first_raw = Path(first["receipt_ref"]["path"]).read_bytes()
    sampler.start()
    second = sampler.stop()
    assert first != second
    assert Path(first["receipt_ref"]["path"]).read_bytes() == first_raw
    assert read_ref(first["receipt_ref"])["start_ref"] != read_ref(second["receipt_ref"])["start_ref"]


def test_invalid_lease_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "uncreated"
    sampler = PeriodicDeviceSampler(output, sample=sample, lease_check=dict)
    with pytest.raises(GpuSamplingError, match="external GPU lease"):
        sampler.start()
    assert not output.exists()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), True])
def test_invalid_interval_is_rejected(tmp_path: Path, interval) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        PeriodicDeviceSampler(tmp_path, interval_s=interval)
