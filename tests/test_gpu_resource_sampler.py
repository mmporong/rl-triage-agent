import subprocess
from types import SimpleNamespace

import pytest

from rl_triage.gpu_resource_sampler import (
    GpuSamplingError,
    integrate_device_energy,
    sample_nvidia_gpus,
)


def test_selective_query_parses_supported_and_unsupported_values() -> None:
    calls = []

    def fake_call(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "2026/11/04 12:00:00.000, GPU-a, 591.00, 73, 21, 1024, 120.5\n"
                "N/A, GPU-b, 591.00, N/A, [Not Supported], 2048, [N/A]\n"
            ),
            stderr="",
        )

    samples = sample_nvidia_gpus(call=fake_call, sampled_at="2026-11-04T12:00:00Z")

    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=timestamp,uuid,driver_version,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["timeout"] == 10
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["errors"] == "replace"
    assert samples[0]["power_draw_w"] == 120.5
    assert samples[0]["measurement_scope"]["power"] == "device_wide"
    assert samples[1]["sample_timestamp"] == "2026-11-04T12:00:00Z"
    assert samples[1]["utilization"] == {"gpu_percent": None, "memory_percent": None}
    assert samples[1]["power_draw_w"] is None


def test_local_device_clock_is_preserved_without_shifting_the_utc_window() -> None:
    result = SimpleNamespace(returncode=0, stdout=(
        "2026/11/04 08:18:52.791, GPU-a, 610.62, 4, 2, 1578, 42.52\n"
    ), stderr="")
    sample = sample_nvidia_gpus(call=lambda *_a, **_k: result,
        sampled_at="2026-11-03T23:18:52.791+00:00")[0]
    assert sample["device_reported_timestamp"] == "2026/11/04 08:18:52.791"
    second = {**sample, "sample_timestamp": "2026-11-03T23:18:53.791+00:00"}
    integrated = integrate_device_energy([sample, second],
        window_start=sample["sample_timestamp"], window_end=second["sample_timestamp"], max_gap_s=2)
    assert integrated["devices"]["GPU-a"]["coverage_fraction"] == 1
    assert integrated["devices"]["GPU-a"]["energy_wh"] == pytest.approx(42.52 / 3600)
    with pytest.raises(GpuSamplingError, match="timezone"):
        integrate_device_energy([sample], window_start="2026/11/04 08:18:52.791",
            window_end="2026/11/04 08:18:53.791", max_gap_s=2)


def test_driver_query_timeout_is_a_recordable_sampling_failure() -> None:
    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 10)
    with pytest.raises(GpuSamplingError, match="실행 실패"):
        sample_nvidia_gpus(call=timeout)


def test_energy_integration_reports_gaps_and_never_attributes_process_power() -> None:
    samples = [
        {"sample_timestamp": 0.0, "device_uuid": "GPU-a", "power_draw_w": 100.0},
        {"sample_timestamp": 2.0, "device_uuid": "GPU-a", "power_draw_w": 200.0},
        {"sample_timestamp": 9.0, "device_uuid": "GPU-a", "power_draw_w": 300.0},
        {"sample_timestamp": 10.0, "device_uuid": "GPU-a", "power_draw_w": None},
        {"sample_timestamp": 0.0, "device_uuid": "GPU-b", "power_draw_w": None},
        {"sample_timestamp": 2.0, "device_uuid": "GPU-b", "power_draw_w": None},
    ]

    report = integrate_device_energy(samples, window_start=0.0, window_end=10.0, max_gap_s=3.0)

    gpu_a = report["devices"]["GPU-a"]
    assert gpu_a["energy_wh"] == pytest.approx(300.0 / 3600)
    assert gpu_a["covered_s"] == 2.0
    assert gpu_a["coverage_fraction"] == 0.2
    assert [gap["reason"] for gap in gpu_a["gaps"]] == ["exceeds_max_gap", "power_unsupported"]
    assert gpu_a["process_attribution"] is None
    assert report["devices"]["GPU-b"]["energy_wh"] is None


def test_bad_cli_shape_and_invalid_energy_window_are_rejected() -> None:
    result = SimpleNamespace(returncode=0, stdout="only,two\n", stderr="")
    with pytest.raises(GpuSamplingError, match="열 수"):
        sample_nvidia_gpus(call=lambda *_args, **_kwargs: result)
    with pytest.raises(GpuSamplingError, match="window_end"):
        integrate_device_energy([], window_start=2, window_end=1, max_gap_s=1)


def test_energy_window_boundary_gaps_are_explicit() -> None:
    report = integrate_device_energy(
        [
            {"sample_timestamp": 2.0, "device_uuid": "GPU-a", "power_draw_w": 100.0},
            {"sample_timestamp": 8.0, "device_uuid": "GPU-a", "power_draw_w": 100.0},
        ],
        window_start=0.0,
        window_end=10.0,
        max_gap_s=10.0,
    )
    device = report["devices"]["GPU-a"]
    assert device["coverage_fraction"] == 0.6
    assert [gap["reason"] for gap in device["gaps"]] == [
        "window_boundary_uncovered",
        "window_boundary_uncovered",
    ]
