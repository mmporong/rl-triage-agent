"""명시적으로 호출할 때만 NVIDIA 장치 자원 샘플을 수집한다."""
from __future__ import annotations

import csv
import math
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from io import StringIO
from itertools import pairwise
from numbers import Real


class GpuSamplingError(RuntimeError):
    """NVIDIA 샘플을 안전하게 읽거나 적분할 수 없을 때 발생한다."""


QUERY_FIELDS = (
    "timestamp",
    "uuid",
    "driver_version",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
)
_UNAVAILABLE = {"", "n/a", "[n/a]", "not supported", "[not supported]", "unsupported"}


def _missing(value: str) -> bool:
    return value.strip().lower() in _UNAVAILABLE


def _text(value: str) -> str | None:
    return None if _missing(value) else value.strip()


def _number(value: str, label: str) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise GpuSamplingError(f"{label} 값을 숫자로 해석할 수 없다: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise GpuSamplingError(f"{label} 값은 유한한 음수 아닌 수여야 한다")
    return number


def sample_nvidia_gpus(
    *,
    call: Callable[..., object] = subprocess.run,
    sampled_at: str | None = None,
) -> list[dict]:
    """selective ``nvidia-smi`` query를 한 번 실행하고 장치별 행을 반환한다."""
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = call(command, capture_output=True, text=True, check=False,
                      encoding="utf-8", errors="replace", timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuSamplingError(f"nvidia-smi 실행 실패: {exc}") from exc
    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", "")
    if returncode != 0 or not isinstance(stdout, str):
        raise GpuSamplingError(f"nvidia-smi query 실패(returncode={returncode}): {stderr}")
    # nvidia-smi timestamp는 로컬 시각이며 UTC offset을 포함하지 않는다.
    # 적분 구간에는 호스트의 aware 시각을 사용하고 장치 원문은 따로 보존한다.
    observed_at = sampled_at or datetime.now(timezone.utc).isoformat()  # noqa: UP017 - SDK Python 3.10 호환
    _seconds(observed_at, "sampled_at")
    rows = list(csv.reader(StringIO(stdout)))
    samples: list[dict] = []
    for index, row in enumerate(rows):
        if not row or all(not column.strip() for column in row):
            continue
        if len(row) != len(QUERY_FIELDS):
            raise GpuSamplingError(f"nvidia-smi 행 {index}의 열 수가 {len(row)}개다")
        timestamp, uuid, driver, gpu_util, memory_util, memory_used, power = row
        samples.append({
            "sample_timestamp": observed_at,
            "device_reported_timestamp": _text(timestamp),
            "device_uuid": _text(uuid),
            "driver_version": _text(driver),
            "utilization": {
                "gpu_percent": _number(gpu_util, "utilization.gpu"),
                "memory_percent": _number(memory_util, "utilization.memory"),
            },
            "memory_used_mib": _number(memory_used, "memory.used"),
            "power_draw_w": _number(power, "power.draw"),
            "measurement_scope": {
                "power": "device_wide",
                "process_vram_attribution": "not_measured",
                "board_power_process_attribution": "not_permitted",
            },
        })
    return samples


def _seconds(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise GpuSamplingError(f"{label}이 유효한 timestamp가 아니다")
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        parsed = None
        for parser in (
            datetime.fromisoformat,
            lambda item: datetime.strptime(item, "%Y/%m/%d %H:%M:%S.%f"),
            lambda item: datetime.strptime(item, "%Y/%m/%d %H:%M:%S"),
        ):
            try:
                parsed = parser(text)
                break
            except ValueError:
                continue
        if parsed is None:
            raise GpuSamplingError(f"{label}이 유효한 timestamp가 아니다: {value!r}")
        if parsed.tzinfo is None:
            raise GpuSamplingError(f"{label}에는 명시적인 timezone offset이 필요하다")
        number = parsed.timestamp()
    else:
        raise GpuSamplingError(f"{label}이 유효한 timestamp가 아니다")
    if not math.isfinite(number):
        raise GpuSamplingError(f"{label}이 유한하지 않다")
    return number


def integrate_device_energy(
    samples: list[dict],
    *,
    window_start: object,
    window_end: object,
    max_gap_s: float,
) -> dict:
    """장치 전체 power.draw를 구간별 사다리꼴로 적분하고 coverage를 보고한다."""
    if not isinstance(samples, list):
        raise GpuSamplingError("samples는 목록이어야 한다")
    start = _seconds(window_start, "window_start")
    end = _seconds(window_end, "window_end")
    if end <= start:
        raise GpuSamplingError("window_end는 window_start보다 커야 한다")
    if isinstance(max_gap_s, bool) or not isinstance(max_gap_s, Real):
        raise GpuSamplingError("max_gap_s는 양의 유한한 수여야 한다")
    gap_limit = float(max_gap_s)
    if not math.isfinite(gap_limit) or gap_limit <= 0:
        raise GpuSamplingError("max_gap_s는 양의 유한한 수여야 한다")
    grouped: dict[str, list[tuple[float, float | None]]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise GpuSamplingError(f"samples[{index}]는 객체여야 한다")
        uuid = sample.get("device_uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise GpuSamplingError(f"samples[{index}].device_uuid가 필요하다")
        timestamp = _seconds(sample.get("sample_timestamp"), f"samples[{index}].sample_timestamp")
        raw_power = sample.get("power_draw_w")
        if raw_power is None:
            power = None
        elif isinstance(raw_power, bool) or not isinstance(raw_power, Real):
            raise GpuSamplingError(f"samples[{index}].power_draw_w가 유효하지 않다")
        else:
            power = float(raw_power)
            if not math.isfinite(power) or power < 0:
                raise GpuSamplingError(f"samples[{index}].power_draw_w가 유효하지 않다")
        if start <= timestamp <= end:
            grouped.setdefault(uuid, []).append((timestamp, power))
    devices: dict[str, dict] = {}
    window_s = end - start
    for uuid, rows in grouped.items():
        rows.sort()
        energy_watt_seconds = 0.0
        covered_s = 0.0
        gaps: list[dict] = []
        if rows[0][0] > start:
            gaps.append({"start_s": start, "end_s": rows[0][0], "reason": "window_boundary_uncovered"})
        for (left_t, left_p), (right_t, right_p) in pairwise(rows):
            interval = right_t - left_t
            if interval <= 0:
                gaps.append({"start_s": left_t, "end_s": right_t, "reason": "duplicate_or_reversed"})
            elif interval > gap_limit:
                gaps.append({"start_s": left_t, "end_s": right_t, "reason": "exceeds_max_gap"})
            elif left_p is None or right_p is None:
                gaps.append({"start_s": left_t, "end_s": right_t, "reason": "power_unsupported"})
            else:
                covered_s += interval
                energy_watt_seconds += (left_p + right_p) * 0.5 * interval
        if rows[-1][0] < end:
            gaps.append({"start_s": rows[-1][0], "end_s": end, "reason": "window_boundary_uncovered"})
        devices[uuid] = {
            "energy_wh": energy_watt_seconds / 3600 if covered_s > 0 else None,
            "scope": "device_wide_estimate",
            "window_s": window_s,
            "covered_s": covered_s,
            "coverage_fraction": covered_s / window_s,
            "max_gap_s": gap_limit,
            "gaps": gaps,
            "sample_count": len(rows),
            "process_attribution": None,
        }
    return {
        "window_start_s": start,
        "window_end_s": end,
        "devices": devices,
        "unsupported_devices_remain_null": True,
    }
