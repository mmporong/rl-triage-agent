"""명시적으로 호출할 때만 NVIDIA 장치 자원 샘플을 수집한다."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from io import StringIO
from itertools import pairwise
from numbers import Real
from pathlib import Path


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


class PeriodicDeviceSampler:
    """호출자가 시작·종료하는 장치 표본 수집기. 작업 실행 권한은 발급하지 않는다."""

    def __init__(
        self, output_root: Path, *, interval_s: float = 2.0, max_gap_s: float = 6.0,
        sample: Callable[[], list[dict]] = sample_nvidia_gpus,
        lease_check: Callable[[], dict] | None = None,
    ) -> None:
        text = str(output_root)
        if (not Path(text).is_absolute() or text.startswith("\\\\") or "://" in text
                or "ri_codex2_tokenfactory_live_160" in text.casefold()):
            raise ValueError("sampler output must be an allowed absolute local path")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value)) or value <= 0
               for value in (interval_s, max_gap_s)):
            raise ValueError("sampling interval and gap must be finite positive seconds")
        self.output_root = Path(output_root)
        self.interval_s, self.max_gap_s = float(interval_s), float(max_gap_s)
        self.sample, self.lease_check = sample, lease_check
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._result: dict | None = None
        self._active = False
        self._samples: list[dict] = []
        self._errors: list[dict] = []

    @staticmethod
    def _ref(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    @staticmethod
    def _new(path: Path, value: dict) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")

    def _lease(self) -> dict:
        if self.lease_check is not None:
            result = self.lease_check()
        else:
            from evals.lineage_source_batch_v1 import lease_context
            result = lease_context()
        if not isinstance(result, dict) or not isinstance(result.get("leaseId"), str):
            raise GpuSamplingError("sampler requires a verified external GPU lease")
        return {"lease_id": result["leaseId"],
                "verification_scope": "existing_live_lease" if self.lease_check is None else "injected_lease_reader"}

    def _collect(self) -> None:
        try:
            rows = self.sample()
            if not isinstance(rows, list):
                raise GpuSamplingError("GPU sample batch must be a list")
            rows = json.loads(json.dumps(rows, allow_nan=False))
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("device_uuid"), str):
                    raise GpuSamplingError("GPU sample device binding is missing")
                _seconds(row.get("sample_timestamp"), "sample timestamp")
            payload = {"status": "sampled", "samples": rows}
            json.dumps(payload, allow_nan=False)
            self._samples.extend(rows)
        except Exception as error:  # noqa: BLE001 - 관측 실패를 자원 사용0으로 바꾸지 않는다.
            payload = {"status": "unavailable", "error_type": type(error).__name__}
            self._errors.append(payload)
        with (self._folder / "samples.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()

    def _loop(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_s):
                self._collect()
        except Exception as error:  # noqa: BLE001 - 스레드의 기록 실패를 종료 receipt에 보존한다.
            self._errors.append({"status": "writer_unavailable", "error_type": type(error).__name__})

    def start(self) -> dict:
        if self._active:
            raise GpuSamplingError("sampler is already active")
        self._admission = self._lease()
        if self.output_root.is_symlink():
            raise GpuSamplingError("sampler output cannot be a symlink")
        self.output_root.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            folder = self.output_root / f"collection-{number:04d}"
            try:
                folder.mkdir()
                break
            except FileExistsError:
                number += 1
        self._folder = folder
        self._samples, self._errors = [], []
        self._result = None
        self._stop_event.clear()
        self._started = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - SDK Python 3.10 호환
        self._new(folder / "start.json", {
            "schema": "periodic_device_sampling_start_v1", "scope": "device_wide",
            "started_utc": self._started, "lease": self._admission,
            "interval_s": self.interval_s, "max_gap_s": self.max_gap_s,
            "reader": "nvidia_smi" if self.sample is sample_nvidia_gpus else "injected_reader",
        })
        self._active = True
        try:
            self._collect()
            self._thread = threading.Thread(target=self._loop, name="rl-triage-device-sampler", daemon=False)
            self._thread.start()
        except BaseException:
            self._active = False
            self._stop_event.set()
            raise
        return {"scope": "device_wide", "receipt_ref": self._ref(folder / "start.json")}

    def stop(self) -> dict:
        if not self._active:
            if self._result is not None:
                return self._result
            raise GpuSamplingError("sampler has not started")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=12.0)
            if self._thread.is_alive():
                raise GpuSamplingError("sampler reader did not stop within its query timeout")
        self._collect()
        ended = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - SDK Python 3.10 호환
        try:
            release = self._lease()
            lease_matches = release["lease_id"] == self._admission["lease_id"]
        except Exception as error:  # noqa: BLE001 - 이미 모은 표본은 보존하고 lease 불확실성을 표시한다.
            release, lease_matches = {"error_type": type(error).__name__}, False
        energy = integrate_device_energy(
            self._samples, window_start=self._started, window_end=ended, max_gap_s=self.max_gap_s,
        )
        device_summary = {}
        for uuid in sorted({row["device_uuid"] for row in self._samples}):
            rows = [row for row in self._samples if row["device_uuid"] == uuid]
            fields = {
                "memory_used_mib_peak": [row.get("memory_used_mib") for row in rows],
                "power_draw_w_peak": [row.get("power_draw_w") for row in rows],
                "gpu_util_percent_sample_mean": [row.get("utilization", {}).get("gpu_percent")
                                                  for row in rows],
            }
            metrics = {}
            for name, values in fields.items():
                valid = [float(value) for value in values if isinstance(value, Real)
                         and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0]
                metrics[name] = {"value": ((sum(valid) / len(valid) if name.endswith("sample_mean")
                                            else max(valid)) if valid else None),
                                 "valid_samples": len(valid), "missing_or_invalid_samples": len(values) - len(valid)}
            device_summary[uuid] = {"scope": "whole_device_including_other_loads", "metrics": metrics}
        result = {
            "schema": "periodic_device_sampling_stop_v1", "scope": "device_wide",
            "started_utc": self._started, "ended_utc": ended,
            "sample_count": len(self._samples), "errors": self._errors,
            "start_ref": self._ref(self._folder / "start.json"),
            "samples_ref": self._ref(self._folder / "samples.jsonl"),
            "release": release, "lease_matches": lease_matches,
            "energy": energy, "device_summary": device_summary,
            "process_gpu_seconds": None, "provider_gpu_seconds": None,
        }
        self._new(self._folder / "stop.json", result)
        self._result = {"scope": "device_wide", "receipt_ref": self._ref(self._folder / "stop.json")}
        self._active = False
        return self._result
