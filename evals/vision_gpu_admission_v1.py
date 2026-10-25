"""Windows UTF-8 GPU 조회 어댑터. 동결 분류 규칙을 유지하며 임대 권한을 발급하지 않는다."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import record_evidence_media as FROZEN  # pyright: ignore[reportMissingImports]
from lease_timestamps import parse_lease_timestamp

SCHEMA = "vision_windows_gpu_observation_v1"
MAX_BYTES = 8 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(raw: str) -> dict:
    def pairs(values):
        value = {}
        for key, item in values:
            if key in value:
                raise ValueError("GPU JSON 중복 key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("GPU 비유한 JSON")))
    if not isinstance(value, dict):
        raise TypeError("GPU JSON 객체가 필요하다")
    return value


def _utf8(raw: bytes) -> str:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("GPU 출력 크기 또는 bytes 형식이 다르다")
    return raw.decode("utf-8-sig", errors="strict")


def _run(arguments: list[str]) -> bytes:
    return subprocess.run(arguments, check=True, capture_output=True, timeout=30,
                          creationflags=FROZEN.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout


def _now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017


def _classify(xml: str, cim: str, profile: str) -> tuple[str, dict]:
    devices = ET.fromstring(xml).findall("gpu")
    if len(devices) != 1:
        raise ValueError("Windows GPU 한 개가 필요하다")
    gpu = devices[0]
    inventories = gpu.findall("processes")
    if len(inventories) != 1 or len(gpu.findall("uuid")) != 1:
        raise ValueError("GPU UUID 또는 process inventory가 없거나 중복이다")
    inventory = inventories[0]
    if ((inventory.text or "").strip() or any(
        child.tag != "process_info" or (child.tail or "").strip() for child in inventory
    )):
        raise ValueError("GPU process inventory에 알 수 없는 내용이 있다")
    uuid = gpu.findtext("uuid")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise ValueError("Windows GPU UUID가 없다")
    classified = FROZEN.classify_gpu(xml, _object(cim), profile)
    classified.pop("observed_at_utc")
    return uuid, classified


def query_windows(profile: str = "desktop_shared_recording_v1", *,
                  run: Callable[[list[str]], bytes] = _run, clock: Callable[[], datetime] = _now) -> dict:
    """live 조회 전후 시각·원문·GPU UUID를 봉인한다. 원 SHA는 부모 원장에 보존한다."""
    started = clock()
    if started.tzinfo is None:
        raise ValueError("GPU 조회 clock에 시간대가 없다")
    xml = _utf8(run(["nvidia-smi", "-q", "-x"]))
    pids = sorted({int(row.text or "") for row in ET.fromstring(xml).findall("gpu/processes/process_info/pid")})
    script = (
        "$ErrorActionPreference='Stop'; "
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
        "$OutputEncoding=[Console]::OutputEncoding; "
        "$engines=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | "
        "Where-Object {$_.UtilizationPercentage -ge 1} | Select-Object Name,UtilizationPercentage); "
        f"$visionGpuPids=@({','.join(str(pid) for pid in pids)}); "
        "$engines | ForEach-Object {if($_.Name -match '^pid_(\\d+)_') {$visionGpuPids += [int]$Matches[1]}}; "
        "$processes=@(Get-CimInstance Win32_Process | Where-Object {$_.ProcessId -in $visionGpuPids} | "
        "Select-Object ProcessId,Name,ExecutablePath,CommandLine); "
        "ConvertTo-Json -InputObject @{processes=$processes; engines=$engines} -Depth 4 -Compress"
    )
    cim = _utf8(run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]))
    finished = clock()
    if finished.tzinfo is None or not 0 <= (finished - started).total_seconds() <= 30:
        raise ValueError("GPU 조회가 유효한30초 시간창을 벗어났다")
    uuid, classified = _classify(xml, cim, profile)
    raw = {"xml_utf8": xml, "cim_utf8": cim, "profile": profile}
    snapshot = {"schema": SCHEMA, "resource": {"kind": "local_windows", "gpu_uuid": uuid},
                "capture_started_utc": started.isoformat(), "capture_finished_utc": finished.isoformat(),
                "raw": raw, "raw_sha256": _digest(raw), "classified": classified, "lease_granted": False}
    return {"snapshot": snapshot, "sha256": _digest(snapshot)}


def verify_fresh(snapshot: dict, expected_snapshot_sha256: str, *, expected_gpu_uuid: str,
                 resource_kind: str, now: datetime, max_age_s: float = 30) -> dict:
    """부모가 보존한 조회 SHA와 현재 시각을 검사한다. Brev용 증거로 전환할 수 없다."""
    if expected_snapshot_sha256 != _digest(snapshot):
        raise ValueError("GPU 조회 원 SHA가 다르다")
    if (set(snapshot) != {"schema", "resource", "capture_started_utc", "capture_finished_utc", "raw", "raw_sha256", "classified", "lease_granted"}
            or snapshot["schema"] != SCHEMA or resource_kind != "local_windows"
            or snapshot["resource"] != {"kind": "local_windows", "gpu_uuid": expected_gpu_uuid}
            or snapshot["lease_granted"] is not False
            or now.tzinfo is None or type(max_age_s) not in (int, float) or not 0 < max_age_s <= 30):
        raise ValueError("Windows GPU 표본의 자원·시각 계약이 다르다")
    start = parse_lease_timestamp(snapshot["capture_started_utc"])
    finish = parse_lease_timestamp(snapshot["capture_finished_utc"])
    if (not 0 <= (finish - start).total_seconds() <= 30
            or not 0 <= (now - start).total_seconds() <= max_age_s
            or not 0 <= (now - finish).total_seconds() <= max_age_s):
        raise ValueError("GPU 표본이 오래됐거나 미래다")
    raw = snapshot["raw"]
    if set(raw) != {"xml_utf8", "cim_utf8", "profile"} or snapshot["raw_sha256"] != _digest(raw):
        raise ValueError("GPU 원시 표본 SHA 또는 형식이 다르다")
    uuid, classified = _classify(raw["xml_utf8"], raw["cim_utf8"], raw["profile"])
    if uuid != expected_gpu_uuid or _canonical(classified) != _canonical(snapshot["classified"]):
        raise ValueError("GPU 처리 표본이 원문·자원과 다르다")
    if classified["busy"]:
        raise ValueError("GPU 자원이 사용 중이다")
    return snapshot
