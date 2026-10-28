"""공급자 관찰 시각과 현재 임대 입장 시각을 분리하는 capture unit.

기존 v2의 canonical 잠금·자식 실행을 유지한다. provider 원문의 신선도는 별도로 검사하고,
자원 proof는 GPU query 뒤의 실제 입장 시각에서 활성 임대를 검증한다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import agentic_play_job_v2 as J
import run_agentic_capture_unit_v1 as V1
import run_agentic_capture_unit_v2 as V2

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = J.REQUIRED_SOURCE_FILES | {
    "bench/run_agentic_capture_unit_v3.py", "tests/test_agentic_capture_unit_v3.py",
}
_LOCK = threading.RLock()
_BASE_ADMISSION = V1.admission


def _now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Isaac Python 3.10


def _provider_time(value: str, current: datetime) -> datetime:
    observed = J._utc(value, "provider observed")
    if not timedelta(0) <= current - observed <= timedelta(seconds=J.FRESH_CAPTURE_S):
        raise ValueError("provider 원문 관찰 시각이 미래이거나 신선도 밖이다")
    return observed


def _protocol(path: Path) -> dict:
    value = J.validate_protocol(path, repo=ROOT)
    if not SOURCE_FILES <= value["source_sha256_lf"].keys():
        raise ValueError("capture unit v3 source closure가 없다")
    return value


def _archive_lease_raw(folder: Path, lease_reference: dict, snapshot: dict) -> None:
    paths = J.lease_paths(snapshot["resource_id"])
    for name, path in {
        "proc-locks.txt": Path("/proc/locks"),
        "owner-stat.txt": Path(f"/proc/{snapshot['owner']['pid']}/stat"),
        "heartbeat.json": paths["heartbeat"],
        "lease-snapshot.json": Path(lease_reference["path"]),
    }.items():
        (folder / name).write_bytes(path.read_bytes())


def _query(command):
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="strict", timeout=30, check=False)


def admission(protocol_path: Path, folder: Path, provider: Path, provider_observed: str,
              lease_reference: dict) -> Path:
    protocol = _protocol(protocol_path)
    provider_at = _provider_time(provider_observed, _now())
    provider_sha = J._sha(provider)
    folder.mkdir(parents=True, exist_ok=False)
    query_receipts = {}
    for stem, command in (("gpu", J.REMOTE.GPU_COMMAND), ("process", J.REMOTE.PROCESS_COMMAND)):
        started = _now()
        J.verify_lease(lease_reference, protocol=protocol, protocol_sha=J._sha(protocol_path, lf=True), now=started)
        result = _query(command)
        finished = _now()
        (folder / f"{stem}.csv").write_text(result.stdout, encoding="utf-8", newline="")
        (folder / f"{stem}.stderr").write_text(result.stderr, encoding="utf-8", newline="")
        receipt = {"command": command, "exit_code": result.returncode,
                   "started_at_utc": started.isoformat(), "observed_at_utc": finished.isoformat(),
                   "stdout_sha256": J._sha(folder / f"{stem}.csv")}
        J.write_new(folder / f"{stem}.receipt.json", receipt)
        query_receipts[stem] = receipt
        if result.returncode or result.stderr:
            raise RuntimeError("GPU 입장 query 실패")
    admitted = _now()
    _provider_time(provider_observed, admitted)
    if J._sha(provider) != provider_sha:
        raise ValueError("provider 원문이 query 동안 바뀌었다")
    J.verify_lease(lease_reference, protocol=protocol, protocol_sha=J._sha(protocol_path, lf=True), now=admitted)
    snapshot = J._read(Path(lease_reference["path"]))
    expires = min(J._utc(protocol["resources"]["deadline_utc"], "deadline"),
                  J._utc(snapshot["expires_at_utc"], "lease expires"), admitted + timedelta(seconds=120))
    proof = J.build_resource_proof(protocol_path, provider_inventory_path=provider,
        nvidia_gpu_csv_path=folder / "gpu.csv", nvidia_process_csv_path=folder / "process.csv",
        observed_at_utc=admitted.isoformat(), expires_at_utc=expires.isoformat(),
        lease_reference=lease_reference, repo=ROOT)
    path = folder / "resource.json"
    J.write_new(path, proof)
    _archive_lease_raw(folder, lease_reference, snapshot)
    J.write_new(folder / "admission-times.json", {
        "schema": "agentic_capture_admission_times_v1",
        "provider_observed_at_utc": provider_at.isoformat(), "provider_inventory_sha256": provider_sha,
        "lease_acquired_at_utc": snapshot["acquired_at_utc"], "query_receipts": query_receipts,
        "admitted_at_utc": admitted.isoformat(), "resource_sha256": J._sha(path),
        "lease": lease_reference, "provider_max_age_s": J.FRESH_CAPTURE_S,
        "resource_observed_time_means": "joint_admission_check_after_gpu_queries",
    })
    return path


@contextmanager
def _admission_scope():
    with _LOCK:
        if V1.admission is not _BASE_ADMISSION:
            raise RuntimeError("capture admission scope가 이미 교체됐다")
        V1.admission = admission
        try:
            yield
        finally:
            V1.admission = _BASE_ADMISSION


def execute(args: argparse.Namespace) -> dict:
    if V1.os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    _protocol(args.protocol)
    with _admission_scope():
        return V2.execute(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["state-smoke", "record"], required=True)
    for name in ("protocol", "provider-inventory", "isaaclab", "python", "stage"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--provider-observed-at", required=True)
    parser.add_argument("--logical-id", required=True)
    print(json.dumps(execute(parser.parse_args()), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
