"""교정 epoch별 남은 F1은 동결 probe로, R1은 별도 3-update 계측으로 검사한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import residual_faults_v1 as FAULTS  # noqa: E402
import residual_training_v1 as TRAINING  # noqa: E402

PROBES = ROOT / "evals" / "probes.py"
TRAINING_ENTRY = ROOT / "bench" / "residual_training_v1.py"


def remaining_from_manifest(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    data = TRAINING.validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    return data, FAULTS.remaining_faults(data["original_faults"], data["applied_corrections"])


def validate_legacy_jobs(path: Path, remaining: tuple[str, ...]) -> dict[str, Any]:
    jobs = json.loads(path.read_text(encoding="utf-8"))
    expected = "F1" if "F1" in remaining else "NONE"
    if jobs.get("fault") != expected:
        raise ValueError(f"동결 probe jobs fault는 남은 F1 상태에 따라 {expected}여야 한다")
    if jobs.get("task", TRAINING.TASK) != TRAINING.TASK:
        raise ValueError("동결 probe task가 고정 Go2 task와 다르다")
    return jobs


def run_legacy(manifest: Path, passthrough: list[str]) -> None:
    data, remaining = remaining_from_manifest(manifest)
    TRAINING.verify_source_binding(data)
    if "--jobs" not in passthrough:
        raise ValueError("동결 probe에는 --jobs가 필요하다")
    jobs_path = Path(passthrough[passthrough.index("--jobs") + 1]).resolve()
    validate_legacy_jobs(jobs_path, remaining)
    original = sys.argv[:]
    try:
        sys.argv = [str(PROBES), *passthrough]
        runpy.run_path(str(PROBES), run_name="__main__")
    finally:
        sys.argv = original


def run_optimizer(manifest: Path) -> dict[str, Any]:
    data, remaining = remaining_from_manifest(manifest)
    if data["mode"] != "inspection":
        raise ValueError("optimizer residual 검사는 inspection manifest만 허용한다")
    process = subprocess.run(
        [sys.executable, "-B", str(TRAINING_ENTRY), "--manifest", str(manifest.resolve())],
        check=False, text=True, encoding="utf-8", errors="replace",
    )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    records_path = Path(data["records_path"])
    readback_path = Path(data["readback_path"])
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    report = TRAINING.OBSERVE.assemble_inspection(rows, TRAINING.observer_manifest(data))
    expected = "abnormal" if "R1" in remaining else "normal"
    if report.get("status") != expected or report.get("records") != 3:
        raise ValueError("optimizer residual 3-update 판정이 남은 R1 상태와 다르다")
    receipt = json.loads(readback_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "residual_training_readback_v1"
            or receipt.get("remaining_faults") != list(remaining)):
        raise ValueError("optimizer residual readback이 manifest 계산과 다르다")
    return {"schema": "residual_optimizer_probe_v1", "status": expected,
            "remaining_faults": list(remaining), "inspection": report,
            "readback_path": str(readback_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="교정 epoch별 남은 결함 검사")
    parser.add_argument("mode", choices=("legacy", "optimizer"))
    parser.add_argument("--manifest", required=True, type=Path)
    args, rest = parser.parse_known_args(argv)
    passthrough = rest[rest.index("--") + 1:] if "--" in rest else rest
    try:
        if args.mode == "legacy":
            run_legacy(args.manifest.resolve(), passthrough)
        else:
            if passthrough:
                raise ValueError("optimizer 모드는 추가 인자를 받지 않는다")
            run_optimizer(args.manifest.resolve())
    except (ValueError, FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
