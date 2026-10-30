"""분리된 repo/runtime/package source namespace로 fresh restart를 실행한다."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from local_f1_restart_base_v3 import (
    ROOT,
    binding,
    judge,
    package_snapshot,
    read,
    runtime_snapshot,
    sha,
    source_map,
    write_new,
)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import brev_recovery_worker_v2 as GENERIC_V2
import brev_recovery_worker_v3 as GENERIC_V3
import local_f1_restart_worker_v2 as V2
import residual_training_v1 as TRAINING

from rl_triage.correction_ledger import CorrectionLedger

CHILD = ROOT / "bench/local_f1_restart_child_v2.py"
GENERIC = ROOT / "bench/brev_recovery_worker_v3.py"
PROTOCOL = ROOT / "bench/protocols/local_f1_restart_s3221_v3.json"
F1_SHA256 = "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731"


def build_jobs(
    auth_path: Path, stage: Path, lab: Path, source_commit: str, isaac_python: Path
) -> tuple[dict, dict]:
    auth, repo_sources = V2.authorization(auth_path), source_map()
    protocol = read(PROTOCOL)
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                str(protocol.get("registration_anchor")),
                source_commit,
            ],
            check=False,
        ).returncode
        == 0
    )
    if (
        protocol.get("schema") != "local_f1_restart_preregistration_v3"
        or protocol.get("status") != "registered"
        or not ancestor
        or protocol.get("timeouts_s")
        != {"train": 6000, "evaluate": 900, "total": 7200, "cleanup": 30}
        or protocol.get("source_fault_checkpoint_sha256") != F1_SHA256
        or protocol.get("reference_checkpoint_sha256")
        != auth["none_checkpoint"]["sha256"]
    ):
        raise ValueError("registered v3 protocol/source provenance 불일치")
    runtime = protocol["runtime"]
    runtime_sources = runtime_snapshot(
        lab, runtime["isaaclab_head"], runtime["source_sha256_lf"]
    )
    package = protocol["python_package"]
    package_sources = package_snapshot(
        isaac_python,
        package["distribution"],
        package["version"],
        package["source_sha256_lf"],
    )
    common = {
        "schema": "brev_recovery_worker_job_v1",
        "stage": str(stage.resolve()),
        "lab": str(lab.resolve()),
        "repo": str(ROOT.resolve()),
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256_lf": sha(PROTOCOL, lf=True),
        "source_commit_sha": source_commit,
        "source_sha256_lf": repo_sources,
        "runtime_source_sha256_lf": runtime_sources,
        "python_package_source_sha256_lf": package_sources,
    }
    train_dir = stage / "train"
    train = {
        **common,
        "mode": "train",
        "arm": "CORRECTED",
        "seed": 3221,
        "run_name": "local_f1_restart_s3221_v3",
        "iterations": 300,
        "manifest_path": str(train_dir / "manifest.json"),
        "records_path": str(train_dir / "records.jsonl"),
        "readback_path": str(train_dir / "readback.json"),
        "telemetry_path": str(train_dir / "telemetry.json"),
        "source_checkpoint_sha256": F1_SHA256,
    }
    evaluate = {
        **common,
        "mode": "evaluate",
        "artifacts": {
            "NONE": {"checkpoint": auth["none_checkpoint"]},
            "CORRECTED": {
                "checkpoint": {
                    "path": str(train_dir / "model_299.pt"),
                    "sha256": "PENDING",
                }
            },
        },
        "fixed_eval_protocol": auth["fixed_eval"]["path"],
        "reports_dir": str(stage / "fixed-eval"),
        "jobs_path": str(stage / "fixed-eval-jobs.json"),
    }
    GENERIC_V3.validate_job(train)
    manifest = GENERIC_V2._residual_manifest(
        train,
        mode="train",
        manifest_path=Path(train["manifest_path"]),
        records_path=Path(train["records_path"]),
        readback_path=Path(train["readback_path"]),
    )
    TRAINING.verify_source_binding(manifest, runtime_root=lab)
    return train, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--lab", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    train, evaluate = build_jobs(
        args.authorization, args.stage, args.lab, args.source_commit, args.isaac_python
    )
    args.stage.mkdir(parents=True, exist_ok=False)
    train_job = args.stage / "train-job.json"
    write_new(train_job, train)
    if args.prepare_only:
        return 0
    if not args.execute:
        raise ValueError("--prepare-only 또는 --execute가 필요하다")
    train_dir = args.stage / "train"
    train_dir.mkdir()
    output, terminal = train_dir / "worker.json", train_dir / "terminal"
    auth = V2.authorization(args.authorization)
    old = {auth["none_checkpoint"]["sha256"], F1_SHA256}
    old_args = [
        part for digest in sorted(old) for part in ("--old-checkpoint-sha256", digest)
    ]
    started = time.monotonic()
    code = V2.run(
        [
            str(args.isaac_python),
            str(CHILD),
            "--job",
            str(train_job),
            "--output",
            str(output),
            "--terminal",
            str(terminal),
            *old_args,
        ],
        args.stage / "train.stdout.txt",
        args.stage / "train.stderr.txt",
        6000,
        args.stage / "train-timeout-failure.json",
    )
    shutdown = V2.TERMINAL.verify(terminal, output, code, old)
    trained = read(output)
    evaluate["artifacts"]["CORRECTED"]["checkpoint"] = trained["checkpoint"]
    eval_job, eval_output = (
        args.stage / "evaluate-job.json",
        args.stage / "evaluate-worker.json",
    )
    write_new(eval_job, evaluate)
    remaining = int(7200 - (time.monotonic() - started))
    if remaining <= 0:
        raise TimeoutError("전체 7200초 예산이 평가 전에 소진됐다")
    code = V2.run(
        [
            str(args.isaac_python),
            str(GENERIC),
            "--job",
            str(eval_job),
            "--output",
            str(eval_output),
        ],
        args.stage / "evaluate.stdout.txt",
        args.stage / "evaluate.stderr.txt",
        min(900, remaining),
        args.stage / "evaluate-timeout-failure.json",
    )
    if code:
        raise RuntimeError(f"fixed evaluation child exit={code}")
    evaluated = read(eval_output)
    reports = evaluated.get("reports", {})
    if evaluated.get("completed") is not True or set(reports) != {"NONE", "CORRECTED"}:
        raise ValueError("fixed evaluation receipt 불일치")
    verdict = judge(
        read(Path(reports["NONE"]["path"])), read(Path(reports["CORRECTED"]["path"]))
    )
    if verdict["outcome"] == "unknown":
        raise ValueError("NONE reference 무효로 회복 판정을 중단한다")
    result_path = args.stage / "result.json"
    write_new(
        result_path,
        {
            "schema": "local_f1_restart_result_v3",
            "status": "complete",
            "fresh_seed": 3221,
            "training": {**binding(output), "shutdown": shutdown},
            "evaluation": binding(eval_output),
            **verdict,
        },
    )
    ledger_path = Path(auth["approval"]["ledger"]["path"])
    receipt = CorrectionLedger(ledger_path).receipt(
        auth["approval"]["request_id"],
        args.stage.name,
        verdict["outcome"],
        sha(result_path),
        gpu_s=None,
        exit_code=0,
    )
    write_new(
        args.stage / "final.json",
        {
            "schema": "local_f1_restart_final_v3",
            "status": "complete",
            "result": binding(result_path),
            "correction_receipt": receipt,
            "ledger": binding(ledger_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
