"""승인된 F1 fresh restart와 고정 평가를 순차 실행한다."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import local_f1_restart_terminal_v2 as TERMINAL
from local_f1_restart_base_v2 import (
    ROOT,
    binding,
    judge,
    read,
    sha,
    source_map,
    write_new,
)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import brev_recovery_worker_v2 as V2
import brev_recovery_worker_v3 as V3

from rl_triage.correction_ledger import CorrectionLedger

CHILD = ROOT / "bench/local_f1_restart_child_v2.py"
GENERIC = ROOT / "bench/brev_recovery_worker_v3.py"
PROTOCOL = ROOT / "bench/protocols/local_f1_restart_s3221_v2.json"
TRAIN_TIMEOUT_S = 6000
EVAL_TIMEOUT_S = 900
TOTAL_TIMEOUT_S = 7200
OLD_F1_SHA256 = "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731"


def old_checkpoint_shas(auth: dict[str, Any]) -> set[str]:
    return {auth["none_checkpoint"]["sha256"], OLD_F1_SHA256}


def authorization(path: Path) -> dict[str, Any]:
    value = read(path)
    if (value.get("schema") != "local_f1_restart_authorization_v2" or value.get("execution_authorized") is not True
            or value.get("fresh_seed") != 3221 or value.get("num_envs") != 4096 or value.get("iterations") != 300
            or value.get("num_steps_per_env") != 24 or value.get("action_scale") != 0.25
            or value.get("correction_id") != "restore_body_frame_reward" or value.get("recovery_mode") != "restart"
            or value.get("approval", {}).get("ledger_state") != "consumed"
            or value.get("approval", {}).get("automatic_model_approval") is not False):
        raise ValueError("authorization 불일치")
    for name in ("model_proposal", "pair", "source_manifest", "source_readback", "source_adapter",
                 "none_checkpoint", "fixed_eval"):
        item = value.get(name, {})
        if not Path(item.get("path", "")).is_file() or sha(Path(item["path"])) != item.get("sha256"):
            raise ValueError(f"authorization binding 변경: {name}")
    ledger = value["approval"]["ledger"]
    if sha(Path(ledger["path"])) != ledger["sha256"]:
        raise ValueError("ledger 변경")
    return value


def build_jobs(auth_path: Path, stage: Path, lab: Path, source_commit: str) -> tuple[dict, dict]:
    auth, sources = authorization(auth_path), source_map()
    protocol = read(PROTOCOL)
    anchor = protocol.get("registration_anchor")
    ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", str(anchor), source_commit],
                              check=False).returncode == 0
    if (protocol.get("status") != "registered" or not ancestor
            or protocol.get("timeouts_s") != {"train": TRAIN_TIMEOUT_S, "evaluate": EVAL_TIMEOUT_S,
                                                "total": TOTAL_TIMEOUT_S, "cleanup": 30}):
        raise ValueError("registered protocol anchor가 runtime source commit의 조상이 아니다")
    common = {"schema": "brev_recovery_worker_job_v1", "stage": str(stage.resolve()),
              "lab": str(lab.resolve()), "repo": str(ROOT.resolve()), "protocol_path": str(PROTOCOL.resolve()),
              "protocol_sha256_lf": sha(PROTOCOL, lf=True), "source_commit_sha": source_commit,
              "source_sha256_lf": sources, "runtime_source_sha256_lf": sources,
              "python_package_source_sha256_lf": sources}
    train_dir = stage / "train"
    train = {**common, "mode": "train", "arm": "CORRECTED", "seed": 3221,
             "run_name": "local_f1_restart_s3221_v2", "iterations": 300,
             "manifest_path": str(train_dir / "manifest.json"), "records_path": str(train_dir / "records.jsonl"),
             "readback_path": str(train_dir / "readback.json"), "telemetry_path": str(train_dir / "telemetry.json"),
             "source_checkpoint_sha256": auth["none_checkpoint"]["sha256"]}
    evaluate = {**common, "mode": "evaluate", "artifacts": {
        "NONE": {"checkpoint": auth["none_checkpoint"]},
        "CORRECTED": {"checkpoint": {"path": str(train_dir / "model_299.pt"), "sha256": "PENDING"}},
    }, "fixed_eval_protocol": auth["fixed_eval"]["path"], "reports_dir": str(stage / "fixed-eval"),
        "jobs_path": str(stage / "fixed-eval-jobs.json")}
    V3.validate_job(train)
    V2._residual_manifest(train, mode="train", manifest_path=Path(train["manifest_path"]),
                          records_path=Path(train["records_path"]), readback_path=Path(train["readback_path"]))
    return train, evaluate


def run(argv: list[str], stdout: Path, stderr: Path, timeout_s: int, failure_path: Path,
        *, popen: Any = subprocess.Popen, cleanup: Any = subprocess.run) -> int:
    with stdout.open("x", encoding="utf-8", newline="\n") as out, stderr.open("x", encoding="utf-8", newline="\n") as err:
        process = popen(argv, stdout=out, stderr=err,
                        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            return int(process.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired as error:
            killed = cleanup(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                             timeout=30, check=False)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired as cleanup_error:
                write_new(failure_path, {"schema": "local_f1_restart_timeout_v2", "status": "failed",
                                         "pid": process.pid, "timeout_s": timeout_s,
                                         "cleanup_returncode": getattr(killed, "returncode", None),
                                         "cleanup_confirmed": False})
                raise TimeoutError(f"owned child tree cleanup timeout pid={process.pid}") from cleanup_error
            write_new(failure_path, {"schema": "local_f1_restart_timeout_v2", "status": "failed",
                                     "pid": process.pid, "timeout_s": timeout_s,
                                     "cleanup_returncode": getattr(killed, "returncode", None),
                                     "cleanup_confirmed": True})
            raise TimeoutError(f"owned child timeout pid={process.pid}") from error


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
    train, evaluate = build_jobs(args.authorization, args.stage, args.lab, args.source_commit)
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
    auth = authorization(args.authorization)
    old = old_checkpoint_shas(auth)
    started = time.monotonic()
    old_args = [part for digest in sorted(old) for part in ("--old-checkpoint-sha256", digest)]
    code = run([str(args.isaac_python), str(CHILD), "--job", str(train_job), "--output", str(output),
                "--terminal", str(terminal), *old_args],
               args.stage / "train.stdout.txt", args.stage / "train.stderr.txt", TRAIN_TIMEOUT_S,
               args.stage / "train-timeout-failure.json")
    shutdown = TERMINAL.verify(terminal, output, code, old)
    trained = read(output)
    evaluate["artifacts"]["CORRECTED"]["checkpoint"] = trained["checkpoint"]
    eval_job = args.stage / "evaluate-job.json"
    write_new(eval_job, evaluate)
    eval_output = args.stage / "evaluate-worker.json"
    remaining = int(TOTAL_TIMEOUT_S - (time.monotonic() - started))
    if remaining <= 0:
        raise TimeoutError("전체 7200초 예산이 평가 전에 소진됐다")
    code = run([str(args.isaac_python), str(GENERIC), "--job", str(eval_job), "--output", str(eval_output)],
               args.stage / "evaluate.stdout.txt", args.stage / "evaluate.stderr.txt",
               min(EVAL_TIMEOUT_S, remaining), args.stage / "evaluate-timeout-failure.json")
    if code:
        raise RuntimeError(f"fixed evaluation child exit={code}")
    evaluated = read(eval_output)
    reports = evaluated.get("reports", {})
    if evaluated.get("completed") is not True or set(reports) != {"NONE", "CORRECTED"}:
        raise ValueError("fixed evaluation receipt 불일치")
    verdict = judge(read(Path(reports["NONE"]["path"])), read(Path(reports["CORRECTED"]["path"])))
    if verdict["outcome"] == "unknown":
        raise ValueError("NONE reference가 불건강하거나 수치가 무효라 회복 판정을 중단한다")
    result_path = args.stage / "result.json"
    write_new(result_path, {"schema": "local_f1_restart_result_v2", "status": "complete", "fresh_seed": 3221,
                            "training": {**binding(output), "shutdown": shutdown},
                            "evaluation": binding(eval_output), **verdict})
    ledger_path = Path(auth["approval"]["ledger"]["path"])
    receipt = CorrectionLedger(ledger_path).receipt(auth["approval"]["request_id"], args.stage.name,
                                                    verdict["outcome"], sha(result_path), gpu_s=None, exit_code=0)
    write_new(args.stage / "final.json", {"schema": "local_f1_restart_final_v2", "status": "complete",
                                          "result": binding(result_path), "correction_receipt": receipt,
                                          "ledger": binding(ledger_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
