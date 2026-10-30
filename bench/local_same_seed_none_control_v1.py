"""seed 3221 NONE 통제를 한 번 학습·평가해 RI213 실패 원인을 분리한다."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / ".git").exists() and (
            parent / "bench/local_f1_restart_base_v3.py"
        ).is_file():
            return parent
    raise FileNotFoundError("rl-triage 저장소 root를 찾지 못했다")


ROOT = _root(Path(__file__))
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))

import brev_recovery_worker_v2 as GENERIC_V2
import brev_recovery_worker_v3 as GENERIC_V3
import local_f1_restart_base_v3 as BASE
import local_f1_restart_worker_v2 as LOCAL
import local_same_seed_none_terminal_v1 as TERMINAL
import residual_training_v1 as TRAINING

PROTOCOL = ROOT / "bench/protocols/local_same_seed_none_control_s3221_v1.json"
CHILD = ROOT / "bench/local_same_seed_none_terminal_v1.py"
GENERIC = ROOT / "bench/brev_recovery_worker_v3.py"
TRAIN_TIMEOUT_S, EVAL_TIMEOUT_S, TOTAL_TIMEOUT_S = 6000, 900, 7200

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *BASE.SOURCE_FILES,
            "bench/local_same_seed_none_control_v1.py",
            "bench/finalize_local_same_seed_none_control_v1.py",
            "bench/local_same_seed_none_terminal_v1.py",
            "bench/protocols/local_same_seed_none_control_s3221_v1.template.json",
            "bench/protocols/local_same_seed_none_control_s3221_v1.json",
        )
    )
)


def source_map(root: Path = ROOT) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source closure 누락: {missing}")
    return {name: BASE.sha(root / name, lf=True) for name in SOURCE_FILES}


def _binding(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if (
        root.resolve() not in path.parents
        or not path.is_file()
        or BASE.sha(path) != expected
    ):
        raise ValueError(f"RI213 input binding 불일치: {relative}")
    return path


def _registered(source_commit: str) -> tuple[dict[str, Any], dict[str, str]]:
    protocol = BASE.read(PROTOCOL)
    anchor = str(protocol.get("registration_anchor", ""))
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                anchor,
                source_commit,
            ],
            check=False,
        ).returncode
        == 0
    )
    if (
        protocol.get("schema") != "local_same_seed_none_control_preregistration_v1"
        or protocol.get("status") != "registered"
        or not ancestor
        or protocol.get("timeouts_s")
        != {"train": 6000, "evaluate": 900, "total": 7200, "cleanup": 30}
        or protocol.get("training")
        != {
            "seed": 3221,
            "num_envs": 4096,
            "iterations": 300,
            "num_steps_per_env": 24,
            "action_scale": 0.25,
        }
        or protocol.get("automatic_retry") != 0
    ):
        raise ValueError("registered NONE control protocol 불일치")
    names = tuple(
        name for name in SOURCE_FILES if name != PROTOCOL.relative_to(ROOT).as_posix()
    )
    current = {name: BASE.sha(ROOT / name, lf=True) for name in names}
    blobs = BASE.git_blob_map(ROOT, anchor, names)
    if current != blobs or current != protocol.get("source_sha256_lf"):
        raise ValueError("registered source blob/current binding 불일치")
    return protocol, source_map()


def build_jobs(
    input_root: Path, stage: Path, lab: Path, source_commit: str, isaac_python: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    stage, input_root = stage.resolve(), input_root.resolve()
    protocol, repo_sources = _registered(source_commit)
    inputs = {
        name: _binding(input_root, row["relative"], row["sha256"])
        for name, row in protocol["ri213_inputs"].items()
    }
    original = BASE.read(inputs["result"])
    if (
        original.get("outcome") != "not_recovered"
        or original.get("ratios") != protocol["original_result"]
    ):
        raise ValueError("RI213 original outcome/ratios 변경")
    prior_evaluation = BASE.read(inputs["evaluate_worker"])
    expected_reports = {
        "NONE": protocol["ri213_inputs"]["none_report"]["sha256"],
        "CORRECTED": protocol["ri213_inputs"]["corrected_report"]["sha256"],
    }
    if (
        prior_evaluation.get("completed") is not True
        or {
            name: row.get("sha256")
            for name, row in prior_evaluation.get("reports", {}).items()
        }
        != expected_reports
    ):
        raise ValueError("RI213 evaluation/report 결합 불일치")
    corrected_worker = BASE.read(inputs["corrected_worker"])
    corrected_checkpoint = corrected_worker.get("checkpoint", {})
    corrected_checkpoint_path = Path(corrected_checkpoint.get("path", ""))
    if (
        corrected_checkpoint.get("sha256") != protocol["old_checkpoint_sha256"][2]
        or not corrected_checkpoint_path.is_file()
        or BASE.sha(corrected_checkpoint_path) != corrected_checkpoint["sha256"]
    ):
        raise ValueError("RI213 corrected checkpoint 결합 불일치")
    fixed = ROOT / protocol["fixed_eval"]["relative"]
    fixed_body = BASE.read(fixed)
    if (
        BASE.sha(fixed, lf=True) != protocol["fixed_eval"]["sha256_lf"]
        or fixed_body.get("eval_seed") != 2026
    ):
        raise ValueError("fixed evaluation protocol 변경")
    runtime = protocol["runtime"]
    runtime_sources = BASE.runtime_snapshot(
        lab, runtime["isaaclab_head"], runtime["source_sha256_lf"]
    )
    package = protocol["python_package"]
    package_sources = BASE.package_snapshot(
        isaac_python,
        package["distribution"],
        package["version"],
        package["source_sha256_lf"],
    )
    common = {
        "schema": "brev_recovery_worker_job_v1",
        "stage": str(stage),
        "lab": str(lab.resolve()),
        "repo": str(ROOT.resolve()),
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256_lf": BASE.sha(PROTOCOL, lf=True),
        "source_commit_sha": source_commit,
        "source_sha256_lf": repo_sources,
        "runtime_source_sha256_lf": runtime_sources,
        "python_package_source_sha256_lf": package_sources,
    }
    train_dir = stage / "train"
    train = {
        **common,
        "mode": "train",
        "arm": "NONE",
        "seed": 3221,
        "run_name": "local_same_seed_none_control_s3221_v1",
        "iterations": 300,
        "manifest_path": str(train_dir / "manifest.json"),
        "records_path": str(train_dir / "records.jsonl"),
        "readback_path": str(train_dir / "readback.json"),
        "telemetry_path": str(train_dir / "telemetry.json"),
        "source_checkpoint_sha256": None,
    }
    evaluate = {
        **common,
        "mode": "evaluate",
        "artifacts": {
            "HEALTHY_NONE_S3221": {
                "checkpoint": {
                    "path": str(train_dir / "model_299.pt"),
                    "sha256": "PENDING",
                }
            }
        },
        "fixed_eval_protocol": str(fixed),
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
    return train, evaluate, inputs


def classify(
    original_none: dict[str, Any], corrected: dict[str, Any], healthy: dict[str, Any]
) -> dict[str, Any]:
    healthy_old = BASE.judge(original_none, healthy)
    corrected_healthy = BASE.judge(healthy, corrected)
    if (
        healthy_old["outcome"] == "not_recovered"
        and corrected_healthy["outcome"] == "recovered"
    ):
        interpretation = "seed_or_command_coverage_consistent"
    elif (
        healthy_old["outcome"] == "recovered"
        and corrected_healthy["outcome"] == "not_recovered"
    ):
        interpretation = "correction_path_influence_possible"
    else:
        interpretation = "inconclusive"
    return {
        "interpretation": interpretation,
        "causal_claim": False,
        "healthy_s3221_vs_original_none_s2221": healthy_old,
        "corrected_s3221_vs_healthy_s3221": corrected_healthy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--lab", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    train, evaluate, inputs = build_jobs(
        args.input_root, args.stage, args.lab, args.source_commit, args.isaac_python
    )
    stage = args.stage.resolve()
    stage.mkdir(parents=True, exist_ok=False)
    train_job = stage / "train-job.json"
    BASE.write_new(train_job, train)
    if args.prepare_only:
        return 0
    if not args.execute:
        raise ValueError("--prepare-only 또는 --execute가 필요하다")
    train_dir = stage / "train"
    train_dir.mkdir()
    output, terminal = train_dir / "worker.json", train_dir / "terminal"
    old = set(BASE.read(PROTOCOL)["old_checkpoint_sha256"][:2])
    old_args = [
        item for digest in sorted(old) for item in ("--old-checkpoint-sha256", digest)
    ]
    started = time.monotonic()
    code = LOCAL.run(
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
        stage / "train.stdout.txt",
        stage / "train.stderr.txt",
        TRAIN_TIMEOUT_S,
        stage / "train-timeout-failure.json",
    )
    shutdown = TERMINAL.verify(terminal, output, code, old)
    trained = BASE.read(output)
    if trained.get("arm") != "NONE" or trained.get("seed") != 3221:
        raise ValueError("NONE control training receipt 불일치")
    evaluate["artifacts"]["HEALTHY_NONE_S3221"]["checkpoint"] = trained["checkpoint"]
    eval_job, eval_output = stage / "evaluate-job.json", stage / "evaluate-worker.json"
    BASE.write_new(eval_job, evaluate)
    remaining = int(TOTAL_TIMEOUT_S - (time.monotonic() - started))
    if remaining <= 0:
        raise TimeoutError("전체 7200초 예산이 평가 전에 소진됐다")
    code = LOCAL.run(
        [
            str(args.isaac_python),
            str(GENERIC),
            "--job",
            str(eval_job),
            "--output",
            str(eval_output),
        ],
        stage / "evaluate.stdout.txt",
        stage / "evaluate.stderr.txt",
        min(EVAL_TIMEOUT_S, remaining),
        stage / "evaluate-timeout-failure.json",
    )
    if code:
        raise RuntimeError(f"fixed evaluation child exit={code}")
    evaluated = BASE.read(eval_output)
    report = evaluated.get("reports", {}).get("HEALTHY_NONE_S3221", {})
    healthy_path = Path(report.get("path", ""))
    if (
        evaluated.get("completed") is not True
        or not healthy_path.is_file()
        or BASE.sha(healthy_path) != report.get("sha256")
    ):
        raise ValueError("healthy NONE fixed evaluation receipt 불일치")
    comparison = classify(
        BASE.read(inputs["none_report"]),
        BASE.read(inputs["corrected_report"]),
        BASE.read(healthy_path),
    )
    corrected_readback = BASE.read(inputs["corrected_worker"])["readback"]
    healthy_readback = trained["readback"]
    prior_job = BASE.read(inputs["train_job"])
    observation = {
        "checkpoint_bytes_equal_prior_corrected": (
            trained["checkpoint"]["sha256"]
            == BASE.read(inputs["corrected_worker"])["checkpoint"]["sha256"]
        ),
        "same_seed_iterations_runtime_and_package": (
            train["seed"] == prior_job["seed"] == 3221
            and train["iterations"] == prior_job["iterations"] == 300
            and train["runtime_source_sha256_lf"]
            == prior_job["runtime_source_sha256_lf"]
            and train["python_package_source_sha256_lf"]
            == prior_job["python_package_source_sha256_lf"]
        ),
        "arm_difference_is_registered": {
            "prior": prior_job["arm"],
            "control": train["arm"],
        }
        == {"prior": "CORRECTED", "control": "NONE"},
        "corrected_reward_restored": corrected_readback["reward"]["before_code_sha256"]
        == corrected_readback["reward"]["after_code_sha256"],
        "healthy_remaining_faults": healthy_readback.get("remaining_faults") == [],
        "causal_claim": False,
    }
    BASE.write_new(
        stage / "result.json",
        {
            "schema": "local_same_seed_none_control_result_v1",
            "status": "complete",
            "development_control_after_observing_seed3221": True,
            "training": {**BASE.binding(output), "shutdown": shutdown},
            "evaluation": BASE.binding(eval_output),
            "original_outcome_unchanged": "not_recovered",
            "comparison": comparison,
            "configuration_and_function_observation": observation,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
