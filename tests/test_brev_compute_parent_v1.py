"""부모 계약·종료·소스 결합 회귀. 모의 자식은 Isaac 실행 증거가 아니다."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "brev_parent", Path(__file__).resolve().parents[1] / "bench/brev_compute_parent_v1.py"
)
assert SPEC and SPEC.loader
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

ENVIRONMENT = {
    "python": "3.10.12",
    "isaacsim": "4.5.0.0",
    "isaaclab": "0.41.3",
    "isaaclab_rl": "0.1.8",
    "torch": "2.7.0+cu128",
    "rsl-rl-lib": "2.3.3",
    "gymnasium": "1.2.0",
}
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def make_lab(path: Path) -> tuple[Path, str]:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    tracked = path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "-m",
        "init",
    )
    return path, git(path, "rev-parse", "HEAD")


def make_case(tmp_path: Path, mode: str = "metadata") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lab, lab_sha = make_lab(tmp_path / "lab")
    worker = tmp_path / "brev_compute_probe_v1.py"
    worker.write_text("pass\n", encoding="utf-8")
    stages = {
        "metadata": {"name": "metadata", "mode": "metadata", "timeout_s": 120},
        "empty-stage": {
            "name": "empty_stage",
            "mode": "empty-stage",
            "physics_steps": 1,
            "robot_envs": 0,
            "timeout_s": 600,
        },
        "train": {
            "name": "normal_s2180",
            "mode": "train",
            "seed": 2180,
            "iterations": 300,
            "timeout_s": 1800,
        },
    }
    environment = {**ENVIRONMENT, "isaaclab_sha": lab_sha}
    parent_file = P.__file__
    assert parent_file is not None
    protocol = {
        "schema": "brev_compute_pilot_protocol_v1",
        "source_binding": {
            P.PARENT_BINDING: P.sha(Path(parent_file).resolve(), lf=True),
            P.WORKER_BINDING: P.sha(worker, lf=True),
        },
        "environment": environment,
        "conditions": {"task": TASK, "num_envs": 4096},
        "stages": list(stages.values()),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    stage = stages[mode]
    job: dict[str, Any] = {
        "schema": "brev_compute_job_v1",
        "protocol": str(protocol_path),
        "protocol_sha256_lf": P.sha(protocol_path, lf=True),
        "stage": stage["name"],
        "mode": mode,
        "timeout_s": stage["timeout_s"],
        "worker": str(worker),
        "worker_sha256_lf": P.sha(worker, lf=True),
        "isaaclab": str(lab),
        "isaaclab_sha": lab_sha,
        "output": str(tmp_path / "worker.json"),
    }
    if mode == "train":
        job.update(seed=2180, iterations=300, run_name="normal-s2180")
    return job, protocol, stage


def valid_report(mode: str, job: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": P.MODES[mode],
        "mode": mode,
        "completed": True,
        "timeout_s": stage["timeout_s"],
        "python": {"version": ENVIRONMENT["python"], "implementation": "CPython"},
        "packages": {key: ENVIRONMENT[key] for key in P.PACKAGE_KEYS},
        "torch": {
            "imported": True,
            "cuda_available": True,
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "compute_capability": [12, 0],
                    "total_memory_bytes": 96 * 1024**3,
                }
            ],
        },
    }
    if mode == "empty-stage":
        report.update(empty_stage_physics_steps=1, robot_environments=0)
    if mode == "train":
        report.update(
            task=TASK,
            observed_task=TASK,
            seed=job["seed"],
            num_envs=4096,
            observed_num_envs=4096,
            requested_iterations=job["iterations"],
            observed_iterations=job["iterations"],
            learn_wall_s=12.5,
            num_steps_per_env=24,
            aggregate_environment_steps=4096 * job["iterations"] * 24,
            checkpoint_path="/tmp/model_300.pt",
            checkpoint_sha256_raw="a" * 64,
        )
    return report


def test_job_binds_protocol_sources_stage_and_output(tmp_path):
    job, _, _ = make_case(tmp_path)
    assert P.validate_job(job)[0] == sys.executable

    worker = Path(job["worker"])
    worker.write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worker source binding"):
        P.validate_job(job)

    job, _, _ = make_case(tmp_path / "fresh")
    protocol_path = Path(job["protocol"])
    protocol_path.write_text(protocol_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol source binding"):
        P.validate_job(job)

    job, protocol, _ = make_case(tmp_path / "wrong_parent")
    protocol["source_binding"][P.PARENT_BINDING] = "0" * 64
    Path(job["protocol"]).write_text(json.dumps(protocol), encoding="utf-8")
    job["protocol_sha256_lf"] = P.sha(Path(job["protocol"]), lf=True)
    with pytest.raises(ValueError, match="parent source binding"):
        P.validate_job(job)

    job, protocol, _ = make_case(tmp_path / "wrong_stage")
    protocol["stages"][0]["timeout_s"] = 121
    Path(job["protocol"]).write_text(json.dumps(protocol), encoding="utf-8")
    job["protocol_sha256_lf"] = P.sha(Path(job["protocol"]), lf=True)
    with pytest.raises(ValueError, match="stage job contract"):
        P.validate_job(job)

    job, _, _ = make_case(tmp_path / "collision")
    Path(job["output"]).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        P.validate_job(job)


@pytest.mark.parametrize("dirty", ["clean", "unstaged", "staged"])
def test_lab_tracked_cleanliness_is_fail_closed_but_untracked_is_allowed(tmp_path, dirty):
    job, _, _ = make_case(tmp_path)
    lab = Path(job["isaaclab"])
    (lab / "untracked.log").write_text("allowed\n", encoding="utf-8")
    if dirty != "clean":
        (lab / "tracked.txt").write_text("changed\n", encoding="utf-8")
    if dirty == "staged":
        git(lab, "add", "tracked.txt")
    if dirty == "clean":
        P.validate_job(job)
    else:
        with pytest.raises(ValueError, match="tracked files are dirty"):
            P.validate_job(job)


@pytest.mark.parametrize("mode", ["metadata", "empty-stage", "train"])
def test_each_mode_accepts_a_semantically_complete_receipt(tmp_path, mode):
    job, protocol, stage = make_case(tmp_path, mode)
    P.validate_receipt_payload(job, valid_report(mode, job, stage), protocol, stage)


def test_train_command_fixes_environment_count_and_protocol_stage(tmp_path):
    job, _, _ = make_case(tmp_path, "train")
    argv = P.validate_job(job)
    assert argv[argv.index("--num-envs") + 1] == "4096"
    assert argv[argv.index("--seed") + 1] == "2180"
    assert argv[argv.index("--iterations") + 1] == "300"


@pytest.mark.parametrize(
    "mode,field,value",
    [
        ("metadata", "packages", {"torch": "2.7.0+cu128"}),
        ("metadata", "python", {"version": "3.11.0"}),
        ("metadata", "torch", {"imported": True, "cuda_available": False, "device_count": 1}),
        ("metadata", "torch", {"imported": True, "cuda_available": True, "device_count": 1, "devices": []}),
        ("empty-stage", "empty_stage_physics_steps", 2),
        ("empty-stage", "robot_environments", 1),
        ("train", "observed_task", "wrong-task"),
        ("train", "observed_num_envs", 4095),
        ("train", "observed_iterations", 299),
        ("train", "learn_wall_s", float("nan")),
        ("train", "aggregate_environment_steps", 1),
        ("train", "checkpoint_sha256_raw", "not-a-sha"),
    ],
)
def test_mode_receipt_mutations_fail_closed(tmp_path, mode, field, value):
    job, protocol, stage = make_case(tmp_path, mode)
    report = valid_report(mode, job, stage)
    report[field] = value
    with pytest.raises(ValueError, match="receipt"):
        P.validate_receipt_payload(job, report, protocol, stage)


def test_completed_snapshot_without_semantic_evidence_fails(tmp_path):
    job, protocol, stage = make_case(tmp_path)
    report = {"schema": P.MODES["metadata"], "mode": "metadata", "completed": True}
    with pytest.raises(ValueError):
        P.validate_receipt_payload(job, report, protocol, stage)


@pytest.mark.skipif(os.name != "posix", reason="실제 프로세스 그룹 회귀는 WSL Linux에서 실행")
@pytest.mark.parametrize(
    "behavior,expected_exit,timed_out,completed",
    [
        ("success", 0, False, True),
        ("no_report", 0, False, False),
        ("nonzero", 7, False, False),
        ("timeout", -15, True, False),
    ],
)
def test_real_child_lifecycle(
    tmp_path, monkeypatch, behavior, expected_exit, timed_out, completed
):
    report_path = tmp_path / "worker.json"
    timeout_s = 0.2 if timed_out else 10
    job = {
        "mode": "metadata",
        "stage": "metadata",
        "output": str(report_path),
        "worker_sha256_lf": "0" * 64,
        "protocol_sha256_lf": "1" * 64,
        "timeout_s": timeout_s,
    }
    stage = {"name": "metadata", "mode": "metadata", "timeout_s": timeout_s}
    protocol = {"environment": copy.deepcopy(ENVIRONMENT)}
    report = valid_report("metadata", job, stage)
    script = "import json,time,sys;from pathlib import Path;"
    if timed_out:
        script += "time.sleep(30);"
    if behavior in {"success", "nonzero"}:
        script += f"Path({str(report_path)!r}).write_text({json.dumps(report)!r});"
    script += "sys.exit(7)" if behavior == "nonzero" else "sys.exit(0)"
    monkeypatch.setattr(P, "validate_job", lambda _job: [sys.executable, "-c", script])
    monkeypatch.setattr(
        P,
        "validate_receipt",
        lambda actual_job, actual_report: P.validate_receipt_payload(
            actual_job, actual_report, protocol, stage
        ),
    )
    output = tmp_path / "parent.json"
    result = P.execute(job, output, read_gpu=list)
    assert result["completed"] is completed
    assert result["timed_out"] is timed_out
    assert result["exit_code"] == expected_exit
    assert result["cleanup_verified"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        P.execute(job, output, read_gpu=list)
