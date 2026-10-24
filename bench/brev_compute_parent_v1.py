"""Brev 파일럿 자식의 계약·종료·GPU 정리를 검증하는 Linux 부모."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODES = {
    "metadata": "brev_compute_metadata_v1",
    "empty-stage": "brev_compute_empty_stage_v1",
    "train": "brev_compute_train_v1",
}
PACKAGE_KEYS = ("isaacsim", "isaaclab", "isaaclab_rl", "torch", "rsl-rl-lib", "gymnasium")
PARENT_BINDING = "bench/brev_compute_parent_v1.py"
WORKER_BINDING = "bench/brev_compute_probe_v1.py"
PROTOCOL_SCHEMA = "brev_compute_pilot_protocol_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(label)
    return Path(value).resolve()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label) from error
    if not isinstance(value, dict):
        raise TypeError(label)
    return value


def _git(lab: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(lab), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Isaac Lab source binding") from error
    if result.returncode != 0:
        raise ValueError("Isaac Lab source binding")
    return result.stdout.strip()


def _validated_contract(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    if job.get("schema") != "brev_compute_job_v1" or job.get("mode") not in MODES:
        raise ValueError("job schema or mode")

    protocol_path = _path(job.get("protocol"), "protocol path")
    protocol_hash = job.get("protocol_sha256_lf")
    if not isinstance(protocol_hash, str) or not SHA256_RE.fullmatch(protocol_hash):
        raise ValueError("protocol source binding")
    if sha(protocol_path, lf=True) != protocol_hash:
        raise ValueError("protocol source binding")
    protocol = _load_json(protocol_path, "protocol document")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("protocol schema")

    source_binding = protocol.get("source_binding")
    if not isinstance(source_binding, dict):
        raise TypeError("protocol source binding")
    parent_hash = source_binding.get(PARENT_BINDING)
    worker_hash = source_binding.get(WORKER_BINDING)
    if parent_hash != sha(Path(__file__).resolve(), lf=True):
        raise ValueError("parent source binding")
    worker = _path(job.get("worker"), "worker path")
    if worker.name != "brev_compute_probe_v1.py":
        raise ValueError("worker source binding")
    if worker_hash != job.get("worker_sha256_lf") or worker_hash != sha(worker, lf=True):
        raise ValueError("worker source binding")

    stage_name = job.get("stage")
    stages = protocol.get("stages")
    if not isinstance(stage_name, str) or not isinstance(stages, list):
        raise TypeError("protocol stage")
    matches = [stage for stage in stages if isinstance(stage, dict) and stage.get("name") == stage_name]
    if len(matches) != 1:
        raise ValueError("protocol stage")
    stage = matches[0]
    mode = job["mode"]
    timeout = job.get("timeout_s")
    if stage.get("mode") != mode or isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("stage job contract")
    if not math.isfinite(float(timeout)) or timeout <= 0 or timeout != stage.get("timeout_s"):
        raise ValueError("stage job contract")
    if mode == "train":
        if type(job.get("seed")) is not int or job.get("seed") != stage.get("seed"):
            raise ValueError("stage job contract")
        if type(job.get("iterations")) is not int or job.get("iterations") != stage.get("iterations"):
            raise ValueError("stage job contract")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", job.get("run_name", "")):
            raise ValueError("training contract")
    elif job.get("seed") is not None or job.get("iterations") is not None:
        raise ValueError("stage job contract")

    environment = protocol.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("protocol environment")
    conditions = protocol.get("conditions")
    if (
        not isinstance(conditions, dict)
        or conditions.get("task") != TASK
        or conditions.get("num_envs") != 4096
    ):
        raise ValueError("protocol training conditions")
    expected_lab_sha = environment.get("isaaclab_sha")
    if job.get("isaaclab_sha") != expected_lab_sha or not isinstance(expected_lab_sha, str):
        raise ValueError("Isaac Lab source binding")
    lab = _path(job.get("isaaclab"), "Isaac Lab path")
    if _git(lab, "rev-parse", "HEAD") != expected_lab_sha:
        raise ValueError("Isaac Lab source binding")
    if _git(lab, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("Isaac Lab tracked files are dirty")
    return protocol, stage, worker, lab


def validate_job(job: dict[str, Any]) -> list[str]:
    _, _, worker, lab = _validated_contract(job)
    output = _path(job.get("output"), "worker output")
    if not Path(job["output"]).is_absolute() or output.exists():
        raise FileExistsError("worker output must be a new absolute path")
    mode = job["mode"]
    argv = [
        sys.executable,
        "-B",
        str(worker),
        "--mode",
        mode,
        "--output",
        str(output),
        "--timeout-s",
        str(job["timeout_s"]),
    ]
    if mode == "train":
        argv += [
            "--isaaclab",
            str(lab),
            "--seed",
            str(job["seed"]),
            "--num-envs",
            "4096",
            "--iterations",
            str(job["iterations"]),
            "--run-name",
            job["run_name"],
        ]
    return argv


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _validate_environment(report: dict[str, Any], environment: dict[str, Any]) -> None:
    python = report.get("python")
    if not isinstance(python, dict) or python.get("version") != environment.get("python"):
        raise ValueError("receipt Python environment")
    packages = report.get("packages")
    expected = {key: environment.get(key) for key in PACKAGE_KEYS}
    if not isinstance(packages, dict) or packages != expected or any(value is None for value in expected.values()):
        raise ValueError("receipt package environment")
    torch = report.get("torch")
    if not isinstance(torch, dict):
        raise TypeError("receipt CUDA environment")
    if (
        torch.get("imported") is not True
        or torch.get("cuda_available") is not True
        or torch.get("device_count") != 1
    ):
        raise ValueError("receipt CUDA environment")
    devices = torch.get("devices")
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], dict):
        raise ValueError("receipt CUDA device")
    device = devices[0]
    capability = device.get("compute_capability")
    if device.get("index") != 0 or not isinstance(device.get("name"), str) or not device["name"]:
        raise ValueError("receipt CUDA device")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(part) is not int or part < 0 for part in capability)
    ):
        raise ValueError("receipt CUDA device")
    memory = device.get("total_memory_bytes")
    if type(memory) is not int or memory <= 0:
        raise ValueError("receipt CUDA device")


def validate_receipt_payload(
    job: dict[str, Any], report: dict[str, Any], protocol: dict[str, Any], stage: dict[str, Any]
) -> None:
    mode = job["mode"]
    if report.get("schema") != MODES[mode] or report.get("mode") != mode or report.get("completed") is not True:
        raise ValueError("receipt completion")
    if report.get("timeout_s") != stage.get("timeout_s"):
        raise ValueError("receipt timeout")
    environment = protocol.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("protocol environment")
    _validate_environment(report, environment)

    if mode == "metadata":
        return
    if mode == "empty-stage":
        if report.get("empty_stage_physics_steps") != stage.get("physics_steps"):
            raise ValueError("empty-stage receipt")
        if report.get("robot_environments") != stage.get("robot_envs"):
            raise ValueError("empty-stage receipt")
        return

    conditions = protocol.get("conditions")
    if not isinstance(conditions, dict):
        raise TypeError("protocol conditions")
    iterations = job["iterations"]
    if report.get("task") != conditions.get("task") or report.get("observed_task") != conditions.get("task"):
        raise ValueError("training task receipt")
    exact = {
        "seed": job["seed"],
        "num_envs": conditions.get("num_envs"),
        "observed_num_envs": conditions.get("num_envs"),
        "requested_iterations": iterations,
        "observed_iterations": iterations,
    }
    if any(report.get(key) != value for key, value in exact.items()):
        raise ValueError("training parameter receipt")
    if not _positive_number(report.get("learn_wall_s")):
        raise ValueError("training timing receipt")
    steps_per_env = report.get("num_steps_per_env")
    aggregate_steps = report.get("aggregate_environment_steps")
    if type(steps_per_env) is not int or steps_per_env <= 0:
        raise ValueError("training steps receipt")
    expected_steps = conditions.get("num_envs") * iterations * steps_per_env
    if type(aggregate_steps) is not int or aggregate_steps != expected_steps:
        raise ValueError("training steps receipt")
    checkpoint_path = report.get("checkpoint_path")
    checkpoint_sha = report.get("checkpoint_sha256_raw")
    if not isinstance(checkpoint_path, str) or not checkpoint_path or not isinstance(checkpoint_sha, str):
        raise ValueError("checkpoint receipt")
    if not SHA256_RE.fullmatch(checkpoint_sha):
        raise ValueError("checkpoint receipt")


def validate_receipt(job: dict[str, Any], report: dict[str, Any]) -> None:
    protocol, stage, _, _ = _validated_contract(job)
    validate_receipt_payload(job, report, protocol, stage)


def gpu_pids() -> list[int]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return sorted(int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdecimal())


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def cleanup_group(pgid: int, proc: subprocess.Popen[Any]) -> bool:
    if not group_alive(pgid):
        return True
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and group_alive(pgid):
        proc.poll()
        time.sleep(0.1)
    if group_alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and group_alive(pgid):
            proc.poll()
            time.sleep(0.1)
    return not group_alive(pgid)


def execute(job: dict[str, Any], output: Path, *, read_gpu=gpu_pids) -> dict[str, Any]:
    if os.name != "posix":
        raise RuntimeError("Linux process group parent required")
    argv = validate_job(job)
    if output.exists() or output.with_suffix(".started.json").exists():
        raise FileExistsError("parent output already exists")
    before = read_gpu()
    if before:
        raise RuntimeError("GPU occupied before pilot")
    write_new(
        output.with_suffix(".started.json"),
        {
            "schema": "brev_compute_parent_started_v1",
            "job_sha256": hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
        },
    )
    stdout, stderr = output.with_suffix(".stdout.log"), output.with_suffix(".stderr.log")
    record: dict[str, Any] = {
        "schema": "brev_compute_parent_v1",
        "completed": False,
        "mode": job["mode"],
        "stage": job.get("stage"),
        "timed_out": False,
        "exit_code": None,
        "cleanup_verified": False,
        "protocol_sha256_lf": job.get("protocol_sha256_lf"),
        "worker_sha256_lf": job["worker_sha256_lf"],
        "gpu_before_pids": before,
        "wall_scope": "owned_child_launch_to_exit_and_process_group_cleanup",
        "timeout_s": job["timeout_s"],
    }
    started = time.perf_counter()
    proc: subprocess.Popen[Any] | None = None
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        try:
            env = {
                **os.environ,
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            proc = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
            try:
                record["exit_code"] = proc.wait(timeout=job["timeout_s"])
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
        except OSError as error:
            record["error_type"] = type(error).__name__
        finally:
            if proc is not None:
                record["cleanup_verified"] = cleanup_group(proc.pid, proc)
                proc.wait(timeout=10)
                record["exit_code"] = proc.returncode
            record["wall_s"] = time.perf_counter() - started
    try:
        after = read_gpu()
        record["gpu_after_pids"] = after
        record["cleanup_verified"] = record["cleanup_verified"] and not after
        report_path = Path(job["output"])
        if report_path.is_file():
            report = _load_json(report_path, "worker receipt")
            record["report_sha256_raw"] = sha(report_path)
            validate_receipt(job, report)
            record["completed"] = (
                record["exit_code"] == 0
                and not record["timed_out"]
                and record["cleanup_verified"]
            )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
        record["completed"] = False
        record["error_type"] = type(error).__name__
    write_new(output, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = execute(json.loads(args.job.read_text(encoding="utf-8")), args.output)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        if not args.output.exists():
            write_new(
                args.output,
                {"schema": "brev_compute_parent_v1", "completed": False, "error_type": type(error).__name__},
            )
        print(json.dumps({"completed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps({key: record[key] for key in ("completed", "exit_code", "timed_out", "cleanup_verified")}))
    return 0 if record["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
