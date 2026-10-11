"""등록된 readback을 Windows에서 실행하고 source/body/owned-job 종료를 대조한다."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import record_evidence_media as M
import runtime_readback_smoke as S

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "bench/protocols/runtime_readback_smoke_v1.json"
SOURCE_FILES = S.PROJECT_SOURCE_FILES


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_committed(repo: Path, paths: set[str]) -> str:
    for relative in sorted(paths):
        S._relative_source_path(repo, relative, "committed source")
        subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
                       check=True, capture_output=True)
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *sorted(paths)],
                            check=True, capture_output=True)
    if status.stdout:
        raise ValueError("source and protocol must be committed before GPU execution")
    return S._git_head(repo)


def _process_creation_filetime(pid: int) -> int | None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None  # A member may exit between the Job query and this read.
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def job_identities(job: M.WindowsJob) -> dict[int, int]:
    capacity = max(job.active_processes() + 8, 16)
    for _ in range(8):
        class ProcessList(ctypes.Structure):
            _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                        ("pids", ctypes.c_size_t * capacity)]
        value = ProcessList()
        success = job._kernel32.QueryInformationJobObject(
            job._handle, 3, ctypes.byref(value), ctypes.sizeof(value), None)
        if success:
            identities = {}
            for pid in value.pids[:value.count]:
                created = _process_creation_filetime(int(pid))
                if created is not None:
                    identities[int(pid)] = created
            return identities
        if ctypes.get_last_error() != 234:
            raise ctypes.WinError(ctypes.get_last_error())
        capacity *= 2
    raise RuntimeError("owned Job PID inventory did not stabilize")


def run_owned(folder: Path, args: list[str], cwd: Path, watchdog_s: float = 360) -> dict:
    """Release the existing worker only after assigning its owned Windows Job."""
    launch = folder / "launch.json"
    write_new(launch, {"args": args, "cwd": str(cwd)})
    started = time.monotonic()
    job = M.WindowsJob()
    process = None
    assigned = False
    timed_out = False
    natural_exit = False
    cleanup = False
    identities: dict[int, int] = {}
    error_type = None
    exit_code = None
    try:
        with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "bench/record_evidence_media.py"), "--_worker", str(launch)],
                stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, creationflags=M.CREATE_NO_WINDOW,
            )
            job.assign(process)
            assigned = True
            identities.update(job_identities(job))
            assert process.stdin is not None
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            while True:
                identities.update(job_identities(job))
                exit_code = process.poll()
                if exit_code is not None and job.active_processes() == 0:
                    natural_exit = True
                    cleanup = True
                    break
                if time.monotonic() - started >= watchdog_s:
                    timed_out = True
                    break
                time.sleep(0.05)
    except BaseException as error:
        error_type = type(error).__name__
    finally:
        try:
            if assigned:
                identities.update(job_identities(job))
                if job.active_processes():
                    job.terminate(124)
                cleanup = job.wait_empty(10)
            elif process is not None:
                # An unreleased GO worker has created no SDK descendants.
                process.terminate()
            if process is not None:
                exit_code = process.wait(timeout=10)
        except BaseException as error:
            error_type = error_type or type(error).__name__
            cleanup = False
        finally:
            job.close()  # KILL_ON_JOB_CLOSE applies only to this assigned tree.
    return {
        "job_assigned_before_release": assigned,
        "natural_exit": natural_exit,
        "timed_out": timed_out,
        "error_type": error_type,
        "exit_code": exit_code,
        "cleanup_verified": cleanup,
        "owned_process_identities": {str(pid): stamp for pid, stamp in sorted(identities.items())},
        "wall_s": time.monotonic() - started,
        "wall_scope": "parent_launch_to_exit_and_owned_job_empty_verification",
    }


def verify_body(body: dict, validated: dict, execution: dict, project_commit: str) -> list[str]:
    reasons = []
    protocol = validated["protocol"]
    paths = validated["paths"]
    identities = execution["owned_process_identities"]
    expected = {
        "schema": "runtime_readback_smoke_body_v1", "completed": True,
        "scope": protocol["scope"], "task": S.TASK,
        "protocol_sha256": validated["protocol_sha256"],
        "manifest_sha256": S._sha256_file(validated["manifest_path"]),
        "project_commit_sha": project_commit,
        "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True,
    }
    for key, value in expected.items():
        if type(body.get(key)) is not type(value) or body.get(key) != value:
            reasons.append(f"body_{key}_mismatch")
    pid = body.get("process_id")
    creation = body.get("process_creation_time_filetime")
    if type(pid) is not int or type(creation) is not int or identities.get(str(pid)) != creation:
        reasons.append("body_pid_creation_not_observed_in_owned_job")
    if body.get("source_receipt") != validated["source_receipt"]:
        reasons.append("body_source_receipt_mismatch")
    if body.get("checkpoint", {}).get("sha256") != protocol["checkpoint_sha256"]:
        reasons.append("body_checkpoint_sha_mismatch")
    configuration_sha = S._sha256_file(paths["configuration_path"])
    if body.get("configuration", {}).get("sha256") != configuration_sha:
        reasons.append("body_configuration_sha_mismatch")
    readback = body.get("collector_report", {})
    bindings = readback.get("source_binding", {})
    expected_bindings = {
        "task_sha256": validated["source_receipt"]["task_binding_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "configuration_sha256": configuration_sha,
        "observer_sha256": protocol["source_files"]["src/rl_triage/runtime_observers.py"],
        "registry_sha256": protocol["registry_sha256"],
    }
    if bindings != expected_bindings:
        reasons.append("collector_source_binding_mismatch")
    verdict = S.assess_smoke(readback, body.get("sdk_clock_receipt", {}), protocol)
    if verdict["smoke_status"] != "pass" or body.get("smoke_validation") != verdict:
        reasons.append("measurement_or_clock_validation_failed")
    if body.get("smoke_status") != "pass" or body.get("measurement_status") != readback.get("measurement_status"):
        reasons.append("body_measurement_verdict_mismatch")
    if (body.get("sdk_versions", {}).get("isaaclab_commit") != S.ISAACLAB_COMMIT or
            body.get("sdk_versions", {}).get("isaacsim") != protocol["isaacsim_version"]):
        reasons.append("body_sdk_version_mismatch")
    return reasons


def run(tag: str, isaaclab: Path, checkpoint: Path) -> dict:
    if os.name != "nt":
        raise RuntimeError("native Windows launcher required")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) is None:
        raise ValueError("tag must use lowercase letters, digits, underscores and hyphens")
    isaaclab = isaaclab.resolve()
    checkpoint = checkpoint.resolve()
    folder = ROOT / "bench/private/runtime_readback" / tag
    if folder.exists():
        raise FileExistsError("use a new tag; an existing run is never overwritten")
    protocol = S._read_json(PROTOCOL)
    if set(protocol["source_files"]) != SOURCE_FILES:
        raise ValueError("required source set differs from the fixed launcher")
    source_receipt = S._validate_protocol(protocol, repo_root=ROOT, isaaclab_root=isaaclab)
    project_commit = verify_committed(ROOT, SOURCE_FILES | {PROTOCOL.relative_to(ROOT).as_posix()})
    if S._sha256_file(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    for argument in (str(checkpoint), str(isaaclab), str(folder), str(ROOT)):
        if any(char in argument for char in '\r\n"%&^'):
            raise ValueError("path contains unsafe Windows batch characters")
    gpu = M.query_gpu()
    if gpu["busy"]:
        raise RuntimeError("GPU workload or unclassified activity present; no process was stopped")
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    manifest = {
        "schema": "runtime_readback_smoke_manifest_v1", "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": S._sha256_file(PROTOCOL), "checkpoint_path": str(checkpoint),
        "output_path": str(folder / "body.json"), "configuration_path": str(folder / "configuration.yaml"),
    }
    write_new(manifest_path, manifest)
    validated = S.validate_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab)
    if validated["source_receipt"] != source_receipt:
        raise ValueError("source changed after the first preflight")
    write_new(folder / "preflight.json", {"project_commit_sha": project_commit, "gpu": gpu,
                                          "source_receipt": source_receipt})
    execution = run_owned(folder, [str(isaaclab / "_isaac_sim/python.bat"),
                                  str(ROOT / "bench/runtime_readback_smoke.py"), "--manifest", str(manifest_path)],
                          isaaclab, protocol["launcher_watchdog_s"])
    reasons = []
    body_sha = None
    try:
        body = S._read_json(validated["paths"]["output_path"])
        body_sha = S._sha256_file(validated["paths"]["output_path"])
        reasons.extend(verify_body(body, validated, execution, project_commit))
    except Exception as error:
        reasons.append("body_validation_" + type(error).__name__)
    if not (execution["job_assigned_before_release"] and execution["natural_exit"] and
            execution["exit_code"] == 0 and execution["cleanup_verified"] and
            not execution["timed_out"] and execution["error_type"] is None):
        reasons.append("owned_execution_did_not_complete_naturally")
    gpu_after = None
    try:
        gpu_after = M.query_gpu()
        owned = set(execution["owned_process_identities"])
        if any(str(row["pid"]) in owned for row in gpu_after["processes"]):
            reasons.append("owned_gpu_process_remaining")
    except Exception as error:
        reasons.append("post_gpu_query_" + type(error).__name__)
    receipt = {
        "schema": "runtime_readback_smoke_parent_v1", "scope": protocol["scope"],
        "status": "invalid" if reasons else "pass", "validation_reasons": reasons,
        "project_commit_sha": project_commit, "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": S._sha256_file(manifest_path), "body_sha256": body_sha,
        "execution": execution, "gpu_after": gpu_after, "quality_or_recovery_evidence": False,
    }
    write_new(folder / "parent_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.isaaclab / (
        "logs/rsl_rl/unitree_go2_flat/2026-09-26_11-31-54_s02_baseline_s42/model_99.pt")
    receipt = run(args.tag, args.isaaclab, checkpoint)
    print(json.dumps({"status": receipt["status"], "execution": receipt["execution"],
                      "validation_reasons": receipt["validation_reasons"]}, ensure_ascii=False))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
