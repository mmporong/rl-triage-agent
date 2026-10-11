"""검증된 기존 체크포인트의 C01 증거 영상을 격리된 경로에 녹화한다."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable


TAG_RE = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,47})\Z")
DEFAULT_PROTOCOL = Path(__file__).with_name("protocols") / "evidence_media_c01_v1.json"
CREATE_NO_WINDOW = 0x08000000
EXPECTED_TASK = "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
EXPECTED_CLIP_IDS = {"before", "true_rollback", "benign_rollback"}
EXPECTED_PROTOCOL_ID = "evidence_media_c01_v1"
EXPECTED_VIDEO_CONTRACT = {
    "width": 1280,
    "height": 720,
    "frame_rate": "50/1",
    "frame_count": 299,
    "duration_s": 5.98,
    "duration_tolerance_s": 0.02,
}
EXPECTED_DISALLOWED_CLAIMS = {
    "normal_walking_policy", "ai_improvement_completed", "fresh_v3_event", "retraining_event",
}


class PreflightError(RuntimeError):
    """실행 전 무결성 또는 자원 게이트 실패."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_file(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol_id", "task", "seed", "num_envs", "video_length", "kit_args",
        "play_py_sha256", "recorder_source", "video_contract", "clips", "public_sources", "interpretation",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise PreflightError(f"protocol fields missing: {', '.join(missing)}")
    if (
        data["schema_version"] != 1
        or data["protocol_id"] != EXPECTED_PROTOCOL_ID
        or data["task"] != EXPECTED_TASK
        or data["seed"] != 42
        or data["num_envs"] != 16
        or data["video_length"] != 300
        or data["kit_args"] != "--/app/vulkan=false --/app/window/hideUi=true"
        or data["video_contract"] != EXPECTED_VIDEO_CONTRACT
    ):
        raise PreflightError("protocol recording invariants changed")
    if len(data["clips"]) != 3 or {clip.get("id") for clip in data["clips"]} != EXPECTED_CLIP_IDS:
        raise PreflightError("protocol clip set changed")
    interpretation = data["interpretation"]
    if (
        interpretation.get("scope") != "old_checkpoint_playback"
        or interpretation.get("same_case_as_active") is not False
        or interpretation.get("retraining_shown") is not False
        or set(interpretation.get("disallowed_claims", [])) != EXPECTED_DISALLOWED_CLAIMS
        or interpretation.get("true_rollback") != {"survival_s": 19.9446, "horizon_s": 20.0, "fall_frac": 0.0024, "label": "healthy"}
        or interpretation.get("benign_rollback") != {"survival_s": 1.0, "horizon_s": 1.0, "fall_frac": 0.0, "label": "undetermined"}
    ):
        raise PreflightError("protocol interpretation contract changed")
    return data


def validate_tag(tag: str) -> str:
    if not TAG_RE.fullmatch(tag):
        raise PreflightError("tag must match [a-z0-9][a-z0-9_-]{0,47}")
    return tag


def _check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise PreflightError(f"missing {label}")
    actual = sha256_file(path)
    if actual != expected:
        raise PreflightError(f"{label} digest mismatch")


def inspect_inputs(protocol: dict[str, Any], repo: Path, isaac_lab: Path) -> list[dict[str, Any]]:
    play_py = isaac_lab / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    launcher = isaac_lab / "_isaac_sim" / "python.bat"
    _check_hash(play_py, protocol["play_py_sha256"], "play.py")
    if not launcher.is_file():
        raise PreflightError("missing Isaac Sim Python launcher")
    recorder_relative = Path(protocol["recorder_source"]["path"])
    if recorder_relative.as_posix() != "bench/record_evidence_media.py":
        raise PreflightError("recorder source path changed")
    if sha256_lf_file(repo / recorder_relative) != protocol["recorder_source"]["sha256_lf"]:
        raise PreflightError("recorder source digest mismatch")

    for source in protocol["public_sources"]:
        relative = Path(source["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PreflightError("public source path must stay inside repository")
        source_path = repo / relative
        if not source_path.is_file():
            raise PreflightError(f"missing public source {relative.as_posix()}")
        if sha256_lf_file(source_path) != source["sha256_lf"]:
            raise PreflightError(f"public source {relative.as_posix()} digest mismatch")

    resolved = []
    runs_root = isaac_lab / "logs" / "rsl_rl" / "unitree_go2_flat"
    for clip in protocol["clips"]:
        for component in (clip["id"], clip["run_basename"], clip["checkpoint"]):
            if component != Path(component).name or component in {".", ".."}:
                raise PreflightError("clip path component is invalid")
        run_dir = runs_root / clip["run_basename"]
        checkpoint = run_dir / clip["checkpoint"]
        params_dir = run_dir / "params"
        env_params = params_dir / "env.yaml"
        agent_params = params_dir / "agent.yaml"
        _check_hash(checkpoint, clip["checkpoint_sha256"], f"checkpoint {clip['id']}")
        _check_hash(env_params, clip["params_env_sha256"], f"env params {clip['id']}")
        _check_hash(agent_params, clip["params_agent_sha256"], f"agent params {clip['id']}")
        resolved.append({"clip": clip, "checkpoint": checkpoint, "params_dir": params_dir})
    return resolved


def query_gpu() -> dict[str, Any]:
    summary = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    rows = [row.strip() for row in summary.stdout.splitlines() if row.strip()]
    if not rows:
        raise PreflightError("nvidia-smi returned no GPU")
    utilization, memory_used = (int(value.strip()) for value in rows[0].split(",", 1))
    workload_pattern = re.compile(r"(?:llama|ollama|python|isaac|kit|cuda)", re.IGNORECASE)
    compute_processes = sum(1 for row in apps.stdout.splitlines() if workload_pattern.search(row))
    return {
        "busy": compute_processes > 0 or utilization >= 10 or memory_used >= 4096,
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
        "compute_process_count": compute_processes,
    }


def preflight(
    protocol: dict[str, Any],
    repo: Path,
    isaac_lab: Path,
    gpu_probe: Callable[[], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = inspect_inputs(protocol, repo, isaac_lab)
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise PreflightError(f"missing tool: {tool}")
    try:
        gpu = (gpu_probe or query_gpu)()
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise PreflightError("GPU status unavailable") from error
    return resolved, gpu


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _replace_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise PreflightError("status temporary file already exists")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _set_status(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        _replace_status(path, payload)
    else:
        _write_new_json(path, payload)


def reserve_output(repo: Path, tag: str, gpu: dict[str, Any]) -> tuple[Path, Path] | None:
    public_dir = repo / "docs" / "media" / f"recovery-c01-{tag}"
    private_dir = repo / "bench" / "private" / f"media_capture_{tag}"
    if private_dir.exists():
        raise PreflightError("private capture tag already exists")
    if public_dir.exists():
        entries = list(public_dir.iterdir())
        waiting = public_dir / "status.json"
        if entries != [waiting] or json.loads(waiting.read_text(encoding="utf-8")).get("state") != "waiting_gpu_idle":
            raise PreflightError("public media tag already exists")
        if gpu["busy"]:
            return None
        return public_dir, private_dir
    if gpu["busy"]:
        _write_new_json(public_dir / "status.json", {"state": "waiting_gpu_idle", "reason": "gpu_busy"})
        return None
    public_dir.mkdir(parents=True)
    return public_dir, private_dir


class WindowsJob:
    """Owns the launched process tree and terminates only that assigned tree on close."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise PreflightError("recording is supported only on Windows")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class BASIC_ACCOUNTING(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong), ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(self._handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        self._accounting_type = BASIC_ACCOUNTING

    def assign(self, process: subprocess.Popen[Any]) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, wintypes.HANDLE(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def active_processes(self) -> int:
        info = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(self._handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ActiveProcesses)

    def terminate(self, exit_code: int = 124) -> None:
        if not self._kernel32.TerminateJobObject(self._handle, exit_code):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait_empty(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.active_processes() == 0:
                return True
            time.sleep(0.05)
        return self.active_processes() == 0

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    data = json.loads(result.stdout)
    if len(data.get("streams", [])) != 1:
        raise RuntimeError("expected exactly one video stream")
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_read_frames"]),
        "duration_s": float(data["format"]["duration"]),
    }


def _validate_video_contract(probe: dict[str, Any], contract: dict[str, Any]) -> None:
    exact = ("width", "height", "frame_rate", "frame_count")
    if any(probe.get(field) != contract.get(field) for field in exact):
        raise RuntimeError("recorded video does not match frame contract")
    if abs(probe["duration_s"] - contract["duration_s"]) > contract["duration_tolerance_s"]:
        raise RuntimeError("recorded video does not match duration contract")


def _worker_main(command_file: Path) -> int:
    if sys.stdin.buffer.readline() != b"GO\n":
        return 125
    command = json.loads(command_file.read_text(encoding="utf-8"))
    completed = subprocess.run(
        command["args"],
        cwd=command["cwd"],
        creationflags=CREATE_NO_WINDOW,
    )
    return completed.returncode


def _record_clip(protocol: dict[str, Any], item: dict[str, Any], isaac_lab: Path, private_dir: Path, timeout_s: int) -> dict[str, Any]:
    clip = item["clip"]
    _check_hash(item["checkpoint"], clip["checkpoint_sha256"], f"checkpoint before launch {clip['id']}")
    _check_hash(item["params_dir"] / "env.yaml", clip["params_env_sha256"], f"env params before launch {clip['id']}")
    _check_hash(item["params_dir"] / "agent.yaml", clip["params_agent_sha256"], f"agent params before launch {clip['id']}")
    clip_dir = private_dir / clip["id"]
    clip_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = clip_dir / clip["checkpoint"]
    shutil.copy2(item["checkpoint"], checkpoint)
    shutil.copytree(item["params_dir"], clip_dir / "params")
    _check_hash(checkpoint, clip["checkpoint_sha256"], f"copied checkpoint {clip['id']}")
    _check_hash(clip_dir / "params" / "env.yaml", clip["params_env_sha256"], f"copied env params {clip['id']}")
    _check_hash(clip_dir / "params" / "agent.yaml", clip["params_agent_sha256"], f"copied agent params {clip['id']}")

    args = [
        str(isaac_lab / "_isaac_sim" / "python.bat"),
        "scripts/reinforcement_learning/rsl_rl/play.py", "--task", protocol["task"],
        "--num_envs", str(protocol["num_envs"]), "--seed", str(protocol["seed"]),
        "--headless", "--video", "--video_length", str(protocol["video_length"]),
        f"--kit_args={protocol['kit_args']}", "--checkpoint", str(checkpoint),
        *clip["overrides"],
    ]
    command_file = clip_dir / "launch.json"
    _write_new_json(command_file, {"args": args, "cwd": str(isaac_lab)})
    stdout_path, stderr_path = clip_dir / "play.stdout.log", clip_dir / "play.stderr.log"
    job = WindowsJob()
    started = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    assigned = False
    cleanup_verified = False
    residual_processes = 0
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--_worker", str(command_file)],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                creationflags=CREATE_NO_WINDOW,
            )
            try:
                job.assign(process)
                assigned = True
            except BaseException as error:
                process.terminate()
                process.wait(timeout=10)
                cleanup_verified = process.poll() is not None and job.active_processes() == 0
                _write_new_json(clip_dir / "execution.json", {
                    "exit_code": 126,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "job_assigned": False,
                    "own_pid_cleanup": cleanup_verified,
                    "residual_processes_observed": 0,
                })
                raise RuntimeError(f"could not assign owned launcher for {clip['id']}") from error
            assert process.stdin is not None
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            try:
                exit_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                exit_code = 124
            if exit_code == 0:
                cleanup_verified = job.wait_empty(5)
                if not cleanup_verified:
                    residual_processes = job.active_processes()
                    job.terminate(125)
                    job.wait_empty(10)
            else:
                job.terminate(exit_code)
                cleanup_verified = job.wait_empty(10)
    finally:
        if assigned and job.active_processes() > 0:
            residual_processes = max(residual_processes, job.active_processes())
            job.terminate(125)
            cleanup_verified = job.wait_empty(10) and cleanup_verified
        job.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
    record = {
        "exit_code": exit_code,
        "elapsed_s": round(time.monotonic() - started, 3),
        "job_assigned": assigned,
        "own_pid_cleanup": cleanup_verified,
        "residual_processes_observed": residual_processes,
    }
    _write_new_json(clip_dir / "execution.json", record)
    if not cleanup_verified or residual_processes:
        raise RuntimeError(f"owned process cleanup was not clean for {clip['id']}")
    if exit_code != 0:
        raise RuntimeError(f"recording failed for {clip['id']} exit={exit_code}")
    videos = sorted((clip_dir / "videos" / "play").glob("*.mp4"), key=lambda path: path.stat().st_mtime_ns)
    if len(videos) != 1:
        raise RuntimeError(f"expected one video for {clip['id']}, found {len(videos)}")
    video_probe = _probe_video(videos[0])
    _validate_video_contract(video_probe, protocol["video_contract"])
    return {"clip": clip, "video": videos[0], "probe": video_probe, **record}


def run_recording(
    protocol: dict[str, Any],
    repo: Path,
    isaac_lab: Path,
    tag: str,
    timeout_s: int,
    gpu_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved, gpu = preflight(protocol, repo, isaac_lab, gpu_probe)
    reserved = reserve_output(repo, tag, gpu)
    if reserved is None:
        return {"state": "waiting_gpu_idle", "reason": "gpu_busy"}
    public_dir, private_dir = reserved
    records = []
    for item in resolved:
        try:
            clip_gpu = (gpu_probe or query_gpu)()
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise PreflightError("GPU status unavailable before clip") from error
        if clip_gpu["busy"]:
            status = public_dir / "status.json"
            if records:
                _set_status(status, {
                    "state": "aborted_gpu_busy",
                    "reason": "gpu_busy_between_clips",
                    "partial_private_clips": len(records),
                    "retry_requires_new_tag": True,
                })
                return {"state": "aborted_gpu_busy", "reason": "gpu_busy_between_clips"}
            if not status.exists():
                _write_new_json(status, {"state": "waiting_gpu_idle", "reason": "gpu_busy"})
            return {"state": "waiting_gpu_idle", "reason": "gpu_busy"}
        if not private_dir.exists():
            private_dir.mkdir(parents=True, exist_ok=False)
        records.append(_record_clip(protocol, item, isaac_lab, private_dir, timeout_s))

    public_clips = []
    for record in records:
        clip = record["clip"]
        destination = public_dir / f"{clip['id']}.mp4"
        with record["video"].open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        public_clips.append({
            "id": clip["id"],
            "run_basename": clip["run_basename"],
            "checkpoint_sha256": clip["checkpoint_sha256"],
            "params_env_sha256": clip["params_env_sha256"],
            "params_agent_sha256": clip["params_agent_sha256"],
            "task": protocol["task"],
            "flags": {"seed": protocol["seed"], "num_envs": protocol["num_envs"], "video_length": protocol["video_length"], "kit_args": protocol["kit_args"], "overrides": clip["overrides"]},
            "file": destination.name,
            "size_bytes": destination.stat().st_size,
            "clip_sha256": sha256_file(destination),
            "ffprobe": record["probe"],
            "execution_exit": record["exit_code"],
            "own_pid_cleanup": record["own_pid_cleanup"],
        })
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol.get("protocol_id", "evidence_media_c01_v1"),
        "scope": "old_checkpoint_playback",
        "interpretation": protocol["interpretation"],
        "same_case_as_active": False,
        "retraining_shown": False,
        "disallowed_claims": protocol["interpretation"]["disallowed_claims"],
        "clips": public_clips,
        "source_bindings": protocol["public_sources"],
        "implementation_bindings": {
            "protocol": {"path": "bench/protocols/evidence_media_c01_v1.json", "sha256_lf": sha256_lf_file(DEFAULT_PROTOCOL)},
            "recorder": protocol["recorder_source"],
            "play_py_sha256": protocol["play_py_sha256"],
        },
    }
    _write_new_json(public_dir / "source_manifest.json", manifest)
    status_path = public_dir / "status.json"
    complete_status = {"state": "complete", "clips": len(public_clips)}
    _set_status(status_path, complete_status)
    return {"state": "complete", "clips": len(public_clips)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--isaac-lab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--timeout-s", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if len(raw_argv) == 2 and raw_argv[0] == "--_worker":
        return _worker_main(Path(raw_argv[1]))
    args = parse_args(raw_argv)
    try:
        tag = validate_tag(args.tag)
        if args.timeout_s < 60:
            raise PreflightError("timeout must be at least 60 seconds")
        repo = Path(__file__).resolve().parents[1]
        protocol = load_protocol(DEFAULT_PROTOCOL.resolve())
        if args.preflight:
            resolved, gpu = preflight(protocol, repo, args.isaac_lab.resolve())
            result = {"state": "ready" if not gpu["busy"] else "waiting_gpu_idle", "inputs_verified": len(resolved), "gpu": gpu}
        else:
            result = run_recording(protocol, repo, args.isaac_lab.resolve(), tag, args.timeout_s)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["state"] in {"ready", "complete"} else 3
    except (PreflightError, FileExistsError, json.JSONDecodeError) as error:
        print(json.dumps({"state": "rejected", "reason": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
