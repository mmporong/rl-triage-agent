"""임대한 Brev VM 안에서 등록 정책 하나의 상태 smoke 또는 native 녹화를 실행한다.

클라우드 생성·Stop·자격증명은 호스트 책임이다. 이 진입점은 실제 lease를 잡고
fresh provider/GPU 원문과 source·checkpoint 계약을 기존 실행기에 전달한다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import agentic_play_job_v1 as JOB
import record_agentic_state_sidecar_v1 as SIDECAR
import run_agentic_play_record_v1 as RUN

ROOT = Path(__file__).resolve().parents[1]


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10


def admission(protocol_path: Path, folder: Path, provider: Path, provider_observed: str,
              lease_reference: dict) -> Path:
    """호스트 provider 관찰과 VM의 실제 nvidia-smi 원문을 하나의 fresh proof로 묶는다."""
    folder.mkdir(parents=True, exist_ok=False)
    for stem, command in (("gpu", JOB.REMOTE.GPU_COMMAND), ("process", JOB.REMOTE.PROCESS_COMMAND)):
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="strict", timeout=30, check=False)
        (folder / f"{stem}.csv").write_text(result.stdout, encoding="utf-8", newline="")
        (folder / f"{stem}.stderr").write_text(result.stderr, encoding="utf-8", newline="")
        JOB.write_new(folder / f"{stem}.receipt.json", {"command": command, "exit_code": result.returncode,
            "observed_at_utc": _stamp(), "stdout_sha256": JOB._sha(folder / f"{stem}.csv")})
        if result.returncode or result.stderr:
            raise RuntimeError("GPU 입장 query 실패")
    current = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    observed = min(JOB._utc(provider_observed, "provider observed"), current)
    expires = min(JOB._utc(JOB._read(protocol_path)["resources"]["deadline_utc"], "deadline"),
                  observed + timedelta(seconds=120))
    proof = JOB.build_resource_proof(protocol_path, provider_inventory_path=provider,
        nvidia_gpu_csv_path=folder / "gpu.csv", nvidia_process_csv_path=folder / "process.csv",
        observed_at_utc=observed.isoformat(), expires_at_utc=expires.isoformat(),
        lease_reference=lease_reference, repo=ROOT)
    path = folder / "resource.json"
    JOB.write_new(path, proof)
    # 종료 뒤에도 실제 측정 시점의 OS lock과 heartbeat 원문을 감사할 수 있게 남긴다.
    snapshot = JOB._read(Path(lease_reference["path"]))
    (folder / "proc-locks.txt").write_bytes(Path("/proc/locks").read_bytes())
    (folder / "owner-stat.txt").write_bytes(Path(f"/proc/{snapshot['owner']['pid']}/stat").read_bytes())
    paths = JOB.lease_paths(snapshot["resource_id"])
    (folder / "heartbeat.json").write_bytes(paths["heartbeat"].read_bytes())
    (folder / "lease-snapshot.json").write_bytes(Path(lease_reference["path"]).read_bytes())
    return path


def _state_smoke(validated: dict, stage: Path, reader) -> dict:
    """App close가 exit code를 바꿔도 결과 파일과 소유 process 종료를 따로 확인한다."""
    output = stage / "state-smoke"
    args = [validated["job"]["python"], "-B", str(ROOT / "bench/record_agentic_state_sidecar_v1.py"),
        "state-dry-run", "--checkpoint", str(validated["checkpoint_path"]), "--checkpoint-sha256",
        validated["checkpoint_sha256"], "--source-id", validated["job"]["job_id"] + "-state-smoke",
        "--out", str(output), "--frames", "1"]
    if reader():
        raise RuntimeError("상태 smoke 직전 GPU가 유휴 상태가 아니다")
    JOB.write_new(stage / "state-smoke-launch.json", {"args": args, "cwd": str(validated["repo"])})
    process = None
    identity = None
    cleanup = False
    try:
        with (stage / "state-smoke.stdout").open("xb") as stdout, (stage / "state-smoke.stderr").open("xb") as stderr:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            env.pop("NVIDIA_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)
            process = subprocess.Popen(args, cwd=validated["repo"], stdout=stdout, stderr=stderr,
                                       env=env, start_new_session=True)
            identity = JOB.OWNED._process_identity(process.pid)
            process.wait(timeout=180)
    finally:
        if process is not None and identity is not None:
            cleanup = JOB.OWNED._safe_cleanup(identity, process)
        if process is not None:
            JOB.write_new(stage / "state-smoke-execution.json", {"exit_code": process.poll(),
                "owned_identity": identity, "cleanup_verified": cleanup, "app_result_required": True})
    result_path = output / "dry_run_result.json"
    if (process is None or process.returncode != 0 or not cleanup or not result_path.is_file()
            or (output / "dry_run_failure.json").exists()):
        raise RuntimeError("상태 smoke의 결과·종료 증거가 완성되지 않았다")
    result = JOB._read(result_path)
    if result.get("completed") is not True or result.get("frames") != 1:
        raise RuntimeError("상태 smoke는 완료되지 않았다")
    manifest, records = SIDECAR.load_sidecar(output / "sidecar")
    if manifest["source"]["checkpoint_sha256"] != validated["checkpoint_sha256"] or len(records) != 1:
        raise RuntimeError("상태 smoke checkpoint·frame 결합")
    if reader():
        raise RuntimeError("상태 smoke 종료 뒤 GPU process가 남았다")
    return {"mode": "state-smoke", "completed": True, "frames": 1, "cleanup_verified": cleanup,
            "result_sha256": JOB._sha(result_path), "behavior_label_created": False}


def execute(args: argparse.Namespace) -> dict:
    if os.name != "posix":
        raise RuntimeError("실제 capture unit은 소유 Linux VM에서 실행한다")
    protocol = JOB.validate_protocol(args.protocol)
    if args.mode == "state-smoke" and protocol["playback_registrations"][args.logical_id]["fault_id"] != "NONE":
        raise ValueError("최소 상태 smoke는 등록된 NONE 대조 정책으로 실행한다")
    stage = args.stage.absolute()
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    started = time.monotonic()
    with JOB.resource_lease(args.protocol, repo=ROOT, ttl_s=900) as lease:
        proof = admission(args.protocol, stage.parent / (stage.name + "-admission"),
                          args.provider_inventory, args.provider_observed_at, lease)
        job = JOB.create_job(args.protocol, proof, JOB._sha(proof), args.isaaclab, args.python,
                             stage, args.logical_id)
        validated = JOB.validate_job(job)
        JOB.write_new(stage / "unit-intent.json", {"mode": args.mode, "source_commit": protocol["source_commit"],
            "protocol_sha256_lf": JOB._sha(args.protocol, lf=True), "job_sha256": JOB._sha(job),
            "checkpoint_sha256": validated["checkpoint_sha256"], "lease": lease})
        try:
            if args.mode == "state-smoke":
                proof_body = JOB._read(proof)
                reader = JOB.REMOTE.StrictRemoteGpuReader(proof_body["capture"], stage / "state-gpu-observations")
                result = _state_smoke(validated, stage, reader)
            else:
                result = RUN.run_job(job)
                if result.get("acceptable_as_live_evidence") is not True:
                    raise RuntimeError("native·sidecar·join·playback가 실제 완료되지 않았다")
        except BaseException as error:  # 원문/부분 산출물을 보존하고 실패를 종료 전에 기록한다.
            JOB.write_new(stage / "unit-failure.json", {"schema": "agentic_capture_unit_failure_v1",
                "error_type": type(error).__name__, "error_message": str(error), "mode": args.mode,
                "wall_s": time.monotonic() - started, "completed": False})
            raise
        value = {"schema": "agentic_capture_unit_result_v1", "mode": args.mode, "completed": True,
                 "wall_s": time.monotonic() - started, "result": result, "new_training": 0, "model_API": 0}
        JOB.write_new(stage / "unit-result.json", value)
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["state-smoke", "record"], required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--provider-inventory", required=True, type=Path)
    parser.add_argument("--provider-observed-at", required=True)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--logical-id", required=True)
    print(json.dumps(execute(parser.parse_args()), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
