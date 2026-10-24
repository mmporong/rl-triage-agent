"""재개 가능한 PPO 모델 결정을 소유 WSL worker와 정리 receipt에 연결한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_resumable_model_v1 as MODEL

SCRIPT = "evals/ppo_resumable_model_v1.py"
TIMEOUT_S = 1000
RECEIPT_SCHEMA = "ppo_resumable_transport_receipt_v1"


class CleanupUnverified(RuntimeError):
    pass


def wsl_path(path: Path) -> str:
    value = Path(path).resolve().as_posix()
    if not re.fullmatch(r"[A-Za-z]:/.*", value):
        raise ValueError("모델 호출에는 Windows 절대 경로가 필요하다")
    return f"/mnt/{value[0].lower()}/{value[3:]}"


def _inventory(folder: Path) -> tuple[int, Path, Path]:
    root = Path(folder) / "transport"
    receipts = sorted((root / "receipts").glob("*.json")) if (root / "receipts").exists() else []
    if [path.name for path in receipts] != [f"{index:04d}.json" for index in range(1, len(receipts) + 1)]:
        raise MODEL.J.JournalIntegrityError("transport receipt 순서가 다르다")
    pid_root = root / "pids"
    pids = sorted(pid_root.glob("*.json")) if pid_root.exists() else []
    if any(path.name not in {receipt.name for receipt in receipts} for path in pids):
        raise MODEL.J.JournalIntegrityError("receipt 없는 과거 worker PID 증거가 있다")
    for index, path in enumerate(receipts, 1):
        receipt = MODEL._read_object(path)
        required = {
            "schema", "ordinal", "packet_sha256", "binding_sha256", "pid_file", "pid_sha256",
            "timeout_s", "status", "cleanup", "returncode", "error_type",
        }
        allowed = required | {"cleanup_error_type"}
        if (set(receipt) not in (required, allowed)
                or receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("ordinal") != index
                or receipt.get("timeout_s") != TIMEOUT_S
                or receipt.get("status") not in {"complete", "invalid"}
                or receipt.get("cleanup") not in {
                    "natural_exit_pid_released", "verified", "unverified", "unverified_no_pid",
                }):
            raise MODEL.J.JournalIntegrityError("과거 transport receipt 형식이 다르다")
        pid_path = pid_root / path.name
        if receipt.get("pid_file") != str(pid_path.resolve()):
            raise MODEL.J.JournalIntegrityError("과거 transport PID 경로 결합이 다르다")
        if pid_path.exists():
            if receipt.get("pid_sha256") != MODEL.J.file_sha(pid_path):
                raise MODEL.J.JournalIntegrityError("과거 worker PID 증거가 변경됐다")
        elif receipt.get("pid_sha256") is not None:
            raise MODEL.J.JournalIntegrityError("과거 worker PID 증거가 사라졌다")
        if receipt["cleanup"] in {"unverified", "unverified_no_pid"} or "cleanup_error_type" in receipt:
            raise CleanupUnverified("과거 worker 정리가 확인되지 않아 새 실행을 거부한다")
        if receipt["cleanup"] == "verified" and not pid_path.is_file():
            raise MODEL.J.JournalIntegrityError("정리한 worker의 PID 소유 증거가 없다")
        if receipt["cleanup"] == "natural_exit_pid_released" and pid_path.exists():
            raise MODEL.J.JournalIntegrityError("자연 종료 receipt에 PID 파일이 남았다")
        if receipt["status"] == "complete" and (receipt["cleanup"] != "natural_exit_pid_released"
                                                   or receipt["error_type"] is not None):
            raise MODEL.J.JournalIntegrityError("완료 transport의 종료 증거가 다르다")
    ordinal = len(receipts) + 1
    return ordinal, root / "pids" / f"{ordinal:04d}.json", root / "receipts" / f"{ordinal:04d}.json"


def _cleanup_worker(pid_file: Path, packet_path: Path) -> str:
    if not pid_file.is_file():
        return "unverified_no_pid"
    code = r'''import hashlib,json,os,signal,sys,time
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); packet=Path(sys.argv[2]); pid=record["pid"]
assert set(record)=={"pid","start_ticks","packet_sha256"} and type(pid) is int and pid>1
assert hashlib.sha256(packet.read_bytes()).hexdigest()==record["packet_sha256"]
proc=Path("/proc")/str(pid)
def same():
 if not (proc/"stat").exists(): return False
 fields=(proc/"stat").read_text().rsplit(")",1)[1].split()
 if fields[19]!=record["start_ticks"]: raise RuntimeError("pid identity changed")
 args=[x.decode() for x in (proc/"cmdline").read_bytes().split(bytes([0])) if x]
 scripts=[i for i,x in enumerate(args) if x.endswith("/ppo_resumable_model_v1.py")]
 if len(scripts)!=1 or Path(args[scripts[0]]).name!="ppo_resumable_model_v1.py": raise RuntimeError("worker name differs")
 def value(flag):
  if args.count(flag)!=1: raise RuntimeError("worker flag differs")
  i=args.index(flag)
  if i+1>=len(args): raise RuntimeError("worker flag value missing")
  return args[i+1]
 if value("--packet")!=str(packet) or value("--pid-file")!=str(Path(sys.argv[1])): raise RuntimeError("worker args differ")
 return True
if same():
 os.kill(pid,signal.SIGTERM)
 for _ in range(30):
  if not proc.exists(): break
  time.sleep(.1)
 if same(): os.kill(pid,signal.SIGKILL)
 for _ in range(30):
  if not proc.exists(): break
  time.sleep(.1)
 if proc.exists(): raise RuntimeError("worker remains")
'''
    command = '"$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.join(
        [code, wsl_path(pid_file), wsl_path(packet_path)])
    subprocess.run(
        ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return "verified"


def _receipt_base(ordinal: int, packet_path: Path, binding_path: Path, pid_file: Path) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "ordinal": ordinal,
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "pid_file": str(pid_file.resolve()),
        "pid_sha256": None,
        "timeout_s": TIMEOUT_S,
        "status": "invalid",
        "cleanup": "unverified",
        "returncode": None,
        "error_type": None,
    }


def model_decision(folder: Path, packet: dict, binding: dict) -> dict:
    """한 번의 journal advance를 WSL worker로 실행하고 자식 소멸을 확인한다."""
    root = Path(folder)
    if root.is_symlink():
        raise MODEL.J.JournalIntegrityError("model transport folder symlink를 거부한다")
    root.mkdir(parents=True, exist_ok=True)
    packet_path, binding_path = root / "packet.json", root / "binding.json"
    MODEL._preserve(packet_path, packet)
    MODEL._preserve(binding_path, binding)
    ordinal, pid_file, receipt_path = _inventory(root)
    if pid_file.exists():
        raise MODEL.J.JournalIntegrityError("새 advance PID 경로가 이미 있다")
    argv = [wsl_path(ROOT / SCRIPT), "--folder", wsl_path(root / "worker"),
            "--packet", wsl_path(packet_path), "--binding", wsl_path(binding_path),
            "--pid-file", wsl_path(pid_file)]
    command = ('. "$HOME/.config/nvidia/env" && '
               'exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(argv))
    receipt = _receipt_base(ordinal, packet_path, binding_path, pid_file)
    try:
        completed = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        receipt["returncode"] = completed.returncode
        if pid_file.exists():
            receipt["cleanup"] = _cleanup_worker(pid_file, packet_path)
            raise CleanupUnverified("자연 종료 뒤 worker PID 파일이 남았다")
        receipt["cleanup"] = "natural_exit_pid_released"
        if completed.returncode != 0:
            raise RuntimeError("모델 worker가 오류로 종료됐다")
        result = json.loads(completed.stdout, parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"JSON constant 거부: {raw}")))
        if not isinstance(result, dict):
            raise TypeError("모델 worker 출력은 JSON object여야 한다")
        receipt["status"] = "complete"
        return result
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        receipt["error_type"] = type(error).__name__
        try:
            receipt["cleanup"] = _cleanup_worker(pid_file, packet_path)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as cleanup_error:
            receipt["cleanup_error_type"] = type(cleanup_error).__name__
        if receipt["cleanup"] != "verified":
            raise CleanupUnverified("모델 worker 정리를 확인하지 못했다") from error
        if isinstance(error, KeyboardInterrupt):
            raise
        raise TimeoutError("모델 결정 시간 상한 초과; 예약 원장을 보존했다") from None
    except BaseException as error:
        if receipt["error_type"] is None:
            receipt["error_type"] = type(error).__name__
        raise
    finally:
        receipt["pid_sha256"] = MODEL.J.file_sha(pid_file) if pid_file.is_file() else None
        MODEL._write_new(receipt_path, receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args(argv)
    result = model_decision(args.folder, MODEL._read_object(args.packet), MODEL._read_object(args.binding))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
