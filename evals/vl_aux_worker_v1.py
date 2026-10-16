"""WSL 모델 worker를 PID·생성 시점으로 묶어 호출하고 종료를 확인한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

MODEL = "nvidia/nemotron-3-super-120b-a12b"


def wsl_path(path: Path) -> str:
    value = Path(path).resolve().as_posix()
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("local Windows drive path required")
    return f"/mnt/{value[0].lower()}/{value[3:]}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cleanup_script() -> str:
    return '''import hashlib,json,os,signal,sys,time
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text()); packet=Path(sys.argv[2]); pid=record['pid']
assert type(pid) is int and pid>1
assert hashlib.sha256(packet.read_bytes()).hexdigest()==record['packet_sha256']
proc=Path('/proc')/str(pid)
def same():
    if not (proc/'stat').exists(): return False
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    if fields[19]!=record['start_ticks']: raise RuntimeError('pid identity changed')
    args=(proc/'cmdline').read_bytes().split(bytes([0]))
    if not any(x.endswith(b'/vl_aux_worker_v1.py') for x in args) or str(packet).encode() not in args: raise RuntimeError('worker args differ')
    return True
was_running=same()
if was_running:
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not proc.exists(): break
        time.sleep(0.1)
    if same(): os.kill(pid,signal.SIGKILL)
    for _ in range(30):
        if not proc.exists(): break
        time.sleep(0.1)
    if proc.exists(): raise RuntimeError('worker remains')
print(json.dumps({'cleanup_verified':True,'was_running':was_running}))
'''


def stop_worker(pid_file: Path, packet: Path, *, runner=subprocess.run) -> dict:
    if not pid_file.is_file():
        raise RuntimeError("worker identity unavailable; no process was terminated")
    command = ('"$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.join(
        [cleanup_script(), wsl_path(pid_file), wsl_path(packet)]))
    process = runner(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                     capture_output=True, encoding="utf-8", errors="replace", timeout=15)
    if process.returncode != 0:
        raise RuntimeError("owned model worker cleanup not verified")
    value = json.loads(process.stdout)
    if value.get("cleanup_verified") is not True:
        raise RuntimeError("owned model worker remains")
    return value


def invoke_worker(kind: str, packet: Path, output: Path, ledger_root: Path | None = None,
                  *, runner=subprocess.run) -> None:
    if kind not in {"diagnosis", "correction", "observation"}:
        raise ValueError("unregistered worker kind")
    output, packet = Path(output), Path(packet)
    pid_file = output.with_suffix(".pid.json")
    if output.exists() or pid_file.exists() or not packet.is_file():
        raise ValueError("model output exists or packet missing")
    argv = [wsl_path(Path(__file__)), "--kind", kind, "--packet", wsl_path(packet),
            "--output", wsl_path(output), "--pid-file", wsl_path(pid_file)]
    if kind == "observation":
        if ledger_root is None:
            raise ValueError("observation ledger required")
        argv += ["--ledger-root", wsl_path(ledger_root)]
    command = '. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(argv)
    try:
        process = runner(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                         capture_output=True, encoding="utf-8", errors="replace", timeout=700)
    except subprocess.TimeoutExpired:
        stop_worker(pid_file, packet, runner=runner)
        raise TimeoutError("model worker timeout; owned process cleanup verified") from None
    if pid_file.is_file():
        receipt = stop_worker(pid_file, packet, runner=runner)
        if receipt["was_running"]:
            raise RuntimeError("model worker did not terminate with its launcher")
    else:
        raise RuntimeError("model worker identity record missing")
    if process.returncode not in ({0} if kind == "observation" else {0, 2}) or not output.is_file():
        raise RuntimeError("model worker did not return a saved result")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("diagnosis", "correction", "observation"), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise RuntimeError("worker runs in registered WSL Linux only")
    identity = {"pid": os.getpid(), "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "packet_sha256": _sha(args.packet)}
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    with args.pid_file.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream)
    # SDK 요청 300초 × 최대 두 번보다 길고 부모 700초보다 짧은 최종 종료 상한.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(650)
    if args.kind == "observation":
        import vl_aux_observer_v1 as observer
        observer.call(args.packet, args.output, args.ledger_root)
        code = 0
    else:
        import feedback_model_call as feedback
        code = feedback.call(args.kind, args.packet, args.output, MODEL)
    signal.alarm(0)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
