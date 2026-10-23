"""D4 조건부 대화를 커밋 봉인과 소유 WSL worker 종료 증거에 연결한다."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import selection_comparison_conditional_call_v1 as API
import selection_comparison_controller_v1 as CONTROL
import selection_comparison_demo_v1 as DEMO

ROOT = DEMO.ROOT
SCRIPT = "evals/selection_comparison_worker_v1.py"
MC = CONTROL.CALL.MC
require = CONTROL.require


def _paths(root: Path, receipt: dict) -> dict[str, Path]:
    ordinal = receipt["ordinal"]
    require(type(ordinal) is int and 1 <= ordinal <= 20, "worker slot ordinal")
    return {
        "capsule": root / "capsules" / f"{receipt['capsule_sha256']}.json",
        "pid": root / "workers" / f"{ordinal:02d}.pid.json",
        "transport": root / "workers" / f"{ordinal:02d}.transport.json",
    }


def _worker_args(root: Path, receipt: dict, anchor: str, protocol_sha: str) -> list[str]:
    paths = _paths(root, receipt)
    return [MC.wsl_path(ROOT / SCRIPT), "--root", MC.wsl_path(root),
            "--slot-id", receipt["slot_id"], "--anchor", anchor,
            "--protocol-sha", protocol_sha,
            "--root-digest", CONTROL.CALL.root_digest(root),
            "--capsule", MC.wsl_path(paths["capsule"]), "--pid-file", MC.wsl_path(paths["pid"]),
            "--approve-model-call"]


def cleanup_worker(pid_file: Path, capsule_path: Path) -> str:
    """새 worker argv·PID 시작 시각·capsule SHA가 같은 소유 프로세스만 정리한다."""
    if not pid_file.is_file():
        return "unverified_no_pid"
    code = r'''import hashlib,json,os,signal,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text()); capsule=Path(sys.argv[2]); pid=r['pid']; proc=Path('/proc')/str(pid)
assert type(pid) is int and pid>1 and hashlib.sha256(capsule.read_bytes()).hexdigest()==r['packet_sha256']
def same():
 if not (proc/'stat').exists(): return False
 fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
 if int(fields[19])!=r['start_ticks']: raise RuntimeError('pid identity changed')
 args=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 script=next((i for i,x in enumerate(args) if x.endswith('/selection_comparison_worker_v1.py')),None)
 if script is None or args[script:]!=r['argv'] or str(capsule) not in args: raise RuntimeError('worker args differ')
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
 if proc.exists(): raise RuntimeError('worker remains')
'''
    command = '"$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.join(
        [code, MC.wsl_path(pid_file), MC.wsl_path(capsule_path)])
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command], check=True,
                   capture_output=True, timeout=15)
    return "verified"


def launch(root: Path, plan: dict, receipt: dict, protocol: dict, anchor: str, timeout_s: float) -> dict:
    require(os.name == "nt", "실제 supervisor는 Windows에서 실행한다")
    require(0 < timeout_s <= MC.TRANSPORT_TIMEOUT_S, "worker timeout 범위")
    root = root.resolve()
    protocol_sha = DEMO.sha(DEMO.PROTOCOL, True)
    DEMO.verify_repository(protocol, anchor, protocol_sha)
    require(protocol["api"]["slot_plan"] == plan and receipt["disposition"] == "executed",
            "실행 slot과 사전등록 결합")
    paths = _paths(root, receipt)
    require(not paths["pid"].exists() and not paths["transport"].exists(), "이미 시작한 worker")
    command = '. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(
        _worker_args(root, receipt, anchor, protocol_sha))
    transport = {"schema": "selection_conditional_transport_v1", "slot_id": receipt["slot_id"],
                 "capsule_sha256": receipt["capsule_sha256"], "status": "invalid",
                 "cleanup": "unverified", "timeout_s": timeout_s}
    try:
        process = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                                 capture_output=True, timeout=timeout_s)
        transport["returncode"] = process.returncode
        require(process.returncode == 0 and not paths["pid"].exists(), "worker 실패 또는 PID 파일 잔여")
        DEMO.verify_repository(protocol, anchor, protocol_sha)
        record = API.review_dialogue(root, plan, receipt)
        transport.update(status="complete", cleanup="natural_exit_pid_released")
        return record
    except Exception as error:
        transport["error_type"] = type(error).__name__
        if paths["pid"].exists():
            try:
                transport["cleanup"] = cleanup_worker(paths["pid"], paths["capsule"])
            except Exception as cleanup_error:
                transport["cleanup_error_type"] = type(cleanup_error).__name__
        raise
    finally:
        DEMO.write_new(paths["transport"], transport)


class ModelCaller:
    """실제 대화와 생략 slot을 분리하며 D3의 원래 snapshot으로 모델 선택을 재검산한다."""

    def __init__(self, root: Path, protocol: dict, anchor: str, budget, *, launcher=launch):
        self.root, self.protocol, self.anchor = Path(root).resolve(), protocol, anchor
        self.plan = API.validate_plan(protocol["api"]["slot_plan"])
        self.budget, self.launcher = budget, launcher
        self.terminals: dict[tuple[str, int], dict] = {}
        self.seen: dict[tuple[str, int], dict] = {}
        self.finalized = False
        API.initialize(self.root, self.plan)

    def __call__(self, snapshot: dict, condition: str, policy: str, epoch: int, ordinal: int) -> dict:
        require(not self.finalized, "종료한 caller 재사용")
        checked = CONTROL.V.checked(snapshot)
        require(checked["inputs"]["policy"] == policy and checked["inputs"]["packet"]["epoch"] == epoch,
                "caller 정책·epoch 불일치")
        observed = len(checked["view"]["observations"])
        if condition == "production":
            suffix = "first" if policy == "model_first_then_rule" else f"open{observed}"
        else:
            suffix = condition
        slot_id = f"{policy}:e{epoch}:{suffix}"
        slot = API._slot(self.plan, slot_id)
        require(type(ordinal) is int and ordinal >= 0, "논리 대화 ordinal")
        self.seen[policy, epoch] = checked
        receipt = API.record_slot(self.root, self.plan, API.materialize_slot(slot, checked))
        require(receipt["disposition"] == "executed", "실행 불필요 slot 호출")
        remaining = self.budget.remaining()["overall_elapsed_s"]
        require(remaining > 0, "전체 경과 예산 소진")
        record = self.launcher(self.root, self.plan, receipt, self.protocol, self.anchor,
                               min(float(MC.TRANSPORT_TIMEOUT_S), remaining))
        reviewed = API.review_dialogue(self.root, self.plan, receipt)
        require(record == reviewed, "worker 반환과 저장 대화 불일치")
        result = CONTROL.replay_dialogue(checked, condition, reviewed["safe_results"])
        require(result["engine"] == reviewed["engine"], "portable와 원래 snapshot의 대화 결과 불일치")
        return result

    def finish_epoch(self, policy: str, epoch: int, snapshot: dict | None, reason: str) -> None:
        require((policy, epoch) not in self.terminals, "epoch 종료 중복")
        if snapshot is not None:
            checked = CONTROL.V.checked(snapshot)
            require(checked["inputs"]["policy"] == policy
                    and checked["inputs"]["packet"]["epoch"] == epoch, "종료 snapshot 정책 결합")
            self.seen[policy, epoch] = checked
            snapshot = checked
        row = {"schema": "selection_policy_epoch_terminal_v1", "policy": policy, "epoch": epoch,
               "reason": reason, "snapshot": deepcopy(snapshot)}
        path = self.root / "terminals" / f"{policy}_e{epoch}.json"
        DEMO.write_new(path, row)
        self.terminals[policy, epoch] = {**row, "sha256": DEMO.sha(path)}

    def finalize(self, *, status: str, arms: dict) -> dict:
        require(not self.finalized, "caller finalize 중복")
        self.finalized = True
        for (policy, epoch), terminal in self.terminals.items():
            path = self.root / "terminals" / f"{policy}_e{epoch}.json"
            expected = {key: value for key, value in terminal.items() if key != "sha256"}
            require(DEMO.sha(path) == terminal["sha256"] and DEMO.read(path) == expected,
                    "epoch 종료 증거 변경")
        for slot in self.plan["slots"]:
            if (self.root / "slots" / f"{slot['ordinal']:02d}.json").exists():
                continue
            key = slot["policy"], slot["epoch"]
            terminal = self.terminals.get(key)
            snapshot = terminal["snapshot"] if terminal else self.seen.get(key)
            if snapshot is None:
                materialized = API.materialize_slot(slot, None)
            elif snapshot["view"]["status"] != "open" or snapshot["view"]["budget_left"] <= 0:
                materialized = API.materialize_slot(slot, snapshot)
                # 종료 처리에서는 생략된 보조 관측을 새로 실행하지 않는다.
                if materialized["disposition"] == "executed":
                    if status != "invalid" or terminal is None:
                        raise ValueError("보조 slot의 종료 근거 누락")
                    materialized = API.materialize_skipped_slot(slot, snapshot, "batch_invalid", terminal["sha256"])
            else:
                if terminal is None:
                    require(status == "invalid", "open epoch 종료 근거 누락")
                    self.finish_epoch(*key, snapshot, "batch_invalid")
                    terminal = self.terminals[key]
                reason = terminal["reason"]
                reason = reason if reason in API.TERMINAL_REASONS - {"view_terminal"} else "batch_invalid"
                materialized = API.materialize_skipped_slot(slot, snapshot, reason, terminal["sha256"])
            API.record_slot(self.root, self.plan, materialized)
        result = API.finalize(self.root, self.plan)
        transports = [DEMO.read(path) for path in (self.root / "workers").glob("*.transport.json")]
        receipts = [DEMO.read(path) for path in (self.root / "slots").glob("*.json")]
        executed = {row["slot_id"] for row in receipts if row["disposition"] == "executed"}
        require({row["slot_id"] for row in transports} == executed
                and len(transports) == len(executed), "worker transport inventory 불일치")
        require(all(row["status"] == "complete" and row["cleanup"] == "natural_exit_pid_released"
                    for row in transports), "완료하지 않은 worker transport")
        require(not list((self.root / "workers").glob("*.pid.json")), "worker PID 증거 잔여")
        result["controller_status"] = status
        result["policies_recorded"] = sorted(arms)
        DEMO.write_new(self.root / "finalize.json", result)
        return result


def worker(args) -> None:
    require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME"))
            and args.approve_model_call, "WSL 모델 호출 승인 필요")
    root = args.root.resolve()
    protocol = DEMO.read(DEMO.PROTOCOL)
    DEMO.verify_repository(protocol, args.anchor, args.protocol_sha)
    require(CONTROL.CALL.root_digest(root) == args.root_digest, "supervisor/worker root 결합")
    plan = API.validate_plan(protocol["api"]["slot_plan"])
    _, receipt = API._load_receipt(root, plan, args.slot_id)
    paths = _paths(root, receipt)
    require(args.capsule.resolve() == paths["capsule"] and args.pid_file.resolve() == paths["pid"],
            "worker capsule·PID 출력 경로")
    identity = MC._worker_identity(paths["capsule"])
    DEMO.write_new(paths["pid"], identity)
    try:
        API.run_dialogue(root, plan, receipt, API.NvidiaProvider())
        DEMO.verify_repository(protocol, args.anchor, args.protocol_sha)
    finally:
        require(DEMO.read(paths["pid"]) == identity, "worker PID 소유자 변경")
        paths["pid"].unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--root-digest", required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--approve-model-call", action="store_true")
    args = parser.parse_args(argv)
    try:
        worker(args)
    except Exception as error:
        print(json.dumps({"status": "invalid", "error_type": type(error).__name__}))
        return 1
    print(json.dumps({"status": "complete", "slot_id": args.slot_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
