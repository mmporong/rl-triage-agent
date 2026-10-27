"""중지된 전송 촬영 VM을 새 Start 증거로만 재사용하는 CPU 준비기."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_vision_transferred_pull_record_v3 as RUN3
import vision_pull_record_job_v1 as RECORD_JOB1
import vision_pull_record_job_v2 as RECORD_JOB2
import vision_pull_record_job_v3 as RECORD_JOB3
import vision_pull_record_job_v4 as RECORD_JOB4
import vision_pull_record_job_v5 as RECORD_JOB5
import vision_transferred_pull_job_v1 as JOB1
import vision_transferred_pull_job_v2 as JOB2
import vision_transferred_pull_job_v3 as JOB3

PROTOCOL_ID = "vision-transferred-pull-record-20261027c-v1"
TARGET = {
    "resource_id": "j1kxncj4e",
    "resource_name": "rl-triage-l40s-transfer-20261027b",
    "instance_type": "gpu-l40s-a.1gpu-16vcpu-64gb",
}
CHECKPOINT_IDS = ("source-s2221-none", "source-s2221-f1")
CHECKPOINTS = {
    "source-s2221-none": {
        "isaaclab_relative": "logs/rsl_rl/unitree_go2_flat/2026-10-26_08-17-51_vision_s2221_none/model_299.pt",
        "sha256": "b346a0c3e3ad6410ce93dcf73e6165ed69e2d6ab95b2c5398943b3a8874cf54d",
    },
    "source-s2221-f1": {
        "isaaclab_relative": "logs/rsl_rl/unitree_go2_flat/2026-10-26_08-21-05_vision_s2221_f1/model_299.pt",
        "sha256": "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731",
    },
}
SOURCE_BINDING_FILES = {
    "bench/run_vision_transferred_reuse_v1.py",
    "tests/test_run_vision_transferred_reuse_v1.py",
}
OLD_START_CUTOFF = datetime.fromisoformat("2026-10-27T08:19:49.019401+00:00")
EXPECTED_QUOTE = Decimal("2.09616")
PRIOR_UPPER = Decimal("24.454911")
FOLLOWUP_MAX = Decimal("5.000000")
PROGRAM_CAP = Decimal("35.000000")
REMOTE_REPO = "$HOME/workspace/rl-triage-record-transfer-20261027c"
REMOTE_LAB = "$HOME/workspace/rl-triage-vision-v1/IsaacLab"
REMOTE_PYTHON = "$HOME/workspace/rl-triage-vision-v1/venv/bin/python"
REMOTE_STATE = "$HOME/workspace/rl-triage-transferred-record-state-20261027c-v1"
EXPECTED_ENVIRONMENT = {
    "gymnasium": "1.2.0",
    "isaaclab": "0.41.3",
    "isaaclab_commit": "90b79bb2d44feb8d833f260f2bf37da3487180ba",
    "isaaclab_rl": "0.1.8",
    "isaacsim": "4.5.0.0",
    "pip": "26.2.1",
    "python": "3.10.12",
    "rsl_rl_lib": "2.3.3",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
}


def _utc(value: Any, anchor: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{anchor} 시각이 문자열이 아니다")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{anchor} 시각 형식이 다르다") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{anchor} 시각에 timezone이 없다")
    return result.astimezone(timezone.utc)


def _read(path: Path) -> dict[str, Any]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"JSON이 단일 regular file이 아니다: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON root가 객체가 아니다")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_lf_bytes(value: bytes) -> str:
    return hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest()


@contextmanager
def seed2221_contract() -> Iterator[None]:
    modules = (
        JOB3, JOB2, JOB1, RECORD_JOB1, RECORD_JOB2, RECORD_JOB3, RECORD_JOB4, RECORD_JOB5,
    )
    previous = [module.CHECKPOINT_IDS for module in modules]
    for module in modules:
        module.CHECKPOINT_IDS = CHECKPOINT_IDS
    try:
        yield
    finally:
        for module, value in zip(modules, previous, strict=True):
            module.CHECKPOINT_IDS = value


def _reuse_child_launcher(popen):
    """RUN5 child를 seed2221 컨텍스트를 다시 여는 reuse 진입점으로 연결한다."""
    old = str(Path(RUN3.PREVIOUS.__file__).resolve())
    current = str(Path(__file__).resolve())

    def launch(command: Sequence[str], **kwargs: Any) -> Any:
        rewritten: list[str] = []
        replaced = False
        for value in command:
            if value == old:
                rewritten.extend([current, "record"])
                replaced = True
            else:
                rewritten.append(value)
        if not replaced:
            raise ValueError("transferred pull 자식 명령에 RUN5 진입점이 없다")
        return popen(rewritten, **kwargs)

    return launch


@contextmanager
def record_contract() -> Iterator[None]:
    previous_launcher = RUN3._child_launcher
    RUN3._child_launcher = _reuse_child_launcher
    try:
        with seed2221_contract():
            yield
    finally:
        RUN3._child_launcher = previous_launcher


def validate_protocol(path: Path, *, repo: Path, now: datetime | None = None) -> dict[str, Any]:
    with seed2221_contract():
        protocol = JOB3.validate_protocol(path, repo=repo, now=now)
    if protocol["protocol_id"] != PROTOCOL_ID or protocol["recording_target"] != TARGET:
        raise ValueError("후속 protocol ID 또는 target이 고정값과 다르다")
    resources = protocol["resources"]
    if (Decimal(resources["compute_quote_usd_per_h"]) != EXPECTED_QUOTE
            or Decimal(resources["batch_budget_usd"]) != FOLLOWUP_MAX):
        raise ValueError("후속 quote 또는 batch budget이 고정값과 다르다")
    expected_rows = [dict({"logical_id": identity}, **CHECKPOINTS[identity]) for identity in CHECKPOINT_IDS]
    if protocol["checkpoints"] != expected_rows:
        raise ValueError("seed 2221 checkpoint catalog가 다르다")
    return protocol


def bind_source(protocol_path: Path, *, repo: Path, source_commit: str, output: Path) -> dict[str, Any]:
    """첫 public commit의 reuse runner/test blob을 후속 protocol draft에 결합한다."""
    repo = Path(repo).resolve()
    if JOB3.COMMON.HEX40.fullmatch(source_commit) is None:
        raise ValueError("source commit이 full SHA가 아니다")
    if JOB3.COMMON._git(repo, "merge-base", "--is-ancestor", source_commit, "HEAD").returncode:
        raise ValueError("source commit이 현재 checkout의 조상이 아니다")
    value = _read(protocol_path)
    sources = dict(value["source_sha256_lf"])
    for relative in sorted(SOURCE_BINDING_FILES):
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{source_commit}:{relative}"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if result.returncode or result.stderr:
            raise ValueError(f"source commit에 결합 파일이 없다: {relative}")
        sources[relative] = _sha_lf_bytes(result.stdout)
    value["source_commit"] = source_commit
    value["source_sha256_lf"] = sources
    _write_new(output, value)
    validate_protocol(output, repo=repo)
    return value


def verify_checkpoint_backup(protocol: Mapping[str, Any], checkpoint_root: Path) -> list[dict[str, Any]]:
    root = Path(checkpoint_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint backup root가 실제 폴더가 아니다")
    verified = []
    for row in protocol["checkpoints"]:
        relative = Path(row["isaaclab_relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint 상대경로가 backup root 밖을 가리킨다")
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"checkpoint backup이 단일 regular file이 아니다: {relative}")
        digest = _sha(path)
        if digest != row["sha256"]:
            raise ValueError(f"checkpoint backup SHA가 다르다: {row['logical_id']}")
        verified.append({"logical_id": row["logical_id"], "sha256": digest, "bytes": path.stat().st_size})
    return verified


def _exact_target(value: Any, anchor: str) -> None:
    if value != TARGET:
        raise ValueError(f"{anchor} target이 고정 자원과 다르다")


def validate_admission(value: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "inventory_before_start", "fresh_quote", "start", "ready_observations",
        "environment", "gpu", "owned_processes", "budget", "pending_intent",
    }
    if set(value) != required or value["schema"] != "vision_transferred_reuse_admission_v1":
        raise ValueError("reuse admission shape가 다르다")
    before = value["inventory_before_start"]
    _exact_target(before.get("target"), "start 전 inventory")
    if before.get("status") != "STOPPED":
        raise ValueError("새 Start 전 target이 STOPPED가 아니다")
    before_at = _utc(before.get("observed_at_utc"), "start 전 inventory")

    quote = value["fresh_quote"]
    if (quote.get("provider") != "nebius" or quote.get("instance_type") != TARGET["instance_type"]
            or Decimal(str(quote.get("usd_per_h"))) != EXPECTED_QUOTE):
        raise ValueError("Start 직전 fresh quote가 사전등록값과 다르다")
    quote_at = _utc(quote.get("observed_at_utc"), "fresh quote")

    start = value["start"]
    _exact_target(start.get("target"), "Start receipt")
    started = _utc(start.get("started_at_utc"), "Start receipt")
    if started <= OLD_START_CUTOFF or not (before_at <= quote_at <= started):
        raise ValueError("새 Start 시각 또는 선행 원문 순서가 다르다")
    if (started - quote_at).total_seconds() > 300:
        raise ValueError("fresh quote가 Start보다 300초 넘게 앞선다")
    if not isinstance(start.get("receipt_sha256"), str) or len(start["receipt_sha256"]) != 64:
        raise ValueError("새 Start receipt SHA가 없다")

    ready = value["ready_observations"]
    if not isinstance(ready, list) or len(ready) != 2:
        raise ValueError("READY 원문은 연속 두 건이어야 한다")
    ready_times = []
    for index, row in enumerate(ready):
        _exact_target(row.get("target"), f"READY {index + 1}")
        if (row.get("status") != "RUNNING" or row.get("build_status") != "COMPLETED"
                or row.get("shell_status") != "READY" or row.get("health") != "HEALTHY"):
            raise ValueError("READY 관찰 상태가 실행 가능 상태가 아니다")
        ready_times.append(_utc(row.get("observed_at_utc"), f"READY {index + 1}"))
    if not started <= ready_times[0] < ready_times[1]:
        raise ValueError("READY 두 건이 새 Start 뒤 연속 시각이 아니다")

    environment = value["environment"]
    if (environment.get("status") != "match" or environment.get("reuse_allowed") is not True
            or environment.get("interpreter") != REMOTE_PYTHON
            or environment.get("isaaclab") != REMOTE_LAB
            or environment.get("versions") != EXPECTED_ENVIRONMENT):
        raise ValueError("원격 interpreter 또는 재사용 환경 버전이 다르다")
    gpu = value["gpu"]
    if (gpu.get("idle") is not True or not isinstance(gpu.get("uuid"), str)
            or not gpu["uuid"].strip() or gpu.get("compute_processes") != []):
        raise ValueError("GPU가 식별된 idle 상태가 아니다")
    if value["owned_processes"] != []:
        raise ValueError("기존 소유 프로세스가 남아 있다")
    budget = value["budget"]
    if (Decimal(str(budget.get("prior_upper_estimate_usd"))) != PRIOR_UPPER
            or Decimal(str(budget.get("followup_max_usd"))) != FOLLOWUP_MAX
            or Decimal(str(budget.get("program_cap_usd"))) != PROGRAM_CAP
            or PRIOR_UPPER + FOLLOWUP_MAX > PROGRAM_CAP):
        raise ValueError("후속 program budget 경계가 다르다")
    if value["pending_intent"] is not None:
        raise ValueError("미확정 intent가 있어 자동 반복을 거부한다")
    if _utc(protocol["resources"]["deadline_utc"], "protocol deadline") <= ready_times[-1]:
        raise ValueError("READY 전에 protocol deadline이 지났다")
    return {"started_at_utc": started.isoformat(), "ready_at_utc": ready_times[-1].isoformat()}


def build_plan(protocol_path: Path, admission_path: Path, checkpoint_root: Path, *, repo: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path, repo=repo)
    checkpoints = verify_checkpoint_backup(protocol, checkpoint_root)
    admission = _read(admission_path)
    admitted = validate_admission(admission, protocol)
    def record_step(phase: str, checkpoint: str, *, only_after: str | None = None) -> dict[str, Any]:
        stage = f"{REMOTE_STATE}/policies/{checkpoint}"
        row = {
            "phase": phase,
            "checkpoint": checkpoint,
            "gpu": True,
            "command": [
                REMOTE_PYTHON,
                f"{REMOTE_REPO}/bench/run_vision_transferred_reuse_v1.py",
                "record",
                "--protocol", f"{REMOTE_REPO}/bench/protocols/vision_transferred_pull_record_20261027c_v1.json",
                "--resource-proof", f"{stage}/resource-proof.json",
                "--resource-sha256", "<fresh-resource-proof-sha256>",
                "--isaaclab", REMOTE_LAB,
                "--python", REMOTE_PYTHON,
                "--stage", stage,
                "--checkpoint-logical-id", checkpoint,
            ],
            "fresh_provider_inventory_required": True,
        }
        if only_after is not None:
            row["only_after"] = only_after
        return row

    steps = [
        record_step("record_none", CHECKPOINT_IDS[0]),
        record_step("record_f1", CHECKPOINT_IDS[1], only_after="record_none_completed"),
        {"phase": "retrieve", "requires_sha256_manifest": True, "gpu": False},
        {"phase": "stop", "command": ["brev", "stop", TARGET["resource_name"]], "gpu": False},
        {"phase": "observe_stopped", "command": ["brev", "ls", "--json"], "expect": "STOPPED", "gpu": False},
    ]
    return {
        "schema": "vision_transferred_reuse_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "target": TARGET,
        "remote": {"repo": REMOTE_REPO, "isaaclab": REMOTE_LAB, "python": REMOTE_PYTHON,
                   "state": REMOTE_STATE},
        "admission": admitted,
        "checkpoints": checkpoints,
        "steps": steps,
        "no_exec_after_stopped": True,
        "unresolved_intent_policy": "hold_without_automatic_replay",
        "training_actions": 0,
        "model_api_requests": 0,
    }


def validate_terminal(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "target", "retrieval", "stop", "stopped", "actions"}
    if set(value) != required or value["schema"] != "vision_transferred_reuse_terminal_v1":
        raise ValueError("reuse terminal shape가 다르다")
    _exact_target(value["target"], "terminal")
    retrieval = value["retrieval"]
    if (not isinstance(retrieval.get("manifest_sha256"), str)
            or len(retrieval["manifest_sha256"]) != 64):
        raise ValueError("회수 manifest SHA가 없다")
    retrieved_at = _utc(retrieval.get("completed_at_utc"), "회수")
    stop = value["stop"]
    stopped = value["stopped"]
    _exact_target(stopped.get("target"), "STOPPED inventory")
    stopped_at = _utc(stopped.get("observed_at_utc"), "STOPPED inventory")
    stop_at = _utc(stop.get("requested_at_utc"), "Stop")
    if (stop.get("command") != ["brev", "stop", TARGET["resource_name"]]
            or not isinstance(stop.get("receipt_sha256"), str)
            or len(stop["receipt_sha256"]) != 64
            or stopped.get("status") != "STOPPED"
            or not retrieved_at <= stop_at <= stopped_at):
        raise ValueError("회수→Stop→STOPPED terminal 경계가 다르다")
    actions = value["actions"]
    if not isinstance(actions, list):
        raise TypeError("terminal actions가 list가 아니다")
    for row in actions:
        at = _utc(row.get("at_utc"), "terminal action")
        if at > stopped_at or (at == stopped_at and row.get("kind") == "remote_exec"):
            raise ValueError("STOPPED 뒤 remote 실행이 기록됐다")
    return {"retrieval_manifest_sha256": retrieval["manifest_sha256"],
            "stopped_at_utc": stopped_at.isoformat()}


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "record":
        with record_contract():
            return RUN3.main(values[1:])
    if values and values[0] == "bind-source":
        parser = argparse.ArgumentParser(description="reuse source blob 결합")
        parser.add_argument("bind-source")
        parser.add_argument("--protocol", type=Path, required=True)
        parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
        parser.add_argument("--source-commit", required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(values)
        bind_source(args.protocol, repo=args.repo, source_commit=args.source_commit,
                    output=args.output)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare", choices=["prepare"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(values)
    plan = build_plan(args.protocol, args.admission, args.checkpoint_root, repo=args.repo)
    _write_new(args.output, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
