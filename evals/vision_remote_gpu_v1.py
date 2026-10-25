"""Brev 자원 원문과 원격 ``nvidia-smi`` 출력을 fail-closed로 검증한다."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PROVIDER_ROW_KEYS = {
    "build_status", "gpu", "health_status", "id", "instance_kind",
    "instance_type", "name", "shell_status", "status",
}
GPU_COMMAND = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]
PROCESS_COMMAND = [
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
]
CAPTURE_KEYS = {
    "resource_id", "gpu_uuid", "provider_inventory_path", "provider_inventory_sha256",
    "nvidia_gpu_csv_path", "nvidia_gpu_csv_sha256", "nvidia_process_csv_path",
    "nvidia_process_csv_sha256",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path: Path, expected_sha: str, anchor: str) -> Path:
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_nlink != 1 or not isinstance(expected_sha, str)
            or HEX64.fullmatch(expected_sha) is None or sha256(path) != expected_sha):
        raise ValueError(f"{anchor} 원본 파일·SHA가 다르다")
    return path.resolve()


def _text(path: Path, expected_sha: str, anchor: str) -> str:
    return _regular(path, expected_sha, anchor).read_text(encoding="utf-8")


def verify_provider_inventory(path: Path, expected_sha: str, resource_id: str) -> dict[str, Any]:
    try:
        body = json.loads(_text(path, expected_sha, "Brev provider inventory"))
    except json.JSONDecodeError as error:
        raise ValueError("Brev provider inventory JSON이 잘못됐다") from error
    if not isinstance(body, dict) or set(body) != {"workspaces"} or not isinstance(body["workspaces"], list):
        raise ValueError("Brev provider inventory schema가 다르다")
    rows = body["workspaces"]
    if any(not isinstance(row, dict) or set(row) != PROVIDER_ROW_KEYS for row in rows):
        raise ValueError("Brev provider workspace 행 schema가 다르다")
    matches = [row for row in rows if row["id"] == resource_id]
    if len(matches) != 1 or matches[0]["status"] != "RUNNING":
        raise ValueError("등록한 Brev 자원이 유일한 RUNNING 상태가 아니다")
    return dict(matches[0])


def parse_gpu_uuid_csv(text: str, expected_uuid: str) -> str:
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0].strip() != lines[0] or lines[0] != expected_uuid:
        raise ValueError("nvidia-smi GPU UUID CSV가 단일 등록 GPU와 다르다")
    return lines[0]


def parse_process_csv(text: str, expected_uuid: str) -> list[int]:
    if text == "":
        return []
    text = text.removesuffix("\n")
    if not text:
        return []
    pids: list[int] = []
    for line in text.split("\n"):
        parts = line.split(", ")
        if len(parts) != 2 or parts[0] != expected_uuid or not parts[1].isdigit():
            raise ValueError("nvidia-smi compute process CSV가 잘못됐다")
        pid = int(parts[1])
        if pid <= 0 or pid in pids:
            raise ValueError("nvidia-smi compute PID가 잘못됐다")
        pids.append(pid)
    return sorted(pids)


def verify_capture_proof(proof: Mapping[str, Any]) -> None:
    if set(proof) != CAPTURE_KEYS:
        raise ValueError("원격 GPU 원문 proof 필드가 다르다")
    resource_id, gpu_uuid = proof["resource_id"], proof["gpu_uuid"]
    if not isinstance(resource_id, str) or not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("원격 GPU resource 식별자가 잘못됐다")
    verify_provider_inventory(
        Path(proof["provider_inventory_path"]), proof["provider_inventory_sha256"], resource_id
    )
    parse_gpu_uuid_csv(
        _text(Path(proof["nvidia_gpu_csv_path"]), proof["nvidia_gpu_csv_sha256"], "GPU UUID CSV"),
        gpu_uuid,
    )
    if parse_process_csv(
        _text(Path(proof["nvidia_process_csv_path"]), proof["nvidia_process_csv_sha256"],
              "GPU process CSV"),
        gpu_uuid,
    ):
        raise ValueError("원격 GPU 입장 원문에 기존 compute process가 있다")


def verify_observation_reference(
    reference: Mapping[str, Any], expected_resource_id: str, expected_gpu_uuid: str
) -> dict[str, Any]:
    if set(reference) != {"path", "sha256"}:
        raise ValueError("GPU 관측 참조 필드가 다르다")
    receipt_path = _regular(Path(reference["path"]), reference["sha256"], "GPU 관측 receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("GPU 관측 receipt JSON이 잘못됐다") from error
    required = {
        "schema", "ordinal", "resource_id", "gpu_uuid", "gpu_command", "gpu_exit_code",
        "gpu_stdout_file", "gpu_stdout_sha256", "gpu_stderr", "process_command",
        "process_exit_code", "process_stdout_file", "process_stdout_sha256", "process_stderr",
    }
    if (not isinstance(receipt, dict) or set(receipt) != required
            or receipt["schema"] != "vision_remote_gpu_observation_v1"
            or type(receipt["ordinal"]) is not int or receipt["ordinal"] <= 0
            or receipt["resource_id"] != expected_resource_id
            or receipt["gpu_uuid"] != expected_gpu_uuid
            or receipt["gpu_command"] != GPU_COMMAND or receipt["process_command"] != PROCESS_COMMAND
            or receipt["gpu_exit_code"] != 0 or receipt["process_exit_code"] != 0
            or receipt["gpu_stderr"] != "" or receipt["process_stderr"] != ""):
        raise ValueError("GPU 관측 receipt 계약이 다르다")
    prefix = f"{receipt['ordinal']:06d}"
    if (receipt["gpu_stdout_file"] != f"{prefix}.gpu.csv"
            or receipt["process_stdout_file"] != f"{prefix}.process.csv"):
        raise ValueError("GPU 관측 원문 파일명이 ordinal과 다르다")
    gpu_path = receipt_path.parent / receipt["gpu_stdout_file"]
    process_path = receipt_path.parent / receipt["process_stdout_file"]
    gpu_text = _text(gpu_path, receipt["gpu_stdout_sha256"], "GPU 관측 UUID CSV")
    process_text = _text(process_path, receipt["process_stdout_sha256"], "GPU 관측 process CSV")
    parse_gpu_uuid_csv(gpu_text, expected_gpu_uuid)
    if parse_process_csv(process_text, expected_gpu_uuid):
        raise ValueError("GPU 관측 시점에 compute process가 남아 있다")
    return receipt


class StrictRemoteGpuReader:
    """각 조회의 원문을 append-only로 남기고 malformed 출력을 거부한다."""

    def __init__(self, proof: Mapping[str, Any], audit_root: Path, *,
                 run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run):
        verify_capture_proof(proof)
        self.proof = dict(proof)
        self.audit_root = Path(audit_root).resolve()
        self.run = run
        self.last_receipt: dict[str, str] | None = None

    def _next(self) -> tuple[int, Path]:
        self.audit_root.mkdir(parents=True, exist_ok=True)
        if self.audit_root.is_symlink():
            raise ValueError("GPU 관측 폴더는 symlink일 수 없다")
        ordinals = []
        for path in self.audit_root.glob("*.receipt.json"):
            if path.is_symlink() or not re.fullmatch(r"[0-9]{6}\.receipt\.json", path.name):
                raise ValueError("GPU 관측 원장 파일명이 잘못됐다")
            ordinals.append(int(path.name[:6]))
        ordinal = max(ordinals, default=0) + 1
        return ordinal, self.audit_root / f"{ordinal:06d}"

    @staticmethod
    def _call(run: Callable[..., subprocess.CompletedProcess[str]], command: list[str]
              ) -> subprocess.CompletedProcess[str]:
        try:
            result = run(command, capture_output=True, text=True, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("nvidia-smi query 실행에 실패했다") from error
        if not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
            raise TypeError("nvidia-smi query 출력 형식이 잘못됐다")
        return result

    def __call__(self) -> list[int]:
        verify_capture_proof(self.proof)
        ordinal, prefix = self._next()
        gpu = self._call(self.run, GPU_COMMAND)
        processes = self._call(self.run, PROCESS_COMMAND)
        gpu_path = prefix.with_suffix(".gpu.csv")
        process_path = prefix.with_suffix(".process.csv")
        for path, value in ((gpu_path, gpu.stdout), (process_path, processes.stdout)):
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        receipt = {
            "schema": "vision_remote_gpu_observation_v1", "ordinal": ordinal,
            "resource_id": self.proof["resource_id"], "gpu_uuid": self.proof["gpu_uuid"],
            "gpu_command": GPU_COMMAND, "gpu_exit_code": gpu.returncode,
            "gpu_stdout_file": gpu_path.name, "gpu_stdout_sha256": sha256(gpu_path),
            "gpu_stderr": gpu.stderr, "process_command": PROCESS_COMMAND,
            "process_exit_code": processes.returncode, "process_stdout_file": process_path.name,
            "process_stdout_sha256": sha256(process_path), "process_stderr": processes.stderr,
        }
        receipt_path = prefix.with_suffix(".receipt.json")
        with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.last_receipt = {"path": str(receipt_path), "sha256": sha256(receipt_path)}
        if gpu.returncode != 0 or processes.returncode != 0 or gpu.stderr or processes.stderr:
            raise ValueError("nvidia-smi query가 깨끗하게 끝나지 않았다")
        parse_gpu_uuid_csv(gpu.stdout, self.proof["gpu_uuid"])
        return parse_process_csv(processes.stdout, self.proof["gpu_uuid"])
