# pyright: reportMissingImports=false
"""20-job source protocol을 소비하는 lineage source training child v2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lineage_source_training_manifest_v2"
PROTOCOL_SCHEMA = "lineage_source_collection_protocol_v2"
ARMS = ("NONE", "F1", "F2", "F3", "F5", "COMPOUND")
JOB_FIELDS = ("job_id", "case_id", "arm", "seed", "run_name", "task", "num_envs", "max_iterations")
REQUIRED = {
    "schema", "purpose", *JOB_FIELDS, "source_commit", "protocol_relative",
    "protocol_sha256_lf", "source_sha256_lf", "runtime_source_sha256_lf",
}
SOURCE_FILES = (
    "bench/lineage_source_training_v2.py",
    "tests/test_lineage_source_training_v2.py",
    "bench/lineage_source_training_v1.py",
    "tests/test_lineage_source_training_v1.py",
    "bench/hidden_faults.py",
    "bench/external_faults.py",
    "bench/isaac_training_guard_v1.py",
)
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _relative(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and not Path(value).is_absolute() and ".." not in Path(value).parts)


def _sha_lf(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _hash_map(value: Any, name: str) -> dict[str, str]:
    if (not isinstance(value, dict) or not value
            or any(not _relative(path) or not _hex(digest, 64) for path, digest in value.items())):
        raise ValueError(f"{name} 형식이 잘못됐다")
    return dict(value)


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("manifest는 JSON 객체여야 한다")
    data = dict(value)
    if set(data) != REQUIRED:
        raise ValueError("manifest 필드가 v2 고정 계약과 다르다")
    if data["schema"] != SCHEMA or data["purpose"] not in {"smoke", "source"} or data["arm"] not in ARMS:
        raise ValueError("manifest schema, purpose 또는 arm이 잘못됐다")
    if any(not isinstance(data[key], str) or _ID.fullmatch(data[key]) is None
           for key in ("job_id", "case_id")):
        raise ValueError("job_id와 case_id 형식이 잘못됐다")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")
    if not isinstance(data["run_name"], str) or _RUN_NAME.fullmatch(data["run_name"]) is None:
        raise ValueError("run_name 형식 또는 길이가 잘못됐다")
    expected_iterations = 1 if data["purpose"] == "smoke" else 300
    if (data["task"] != TASK or data["num_envs"] != 4096
            or data["max_iterations"] != expected_iterations
            or (data["purpose"] == "smoke" and data["arm"] != "NONE")):
        raise ValueError("task·환경 수·update 예산이 v2 source 계약과 다르다")
    if not _hex(data["source_commit"], 40) or not _hex(data["protocol_sha256_lf"], 64):
        raise ValueError("source commit 또는 protocol SHA 형식이 다르다")
    if not _relative(data["protocol_relative"]):
        raise ValueError("protocol_relative는 저장소 상대경로여야 한다")
    source = _hash_map(data["source_sha256_lf"], "source_sha256_lf")
    runtime = _hash_map(data["runtime_source_sha256_lf"], "runtime_source_sha256_lf")
    if not set(SOURCE_FILES) <= set(source):
        raise ValueError("v2 child source closure가 누락됐다")
    data["source_sha256_lf"], data["runtime_source_sha256_lf"] = source, runtime
    return data


def manifest_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(validate_manifest(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True)


def _protocol_job(protocol: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise TypeError("v2 source protocol 객체가 필요하다")
    if data["purpose"] == "smoke":
        row = protocol.get("smoke_job")
        matches = [row] if isinstance(row, dict) and row.get("job_id") == data["job_id"] else []
    else:
        rows = protocol.get("training_jobs")
        if not isinstance(rows, list) or len(rows) != 20:
            raise ValueError("v2 protocol.training_jobs는 20개 배열이어야 한다")
        matches = [row for row in rows if isinstance(row, dict) and row.get("job_id") == data["job_id"]]
    if len(matches) != 1:
        raise ValueError("v2 protocol에서 유일한 job_id를 찾지 못했다")
    expected = {field: data[field] for field in JOB_FIELDS}
    actual = {field: matches[0].get(field) for field in JOB_FIELDS}
    if actual != expected:
        raise ValueError("v2 protocol job과 manifest 인자가 다르다")
    return dict(matches[0])


def verify_binding(value: Mapping[str, Any], *, repo: Path, lab: Path) -> None:
    data = validate_manifest(value)
    repo, lab = Path(repo).resolve(), Path(lab).resolve()
    commit = data["source_commit"]
    _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode:
        raise ValueError("source commit이 현재 HEAD의 조상이 아니다")
    commitments = {data["protocol_relative"]: data["protocol_sha256_lf"], **data["source_sha256_lf"]}
    for relative, expected in commitments.items():
        path = repo / relative
        if not path.is_file() or _sha_lf(path) != expected:
            raise ValueError(f"현재 source binding이 다르다: {relative}")
        blob = _git(repo, "show", f"{commit}:{relative}").stdout
        normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"anchor source blob이 다르다: {relative}")
    for relative, expected in data["runtime_source_sha256_lf"].items():
        path = lab / relative
        if not path.is_file() or _sha_lf(path) != expected:
            raise ValueError(f"runtime source binding이 다르다: {relative}")
    protocol = json.loads((repo / data["protocol_relative"]).read_text(encoding="utf-8"))
    _protocol_job(protocol, data)


def _isolated_v1() -> ModuleType:
    """공유 import 전역을 바꾸지 않는 private v1 실행 본문을 만든다."""
    path = ROOT / "bench/lineage_source_training_v1.py"
    spec = importlib.util.spec_from_file_location("_lineage_source_training_v1_for_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__dict__["SCHEMA"] = SCHEMA
    module.__dict__["validate_manifest"] = validate_manifest
    module.__dict__["manifest_fingerprint"] = manifest_fingerprint
    module.__dict__["verify_binding"] = verify_binding
    return module


def execute(manifest_path: Path, lab: Path, out: Path) -> dict[str, Any]:
    """v2 결합을 확인한 뒤 기존 학습·결함 주입·종료 본문을 격리 실행한다."""
    module = _isolated_v1()
    return module.execute(manifest_path, lab, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        value = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        if args.validate_only:
            print(json.dumps({"status": "valid", "schema": value["schema"]}))
            return 0
        execute(args.manifest, args.lab, args.out)
        return 0
    except BaseException as error:  # noqa: BLE001
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

