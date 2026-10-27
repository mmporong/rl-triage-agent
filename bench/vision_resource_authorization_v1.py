"""과거 학습 source와 새 촬영 자원 승인 기한을 분리한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_vision_brev_batch_v1 as SOURCE

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vision_resource_authorization_v1"
OPERATION = "existing_checkpoint_recording_only"
MINIMUM_REMAINING_S = SOURCE.ACTION_AUTHORIZATION_S
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def _exact(value: Any, keys: set[str], anchor: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{anchor} 필드가 다르다")
    return value


def _relative(value: Any, anchor: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{anchor}가 문자열이 아니다")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{anchor}는 repository 상대 경로여야 한다")
    return path.as_posix()


def _reference(value: Any, anchor: str) -> dict[str, str]:
    row = _exact(value, {"path", "sha256_lf"}, anchor)
    path = _relative(row["path"], f"{anchor}.path")
    digest = row["sha256_lf"]
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        raise ValueError(f"{anchor}.sha256_lf 형식이 다르다")
    return {"path": path, "sha256_lf": digest}


def _regular(repo: Path, reference: Mapping[str, str], anchor: str) -> Path:
    path = repo / reference["path"]
    if not path.is_file() or path.is_symlink() or _sha(path, lf=True) != reference["sha256_lf"]:
        raise ValueError(f"{anchor} 원본 또는 LF SHA가 다르다")
    return path


def _checkpoint_catalog(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("checkpoint catalog가 비었다")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in value:
        row = _exact(raw, {"logical_id", "isaaclab_relative", "sha256"}, "checkpoint")
        logical_id = row["logical_id"]
        relative = _relative(row["isaaclab_relative"], "checkpoint path")
        digest = row["sha256"]
        if (not isinstance(logical_id, str) or SLUG.fullmatch(logical_id) is None
                or logical_id in seen or not isinstance(digest, str)
                or HEX64.fullmatch(digest) is None):
            raise ValueError("checkpoint catalog 형식 또는 중복이 잘못됐다")
        seen.add(logical_id)
        result.append({"logical_id": logical_id, "isaaclab_relative": relative, "sha256": digest})
    return result


def validate(path: Path, *, repo: Path = ROOT, now: datetime | None = None,
             require_current: bool = True) -> dict[str, Any]:
    repo = Path(repo).resolve()
    raw = _exact(_read(path), {
        "schema", "authorization_id", "registered_before_execution", "registered_at_utc",
        "source_protocol", "execution_protocol", "historical_source_deadline_utc",
        "authorization_deadline_utc", "minimum_remaining_s", "resource_id",
        "resource_name", "scope", "checkpoint_catalog",
    }, "resource authorization")
    if (raw["schema"] != SCHEMA or raw["registered_before_execution"] is not True
            or not isinstance(raw["authorization_id"], str)
            or SLUG.fullmatch(raw["authorization_id"]) is None
            or raw["minimum_remaining_s"] != MINIMUM_REMAINING_S):
        raise ValueError("resource authorization 기본 계약이 다르다")
    source_ref = _reference(raw["source_protocol"], "source protocol")
    execution_ref = _reference(raw["execution_protocol"], "execution protocol")
    source_path = _regular(repo, source_ref, "source protocol")
    execution_path = _regular(repo, execution_ref, "execution protocol")
    source = SOURCE.validate_protocol(_read(source_path))
    execution = _read(execution_path)
    if (execution.get("registered_before_measurement") is not True
            or execution.get("source_protocol") != source_ref):
        raise ValueError("새 실행 protocol이 과거 source 계보에 결합되지 않았다")
    source_commit = execution.get("source_commit")
    if not isinstance(source_commit, str) or HEX40.fullmatch(source_commit) is None:
        raise ValueError("새 실행 source commit 형식이 다르다")
    if (repo / ".git").exists() and subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", source_commit, "HEAD"],
        check=False, capture_output=True, text=True, encoding="utf-8", errors="strict",
    ).returncode:
        raise ValueError("새 실행 source commit이 현재 HEAD의 조상이 아니다")
    source_deadline = SOURCE._stamp(source["resources"]["deadline_utc"])
    if SOURCE._stamp(raw["historical_source_deadline_utc"]) != source_deadline:
        raise ValueError("과거 source deadline 원문이 바뀌었다")
    registered = SOURCE._stamp(raw["registered_at_utc"])
    deadline = SOURCE._stamp(raw["authorization_deadline_utc"])
    if registered >= deadline:
        raise ValueError("새 자원 승인이 등록 시각 뒤 유효 구간을 갖지 않는다")
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    current = current.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10
    if registered > current:
        raise ValueError("새 자원 승인이 미래에 등록된 것으로 되어 있다")
    if require_current and (deadline - current).total_seconds() < MINIMUM_REMAINING_S:
        raise ValueError("새 자원 승인의 첫 action 유효 시간이 부족하다")
    resources = execution.get("resources")
    source_resources = source["resources"]
    if (not isinstance(resources, dict)
            or raw["resource_id"] != source_resources["owned_instance_id"]
            or resources.get("owned_instance_id") != raw["resource_id"]
            or resources.get("owned_instance_name") != raw["resource_name"]
            or resources.get("max_owned_instances") != 1):
        raise ValueError("새 실행과 source의 소유 자원 결합이 다르다")
    scope = _exact(raw["scope"], {
        "operation", "training_actions", "model_api_requests", "max_full_recordings",
        "max_wall_s_per_recording", "max_compute_usd_batch",
    }, "authorization scope")
    limits = execution.get("limits")
    if (scope["operation"] != OPERATION or scope["training_actions"] != 0
            or scope["model_api_requests"] != 0 or not isinstance(limits, dict)
            or limits.get("training_actions") != 0 or limits.get("model_api_requests") != 0
            or scope["max_full_recordings"] != limits.get("max_policies")
            or scope["max_wall_s_per_recording"] != limits.get("max_wall_s_per_policy")
            or str(scope["max_compute_usd_batch"]) != str(limits.get("max_wall_compute_usd_batch"))):
        raise ValueError("새 승인이 촬영 전용 실행 상한과 다르다")
    catalog = _checkpoint_catalog(raw["checkpoint_catalog"])
    if catalog != _checkpoint_catalog(execution.get("checkpoints")):
        raise ValueError("새 승인이 실행 checkpoint 계보와 다르다")
    return {
        **raw,
        "source_protocol": source_ref,
        "execution_protocol": execution_ref,
        "checkpoint_catalog": catalog,
        "_source": source,
        "_execution": execution,
        "_source_path": source_path,
        "_execution_path": execution_path,
    }


def build(source_path: Path, execution_path: Path, *, authorization_id: str,
          registered_at_utc: str, authorization_deadline_utc: str,
          repo: Path = ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    source_path, execution_path = Path(source_path).resolve(), Path(execution_path).resolve()
    source = SOURCE.validate_protocol(_read(source_path))
    execution = _read(execution_path)
    limits = execution.get("limits", {})
    return {
        "schema": SCHEMA,
        "authorization_id": authorization_id,
        "registered_before_execution": True,
        "registered_at_utc": registered_at_utc,
        "source_protocol": {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": _sha(source_path, lf=True)},
        "execution_protocol": {"path": execution_path.relative_to(repo).as_posix(), "sha256_lf": _sha(execution_path, lf=True)},
        "historical_source_deadline_utc": source["resources"]["deadline_utc"],
        "authorization_deadline_utc": authorization_deadline_utc,
        "minimum_remaining_s": MINIMUM_REMAINING_S,
        "resource_id": source["resources"]["owned_instance_id"],
        "resource_name": execution.get("resources", {}).get("owned_instance_name"),
        "scope": {
            "operation": OPERATION,
            "training_actions": 0,
            "model_api_requests": 0,
            "max_full_recordings": limits.get("max_policies"),
            "max_wall_s_per_recording": limits.get("max_wall_s_per_policy"),
            "max_compute_usd_batch": limits.get("max_wall_compute_usd_batch"),
        },
        "checkpoint_catalog": execution.get("checkpoints"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--registered-at-utc", required=True)
    parser.add_argument("--authorization-deadline-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build(args.source, args.execution, authorization_id=args.authorization_id,
                  registered_at_utc=args.registered_at_utc,
                  authorization_deadline_utc=args.authorization_deadline_utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    validate(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
