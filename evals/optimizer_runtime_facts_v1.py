"""검증된 PPO observer 레코드를 모델용 숫자 fact로 봉인한다.

학습, GPU, 모델 호출, 승인, 실행 권한 또는 원문 변경을 수행하지 않는다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import isaac_training_guard_v1 as GUARD
import lineage_recovery_inputs_v1 as RECOVERY

SCHEMA = "optimizer_runtime_facts_v1"
METADATA_SCHEMA = "optimizer_runtime_facts_metadata_v1"
ARTIFACT_NAME = "artifact.json"
METADATA_NAME = "metadata.json"
ROLES = {"case", "reference"}
SOURCE_CONTRACT = "registered_training_runtime_fact_projection"
_TRAINING_CHECK_LOCK = threading.RLock()
FACT_NAMES = (
    "available",
    "updates_observed",
    "expected_optimizer_steps",
    "optimizer_steps_min",
    "optimizer_steps_max",
    "expected_minibatches",
    "minibatches_min",
    "minibatches_max",
)
FACT_IDS = tuple(f"runtime:{role}:{name}" for role in ("case", "reference") for name in FACT_NAMES)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"비유한 JSON 숫자는 허용하지 않는다: {value}")))
    _require(isinstance(value, dict), f"{path.name}은 JSON 객체여야 한다")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자는 허용하지 않는다: {token}")))
        _require(isinstance(value, dict), "observer record는 JSON 객체여야 한다")
        rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, value: object) -> None:
    _require(not path.exists(), f"{path.name}은 새 파일이어야 한다")
    path.write_bytes(_canonical(value))


def _role(role: str) -> str:
    _require(role in ROLES, "role은 case 또는 reference여야 한다")
    return role


def _count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label}은 정수여야 한다")
    _require(value >= 0, f"{label}은 음수일 수 없다")
    return value


def _validate_facts(facts: object, role: str) -> dict[str, bool | int | None]:
    _require(isinstance(facts, dict), "facts는 객체여야 한다")
    typed_facts = cast(dict[str, Any], facts)
    expected_ids = {f"runtime:{role}:{name}" for name in FACT_NAMES}
    _require(set(typed_facts) == expected_ids, "role별 fact ID는 정확히 8개여야 한다")
    available_id = f"runtime:{role}:available"
    available = typed_facts[available_id]
    _require(type(available) is bool, "available fact는 bool이어야 한다")
    for fact_id, value in typed_facts.items():
        if fact_id == available_id:
            continue
        if not available:
            _require(value is None, "unavailable 숫자 fact는 0이 아니라 null이어야 한다")
        else:
            _count(value, fact_id)
    return cast(dict[str, bool | int | None], typed_facts)


def unavailable_facts(role: str) -> dict[str, bool | int | None]:
    """관측이 없을 때 쓰는 8개 고정 fact를 반환한다."""
    role = _role(role)
    return {
        f"runtime:{role}:{name}": False if name == "available" else None
        for name in FACT_NAMES
    }


def _observed_facts(
    manifest: dict[str, Any], evidence: dict[str, Any], role: str,
) -> tuple[dict[str, bool | int | None], Path]:
    records_path = Path(manifest["records_path"]).resolve()
    residual_records = evidence.get("residual_records")
    _require(isinstance(residual_records, dict), "training evidence residual_records가 없다")
    residual_records = cast(dict[str, Any], residual_records)
    _require(Path(str(residual_records.get("path", ""))).resolve() == records_path,
             "evidence와 manifest의 observer records 경로가 다르다")
    _require(records_path.is_file() and residual_records.get("sha256") == _sha(records_path),
             "observer records bytes가 검증된 evidence와 다르다")

    configured_epochs = _count(manifest.get("configured_learning_epochs"), "configured epochs")
    configured_batches = _count(manifest.get("configured_mini_batches"), "configured minibatches")
    expected = configured_epochs * configured_batches
    rows = _rows(records_path)
    _require(len(rows) == manifest.get("runtime_updates_to_observe") == 3,
             "optimizer observer는 update 0/1/2 세 행이어야 한다")
    optimizer_steps: list[int] = []
    minibatches: list[int] = []
    for index, row in enumerate(rows):
        _require(row.get("schema") == "optimizer_observability_record_v2",
                 "optimizer observer record schema가 다르다")
        _require(row.get("update_index") == index, "observer update index는 0/1/2 순서여야 한다")
        _require(row.get("error") is None, "observer record에 오류가 있다")
        row_expected = row.get("expected")
        _require(isinstance(row_expected, dict)
                 and row_expected.get("optimizer_steps") == expected
                 and row_expected.get("minibatches") == expected,
                 "observer expected 수가 검증된 configured epochs×minibatches와 다르다")
        observed = row.get("observed")
        _require(isinstance(observed, dict), "observer observed 객체가 없다")
        observed = cast(dict[str, Any], observed)
        optimizer_steps.append(_count(observed.get("optimizer_steps"), "observed optimizer steps"))
        minibatches.append(_count(observed.get("minibatches"), "observed minibatches"))

    prefix = f"runtime:{role}:"
    facts: dict[str, bool | int | None] = {
        prefix + "available": True,
        prefix + "updates_observed": len(rows),
        prefix + "expected_optimizer_steps": expected,
        prefix + "optimizer_steps_min": min(optimizer_steps),
        prefix + "optimizer_steps_max": max(optimizer_steps),
        prefix + "expected_minibatches": expected,
        prefix + "minibatches_min": min(minibatches),
        prefix + "minibatches_max": max(minibatches),
    }
    return _validate_facts(facts, role), records_path


def _source_files(
    manifest_path: Path, evidence_dir: Path, owned_result_path: Path,
    manifest: dict[str, Any], evidence: dict[str, Any], records_path: Path,
) -> dict[str, Path]:
    return {
        "manifest": manifest_path,
        "owned_result": owned_result_path,
        "training_evidence": evidence_dir / GUARD.EVIDENCE,
        "guard_records": evidence_dir / GUARD.RECORDS,
        "close_marker": evidence_dir / GUARD.CLOSE_READY,
        "observer_records": records_path,
        "readback": Path(manifest["readback_path"]).resolve(),
        "checkpoint": Path(evidence["checkpoint"]["path"]).resolve(),
    }


def _check_registered_training(
    manifest_path: Path, evidence_dir: Path, owned_result_path: Path, lab: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """복구 receipt를 만들지 않고 등록 학습 arm 전체에 공통 실증 검증만 적용한다."""
    allowed_arms = set(RECOVERY.TRAIN.RESIDUAL.ARMS)
    with _TRAINING_CHECK_LOCK:
        previous = RECOVERY.ALLOWED_ARMS
        RECOVERY.ALLOWED_ARMS = allowed_arms
        try:
            return RECOVERY._check_training(manifest_path, evidence_dir, owned_result_path, lab)
        finally:
            RECOVERY.ALLOWED_ARMS = previous


def build(
    manifest_path: Path,
    evidence_dir: Path,
    owned_result_path: Path,
    *,
    role: str,
    lab: Path,
    expected_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """실제 학습 원문을 검증하고 공개 envelope와 비공개 metadata를 만든다."""
    role = _role(role)
    manifest_path = Path(manifest_path).resolve()
    evidence_dir = Path(evidence_dir).resolve()
    owned_result_path = Path(owned_result_path).resolve()
    lab = Path(lab).resolve()
    _require(isinstance(expected_checkpoint_sha256, str) and len(expected_checkpoint_sha256) == 64,
             "expected checkpoint SHA-256은 필수다")
    manifest, evidence, _completion, _residual = _check_registered_training(
        manifest_path, evidence_dir, owned_result_path, lab)
    checkpoint_sha = evidence["checkpoint"]["sha256"]
    _require(checkpoint_sha == expected_checkpoint_sha256,
             "runtime facts checkpoint가 요청한 진단 모델 checkpoint와 다르다")
    facts, records_path = _observed_facts(manifest, evidence, role)
    sources = _source_files(
        manifest_path, evidence_dir, owned_result_path, manifest, evidence, records_path)
    raw_binding = {name: _sha(path) for name, path in sorted(sources.items())}
    binding_core = {
        "source_contract": SOURCE_CONTRACT,
        "accepted_arms": sorted(RECOVERY.TRAIN.RESIDUAL.ARMS),
        "role": role,
        "checkpoint_sha256": checkpoint_sha,
        "raw_source_sha256": raw_binding,
    }
    binding_digest = hashlib.sha256(_canonical(binding_core)).hexdigest()
    envelope = {
        "schema": SCHEMA,
        "model_view": {"facts": facts},
        "source_binding": {
            "source_contract": SOURCE_CONTRACT,
            "accepted_arms": sorted(RECOVERY.TRAIN.RESIDUAL.ARMS),
            "binding_digest": binding_digest,
            "checkpoint_sha256": checkpoint_sha,
        },
        "scope": {
            "activity": "cpu_optimizer_runtime_fact_materialization_only",
            "approval": False,
            "execution_allowed": False,
        },
    }
    metadata = {
        "schema": METADATA_SCHEMA,
        "source_contract": SOURCE_CONTRACT,
        "accepted_arms": sorted(RECOVERY.TRAIN.RESIDUAL.ARMS),
        "role": role,
        "sources": {name: str(path) for name, path in sorted(sources.items())},
        "raw_source_sha256": raw_binding,
        "binding_digest": binding_digest,
        "checkpoint_sha256": checkpoint_sha,
        "lab": str(lab),
    }
    return deepcopy(envelope), deepcopy(metadata)


def prepare(
    manifest_path: Path,
    evidence_dir: Path,
    owned_result_path: Path,
    *,
    role: str,
    lab: Path,
    expected_checkpoint_sha256: str,
    out: Path,
) -> dict[str, Any]:
    """공개 artifact와 별도 private metadata를 fresh 디렉터리에 쓴다."""
    out = Path(out).resolve()
    _require(not out.exists(), "out은 존재하지 않는 fresh 경로여야 한다")
    envelope, metadata = build(
        manifest_path, evidence_dir, owned_result_path, role=role, lab=lab,
        expected_checkpoint_sha256=expected_checkpoint_sha256)
    out.mkdir(parents=True, exist_ok=False)
    metadata_path = out / METADATA_NAME
    _write_new(metadata_path, metadata)
    artifact = {
        "envelope": envelope,
        "private_metadata_ref": {"file": METADATA_NAME, "sha256": _sha(metadata_path)},
    }
    _write_new(out / ARTIFACT_NAME, artifact)
    return deepcopy(artifact)


def checked(
    artifact_path: Path,
    *,
    expected_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, bool | int | None]]:
    """봉인 artifact, metadata, 원문 및 현재 checkpoint를 모두 다시 검증한다."""
    artifact_path = Path(artifact_path).resolve()
    artifact = _read_json(artifact_path)
    _require(set(artifact) == {"envelope", "private_metadata_ref"}, "artifact shape가 다르다")
    metadata_ref = artifact["private_metadata_ref"]
    _require(isinstance(metadata_ref, dict)
             and set(metadata_ref) == {"file", "sha256"}
             and metadata_ref["file"] == METADATA_NAME,
             "private metadata ref shape가 다르다")
    metadata_ref = cast(dict[str, Any], metadata_ref)
    metadata_path = artifact_path.parent / METADATA_NAME
    _require(metadata_path.is_file() and _sha(metadata_path) == metadata_ref["sha256"],
             "private metadata bytes가 artifact ref와 다르다")
    metadata = _read_json(metadata_path)
    _require(set(metadata) == {"schema", "source_contract", "accepted_arms", "role",
                               "sources", "raw_source_sha256",
                               "binding_digest", "checkpoint_sha256", "lab"}
             and metadata.get("schema") == METADATA_SCHEMA
             and metadata.get("source_contract") == SOURCE_CONTRACT
             and metadata.get("accepted_arms") == sorted(RECOVERY.TRAIN.RESIDUAL.ARMS),
             "private metadata shape가 다르다")
    sources = metadata.get("sources")
    _require(isinstance(sources, dict) and set(sources) == {
        "manifest", "owned_result", "training_evidence", "guard_records", "close_marker",
        "observer_records", "readback", "checkpoint"}, "private source map shape가 다르다")
    sources = cast(dict[str, str], sources)
    role = metadata.get("role")
    lab = metadata.get("lab")
    if not isinstance(role, str) or not isinstance(lab, str):
        raise TypeError("metadata role/lab 형식이 다르다")
    expected_envelope, expected_metadata = build(
        Path(sources["manifest"]), Path(sources["training_evidence"]).parent,
        Path(sources["owned_result"]), role=role, lab=Path(lab),
        expected_checkpoint_sha256=expected_checkpoint_sha256)
    _require(metadata == expected_metadata, "private metadata가 현재 검증 원문과 다르다")
    envelope = artifact.get("envelope")
    _require(envelope == expected_envelope, "공개 envelope가 현재 검증 facts와 다르다")
    envelope = cast(dict[str, Any], envelope)
    _require(envelope.get("scope") == {
        "activity": "cpu_optimizer_runtime_fact_materialization_only",
        "approval": False, "execution_allowed": False,
    }, "artifact가 승인 또는 실행 권한을 만들었다")
    facts = envelope.get("model_view", {}).get("facts")
    return deepcopy(envelope), deepcopy(_validate_facts(facts, role))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    for name in ("manifest", "evidence_dir", "owned_result", "lab", "out"):
        build_parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    build_parser.add_argument("--role", choices=sorted(ROLES), required=True)
    build_parser.add_argument("--expected-checkpoint-sha256", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--artifact", type=Path, required=True)
    verify_parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = prepare(
            args.manifest, args.evidence_dir, args.owned_result, role=args.role, lab=args.lab,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256, out=args.out)
    else:
        envelope, facts = checked(
            args.artifact, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        result = {"envelope": envelope, "facts": facts}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
