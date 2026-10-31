"""저장된 PPO gradient 개발 기록을 CPU probe receipt로 재검증한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import optimizer_observability as OBS
import ppo_gradient_invariant_v1 as GRADIENT

PLAN_SCHEMA = "recorded_ppo_gradient_probe_plan_v1"
RECEIPT_SCHEMA = "recorded_ppo_gradient_probe_receipt_v1"
PROBE = "P_ppo_gradient"
EVIDENCE_KIND = "development_fixture"
RAW_FILES = ("manifest.json", "records.jsonl", "training_evidence.json", "close_ready.json")
SCOPE = {
    "reused_development_gradient_records": True,
    "new_training": False,
    "gpu_execution": False,
    "new_observation": False,
    "recovery_claim": False,
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda item: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 값: {item}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        require(bool(line.strip()), f"records {index} 빈 행")
        value = json.loads(line, parse_constant=lambda item: (
            _ for _ in ()).throw(ValueError(f"비유한 JSON 값: {item}")))
        require(isinstance(value, dict), f"records {index}은 객체여야 한다")
        rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    require(not path.exists(), f"{path.name}은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _validate_completion(folder: Path, manifest: Mapping[str, Any]) -> None:
    evidence = _read(folder / "training_evidence.json")
    close = _read(folder / "close_ready.json")
    fingerprint = OBS.manifest_fingerprint(manifest)
    updates = manifest["runtime_updates_to_observe"]
    require(
        evidence.get("schema") == "ppo_gradient_training_evidence_v1"
        and evidence.get("manifest_sha256") == fingerprint
        and evidence.get("arm") == manifest["arm"]
        and evidence.get("updates") == updates
        and evidence.get("observed") is True
        and evidence.get("learning_completed") is True
        and evidence.get("recorded_before_app_close") is True
        and evidence.get("method_cleanup") is True,
        "gradient training evidence 완료 결합 불일치",
    )
    require(
        close.get("schema") == "ppo_gradient_close_ready_v4"
        and close.get("manifest_sha256") == fingerprint
        and close.get("updates") == updates
        and close.get("evidence_sha256_lf") == _lf_sha(folder / "training_evidence.json")
        and close.get("records_sha256_lf") == _lf_sha(folder / "records.jsonl"),
        "gradient close-ready raw SHA 결합 불일치",
    )


def _source(evidence_dir: Path) -> dict[str, Any]:
    folder = Path(evidence_dir).resolve()
    require(folder.is_dir(), "gradient evidence 디렉터리가 없다")
    paths = {name: folder / name for name in RAW_FILES}
    require(all(path.is_file() for path in paths.values()), "gradient raw 파일이 모두 필요하다")
    raw = {name: _sha(path) for name, path in paths.items()}
    manifest = GRADIENT.validate_manifest(_read(paths["manifest.json"]))
    require(manifest.get("observer_enabled") is True,
            "저장 gradient 소비에는 observer-enabled 기록이 필요하다")
    _validate_completion(folder, manifest)
    inspection = GRADIENT.inspect_records(_records(paths["records.jsonl"]), manifest)
    count = {
        name: deepcopy(inspection["primitives"][name])
        for name in ("optimizer_step_count", "minibatch_count")
    }
    gradient = deepcopy(inspection["primitives"][GRADIENT.PRIMITIVE])
    require(gradient.get("status") in {"normal", "abnormal", "unknown"},
            "gradient primitive status가 지원 범위 밖이다")
    return {
        "evidence_kind": EVIDENCE_KIND,
        "source_binding": {
            "manifest_fingerprint": OBS.manifest_fingerprint(manifest),
            "arm": manifest["arm"],
            "seed": manifest["seed"],
            "run_id": manifest.get("run_id"),
            "source_commit_sha": manifest["source_commit_sha"],
            "protocol_path": manifest["protocol_path"],
            "protocol_sha256_lf": manifest["protocol_sha256_lf"],
            "source_closure_digest": _digest(manifest["source_sha256_lf"]),
            "raw_source_sha256": raw,
        },
        "inspection": {
            "records": inspection["records"],
            "count_primitives": count,
            "gradient_primitive": gradient,
            "count_status": (
                "abnormal" if any(row["status"] == "abnormal" for row in count.values())
                else "normal" if all(row["status"] == "normal" for row in count.values())
                else "unknown"
            ),
            "gradient_status": gradient["status"],
        },
        "private_source": {
            "evidence_dir": str(folder),
            "raw_files": {name: str(path) for name, path in paths.items()},
        },
    }


def prepare(evidence_dir: Path, *, request_id: str, out: Path) -> dict[str, Any]:
    """기존 개발 기록을 새 관측으로 세지 않는 CPU 재사용 계획을 만든다."""
    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request_id) is not None,
            "request ID 형식")
    source = _source(evidence_dir)
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "prepared",
        "probe": PROBE,
        "request_id": request_id,
        "evidence_kind": EVIDENCE_KIND,
        "source": source,
        "scope": deepcopy(SCOPE),
    }
    _write_new(Path(out), plan)
    return deepcopy(plan)


def _checked_plan(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    plan = _read(path)
    require(
        plan.get("schema") == PLAN_SCHEMA
        and plan.get("status") == "prepared"
        and plan.get("probe") == PROBE
        and plan.get("evidence_kind") == EVIDENCE_KIND
        and plan.get("scope") == SCOPE,
        "recorded gradient plan 계약 불일치",
    )
    source = plan.get("source")
    if not isinstance(source, dict):
        raise TypeError("recorded gradient plan source가 없다")
    private_source = source.get("private_source")
    if (not isinstance(private_source, dict)
            or not isinstance(private_source.get("evidence_dir"), str)):
        raise TypeError("recorded gradient private source가 없다")
    require(source == _source(Path(private_source["evidence_dir"])),
            "recorded gradient plan이 현재 원문과 다르다")
    return plan


def _receipt(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    inspection = plan["source"]["inspection"]
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "probe": PROBE,
        "request_id": plan["request_id"],
        "evidence_kind": EVIDENCE_KIND,
        "outcome": inspection["gradient_status"],
        "count_status": inspection["count_status"],
        "count_primitives": deepcopy(inspection["count_primitives"]),
        "gradient_primitive": deepcopy(inspection["gradient_primitive"]),
        "records": inspection["records"],
        "source_binding": deepcopy(plan["source"]["source_binding"]),
        "plan": {"path": str(plan_path), "sha256": _sha(plan_path)},
        "scope": deepcopy(SCOPE),
    }


def execute(plan_path: Path, *, out: Path) -> dict[str, Any]:
    """현재 raw 기록을 다시 계산해 gradient receipt를 write-once로 만든다."""
    plan_path = Path(plan_path).resolve()
    plan = _checked_plan(plan_path)
    receipt = _receipt(plan_path, plan)
    _write_new(Path(out), receipt)
    return deepcopy(receipt)


def checked_receipt(plan_path: Path, receipt_path: Path) -> dict[str, Any]:
    """plan, 원문, receipt SHA와 count/gradient 분리 판정을 다시 확인한다."""
    plan_path, receipt_path = Path(plan_path).resolve(), Path(receipt_path).resolve()
    plan = _checked_plan(plan_path)
    receipt = _read(receipt_path)
    require(receipt == _receipt(plan_path, plan),
            "recorded gradient receipt가 현재 원문·plan과 다르다")
    return deepcopy(receipt)


__all__ = ["checked_receipt", "execute", "prepare"]
