"""영상 선택과 기존 승인 전 host request 사이의 불변 bridge."""
from __future__ import annotations

import json
import math
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import selection_comparison_controller_v1 as CONTROLLER
import vision_evidence_contract_v1 as VISION
import vision_selection_contract_v1 as CONTRACT

SCHEMA = "vision_selection_bridge_receipt_v1"
REFERENCE_FIELDS = frozenset({"path", "sha256", "status"})


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("bridge receipt의 JSON key가 중복됐다")
        value[key] = item
    return value


def _validate_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("bridge receipt에 비유한 수치가 있다")
        return
    if isinstance(value, list):
        for child in value:
            _validate_json(child)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("bridge receipt object key는 문자열이어야 한다")
        for child in value.values():
            _validate_json(child)
        return
    raise TypeError("bridge receipt는 JSON 값이어야 한다")


def _canonical_file(path: Path) -> Path:
    target = Path(path)
    if not target.is_absolute() or target != target.resolve(strict=True):
        raise ValueError("bridge receipt 경로는 canonical absolute path여야 한다")
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError("bridge receipt의 symlink 경로를 거부한다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("bridge receipt는 단일 regular file이어야 한다")
    return target


def _target(path: Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("bridge receipt 출력은 absolute path여야 한다")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    existing = next((parent for parent in target.parents if parent.exists()), None)
    if existing is None or existing.is_symlink() or any(parent.is_symlink() for parent in existing.parents):
        raise ValueError("bridge receipt 출력 parent의 symlink를 거부한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent != target.parent.resolve(strict=True):
        raise ValueError("bridge receipt 출력 parent는 canonical path여야 한다")
    return target


def _read(path: Path) -> dict[str, Any]:
    target = _canonical_file(path)
    try:
        value = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"bridge receipt JSON constant 거부: {raw}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("bridge receipt를 읽을 수 없다") from error
    if not isinstance(value, dict):
        raise TypeError("bridge receipt 객체가 아니다")
    _validate_json(value)
    return value


def _write_new(path: Path, value: dict) -> None:
    _validate_json(value)
    target = _target(path)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    _canonical_file(target)


def _reference(value: object) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != REFERENCE_FIELDS:
        raise TypeError("seal_bridge가 반환한 path·sha256·status reference가 필요하다")
    path, sha256, status = value["path"], value["sha256"], value["status"]
    if not isinstance(path, str) or not isinstance(sha256, str) or not isinstance(status, str):
        raise TypeError("bridge reference 값 형식이 다르다")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValueError("bridge reference SHA 형식이 다르다")
    if status not in {"choice_ready", "HOLD"}:
        raise ValueError("bridge reference status가 다르다")
    target = _canonical_file(Path(path))
    if CONTROLLER.digest_file(target) != sha256:
        raise ValueError("bridge reference SHA가 현재 파일과 다르다")
    return target, {"path": str(target), "sha256": sha256, "status": status}


def _hold(bundle: dict, code: str) -> dict:
    base = bundle["operational_base"]
    return {
        "schema": "selection_controller_choice_v1",
        "context_digest": base["context_digest"],
        "view_digest": base["base_view_digest"],
        "policy": base["policy"],
        "selected_by": "hold",
        "action": "hold",
        "probe": None,
        "fact_ids": [],
        "model_status": "invalid",
        "codes": [code],
        "reason": "model_invalid",
        "used_for_decision": True,
        "approval_granted": False,
    }


def _record(snapshot: dict, contract_path: Path, source_binding: dict,
            view_kind: str, model_content: object) -> dict:
    bundle = CONTRACT.prepare(snapshot, contract_path, source_binding, view_kind)
    sent = CONTRACT.messages(bundle, snapshot, contract_path, source_binding)
    safe_content = model_content if isinstance(model_content, str) else None
    try:
        decision = CONTRACT.validate_choice(bundle, safe_content)
        choice = CONTRACT.project_choice(bundle, decision, snapshot, contract_path, source_binding)
        status = "choice_ready" if decision["action"] == "continue" else "HOLD"
        reason, code = ("model_valid" if status == "choice_ready" else "model_hold"), None
    except CONTRACT.ChoiceError as error:
        decision = None
        code = str(error)
        choice = _hold(bundle, code)
        status, reason = "HOLD", "model_invalid"
    return {
        "schema": SCHEMA,
        "status": status,
        "reason": reason,
        "validation_code": code,
        "view_kind": view_kind,
        "bundle_digest": CONTROLLER.L.digest(bundle),
        "operational_base_digest": bundle["operational_base_digest"],
        "base_snapshot_digest": bundle["operational_base"]["snapshot_digest"],
        "base_view_digest": bundle["operational_base"]["base_view_digest"],
        "visual_view_digest": bundle["selection_view_digest"],
        "messages_digest": CONTROLLER.L.digest(sent),
        "v01_contract_sha256": VISION.sha256(Path(contract_path)),
        "model_content": safe_content,
        "decision": decision,
        "projected_choice": choice,
        "approval_granted": False,
        "rule_fallback_used": False,
    }


def seal_bridge(path: Path, snapshot: dict, contract_path: Path, source_binding: dict,
                view_kind: str, model_content: object) -> dict:
    """모델 결과와 V01/base 결합을 승인 전 write-once receipt로 남긴다."""
    value = _record(snapshot, Path(contract_path), source_binding, view_kind, model_content)
    target = Path(path)
    _write_new(target, value)
    return {"path": str(target.resolve(strict=True)), "sha256": CONTROLLER.digest_file(target),
            "status": value["status"]}


def verify_bridge(reference: dict, snapshot: dict, contract_path: Path, source_binding: dict) -> dict:
    """현재 원장 상태와 V01 바이트에서 bridge를 다시 계산해 stale·변조를 거부한다."""
    path, checked_reference = _reference(reference)
    saved = _read(path)
    if saved.get("schema") != SCHEMA or saved.get("view_kind") not in CONTRACT.VIEW_KINDS:
        raise ValueError("bridge receipt 형식이 다르다")
    if saved.get("status") != checked_reference["status"]:
        raise ValueError("bridge reference status와 receipt가 다르다")
    expected = _record(
        snapshot, Path(contract_path), source_binding, saved["view_kind"], saved.get("model_content"),
    )
    if CONTRACT._canonical(saved) != CONTRACT._canonical(expected):
        raise ValueError("bridge receipt가 현재 V01 또는 selection 상태와 다르다")
    return deepcopy(expected)


def host_request_spec(reference: dict, snapshot: dict, contract_path: Path, source_binding: dict) -> dict:
    """유효 bridge만 기존 host canonical request_spec으로 넘긴다."""
    bridge = verify_bridge(reference, snapshot, contract_path, source_binding)
    if bridge["status"] != "choice_ready":
        return {"status": "HOLD", "reason": bridge["reason"], "spec": None,
                "approval_granted": False, "rule_fallback_used": False}
    spec = CONTROLLER.request_spec(snapshot, bridge["projected_choice"])
    return {"status": "request_ready", "reason": "validated_model_choice", "spec": spec,
            "approval_granted": False, "rule_fallback_used": False}
