"""전향적 진단 source와 세 정책 비교 원문을 경로 없는 공개 결과로 내보낸다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]

import evidence_guided_selector_v1 as SEL
import run_prospective_agentic_comparison_v1 as RUN
import runtime_observation_view_v1 as RV

SUMMARY_SCHEMA = "prospective_agentic_public_summary_v1"
CASE_SCHEMA = "prospective_agentic_public_case_v1"
SOURCE_SCHEMA = "prospective_agentic_public_source_v1"
MANIFEST_SCHEMA = "prospective_agentic_public_manifest_v1"
SANITIZED_SCHEMA = "prospective_agentic_sanitized_payload_v1"
POLICIES = RUN.COMP.POLICIES
FORBIDDEN = re.compile(
    r"(?:[A-Za-z]:[\\/]|/(?:home|mnt|Users)/|bench[\\/]private|AppData|claude_reward_track)",
    re.IGNORECASE,
)
SECRET_KEYS = {"api_key", "apikey", "authorization", "access_token", "secret", "token"}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write(path: Path, value: object) -> None:
    path = Path(path).resolve()
    _require(not path.exists(), f"{path.name}은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _safe_public(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _require(FORBIDDEN.search(encoded) is None, "공개 JSON에 개인·비공개·host 경로가 남았다")


def _sanitize(value: object, key: str | None = None) -> tuple[object, bool]:
    if key is not None and key.lower() in SECRET_KEYS:
        return "<redacted-secret>", True
    if isinstance(value, dict):
        result: dict[str, object] = {}
        changed = False
        for name, item in value.items():
            clean, item_changed = _sanitize(item, str(name))
            result[str(name)] = clean
            changed |= item_changed
        return result, changed
    if isinstance(value, list):
        rows, changed = [], False
        for item in value:
            clean, item_changed = _sanitize(item)
            rows.append(clean)
            changed |= item_changed
        return rows, changed
    if isinstance(value, str) and FORBIDDEN.search(value):
        return "<redacted-path>", True
    return deepcopy(value), False


def _logical(kind: str, name: str, raw_sha256: str) -> dict[str, str]:
    return {"logical_source": f"{kind}/{name}", "raw_sha256": raw_sha256}


def _copy_payload(source: Path, target: Path, logical: str) -> dict[str, Any]:
    """안전하면 JSON bytes를 복사하고, 아니면 원본 SHA를 보존한 별도 wrapper를 쓴다."""
    source = Path(source).resolve()
    raw = source.read_bytes()
    value = _read(source)
    clean, changed = _sanitize(value)
    if not changed:
        _safe_public(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        mode = "raw_byte_copy"
    else:
        wrapper = {"schema": SANITIZED_SCHEMA, "original_sha256": hashlib.sha256(raw).hexdigest(),
                   "public_payload": clean, "redactions": ["path_or_secret"]}
        _safe_public(wrapper)
        _write(target, wrapper)
        mode = "sanitized_derived_json"
    return {"path": f"dialogues/{logical}", "sha256": _sha(target), "mode": mode,
            "source": _logical("comparison", logical, hashlib.sha256(raw).hexdigest())}


def _finite(value: object, name: str) -> float:
    _require(type(value) in (int, float) and isinstance(value, (int, float))
             and math.isfinite(value) and value >= 0, f"{name}은 확인된 비음수 수치여야 한다")
    return float(cast(int | float, value))


def _row_order() -> list[tuple[str, str]]:
    return [(f"c{case:02d}", policy) for case in range(1, 9)
            for policy in RUN.COMP.policy_order(case - 1)]


def _comparison_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("rows")
    _require(value.get("schema") == RUN.RESULT_SCHEMA and value.get("status") in {"complete", "invalid"}
             and isinstance(rows, list) and len(rows) <= 24, "comparison result/failure schema가 다르다")
    typed = cast(list[dict[str, Any]], rows)
    expected = _row_order()
    _require([(row.get("case_id"), row.get("policy")) for row in typed] == expected[:len(typed)],
             "comparison row 순서·중복·분모가 등록 계획과 다르다")
    if value["status"] == "complete":
        _require(len(typed) == 24 and value.get("planned_cases") == 8
                 and value.get("planned_policy_rows") == 24, "complete 비교의 24행 분모가 없다")
    return typed


def _verify_epoch(root: Path, row: dict[str, Any], category: str) -> tuple[dict[str, Any] | None,
                                                                          list[Path]]:
    epoch_root = root / row["case_id"] / row["policy"]
    if not row.get("gate", {}).get("run"):
        _require(not epoch_root.exists() and row.get("model_requests") == 0
                 and row.get("paired_probes") == 0 and row.get("decision_outcome") == "HOLD",
                 "gate 중단 행에 모델·검사 실행 흔적이 있다")
        return None, []
    result_path = epoch_root / "result.json"
    _require(result_path.is_file(), "실행된 comparison epoch result가 없다")
    epoch = _read(result_path)
    _require(epoch.get("schema") == "prospective_agentic_epoch_v1"
             and epoch.get("status") == "complete" and epoch.get("policy") == row["policy"],
             "epoch result schema/status/policy가 다르다")
    expected_score = RUN._score(category, epoch)
    expected_validity = RUN._validity(epoch_root, epoch)
    for key, item in {**expected_score, **expected_validity}.items():
        _require(row.get(key) == item, f"comparison row {key}가 raw 재계산과 다르다")
    mappings = {
        "model_requests": "request_count", "model_actions": "model_actions",
        "paired_probes": "probe_count", "decision_outcome": "decision_outcome",
        "parent_probe_wall_s": "parent_probe_wall_s",
        "owned_child_wall_s": "owned_child_wall_s",
    }
    for public_key, raw_key in mappings.items():
        _require(row.get(public_key) == epoch.get(raw_key),
                 f"comparison row {public_key}가 epoch 원문과 다르다")
    _finite(epoch.get("parent_probe_wall_s"), "parent probe wall")
    _finite(epoch.get("owned_child_wall_s"), "owned child wall")
    dialogue_paths: list[Path] = []
    for index, step in enumerate(epoch.get("steps", []), 1):
        after_path = epoch_root / f"step-{index:02d}/state-after.json"
        _require(after_path.is_file(), "step state-after 원문이 없다")
        after = _read(after_path)
        SEL.context(after)  # receipt proof, checkpoint, request ID, state replay를 함께 재검증한다.
        _require(after.get("state_digest") == step.get("state_after_digest"),
                 "step state digest가 result와 다르다")
        if "dialogue_sha256" in step:
            dialogue = epoch_root / f"step-{index:02d}/dialogue.json"
            _require(dialogue.is_file() and _sha(dialogue) == step["dialogue_sha256"],
                     "dialogue SHA가 epoch와 다르다")
            dialogue_paths.append(dialogue)
    final_state = _read(epoch_root / "final-state.json")
    SEL.context(final_state)
    _require(final_state.get("status") == epoch.get("final_status")
             and final_state.get("hypotheses") == epoch.get("remaining_candidates")
             and final_state.get("observed") == epoch.get("observations"),
             "final selector state가 epoch result와 다르다")
    return epoch, dialogue_paths


def export(source_batch_result: Path, comparison_artifact: Path, *, protocol_path: Path,
           anchor: str, out: Path) -> dict[str, Any]:
    """complete 또는 partial-failure 원문을 검증하고 새 공개 폴더에 write-once로 내보낸다."""
    source_batch_result = Path(source_batch_result).resolve()
    comparison_artifact = Path(comparison_artifact).resolve()
    protocol_path = Path(protocol_path).resolve()
    comparison_root = comparison_artifact.parent
    out = Path(out).resolve()
    _require(not out.exists(), "public export out은 fresh 경로여야 한다")
    protocol = RUN._read(protocol_path)
    cases = RUN._source_cases(source_batch_result, protocol, anchor)
    _require(len(cases) == 8, "source case 수가 8이 아니다")
    comparison = _read(comparison_artifact)
    rows = _comparison_rows(comparison)
    if comparison["status"] == "complete":
        _require(comparison.get("protocol") == RUN._ref(protocol_path)
                 and comparison.get("source_batch") == RUN._ref(source_batch_result)
                 and comparison.get("anchor") == anchor,
                 "comparison protocol/source/anchor ref가 다르다")

    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    public_sources: list[dict[str, Any]] = []
    for case in cases:
        fixed_clean, fixed_changed = _sanitize(case["fixed_report"])
        source = {
            "schema": SOURCE_SCHEMA, "case_id": case["case_id"], "seed": case["seed"],
            "category": case["category"], "checkpoint_sha256": case["checkpoint"]["sha256"],
            "fixed_report": fixed_clean,
            "fixed_report_is_sanitized": fixed_changed,
            "source_result": _logical("source", f"{case['case_id']}/result.json", _sha(case["result"])),
            "fixed_report_source": _logical(
                "source", f"{case['case_id']}/fixed-report.json", _sha(case["fixed"])),
        }
        _safe_public(source)
        target = out / "source" / f"{case['case_id']}.json"
        _write(target, source)
        files[target.relative_to(out).as_posix()] = {
            "sha256": _sha(target), "mode": "derived_verified_source_summary",
            "source": source["source_result"],
        }
        public_sources.append(source)

    case_by_id = {row["case_id"]: row for row in cases}
    public_rows: list[dict[str, Any]] = []
    dialogue_count = 0
    with RV.view_scope(ROOT / protocol["protocols"]["view"]["path"], anchor):
        for row in rows:
            case = case_by_id[row["case_id"]]
            epoch, dialogues = _verify_epoch(comparison_root, row, case["category"])
            public = {"schema": CASE_SCHEMA, **deepcopy(row),
                      "source": _logical("comparison", f"{row['case_id']}/{row['policy']}",
                                         _sha(comparison_artifact)),
                      "raw_epoch_present": epoch is not None}
            _safe_public(public)
            target = out / "cases" / row["case_id"] / f"{row['policy']}.json"
            _write(target, public)
            files[target.relative_to(out).as_posix()] = {
                "sha256": _sha(target), "mode": "derived_verified_comparison_row",
                "source": public["source"],
            }
            public_rows.append(public)
            for index, dialogue in enumerate(dialogues, 1):
                destination = (out / "dialogues" / row["case_id"] / row["policy"]
                               / f"dialogue-{index:02d}.json")
                entry = _copy_payload(
                    dialogue, destination,
                    f"{row['case_id']}/{row['policy']}/{dialogue.parent.name}/dialogue.json")
                files[destination.relative_to(out).as_posix()] = entry
                dialogue_count += 1

    planned = _row_order()
    completed = {(row["case_id"], row["policy"]) for row in rows}
    missing = [{"case_id": case, "policy": policy} for case, policy in planned
               if (case, policy) not in completed]
    summary = {
        "schema": SUMMARY_SCHEMA, "status": comparison["status"],
        "planned_source_training_jobs": 9, "planned_fixed_evaluations": 8,
        "planned_policy_rows": 24, "exported_policy_rows": len(public_rows),
        "missing_policy_rows": missing, "dialogue_files": dialogue_count,
        "rows": public_rows,
        "timing_scope": {"parent_probe_wall_s": "host orchestration wall",
                         "owned_child_wall_s": "owned child wall; not CUDA kernel time"},
        "claims": {"recovery": False, "model_version_advantage": False,
                   "system_effect_equals_model_effect": False,
                   "missing_or_excluded_is_success": False},
    }
    _safe_public(summary)
    _write(out / "summary.json", summary)
    files["summary.json"] = {"sha256": _sha(out / "summary.json"),
                             "mode": "derived_verified_summary",
                             "source": _logical("comparison", comparison_artifact.name,
                                                _sha(comparison_artifact))}
    manifest = {
        "schema": MANIFEST_SCHEMA, "files": files,
        "raw_sources": {
            "source_batch": _logical("source", source_batch_result.name, _sha(source_batch_result)),
            "comparison": _logical("comparison", comparison_artifact.name,
                                   _sha(comparison_artifact)),
            "protocol": {"logical_source": "protocol/comparison",
                         "raw_sha256": _sha(protocol_path)},
        },
        "derivation": {"raw_bytes_copied_only_when_safe": True,
                       "sanitized_payloads_carry_original_and_public_sha": True,
                       "checkpoint_binaries_included": False,
                       "training_stdout_included": False},
    }
    _safe_public(manifest)
    _write(out / "artifact_manifest.json", manifest)
    return deepcopy(manifest)
