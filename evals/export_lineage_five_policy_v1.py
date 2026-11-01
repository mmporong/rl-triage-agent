# pyright: reportMissingImports=false
"""R3 five-policy 결과를 원문 경로 없이 공개 JSON 네 파일로 투영한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]

import lineage_five_policy_summary_v1 as SUMMARY

SCHEMA = "lineage_five_policy_public_release_v1"
CASES_SCHEMA = "lineage_five_policy_public_cases_v1"
PROVENANCE_SCHEMA = "lineage_five_policy_public_provenance_v1"
MANIFEST_SCHEMA = "lineage_five_policy_public_manifest_v1"
SOURCE_FILES = (
    "evals/export_lineage_five_policy_v1.py",
    "tests/test_export_lineage_five_policy_v1.py",
    "evals/lineage_five_policy_summary_v1.py",
    "tests/test_lineage_five_policy_summary_v1.py",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자: {token}")
        ),
    )
    _require(isinstance(value, dict), "JSON object가 필요하다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _digest_ref(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    return {"sha256": _sha(target), "size_bytes": target.stat().st_size}


def _public_checkpoint_bindings(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    _require(isinstance(value, Mapping) and set(value) == {"case", "reference"},
             "checkpoint binding 역할")
    result: dict[str, Any] = {}
    for role in ("case", "reference"):
        row = cast(Mapping[str, Any], value)[role]
        _require(isinstance(row, Mapping), f"{role} checkpoint binding")
        checkpoint = row.get("checkpoint_sha256")
        _require(isinstance(checkpoint, str) and len(checkpoint) == 64,
                 f"{role} checkpoint SHA")
        result[role] = {"checkpoint_sha256": checkpoint}
        for name in ("telemetry_sha256", "fixed_result_sha256"):
            if name in row:
                item = row[name]
                _require(isinstance(item, str) and len(item) == 64, f"{role} {name}")
                result[role][name] = item
    return result


def _public_step(value: object) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "probe step")
    row = cast(Mapping[str, Any], value)
    facts = row.get("fact_ids", [])
    _require(isinstance(facts, list) and all(isinstance(item, str) for item in facts),
             "probe fact IDs")
    return {
        "probe": row.get("probe"),
        "outcome": row.get("outcome"),
        "fact_ids": deepcopy(facts),
        "owned_child_wall_s": row.get("owned_child_wall_s"),
    }


def _public_case(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "policy": row["policy"],
        "qualification": deepcopy(row["qualification"]),
        "scoring": deepcopy(row["scoring"]),
        "correct_any_family": row["correct"],
        "wrong_confirmation": row["wrong"],
        "compound_partial": row["compound_partial"],
        "compound_full": row["compound_full"],
        "normal_shadow_proposal": row["normal_shadow_proposal"],
        "normal_shadow_observed": row["normal_shadow_observed"],
        "normal_host_gate": row["host_gate"],
        "contract_invalid_events": row["contract_invalid_events"],
        "infra_unknown_events": row["infra_unknown_events"],
        "provider_unknown_events": row["provider_unknown_events"],
        "source_exclusion": row["source_exclusion"],
        "hold": row["hold"],
        "hold_reason": row["hold_reason"],
        "paired_probes": row["paired_probes"],
        "model_calls": row["model_calls"],
        "provider_tokens": deepcopy(row["provider_tokens"]),
        "provider_usage_unknown_calls": row["provider_usage_unknown_calls"],
        "provider_wall_s": row["provider_wall_s"],
        "owned_child_wall_s": row["owned_child_wall_s"],
        "owned_child_wall_unknown_probes": row["owned_child_wall_unknown_probes"],
        "parent_wall_s": row["parent_wall_s"],
        "frame_fact_citations": row["frame_fact_citations"],
        "source_digest": row["source_digest"],
        "checkpoint_bindings": _public_checkpoint_bindings(row["checkpoint_bindings"]),
        "confirmed_subtype": row["confirmed_subtype"],
        "steps": [_public_step(step) for step in row["steps"]],
        "technical_missing": row["technical_missing"],
    }


def _unavailable_cases(batch_path: Path) -> list[dict[str, Any]]:
    batch = _read(batch_path)
    rows = batch.get("rows")
    _require(isinstance(rows, list) and len(rows) == 50, "input-unavailable 50행")
    public = []
    for value in cast(list[Any], rows):
        _require(isinstance(value, Mapping), "batch row")
        row = cast(Mapping[str, Any], value)
        detail_path = SUMMARY._bound_file(row.get("result"), "public missing row")
        detail = SUMMARY._read(detail_path)
        public.append({
            "case_id": row.get("case_id"), "policy": row.get("arm"),
            "qualification": deepcopy(detail.get("qualification")),
            "scoring": deepcopy(detail.get("scoring")),
            "hold": detail.get("end_state") == "HOLD",
            "hold_reason": detail.get("hold_reason"),
            "input_unavailable": detail.get("input_unavailable"),
            "model_calls": detail.get("model_calls"),
            "paired_probes": len(detail.get("steps", [])),
            "source_digest": detail.get("source_digest"),
            "checkpoint_bindings": _public_checkpoint_bindings(
                detail.get("checkpoint_bindings")),
        })
    return public


def _public_summary(recomputed: Mapping[str, Any], batch: Path, study: Path) -> dict[str, Any]:
    result = deepcopy(dict(recomputed))
    result["batch"] = _digest_ref(batch)
    result["study"] = _digest_ref(study)
    unavailable = result.get("input_unavailable")
    if isinstance(unavailable, dict):
        for name in ("missing_rows", "observed_rows"):
            rows = unavailable.get(name)
            _require(isinstance(rows, list), f"input_unavailable {name}")
            for row in cast(list[Any], rows):
                _require(isinstance(row, dict) and isinstance(row.get("case_id"), str)
                         and isinstance(row.get("policy"), str),
                         f"input_unavailable {name} logical row")
                raw = SUMMARY._bound_file(row.get("result"), f"input_unavailable {name} result")
                row["result"] = _digest_ref(raw)
    result["public_projection"] = {
        "private_paths_removed": True,
        "raw_provider_responses_included": False,
        "reasoning_text_included": False,
        "api_keys_included": False,
        "numeric_and_null_values_reinterpreted": False,
    }
    return result


def _repo_relative(path: Path) -> str | None:
    target = Path(path).resolve(strict=True)
    try:
        return target.relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _public_protocol(value: object, *, allow_fixture: bool) -> dict[str, str] | None:
    if value is None:
        return None
    protocol_path = SUMMARY._bound_file(value, "execution protocol")
    relative_value = _repo_relative(protocol_path)
    _require(relative_value is not None, "공개 execution protocol은 저장소 파일이어야 한다")
    relative = cast(str, relative_value)
    if not allow_fixture:
        _require(relative.startswith("bench/protocols/") and relative.count("/") >= 2,
                 "actual execution protocol은 bench/protocols 공개 파일이어야 한다")
    return {"path": relative, "sha256": _sha(protocol_path)}


def _assert_public(*values: object) -> None:
    forbidden_key = re.compile(r"^(api_?key|authorization|credential|secret|access_?token)$", re.IGNORECASE)
    forbidden_value = re.compile(
        r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|tmp)/|bench[\\/]private|"
        r"Bearer\s+|nvapi-|sk-[A-Za-z0-9])", re.IGNORECASE)

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _require(not (isinstance(key, str) and forbidden_key.fullmatch(key)),
                         "공개 JSON에 자격증명 key가 있다")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            _require(forbidden_value.search(value) is None,
                     "공개 JSON에 개인·private 경로 또는 자격증명 값이 있다")

    for value in values:
        visit(value)


def export(batch_result: Path, study: Path, summary: Path, out: Path, *,
           allow_fixture: bool = False) -> dict[str, Any]:
    """원문을 다시 검증한 뒤 경로 없는 공개 summary/cases/provenance/manifest를 쓴다."""
    batch_path = Path(batch_result).resolve(strict=True)
    study_path = Path(study).resolve(strict=True)
    summary_path = Path(summary).resolve(strict=True)
    out = Path(out).resolve()
    _require(not out.exists(), "public output은 fresh 경로여야 한다")
    if not allow_fixture:
        _require(out.parent == (ROOT / "evals/results").resolve(),
                 "actual 공개 결과는 evals/results/<tag>에 써야 한다")

    supplied = _read(summary_path)
    with tempfile.TemporaryDirectory(prefix="lineage-five-policy-export-") as folder:
        verified_path = Path(folder) / "summary.json"
        recomputed = SUMMARY.summarize(
            batch_path, study_path, verified_path, allow_fixture=allow_fixture)
    _require(supplied == recomputed, "공급 summary가 원문 재계산 결과와 다르다")
    expected_scope = "cpu_fixture_development_measurement" if allow_fixture else None
    if allow_fixture:
        _require(recomputed.get("scope") == expected_scope, "fixture scope")
    else:
        _require(str(recomputed.get("scope", "")).startswith("actual_backend"),
                 "production export는 actual scope만 허용한다")

    if recomputed.get("scope") == "actual_backend_inconclusive_input_unavailable":
        cases = _unavailable_cases(batch_path)
    else:
        _batch, verified_rows = SUMMARY._grid(batch_path, allow_fixture=allow_fixture)
        cases = [_public_case(row) for row in verified_rows]
    _require(len(cases) == 50, "공개 case×policy 50행")

    batch_value = _read(batch_path)
    protocol_public = _public_protocol(
        batch_value.get("execution_protocol"), allow_fixture=allow_fixture)
    anchor = batch_value.get("execution_anchor")
    if not allow_fixture:
        _require(isinstance(anchor, str) and len(anchor) == 40, "execution anchor")

    public_summary = _public_summary(recomputed, batch_path, study_path)
    cases_value = {
        "schema": CASES_SCHEMA, "status": "complete", "scope": recomputed["scope"],
        "planned_cases": 10, "planned_policies": 5, "planned_rows": 50,
        "independent_unit": "source_lineage_not_row_or_call_or_frame",
        "rows": cases,
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA, "status": "complete",
        "scope": recomputed["scope"], "batch": _digest_ref(batch_path),
        "study": _digest_ref(study_path), "supplied_summary": _digest_ref(summary_path),
        "recomputed_summary_digest": _digest(recomputed),
        "execution_anchor": anchor, "execution_protocol": protocol_public,
        "source_files_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "claims": deepcopy(recomputed["claims"]),
        "fixture_promoted_to_actual": False,
        "raw_provider_responses_included": False,
    }

    _assert_public(public_summary, cases_value, provenance)

    out.mkdir(parents=True)
    _write_new(out / "summary.json", public_summary)
    _write_new(out / "cases.json", cases_value)
    _write_new(out / "provenance.json", provenance)
    files = {
        name: {"sha256": _sha(out / name), "size_bytes": (out / name).stat().st_size}
        for name in ("summary.json", "cases.json", "provenance.json")
    }
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "complete", "release_schema": SCHEMA,
        "scope": recomputed["scope"], "files": files,
        "private_paths_in_public_files": False, "overwrite_allowed": False,
    }
    _assert_public(manifest)
    _write_new(out / "manifest.json", manifest)
    return deepcopy(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-result", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.batch_result, args.study, args.summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
