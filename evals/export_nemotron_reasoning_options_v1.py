"""Nemotron 3.5 reasoning-mode development study를 검증해 trace 없이 공개한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from evals import export_prospective_agentic_comparison_v1 as PRIV
from evals import nemotron_reasoning_options_study_v1 as V1
from evals import nemotron_reasoning_options_study_v2 as V2

ROOT = V1.ROOT
SUMMARY_SCHEMA = "nemotron_reasoning_options_public_summary_v1"
MANIFEST_SCHEMA = "nemotron_reasoning_options_public_manifest_v1"
MODES = ("legacy_none", "enabled_low")
EXPECTED_CALLS = 32


def _require(ok: bool, reason: str) -> None:
    V1.require(ok, reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return V1._read(Path(path))


def _public_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_relative_to(ROOT), "공개 provenance ref가 저장소를 이탈했다")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}


def _verify_plan(plan_path: Path, corpus_path: Path, anchor: str) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "source anchor 형식")
    plan = V2.verify_plan(plan_path, corpus_path)
    relative = Path(plan_path).resolve().relative_to(ROOT).as_posix()
    V1._git("cat-file", "-e", f"{anchor}^{{commit}}")
    V1._git("merge-base", "--is-ancestor", anchor, "HEAD")
    _require(V1._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
             == Path(plan_path).read_bytes().replace(b"\r\n", b"\n"),
             "등록 anchor plan blob 불일치")
    for source, expected in plan["source_sha256_lf"].items():
        blob = V1._git("show", f"{anchor}:{source}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == expected,
                 f"등록 anchor source blob 불일치: {source}")
    return plan


def _bind_messages(path: Path, messages: list[dict[str, str]], raw_sha: str,
                   canonical_digest: str) -> None:
    _require(_sha(path) == raw_sha, "messages raw file SHA 불일치")
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(loaded == messages and V1.O2.O1._digest(messages) == canonical_digest,
             "messages payload/canonical digest 불일치")


def _result_rows(value: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("calls")
    _require(value.get("schema") == V2.RESULT_SCHEMA and value.get("status") == "complete"
             and value.get("request_count") == EXPECTED_CALLS
             and value.get("api_cap") == EXPECTED_CALLS
             and isinstance(rows, list) and len(rows) == EXPECTED_CALLS,
             "reasoning study complete 32행 분모 불일치")
    typed = cast(list[dict[str, Any]], rows)
    planned = plan["calls"]
    for row, expected in zip(typed, planned, strict=True):
        for key in ("call_index", "case_id", "mode", "replicate", "evaluation_role"):
            _require(row.get(key) == expected[key], f"result/plan {key} 결합 불일치")
    return typed


def _finite_number(value: object, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(cast(float, value))
             and cast(float, value) >= 0, f"{label}은 유한 비음수여야 한다")
    return float(cast(float, value))


def _validate_call(call_root: Path, planned: dict[str, Any], result_row: dict[str, Any],
                   case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    intent_path, private_path, public_path = (call_root / "intent.json",
                                               call_root / "private.json",
                                               call_root / "public.json")
    intent, private, public = _read(intent_path), _read(private_path), _read(public_path)
    request = V1.build_request(case["messages"], planned["mode"])
    request_sha = V1.digest(request)
    _require(intent.get("schema") == "nemotron_reasoning_options_intent_v2"
             and intent.get("planned") == planned and intent.get("request") == request
             and intent.get("messages_file_sha256") == planned["messages_file_sha256"]
             and intent.get("messages_digest") == planned["messages_digest"]
             and intent.get("request_sha256") == request_sha,
             "intent/plan/request/messages 결합 불일치")
    _require(private.get("schema") == V2.CALL_SCHEMA and private.get("planned") == planned
             and private.get("request_sha256") == request_sha,
             "private call/plan/request 결합 불일치")
    response = private.get("response")
    _require(isinstance(response, dict), "private response 형식")
    response = cast(dict[str, Any], response)
    review, action = V1._review(response, case["state"])
    _require(private.get("review") == review and private.get("model_action") == action
             and private.get("response_field_names") == sorted(response)
             and private.get("actual_system_gate_executed") is False
             and private.get("approval") is False and private.get("repair") is False
             and private.get("recovery") is False,
             "private review/action/권한 재생 불일치")
    _require(not (V1._walk_keys(review) & {"reasoning", "reasoning_content"}),
             "review에 private reasoning trace 필드가 남았다")
    expected_public = V1._public_call(private)
    _require(public == expected_public, "public call projection이 private 원문과 다르다")
    public_response = public.get("response", {})
    _require(isinstance(public_response, dict)
             and "reasoning_content" not in public_response and "reasoning" not in public_response
             and public.get("public_projection") == "reasoning_trace_removed",
             "공개 call에 reasoning body가 남았다")
    _require(_sha(private_path) == result_row.get("private_sha256")
             and _sha(public_path) == result_row.get("public_sha256"),
             "result call raw SHA 불일치")
    expected_row = {"call_index": planned["call_index"], "case_id": planned["case_id"],
                    "mode": planned["mode"], "replicate": planned["replicate"],
                    "evaluation_role": planned["evaluation_role"],
                    "valid": review.get("status") == "valid", "action": action,
                    "finish_reason": response.get("finish_reason"),
                    "provider_model": response.get("provider_model"),
                    "input_tokens": response.get("input_tokens"),
                    "output_tokens": response.get("output_tokens"),
                    "total_tokens": response.get("total_tokens"),
                    "reasoning_tokens": response.get("reasoning_tokens"),
                    "reasoning_tokens_status": response.get("reasoning_tokens_status"),
                    "wall_s": response.get("wall_s"),
                    "private_sha256": _sha(private_path), "public_sha256": _sha(public_path)}
    _require(result_row == expected_row, "result row가 raw call 재계산과 다르다")
    _finite_number(response.get("wall_s"), "call wall_s")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        _require(type(response.get(key)) is int and response[key] >= 0, f"{key} 형식")
    return public, {"intent_sha256": _sha(intent_path),
                    "private_sha256": _sha(private_path), "public_sha256": _sha(public_path)}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        fault = [row for row in selected if row["evaluation_role"] == "fault_development"]
        normal = [row for row in selected if row["evaluation_role"] == "normal_model_only"]
        modes[mode] = {"calls": len(selected), "valid": sum(row["valid"] is True for row in selected),
            "fault_calls": len(fault), "fault_valid": sum(row["valid"] is True for row in fault),
            "fault_read_or_probe": sum(row["valid"] is True and row["action"] in {
                "read_evidence", "run_probe"} for row in fault),
            "normal_calls": len(normal),
            "normal_valid_probe": sum(row["valid"] is True and row["action"] == "run_probe"
                                      for row in normal),
            "normal_read": sum(row["valid"] is True and row["action"] == "read_evidence"
                               for row in normal),
            "normal_hold": sum(row["valid"] is True and row["action"] == "hold" for row in normal),
            "incomplete": sum(row.get("finish_reason") != "stop" for row in selected),
            "reported_positive_reasoning_tokens": sum(
                type(row.get("reasoning_tokens")) is int and row["reasoning_tokens"] > 0
                for row in selected),
            "reported_zero_reasoning_tokens": sum(row.get("reasoning_tokens") == 0
                                                  for row in selected),
            "mean_parent_wall_s": sum(float(row["wall_s"]) for row in selected) / len(selected),
            "input_tokens": sum(int(row["input_tokens"]) for row in selected),
            "output_tokens": sum(int(row["output_tokens"]) for row in selected),
            "total_tokens": sum(int(row["total_tokens"]) for row in selected)}
    pairs = []
    for case in [f"c{i:02d}" for i in range(1, 9)]:
        for replicate in (1, 2):
            pair = [row for row in rows if row["case_id"] == case
                    and row["replicate"] == replicate]
            _require(len(pair) == 2 and {row["mode"] for row in pair} == set(MODES),
                     "case/replicate paired mode 결합 불일치")
            by_mode = {row["mode"]: row for row in pair}
            pairs.append({"case_id": case, "replicate": replicate,
                          "evaluation_role": pair[0]["evaluation_role"],
                          "both_valid": all(row["valid"] is True for row in pair),
                          "legacy_none_action": by_mode["legacy_none"]["action"],
                          "enabled_low_action": by_mode["enabled_low"]["action"],
                          "action_changed": (by_mode["legacy_none"]["action"]
                                             != by_mode["enabled_low"]["action"])})
    return {"modes": modes, "pairs": pairs}


def _adoption(plan: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    legacy, low = metrics["modes"]["legacy_none"], metrics["modes"]["enabled_low"]
    fault_pairs = [row for row in metrics["pairs"] if row["evaluation_role"] == "fault_development"]
    distinct = {row["case_id"] for row in fault_pairs if row["both_valid"]
                and row["enabled_low_action"] in {"read_evidence", "run_probe"}}
    checks = {"complete_calls": legacy["calls"] + low["calls"] == 32,
              "enabled_low_fault_valid_at_least_10_of_12": low["fault_valid"] >= 10,
              "enabled_low_fault_valid_not_below_legacy": low["fault_valid"] >= legacy["fault_valid"],
              "enabled_low_fault_read_or_probe_above_legacy":
                  low["fault_read_or_probe"] > legacy["fault_read_or_probe"],
              "distinct_fault_cases_with_valid_read_or_probe_at_least_2": len(distinct) >= 2,
              "normal_valid_probe_proposals_not_above_legacy":
                  low["normal_valid_probe"] <= legacy["normal_valid_probe"],
              "enabled_low_incomplete_not_above_legacy": low["incomplete"] <= legacy["incomplete"],
              "reasoning_surface_or_positive_reported_reasoning_tokens_required":
                  low["reported_positive_reasoning_tokens"] > 0,
              "actual_probe_confirmation_before_diagnosis_benefit": True}
    registered = plan["analysis"]["adoption_rule"]
    _require(registered["complete_calls"] == 32
             and registered["enabled_low_fault_valid_at_least"] == "10/12"
             and registered["distinct_fault_cases_with_valid_read_or_probe_at_least"] == 2,
             "등록 adoption rule 값 불일치")
    return {"registered_rule": deepcopy(registered), "checks": checks,
            "action_candidate": all(checks.values()),
            "scope": "development_action_candidate_only_not_default",
            "diagnosis_benefit": False, "actual_probe_confirmation_performed": False}


def export(result_path: Path, plan_path: Path, corpus_path: Path, *, anchor: str,
           out: Path) -> dict[str, Any]:
    result_path, plan_path, corpus_path, out = map(
        Path.resolve, (Path(result_path), Path(plan_path), Path(corpus_path), Path(out)))
    _require(not out.exists(), "public reasoning option out은 fresh 경로여야 한다")
    plan = _verify_plan(plan_path, corpus_path, anchor)
    corpus, cases = V1._corpus_ref(corpus_path, load_payloads=True)
    case_by_id = {row["case_id"]: row for row in cases}
    result = _read(result_path)
    rows = _result_rows(result, plan)
    _require(result.get("plan") == {"sha256": _sha(plan_path)}
             and result.get("corpus") == {"sha256": _sha(corpus_path)}
             and result.get("prior_invalid_api_requests") == 0,
             "result plan/corpus/prior-invalid 결합 불일치")
    execution_root = result_path.parent
    verified_public: list[dict[str, Any]] = []
    proof_rows: list[dict[str, Any]] = []
    for planned, row in zip(plan["calls"], rows, strict=True):
        case = case_by_id[planned["case_id"]]
        messages_path = V1._relative(corpus_path.parent, next(
            item for item in corpus["entries"] if item["case_id"] == planned["case_id"])["messages"])
        _bind_messages(messages_path, case["messages"], planned["messages_file_sha256"],
                       planned["messages_digest"])
        public, proofs = _validate_call(
            execution_root / "calls" / f"{planned['call_index']:02d}", planned, row, case)
        PRIV._safe_public(public)
        verified_public.append(public)
        proof_rows.append({"call_index": planned["call_index"], **proofs})
    metrics = _metrics(rows)
    adoption = _adoption(plan, metrics)
    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str]] = {}
    for case in cases:
        PRIV._safe_public(case["messages"])
        source = V1._relative(corpus_path.parent, next(
            item for item in corpus["entries"] if item["case_id"] == case["case_id"])["messages"])
        target = out / "source" / "messages" / f"{case['case_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        relative = target.relative_to(out).as_posix()
        files[relative] = {"sha256": _sha(target), "mode": "exact_source_messages_bytes"}
    for public, planned in zip(verified_public, plan["calls"], strict=True):
        target = out / "calls" / f"call-{planned['call_index']:02d}.json"
        source = execution_root / "calls" / f"{planned['call_index']:02d}" / "public.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        relative = target.relative_to(out).as_posix()
        files[relative] = {"sha256": _sha(target), "mode": "verified_public_projection"}
    case_mode = []
    for case in [f"c{i:02d}" for i in range(1, 9)]:
        for mode in MODES:
            selected = [row for row in rows if row["case_id"] == case and row["mode"] == mode]
            case_mode.append({"case_id": case, "mode": mode, "calls": len(selected),
                              "valid": sum(row["valid"] is True for row in selected),
                              "actions": {action: sum(row["action"] == action for row in selected)
                                          for action in ("hold", "read_evidence", "run_probe")},
                              "input_tokens": sum(row["input_tokens"] for row in selected),
                              "output_tokens": sum(row["output_tokens"] for row in selected),
                              "total_tokens": sum(row["total_tokens"] for row in selected),
                              "mean_parent_wall_s": sum(row["wall_s"] for row in selected) / 2})
    summary = {"schema": SUMMARY_SCHEMA, "status": "complete", "calls": 32,
               "modes": metrics["modes"], "case_mode": case_mode,
               "fault_paired_action_changes": [row for row in metrics["pairs"]
                                               if row["evaluation_role"] == "fault_development"],
               "adoption": adoption,
               "normal_model_only_counterfactual": True,
               "normal_host_gate_model_calls_in_primary_study": 0,
               "claims": {"first_action_proposals_only": True, "probe_confirmation": False,
                          "correction": False, "recovery": False, "speed_improvement": False,
                          "throughput_improvement": False, "deeper_rl_success": False,
                          "model_version_comparison": False, "same_model_different_request_mode": True,
                          "two_seed_repeated_conditions_correlated": True,
                          "default_adoption": False, "post_exposure_development": True}}
    PRIV._safe_public(summary)
    V1._write_new(out / "summary.json", summary)
    files["summary.json"] = {"sha256": _sha(out / "summary.json"), "mode": "derived_summary"}
    provenance = {"schema": "nemotron_reasoning_options_public_provenance_v1",
                  "creator_sha256": _sha(Path(__file__)), "source_anchor": anchor,
                  "result_sha256": _sha(result_path), "plan": _public_ref(plan_path),
                  "corpus_sha256": _sha(corpus_path),
                  "corpus_entries_digest": plan["corpus"]["entries_digest"],
                  "logical_source_ids": [f"call-{i:02d}" for i in range(1, 33)],
                  "raw_call_proofs": proof_rows,
                  "prior_invalid": deepcopy(plan["prior_invalid"])}
    PRIV._safe_public(provenance)
    V1._write_new(out / "source_provenance.json", provenance)
    files["source_provenance.json"] = {"sha256": _sha(out / "source_provenance.json"),
                                       "mode": "derived_provenance"}
    manifest = {"schema": MANIFEST_SCHEMA, "files": files,
                "summary": {"path": "summary.json", "sha256": _sha(out / "summary.json")},
                "provenance": {"path": "source_provenance.json",
                               "sha256": _sha(out / "source_provenance.json")},
                "privacy": {"private_paths": False, "reasoning_trace_body": False,
                            "reasoning_presence_and_tokens_only": True}}
    PRIV._safe_public(manifest)
    V1._write_new(out / "artifact_manifest.json", manifest)
    actual = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact_manifest.json"}, "manifest 누락/추가 파일")
    _require(all(_sha(out / name) == item["sha256"] for name, item in files.items()),
             "manifest 파일 SHA 재검증 실패")
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.result, args.plan, args.corpus, anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
