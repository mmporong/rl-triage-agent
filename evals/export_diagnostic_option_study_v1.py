"""V2 diagnostic option development study를 원문 재검증 후 공개한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from evals import export_prospective_agentic_comparison_v1 as PRIV
from evals import export_registry_action_study_v1 as OLD
from evals import registry_action_loop_v2 as LOOP
from evals import registry_action_options_v2 as OPTIONS
from evals import run_diagnostic_option_study_v1 as STUDY

ROOT = PRIV.ROOT
SUMMARY_SCHEMA = "diagnostic_option_study_public_summary_v1"
CASE_SCHEMA = "diagnostic_option_study_public_case_v1"
SOURCE_SCHEMA = "diagnostic_option_study_public_source_v1"
MANIFEST_SCHEMA = "diagnostic_option_study_public_manifest_v1"


def _require(ok: bool, reason: str) -> None:
    PRIV._require(ok, reason)


def _ref(path: Path) -> dict[str, str]:
    return STUDY.BASE._ref(Path(path).resolve())


def _verify_protocol(path: Path, anchor: str) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    value = PRIV._read(path)
    parent_ref = value.get("parent_study_protocol")
    _require(value.get("schema") == STUDY.SCHEMA and isinstance(parent_ref, dict),
             "diagnostic option protocol schema 불일치")
    parent_ref = cast(dict[str, str], parent_ref)
    prior_sha = value.get("prior_option_result_sha256")
    _require(isinstance(prior_sha, str), "prior option result SHA가 없다")
    parent_path = ROOT / parent_ref["path"]
    _require(value == STUDY.prepare_protocol(parent_path, cast(str, prior_sha)),
             "diagnostic option protocol/current source closure 불일치")
    STUDY.BASE._git("cat-file", "-e", f"{anchor}^{{commit}}")
    STUDY.BASE._git("merge-base", "--is-ancestor", anchor, "HEAD")
    STUDY.BASE._verify_anchor_file(path, anchor)
    for mapping in (value["source_sha256_lf"], value["inherited_source_sha256_lf"]):
        for relative, digest in mapping.items():
            blob = STUDY.BASE._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
            _require(hashlib.sha256(blob).hexdigest() == digest,
                     f"anchor source blob 불일치: {relative}")
    parent_blob = STUDY.BASE._git("show", f"{anchor}:{parent_ref['path']}").replace(b"\r\n", b"\n")
    _require(hashlib.sha256(parent_blob).hexdigest() == parent_ref["sha256_lf"],
             "parent study protocol anchor blob 불일치")
    return value


def _rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("rows")
    _require(value.get("schema") == STUDY.RESULT_SCHEMA and value.get("status") == "complete"
             and isinstance(rows, list) and len(rows) == 8,
             "study는 complete 8행이어야 공개할 수 있다")
    typed = cast(list[dict[str, Any]], rows)
    expected = [(f"c{i:02d}", STUDY.POLICY) for i in range(1, 9)]
    _require([(row.get("case_id"), row.get("policy")) for row in typed] == expected,
             "study row 순서·중복·policy가 등록 계획과 다르다")
    _require(value.get("planned_rows") == 8 and value.get("fault_rows") == 6
             and value.get("normal_host_gate_rows") == 2,
             "complete study의 8/6/2 분모가 없다")
    return typed


def _parent_option_dialogue(parent_root: Path, case_id: str, policy: str) -> Path:
    epoch_root = Path(parent_root).resolve() / case_id / policy / "epoch"
    result = PRIV._read(epoch_root / "result.json")
    _require(result.get("schema") == OLD.LOOP.SCHEMA and result.get("status") == "complete"
             and result.get("run_id") == f"{case_id}.{policy}",
             "parent option loop schema/status/run 결합 불일치")
    steps = result.get("steps")
    _require(isinstance(steps, list) and len(steps) == result.get("model_actions")
             and bool(steps), "parent option loop step이 없다")
    first = cast(list[dict[str, Any]], steps)[0]
    stage = epoch_root / "step-01"
    before = PRIV._read(stage / "state-before.json")
    OLD.STUDY.SEL.context(before)
    _require(before.get("state_digest") == first.get("state_before_digest"),
             "parent option state-before digest 불일치")
    _require(PRIV._read(stage / "option-context.json") == OLD.OPTIONS.context(before),
             "parent option context/state 불일치")
    dialogue_path = OLD._checked_local_ref(first.get("dialogue"), epoch_root,
                                           "parent option dialogue")
    _require(dialogue_path == (stage / "dialogue.json").resolve(),
             "parent option 첫 dialogue 단계 경로 불일치")
    dialogue = PRIV._read(dialogue_path)
    _require(dialogue.get("schema") == "registry_action_options_dialogue_v1"
             and dialogue.get("state_digest") == before.get("state_digest")
             and dialogue.get("action") == first.get("action")
             and OLD.OPTIONS.select_model(dialogue["final_review"], before) == first.get("action"),
             "parent option dialogue review/action 재생 불일치")
    return dialogue_path


def _verify_epoch(root: Path, row: dict[str, Any], category: str,
                  prior_root: Path) -> tuple[dict[str, Any] | None, list[Path], dict[str, Any]]:
    row_root = root / row["case_id"] / STUDY.POLICY
    epoch_root = row_root / "epoch"
    if not row.get("gate", {}).get("run"):
        _require(not row_root.exists() and row.get("model_requests") == 0
                 and row.get("paired_probes") == 0 and row.get("decision_outcome") == "HOLD"
                 and row.get("normal_gate_model_success") is False,
                 "normal host gate가 모델 성공/실행으로 기록됐다")
        return None, [], {"first_action": None, "first_response_valid": None,
                          "post_feedback_valid": None, "valid_hold": False,
                          "observed_actions": []}
    result_path = epoch_root / "result.json"
    _require(result_path.is_file(), "실행 study epoch result가 없다")
    epoch = PRIV._read(result_path)
    _require(epoch.get("schema") == LOOP.SCHEMA and epoch.get("status") == "complete"
             and epoch.get("run_id") == f"{row['case_id']}.{STUDY.POLICY}",
             "V2 loop result schema/status/run 결합 불일치")
    for field in ("decision_outcome", "model_actions", "paired_probes",
                  "parent_probe_wall_s", "owned_child_wall_s", "final_status"):
        _require(row.get(field) == epoch.get(field), f"study row {field} raw 불일치")
    for key, item in STUDY.BASE._score(category, epoch).items():
        _require(row.get(key) == item, f"study row {key} raw 재계산 불일치")
    steps = epoch.get("steps")
    _require(isinstance(steps, list) and len(steps) == epoch.get("model_actions"),
             "loop step/model action 수 불일치")
    dialogues: list[Path] = []
    previous_digest: str | None = None
    first_response_valid: bool | None = None
    post_feedback_valid: bool | None = None
    first_action: str | None = None
    model_requests = 0
    observed_actions: list[str] = []
    for index, step in enumerate(cast(list[dict[str, Any]], steps), 1):
        stage = epoch_root / f"step-{index:02d}"
        before = PRIV._read(stage / "state-before.json")
        STUDY.BASE.SEL.context(before)
        _require(before.get("state_digest") == step.get("state_before_digest")
                 and (previous_digest is None or before.get("state_digest") == previous_digest),
                 "loop state-before chain 불일치")
        _require(PRIV._read(stage / "option-context.json") == OPTIONS.context(before),
                 "V2 option context/state 불일치")
        dialogue_path = OLD._checked_local_ref(step.get("dialogue"), epoch_root, "dialogue")
        dialogue = PRIV._read(dialogue_path)
        _require(dialogue.get("schema") == "registry_action_options_dialogue_v2"
                 and dialogue.get("state_digest") == before["state_digest"]
                 and dialogue.get("action") == step.get("action")
                 and dialogue.get("model_valid") == step.get("model_valid"),
                 "V2 dialogue/action/state 결합 불일치")
        calls = cast(list[dict[str, Any]], dialogue.get("calls"))
        _require(1 <= len(calls) <= 2, "dialogue calls 상한 불일치")
        model_requests += len(calls)
        _require(OPTIONS.select_model(dialogue["final_review"], before) == step.get("action"),
                 "dialogue raw review 재생 action 불일치")
        if index == 1:
            first_response_valid = calls[0].get("review", {}).get("status") == "valid"
            post_feedback_valid = dialogue.get("model_valid") is True
            first_action = step.get("action", {}).get("action")
            prior = PRIV._read(_parent_option_dialogue(
                prior_root, row["case_id"], STUDY.POLICY))
            _require(OLD._dialogue_facts(dialogue) == OLD._dialogue_facts(prior),
                     "parent/new visible facts 원자료가 다르다")
        selected = PRIV._read(stage / "selected-action.json")
        _require(selected.get("action") == step.get("action")
                 and selected.get("dialogue") == step.get("dialogue"),
                 "selected action/dialogue ref 불일치")
        host_path = OLD._checked_local_ref(step.get("host_action"), epoch_root, "host action")
        _require(PRIV._read(host_path).get("action") == step.get("action"), "host action 불일치")
        if step.get("action", {}).get("action") == "run_probe":
            proof_path = stage / "paired-proof.json"
            _require(proof_path.is_file(), "run_probe paired proof가 없다")
            proof = PRIV._read(proof_path)
            for label, item in proof.items():
                OLD._checked_local_ref(item, epoch_root, f"paired proof {label}")
        after_path = OLD._checked_local_ref(step.get("state_after"), epoch_root, "state after")
        after = PRIV._read(after_path)
        STUDY.BASE.SEL.context(after)
        _require(after.get("state_digest") == step.get("state_after_digest"),
                 "state-after digest 불일치")
        previous_digest = cast(str, after["state_digest"])
        observed_actions.append(cast(str, step["action"]["action"]))
        dialogues.append(dialogue_path)
    final = PRIV._read(epoch_root / "final-state.json")
    STUDY.BASE.SEL.context(final)
    _require(final.get("state_digest") == previous_digest
             and final.get("status") == epoch.get("final_status"), "final state chain 불일치")
    _require(row.get("model_requests") == model_requests
             and row.get("first_decision_valid") == first_response_valid
             and row.get("post_feedback_first_decision_valid") == post_feedback_valid,
             "row first/post-feedback/model request 집계 불일치")
    return epoch, dialogues, {"first_action": first_action,
        "first_response_valid": first_response_valid,
        "post_feedback_valid": post_feedback_valid,
        "valid_hold": first_response_valid is True and first_action == "hold",
        "observed_actions": observed_actions}


def _candidate(rows: list[dict[str, Any]], prior_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fault = [row for row in rows if row.get("gate", {}).get("run")]
    normal = [row for row in rows if not row.get("gate", {}).get("run")]
    _require(len(fault) == 6 and len(normal) == 2, "candidate의 6 fault/2 normal 분모 불일치")
    first_valid = sum(row["first_response_valid"] is True for row in fault)
    correct = sum(row.get("correct_confirmed_family") is True for row in fault)
    wrong = sum(row.get("wrong_family_confirmation") is True for row in fault)
    parent_fault = [row for row in prior_rows if row.get("policy") == STUDY.POLICY
                    and row.get("gate", {}).get("run")]
    parent_correct = sum(row.get("correct_confirmed_family") is True for row in parent_fault)
    normal_clean = all(row.get("model_requests") == 0 and row.get("paired_probes") == 0
                       for row in normal)
    all_hold = all(row["first_action"] == "hold" for row in fault)
    checks = {"normal_rows_api_and_probe_zero": normal_clean,
              "wrong_family_confirmations_zero": wrong == 0,
              "first_valid_at_least_5_of_6": first_valid >= 5,
              "correct_family_strictly_above_parent": correct > parent_correct,
              "all_six_hold": all_hold}
    return {"candidate": (all(checks[key] for key in checks if key != "all_six_hold")
                          and not all_hold), "checks": checks, "first_valid": first_valid,
            "correct": correct, "wrong": wrong, "parent_correct": parent_correct,
            "normal_clean": normal_clean, "all_hold": all_hold}


def export(source_batch: Path, prior_result: Path, study_result: Path, *,
           protocol_path: Path, anchor: str, out: Path) -> dict[str, Any]:
    source_batch, prior_result = Path(source_batch).resolve(), Path(prior_result).resolve()
    study_result, protocol_path, out = (Path(study_result).resolve(),
                                        Path(protocol_path).resolve(), Path(out).resolve())
    _require(not out.exists(), "public diagnostic option out은 fresh 경로여야 한다")
    with STUDY.COMP.CACHE.proof_scope() as cache_stats:
        protocol = _verify_protocol(protocol_path, anchor)
        _require(PRIV._sha(prior_result) == protocol["prior_option_result_sha256"],
                 "parent option result SHA 불일치")
        prior = STUDY._prior(prior_result, protocol)
        parent = PRIV._read(ROOT / protocol["parent_study_protocol"]["path"])
        base = PRIV._read(ROOT / parent["base_comparison_protocol"]["path"])
        cases = STUDY.RUN._source_cases(source_batch, protocol=base, anchor=anchor)
        value = PRIV._read(study_result)
        rows = _rows(value)
        _require(value.get("protocol") == _ref(protocol_path)
                 and value.get("source_batch") == _ref(source_batch)
                 and value.get("prior_result") == _ref(prior_result),
                 "study protocol/source/parent ref 불일치")
        out.mkdir(parents=True, exist_ok=False)
        files: dict[str, dict[str, Any]] = {}
        case_by_id = {case["case_id"]: case for case in cases}
        public_rows, dialogue_count = [], 0
        for case in cases:
            fixed, changed = PRIV._sanitize(case["fixed_report"])
            source = {"schema": SOURCE_SCHEMA, "case_id": case["case_id"],
                      "seed": case["seed"], "category": case["category"],
                      "checkpoint_sha256": case["checkpoint"]["sha256"],
                      "fixed_report": fixed, "fixed_report_is_sanitized": changed,
                      "source_result_sha256": PRIV._sha(case["result"]),
                      "fixed_report_sha256": PRIV._sha(case["fixed"])}
            PRIV._safe_public(source)
            target = out / "source" / f"{case['case_id']}.json"
            PRIV._write(target, source)
            files[target.relative_to(out).as_posix()] = {
                "sha256": PRIV._sha(target), "mode": "derived_verified_source_summary"}
        for row in rows:
            case = case_by_id[row["case_id"]]
            epoch, dialogues, metrics = _verify_epoch(
                study_result.parent, row, case["category"], prior_result.parent)
            public = {"schema": CASE_SCHEMA, **deepcopy(row), **metrics,
                      "raw_epoch_present": epoch is not None,
                      "source": {"logical_source": f"study/{row['case_id']}/{STUDY.POLICY}",
                                 "raw_result_sha256": PRIV._sha(study_result)}}
            PRIV._safe_public(public)
            target = out / "cases" / f"{row['case_id']}.json"
            PRIV._write(target, public)
            files[target.relative_to(out).as_posix()] = {
                "sha256": PRIV._sha(target), "mode": "derived_verified_study_row"}
            public_rows.append(public)
            for index, dialogue in enumerate(dialogues, 1):
                target = out / "dialogues" / row["case_id"] / f"dialogue-{index:02d}.json"
                relative = target.relative_to(out).as_posix()
                copied = PRIV._copy_payload(
                    dialogue, target, f"{row['case_id']}/{dialogue.parent.name}/dialogue.json")
                copied["path"] = relative
                files[relative] = copied
                dialogue_count += 1
        fault = [row for row in public_rows if row.get("gate", {}).get("run")]
        candidate = _candidate(public_rows, prior["rows"])
        _require(value.get("validity", {}).get("first_decision_valid") == candidate["first_valid"]
                 and value.get("validity", {}).get("first_decision_denominator") == 6,
                 "parent first-decision validity 재계산 불일치")
        summary = {"schema": SUMMARY_SCHEMA, "status": "complete",
                   "planned_rows": 8, "exported_rows": 8, "fault_rows": 6,
                   "normal_host_gate_rows": 2, "dialogue_files": dialogue_count,
                   "first_decision": {"valid": candidate["first_valid"], "denominator": 6,
                                      "valid_hold": sum(row["valid_hold"] is True for row in fault)},
                   "actions": {name: sum(name in row["observed_actions"] for row in fault)
                               for name in ("read_evidence", "run_probe", "hold")},
                   "confirmation": {"correct_confirmed_family": candidate["correct"],
                                    "parent_guided_correct_confirmed_family": candidate["parent_correct"],
                                    "compound_partial": sum(row.get("compound_partial") is True
                                                            for row in fault),
                                    "wrong_family_confirmation": candidate["wrong"]},
                   "development_candidate": candidate["candidate"],
                   "candidate_rule_result": candidate["checks"],
                   "rows": public_rows,
                   "claims": {"post_exposure_development": True,
                              "registry_prediction_metadata_newly_exposed": True,
                              "first_format_validity_is_confirmation": False,
                              "default_adoption": False, "model_version_effect": False,
                              "recovery": False, "whole_loop_recovery": False}}
        PRIV._safe_public(summary)
        PRIV._write(out / "summary.json", summary)
        files["summary.json"] = {"sha256": PRIV._sha(out / "summary.json"),
                                 "mode": "derived_verified_summary"}
    provenance = {"schema": "diagnostic_option_study_export_provenance_v1",
                  "source_anchor": anchor,
                  "registered_protocol": STUDY.BASE._public_ref(protocol_path),
                  "parent_study_protocol": protocol["parent_study_protocol"],
                  "protocol_sha256": PRIV._sha(protocol_path),
                  "source_batch_sha256": PRIV._sha(source_batch),
                  "parent_option_result_sha256": PRIV._sha(prior_result),
                  "study_result_sha256": PRIV._sha(study_result),
                  "immutable_git_blob_cache": deepcopy(cache_stats),
                  "mutable_runtime_checks_cached": False}
    PRIV._safe_public(provenance)
    PRIV._write(out / "source_provenance.json", provenance)
    files["source_provenance.json"] = {"sha256": PRIV._sha(out / "source_provenance.json"),
                                       "mode": "derived_source_provenance"}
    manifest = {"schema": MANIFEST_SCHEMA, "files": files,
                "summary": {"path": "summary.json", "sha256": PRIV._sha(out / "summary.json")},
                "source_provenance": {"path": "source_provenance.json",
                                      "sha256": PRIV._sha(out / "source_provenance.json")},
                "privacy": {"private_paths_in_public": False, "checkpoint_binaries": False,
                            "training_logs": False, "safe_dialogue_copy_or_sanitized": True}}
    PRIV._safe_public(manifest)
    PRIV._write(out / "artifact_manifest.json", manifest)
    actual = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact_manifest.json"},
             "public artifact manifest 누락/추가 파일 불일치")
    for relative, item in files.items():
        _require(PRIV._sha(out / relative) == item["sha256"],
                 f"public artifact manifest SHA 불일치: {relative}")
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--study-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.source_batch, args.prior_result, args.study_result,
           protocol_path=args.protocol, anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
