"""등록 option 계약 development study를 원문 재검증 후 공개한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from evals import export_prospective_agentic_comparison_v1 as PRIV
from evals import registry_action_loop_v1 as LOOP
from evals import registry_action_options_v1 as OPTIONS
from evals import run_registry_action_study_v1 as STUDY

ROOT = PRIV.ROOT
SUMMARY_SCHEMA = "registry_action_contract_public_summary_v1"
CASE_SCHEMA = "registry_action_contract_public_case_v1"
SOURCE_SCHEMA = "registry_action_contract_public_source_v1"
MANIFEST_SCHEMA = "registry_action_contract_public_manifest_v1"
POLICIES = STUDY.POLICIES


def _require(ok: bool, reason: str) -> None:
    PRIV._require(ok, reason)


def _ref(path: Path) -> dict[str, str]:
    return STUDY._ref(Path(path).resolve())


def _row_order() -> list[tuple[str, str]]:
    return [(f"c{case + 1:02d}", policy) for case in range(8)
            for policy in STUDY.COMP.policy_order(case) if policy in POLICIES]


def _verify_protocol(path: Path, anchor: str) -> dict[str, Any]:
    """TF/lab tree를 읽지 않고 공개 protocol·source immutable blob만 검증한다."""
    value = PRIV._read(path)
    base_ref = value.get("base_comparison_protocol")
    _require(value.get("schema") == STUDY.SCHEMA and isinstance(base_ref, dict),
             "study protocol schema 불일치")
    base_ref = cast(dict[str, str], base_ref)
    prior_sha = value.get("prior_comparison_result_sha256")
    _require(isinstance(prior_sha, str), "prior comparison SHA가 없다")
    base_path = ROOT / base_ref["path"]
    expected = STUDY.prepare_protocol(base_path, cast(str, prior_sha))
    _require(value == expected, "study protocol이 현재 source closure와 다르다")
    STUDY._git("cat-file", "-e", f"{anchor}^{{commit}}")
    STUDY._git("merge-base", "--is-ancestor", anchor, "HEAD")
    STUDY._verify_anchor_file(path, anchor)
    for mapping in (value["source_sha256_lf"], value["inherited_source_sha256_lf"]):
        for relative, digest in mapping.items():
            blob = STUDY._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
            _require(hashlib.sha256(blob).hexdigest() == digest,
                     f"study anchor source blob 불일치: {relative}")
    base_blob = STUDY._git("show", f"{anchor}:{base_ref['path']}").replace(b"\r\n", b"\n")
    _require(hashlib.sha256(base_blob).hexdigest() == base_ref["sha256_lf"],
             "base comparison protocol anchor blob 불일치")
    return value


def _rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("rows")
    _require(value.get("schema") == STUDY.RESULT_SCHEMA
             and value.get("status") in {"complete", "invalid"}
             and isinstance(rows, list) and len(rows) <= 16, "study result schema/rows 불일치")
    typed = cast(list[dict[str, Any]], rows)
    expected = _row_order()
    _require([(row.get("case_id"), row.get("policy")) for row in typed] == expected[:len(typed)],
             "study row 순서·중복이 등록 계획과 다르다")
    if value["status"] == "complete":
        _require(len(typed) == 16 and value.get("planned_rows") == 16
                 and value.get("fault_rows") == 12
                 and value.get("normal_host_gate_rows") == 4,
                 "complete study의 16/12/4 분모가 없다")
    return typed


def _checked_local_ref(ref: object, root: Path, label: str) -> Path:
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, f"{label} ref shape")
    ref = cast(dict[str, str], ref)
    path = Path(ref["path"]).resolve()
    _require(path.is_relative_to(root.resolve()) and path.is_file()
             and PRIV._sha(path) == ref["sha256"], f"{label} ref/SHA/root 불일치")
    return path


def _dialogue_facts(dialogue: dict[str, Any]) -> dict[str, Any]:
    calls = dialogue.get("calls")
    _require(isinstance(calls, list) and 1 <= len(calls) <= 2, "dialogue calls 수 불일치")
    first = cast(list[dict[str, Any]], calls)[0]
    messages = first.get("messages")
    _require(isinstance(messages, list) and len(messages) >= 2
             and messages[1].get("role") == "user", "dialogue user context가 없다")
    messages = cast(list[dict[str, Any]], messages)
    context = json.loads(messages[1]["content"])
    _require(isinstance(context, dict) and isinstance(context.get("evidence_facts"), dict),
             "dialogue visible facts가 없다")
    return cast(dict[str, Any], context["evidence_facts"])


def _prior_dialogue(prior_root: Path, case_id: str, policy: str) -> Path:
    epoch_root = prior_root / case_id / policy
    epoch = PRIV._read(epoch_root / "result.json")
    for index, step in enumerate(epoch.get("steps", []), 1):
        digest = step.get("dialogue_sha256")
        if digest is None:
            continue
        path = epoch_root / f"step-{index:02d}" / "dialogue.json"
        _require(path.is_file() and PRIV._sha(path) == digest, "prior dialogue SHA 불일치")
        return path
    raise ValueError("prior fault row 첫 dialogue가 없다")


def _verify_epoch(root: Path, row: dict[str, Any], category: str,
                  prior_root: Path) -> tuple[dict[str, Any] | None, list[Path], dict[str, Any]]:
    row_root = root / row["case_id"] / row["policy"]
    epoch_root = row_root / "epoch"
    if not row.get("gate", {}).get("run"):
        _require(not row_root.exists() and row.get("model_requests") == 0
                 and row.get("paired_probes") == 0 and row.get("decision_outcome") == "HOLD"
                 and row.get("normal_gate_model_success") is False,
                 "normal host gate가 모델 성공/실행으로 기록됐다")
        return None, [], {"first_action": None, "first_response_valid": None,
                          "post_feedback_valid": None, "valid_hold": False}
    result_path = epoch_root / "result.json"
    _require(result_path.is_file(), "실행 study epoch result가 없다")
    epoch = PRIV._read(result_path)
    _require(epoch.get("schema") == LOOP.SCHEMA and epoch.get("status") == "complete"
             and epoch.get("run_id") == f"{row['case_id']}.{row['policy']}",
             "loop result schema/status/run 결합 불일치")
    mappings = {
        "decision_outcome": "decision_outcome", "model_actions": "model_actions",
        "paired_probes": "paired_probes", "parent_probe_wall_s": "parent_probe_wall_s",
        "owned_child_wall_s": "owned_child_wall_s", "final_status": "final_status",
    }
    for public, raw in mappings.items():
        _require(row.get(public) == epoch.get(raw), f"study row {public} raw 불일치")
    expected_score = STUDY._score(category, epoch)
    for key, item in expected_score.items():
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
    for index, step in enumerate(cast(list[dict[str, Any]], steps), 1):
        stage = epoch_root / f"step-{index:02d}"
        before_path = stage / "state-before.json"
        _require(before_path.is_file(), "loop state-before가 없다")
        before = PRIV._read(before_path)
        STUDY.SEL.context(before)
        _require(before.get("state_digest") == step.get("state_before_digest")
                 and (previous_digest is None or before.get("state_digest") == previous_digest),
                 "loop state-before chain 불일치")
        context_path = stage / "option-context.json"
        context = PRIV._read(context_path)
        _require(context == OPTIONS.context(before), "option context가 state 원문과 다르다")
        dialogue_path = _checked_local_ref(step.get("dialogue"), epoch_root, "dialogue")
        _require(dialogue_path == (stage / "dialogue.json").resolve(), "dialogue 단계 경로 불일치")
        dialogue = PRIV._read(dialogue_path)
        _require(dialogue.get("state_digest") == before["state_digest"]
                 and dialogue.get("action") == step.get("action")
                 and dialogue.get("model_valid") == step.get("model_valid"),
                 "dialogue/action/state 결합 불일치")
        calls = cast(list[dict[str, Any]], dialogue["calls"])
        _require(1 <= len(calls) <= 2, "dialogue calls 상한 불일치")
        model_requests += len(calls)
        if index == 1:
            first_response_valid = calls[0].get("review", {}).get("status") == "valid"
            post_feedback_valid = dialogue.get("model_valid") is True
            first_action = step.get("action", {}).get("action")
            prior = PRIV._read(_prior_dialogue(prior_root, row["case_id"], row["policy"]))
            _require(_dialogue_facts(dialogue) == _dialogue_facts(prior),
                     "old/new visible facts 원자료가 다르다")
        selected = PRIV._read(stage / "selected-action.json")
        _require(selected.get("action") == step.get("action")
                 and selected.get("dialogue") == step.get("dialogue"),
                 "selected action/dialogue ref 불일치")
        host_action_path = _checked_local_ref(step.get("host_action"), epoch_root, "host action")
        host_action = PRIV._read(host_action_path)
        _require(host_action.get("action") == step.get("action"), "host action 불일치")
        after_path = _checked_local_ref(step.get("state_after"), epoch_root, "state after")
        after = PRIV._read(after_path)
        STUDY.SEL.context(after)
        _require(after.get("state_digest") == step.get("state_after_digest"),
                 "state-after digest 불일치")
        previous_digest = cast(str, after["state_digest"])
        dialogues.append(dialogue_path)
    final = PRIV._read(epoch_root / "final-state.json")
    STUDY.SEL.context(final)
    _require(final.get("state_digest") == previous_digest
             and final.get("status") == epoch.get("final_status"), "final state chain 불일치")
    _require(row.get("model_requests") == model_requests
             and row.get("first_decision_valid") == first_response_valid
             and row.get("post_feedback_first_decision_valid") == post_feedback_valid,
             "row first/post-feedback/model request 집계 불일치")
    metrics = {"first_action": first_action, "first_response_valid": first_response_valid,
               "post_feedback_valid": post_feedback_valid,
               "valid_hold": first_response_valid is True and first_action == "hold"}
    return epoch, dialogues, metrics


def export(source_batch: Path, prior_result: Path, study_result: Path, *,
           protocol_path: Path, anchor: str, out: Path) -> dict[str, Any]:
    source_batch, prior_result = Path(source_batch).resolve(), Path(prior_result).resolve()
    study_result, protocol_path = Path(study_result).resolve(), Path(protocol_path).resolve()
    out = Path(out).resolve()
    _require(not out.exists(), "public study out은 fresh 경로여야 한다")
    with STUDY.COMP.CACHE.proof_scope() as cache_stats:
        protocol = _verify_protocol(protocol_path, anchor)
        _require(PRIV._sha(prior_result) == protocol["prior_comparison_result_sha256"],
                 "prior comparison SHA가 protocol과 다르다")
        prior = STUDY._prior(prior_result, protocol)
        cases = STUDY.RUN._source_cases(
            source_batch, PRIV._read(ROOT / protocol["base_comparison_protocol"]["path"]), anchor)
        value = PRIV._read(study_result)
        rows = _rows(value)
        if value["status"] == "complete":
            _require(value.get("protocol") == _ref(protocol_path)
                     and value.get("source_batch") == _ref(source_batch)
                     and value.get("prior_result") == _ref(prior_result),
                     "study protocol/source/prior ref 불일치")
        out.mkdir(parents=True, exist_ok=False)
        files: dict[str, dict[str, Any]] = {}
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
            files[target.relative_to(out).as_posix()] = {"sha256": PRIV._sha(target),
                "mode": "derived_verified_source_summary"}
        case_by_id = {case["case_id"]: case for case in cases}
        public_rows: list[dict[str, Any]] = []
        dialogue_count = 0
        prior_root = prior_result.parent
        for row in rows:
            case = case_by_id[row["case_id"]]
            epoch, dialogues, metrics = _verify_epoch(
                study_result.parent, row, case["category"], prior_root)
            public = {"schema": CASE_SCHEMA, **deepcopy(row), **metrics,
                      "raw_epoch_present": epoch is not None,
                      "source": {"logical_source": f"study/{row['case_id']}/{row['policy']}",
                                 "raw_result_sha256": PRIV._sha(study_result)}}
            PRIV._safe_public(public)
            target = out / "cases" / row["case_id"] / f"{row['policy']}.json"
            PRIV._write(target, public)
            files[target.relative_to(out).as_posix()] = {"sha256": PRIV._sha(target),
                "mode": "derived_verified_study_row"}
            public_rows.append(public)
            for index, dialogue in enumerate(dialogues, 1):
                target = out / "dialogues" / row["case_id"] / row["policy"] / f"dialogue-{index:02d}.json"
                files[target.relative_to(out).as_posix()] = PRIV._copy_payload(
                    dialogue, target,
                    f"{row['case_id']}/{row['policy']}/{dialogue.parent.name}/dialogue.json")
                dialogue_count += 1
        missing = [{"case_id": case, "policy": policy} for case, policy in _row_order()
                   if (case, policy) not in {(r["case_id"], r["policy"]) for r in rows}]
        fault = [row for row in public_rows if row.get("gate", {}).get("run")]
        first_valid = sum(row["first_response_valid"] is True for row in fault)
        prior_first = STUDY._prior_first_decisions(prior_result, prior["rows"])
        expected_candidate = first_valid >= 10 and first_valid > prior_first
        if value["status"] == "complete":
            _require(value.get("validity", {}).get("first_decision_valid") == first_valid
                     and value.get("validity", {}).get("first_decision_denominator") == 12
                     and value.get("format_candidate") == expected_candidate,
                     "parent first-decision/format candidate 재계산 불일치")
        summary = {
            "schema": SUMMARY_SCHEMA, "status": value["status"], "planned_rows": 16,
            "exported_rows": len(public_rows), "missing_rows": missing,
            "fault_rows_exported": len(fault),
            "normal_host_gate_rows": sum(not row.get("gate", {}).get("run") for row in public_rows),
            "dialogue_files": dialogue_count,
            "first_decision": {"valid": first_valid, "denominator": len(fault),
                               "valid_hold": sum(row["valid_hold"] is True for row in fault),
                               "prior_first_valid": prior_first,
                               "format_candidate": expected_candidate,
                               "format_candidate_is_diagnostic_benefit": False},
            "actions": {name: sum(row["first_action"] == name for row in fault)
                        for name in ("read_evidence", "run_probe", "hold")},
            "rows": public_rows,
            "claims": {"post_exposure_development": True, "default_adoption": False,
                       "model_version_effect": False, "recovery": False,
                       "strong_rule_time_advantage": False,
                       "normal_host_gate_is_model_success": False},
        }
        PRIV._safe_public(summary)
        PRIV._write(out / "summary.json", summary)
        files["summary.json"] = {"sha256": PRIV._sha(out / "summary.json"),
                                  "mode": "derived_verified_summary"}
    provenance = {"schema": "registry_action_contract_export_provenance_v1",
                  "source_batch_sha256": PRIV._sha(source_batch),
                  "prior_comparison_sha256": PRIV._sha(prior_result),
                  "study_result_sha256": PRIV._sha(study_result),
                  "protocol_sha256": PRIV._sha(protocol_path),
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
