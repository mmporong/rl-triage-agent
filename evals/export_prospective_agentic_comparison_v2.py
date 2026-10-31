"""host eager exposure를 재생 검증해 기존 비교 공개 export를 완성한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from evals import export_prospective_agentic_comparison_v1 as V1

ROOT = V1.ROOT
RUN = V1.RUN
RV = V1.RV
SEL = V1.SEL
EXPORTER_SCHEMA = "prospective_agentic_comparison_exporter_v2"
HOST_EAGER_ACTION = {"action": "read_evidence", "target": SEL.RESOURCE, "fact_ids": []}


def _require(ok: bool, reason: str) -> None:
    V1._require(ok, reason)


def _host_eager(step: object, next_step: object, *, policy: str) -> None:
    _require(policy == "eager_model" and isinstance(step, dict),
             "state-after 없는 step은 eager host exposure만 허용한다")
    step = cast(dict[str, Any], step)
    expected_keys = {"kind", "action", "state_digest", "model_request"}
    _require(set(step) == expected_keys
             and step.get("kind") == "host_eager_exposure"
             and step.get("model_request") is False
             and step.get("action") == HOST_EAGER_ACTION,
             "eager host exposure shape/action이 다르다")
    _require(isinstance(next_step, dict), "eager exposure 뒤 첫 actual step이 없다")
    next_step = cast(dict[str, Any], next_step)
    _require(isinstance(next_step.get("state_before"), dict),
             "eager exposure 뒤 첫 actual state_before가 없다")
    before = next_step["state_before"]
    SEL.context(before)
    _require(before.get("state_digest") == step.get("state_digest"),
             "eager exposure digest가 첫 actual state_before와 다르다")
    history = before.get("history")
    _require(isinstance(history, list) and bool(history)
             and history[-1] == HOST_EAGER_ACTION,
             "첫 actual state가 eager read action을 재생하지 않는다")


def _verify_epoch(root: Path, row: dict[str, Any], category: str) -> tuple[
        dict[str, Any] | None, list[Path]]:
    """V1 검증을 유지하되 정확한 eager host event만 state-after 예외로 허용한다."""
    epoch_root = root / row["case_id"] / row["policy"]
    if not row.get("gate", {}).get("run"):
        _require(not epoch_root.exists() and row.get("model_requests") == 0
                 and row.get("paired_probes") == 0 and row.get("decision_outcome") == "HOLD",
                 "gate 중단 행에 모델·검사 실행 흔적이 있다")
        return None, []
    result_path = epoch_root / "result.json"
    _require(result_path.is_file(), "실행된 comparison epoch result가 없다")
    epoch = V1._read(result_path)
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
    V1._finite(epoch.get("parent_probe_wall_s"), "parent probe wall")
    V1._finite(epoch.get("owned_child_wall_s"), "owned child wall")
    steps = epoch.get("steps")
    _require(isinstance(steps, list), "epoch steps가 list가 아니다")
    steps = cast(list[dict[str, Any]], steps)
    dialogue_paths: list[Path] = []
    for index, step in enumerate(steps, 1):
        _require(isinstance(step, dict), "epoch step이 객체가 아니다")
        after_path = epoch_root / f"step-{index:02d}/state-after.json"
        if not after_path.is_file():
            _require(index == 1 and len(steps) > 1,
                     "일반 step state-after 원문이 없다")
            _host_eager(step, steps[1], policy=row["policy"])
            continue
        after = V1._read(after_path)
        SEL.context(after)
        _require(after.get("state_digest") == step.get("state_after_digest"),
                 "step state digest가 result와 다르다")
        if "dialogue_sha256" in step:
            dialogue = epoch_root / f"step-{index:02d}/dialogue.json"
            _require(dialogue.is_file() and V1._sha(dialogue) == step["dialogue_sha256"],
                     "dialogue SHA가 epoch와 다르다")
            dialogue_paths.append(dialogue)
    final_state = V1._read(epoch_root / "final-state.json")
    SEL.context(final_state)
    _require(final_state.get("status") == epoch.get("final_status")
             and final_state.get("hypotheses") == epoch.get("remaining_candidates")
             and final_state.get("observed") == epoch.get("observations"),
             "final selector state가 epoch result와 다르다")
    return epoch, dialogue_paths


def export(source_batch_result: Path, comparison_artifact: Path, *, protocol_path: Path,
           anchor: str, out: Path) -> dict[str, Any]:
    source_batch_result = Path(source_batch_result).resolve()
    comparison_artifact = Path(comparison_artifact).resolve()
    protocol_path = Path(protocol_path).resolve()
    comparison_root = comparison_artifact.parent
    out = Path(out).resolve()
    _require(not out.exists(), "public export out은 fresh 경로여야 한다")
    protocol = RUN._read(protocol_path)
    with RUN.COMP.CACHE.proof_scope() as cache_stats:
        cases = RUN._source_cases(source_batch_result, protocol, anchor)
        _require(len(cases) == 8, "source case 수가 8이 아니다")
        comparison = V1._read(comparison_artifact)
        rows = V1._comparison_rows(comparison)
        if comparison["status"] == "complete":
            _require(comparison.get("protocol") == RUN._ref(protocol_path)
                     and comparison.get("source_batch") == RUN._ref(source_batch_result)
                     and comparison.get("anchor") == anchor,
                     "comparison protocol/source/anchor ref가 다르다")
        out.mkdir(parents=True, exist_ok=False)
        files: dict[str, dict[str, Any]] = {}
        for case in cases:
            fixed_clean, fixed_changed = V1._sanitize(case["fixed_report"])
            source = {
                "schema": V1.SOURCE_SCHEMA, "case_id": case["case_id"], "seed": case["seed"],
                "category": case["category"], "checkpoint_sha256": case["checkpoint"]["sha256"],
                "fixed_report": fixed_clean, "fixed_report_is_sanitized": fixed_changed,
                "source_result": V1._logical("source", f"{case['case_id']}/result.json",
                                             V1._sha(case["result"])),
                "fixed_report_source": V1._logical(
                    "source", f"{case['case_id']}/fixed-report.json", V1._sha(case["fixed"])),
            }
            V1._safe_public(source)
            target = out / "source" / f"{case['case_id']}.json"
            V1._write(target, source)
            files[target.relative_to(out).as_posix()] = {
                "sha256": V1._sha(target), "mode": "derived_verified_source_summary",
                "source": source["source_result"]}
        case_by_id = {row["case_id"]: row for row in cases}
        public_rows: list[dict[str, Any]] = []
        dialogue_count = 0
        with RV.view_scope(ROOT / protocol["protocols"]["view"]["path"], anchor):
            for row in rows:
                case = case_by_id[row["case_id"]]
                epoch, dialogues = _verify_epoch(comparison_root, row, case["category"])
                public = {"schema": V1.CASE_SCHEMA, **deepcopy(row),
                          "source": V1._logical(
                              "comparison", f"{row['case_id']}/{row['policy']}",
                              V1._sha(comparison_artifact)), "raw_epoch_present": epoch is not None}
                V1._safe_public(public)
                target = out / "cases" / row["case_id"] / f"{row['policy']}.json"
                V1._write(target, public)
                files[target.relative_to(out).as_posix()] = {
                    "sha256": V1._sha(target), "mode": "derived_verified_comparison_row",
                    "source": public["source"]}
                public_rows.append(public)
                for index, dialogue in enumerate(dialogues, 1):
                    destination = (out / "dialogues" / row["case_id"] / row["policy"]
                                   / f"dialogue-{index:02d}.json")
                    entry = V1._copy_payload(
                        dialogue, destination,
                        f"{row['case_id']}/{row['policy']}/{dialogue.parent.name}/dialogue.json")
                    files[destination.relative_to(out).as_posix()] = entry
                    dialogue_count += 1
        planned = V1._row_order()
        completed = {(row["case_id"], row["policy"]) for row in rows}
        missing = [{"case_id": case, "policy": policy} for case, policy in planned
                   if (case, policy) not in completed]
        summary = {
            "schema": V1.SUMMARY_SCHEMA, "status": comparison["status"],
            "planned_source_training_jobs": 9, "planned_fixed_evaluations": 8,
            "planned_policy_rows": 24, "exported_policy_rows": len(public_rows),
            "missing_policy_rows": missing, "dialogue_files": dialogue_count, "rows": public_rows,
            "timing_scope": {"parent_probe_wall_s": "host orchestration wall",
                             "owned_child_wall_s": "owned child wall; not CUDA kernel time"},
            "claims": {"recovery": False, "model_version_advantage": False,
                       "system_effect_equals_model_effect": False,
                       "missing_or_excluded_is_success": False},
        }
        V1._safe_public(summary)
        V1._write(out / "summary.json", summary)
        files["summary.json"] = {
            "sha256": V1._sha(out / "summary.json"), "mode": "derived_verified_summary",
            "source": V1._logical("comparison", comparison_artifact.name,
                                  V1._sha(comparison_artifact))}
    provenance = {
        "schema": EXPORTER_SCHEMA,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_exporter_sha256": V1._sha(Path(V1.__file__)),
        "eager_host_exposure_rule": "exact_read_event_replayed_into_first_actual_state",
        "immutable_git_blob_cache": deepcopy(cache_stats),
        "mutable_runtime_checks_cached": False,
    }
    V1._safe_public(provenance)
    V1._write(out / "export_provenance.json", provenance)
    files["export_provenance.json"] = {
        "sha256": V1._sha(out / "export_provenance.json"),
        "mode": "derived_export_provenance",
        "source": V1._logical("exporter", "v2", provenance["exporter_sha256"])}
    manifest = {
        "schema": V1.MANIFEST_SCHEMA, "files": files,
        "raw_sources": {
            "source_batch": V1._logical("source", source_batch_result.name,
                                        V1._sha(source_batch_result)),
            "comparison": V1._logical("comparison", comparison_artifact.name,
                                      V1._sha(comparison_artifact)),
            "protocol": {"logical_source": "protocol/comparison",
                         "raw_sha256": V1._sha(protocol_path)}},
        "derivation": {"raw_bytes_copied_only_when_safe": True,
                       "sanitized_payloads_carry_original_and_public_sha": True,
                       "checkpoint_binaries_included": False,
                       "training_stdout_included": False,
                       "eager_host_exposure_replayed": True},
        "exporter_provenance": {"path": "export_provenance.json",
                                "sha256": V1._sha(out / "export_provenance.json")},
    }
    V1._safe_public(manifest)
    V1._write(out / "artifact_manifest.json", manifest)
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.source_batch, args.comparison, protocol_path=args.protocol,
           anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
