"""전송 영상 선택 결과를 actual 평가·검사 원문과 결합하는 CPU 집계기."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import residual_demo_v1 as RECOVERY
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as METHOD
import vision_selection_execution_v1 as EXECUTION
import vision_transferred_input_acceptance_v2 as V2
import vision_transferred_probe_v1 as PROBE
import vision_transferred_selection_v1 as SELECTION

from rl_triage import grounded_recovery as GROUNDED

SCHEMA = "vision_transferred_summary_v1"
GROUP_SEED = {"source-01": 2220, "source-02": 2221}
ARM_ROLE = {"NONE": "normal", "F1": "registered-f1"}
SOURCE_PREFIX = Path(
    "rl-triage-brev-e2e-20261024/repo/.triage-runtime/vision-brev/"
    "vision-source-20261026x"
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _source_path(archive: Path, relative: Path) -> Path:
    tree = (Path(archive).resolve(strict=True) / "tree").resolve(strict=True)
    target = (tree / relative).resolve(strict=True)
    target.relative_to(tree)
    return target


def _metrics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        isinstance(left[key], (int, float)) and not isinstance(left[key], bool)
        and isinstance(right[key], (int, float)) and not isinstance(right[key], bool)
        and math.isclose(float(left[key]), float(right[key]), rel_tol=1e-12, abs_tol=1e-12)
        for key in left
    )


def source_labels(archive: Path, summary: Mapping[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """437-file archive와 평가 raw/report를 다시 읽어 네 행동 label을 계산한다."""
    proof = V2.verify_archive(Path(archive), dict(summary))
    if proof.get("files_verified") != 437 or proof.get("all_match") is not True:
        raise ValueError("source archive 437-file full inventory 검증이 아니다")
    bands_path = ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json"
    bands_summary = read(bands_path)
    source_protocol = read(ROOT / "bench/protocols/vision_brev_source_20261026_v8.json")
    registered_bands = source_protocol.get("evaluation", {})
    if (registered_bands.get("bands_path") != bands_path.relative_to(ROOT).as_posix()
            or PROBE.file_sha(bands_path, lf=True) != registered_bands.get("bands_sha256")):
        raise ValueError("source protocol의 frozen bands 결합이 다르다")
    checkpoints = {(row["seed_group"], row["source_role"]): row["checkpoint_sha256"]
                   for row in inputs}
    if set(checkpoints) != {(group, role) for group in METHOD.SEED_GROUPS for role in METHOD.ROLES}:
        raise ValueError("selection input의 네 source checkpoint 결합이 불완전하다")
    training = {(row.get("seed"), row.get("arm")): row for row in summary.get("training", [])
                if isinstance(row, Mapping)}
    evaluations = {row.get("seed"): row for row in summary.get("evaluations", [])
                   if isinstance(row, Mapping)}
    if set(evaluations) != set(GROUP_SEED.values()):
        raise ValueError("source summary evaluation seed 집합이 다르다")
    labels: dict[str, Any] = {}
    raw_hashes: set[str] = set()
    report_hashes: set[str] = set()
    for group, seed in GROUP_SEED.items():
        number = seed - 2216
        action = _source_path(archive, SOURCE_PREFIX / "actions" / f"0{number}_eval-s{seed}" / "raw.json")
        raw = read(action)
        public = evaluations[seed]
        action_sha = PROBE.file_sha(action)
        if (action_sha != public.get("raw_sha256") or action_sha in raw_hashes
                or raw.get("kind") != "evaluate" or raw.get("seed") != seed
                or raw.get("job_id") != f"eval-s{seed}"):
            raise ValueError(f"seed {seed} source evaluation raw 결합이 다르다")
        raw_hashes.add(action_sha)
        reports: dict[str, dict[str, Any]] = {}
        labels[group] = {}
        for arm, role in ARM_ROLE.items():
            report_path = _source_path(
                archive, SOURCE_PREFIX / "evaluate" / f"eval-s{seed}" / "reports" / f"{arm}.json",
            )
            report = read(report_path)
            report_sha = PROBE.file_sha(report_path)
            expected_checkpoint = checkpoints[group, role]
            public_training = training.get((seed, arm))
            if (report_sha in report_hashes or not isinstance(public_training, Mapping)
                    or public_training.get("checkpoint_sha256") != expected_checkpoint
                    or report.get("checkpoint", {}).get("sha256") != expected_checkpoint
                    or raw.get("evaluation", {}).get("reports", {}).get(arm) != report
                    or not _metrics_equal(public.get("metrics", {}).get(arm, {}), report.get("metrics", {}))
                    or public.get("conditions_per_policy") != len(report.get("by_condition", []))):
                raise ValueError(f"seed {seed} {arm} 평가 report/checkpoint 결합이 다르다")
            GROUNDED.validate_fixed_report(report, {
                "name": arm, "checkpoint_file": "model_299.pt",
                "checkpoint_sha256": expected_checkpoint, "action_scale": 0.25,
                "protocol_sha256_lf": source_protocol["source_sha256_lf"][
                    source_protocol["fixed_eval_protocol"]
                ],
            })
            report_hashes.add(report_sha)
            reports[arm] = report
        source_gate = RECOVERY.recovery(reports["F1"], reports["NONE"], bands_summary["bands"])
        if raw.get("source_gate") != source_gate or public.get("source_gate") != source_gate:
            raise ValueError(f"seed {seed} source gate 재계산이 다르다")
        verdicts = {
            "NONE": RECOVERY.recovery(reports["NONE"], reports["NONE"], bands_summary["bands"])["verdict"],
            "F1": source_gate["verdict"],
        }
        for arm, role in ARM_ROLE.items():
            labels[group][role] = {
                "label": verdicts[arm]["label"],
                "checkpoint_sha256": checkpoints[group, role],
                "fixed_eval_artifact_sha256": PROBE.file_sha(_source_path(
                    archive, SOURCE_PREFIX / "evaluate" / f"eval-s{seed}" / "reports" / f"{arm}.json",
                )),
            }
    if any(labels[group]["normal"]["label"] != "healthy"
           or labels[group]["registered-f1"]["label"] != "unhealthy" for group in METHOD.SEED_GROUPS):
        raise ValueError("source label 실측 재계산이 등록 비교 전제와 다르다")
    return {"labels": labels, "archive_verification": proof,
            "evaluation_raw_sha256": sorted(raw_hashes), "fixed_eval_sha256": sorted(report_hashes),
            "bands_sha256_lf": PROBE.file_sha(bands_path, lf=True)}


def probe_support(root: Path, protocol_path: Path, archive: Path) -> dict[str, Any]:
    """새 probe pair를 validate_pair로 재생하고 METHOD 입력으로만 투영한다."""
    protocol = PROBE.validate_protocol(protocol_path, require_bound=True)
    protocol["_path"] = str(Path(protocol_path).resolve(strict=True))
    PROBE.verify_repository_binding(root, protocol)
    source_inputs = PROBE.verify_source_inputs(root, archive, protocol)
    support: dict[str, dict[str, Any]] = {}
    missing = []
    for group in METHOD.SEED_GROUPS:
        stage = Path(root) / str(protocol["execution_root"]) / group
        if not stage.is_dir() or not (stage / "result.json").is_file():
            missing.append(group)
            continue
        verified = PROBE.validate_pair(root, protocol, source_inputs, stage)
        stored = read(stage / "result.json")
        if stored != verified or verified.get("group") != group:
            raise ValueError(f"{group} probe result 저장본과 actual 재검산이 다르다")
        request = read(stage / "authorization.json").get("request", {})
        support[group] = {
            "probe": verified["probe"], "outcome": verified["outcomes"]["F1"],
            "checkpoint_sha256": request.get("checkpoints", {}).get("F1"),
            "approval_consumed": True, "usable": verified["usable"],
            "receipt_sha256": PROBE.file_sha(stage / "approval.ledger.jsonl"),
            "authorization_sha256": PROBE.file_sha(stage / "authorization.json"),
        }
    return {"support": support, "missing_groups": missing,
            "status": "complete" if not missing else "inconclusive"}


def selection_results(protocol: Path, protocol_sha256_lf: str, source_anchor: str,
                      input_locations: Path, *, root: Path = ROOT,
                      verify_raw_rgb: bool = True) -> dict[str, Any]:
    """전송 selection의 prepare·inspect·project_results를 같은 입력에서 수행한다."""
    prepared = SELECTION.prepare(protocol, protocol_sha256_lf, source_anchor, input_locations,
                                 root=root, verify_raw_rgb=verify_raw_rgb)
    inspection = SELECTION.inspect(protocol, protocol_sha256_lf, source_anchor, input_locations,
                                   root=root, verify_raw_rgb=verify_raw_rgb)
    repository, stage, manifest = CALL._plan(prepared["plan"])
    context = {"repository": repository, "stage": stage, "manifest": manifest,
               "prepared": prepared, "protocol_path": Path(protocol).resolve(strict=True),
               "protocol_sha256_lf": protocol_sha256_lf, "source_anchor": source_anchor,
               "input_locations": Path(input_locations).resolve(strict=True)}
    rows = EXECUTION.project_results(context)
    usage = EXECUTION._usage(context)
    statuses = Counter(row["model_status"] for row in rows)
    return {"prepared": prepared, "inspection": inspection, "rows": rows, "api_usage": usage,
            "validity": {"denominator": 24, "terminal": sum(status != "unknown_pending" for status in (
                row["model_status"] for row in rows)), "valid": statuses["valid"],
                "semantic_invalid": statuses["semantic_invalid"],
                "infrastructure_failed": statuses["infrastructure_failed"],
                "unknown_pending": statuses["unknown_pending"], "valid_rate": statuses["valid"] / 24}}


def summarize(protocol: Path, protocol_sha256_lf: str, source_anchor: str, input_locations: Path,
              archive: Path, probe_protocol: Path, *, root: Path = ROOT,
              verify_raw_rgb: bool = True) -> dict[str, Any]:
    required = {"selection_protocol": protocol, "input_locations": input_locations,
                "source_archive": archive, "probe_protocol": probe_protocol}
    missing = [name for name, path in required.items() if not Path(path).exists()]
    if missing:
        return {"schema": SCHEMA, "status": "HOLD", "reason": "inputs_insufficient",
                "missing": missing, "selection": None, "source_labels": None,
                "probe_support": None, "comparison_summary": None}
    selected = selection_results(protocol, protocol_sha256_lf, source_anchor, input_locations,
                                 root=root, verify_raw_rgb=verify_raw_rgb)
    input_rows = selected["prepared"]["protocol"]["rows"]
    probe_protocol_body = PROBE.validate_protocol(probe_protocol)
    summary_ref = probe_protocol_body["source_training"]["summary"]
    summary_path = root / summary_ref["path"]
    if not summary_path.is_file() or PROBE.file_sha(summary_path) != summary_ref["sha256"]:
        raise ValueError("probe protocol의 source summary 결합이 다르다")
    labels = source_labels(archive, read(summary_path), input_rows)
    probes = probe_support(root, probe_protocol, archive)
    method_summary = METHOD.summarize(
        input_rows, selected["rows"],
        method_protocol_sha256=input_rows[0]["method_protocol_sha256"],
        source_protocol_sha256=input_rows[0]["source_protocol_sha256"],
        source_labels=labels["labels"], probe_support=probes["support"],
    )
    complete = (selected["inspection"].get("status") == "complete"
                and method_summary["execution_status"] == "complete" and probes["status"] == "complete")
    return {
        "schema": SCHEMA, "status": "complete" if complete else "HOLD",
        "reason": None if complete else "evidence_inconclusive",
        "selection": {"inspection": selected["inspection"], "results": selected["rows"],
                      "api_usage": selected["api_usage"], "validity": selected["validity"],
                      "view_counts": method_summary["view_counts"]},
        "source_labels": labels, "probe_support": probes,
        "comparison_summary": method_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--input-locations", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--probe-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-raw-rgb", action="store_true")
    args = parser.parse_args(argv)
    result = summarize(args.protocol, args.protocol_sha256_lf, args.source_anchor,
                       args.input_locations, args.archive, args.probe_protocol,
                       verify_raw_rgb=not args.skip_raw_rgb)
    if args.output:
        PROBE.write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
