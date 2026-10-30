"""18개 source 고정 평가를 같은-seed 정상 참조 10사례 gate로 합친다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import residual_demo_v2 as RESIDUAL

SCHEMA = "lineage_evaluation_gate_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
FIXED_PROTOCOL = "bench/protocols/fixed_eval_v1.json"
FIXED_PROTOCOL_SHA256_LF = "8d24652eeed47f462d78244fceec785164b7695e5b757ae8a4e9bd7947aa8a46"
BANDS_SHA256 = "7589a132d4e1a55e159527389a8b635a1766472ecde558da0a1a704c286f3f23"
EXPECTED_COMMANDS = {
    (float(vx), float(vy), float(yaw))
    for vx in (-1, 0, 1) for vy in (-1, 0, 1) for yaw in (-1, 0, 1)
    if (vx, vy, yaw) != (0, 0, 0)
}
SUPPORTED = {"F1", "F2", "COMPOUND"}
OUT_OF_SUPPORT = {"F3", "F5"}


def _read(value: Any, name: str) -> dict:
    if isinstance(value, (str, os.PathLike)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name}은 JSON 객체 또는 경로여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _report_path(value: Any, repo: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("report_path가 비었다")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _bands(repo: Path, bands_path: Path, protocol: Mapping[str, Any]) -> dict:
    fixed = repo / FIXED_PROTOCOL
    if (not fixed.is_file() or _sha(fixed, lf=True) != FIXED_PROTOCOL_SHA256_LF
            or protocol.get("source_sha256_lf", {}).get(FIXED_PROTOCOL) != FIXED_PROTOCOL_SHA256_LF):
        raise ValueError("고정 26조건 protocol SHA 결합이 다르다")
    if not bands_path.is_file() or _sha(bands_path) != BANDS_SHA256:
        raise ValueError("원래 고정 평가 bands SHA가 다르다")
    value = _read(bands_path, "bands")
    bands = value.get("bands")
    expected = {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}
    if not isinstance(bands, dict) or set(bands) != expected:
        raise ValueError("원래 bands 필드가 다르다")
    return bands


def _validate_protocol(protocol: Mapping[str, Any]) -> tuple[list[dict], dict[str, dict]]:
    cases, jobs = protocol.get("cases"), protocol.get("training_jobs")
    if (protocol.get("schema") != "lineage_source_collection_protocol_v1"
            or not isinstance(cases, list) or len(cases) != 10
            or not isinstance(jobs, list) or len(jobs) != 18
            or protocol.get("next_stage", {}).get("registered") is not False):
        raise ValueError("source protocol의 10사례·18학습 계약이 다르다")
    by_job = {row.get("job_id"): row for row in jobs if isinstance(row, dict)}
    if len(by_job) != 18 or None in by_job:
        raise ValueError("source protocol job_id가 중복되거나 비었다")
    for case in cases:
        if (not isinstance(case, dict) or case.get("source_job") not in by_job
                or case.get("reference_job") not in by_job):
            raise ValueError("case의 source/reference job이 등록되지 않았다")
        source, reference = by_job[case["source_job"]], by_job[case["reference_job"]]
        if (source.get("case_id") != case.get("case_id") or reference.get("case_id") != case.get("case_id")
                or source.get("seed") != case.get("seed") or reference.get("seed") != case.get("seed")
                or source.get("arm") != case.get("arm") or reference.get("arm") != "NONE"):
            raise ValueError("같은-seed 정상 참조 또는 case/job 결합이 다르다")
        if case["arm"] == "NONE" and case["source_job"] != case["reference_job"]:
            raise ValueError("NONE case는 자기 참조여야 한다")
    return cases, by_job


def _source_records(source_result: Mapping[str, Any], jobs: Mapping[str, dict]) -> dict[str, dict]:
    records = source_result.get("records")
    if (source_result.get("status") != "complete" or source_result.get("source_jobs_completed") != 18
            or not isinstance(records, list)):
        raise ValueError("source batch 완료 receipt가 아니다")
    selected = [row for row in records if isinstance(row, dict) and row.get("purpose") == "source"]
    by_job = {row.get("job_id"): row for row in selected}
    if len(selected) != 18 or set(by_job) != set(jobs):
        raise ValueError("source 18개 receipt가 모두 있어야 한다")
    for job_id, row in by_job.items():
        job, checkpoint = jobs[job_id], row.get("checkpoint")
        if (row.get("arm") != job["arm"] or row.get("seed") != job["seed"]
                or not isinstance(checkpoint, dict) or not _hex(checkpoint.get("sha256"))):
            raise ValueError("source receipt의 arm·seed·checkpoint가 등록 job과 다르다")
    return by_job


def _validate_report(report: Mapping[str, Any], job: Mapping[str, Any], checkpoint_sha: str) -> dict:
    if (report.get("task") != TASK or report.get("eval_seed") != 2026
            or report.get("protocol") != "fixed_eval_v1"
            or report.get("protocol_sha256_lf") != FIXED_PROTOCOL_SHA256_LF
            or report.get("num_envs") != 1040 or report.get("horizon_steps") != 1000
            or report.get("step_dt") != 0.02 or report.get("action_scale") != 0.25
            or report.get("name") not in {job["job_id"], job["run_name"]}
            or report.get("checkpoint", {}).get("sha256") != checkpoint_sha
            or not isinstance(report.get("versions"), dict) or not report["versions"]):
        raise ValueError("고정 평가 task·seed·checkpoint·protocol binding이 다르다")
    rows = report.get("by_condition")
    if not isinstance(rows, list) or len(rows) != 26:
        raise ValueError("고정 평가 26개 조건이 모두 있어야 한다")
    commands = []
    for row in rows:
        command = row.get("command") if isinstance(row, dict) else None
        if (not isinstance(command, list) or len(command) != 3 or row.get("num_envs") != 40
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in command)):
            raise ValueError("고정 평가 개별 condition 행이 잘못됐다")
        commands.append(tuple(float(value) for value in command))
    if len(set(commands)) != 26 or set(commands) != EXPECTED_COMMANDS:
        raise ValueError("고정 평가 condition 행이 누락·중복·교체됐다")
    for field in ("metrics", "standstill"):
        if not isinstance(report.get(field), dict):
            raise TypeError("고정 평가 behavior 필드가 없다")
    return dict(report)


def _reports(index: Mapping[str, Any], records: Mapping[str, dict], jobs: Mapping[str, dict],
             repo: Path) -> dict[str, dict]:
    if set(index) != set(jobs):
        raise ValueError("평가 index는 source 18개를 명시 등록해야 한다")
    result = {}
    for job_id, job in jobs.items():
        entry = index[job_id]
        if not isinstance(entry, dict) or set(entry) != {"report_path", "report_sha256", "checkpoint_sha256"}:
            raise ValueError("평가 index entry 필드가 다르다")
        checkpoint = records[job_id]["checkpoint"]["sha256"]
        if entry["checkpoint_sha256"] != checkpoint or not _hex(entry.get("report_sha256")):
            raise ValueError("평가 index checkpoint/report SHA 형식이 다르다")
        path = _report_path(entry["report_path"], repo)
        if not path.is_file() or _sha(path) != entry["report_sha256"]:
            raise ValueError("평가 report 파일 SHA가 index와 다르다")
        result[job_id] = _validate_report(_read(path, "report"), job, checkpoint)
    return result


def _classify(case: Mapping[str, Any], source: dict, reference: dict, bands: dict) -> dict:
    arm = case["arm"]
    baseline_walks = RESIDUAL.D1.V.R1.U.walks(reference)
    if not baseline_walks:
        label, reason, intervention = "baseline_invalid", "same_seed_reference_does_not_walk", 0
        verdict = None
    elif arm == "NONE":
        label, reason, intervention = "eligible", "self_reference_normal_control", 0
        verdict = RESIDUAL.recovery(source, reference, bands)["verdict"]
        if verdict["label"] != "healthy":
            label, reason = "baseline_invalid", "none_self_reference_not_healthy"
    else:
        verdict = RESIDUAL.recovery(source, reference, bands)["verdict"]
        if verdict["label"] == "healthy":
            label, reason, intervention = "source_failure_absent", "fixed_failure_not_present", 0
        elif verdict["label"] == "undetermined":
            label, reason, intervention = "measurement_unknown", "fixed_measurement_undetermined", 0
        elif verdict["label"] != "unhealthy":
            raise ValueError("등록되지 않은 fixed verdict label")
        elif arm in SUPPORTED:
            label, reason, intervention = "eligible", "registered_failure_present", 1
        elif arm in OUT_OF_SUPPORT:
            label, reason, intervention = "out_of_support", "failure_present_but_no_registered_correction", 0
        else:
            raise ValueError("등록되지 않은 source arm")
    return {
        "case_id": case["case_id"], "seed": case["seed"], "arm": arm,
        "source_job": case["source_job"], "reference_job": case["reference_job"],
        "label": label, "reason": reason, "baseline_walks": baseline_walks,
        "fixed_verdict": verdict, "expected_correction_interventions": intervention,
        "diagnosis_eligible": label in {"eligible", "out_of_support"} and arm != "NONE",
        "diagnosis_claimed": False,
        "expected_action": "HOLD" if label in {"out_of_support", "measurement_unknown"}
        else "no_intervention" if intervention == 0 else "eligible_for_registered_loop",
    }


def consume_gate(source_result, protocol, report_index, *, repo, bands_path, out=None) -> dict:
    """명시 index의 18 report를 검증하고 같은-seed 10 case gate를 반환한다."""
    repo, bands_path = Path(repo).resolve(), Path(bands_path).resolve()
    protocol_value = _read(protocol, "protocol")
    cases, jobs = _validate_protocol(protocol_value)
    bands = _bands(repo, bands_path, protocol_value)
    records = _source_records(_read(source_result, "source_result"), jobs)
    reports = _reports(_read(report_index, "report_index"), records, jobs, repo)
    rows = [_classify(case, reports[case["source_job"]], reports[case["reference_job"]], bands)
            for case in cases]
    source_checkpoints = {case["case_id"]: records[case["source_job"]]["checkpoint"]["sha256"]
                          for case in cases}
    grouped: dict[str, list[str]] = {}
    for case_id, checkpoint in source_checkpoints.items():
        grouped.setdefault(checkpoint, []).append(case_id)
    duplicates = [{"checkpoint_sha256": sha, "case_ids": ids, "dedup_required": True}
                  for sha, ids in grouped.items() if len(ids) > 1]
    duplicate_excess = sum(len(row["case_ids"]) - 1 for row in duplicates)
    excluded_labels = {"baseline_invalid", "source_failure_absent", "measurement_unknown"}
    valid_labels = {"eligible", "out_of_support"}
    excluded = [row for row in rows if row["label"] in excluded_labels]
    valid_checkpoint_count = len({source_checkpoints[row["case_id"]]
                                  for row in rows if row["label"] in valid_labels})
    result = {
        "schema": SCHEMA, "status": "complete", "planned_cases": 10,
        "valid_pairs": sum(row["label"] in valid_labels for row in rows),
        "independent_valid_pairs": valid_checkpoint_count,
        "excluded_total": len(excluded), "excluded": excluded, "cases": rows,
        "label_counts": {label: sum(row["label"] == label for row in rows)
                         for label in ("baseline_invalid", "source_failure_absent", "measurement_unknown",
                                       "eligible", "out_of_support")},
        "duplicate_checkpoint_groups": duplicates,
        "independent_case_count": len(rows) - duplicate_excess,
        "cause_scope": {"source_arm_is_preregistered_metadata": True, "diagnosis_performed": False,
                        "f3_f5_expected_action": "HOLD_out_of_registered_correction_support"},
        "development_coupling": {"scope": "label_development_after_source_collection",
                                 "heldout_claim": False, "generalization_claim": False,
                                 "source_protocol_next_stage_registered": False,
                                 "report_index_is_post_collection_input": True},
        "fixed_evaluation": {"conditions": 26, "protocol_sha256_lf": FIXED_PROTOCOL_SHA256_LF,
                             "bands_sha256": BANDS_SHA256},
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--report-index", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    consume_gate(args.source_result, args.protocol, args.report_index, repo=ROOT,
                 bands_path=args.bands, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
