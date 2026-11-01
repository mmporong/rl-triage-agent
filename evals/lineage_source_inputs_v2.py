"""v2 lineage source 20개를 진단 비교용 CPU 입력으로 봉인한다.

학습·GPU·모델 호출·승인·실행 권한을 만들지 않는다. 공개 packet은 등록된
telemetry scalar와 결측 runtime fact만 포함하고, 결함·평가·경로 정보는 private
metadata에만 둔다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_evaluation_gate_v1 as GATE
import lineage_selection_inputs_v1 as SELECT
import lineage_source_batch_v2 as BATCH
import optimizer_runtime_facts_v1 as RUNTIME

SCHEMA = "lineage_source_inputs_v2"
METADATA_SCHEMA = "lineage_source_inputs_metadata_v2"
PACKET_SCHEMA = "lineage_source_case_packet_v2"
COLLECTED = "collected.json"
METADATA = "metadata.json"
FIXED_BANDS = ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PRIVATE_TOKENS = {
    "fault", "arm", "seed", "label", "fixed_verdict", "effective_fault_metadata",
    "source_failure_absent", "not_applicable", "baseline_invalid", "out_of_support",
    "F1", "F2", "F5", "COMPOUND",
}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _owned_ok(value: object) -> bool:
    return (isinstance(value, dict) and value.get("exit_code") == 0
            and value.get("natural_exit") is True and value.get("cleanup_verified") is True
            and value.get("job_assigned_before_release") is True
            and value.get("timed_out") is False and value.get("error_type") is None)


def _validate_report(report: dict[str, Any], checkpoint_sha: str) -> dict[str, Any]:
    name = report.get("name")
    _require(isinstance(name, str) and bool(name), "fixed26 report name이 없다")
    return GATE._validate_report(report, {"job_id": name, "run_name": name}, checkpoint_sha)


def _materialize_telemetry(stage: Path, evidence: dict[str, Any], out: Path,
                           isaac_python: Path) -> dict[str, Any]:
    log_dir = Path(evidence["log_dir"]).resolve()
    event = SELECT._event(log_dir)
    event_sha = _sha(event)
    out.mkdir(parents=True, exist_ok=False)
    event_copy = out / "source.events.out.tfevents.raw"
    shutil.copyfile(event, event_copy)
    _require(_sha(event_copy) == event_sha, "복사한 TensorBoard event SHA가 다르다")
    telemetry = out / "telemetry.json"
    argv = [str(isaac_python), str(SELECT.EXTRACTOR), str(log_dir), str(telemetry)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    completed = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="strict", check=False,
        env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=SELECT.EXTRACT_TIMEOUT_S)
    (out / "extract.stdout.txt").write_text(completed.stdout, encoding="utf-8", newline="")
    (out / "extract.stderr.txt").write_text(completed.stderr, encoding="utf-8", newline="")
    receipt = {
        "schema": "lineage_source_inputs_extract_receipt_v2", "cpu_only": True,
        "timeout_s": SELECT.EXTRACT_TIMEOUT_S, "returncode": completed.returncode,
        "event_sha256_before": event_sha, "event_sha256_after": _sha(event),
        "extractor_sha256_lf": _sha(SELECT.EXTRACTOR, lf=True),
    }
    if completed.returncode != 0 or not telemetry.is_file():
        _write_new(out / "extract.receipt.json", receipt)
        raise RuntimeError("TensorBoard telemetry 추출이 성공하지 않았다")
    _require(receipt["event_sha256_after"] == event_sha, "추출 중 source event가 바뀌었다")
    telemetry_value = _read(telemetry)
    SELECT._telemetry_projection("case", telemetry_value)
    receipt["telemetry_sha256"] = _sha(telemetry)
    _write_new(out / "extract.receipt.json", receipt)
    return {
        "stage": str(stage.resolve()), "event": _ref(event), "event_copy": _ref(event_copy),
        "telemetry": _ref(telemetry), "receipt": _ref(out / "extract.receipt.json"),
    }


def _effective_metadata(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    arm = manifest["arm"]
    ppo = evidence.get("ppo_readback")
    expected_effective = {"learning_epochs": 1 if arm == "COMPOUND" else 5,
                          "mini_batches": 4}
    _require(isinstance(ppo, dict)
             and ppo.get("configured") == {"learning_epochs": 5, "mini_batches": 4}
             and ppo.get("effective") == expected_effective,
             "source PPO configured/effective readback이 등록 arm과 다르다")
    result: dict[str, Any] = {
        "registered_arm": arm, "application_readback_available": False,
        "applicable": None, "source": "registered_injection_plus_training_evidence",
        "ppo_configured_effective_readback": deepcopy(ppo),
        "optimizer_step_counts_derived": False,
    }
    if arm == "NONE":
        result.update(application_readback_available=True, applicable=True,
                      mechanism="no_fault_normal_source")
    elif arm == "F5":
        result.update(mechanism="physics_low_friction_static_dynamic_0.25x",
                      repair_support="unsupported", termination_fault=False)
    elif arm == "COMPOUND":
        patch = evidence.get("patch_readback")
        result.update(mechanism="reward_plus_optimizer_epoch_reduction",
                      optimizer_patch_readback=deepcopy(patch),
                      application_readback_available=isinstance(patch, dict),
                      applicable=True if isinstance(patch, dict) and patch.get("r1_applied") is True else None)
    else:
        result["mechanism"] = {"F1": "reward_frame", "F2": "actuator_torque_speed"}.get(
            arm, "registered_source_fault")
    explicit = evidence.get("effective_fault_metadata")
    if isinstance(explicit, dict):
        result.update(deepcopy(explicit))
        result["source"] = "training_evidence_effective_fault_metadata"
    return result


def _classify(case: dict[str, Any], source_report: dict[str, Any],
              reference_report: dict[str, Any], bands: dict[str, Any],
              effective: dict[str, Any], source_cp: str, reference_cp: str) -> dict[str, Any]:
    arm = case["arm"]
    baseline_walks = GATE.RESIDUAL.D1.V.R1.U.walks(reference_report)
    verdict = None
    if source_cp == reference_cp:
        label, reason = "duplicate_checkpoint", "case_and_reference_checkpoint_bytes_equal"
    elif not baseline_walks:
        label, reason = "baseline_invalid", "independent_reference_does_not_walk"
    elif effective.get("applicable") is False:
        label, reason = "not_applicable", "effective_fault_metadata_reports_not_applicable"
    else:
        verdict = GATE.RESIDUAL.recovery(source_report, reference_report, bands)["verdict"]
        if arm == "NONE":
            if verdict["label"] == "healthy":
                label, reason = "eligible", "independent_normal_case_and_reference_healthy"
            else:
                label, reason = "baseline_invalid", "normal_case_not_healthy_against_independent_reference"
        elif verdict["label"] == "healthy":
            label, reason = "source_failure_absent", "fixed26_failure_not_present"
        elif verdict["label"] == "undetermined":
            label, reason = "measurement_unknown", "fixed26_measurement_undetermined"
        elif verdict["label"] != "unhealthy":
            raise ValueError("등록되지 않은 fixed26 verdict label")
        elif arm == "F5":
            label, reason = "out_of_support", "physics_low_friction_failure_repair_unsupported"
        else:
            label, reason = "eligible", "registered_failure_present"
    diagnosis_eligible = label in {"eligible", "out_of_support"} and arm != "NONE"
    return {
        "case_id": case["case_id"], "arm": arm, "seed": case["seed"],
        "source_job": case["source_job"], "reference_job": case["reference_job"],
        "reference_relation": case["reference_relation"], "label": label, "reason": reason,
        "baseline_walks": baseline_walks, "fixed_verdict": verdict,
        "diagnosis_eligible": diagnosis_eligible,
        "correction_support": "unsupported" if arm == "F5" else "registered" if arm != "NONE" else "not_applicable",
        "compound_confirmation": ({"partial": None, "full": None,
                                    "status": "not_measured_source_stage"}
                                   if arm == "COMPOUND" else None),
        "actual_3_update_check": "unavailable_requires_separate_gpu_run",
    }


def validate_batch(source_root: Path, protocol_path: Path, *, lab: Path,
                   anchor: str) -> dict[str, Any]:
    """완료 v2 batch의 20 source·fixed26·원문 결합을 기존 검증기로 재소비한다."""
    source_root, protocol_path = Path(source_root).resolve(), Path(protocol_path).resolve()
    lab = Path(lab).resolve()
    _require(HEX40.fullmatch(anchor) is not None, "anchor 형식이 잘못됐다")
    protocol = BATCH.verify(protocol_path, lab, anchor)
    result, intent = _read(source_root / "result.json"), _read(source_root / "intent.json")
    jobs = {row["job_id"]: row for row in protocol["training_jobs"]}
    records = result.get("records")
    _require(result.get("schema") == BATCH.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("mode") == "all" and result.get("source_jobs_completed") == 20
             and result.get("fixed26_completed") == 20 and isinstance(records, list)
             and len(records) == 20 and result.get("runtime_facts_fabricated") is False
             and result.get("model_api") == 0 and result.get("repairs") == 0,
             "v2 batch 완료 20 source·20 fixed26 receipt가 아니다")
    _require(intent.get("anchor") == anchor
             and intent.get("protocol_sha256_lf") == _sha(protocol_path, lf=True),
             "batch intent의 anchor·protocol 결합이 다르다")
    by_job = {row.get("job_id"): row for row in cast(list[Any], records) if isinstance(row, dict)}
    _require(len(by_job) == 20 and set(by_job) == set(jobs), "batch record job 집합이 다르다")
    bands = GATE._bands(ROOT, FIXED_BANDS, protocol)
    sources: dict[str, dict[str, Any]] = {}
    for job_id, job in jobs.items():
        stage = source_root / job_id
        manifest = BATCH.CHILD.validate_manifest(_read(stage / "manifest.json"))
        expected_manifest = BATCH.manifest(protocol_path, protocol, job, anchor, purpose="source")
        _require(manifest == expected_manifest, f"{job_id} manifest가 등록 job과 다르다")
        owned = _read(stage / "owned-result.json")
        evidence = BATCH.completed(stage, manifest, owned)
        row = by_job[job_id]
        fixed_path = stage / "fixed-result.json"
        fixed = _read(fixed_path)
        report, fixed_execution = fixed.get("report"), fixed.get("execution")
        _require(isinstance(report, dict) and _owned_ok(fixed_execution),
                 f"{job_id} fixed26 owned 완료가 아니다")
        checkpoint = evidence["checkpoint"]
        checked_report = _validate_report(cast(dict[str, Any], report), checkpoint["sha256"])
        _require(row.get("purpose") == "source" and row.get("arm") == job["arm"]
                 and row.get("seed") == job["seed"] and row.get("role") == job["role"]
                 and row.get("checkpoint") == checkpoint and row.get("runtime_facts") is None
                 and row.get("evidence_sha256") == _sha(stage / "training" / BATCH.GUARD.EVIDENCE)
                 and row.get("records_sha256") == _sha(stage / "training" / BATCH.GUARD.RECORDS)
                 and row.get("fixed26", {}).get("conditions") == 26
                 and row.get("fixed26", {}).get("execution") == fixed_execution
                 and row.get("fixed26", {}).get("result_sha256") == _sha(fixed_path),
                 f"{job_id} batch row가 학습·fixed26 원문과 다르다")
        sources[job_id] = {
            "stage": stage, "manifest": manifest, "evidence": evidence,
            "checkpoint": checkpoint, "report": checked_report,
            "effective_fault_metadata": _effective_metadata(manifest, evidence),
        }
    cases = []
    for case in protocol["cases"]:
        source, reference = sources[case["source_job"]], sources[case["reference_job"]]
        cases.append(_classify(
            case, source["report"], reference["report"], bands,
            source["effective_fault_metadata"], source["checkpoint"]["sha256"],
            reference["checkpoint"]["sha256"]))
    checkpoint_groups: dict[str, list[str]] = {}
    for row in cases:
        checkpoint_groups.setdefault(sources[row["source_job"]]["checkpoint"]["sha256"], []).append(
            row["case_id"])
    duplicate_groups = [
        {"checkpoint_sha256": digest, "case_ids": ids}
        for digest, ids in checkpoint_groups.items() if len(ids) > 1
    ]
    duplicate_ids = {case_id for group in duplicate_groups for case_id in group["case_ids"][1:]}
    for row in cases:
        row["duplicate_across_cases"] = row["case_id"] in duplicate_ids
        row["eligible_after_dedup"] = row["label"] in {"eligible", "out_of_support"} and not row[
            "duplicate_across_cases"]
        row["exclusion_reason"] = (
            "duplicate_checkpoint_across_cases" if row["duplicate_across_cases"]
            else None if row["eligible_after_dedup"] else row["label"])
    return {"protocol": protocol, "result": result, "sources": sources, "cases": cases,
            "duplicate_checkpoint_groups": duplicate_groups}


def collect_batch(source_root: Path, protocol_path: Path, *, lab: Path, anchor: str,
                  isaac_python: Path, out: Path) -> dict[str, Any]:
    """검증된 v2 batch를 20 telemetry와 10 case index로 write-once 봉인한다."""
    out, isaac_python = Path(out).resolve(), Path(isaac_python).resolve()
    _require(not out.exists(), "진단 입력 out은 fresh 경로여야 한다")
    _require(isaac_python.is_file(), "Isaac Python 실행 파일이 없다")
    context = validate_batch(source_root, protocol_path, lab=lab, anchor=anchor)
    out.mkdir(parents=True, exist_ok=False)
    source_metadata: dict[str, dict[str, Any]] = {}
    public_sources: dict[str, dict[str, str]] = {}
    ordered_jobs = [row["job_id"] for row in context["protocol"]["training_jobs"]]
    for index, job_id in enumerate(ordered_jobs, start=1):
        source = context["sources"][job_id]
        public_id = f"source-{index:02d}"
        material = _materialize_telemetry(source["stage"], source["evidence"],
                                          out / "sources" / job_id, isaac_python)
        binding = {
            "checkpoint_sha256": source["checkpoint"]["sha256"],
            "telemetry_sha256": material["telemetry"]["sha256"],
            "manifest_sha256": _sha(source["stage"] / "manifest.json"),
            "evidence_sha256": _sha(source["stage"] / "training" / BATCH.GUARD.EVIDENCE),
            "records_sha256": _sha(source["stage"] / "training" / BATCH.GUARD.RECORDS),
            "fixed_result_sha256": _sha(source["stage"] / "fixed-result.json"),
        }
        public_sources[public_id] = deepcopy(binding)
        source_metadata[job_id] = {
            "public_id": public_id, "binding": binding, "material": material,
            "manifest": deepcopy(source["manifest"]),
            "effective_fault_metadata": deepcopy(source["effective_fault_metadata"]),
        }
    metadata = {
        "schema": METADATA_SCHEMA, "fixture": False,
        "consumer_sha256_lf": _sha(Path(__file__), lf=True),
        "protocol": _ref(protocol_path), "source_result": _ref(Path(source_root) / "result.json"),
        "anchor": anchor, "sources": source_metadata, "cases": context["cases"],
        "duplicate_checkpoint_groups": context["duplicate_checkpoint_groups"],
        "runtime_observation": {
            "actual_3_update_check": "unavailable_requires_separate_gpu_run",
            "reason": "v2 source child did not collect optimizer observer records",
            "fabricated_counts": False,
        },
    }
    _write_new(out / METADATA, metadata)
    public = {
        "schema": SCHEMA, "fixture": False, "planned_cases": 10, "source_count": 20,
        "sources": public_sources, "case_ids": [row["case_id"] for row in context["cases"]],
        "metadata_ref": {"file": METADATA, "sha256": _sha(out / METADATA)},
        "scope": {"activity": "cpu_diagnosis_input_materialization_only", "approval": False,
                  "execution_allowed": False, "fault_label_exposed": False,
                  "evaluation_truth_exposed": False, "physical_path_exposed": False},
    }
    _write_new(out / COLLECTED, public)
    return deepcopy(public)


def checked(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """봉인된 index와 20 telemetry·원문 SHA를 다시 확인한다."""
    path = Path(path).resolve()
    public = _read(path)
    _require(public.get("schema") == SCHEMA and public.get("fixture") is False
             and public.get("planned_cases") == 10 and public.get("source_count") == 20,
             "v2 collected 공개 shape가 다르다")
    metadata_path = path.parent / METADATA
    _require(public.get("metadata_ref") == {"file": METADATA, "sha256": _sha(metadata_path)},
             "private metadata ref가 다르다")
    metadata = _read(metadata_path)
    _require(metadata.get("schema") == METADATA_SCHEMA and metadata.get("fixture") is False,
             "private metadata schema가 다르다")
    _require(metadata.get("consumer_sha256_lf") == _sha(Path(__file__), lf=True),
             "consumer source bytes가 수집 시점과 다르다")
    sources = metadata.get("sources")
    _require(isinstance(sources, dict) and len(sources) == 20
             and len(public.get("sources", {})) == 20, "source index 집합이 다르다")
    sources = cast(dict[str, dict[str, Any]], sources)
    for job_id, private in sources.items():
        binding = private["binding"]
        public_id = private.get("public_id")
        _require(isinstance(public_id, str) and public.get("sources", {}).get(public_id) == binding,
                 f"{job_id} 공개 binding이 다르다")
        material = private["material"]
        for name in ("event", "event_copy", "telemetry", "receipt"):
            ref = material[name]
            _require(_sha(Path(ref["path"])) == ref["sha256"], f"{job_id} {name} bytes가 바뀌었다")
        _require(material["event"]["sha256"] == material["event_copy"]["sha256"]
                 and material["telemetry"]["sha256"] == binding["telemetry_sha256"],
                 f"{job_id} event/telemetry 결합이 다르다")
        telemetry = _read(Path(material["telemetry"]["path"]))
        SELECT._telemetry_projection("case", telemetry)
    _require(metadata.get("runtime_observation", {}).get("fabricated_counts") is False,
             "runtime 결측값을 추정치로 바꿀 수 없다")
    return deepcopy(public), deepcopy(metadata)


def _model_safe(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _require(not re.search(r"(?:[A-Za-z]:[\\/]|/(?:home|mnt|Users)/)", encoded),
             "모델 packet에 물리 경로가 노출됐다")
    lowered = encoded.lower()
    _require(not any(re.search(rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])", lowered)
                     for token in PRIVATE_TOKENS), "모델 packet에 결함·평가 정답이 노출됐다")


def make_case_packet(index_path: Path, case_id: str, *, budget_left: int = 7,
                     execution_budget_s: float = 1200) -> dict[str, Any]:
    """한 case/reference의 48 telemetry scalar와 결측 runtime facts를 만든다."""
    public, metadata = checked(index_path)
    cases = [row for row in metadata["cases"] if row.get("case_id") == case_id]
    _require(len(cases) == 1, "case_id는 계획 10건 중 하나여야 한다")
    case = cases[0]
    source_private = metadata["sources"][case["source_job"]]
    reference_private = metadata["sources"][case["reference_job"]]
    source_telemetry = _read(Path(source_private["material"]["telemetry"]["path"]))
    reference_telemetry = _read(Path(reference_private["material"]["telemetry"]["path"]))
    source_facts, source_coverage = SELECT._telemetry_projection("case", source_telemetry)
    reference_facts, reference_coverage = SELECT._telemetry_projection("reference", reference_telemetry)
    facts = {**source_facts, **reference_facts,
             **RUNTIME.unavailable_facts("case"), **RUNTIME.unavailable_facts("reference")}
    _require(len(source_facts) == len(reference_facts) == 24, "공개 telemetry는 role별 24 scalar여야 한다")
    packet = SELECT.V.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None, facts=facts,
        input_bindings={
            "case": {"checkpoint_sha256": source_private["binding"]["checkpoint_sha256"],
                     "telemetry_sha256": source_private["binding"]["telemetry_sha256"]},
            "reference": {"checkpoint_sha256": reference_private["binding"]["checkpoint_sha256"],
                          "telemetry_sha256": reference_private["binding"]["telemetry_sha256"]},
        }, budget_left=budget_left, execution_budget_s=execution_budget_s)
    _model_safe(packet)
    return {
        "schema": PACKET_SCHEMA, "case_id": case_id, "packet": packet,
        "source_binding": {
            "case": deepcopy(public["sources"][source_private["public_id"]]),
            "reference": deepcopy(public["sources"][reference_private["public_id"]])},
        "private_metadata": {
            "index": _ref(index_path), "classification": deepcopy(case),
            "coverage": {"case": source_coverage, "reference": reference_coverage},
            "actual_3_update_check": "unavailable_requires_separate_gpu_run",
        },
        "scope": {"approval": False, "execution_allowed": False,
                  "fault_label_exposed_to_model": False,
                  "evaluation_truth_exposed_to_model": False,
                  "physical_path_exposed_to_model": False,
                  "telemetry_scalar_facts": 48, "runtime_counts_fabricated": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    for name in ("source-root", "protocol", "lab", "isaac-python", "out"):
        collect.add_argument(f"--{name}", type=Path, required=True)
    collect.add_argument("--anchor", required=True)
    check = sub.add_parser("check")
    check.add_argument("--index", type=Path, required=True)
    packet = sub.add_parser("packet")
    packet.add_argument("--index", type=Path, required=True)
    packet.add_argument("--case-id", required=True)
    packet.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect_batch(args.source_root, args.protocol, lab=args.lab, anchor=args.anchor,
                      isaac_python=args.isaac_python, out=args.out)
    elif args.command == "check":
        checked(args.index)
    else:
        _write_new(args.out, make_case_packet(args.index, args.case_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
