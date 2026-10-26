"""first-class record readiness를 사용하는 영상 선택 실행·집계 진입점."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "src", ROOT / "evals", ROOT / "bench"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import vision_record_selection_worker_v1 as WORKER  # pyright: ignore[reportMissingImports]
import vision_record_source_readiness_v1 as READINESS  # pyright: ignore[reportMissingImports]
import vision_selection_execution_v1 as BASE  # pyright: ignore[reportMissingImports]

CALL = BASE.CALL
COMPARISON = BASE.COMPARISON
CONTROLLER = BASE.CONTROLLER
GROUNDED = BASE.GROUNDED
JOURNAL = BASE.JOURNAL
RECOVERY = BASE.RECOVERY
REMOTE = BASE.REMOTE
SOURCE = BASE.SOURCE
EVIDENCE_SCHEMA = BASE.EVIDENCE_SCHEMA

_read = BASE._read
_ref = BASE._ref
_finite = BASE._finite
_source_artifact = BASE._source_artifact
_probe_execution_source = BASE._probe_execution_source
_portable_probe = BASE._portable_probe
_owner_present = BASE._owner_present
_usage = BASE._usage
_check_output_pair = BASE._check_output_pair
_write_once = BASE._write_once
inspect_run = BASE.inspect_run
project_results = BASE.project_results


def load_context(
    protocol: Path, protocol_sha_lf: str, source_anchor: str, input_locations: Path
) -> dict[str, Any]:
    """새 worker 준비 계약으로 읽기 전용 실행 문맥을 만든다."""
    prepared = WORKER.prepare(
        protocol, protocol_sha_lf, source_anchor, input_locations
    )
    repository, stage, manifest = CALL._plan(prepared["plan"])
    return {
        "repository": repository,
        "stage": stage,
        "manifest": manifest,
        "prepared": prepared,
        "protocol_path": Path(protocol).resolve(strict=True),
        "protocol_sha256_lf": protocol_sha_lf,
        "source_anchor": source_anchor,
        "input_locations": Path(input_locations).resolve(strict=True),
    }


def remaining_commands(
    context: dict[str, Any], now_utc: datetime | None = None
) -> dict[str, Any]:
    """기존 상태 판정으로 새 worker CLI의 다음 한 명령만 만든다."""
    state = inspect_run(context, now_utc)
    commands: list[list[str]] = []
    if state["status"] == "runnable":
        commands.append(
            [
                "setsid",
                str(Path(sys.executable).resolve()),
                str((ROOT / "evals/vision_record_selection_worker_v1.py").resolve()),
                "advance",
                "--protocol",
                str(context["protocol_path"]),
                "--protocol-sha256-lf",
                context["protocol_sha256_lf"],
                "--source-anchor",
                context["source_anchor"],
                "--input-locations",
                str(context["input_locations"]),
                "--call-id",
                state["next_call_id"],
            ]
        )
    return {**state, "commands": commands}


def _input_artifacts(root: Path, context: dict[str, Any], group: str,
                     input_rows: list[dict], source_protocol: dict) -> dict[str, dict]:
    """동결 input-locations가 가리키는 readiness에서 학습 artifact를 다시 결합한다."""
    locations = _read(Path(context["input_locations"]))
    if (set(locations) != {"schema", "execution_id", "rows"}
            or not isinstance(locations["rows"], list)
            or len(locations["rows"]) != len(input_rows)):
        raise ValueError("input locations 형식·row 수가 다르다")
    by_call = {}
    for public, location in zip(input_rows, locations["rows"], strict=True):
        if (not isinstance(location, dict) or set(location) != {"call_id", "input_manifest", "capsule"}
                or location.get("call_id") != public["call_id"]):
            raise ValueError("input locations 순서·call 결합이 다르다")
        by_call[public["call_id"]] = location
    outputs: dict[str, dict] = {}
    for role, arm in (("normal", "NONE"), ("registered-f1", "F1")):
        rows = [row for row in input_rows if row["seed_group"] == group and row["source_role"] == role]
        if not rows or any(row["checkpoint_sha256"] != rows[0]["checkpoint_sha256"] for row in rows):
            raise ValueError("comparison input checkpoint가 role 안에서 다르다")
        identities = set()
        selected: tuple[dict, str] | None = None
        for row in rows:
            location = by_call[row["call_id"]]
            manifest_path, _ = _ref(root, location["input_manifest"], "input manifest")
            manifest = _read(manifest_path)
            if set(manifest) != {"schema", "snapshot", "contract", "trace", "camera_pose_audit", "model_readiness"}:
                raise ValueError("input manifest 형식이 다르다")
            readiness_path, readiness_sha = _ref(root, manifest["model_readiness"], "model readiness")
            readiness = _read(readiness_path)
            proof = READINESS.verify_model_readiness(root, readiness_path, source_protocol)
            if (proof.get("checkpoint_sha256") != row["checkpoint_sha256"]
                    or proof.get("source_seed") != row["source_seed"]
                    or proof.get("source_role") != role):
                raise ValueError("model readiness 재검증과 comparison input 계보가 다르다")
            artifact_path, artifact_sha = _ref(
                root, readiness.get("source_training_artifact"), "source training artifact",
            )
            artifact = _read(artifact_path)
            snapshot_path, _ = _ref(root, manifest["snapshot"], "selection snapshot")
            snapshot = _read(snapshot_path)
            bindings = snapshot.get("inputs", {}).get("packet", {}).get("input_bindings")
            case = bindings.get("case") if isinstance(bindings, dict) else None
            telemetry_sha = artifact.get("telemetry", {}).get("sha256")
            if (not isinstance(case, dict)
                    or case != {"checkpoint_sha256": row["checkpoint_sha256"],
                                "telemetry_sha256": telemetry_sha}):
                raise ValueError("snapshot case와 readiness 학습 artifact 계보가 다르다")
            _source_artifact(artifact, row, arm, str(telemetry_sha))
            identity = (readiness_sha, artifact_sha, str(telemetry_sha))
            identities.add(identity)
            selected = (artifact, artifact_sha)
        if len(identities) != 1 or selected is None:
            raise ValueError("같은 role 입력이 다른 readiness 학습 artifact를 쓴다")
        outputs[arm] = selected[0]
    return outputs


def _evaluate_evidence(root: Path, context: dict[str, Any], group: str, evaluation: dict,
                       source_protocol: dict, input_rows: list[dict]) -> tuple[dict, dict]:
    fields = {"intent", "raw", "receipt", "resource",
              "gpu_before_observation", "gpu_after_observation"}
    if not isinstance(evaluation, dict) or set(evaluation) != fields:
        raise ValueError("evaluate evidence portable reference가 완전하지 않다")
    intent_path, intent_sha = _ref(root, evaluation["intent"], f"{group} evaluate intent")
    raw_path, raw_sha = _ref(root, evaluation["raw"], f"{group} evaluate raw")
    receipt_path, _ = _ref(root, evaluation["receipt"], f"{group} evaluate receipt")
    resource_path, resource_sha = _ref(root, evaluation["resource"], f"{group} resource")
    before_path, before_sha = _ref(root, evaluation["gpu_before_observation"], f"{group} GPU before")
    after_path, after_sha = _ref(root, evaluation["gpu_after_observation"], f"{group} GPU after")
    intent, raw, receipt, resource = map(_read, (intent_path, raw_path, receipt_path, resource_path))
    seeds = {row["source_seed"] for row in input_rows if row["seed_group"] == group}
    if len(seeds) != 1:
        raise ValueError("comparison input source seed가 group 안에서 유일하지 않다")
    seed = next(iter(seeds))
    actions = [(ordinal, action) for ordinal, action in enumerate(source_protocol["source_action_order"])
               if action.get("kind") == "evaluate" and action.get("seed") == seed]
    if len(actions) != 1:
        raise ValueError("source protocol evaluate action이 유일하지 않다")
    ordinal, action = actions[0]
    limit = source_protocol["resources"]["action_limits"]["evaluate"]
    expected_intent = {
        "schema": SOURCE.INTENT_SCHEMA, "action_id": action["action_id"], "kind": "evaluate",
        "ordinal": ordinal, "protocol_sha256_lf": source_protocol["_verified_sha256_lf"],
        "resource_sha256": resource_sha, "reserved_max_wall_s": limit["max_wall_s"],
        "reserved_max_compute_usd": limit["max_compute_usd"],
        "cost_scope": source_protocol["resources"]["cost_scope"],
    }
    if (set(intent) != {*expected_intent, "cost_before_usd", "started_at_utc"}
            or any(intent.get(key) != value for key, value in expected_intent.items())):
        raise ValueError("evaluate intent와 등록 action/resource 결합이 다르다")
    if (set(receipt) != {"schema", "action_id", "intent_sha256", "raw_sha256",
                         "cost_after_usd", "finished_at_utc"}
            or receipt.get("schema") != SOURCE.RECEIPT_SCHEMA
            or receipt.get("action_id") != action["action_id"]
            or receipt.get("intent_sha256") != intent_sha or receipt.get("raw_sha256") != raw_sha):
        raise ValueError("evaluate receipt SHA 결합이 다르다")
    started, finished = SOURCE._stamp(intent["started_at_utc"]), SOURCE._stamp(receipt["finished_at_utc"])
    compute_started = SOURCE._stamp(resource["compute_started_at_utc"])
    deadline = SOURCE._stamp(source_protocol["resources"]["deadline_utc"])
    quote = _finite(source_protocol["resources"]["compute_quote_usd_per_h"], "compute quote")
    before_cost = (started - compute_started).total_seconds() * quote / 3600
    after_cost = (finished - compute_started).total_seconds() * quote / 3600
    if (not 0 <= (finished - started).total_seconds() <= limit["max_wall_s"] or finished > deadline
            or abs(_finite(intent["cost_before_usd"], "cost before") - before_cost) > 1e-9
            or abs(_finite(receipt["cost_after_usd"], "cost after") - after_cost) > 1e-9
            or not 0 <= before_cost <= after_cost <= source_protocol["resources"]["whole_batch_budget_usd"]):
        raise ValueError("evaluate receipt 시각·비용이 등록 자원과 다르다")
    execution = raw.get("evaluation", {}).get("execution")
    before_ref = execution.get("gpu_before_observation") if isinstance(execution, dict) else None
    after_ref = execution.get("gpu_after_observation") if isinstance(execution, dict) else None
    expected_execution = {
        "schema": "brev_recovery_posix_execution_v1", "mode": "evaluate", "completed": True,
        "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "gpu_before": [], "gpu_after": [],
    }
    if (raw.get("kind") != "evaluate" or raw.get("action_id") != action["action_id"]
            or raw.get("seed") != seed or not isinstance(execution, dict)
            or any(execution.get(key) != value for key, value in expected_execution.items())
            or execution.get("source_before") != execution.get("source_after")
            or not isinstance(execution.get("launch"), dict)
            or _finite(execution.get("wall_s"), "evaluate wall") <= 0
            or not isinstance(before_ref, dict) or before_ref.get("sha256") != before_sha
            or not isinstance(after_ref, dict) or after_ref.get("sha256") != after_sha):
        raise ValueError("evaluate raw 실행 완료·GPU 해제 결합이 다르다")
    resource_id, gpu_uuid = resource.get("resource_id"), resource.get("gpu_uuid")
    if not isinstance(resource_id, str) or not isinstance(gpu_uuid, str):
        raise TypeError("evaluate resource identity가 없다")
    REMOTE.verify_observation_reference({"path": str(before_path), "sha256": before_sha}, resource_id, gpu_uuid)
    REMOTE.verify_observation_reference({"path": str(after_path), "sha256": after_sha}, resource_id, gpu_uuid)
    reports = raw.get("evaluation", {}).get("reports")
    if not isinstance(reports, dict) or set(reports) != {"NONE", "F1"}:
        raise ValueError("evaluate NONE/F1 원본 report가 없다")
    bands_path = (root / source_protocol["evaluation"]["bands_path"]).resolve(strict=True)
    bands_path.relative_to(root.resolve(strict=True))
    bands = _read(bands_path).get("bands")
    if not isinstance(bands, dict) or raw.get("source_gate") != RECOVERY.recovery(
            reports["F1"], reports["NONE"], bands):
        raise ValueError("evaluate source failure gate 재계산이 raw와 다르다")
    artifacts = _input_artifacts(root, context, group, input_rows, source_protocol)
    fixed_path = (root / source_protocol["fixed_eval_protocol"]).resolve(strict=True)
    fixed_path.relative_to(root.resolve(strict=True))
    fixed_sha = SOURCE.sha256(fixed_path, lf=True)
    for arm in ("NONE", "F1"):
        checkpoint = artifacts[arm]["checkpoint"]
        reports[arm] = GROUNDED.validate_fixed_report(reports[arm], {
            "name": arm, "checkpoint_file": Path(checkpoint["path"]).name,
            "checkpoint_sha256": checkpoint["sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": fixed_sha,
        })
    training = {row["job_id"]: row for row in source_protocol["training_plan"]}
    if (training[action["none_job_id"]]["seed"] != seed or training[action["none_job_id"]]["arm"] != "NONE"
            or training[action["f1_job_id"]]["seed"] != seed or training[action["f1_job_id"]]["arm"] != "F1"):
        raise ValueError("evaluate action의 NONE/F1 training 계보가 다르다")
    return reports, artifacts


def _evidence(context: dict[str, Any], path: Path) -> tuple[dict, dict, str]:
    manifest = _read(path)
    expected = {"schema", "protocol_sha256_lf", "probe_execution", "source_groups"}
    if set(manifest) != expected or manifest.get("schema") != EVIDENCE_SCHEMA \
            or manifest.get("protocol_sha256_lf") != context["protocol_sha256_lf"]:
        raise ValueError("evidence locations와 actual protocol 결합이 다르다")
    groups = manifest["source_groups"]
    if not isinstance(groups, dict) or set(groups) != set(COMPARISON.SEED_GROUPS):
        raise ValueError("두 source group evidence가 필요하다")
    labels, probes = {}, {}
    input_rows = context["prepared"]["protocol"]["rows"]
    checkpoints = {(row["seed_group"], row["source_role"]): row["checkpoint_sha256"] for row in input_rows}
    source_protocol_sha = input_rows[0]["source_protocol_sha256"]
    source_ref = context["prepared"]["protocol"]["source_protocol"]
    _, _, source_protocol = BASE.WORKER._protocol_reference(ROOT, source_ref, "source protocol")
    source_protocol["_verified_sha256_lf"] = source_ref["sha256_lf"]
    expected_probe_source = _probe_execution_source(ROOT, manifest["probe_execution"])
    bands = _read((ROOT / source_protocol["evaluation"]["bands_path"]).resolve(strict=True))["bands"]
    for group in COMPARISON.SEED_GROUPS:
        item = groups[group]
        if not isinstance(item, dict) or set(item) != {"evaluation", "probe_evidence"}:
            raise ValueError("source group evidence 필드가 다르다")
        reports, _ = _evaluate_evidence(ROOT, context, group, item["evaluation"], source_protocol, input_rows)
        measured = RECOVERY.recovery(reports["F1"], reports["NONE"], bands)
        normal = RECOVERY.recovery(reports["NONE"], reports["NONE"], bands)
        label = {"recovered": "healthy", "not_recovered": "unhealthy", "unknown": "unknown"}
        labels[group] = {
            "normal": {"checkpoint_sha256": checkpoints[(group, "normal")],
                       "fixed_eval_artifact_sha256": JOURNAL.digest(reports["NONE"]),
                       "label": label[normal["outcome"]]},
            "registered-f1": {"checkpoint_sha256": checkpoints[(group, "registered-f1")],
                              "fixed_eval_artifact_sha256": JOURNAL.digest(reports["F1"]),
                              "label": label[measured["outcome"]]},
        }
        auth_ref = item["probe_evidence"].get("authorization") if isinstance(item["probe_evidence"], dict) else None
        auth_path, _ = _ref(ROOT, auth_ref, f"{group} probe authorization")
        snapshot = _read(auth_path).get("snapshot")
        if not isinstance(snapshot, dict):
            raise TypeError("probe authorization snapshot이 object가 아니다")
        verified, seal_sha = _portable_probe(
            ROOT, item["probe_evidence"], snapshot, checkpoints[(group, "registered-f1")],
            checkpoints[(group, "normal")], expected_probe_source,
        )
        probes[group] = {"checkpoint_sha256": checkpoints[(group, "registered-f1")],
                         "receipt_sha256": seal_sha,
                         "authorization_sha256": item["probe_evidence"]["authorization"]["sha256"],
                         "probe": verified["probe"], "outcome": verified["outcome"],
                         "approval_consumed": True, "usable": verified["outcome"] != "unknown"}
    return labels, probes, source_protocol_sha


def summarize(context: dict[str, Any], evidence_manifest: Path, output_dir: Path,
              finalize: bool = False) -> dict[str, Any]:
    """원본 판단·평가·승인 probe를 재검증해 경로 없는 집계만 만든다."""
    state = inspect_run(context)
    stage = Path(context["stage"])
    decision = stage / "decisions" / state["next_call_id"] if state.get("next_call_id") else None
    if _owner_present(stage, decision):
        raise RuntimeError("owner가 남은 실행은 집계할 수 없다")
    labels, probes, source_protocol_sha = _evidence(context, evidence_manifest)
    choices = project_results(context)
    summary = COMPARISON.summarize(
        context["prepared"]["protocol"]["rows"], choices,
        method_protocol_sha256=context["prepared"]["protocol"]["rows"][0]["method_protocol_sha256"],
        source_protocol_sha256=source_protocol_sha, source_labels=labels, probe_support=probes,
    )
    summary["api_usage"] = _usage(context)
    summary["evidence_manifest_sha256"] = JOURNAL.file_sha(evidence_manifest)
    result = {"status": state["status"], "choice_results": choices, "comparison_summary": summary,
              "finalized": False}
    if not finalize:
        return result
    terminal_boundary = state["counts"]["terminal"] == 24 or state["api"]["remaining"] == 0
    permanent_orphan = state["status"] == "blocked_pending" and state["api"]["pending"] > 0
    if not (terminal_boundary or permanent_orphan):
        raise RuntimeError("finalize 경계에 도달하지 않았다")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    choice_doc = {"schema": "vision_selection_choice_results_v1", "rows": choices,
                  "api_usage": summary["api_usage"]}
    if not _check_output_pair(output, choice_doc, summary):
        _write_once(output / "choice_results.json", choice_doc)
        _write_once(output / "comparison_summary.json", summary)
    result["finalized"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "remaining_commands", "summarize"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--input-locations", required=True, type=Path)
    parser.add_argument("--evidence-locations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    context = load_context(
        args.protocol, args.protocol_sha256_lf, args.source_anchor, args.input_locations
    )
    if args.command == "verify":
        result = inspect_run(context)
    elif args.command == "remaining_commands":
        result = remaining_commands(context)
    else:
        if args.evidence_locations is None or args.output_dir is None:
            parser.error("summarize에는 --evidence-locations와 --output-dir가 필요하다")
        result = summarize(
            context, args.evidence_locations, args.output_dir, args.finalize
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
