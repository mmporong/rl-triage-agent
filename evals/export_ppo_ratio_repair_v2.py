"""PPO ratio repair v2 결과를 재검증해 공개 allowlist JSON으로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast, overload

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals", ROOT / "bench"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import ppo_ratio_repair_contract_v2 as CONTRACT  # noqa: E402
import ppo_ratio_repair_demo_v2 as DEMO  # noqa: E402
import ppo_ratio_repair_execution as EXECUTION  # noqa: E402
import ppo_ratio_repair_model_call_v2 as MODEL_CALL  # noqa: E402
import ppo_ratio_repair_training as TRAINING  # pyright: ignore[reportMissingImports]  # noqa: E402

from rl_triage import grounded_recovery as G  # noqa: E402
from rl_triage import recovery as REC  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger  # noqa: E402

TAG = "ppo_ratio_repair_s2163_20261023"
RUN = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_repair/runs" / TAG
PROTOCOL = ROOT / "bench/protocols/ppo_ratio_repair_demo_v2.json"
BANDS = ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json"
OUTPUT = ROOT / "evals/results" / TAG
AUDIT = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_repair/exports" / TAG
PUBLIC_FILES = {
    "summary.json",
    "evaluations.json",
    "model_evidence.json",
    "probe_correction.json",
    "provenance.json",
}
FIXED_EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
FIXED_EVAL_SHA256_LF = "8d24652eeed47f462d78244fceec785164b7695e5b757ae8a4e9bd7947aa8a46"
TRAINING_MANIFEST_SCHEMA = "ppo_ratio_repair_training_manifest_v1"
TRAINING_MANIFEST_KEYS = {
    "schema", "mode", "arm", "seed", "run_name", "task", "num_envs", "max_iterations",
    "configured_learning_epochs", "configured_mini_batches", "runtime_updates_to_observe",
    "source_commit_sha", "protocol_path", "protocol_sha256_lf", "source_sha256_lf",
    "runtime_source_sha256_lf", "records_path", "readback_path", "source_checkpoint_sha256",
}
EXPECTED_ARMS = {
    "NONE": (2163, "training/none", f"{TAG}_none", None),
    DEMO.FAULT: (2163, "training/inverse_ratio", f"{TAG}_inverse_ratio", None),
    "CORRECTED": (3163, "correction_training", f"{TAG}_corrected", "fault"),
}

_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/", re.IGNORECASE)
_PRIVATE = re.compile(r"bench[\\/]private|claude_reward_track|codex_support", re.IGNORECASE)
_DANGEROUS_PATH = re.compile(r"^(?:/|\\\\)|(?:^|[\\/])\.\.(?:[\\/]|$)")
_FORBIDDEN_KEYS = {
    "path", "run_dir", "command", "args", "argv", "environment", "env", "stack", "traceback",
    "api_key", "authorization", "secret", "response_id", "content", "reasoning_content",
}


def sha256(path: Path, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


@overload
def git(*args: str, binary: Literal[False] = False) -> str: ...


@overload
def git(*args: str, binary: Literal[True]) -> bytes: ...


def git(*args: str, binary: bool = False) -> str | bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
    )
    return process.stdout if binary else process.stdout.strip()


def _source(manifest: dict[str, str], logical_id: str, path: Path, *, lf: bool = False) -> None:
    if not path.is_file():
        raise ValueError(f"source artifact missing: {logical_id}")
    if logical_id in manifest:
        raise ValueError(f"duplicate source artifact id: {logical_id}")
    manifest[logical_id] = sha256(path, lf)


def _verify_protocol(
    protocol: dict,
    result: dict,
    manifest: dict[str, str],
    protocol_path: Path,
) -> None:
    protocol_sha = sha256(protocol_path, True)
    if (
        protocol.get("schema") != "ppo_ratio_repair_demo_protocol_v2"
        or protocol.get("seed") != result.get("seed")
        or protocol.get("restart_seed") != result.get("restart_seed")
        or result.get("protocol_sha256_lf") != protocol_sha
        or result.get("design_coupling") != protocol.get("design_coupling")
    ):
        raise ValueError("result and protocol binding differs")
    commit = result["experiment_commit"]
    git("cat-file", "-e", f"{commit}^{{commit}}")
    git("merge-base", "--is-ancestor", commit, str(git("rev-parse", "HEAD")))
    protocol_relative = protocol_path.resolve().relative_to(ROOT).as_posix()
    committed_protocol = git("show", f"{commit}:{protocol_relative}", binary=True)
    if hashlib.sha256(committed_protocol.replace(b"\r\n", b"\n")).hexdigest() != protocol_sha:
        raise ValueError("committed protocol blob differs")
    head_protocol = git("show", f"HEAD:{protocol_relative}", binary=True)
    if hashlib.sha256(head_protocol.replace(b"\r\n", b"\n")).hexdigest() != protocol_sha:
        raise ValueError("current HEAD protocol blob differs")
    for field in ("source_sha256_lf", "evidence_sha256_lf"):
        entries = protocol.get(field)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"protocol {field} is missing")
        for relative, expected in entries.items():
            path = ROOT / relative
            if not path.is_file() or sha256(path, True) != expected:
                raise ValueError(f"protocol-bound source differs: {relative}")
            committed = git("show", f"{commit}:{relative}", binary=True)
            if hashlib.sha256(committed.replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError(f"committed source differs: {relative}")
            head = git("show", f"HEAD:{relative}", binary=True)
            if hashlib.sha256(head.replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError(f"current HEAD source differs: {relative}")
            _source(manifest, f"protocol_{field}:{relative}", path, lf=True)
    _source(manifest, "protocol", protocol_path, lf=True)


def _verify_begin_and_preflight(run: Path, result: dict, protocol: dict, manifest: dict[str, str]) -> dict:
    begin_path, seed_path, launch_path = run / "begin.json", run / "seed_freshness.json", run / "launch_paths.json"
    for logical_id, path in (("begin", begin_path), ("seed_freshness", seed_path), ("launch_paths", launch_path)):
        _source(manifest, logical_id, path)
    begin, freshness, launch = read(begin_path), read(seed_path), read(launch_path)
    if begin != {
        "tag": result["tag"],
        "experiment_commit": result["experiment_commit"],
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "approver": begin.get("approver"),
        "identity_attested": False,
        "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임",
    } or not isinstance(begin["approver"], str) or not begin["approver"].strip():
        raise ValueError("begin binding differs")
    if (
        freshness.get("schema") != "preregistered_seed_freshness_v1"
        or freshness.get("status") != "fresh"
        or freshness.get("seeds") != [protocol["seed"], protocol["restart_seed"]]
        or freshness.get("evidence") != []
    ):
        raise ValueError("seed freshness evidence differs")
    if launch.get("processes_started") != 0 or launch.get("gpu_queries") != 0:
        raise ValueError("launch path preflight has side effects")
    return {"begin": begin, "freshness": freshness, "launch": launch}


def _training_evidence(run: Path, result: dict, protocol: dict, manifest: dict[str, str]) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    fault_checkpoint = None
    for arm, (seed, folder_name, run_name, source_kind) in EXPECTED_ARMS.items():
        artifact_path = run / f"artifact_{arm.lower()}.json"
        manifest_path = run / folder_name / "manifest.json"
        if not artifact_path.is_file() and not manifest_path.is_file():
            continue
        _source(manifest, f"training_manifest:{arm}", manifest_path)
        _source(manifest, f"artifact:{arm}", artifact_path)
        training, artifact = read(manifest_path), read(artifact_path)
        if set(training) != TRAINING_MANIFEST_KEYS:
            raise ValueError(f"training manifest field inventory differs: {arm}")
        expected = {
            "schema": TRAINING_MANIFEST_SCHEMA,
            "mode": "train",
            "arm": arm,
            "seed": seed,
            "run_name": run_name,
            "task": protocol["task"],
            "num_envs": protocol["num_envs"],
            "max_iterations": protocol["training_updates"],
            "configured_learning_epochs": protocol["inspection"]["epochs"],
            "configured_mini_batches": protocol["inspection"]["mini_batches"],
            "runtime_updates_to_observe": protocol["inspection"]["updates"],
            "source_commit_sha": result["experiment_commit"],
            "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256_lf": result["protocol_sha256_lf"],
            "source_sha256_lf": protocol["source_sha256_lf"],
            "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        }
        if any(training.get(key) != value for key, value in expected.items()):
            raise ValueError(f"training manifest binding differs: {arm}")
        expected_private_files = {
            "records_path": (manifest_path.parent / "records.jsonl").resolve(),
            "readback_path": (manifest_path.parent / "readback.json").resolve(),
        }
        if any(Path(training[key]).resolve() != value for key, value in expected_private_files.items()):
            raise ValueError(f"training manifest execution path differs: {arm}")
        records_path = expected_private_files["records_path"]
        readback_path = expected_private_files["readback_path"]
        checkpoint_receipt_path = manifest_path.parent / "checkpoint.json"
        for label, path in (
            (f"training_records:{arm}", records_path),
            (f"training_readback:{arm}", readback_path),
            (f"training_checkpoint_receipt:{arm}", checkpoint_receipt_path),
        ):
            _source(manifest, label, path)
        measured = EXECUTION.inspection(records_path, training)
        measured_inspection = {
            key: measured[key]
            for key in (
                "count_status", "gradient_status", "first_mismatch_update", "gradient_batches",
                "optimizer_steps", "minibatches",
            )
        }
        raw_readback = TRAINING.validate_readback(read(readback_path), arm)
        raw_checkpoint_evidence = read(checkpoint_receipt_path)
        source_checkpoint = training.get("source_checkpoint_sha256")
        if source_kind is None and source_checkpoint is not None:
            raise ValueError(f"source training unexpectedly loads a checkpoint: {arm}")
        if source_kind == "fault" and source_checkpoint != fault_checkpoint:
            raise ValueError("corrected training source checkpoint differs")
        inspection = artifact.get("inspection")
        expected_gradient = "abnormal" if arm == DEMO.FAULT else "normal"
        expected_mismatch = 0 if arm == DEMO.FAULT else None
        expected_inspection = {
            "count_status": "normal",
            "gradient_status": expected_gradient,
            "first_mismatch_update": expected_mismatch,
            "gradient_batches": 60,
            "optimizer_steps": [20, 20, 20],
            "minibatches": [20, 20, 20],
        }
        checkpoint = artifact.get("checkpoint")
        checkpoint_evidence = artifact.get("checkpoint_evidence")
        execution = artifact.get("execution")
        readback = artifact.get("readback")
        report = artifact.get("report")
        if (
            artifact.get("arm") != arm
            or artifact.get("seed") != seed
            or artifact.get("run_name") != run_name
            or inspection != expected_inspection
            or inspection != measured_inspection
            or not isinstance(checkpoint, dict)
            or Path(str(checkpoint.get("path", ""))).name != "model_299.pt"
            or not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256", "")))
            or artifact.get("configuration") != {
                "seed": seed,
                "max_iterations": protocol["training_updates"],
                "num_learning_epochs": protocol["inspection"]["epochs"],
                "num_mini_batches": protocol["inspection"]["mini_batches"],
            }
        ):
            raise ValueError(f"training artifact binding differs: {arm}")
        expected_mutation_arm = {
            "NONE": "NONE", DEMO.FAULT: "INVERSE_RATIO", "CORRECTED": "RESTORED",
        }[arm]
        mutation = readback.get("mutation") if isinstance(readback, dict) else None
        expected_readback_configuration = {
            "learning_epochs": protocol["inspection"]["epochs"],
            "mini_batches": protocol["inspection"]["mini_batches"],
            "clip_param": 0.2,
        }
        expected_primitive_status = "abnormal" if arm == DEMO.FAULT else "normal"
        checkpoint_path = Path(str(checkpoint["path"])).resolve()
        first_checkpoint = Path(str(raw_checkpoint_evidence.get("first", {}).get("path", ""))).resolve()
        final_checkpoint = Path(str(raw_checkpoint_evidence.get("final", {}).get("path", ""))).resolve()
        if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint["sha256"]:
            raise ValueError(f"training checkpoint bytes differ: {arm}")
        if not first_checkpoint.is_file() or sha256(first_checkpoint) != raw_checkpoint_evidence.get("first", {}).get("sha256"):
            raise ValueError(f"training first checkpoint bytes differ: {arm}")
        _source(manifest, f"training_checkpoint:{arm}", checkpoint_path)
        _source(manifest, f"training_first_checkpoint:{arm}", first_checkpoint)
        if (
            not isinstance(execution, dict)
            or execution.get("natural_exit") is not True
            or execution.get("timed_out") is not False
            or execution.get("exit_code") != 0
            or execution.get("cleanup_verified") is not True
            or not isinstance(readback, dict)
            or artifact.get("records_path") is None
            or Path(str(artifact["records_path"])).resolve() != records_path
            or readback != raw_readback
            or readback.get("arm") != arm
            or readback.get("configuration") != expected_readback_configuration
            or not isinstance(mutation, dict)
            or mutation.get("arm") != expected_mutation_arm
            or mutation.get("mutation") != "inverse_importance_ratio_v1"
            or mutation.get("changed_ratio_assignments") != 1
            or mutation.get("active_mutation") is not (arm == DEMO.FAULT)
            or mutation.get("restored_original") is not (arm == "CORRECTED")
            or not isinstance(report, dict)
            or report != measured["report"]
            or report.get("schema") != "optimizer_observability_inspection_v1"
            or report.get("records") != protocol["inspection"]["updates"]
            or report.get("status") != expected_primitive_status
            or report.get("manifest", {}).get("arm") != arm
            or report.get("manifest", {}).get("seed") != seed
            or report.get("manifest", {}).get("source_commit_sha") != result["experiment_commit"]
            or report.get("manifest", {}).get("protocol_sha256_lf") != result["protocol_sha256_lf"]
            or report.get("primitives", {}).get("optimizer_step_count", {}).get("status") != "normal"
            or report.get("primitives", {}).get("minibatch_count", {}).get("status") != "normal"
            or report.get("primitives", {}).get("ppo_surrogate_gradient", {}).get("status")
            != expected_primitive_status
            or not isinstance(checkpoint_evidence, dict)
            or checkpoint_evidence != raw_checkpoint_evidence
            or checkpoint_evidence.get("device") != "cpu"
            or checkpoint_evidence.get("first", {}).get("iteration") != 0
            or checkpoint_evidence.get("final", {}).get("iteration") != 299
            or checkpoint_evidence.get("final", {}).get("sha256") != checkpoint["sha256"]
            or final_checkpoint != checkpoint_path
            or first_checkpoint.name != "model_0.pt"
        ):
            raise ValueError(f"training artifact execution evidence differs: {arm}")
        artifacts[arm] = artifact
        if arm == DEMO.FAULT:
            fault_checkpoint = artifact["checkpoint"]["sha256"]
    if "CORRECTED" in artifacts and DEMO.FAULT not in artifacts:
        raise ValueError("corrected training exists without fault source")
    return artifacts


def _public_evaluation(arm: str, report: dict, verdict: dict, source_sha: str) -> dict:
    conditions = []
    for row in report["by_condition"]:
        conditions.append({
            "target_velocity": row["command"],
            **{
                key: row[key]
                for key in (
                    "num_envs", "fall_count", "alive_steps_sum", "lin_error_sq_sum", "yaw_error_sq_sum",
                    "fall_rate", "lin_vel_rmse_mps", "yaw_rate_rmse_radps",
                )
            },
        })
    return {
        "arm": arm,
        "label": verdict["label"],
        "verdict": verdict,
        "protocol": {
            key: report[key]
            for key in (
                "protocol", "protocol_sha256_lf", "task", "eval_seed", "num_envs", "horizon_steps",
                "step_dt", "action_scale",
            )
        },
        "checkpoint": {
            "file": report["checkpoint"]["file"],
            "sha256": report["checkpoint"]["sha256"],
        },
        "metrics": report["metrics"],
        "standstill": report["standstill"],
        "condition_count": len(conditions),
        "conditions": conditions,
        "source_artifact_sha256": source_sha,
    }


def _evaluations(
    run: Path,
    result: dict,
    manifest: dict[str, str],
    artifacts: dict[str, dict],
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    locations = {
        "NONE": run / "source_evaluation/none.json",
        DEMO.FAULT: run / "source_evaluation/inverse_ratio.json",
        "CORRECTED": run / "correction_evaluation/corrected.json",
    }
    reports: dict[str, dict] = {}
    checked: dict[str, dict] = {}
    for arm, report_path in locations.items():
        if not report_path.is_file():
            continue
        if arm not in artifacts:
            raise ValueError(f"fixed evaluation exists without training artifact: {arm}")
        _source(manifest, f"fixed_evaluation:{arm}", report_path)
        artifact, report = artifacts[arm], read(report_path)
        expected_names = {
            "NONE": f"{TAG}_none",
            DEMO.FAULT: f"{TAG}_inverse_ratio",
            "CORRECTED": f"{TAG}_fixed_corrected",
        }
        expected = {
            "name": expected_names[arm],
            "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": artifact["checkpoint"]["sha256"],
            "action_scale": 0.25,
            "protocol_sha256_lf": FIXED_EVAL_SHA256_LF,
        }
        reports[arm] = report
        checked[arm] = G.validate_fixed_report(report, expected)
    if "source_gate" in result and set(checked) < {"NONE", DEMO.FAULT}:
        raise ValueError("source gate exists without both source fixed evaluations")
    if "recovery" in result and "CORRECTED" not in checked:
        raise ValueError("recovery exists without corrected fixed evaluation")
    if not checked:
        return {"schema": "public_ppo_ratio_repair_evaluations_v2", "arms": {}}, checked, {}
    bands = read(BANDS)["bands"]
    _source(manifest, "recovery_bands", BANDS)
    if "NONE" not in checked:
        raise ValueError("a fixed evaluation exists without NONE reference")
    verdicts = {
        arm: REC.verdict(REC.compare(G._behavior(report), G._behavior(checked["NONE"])), bands)
        for arm, report in checked.items()
    }
    if len({json.dumps(row["versions"], sort_keys=True) for row in checked.values()}) != 1:
        raise ValueError("fixed evaluation runtime versions differ")
    if "source_gate" in result:
        recomputed_gate = DEMO.source_gate(checked, bands)
        if result["source_gate"] != recomputed_gate:
            raise ValueError("stored source gate differs from fixed evaluation recomputation")
        source_gate_path = run / "source_gate.json"
        _source(manifest, "source_gate", source_gate_path)
        if read(source_gate_path) != recomputed_gate:
            raise ValueError("source gate artifact differs from fixed evaluation recomputation")
    if "recovery" in result:
        recovery = result["recovery"]
        expected_outcome = "recovered" if verdicts["CORRECTED"]["label"] == "healthy" else "not_recovered"
        expected_recovery = {
            "outcome": expected_outcome,
            "verdict": verdicts["CORRECTED"],
            "source_metrics": checked[DEMO.FAULT]["metrics"],
            "reference_metrics": checked["NONE"]["metrics"],
            "corrected_metrics": checked["CORRECTED"]["metrics"],
            "checkpoint_sha256": artifacts["CORRECTED"]["checkpoint"]["sha256"],
        }
        if recovery != expected_recovery or result.get("outcome") not in {expected_outcome, "correction_unknown"}:
            raise ValueError("stored recovery differs from fixed evaluation recomputation")
    public = {
        "schema": "public_ppo_ratio_repair_evaluations_v2",
        "arms": {
            arm: _public_evaluation(arm, checked[arm], verdicts[arm], manifest[f"fixed_evaluation:{arm}"])
            for arm in checked
        },
    }
    return public, checked, verdicts


def _call_cost(call: dict) -> dict:
    out = {
        key: call.get(key)
        for key in ("provider_model", "input_tokens", "output_tokens", "total_tokens", "wall_s")
    }
    numeric = (out["input_tokens"], out["output_tokens"], out["total_tokens"])
    if all(type(value) is int for value in numeric):
        input_tokens = cast(int, out["input_tokens"])
        output_tokens = cast(int, out["output_tokens"])
        total_tokens = cast(int, out["total_tokens"])
        if total_tokens != input_tokens + output_tokens:
            raise ValueError("model token accounting differs")
    elif any(value is not None for value in numeric):
        raise ValueError("model token accounting is partial")
    if out["wall_s"] is not None and (
        isinstance(out["wall_s"], bool)
        or not isinstance(out["wall_s"], (int, float))
        or not math.isfinite(out["wall_s"])
        or out["wall_s"] < 0
    ):
        raise ValueError("model wall time is invalid")
    return out


def _validate_reservation(kind: str, packet_data: bytes, reservation: dict) -> None:
    if not isinstance(reservation.get("at"), str):
        raise ValueError("model reservation timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(reservation["at"])
    except ValueError as error:
        raise ValueError("model reservation timestamp is invalid") from error
    if timestamp.utcoffset() is None:
        raise ValueError("model reservation timestamp lacks a UTC offset")
    expected = MODEL_CALL.reservation_for(kind, packet_data)
    expected["at"] = reservation["at"]
    if reservation != expected:
        raise ValueError("orphan model reservation differs from packet binding")


def _validate_transport_receipt(value: dict) -> None:
    allowed = {"type", "worker_cleanup"}
    if set(value) not in (allowed, allowed | {"cleanup_error_type"}):
        raise ValueError("model transport receipt field inventory differs")
    if value.get("type") != "dialogue_timeout" or value.get("worker_cleanup") not in {
        "verified", "unverified", "unverified_no_pid",
    }:
        raise ValueError("model transport receipt type or cleanup status differs")
    if "cleanup_error_type" in value and (
        value["worker_cleanup"] != "unverified"
        or not isinstance(value["cleanup_error_type"], str)
        or not value["cleanup_error_type"]
    ):
        raise ValueError("model transport cleanup error is malformed")


def _models(run: Path, result: dict, manifest: dict[str, str]) -> dict:
    model_root = run / "model"
    model_files = sorted(model_root.iterdir(), key=lambda path: path.name) if model_root.is_dir() else []
    if any(not path.is_file() or path.is_symlink() for path in model_files):
        raise ValueError("model evidence inventory contains a directory or symlink")
    allowed_name = re.compile(
        r"(?:diagnosis|correction)\.(?:packet\.json|feedback[12](?:\.reserved|\.pid)?\.json|review[12]\.json|transport_error\.json)"
    )
    if any(allowed_name.fullmatch(path.name) is None for path in model_files):
        raise ValueError("model evidence inventory contains an unexpected file")
    model_names = {path.name for path in model_files}
    for path in model_files:
        if path.name.endswith(".pid.json"):
            reservation_name = path.name.replace(".pid.json", ".reserved.json")
            if reservation_name not in model_names:
                raise ValueError("orphan model PID lacks its reservation")
        review_match = re.fullmatch(r"(diagnosis|correction)\.review([12])\.json", path.name)
        if review_match:
            kind, index = review_match.groups()
            record_name = f"{kind}.feedback{index}.json"
            attempts = result.get("model", {}).get(kind, {}).get("attempts", [])
            if record_name not in model_names or sum(
                isinstance(row, dict) and row.get("record") == record_name for row in attempts
            ) != 1:
                raise ValueError("orphan model review lacks its record and result attempt")
    for path in model_files:
        _source(manifest, f"model_file:{path.name}", path)
    for kind in ("diagnosis", "correction"):
        packet_path = model_root / f"{kind}.packet.json"
        reservations = sorted(model_root.glob(f"{kind}.feedback[12].reserved.json")) \
            if model_root.is_dir() else []
        if reservations and not packet_path.is_file():
            raise ValueError("model reservation exists without its packet")
        packet_data = packet_path.read_bytes() if packet_path.is_file() else b""
        for reservation_path in reservations:
            _validate_reservation(kind, packet_data, read(reservation_path))
            pid_path = reservation_path.with_name(reservation_path.name.replace(".reserved.json", ".pid.json"))
            if pid_path.is_file():
                pid = read(pid_path)
                if (
                    set(pid) != {"pid", "start_ticks", "packet_sha256"}
                    or type(pid["pid"]) is not int
                    or pid["pid"] <= 1
                    or not isinstance(pid["start_ticks"], str)
                    or not pid["start_ticks"].isdigit()
                    or pid["packet_sha256"] != hashlib.sha256(packet_data).hexdigest()
                ):
                    raise ValueError("model worker identity differs from packet binding")
        transport_path = model_root / f"{kind}.transport_error.json"
        if transport_path.is_file():
            _validate_transport_receipt(read(transport_path))
            pending_for_kind = [
                path for path in reservations
                if not path.with_name(path.name.replace(".reserved.json", ".json")).is_file()
            ]
            if len(pending_for_kind) != 1:
                raise ValueError("model transport receipt lacks one packet-bound pending reservation")
    records = [path for path in model_files if re.fullmatch(r"(?:diagnosis|correction)\.feedback[12]\.json", path.name)]
    public = []
    request_count = 0
    records_by_kind: dict[str, list[str]] = {"diagnosis": [], "correction": []}
    for record_path in records:
        saved = read(record_path)
        kind = saved.get("kind")
        if kind not in ("diagnosis", "correction"):
            raise ValueError("model record kind differs")
        packet_path = model_root / f"{kind}.packet.json"
        reservation_path = record_path.with_suffix(".reserved.json")
        review = MODEL_CALL.review(saved, packet_path.read_bytes(), kind, read(reservation_path))
        part = result.get("model", {}).get(kind)
        if not isinstance(part, dict) or not isinstance(part.get("attempts"), list):
            raise ValueError("result is missing model attempt evidence")
        attempt = next((row for row in part["attempts"] if row.get("record") == record_path.name), None)
        costs = [_call_cost(call) for call in saved.get("calls", [])]
        attempt_match = re.search(r"feedback([12])", record_path.name)
        if attempt_match is None:
            raise ValueError("model attempt filename differs")
        attempt_index = int(attempt_match.group(1))
        review_path = model_root / f"{kind}.review{attempt_index}.json"
        if (
            attempt is None
            or attempt.get("record_sha256") != sha256(record_path)
            or attempt.get("review") != review
            or attempt.get("calls") != costs
            or not review_path.is_file()
            or read(review_path) != attempt
        ):
            raise ValueError("stored model attempt differs from raw record recomputation")
        execution_wall_s = saved.get("execution_wall_s")
        if (
            isinstance(execution_wall_s, bool)
            or not isinstance(execution_wall_s, (int, float))
            or not math.isfinite(execution_wall_s)
            or execution_wall_s < 0
        ):
            raise ValueError("model execution wall time is invalid")
        request_count += saved.get("requests", 0)
        records_by_kind[kind].append(record_path.name)
        public.append({
            "kind": kind,
            "dialogue": len([row for row in public if row["kind"] == kind]) + 1,
            "review": review,
            "feedback": saved.get("feedback"),
            "calls": costs,
            "execution_wall_s": execution_wall_s,
            "source_record_sha256": manifest[f"model_file:{record_path.name}"],
        })
    expected_saved = result.get("api_requests_in_saved_records", 0)
    if request_count != expected_saved:
        raise ValueError("raw model request count differs from result")
    actual_pending = sorted(
        path.name
        for path in model_root.glob("*.reserved.json")
        if not path.with_name(path.name.replace(".reserved.json", ".json")).is_file()
    ) if model_root.is_dir() else []
    actual_transport = sorted(path.name for path in model_root.glob("*.transport_error.json")) \
        if model_root.is_dir() else []
    pending = result.get("pending_model_records", [])
    transport = result.get("transport_errors", [])
    if pending != actual_pending or transport != actual_transport:
        raise ValueError("pending model reservation or transport evidence differs")
    expected_final = request_count if not pending and not transport else None
    if result.get("api_requests") != expected_final:
        raise ValueError("completed model request count differs")
    for kind, part in result.get("model", {}).items():
        if kind not in records_by_kind:
            raise ValueError(f"model result has an unknown kind: {kind}")
        if not isinstance(part, dict) or not part.get("attempts"):
            raise ValueError(f"model result is malformed: {kind}")
        if part.get("review") != part["attempts"][-1].get("review"):
            raise ValueError("final model review differs from last attempt")
        attempt_names = [row.get("record") for row in part["attempts"]]
        if attempt_names != records_by_kind.get(kind, []):
            raise ValueError("model attempt order or inventory differs")
    if any(records_by_kind[kind] and kind not in result.get("model", {}) for kind in records_by_kind):
        raise ValueError("raw model record is absent from result")
    return {
        "schema": "public_ppo_ratio_repair_model_evidence_v2",
        "request_model": MODEL_CALL.MODEL,
        "accepted_provider_models": sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "api_requests": result.get("api_requests"),
        "api_requests_in_saved_records": request_count,
        "pending_record_count": len(pending),
        "transport_error_count": len(transport),
        "dialogues": public,
    }


def _lifecycle(events: list[dict], request_id: str) -> list[str]:
    return [row["event"] for row in events if row.get("request_id") == request_id]


def _approval_matches(events: list[dict], request_id: str, approver: str) -> bool:
    approvals = [row for row in events if row.get("request_id") == request_id and row["event"] == "approved"]
    return len(approvals) == 1 and approvals[0].get("approver") == approver


def _probe_correction(
    run: Path,
    result: dict,
    manifest: dict[str, str],
    protocol: dict,
    artifacts: dict[str, dict],
) -> tuple[dict, dict | None]:
    probes_path, corrections_path = run / "probes.ledger.jsonl", run / "correction.ledger.jsonl"
    approver = read(run / "begin.json")["approver"]
    probe_rows = []
    allowed_missing_failed_receipt = False
    if probes_path.is_file():
        _source(manifest, "probe_ledger", probes_path)
        ledger = CONTRACT.M.RegistryLedger(probes_path, CONTRACT.registry())
        events, requests = ledger.events(), ledger.requests()
        if any(request["state"] != "done" for request in requests.values()):
            raise ValueError("probe ledger has an unfinished request")
        stored_step_list = result.get("probe_steps", [])
        if not isinstance(stored_step_list, list):
            raise ValueError("result probe steps are not a list")
        stored_steps = {step.get("request_id"): step for step in stored_step_list}
        if len(stored_steps) != len(stored_step_list):
            raise ValueError("result probe steps contain duplicate request ids")
        proposed_ids = [row["request_id"] for row in events if row["event"] == "proposed"]
        expected_steps = []
        for index, request_id in enumerate(proposed_ids, 1):
            request = requests[request_id]
            args_digest = CONTRACT.L.digest(CONTRACT.registry()["probes"][request["probe"]]["args"])
            prereg = {
                "checkpoint_sha256": artifacts[DEMO.FAULT]["checkpoint"]["sha256"],
                "seed": protocol["seed"],
                "protocol_sha256_lf": result["protocol_sha256_lf"],
                "experiment_commit": result["experiment_commit"],
                "registry_digest": ledger.registry_digest,
            }
            prereg_digest = CONTRACT.L.digest(prereg)
            if (
                request.get("registry_digest") != ledger.registry_digest
                or request.get("args_digest") != args_digest
                or request.get("prereg_digest") != prereg_digest
                or request.get("effect_key") != ledger.effect_key(prereg_digest, request["probe"], args_digest)
                or request.get("budget_gpu_s") != protocol["gpu"]["probe_execution_budget_s"]
            ):
                raise ValueError("probe request registry, args, preregistration, or effect binding differs")
            receipt = next((row for row in events if row.get("request_id") == request_id and row["event"] == "receipt"), None)
            expected_step = {**receipt, "probe": request["probe"]} if receipt is not None else None
            expected_steps.append(expected_step)
            stored_step = stored_steps.get(request_id)
            if receipt is None or (stored_step is not None and stored_step != expected_step):
                raise ValueError("probe result and ledger receipt differ")
            if stored_step is None and (receipt["outcome"] != "unknown" or result.get("outcome") != "invalid"):
                raise ValueError("successful probe receipt is absent from result")
            lifecycle = _lifecycle(events, request_id)
            gpu_s = receipt.get("gpu_s")
            over_budget = isinstance(gpu_s, (int, float)) and not isinstance(gpu_s, bool) \
                and gpu_s > request["budget_gpu_s"]
            if (
                lifecycle != ["proposed", "approved", "consumed", "receipt"]
                or not _approval_matches(events, request_id, approver)
                or isinstance(gpu_s, bool)
                or not isinstance(gpu_s, (int, float))
                or not math.isfinite(gpu_s)
                or gpu_s < 0
                or receipt.get("over_budget") is not over_budget
                or ((receipt.get("exit_code") != 0 or over_budget) and receipt.get("outcome") != "unknown")
            ):
                raise ValueError("probe approval lifecycle differs")
            probe_rows.append({
                "order": index,
                "probe": request["probe"],
                "request_id": request_id,
                "approval_events": lifecycle,
                "outcome": receipt["outcome"],
                "measurement": receipt["measurement"],
                "gpu_s": receipt["gpu_s"],
                "exit_code": receipt["exit_code"],
                "over_budget": receipt["over_budget"],
            })
        if set(stored_steps) - set(requests):
            raise ValueError("result contains a probe absent from the ledger")
        allowed_missing_failed_receipt = (
            result.get("outcome") == "invalid"
            and isinstance(result.get("failure"), dict)
            and len(expected_steps) == len(stored_step_list) + 1
            and expected_steps[-1] is not None
            and expected_steps[-1].get("outcome") == "unknown"
            and expected_steps[-1].get("exit_code") != 0
            and stored_step_list == expected_steps[:-1]
        )
        if stored_step_list != expected_steps and not allowed_missing_failed_receipt:
            raise ValueError("result probe step order differs from ledger receipts")
    elif result.get("probe_steps"):
        raise ValueError("result has probe steps without a ledger")

    corrections = []
    replayed_loop = None
    if corrections_path.is_file():
        _source(manifest, "correction_ledger", corrections_path)
        ledger = CorrectionLedger(corrections_path, CONTRACT.correction_registry())
        events, requests = ledger.events(), ledger.requests()
        if len(requests) != 1:
            raise ValueError("correction request inventory differs from the single-action contract")
        for request_id, request in requests.items():
            if request["state"] != "done":
                raise ValueError("correction ledger has an unfinished request")
            receipt = request["receipt"]
            lifecycle = _lifecycle(events, request_id)
            gpu_s = receipt.get("gpu_s")
            over_budget = isinstance(gpu_s, (int, float)) and not isinstance(gpu_s, bool) \
                and gpu_s > request["budget_gpu_s"]
            if (
                lifecycle != ["proposed", "approved", "consumed", "receipt"]
                or not _approval_matches(events, request_id, approver)
                or (gpu_s is not None and (
                    isinstance(gpu_s, bool) or not isinstance(gpu_s, (int, float))
                    or not math.isfinite(gpu_s) or gpu_s < 0
                ))
                or receipt.get("over_budget") is not over_budget
                or ((receipt.get("exit_code") != 0 or over_budget) and receipt.get("verdict") != "unknown")
            ):
                raise ValueError("correction approval lifecycle differs")
            if receipt["tag"] != result["tag"]:
                raise ValueError("correction receipt tag differs")
            registry = CONTRACT.correction_registry().get(request.get("confirmed_mechanism"))
            correction_review = result.get("model", {}).get("correction")
            proposal = (correction_review or {}).get("review", {}).get("decision", {}).get("proposal", {})
            packet_path = run / "model/correction.packet.json"
            expected_evidence = {
                "packet_digest": CONTRACT.L.digest(read(packet_path)),
                "model_review_digest": CONTRACT.L.digest(correction_review),
                "protocol_sha256_lf": result["protocol_sha256_lf"],
                "checkpoint_sha256": artifacts[DEMO.FAULT]["checkpoint"]["sha256"],
            }
            if (
                registry is None
                or request.get("registry_digest") != ledger.registry_digest
                or request.get("correction_id") != registry["correction_id"]
                or request.get("recovery_mode") not in registry["recovery_modes"]
                or request.get("budget_gpu_s") != protocol["gpu"]["correction_execution_budget_s"]
                or request.get("protocol_case_arm") != {"case": registry["protocol_case"], "arm": "corrected"}
                or request.get("effect_key") != ledger.effect_key(
                    request["loop_state_digest"], request["correction_id"], request["recovery_mode"]
                )
                or request.get("loop_state_digest") != CONTRACT.L.digest(result.get("loop"))
                or request.get("receipts_digest") != CONTRACT.L.digest(result.get("probe_steps", []))
                or request.get("evidence") != expected_evidence
                or proposal.get("correction_id") != request.get("correction_id")
                or proposal.get("recovery_mode") != request.get("recovery_mode")
            ):
                raise ValueError("correction request registry, receipt, or effect binding differs")
            recovery_path = run / "recovery.json"
            receipt_verdict = receipt.get("verdict")
            result_sha = receipt.get("result_sha")
            if receipt_verdict in {"recovered", "not_recovered"}:
                if result_sha is None:
                    raise ValueError("completed correction receipt is missing a result hash")
                _source(manifest, "recovery", recovery_path)
                if result_sha != manifest["recovery"] or read(recovery_path) != result.get("recovery"):
                    raise ValueError("correction receipt result binding differs")
                if receipt_verdict != result["recovery"]["outcome"] or result.get("outcome") != receipt_verdict:
                    raise ValueError("correction receipt verdict differs from recovery outcome")
            elif receipt_verdict == "unknown":
                if result.get("outcome") != "correction_unknown":
                    raise ValueError("unknown correction receipt is not separated from a completed outcome")
                if result_sha is not None:
                    _source(manifest, "recovery", recovery_path)
                    if result_sha != manifest["recovery"] or read(recovery_path) != result.get("recovery"):
                        raise ValueError("unknown correction receipt result binding differs")
                elif recovery_path.exists() or "recovery" in result:
                    raise ValueError("unbound recovery evidence accompanies an unknown correction receipt")
            else:
                raise ValueError("correction receipt verdict is outside the registry")
            corrections.append({
                "request_id": request_id,
                "correction_id": request["correction_id"],
                "recovery_mode": request["recovery_mode"],
                "confirmed_mechanism": request["confirmed_mechanism"],
                "approval_events": lifecycle,
                "verdict": receipt["verdict"],
                "gpu_s": receipt["gpu_s"],
                "exit_code": receipt["exit_code"],
                "over_budget": receipt["over_budget"],
            })
    diagnosis = result.get("model", {}).get("diagnosis", {}).get("review", {})
    decision = diagnosis.get("decision")
    completed_loop_artifacts = 0
    if probe_rows:
        if not isinstance(decision, dict) or decision.get("action") != "continue":
            raise ValueError("probe execution exists without an accepted diagnosis")
        state = CONTRACT.initial_state(decision)
        for index, row in enumerate(probe_rows, 1):
            if row["probe"] != state["next_probe"]:
                raise ValueError("probe order differs from replayed registry loop")
            state = CONTRACT.advance(state, row["probe"], row["outcome"])
            loop_path = run / f"loop_{index:02d}.json"
            if loop_path.is_file():
                _source(manifest, f"loop:{index:02d}", loop_path)
                if read(loop_path) != state:
                    raise ValueError("stored loop artifact differs from diagnosis and ordered receipts")
                completed_loop_artifacts += 1
            elif not (
                allowed_missing_failed_receipt
                and index == len(probe_rows)
                and row["outcome"] == "unknown"
                and row["exit_code"] != 0
            ):
                raise ValueError("stored loop artifact is missing")
        replayed_loop = state
        if result.get("loop") != replayed_loop and not (
            allowed_missing_failed_receipt and result.get("loop") is None
        ):
            raise ValueError("stored loop differs from diagnosis and ordered receipts")
    loop_files = sorted(run.glob("loop_*.json"))
    expected_loop_count = completed_loop_artifacts if probe_rows else 0
    if [path.name for path in loop_files] != [f"loop_{index:02d}.json" for index in range(1, expected_loop_count + 1)]:
        raise ValueError("loop artifact inventory differs from probe receipt inventory")
    public = {
        "schema": "public_ppo_ratio_repair_probe_correction_v2",
        "probes": probe_rows,
        "corrections": corrections,
    }
    return public, replayed_loop


def _require_stage_evidence(
    run: Path,
    result: dict,
    artifacts: dict[str, dict],
    evaluations: dict[str, dict],
    model_public: dict,
    probe_public: dict,
    manifest: dict[str, str],
) -> None:
    outcome = result["outcome"]
    source_outcomes = {
        "baseline_invalid", "source_failure_absent", "model_infra_failed", "model_invalid", "diagnosis_hold",
        "loop_not_confirmed", "correction_scope_hold", "correction_not_proposed", "correction_unknown",
        "not_recovered", "recovered",
    }
    diagnosis_outcomes = source_outcomes - {"baseline_invalid", "source_failure_absent"}
    probe_outcomes = {
        "loop_not_confirmed", "correction_scope_hold", "correction_not_proposed", "correction_unknown",
        "not_recovered", "recovered",
    }
    correction_model_outcomes = {"correction_not_proposed", "correction_unknown", "not_recovered", "recovered"}
    correction_receipt_outcomes = {"correction_unknown", "not_recovered", "recovered"}
    if outcome in source_outcomes and (
        set(artifacts) < {"NONE", DEMO.FAULT}
        or set(evaluations) < {"NONE", DEMO.FAULT}
        or "source_gate" not in result
    ):
        raise ValueError("outcome is missing required source training or evaluation evidence")
    if outcome in diagnosis_outcomes and not (
        any(row["kind"] == "diagnosis" for row in model_public["dialogues"])
        or model_public["pending_record_count"]
        or model_public["transport_error_count"]
    ):
        raise ValueError("outcome is missing required diagnosis evidence")
    if outcome in probe_outcomes and not probe_public["probes"]:
        raise ValueError("outcome is missing required probe evidence")
    if outcome in correction_model_outcomes and not any(
        row["kind"] == "correction" for row in model_public["dialogues"]
    ):
        raise ValueError("outcome is missing required correction model evidence")
    if outcome in correction_receipt_outcomes and not probe_public["corrections"]:
        raise ValueError("outcome is missing required correction receipt evidence")
    if outcome in {"not_recovered", "recovered"} and "CORRECTED" not in artifacts:
        raise ValueError("outcome is missing required corrected training evidence")
    if outcome in {"not_recovered", "recovered"} and (
        "CORRECTED" not in evaluations or "recovery" not in result
    ):
        raise ValueError("outcome is missing required corrected evaluation evidence")
    if result.get("source_gate", {}).get("outcome") == "failure_present":
        reference = run / "reference/report.json"
        _source(manifest, "legacy_reference", reference)


def _validated_failure(
    run: Path,
    result: dict,
    probes: list[dict],
    corrections: list[dict],
    manifest: dict[str, str],
) -> tuple[dict | None, dict | None]:
    failure = result.get("failure")
    if failure is None:
        return None, None
    if not isinstance(failure, dict) or set(failure) != {"stage", "type", "site"}:
        raise ValueError("failure receipt field inventory differs")
    stage, error_type, site = failure["stage"], failure["type"], failure["site"]
    if (
        not isinstance(stage, str)
        or not stage
        or not isinstance(error_type, str)
        or not error_type
        or not isinstance(site, dict)
        or set(site) != {"file", "line"}
        or not isinstance(site["file"], str)
        or _ABSOLUTE.search(site["file"])
        or _DANGEROUS_PATH.search(site["file"])
        or type(site["line"]) is not int
        or site["line"] <= 0
    ):
        raise ValueError("failure receipt is malformed")
    failure_source = ROOT / site["file"]
    source_manifest_key = f"protocol_source_sha256_lf:{site['file']}"
    if (
        not failure_source.is_file()
        or source_manifest_key not in manifest
        or site["line"] > len(failure_source.read_text(encoding="utf-8").splitlines())
    ):
        raise ValueError("failure site is not bound to a registered source line")
    probe_match = re.fullmatch(r"probe_(\d+)_(P_[A-Za-z0-9_]+)", stage)
    if probe_match is not None:
        index, probe = int(probe_match.group(1)), probe_match.group(2)
        if (
            index != len(probes)
            or probes[-1]["order"] != index
            or probes[-1]["probe"] != probe
            or probes[-1]["outcome"] != "unknown"
            or probes[-1]["exit_code"] == 0
            or corrections
        ):
            raise ValueError("probe failure stage differs from its terminal receipt")
    elif stage not in {
        "source_training", "source_evaluation", "reference", "diagnosis", "correction_model",
        "correction_training", "correction_evaluation",
    }:
        raise ValueError("failure stage is outside the registered flow")
    failed_execution = None
    execution_path = run / stage / "execution.json"
    if probe_match is not None:
        _source(manifest, "failed_stage_execution", execution_path)
        execution = read(execution_path)
        wall_s = execution.get("wall_s")
        if (
            execution.get("natural_exit") is not True
            or execution.get("timed_out") is not False
            or execution.get("exit_code") != 0
            or execution.get("cleanup_verified") is not True
            or isinstance(wall_s, bool)
            or not isinstance(wall_s, (int, float))
            or not math.isfinite(wall_s)
            or wall_s < 0
        ):
            raise ValueError("failed probe execution receipt is malformed")
        report_path = run / stage / "report.json"
        _source(manifest, "failed_stage_report", report_path)
        failed_execution = {
            "stage": stage,
            "parent_wall_s": wall_s,
            "execution_receipt_sha256": manifest["failed_stage_execution"],
            "unaccepted_report_sha256": manifest["failed_stage_report"],
            "accepted_as_probe_result": False,
        }
    return failure, failed_execution


def _derive_outcome(result: dict, replayed_loop: dict | None, probe_public: dict, failure: dict | None) -> str:
    corrections = probe_public["corrections"]
    if failure is not None:
        if failure["type"] == "TimeoutError":
            return "model_infra_failed"
        if corrections and corrections[-1]["verdict"] == "unknown":
            return "correction_unknown"
        return "invalid"
    source_outcome = result.get("source_gate", {}).get("outcome")
    if source_outcome in {"baseline_invalid", "source_failure_absent"}:
        return source_outcome
    if source_outcome != "failure_present":
        return "invalid"
    diagnosis = result.get("model", {}).get("diagnosis", {}).get("review", {})
    diagnosis_status = diagnosis.get("status")
    if diagnosis_status in {"infra_error", "call_error"}:
        return "model_infra_failed"
    if diagnosis_status == "invalid":
        return "model_invalid"
    if diagnosis_status != "valid":
        return "invalid"
    decision = diagnosis.get("decision", {})
    if decision.get("action") != "continue":
        return "diagnosis_hold"
    if replayed_loop is None:
        return "invalid"
    if replayed_loop.get("status") != "confirmed":
        return "loop_not_confirmed"
    if not CONTRACT.correction_allowed(replayed_loop):
        return "correction_scope_hold"
    correction = result.get("model", {}).get("correction", {}).get("review", {})
    correction_status = correction.get("status")
    if correction_status in {"infra_error", "call_error"}:
        return "model_infra_failed"
    if correction_status == "invalid":
        return "model_invalid"
    if correction_status != "valid":
        return "invalid"
    correction_decision = correction.get("decision", {})
    if correction_decision.get("decision") != "propose" or correction_decision.get("agrees_with_rule") is not True:
        return "correction_not_proposed"
    if len(corrections) != 1:
        return "invalid"
    verdict = corrections[0]["verdict"]
    return "correction_unknown" if verdict == "unknown" else verdict


def _derive_conditions(
    result: dict,
    protocol: dict,
    preflight: dict,
    artifacts: dict[str, dict],
    evaluations: dict[str, dict],
    model_public: dict,
    probe_public: dict,
    replayed_loop: dict | None,
) -> dict[str, bool]:
    model_reviews = result.get("model", {})
    diagnosis = model_reviews.get("diagnosis", {}).get("review", {})
    correction = model_reviews.get("correction", {}).get("review", {})
    probes = probe_public["probes"]
    corrections = probe_public["corrections"]
    corrected_inspection = artifacts.get("CORRECTED", {}).get("inspection", {})
    expected_costs = set()
    if "NONE" in artifacts:
        expected_costs.add("train_NONE")
    if DEMO.FAULT in artifacts:
        expected_costs.add(f"train_{DEMO.FAULT}")
    if {"NONE", DEMO.FAULT} <= set(evaluations):
        expected_costs.add("source_evaluation")
    if result.get("source_gate", {}).get("outcome") == "failure_present":
        expected_costs.add("reference")
    expected_costs.update(
        f"probe_{row['order']}_{row['probe']}" for row in probes if row["exit_code"] == 0
    )
    if "CORRECTED" in artifacts:
        expected_costs.add("correction_training")
    if "CORRECTED" in evaluations:
        expected_costs.add("correction_evaluation")
    costs = result["costs"]
    costs_recorded = (
        bool(costs)
        and set(costs) == expected_costs
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 for value in costs.values())
        and model_public["pending_record_count"] == 0
        and model_public["transport_error_count"] == 0
        and all(
            call["input_tokens"] is not None and call["output_tokens"] is not None and call["wall_s"] is not None
            for dialogue in model_public["dialogues"]
            for call in dialogue["calls"]
        )
    )
    accepted_diagnosis = diagnosis.get("status") == "valid"
    accepted_correction = correction.get("status") == "valid"
    return {
        "committed_source_and_protocol": True,
        "fresh_seeds": preflight["freshness"]["status"] == "fresh",
        "launch_paths_checked": preflight["launch"].get("processes_started") == 0,
        "source_failure_present": result.get("source_gate", {}).get("outcome") == "failure_present",
        "accepted_fresh_diagnosis": accepted_diagnosis,
        "first_probe_from_model": bool(probes) and accepted_diagnosis
        and probes[0]["probe"] == diagnosis["decision"]["probe"],
        "approved_probe_receipts": bool(probes)
        and all(row["approval_events"] == ["proposed", "approved", "consumed", "receipt"] for row in probes),
        "optimizer_surrogate_confirmed": replayed_loop is not None
        and replayed_loop["status"] == "confirmed"
        and replayed_loop["hypotheses"] == [CONTRACT.SURROGATE],
        "accepted_fresh_correction": accepted_correction,
        "approved_correction_consumed_once": len(corrections) == 1
        and corrections[0]["approval_events"] == ["proposed", "approved", "consumed", "receipt"],
        "restored_surrogate_gradient": corrected_inspection.get("count_status") == "normal"
        and corrected_inspection.get("gradient_status") == "normal"
        and corrected_inspection.get("optimizer_steps") == [20, 20, 20]
        and corrected_inspection.get("minibatches") == [20, 20, 20],
        "corrected_fixed_eval_healthy": evaluations.get("CORRECTED", {}).get("label") == "healthy",
        "costs_recorded": costs_recorded,
    }


def _verify_conditions(result: dict, protocol: dict, derived: dict[str, bool]) -> None:
    stored = result["conditions"]
    if set(stored) - set(protocol["success_conditions"]):
        raise ValueError("stored completion condition inventory differs")
    if any(stored[key] is not derived[key] for key in stored):
        raise ValueError("stored completion condition differs from raw evidence")
    if result["outcome"] == "recovered" and (
        set(stored) != set(protocol["success_conditions"])
        or not all(derived[key] is True for key in protocol["success_conditions"])
    ):
        raise ValueError("recovered result is missing a verified success condition")
    recomputed_success = result["outcome"] == "recovered" and all(
        derived[key] is True for key in protocol["success_conditions"]
    )
    if result.get("demo_success") is not recomputed_success:
        raise ValueError("demo success differs from verified completion conditions")


def verify_public_payloads(payloads: dict[str, dict]) -> None:
    if set(payloads) != PUBLIC_FILES:
        raise ValueError("public artifact inventory differs")

    def visit(value: Any, label: str = "root") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in _FORBIDDEN_KEYS:
                    raise ValueError(f"forbidden public field: {label}.{key}")
                visit(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")
        elif isinstance(value, str) and (
            _ABSOLUTE.search(value) or _PRIVATE.search(value) or _DANGEROUS_PATH.search(value)
        ):
            raise ValueError(f"private or unsafe path in public value: {label}")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite public number: {label}")

    visit(payloads)


def serialized(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def verify_and_build(run: Path = RUN, protocol_path: Path = PROTOCOL) -> dict[str, dict]:
    protocol_path = Path(protocol_path)
    result_path = Path(run) / "result.json"
    result, protocol = read(result_path), read(protocol_path)
    manifest: dict[str, str] = {}
    _source(manifest, "result", result_path)
    if (
        result.get("schema") != "ppo_ratio_repair_demo_result_v2"
        or result.get("tag") != TAG
        or result.get("outcome") not in protocol.get("outcome_wording", {})
        or result.get("wording") != protocol["outcome_wording"][result["outcome"]]
        or not re.fullmatch(r"[0-9a-f]{40}", str(result.get("experiment_commit", "")))
    ):
        raise ValueError("stored result identity or outcome differs")
    _verify_protocol(protocol, result, manifest, protocol_path)
    preflight = _verify_begin_and_preflight(Path(run), result, protocol, manifest)
    costs = result.get("costs")
    if not isinstance(costs, dict) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in costs.values()
    ):
        raise ValueError("stored execution costs are invalid")
    conditions = result.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("stored completion conditions are not an object")
    artifacts = _training_evidence(Path(run), result, protocol, manifest)
    evaluations, _checked, _verdicts = _evaluations(Path(run), result, manifest, artifacts)
    models = _models(Path(run), result, manifest)
    probe_correction, replayed_loop = _probe_correction(Path(run), result, manifest, protocol, artifacts)
    failure, failed_execution = _validated_failure(
        Path(run), result, probe_correction["probes"], probe_correction["corrections"], manifest
    )
    derived_outcome = _derive_outcome(result, replayed_loop, probe_correction, failure)
    if result["outcome"] != derived_outcome:
        raise ValueError("stored outcome differs from raw stage evidence")
    _require_stage_evidence(
        Path(run), result, artifacts, evaluations["arms"], models, probe_correction, manifest
    )
    derived = _derive_conditions(
        result, protocol, preflight, artifacts, evaluations["arms"], models, probe_correction, replayed_loop
    )
    _verify_conditions(result, protocol, derived)
    completed_cost_total = math.fsum(float(value) for value in costs.values())
    failed_stage_cost = 0.0 if failed_execution is None else float(failed_execution["parent_wall_s"])
    partial_observations = {
        "terminal_outcome": result["outcome"],
        "source_gate": result.get("source_gate", {}).get("outcome"),
        "diagnosis_status": result.get("model", {}).get("diagnosis", {}).get("review", {}).get("status"),
        "accepted_probe_outcomes": {
            outcome: sum(row["outcome"] == outcome for row in probe_correction["probes"])
            for outcome in ("normal", "abnormal", "unknown")
        },
    }
    summary = {
        "schema": "public_ppo_ratio_repair_summary_v2",
        "status": "complete",
        "tag": result["tag"],
        "outcome": result["outcome"],
        "wording": result["wording"],
        "demo_success": result["demo_success"],
        "source_training_seed": result["seed"],
        "corrected_restart_seed": result["restart_seed"],
        "completion_conditions": conditions,
        "source_gate": result.get("source_gate"),
        "recovery": result.get("recovery"),
        "failure": failure,
        "stage_observations": partial_observations,
        "execution_costs": {
            "completed_stage_parent_wall_s": costs,
            "completed_stage_total_parent_wall_s": completed_cost_total,
            "failed_stage_execution": failed_execution,
            "known_total_parent_wall_s": completed_cost_total + failed_stage_cost,
            "scope": "부모 프로세스 벽시계 시간. 완료 단계와 실패 단계 실행 receipt를 분리한다.",
        },
        "model_request_accounting": {
            "completed": result.get("api_requests"),
            "saved": result.get("api_requests_in_saved_records"),
        },
        "interpretation": protocol["interpretation"],
        "design_coupling": protocol["design_coupling"],
        "source_commit_sha": result["experiment_commit"],
        "protocol_sha256_lf": result["protocol_sha256_lf"],
    }
    payloads = {
        "summary.json": summary,
        "evaluations.json": evaluations,
        "model_evidence.json": models,
        "probe_correction.json": probe_correction,
        "provenance.json": {
            "schema": "public_ppo_ratio_repair_provenance_v2",
            "export_contract": "allowlisted_fields_only_no_private_or_user_paths",
            "source_commit_sha": result["experiment_commit"],
            "protocol_sha256_lf": result["protocol_sha256_lf"],
            "source_artifact_sha256": manifest,
        },
    }
    verify_public_payloads(payloads)
    return payloads


def write_new(payloads: dict[str, dict], output: Path = OUTPUT, audit: Path = AUDIT) -> None:
    verify_public_payloads(payloads)
    if output.exists() or audit.exists():
        raise FileExistsError("public output or private export audit already exists")
    output.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(payloads.items()):
        path = output / relative
        path.write_bytes(serialized(payload))
    public_hashes = {relative: sha256(output / relative) for relative in sorted(payloads)}
    audit.mkdir(parents=True, exist_ok=False)
    receipt = {
            "schema": "ppo_ratio_repair_public_export_receipt_v2",
            "tag": TAG,
            "source_artifact_sha256": payloads["provenance.json"]["source_artifact_sha256"],
            "public_artifact_sha256": public_hashes,
    }
    (audit / "receipt.json").write_bytes(serialized(receipt))


def verify_export(
    output: Path = OUTPUT,
    run: Path = RUN,
    protocol_path: Path = PROTOCOL,
    audit: Path = AUDIT,
) -> dict[str, str]:
    expected = verify_and_build(run, protocol_path)
    entries = list(output.iterdir()) if output.is_dir() else []
    actual = {path.name for path in entries}
    if (
        actual != PUBLIC_FILES
        or any(not path.is_file() or path.is_symlink() for path in entries)
        or any(path.is_dir() for path in output.rglob("*"))
    ):
        raise ValueError("exported public artifact inventory differs")
    for relative, payload in expected.items():
        if (output / relative).read_bytes() != serialized(payload):
            raise ValueError(f"exported public artifact differs: {relative}")
    public_hashes = {relative: sha256(output / relative) for relative in sorted(expected)}
    audit_entries = list(audit.iterdir()) if audit.is_dir() else []
    if (
        [path.name for path in audit_entries] != ["receipt.json"]
        or audit_entries[0].is_symlink()
        or not audit_entries[0].is_file()
    ):
        raise ValueError("private export audit inventory differs")
    expected_receipt = {
        "schema": "ppo_ratio_repair_public_export_receipt_v2",
        "tag": TAG,
        "source_artifact_sha256": expected["provenance.json"]["source_artifact_sha256"],
        "public_artifact_sha256": public_hashes,
    }
    if audit_entries[0].read_bytes() != serialized(expected_receipt):
        raise ValueError("private export audit differs")
    return public_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "check":
        payloads = verify_and_build()
        value = {"status": "pass", "public_files": sorted(payloads)}
    elif args.command == "export":
        payloads = verify_and_build()
        write_new(payloads)
        value = {"status": "exported", "public_files": sorted(payloads)}
    else:
        value = {"status": "pass", "public_sha256": verify_export()}
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
