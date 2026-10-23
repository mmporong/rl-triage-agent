"""PPO ratio repair C3 결과를 v2 공개 검증 경계와 C3 GPU receipt로 재검증한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals", ROOT / "bench"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import export_ppo_ratio_repair_v2 as V2  # noqa: E402
import ppo_ratio_repair_demo_v3 as DEMO  # noqa: E402
import ppo_ratio_repair_execution_v3 as EXECUTION_V3  # noqa: E402

TAG = "ppo_ratio_repair_s2164_20261023"
RUN = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_repair/runs" / TAG
PROTOCOL = ROOT / "bench/protocols/ppo_ratio_repair_demo_v3.json"
OUTPUT = ROOT / "evals/results" / TAG
AUDIT = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_repair/exports" / TAG
PUBLIC_FILES = set(V2.PUBLIC_FILES)
EXPECTED_ARMS = {
    "NONE": (2164, "training/none", f"{TAG}_none", None),
    DEMO.FAULT: (2164, "training/inverse_ratio", f"{TAG}_inverse_ratio", None),
    "CORRECTED": (3164, "correction_training", f"{TAG}_corrected", "fault"),
}

# 테스트 fixture와 공개 scrub이 v2에서 사용한 동일 구현을 참조한다.
CONTRACT, MODEL_CALL, TRAINING = V2.CONTRACT, V2.MODEL_CALL, V2.TRAINING
REC, G, CorrectionLedger = V2.REC, V2.G, V2.CorrectionLedger
BANDS, TRAINING_MANIFEST_SCHEMA = V2.BANDS, V2.TRAINING_MANIFEST_SCHEMA
sha256, read, serialized = V2.sha256, V2.read, V2.serialized


def _isolated_v2() -> ModuleType:
    """전역 치환이 이미 import된 v2 exporter로 새지 않는 별도 module을 만든다."""
    path = ROOT / "evals/export_ppo_ratio_repair_v2.py"
    name = "_ppo_ratio_repair_export_v2_isolated"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("v2 exporter module을 격리해 읽을 수 없다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configured_base(run: Path, protocol_path: Path) -> ModuleType:
    base = _isolated_v2()
    for name, value in {
        "TAG": TAG,
        "RUN": Path(run),
        "PROTOCOL": Path(protocol_path),
        "OUTPUT": OUTPUT,
        "AUDIT": AUDIT,
        "DEMO": DEMO,
        "EXPECTED_ARMS": EXPECTED_ARMS,
    }.items():
        setattr(base, name, value)
    original_read = base.read
    result_path = (Path(run) / "result.json").resolve()
    protocol_resolved = Path(protocol_path).resolve()

    def compatible_read(path: Path) -> dict:
        value = original_read(path)
        resolved = Path(path).resolve()
        if resolved == result_path and value.get("schema") == "ppo_ratio_repair_demo_result_v3":
            value = {**value, "schema": "ppo_ratio_repair_demo_result_v2"}
        elif resolved == protocol_resolved and value.get("schema") == "ppo_ratio_repair_demo_protocol_v3":
            value = {**value, "schema": "ppo_ratio_repair_demo_protocol_v2"}
        return value

    setattr(base, "read", compatible_read)
    original_training = base._training_evidence

    def compatible_training(where: Path, result: dict, protocol: dict, manifest: dict[str, str]) -> dict[str, dict]:
        partial = (
            result.get("outcome") == "invalid"
            and result.get("failure", {}).get("stage") == "source_training"
            and (Path(where) / "training/none/manifest.json").is_file()
            and not (Path(where) / "artifact_none.json").exists()
        )
        if not partial:
            return original_training(where, result, protocol, manifest)
        _validate_partial_source_training(base, Path(where), result, protocol, manifest)
        return {}

    setattr(base, "_training_evidence", compatible_training)
    return base


def _validate_partial_source_training(
    base: ModuleType,
    run: Path,
    result: dict,
    protocol: dict,
    source_manifest: dict[str, str],
) -> None:
    """GPU 사후 gate에서 거절된 NONE 학습의 원시 산출물만 검증하고 정책 artifact로 승격하지 않는다."""
    if any((run / relative).exists() for relative in (
        "artifact_inverse_ratio.json", "artifact_corrected.json", "training/inverse_ratio", "correction_training",
    )):
        raise ValueError("source-training failure contains a later training arm")
    folder = run / "training/none"
    paths = {
        "training_manifest:NONE": folder / "manifest.json",
        "training_records:NONE": folder / "records.jsonl",
        "training_readback:NONE": folder / "readback.json",
        "training_checkpoint_receipt:NONE": folder / "checkpoint.json",
        "partial_training_execution:NONE": folder / "execution.json",
        "partial_checkpoint_execution:NONE": folder / "checkpoint_execution.json",
        "partial_extract_execution:NONE": folder / "extract_execution.json",
        "partial_training_telemetry:NONE": folder / "telemetry.json",
    }
    for logical_id, path in paths.items():
        base._source(source_manifest, logical_id, path)
    training = base.read(paths["training_manifest:NONE"])
    seed, _folder_name, run_name, _source_kind = EXPECTED_ARMS["NONE"]
    expected = {
        "schema": base.TRAINING_MANIFEST_SCHEMA,
        "mode": "train",
        "arm": "NONE",
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
    if set(training) != base.TRAINING_MANIFEST_KEYS or any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("partial training manifest binding differs")
    if training.get("source_checkpoint_sha256") is not None:
        raise ValueError("partial NONE training unexpectedly loads a checkpoint")
    if (
        Path(str(training.get("records_path", ""))).resolve() != paths["training_records:NONE"].resolve()
        or Path(str(training.get("readback_path", ""))).resolve() != paths["training_readback:NONE"].resolve()
    ):
        raise ValueError("partial training execution path differs")
    measured = base.EXECUTION.inspection(paths["training_records:NONE"], training)
    report = measured["report"]
    if (
        measured.get("count_status") != "normal"
        or measured.get("gradient_status") != "normal"
        or measured.get("first_mismatch_update") is not None
        or measured.get("gradient_batches") != 60
        or measured.get("optimizer_steps") != [20, 20, 20]
        or measured.get("minibatches") != [20, 20, 20]
        or report.get("manifest", {}).get("source_commit_sha") != result["experiment_commit"]
        or report.get("manifest", {}).get("protocol_sha256_lf") != result["protocol_sha256_lf"]
    ):
        raise ValueError("partial training records differ from registered NONE inspection")
    readback = base.TRAINING.validate_readback(base.read(paths["training_readback:NONE"]), "NONE")
    mutation = readback.get("mutation")
    if (
        readback.get("configuration") != {"learning_epochs": 5, "mini_batches": 4, "clip_param": 0.2}
        or not isinstance(mutation, dict)
        or mutation.get("arm") != "NONE"
        or mutation.get("mutation") != "inverse_importance_ratio_v1"
        or mutation.get("active_mutation") is not False
        or mutation.get("restored_original") is not False
    ):
        raise ValueError("partial training readback differs")
    checkpoint = base.read(paths["training_checkpoint_receipt:NONE"])
    first = Path(str(checkpoint.get("first", {}).get("path", ""))).resolve()
    final = Path(str(checkpoint.get("final", {}).get("path", ""))).resolve()
    if (
        checkpoint.get("device") != "cpu"
        or checkpoint.get("first", {}).get("iteration") != 0
        or checkpoint.get("final", {}).get("iteration") != 299
        or first.name != "model_0.pt"
        or final.name != "model_299.pt"
        or not first.is_file()
        or not final.is_file()
        or sha256(first) != checkpoint["first"].get("sha256")
        or sha256(final) != checkpoint["final"].get("sha256")
    ):
        raise ValueError("partial training checkpoint evidence differs")
    base._source(source_manifest, "partial_training_first_checkpoint:NONE", first)
    base._source(source_manifest, "partial_training_final_checkpoint:NONE", final)
    _partial_execution_costs(run, result)
    telemetry = base.read(paths["partial_training_telemetry:NONE"])
    summaries = telemetry.get("summary")
    if not isinstance(summaries, dict) or not summaries or any(
        not isinstance(row, dict) or row.get("n") != protocol["training_updates"]
        for row in summaries.values()
    ):
        raise ValueError("partial training telemetry differs")


def _execution_wall(path: Path) -> float:
    value = read(path)
    wall_s = value.get("wall_s")
    if (
        value.get("natural_exit") is not True
        or value.get("timed_out") is not False
        or value.get("exit_code") != 0
        or value.get("cleanup_verified") is not True
        or isinstance(wall_s, bool)
        or not isinstance(wall_s, (int, float))
        or not math.isfinite(wall_s)
        or wall_s < 0
    ):
        raise ValueError("partial training execution receipt differs")
    return float(wall_s)


def _partial_execution_costs(run: Path, result: dict) -> dict[str, float]:
    if not (
        result.get("outcome") == "invalid"
        and result.get("failure", {}).get("stage") == "source_training"
    ):
        return {}
    folder = Path(run) / "training/none"
    return {
        "owned_training_parent_wall_s": _execution_wall(folder / "execution.json"),
        "checkpoint_readback_parent_wall_s": _execution_wall(folder / "checkpoint_execution.json"),
        "telemetry_extract_parent_wall_s": _execution_wall(folder / "extract_execution.json"),
    }


def _strict_sample(sample: Any) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise ValueError("GPU raw sample is not an object")
    required = {
        "busy", "busy_reasons", "admission_policy", "gpu_exclusive", "performance_measurement",
        "utilization_percent", "memory_used_mib", "compute_process_count", "processes", "active_engines",
    }
    if (
        not required <= set(sample)
        or type(sample["busy"]) is not bool
        or not isinstance(sample["busy_reasons"], list)
        or sample["admission_policy"] != EXECUTION_V3.GPU_PROFILE
        or sample["gpu_exclusive"] is not False
        or sample["performance_measurement"] is not False
        or type(sample["utilization_percent"]) is not int
        or type(sample["memory_used_mib"]) is not int
        or type(sample["compute_process_count"]) is not int
        or not isinstance(sample["processes"], list)
        or not isinstance(sample["active_engines"], list)
    ):
        raise ValueError("GPU raw sample fields differ from the registered strict query")
    return sample


def _identity(value: Any, pid: int) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"ProcessId", "CreationDate", "SessionId"}
        or value.get("ProcessId") != pid
        or not isinstance(value.get("CreationDate"), str)
        or not value["CreationDate"]
        or type(value.get("SessionId")) is not int
        or value["SessionId"] < 0
    ):
        raise ValueError("CSRSS process identity is malformed")
    return value


def _validate_gpu_receipt(value: dict) -> str:
    samples = value.get("samples")
    bounded = value.get("bounded_resample")
    if not isinstance(samples, list) or not samples or len(samples) > EXECUTION_V3.MAX_SAMPLES:
        raise ValueError("GPU bounded sample count differs")
    samples = [_strict_sample(sample) for sample in samples]
    if not isinstance(bounded, dict) or set(bounded) != {
        "attempted", "accepted", "reason", "interval_s", "max_samples", "sample_count",
        "required_consecutive_clear", "identities",
    }:
        raise ValueError("GPU bounded receipt fields differ")
    if (
        type(bounded["attempted"]) is not bool
        or type(bounded["accepted"]) is not bool
        or not isinstance(bounded["reason"], str)
        or bounded["interval_s"] != EXECUTION_V3.RESAMPLE_INTERVAL_S
        or bounded["max_samples"] != EXECUTION_V3.MAX_SAMPLES
        or bounded["sample_count"] != len(samples)
        or bounded["required_consecutive_clear"] != 2
        or bounded["attempted"] is not (len(samples) > 1)
        or not isinstance(bounded["identities"], list)
    ):
        raise ValueError("GPU bounded receipt metadata differs")
    expected = {**samples[-1], "bounded_resample": bounded, "samples": samples}
    if value != expected:
        raise ValueError("GPU receipt is not the exact last sample projection")

    first = samples[0]
    first_memory = first["memory_used_mib"]
    # 첫 표본은 동결 classifier의 결과를 보존한다. 허용 화면 프로세스의 3D 사용률은 0이 아닐 수 있다.
    initial_clear = first["busy"] is False and first["busy_reasons"] == []
    pid = EXECUTION_V3._candidate_pid(first)
    identities = bounded["identities"]
    if initial_clear:
        if not (
            len(samples) == 1
            and identities == []
            and bounded["accepted"] is True
            and bounded["reason"] == "initial_clear"
        ):
            raise ValueError("initial-clear GPU receipt differs")
        return "initial_clear"

    if pid is None:
        if not (
            len(samples) == 1
            and identities == []
            and bounded["accepted"] is False
            and bounded["reason"] == "strict_busy_not_registered_transient"
        ):
            raise ValueError("unregistered busy GPU receipt differs")
        return "rejected"

    checked_identities = [_identity(row, pid) for row in identities]
    passed = (
        len(samples) == EXECUTION_V3.MAX_SAMPLES
        and len(checked_identities) == EXECUTION_V3.MAX_SAMPLES
        and all(row == checked_identities[0] for row in checked_identities)
        and all(EXECUTION_V3._clear_followup(sample, pid, first_memory) for sample in samples[1:])
    )
    if passed:
        if bounded["accepted"] is not True or bounded["reason"] != "two_consecutive_strict_clear":
            raise ValueError("resampled-clear GPU receipt differs")
        return "resampled_clear"
    if bounded["accepted"] is not False or bounded["reason"] in {
        "initial_clear", "two_consecutive_strict_clear", "strict_busy_not_registered_transient",
    }:
        raise ValueError("rejected CSRSS resample receipt differs")
    return "rejected"


def _required_execution_groups(run: Path, result: dict) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    costs = result.get("costs", {})
    fixed = {
        "train_NONE": run / "training/none",
        f"train_{DEMO.FAULT}": run / "training/inverse_ratio",
        "source_evaluation": run / "source_evaluation",
        "reference": run / "reference",
        "correction_training": run / "correction_training",
        "correction_evaluation": run / "correction_evaluation",
    }
    for key, folder in fixed.items():
        if key in costs:
            groups.append((key, [folder]))
    for index, row in enumerate(result.get("probe_steps", []), 1):
        key = f"probe_{index}_{row.get('probe')}"
        if key not in costs:
            continue
        folder = run / key
        candidates = [folder, folder / "none", folder / "inverse_ratio"]
        groups.append((key, candidates))
    return groups


def _verify_gpu_receipts(run: Path, result: dict) -> tuple[dict[str, str], dict[str, int]]:
    run = Path(run).resolve()
    execution_files = sorted(run.rglob("execution.json"))
    if any(path.is_symlink() or not path.is_file() for path in execution_files):
        raise ValueError("GPU execution inventory contains a non-file")
    parents = {path.parent.resolve() for path in execution_files}
    for key, candidates in _required_execution_groups(run, result):
        if not any(candidate.resolve() in parents for candidate in candidates):
            raise ValueError(f"completed stage lacks execution evidence: {key}")
    gpu_files = sorted(path for path in run.rglob("gpu_*.json"))
    expected = {parent / name for parent in parents for name in ("gpu_before.json", "gpu_after.json")}
    if set(gpu_files) != expected or any(path.is_symlink() or not path.is_file() for path in gpu_files):
        raise ValueError("GPU receipt inventory differs from execution stages")
    hashes: dict[str, str] = {}
    counts = {"initial_clear": 0, "resampled_clear": 0, "rejected": 0}
    for execution in execution_files:
        relative = execution.relative_to(run).as_posix()
        hashes[f"gpu_execution:{relative}"] = sha256(execution)
    for path in gpu_files:
        relative = path.relative_to(run).as_posix()
        status = _validate_gpu_receipt(read(path))
        counts[status] += 1
        hashes[f"gpu_receipt:{relative}"] = sha256(path)
    if result.get("outcome") != "invalid" and counts["rejected"]:
        raise ValueError("non-invalid outcome contains a rejected GPU admission receipt")
    return hashes, counts


def _v3_payloads(payloads: dict[str, dict], gpu_hashes: dict[str, str], gpu_counts: dict[str, int]) -> dict[str, dict]:
    schemas = {
        "summary.json": "public_ppo_ratio_repair_summary_v3",
        "evaluations.json": "public_ppo_ratio_repair_evaluations_v3",
        "model_evidence.json": "public_ppo_ratio_repair_model_evidence_v3",
        "probe_correction.json": "public_ppo_ratio_repair_probe_correction_v3",
        "provenance.json": "public_ppo_ratio_repair_provenance_v3",
    }
    converted = {name: {**value, "schema": schemas[name]} for name, value in payloads.items()}
    converted["summary.json"]["gpu_admission_evidence"] = {
        "validated_receipts": sum(gpu_counts.values()),
        **gpu_counts,
        "used_for_performance_claim": False,
    }
    provenance = converted["provenance.json"]
    provenance["export_contract"] = "v3_allowlisted_fields_only_no_private_or_user_paths"
    provenance["source_artifact_sha256"] = {
        **provenance["source_artifact_sha256"],
        **gpu_hashes,
    }
    V2.verify_public_payloads(converted)
    return converted


def verify_and_build(run: Path = RUN, protocol_path: Path = PROTOCOL) -> dict[str, dict]:
    run, protocol_path = Path(run), Path(protocol_path)
    original_result = read(run / "result.json")
    if original_result.get("schema") != "ppo_ratio_repair_demo_result_v3":
        raise ValueError("C3 result schema differs")
    protocol = read(protocol_path)
    if (
        protocol.get("schema") != "ppo_ratio_repair_demo_protocol_v3"
        or protocol.get("seed") != 2164
        or protocol.get("restart_seed") != 3164
        or protocol.get("model", {}).get("total_http_request_cap") != 4
        or protocol.get("model", {}).get("infra_dialogue_attempts") != 1
        or protocol.get("model", {}).get("feedback_retries") != 1
    ):
        raise ValueError("C3 protocol identity or request cap differs")
    gpu_hashes, gpu_counts = _verify_gpu_receipts(run, original_result)
    base = _configured_base(run, protocol_path)
    payloads = base.verify_and_build(run, protocol_path)
    model = payloads["model_evidence.json"]
    saved = model.get("api_requests_in_saved_records")
    if type(saved) is not int or saved < 0 or saved > 4:
        raise ValueError("C3 saved API request count exceeds the registered cap")
    for kind, part in original_result.get("model", {}).items():
        attempts = part.get("attempts") if isinstance(part, dict) else None
        if (
            kind not in {"diagnosis", "correction"}
            or not isinstance(attempts, list)
            or len(attempts) != 1
            or attempts[0].get("record") != f"{kind}.feedback1.json"
        ):
            raise ValueError("C3 model kind must use one registered dialogue")
    converted = _v3_payloads(payloads, gpu_hashes, gpu_counts)
    partial_costs = _partial_execution_costs(run, original_result)
    if partial_costs:
        execution_costs = converted["summary.json"]["execution_costs"]
        execution_costs["partial_unaccepted_stage_parent_wall_s"] = partial_costs
        execution_costs["known_total_parent_wall_s"] = math.fsum(partial_costs.values())
        execution_costs["scope"] = (
            "부모 프로세스 벽시계 시간. 결과에 수용되지 않은 source training의 확인 가능한 실행 receipt를 분리한다."
        )
        converted["summary.json"]["stage_observations"]["partial_training_artifact_accepted"] = False
        V2.verify_public_payloads(converted)
    return converted


def write_new(payloads: dict[str, dict], output: Path = OUTPUT, audit: Path = AUDIT) -> None:
    V2.verify_public_payloads(payloads)
    if output.exists() or audit.exists():
        raise FileExistsError("public output or private export audit already exists")
    output.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(payloads.items()):
        (output / relative).write_bytes(serialized(payload))
    public_hashes = {relative: sha256(output / relative) for relative in sorted(payloads)}
    audit.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "ppo_ratio_repair_public_export_receipt_v3",
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
    if (
        {path.name for path in entries} != PUBLIC_FILES
        or any(not path.is_file() or path.is_symlink() for path in entries)
        or any(path.is_dir() for path in output.rglob("*"))
    ):
        raise ValueError("exported public artifact inventory differs")
    for relative, payload in expected.items():
        if (output / relative).read_bytes() != serialized(payload):
            raise ValueError(f"exported public artifact differs: {relative}")
    public_hashes = {relative: sha256(output / relative) for relative in sorted(expected)}
    receipt_path = audit / "receipt.json"
    audit_entries = list(audit.iterdir()) if audit.is_dir() else []
    receipt = {
        "schema": "ppo_ratio_repair_public_export_receipt_v3",
        "tag": TAG,
        "source_artifact_sha256": expected["provenance.json"]["source_artifact_sha256"],
        "public_artifact_sha256": public_hashes,
    }
    if (
        [path.name for path in audit_entries] != ["receipt.json"]
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.read_bytes() != serialized(receipt)
    ):
        raise ValueError("private export audit differs")
    return public_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "check":
        value = {"status": "pass", "public_files": sorted(verify_and_build())}
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
