# pyright: reportMissingImports=false
"""R4 최종 outcome에 따라 recovered target만 same-Play로 순차 녹화한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_five_policy_recovery_v1 as EXPORT
import lineage_gpu_admission_v1 as GPU
import lineage_source_batch_v1 as LEASE
import recovery_same_play_capture_v1 as CAPTURE

B = CAPTURE.V1
CONFIG_SCHEMA = "recovery_same_play_capture_config_v1"
RESULT_SCHEMA = "recovery_same_play_capture_batch_result_v1"


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def bound(ref: object, label: str) -> Path:
    require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}, f"{label} ref")
    path = Path(str(ref["path"])).resolve(strict=True)
    require(B._sha(path) == ref["sha256"], f"{label} SHA")
    return path


def _export_files(folder: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = B._read(folder / "manifest.json")
    require(manifest.get("schema") == EXPORT.MANIFEST_SCHEMA and manifest.get("status") == "complete",
            "R4 export manifest")
    values = []
    for name in ("summary.json", "cases.json", "provenance.json"):
        path = folder / name
        ref = manifest.get("files", {}).get(name)
        require(path.is_file() and isinstance(ref, Mapping) and B._sha(path) == ref.get("sha256")
                and path.stat().st_size == ref.get("size_bytes"), f"R4 export {name}")
        values.append(B._read(path))
    return values[0], values[1], values[2]


def _final_checkpoint(row: Mapping[str, Any], result_path: Path, *, second: bool) -> tuple[Path, str]:
    result = B._read(result_path)
    receipt = B._read(bound(result.get("receipt"), "recovery receipt"))
    require(result.get("status") == "complete" and result.get("outcome") == "recovered"
            and receipt.get("status") == "recovered", "recovered result/receipt")
    training = receipt.get("training")
    require(isinstance(training, Mapping), "recovery training")
    checkpoint_ref = training.get("checkpoint")
    require(isinstance(checkpoint_ref, Mapping), "recovery checkpoint ref")
    checkpoint = Path(str(checkpoint_ref.get("path", ""))).resolve(strict=True)
    sha = str(checkpoint_ref.get("sha256", ""))
    require(checkpoint.is_file() and not checkpoint.is_symlink() and B._sha(checkpoint) == sha,
            "final checkpoint raw")
    expected = (row.get("second_training", {}).get("new_checkpoint_sha256") if second
                else row.get("first_new_checkpoint_sha256"))
    require(sha == expected, "export/final checkpoint SHA")
    return checkpoint, sha


def validated_targets(design_path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    design = CAPTURE.validate_design(design_path)
    first_path = bound(config.get("first_result"), "first raw result")
    second_ref = config.get("second_result")
    second_path = bound(second_ref, "second raw result") if second_ref is not None else None
    export_dir = Path(str(config.get("recovery_export", ""))).resolve(strict=True)
    summary, cases, provenance = _export_files(export_dir)
    require(summary.get("schema") == EXPORT.SUMMARY_SCHEMA and summary.get("scope") == "actual_backend",
            "actual R4 export")
    require(cases.get("schema") == EXPORT.CASES_SCHEMA, "R4 cases schema")
    require(provenance.get("schema") == EXPORT.PROVENANCE_SCHEMA
            and provenance.get("first_result", {}).get("sha256") == B._sha(first_path)
            and ((second_path is None and provenance.get("second_result") is None)
                 or (second_path is not None and provenance.get("second_result", {}).get("sha256") == B._sha(second_path))),
            "R4 raw/export 결합")
    for key, provenance_key in (("first_protocol", "first_protocol"),
                                ("prior_conditional_design", "conditional_design")):
        relative = design["recovery_inputs"][key]
        expected_path = (ROOT / relative).resolve(strict=True)
        public_ref = provenance.get(provenance_key)
        require(isinstance(public_ref, Mapping) and public_ref.get("path") == relative
                and public_ref.get("sha256") == B._sha(expected_path),
                f"R4 export {provenance_key} 고정 원문")
    first = B._read(first_path)
    second = B._read(second_path) if second_path else None
    require(first.get("schema") == EXPORT.PARENT.FIRST_RESULT_SCHEMA and first.get("status") == "complete",
            "first raw parent")
    require(second is None or (second.get("schema") == EXPORT.PARENT.SECOND_RESULT_SCHEMA
                               and second.get("status") == "complete"), "second raw parent")
    first_rows = {f"{r['case_id']}:{r['policy']}": r for r in first.get("records", [])}
    second_rows = ({f"{r['case_id']}:{r['policy']}": r for r in second.get("records", [])}
                   if second else {})
    public_rows = {f"{r['case_id']}:{r['policy']}": r for r in cases.get("rows", [])}
    require(set(design["target_order"]) <= set(first_rows) and set(design["target_order"]) <= set(public_rows),
            "선택 target 누락")
    selected = []
    for key in design["target_order"]:
        registered = design["targets"][key]
        raw_first, public = first_rows[key], public_rows[key]
        require(public.get("source_arm") == registered["source_arm"]
                and public.get("original_source_checkpoint_sha256")
                == public.get("first_training", {}).get("original_source_checkpoint_sha256",
                                                        public.get("original_source_checkpoint_sha256")),
                "original checkpoint/source arm")
        first_outcome = raw_first.get("outcome")
        require(first_outcome == public.get("first_outcome") and first_outcome in {"recovered", "not_recovered", "HOLD"},
                "first outcome")
        base = {"key": key, "case_id": registered["case_id"], "policy": "logs_only",
                "source_arm": registered["source_arm"],
                "original_checkpoint_sha256": public["original_source_checkpoint_sha256"],
                "input_pair": registered["source_pair"]}
        if first_outcome == "recovered":
            require(raw_first.get("registered_seed") == registered["first_seed"]
                    and public.get("final_outcome") == "recovered", "first recovered seed/outcome")
            cp, sha = _final_checkpoint(public, bound(raw_first.get("result"), "first result"), second=False)
            selected.append({**base, "status": "registered_recovered", "final_epoch": "first",
                             "registered_seed": registered["first_seed"],
                             "checkpoint_path": str(cp), "checkpoint_sha256": sha})
        elif first_outcome == "not_recovered":
            require(second is not None and key in second_rows, "not_recovered 뒤 second 결과")
            raw_second = second_rows[key]
            final = raw_second.get("outcome")
            require(final == public.get("final_outcome") and final in {"recovered", "not_recovered", "HOLD"},
                    "second final outcome")
            if final == "recovered":
                require(raw_second.get("registered_seed") == registered["second_seed"], "second recovered seed")
                cp, sha = _final_checkpoint(public, bound(raw_second.get("result"), "second result"), second=True)
                selected.append({**base, "status": "registered_recovered", "final_epoch": "second",
                                 "registered_seed": registered["second_seed"],
                                 "checkpoint_path": str(cp), "checkpoint_sha256": sha})
            else:
                selected.append({**base, "status": "HOLD", "final_outcome": final,
                                 "replacement_target": False})
        else:
            selected.append({**base, "status": "HOLD", "final_outcome": "HOLD",
                             "replacement_target": False})
    return selected


def validate_input_pairs(config: Mapping[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    batch = B._read(bound(config.get("input_capture_batch"), "R2 capture batch"))
    source_index_path = bound(config.get("input_source_index"), "R2 source index")
    source_index = B._read(source_index_path)
    require(batch.get("schema") == "lineage_same_play_input_batch_result_v1"
            and batch.get("status") == "complete" and batch.get("source_artifacts") == 20
            and len(batch.get("pairs", {})) == 10, "R2 actual batch")
    require(source_index.get("schema") == "lineage_source_inputs_v2"
            and source_index.get("source_count") == 20
            and set(source_index.get("sources", {})) == {f"source-{i:02d}" for i in range(1, 21)},
            "R2 actual source index")
    pairs = {}
    for target in targets:
        case = target["case_id"]
        pair_path = bound(batch["pairs"].get(case), f"R2 {case} pair")
        pair = B._read(pair_path)
        expected = target["input_pair"]
        artifacts = pair.get("provenance", {}).get("artifacts", {})
        require(pair.get("schema") == "lineage_same_play_pair_inputs_result_v1"
                and pair.get("status") == "prepared" and pair.get("replacement_capture") is False
                and pair.get("provenance", {}).get("source_index") == B._ref(source_index_path)
                and artifacts.get("reference", {}).get("value", {}).get("source_id") == expected["reference"]
                and artifacts.get("case", {}).get("value", {}).get("source_id") == expected["case"]
                and artifacts["reference"]["value"].get("checkpoint_sha256")
                    == source_index["sources"][expected["reference"]]["checkpoint_sha256"]
                and artifacts["case"]["value"].get("checkpoint_sha256")
                    == source_index["sources"][expected["case"]]["checkpoint_sha256"],
                f"R2 {case} source pair")
        pairs[case] = B._ref(pair_path)
    return pairs


def prepare_config(design_path: Path, first_result: Path, recovery_export: Path,
                   input_capture_batch: Path, input_source_index: Path, lab: Path,
                   isaac_python: Path, out: Path, *, second_result: Path | None = None) -> dict[str, Any]:
    CAPTURE.validate_design(design_path)
    require(not Path(out).exists() and Path(isaac_python).is_file() and Path(lab).is_dir(),
            "fresh config/Isaac launcher/lab")
    config: dict[str, Any] = {
        "schema": CONFIG_SCHEMA, "design": B._ref(design_path),
        "first_result": B._ref(first_result),
        "second_result": B._ref(second_result) if second_result is not None else None,
        "recovery_export": str(Path(recovery_export).resolve(strict=True)),
        "input_capture_batch": B._ref(input_capture_batch),
        "input_source_index": B._ref(input_source_index),
        "lab": str(Path(lab).resolve()), "isaac_python": str(Path(isaac_python).resolve()),
    }
    targets = validated_targets(design_path, config)
    config["input_pairs"] = validate_input_pairs(config, targets)
    B._write_new(out, config)
    return config


def preflight(design_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    design, config = CAPTURE.validate_design(design_path), B._read(config_path)
    require(config.get("schema") == CONFIG_SCHEMA and config.get("design") == B._ref(design_path),
            "private config/design")
    require(Path(str(config.get("isaac_python", ""))).is_file(), "Isaac launcher")
    targets = validated_targets(design_path, config)
    pairs = validate_input_pairs(config, targets)
    require(config.get("input_pairs") == pairs, "R2 pair 사전등록")
    recovered = [row for row in targets if row["status"] == "registered_recovered"]
    require(len(recovered) <= 2 and (0 if not recovered else 1 + len(recovered)) <= design["budget"]["sdk_cap"],
            "SDK batch cap")
    return design, config, targets


def execute(design_path: Path, config_path: Path, source_anchor: str,
            execution_anchor: str, out: Path) -> dict[str, Any]:
    design, config, targets = preflight(design_path, config_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = LEASE.lease_context()
    require(initial.get("owner") == LEASE.OWNER, "capture batch lease owner")
    B._write_new(out / "lease-initial.json", initial)
    started, sdk, seen = time.monotonic(), 0, set()
    recovered = [row for row in targets if row["status"] == "registered_recovered"]

    def boundary() -> None:
        current = LEASE.lease_context()
        require(current.get("owner") == initial.get("owner")
                and current.get("leaseId") == initial.get("leaseId")
                and time.monotonic() - started <= design["budget"]["wall_cap_s"]
                and B._ref(design_path) == config["design"], "capture batch source/time/lease boundary")

    def record(target: dict[str, Any], mode: str) -> None:
        nonlocal sdk
        boundary()
        reservation = (target["key"], mode)
        require(reservation not in seen and sdk < design["budget"]["sdk_cap"], "중복 또는 초과 SDK 예약")
        seen.add(reservation)
        sdk += 1
        B._write_new(out / "sdk-intents" / f"{sdk:02d}.json", {"target": target["key"], "mode": mode})
        GPU.snapshot(out / "gpu-admission" / f"{sdk:02d}")
        stage = out / ("smoke" if mode == "smoke" else target["case_id"])
        CAPTURE.execute(design_path, target, source_anchor, execution_anchor, stage,
                        Path(config["isaac_python"]), mode=mode)

    try:
        if recovered:
            record(recovered[0], "smoke")
            for target in recovered:
                record(target, "full")
        result = {"schema": RESULT_SCHEMA, "status": "technical_complete" if recovered else "HOLD",
                  "sdk_intents": sdk, "api": 0, "recovered_targets": len(recovered),
                  "targets": targets, "input_pairs": config["input_pairs"],
                  "fault_reference_recaptures": 0, "replacement_targets": 0,
                  "capture_approved": False, "model_ready": False, "used_for_decision": False}
        B._write_new(out / "result.json", result)
        return result
    except BaseException as error:
        B._write_new(out / "failure.json", {"status": "invalid", "error_type": type(error).__name__,
                                            "message": str(error), "sdk_intents": sdk,
                                            "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("protocol", "first-result", "recovery-export", "input-capture-batch",
                 "input-source-index", "lab", "isaac-python", "out"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--second-result", type=Path)
    for command in ("preflight", "execute"):
        child = sub.add_parser(command)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
        if command == "execute":
            child.add_argument("--source-anchor", required=True)
            child.add_argument("--execution-anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_config(args.protocol, args.first_result, args.recovery_export,
                       args.input_capture_batch, args.input_source_index, args.lab,
                       args.isaac_python, args.out, second_result=args.second_result)
    elif args.command == "preflight":
        _design, _config, targets = preflight(args.protocol, args.config)
        B._write_new(args.out, {"status": "complete", "gpu": 0, "api": 0, "targets": targets})
    else:
        execute(args.protocol, args.config, args.source_anchor, args.execution_anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
