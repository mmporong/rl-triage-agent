"""완료된 lineage 교정 학습 artifact를 CPU TensorBoard 입력으로 봉인한다.

원본 source 수집기가 아니며 학습, GPU, 모델 호출, 승인 또는 원장 쓰기를 하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import isaac_training_guard_v1 as GUARD
import lineage_probe_backend_v1 as P
import lineage_recovery_training_v1 as TRAIN
import lineage_selection_inputs_v1 as INPUT
import residual_execution_v1 as RESIDUAL

SCHEMA = "lineage_recovery_collected_v1"
RECEIPT_SCHEMA = "lineage_recovery_tensorboard_extract_receipt_v1"
ALLOWED_ARMS = {"NONE", "FIRST", "BOTH"}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    _require(all(isinstance(row, dict) for row in rows), f"{Path(path).name} 행 형식이 잘못됐다")
    return rows


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _check_training(
    manifest_path: Path,
    evidence_dir: Path,
    owned_result_path: Path,
    lab: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = TRAIN.validate_manifest(_json(manifest_path))
    _require(manifest["mode"] == "train" and manifest["max_iterations"] == 300,
             "recovery 입력은 300 update train artifact만 허용한다")
    _require(manifest["arm"] in ALLOWED_ARMS, "recovery 입력 arm은 NONE/FIRST/BOTH만 허용한다")
    TRAIN.verify_binding(manifest, lab=lab)
    fingerprint = TRAIN.manifest_fingerprint(manifest)
    GUARD.validate_completion(evidence_dir, fingerprint)

    owned = _json(owned_result_path)
    _require(P._execution_ok(owned), "owned 학습 실행 영수증이 정상 종료 계약과 다르다")
    evidence_path = evidence_dir / GUARD.EVIDENCE
    guard_records_path = evidence_dir / GUARD.RECORDS
    evidence = _json(evidence_path)
    guard_rows = _rows(guard_records_path)
    _require(len(guard_rows) == 1, "guard records는 완료 한 행이어야 한다")
    completion = guard_rows[0]
    _require(completion.get("schema") == TRAIN.COMPLETION_SCHEMA,
             "guard completion schema가 lineage recovery wrapper와 다르다")

    remaining = list(TRAIN.RESIDUAL.FAULTS.remaining_faults(
        manifest["original_faults"], manifest["applied_corrections"]))
    common = {
        "arm": manifest["arm"], "seed": manifest["seed"],
        "original_faults": manifest["original_faults"],
        "applied_corrections": manifest["applied_corrections"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
    }
    _require(evidence.get("schema") == TRAIN.EVIDENCE_SCHEMA,
             "training evidence schema가 다르다")
    _require(all(evidence.get(key) == value for key, value in common.items()),
             "training evidence의 arm/seed/fault/correction/source checkpoint 결합이 다르다")
    _require(evidence.get("remaining_faults") == remaining,
             "training evidence remaining_faults가 재계산값과 다르다")
    _require(evidence.get("updates") == evidence.get("update_count") == 300,
             "training evidence update 수가 300과 다르다")
    _require(all(evidence.get(field) is True for field in (
        "learn_returned", "learning_completed", "recorded_before_app_close",
        "fresh_initialization")), "fresh 완료 evidence가 부족하다")
    _require(evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False,
             "checkpoint load 또는 resume가 감지됐다")
    for field, expected in {**common, "remaining_faults": remaining, "updates": 300,
                            "update_count": 300}.items():
        _require(completion.get(field) == expected,
                 f"guard completion {field}가 evidence/manifest와 다르다")
    _require(all(completion.get(field) is True for field in (
        "learn_returned", "learning_completed", "recorded_before_app_close",
        "fresh_initialization")), "guard completion의 fresh 완료 증거가 부족하다")
    _require(completion.get("checkpoint_loaded") is False
             and completion.get("resume_requested") is False,
             "guard completion에 checkpoint load 또는 resume가 있다")

    checkpoint = evidence.get("checkpoint")
    _require(isinstance(checkpoint, dict) and checkpoint == completion.get("checkpoint"),
             "evidence와 guard completion checkpoint가 다르다")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    log_dir = Path(str(evidence.get("log_dir", ""))).resolve()
    _require(_same_path(checkpoint.get("log_dir", ""), log_dir)
             and checkpoint_path.parent == log_dir and checkpoint.get("iteration") == 299,
             "최종 checkpoint 위치/log_dir/iteration 결합이 다르다")
    _require(checkpoint_path.is_file()
             and INPUT._sha(checkpoint_path) == checkpoint.get("sha256")
             and checkpoint_path.stat().st_size == checkpoint.get("size_bytes"),
             "최종 checkpoint bytes 또는 크기가 evidence와 다르다")
    configuration = RESIDUAL.V1._verify_saved_configuration(log_dir, manifest["seed"], 300)
    _require(configuration == {"seed": manifest["seed"], "max_iterations": 300,
                               "num_learning_epochs": 5, "num_mini_batches": 4},
             "저장된 fresh 학습 설정이 등록 계약과 다르다")

    residual = TRAIN._validate_residual_outputs(manifest)
    normalized_report = json.loads(json.dumps(
        residual["report"], ensure_ascii=False, allow_nan=False))
    _require(residual["remaining_faults"] == remaining
             and completion.get("residual_inspection") == normalized_report,
             "wrapper residual inspection이 guard completion과 다르다")
    _require(evidence.get("residual_records") == residual["records"]
             and evidence.get("residual_readback") == residual["readback"],
             "wrapper residual 원문 SHA가 evidence와 다르다")
    readback = RESIDUAL._validate_readback(
        _json(Path(manifest["readback_path"])), manifest)
    _require(readback.get("remaining_faults") == remaining,
             "reward/optimizer readback 의미가 remaining_faults와 다르다")
    _require(Path(manifest["records_path"]).resolve() != guard_records_path.resolve(),
             "observer records와 guard records는 서로 다른 파일이어야 한다")
    return manifest, evidence, completion, residual


def _raw_inputs(
    manifest_path: Path,
    evidence_dir: Path,
    owned_result_path: Path,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Path]:
    checkpoint = Path(evidence["checkpoint"]["path"]).resolve()
    log_dir = Path(evidence["log_dir"]).resolve()
    return {
        "manifest": manifest_path,
        "owned_result": owned_result_path,
        "training_evidence": evidence_dir / GUARD.EVIDENCE,
        "guard_records": evidence_dir / GUARD.RECORDS,
        "close_marker": evidence_dir / GUARD.CLOSE_READY,
        "checkpoint": checkpoint,
        "source_event": INPUT._event(log_dir),
        "observer_records": Path(manifest["records_path"]).resolve(),
        "readback": Path(manifest["readback_path"]).resolve(),
    }


def collect_recovery(
    manifest_path: Path,
    evidence_dir: Path,
    owned_result_path: Path,
    out: Path,
    isaac_python: Path,
    *,
    lab: Path,
) -> dict[str, Any]:
    """완료된 recovery 학습을 검증하고 CPU TensorBoard 입력으로 봉인한다."""
    manifest_path = Path(manifest_path).resolve()
    evidence_dir = Path(evidence_dir).resolve()
    owned_result_path = Path(owned_result_path).resolve()
    out = Path(out).resolve()
    isaac_python = Path(isaac_python).resolve()
    lab = Path(lab).resolve()
    _require(not out.exists(), "out은 존재하지 않는 fresh 경로여야 한다")
    _require(isaac_python.is_file(), "Isaac Python 실행 파일이 없다")
    manifest, evidence, _completion, residual = _check_training(
        manifest_path, evidence_dir, owned_result_path, lab)
    raw = _raw_inputs(manifest_path, evidence_dir, owned_result_path, manifest, evidence)
    before = {name: INPUT._sha(path) for name, path in raw.items()}

    out.mkdir(parents=True, exist_ok=False)
    event_copy = out / "recovery.events.out.tfevents.raw"
    shutil.copyfile(raw["source_event"], event_copy)
    _require(INPUT._sha(event_copy) == before["source_event"],
             "복사한 TensorBoard event SHA가 원문과 다르다")
    telemetry = out / "telemetry.json"
    argv = [str(isaac_python), str(INPUT.EXTRACTOR),
            str(Path(evidence["log_dir"]).resolve()), str(telemetry)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "argv": argv, "cpu_only": True,
        "timeout_s": INPUT.EXTRACT_TIMEOUT_S,
        "event_sha256_before": before["source_event"],
        "extractor_sha256_lf": INPUT._sha(INPUT.EXTRACTOR, lf=True),
        "collector_sha256_lf": INPUT._sha(Path(__file__), lf=True),
        "wrapper_sha256_lf": INPUT._sha(Path(TRAIN.__file__), lf=True),
    }
    try:
        completed = INPUT.subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="strict",
            check=False, env=environment,
            creationflags=getattr(INPUT.subprocess, "CREATE_NO_WINDOW", 0),
            timeout=INPUT.EXTRACT_TIMEOUT_S,
        )
        stdout, stderr = completed.stdout, completed.stderr
        receipt.update(returncode=completed.returncode, error_type=None)
    except Exception as exc:
        stdout = INPUT._subprocess_text(getattr(exc, "stdout", None))
        stderr = INPUT._subprocess_text(getattr(exc, "stderr", None))
        receipt.update(returncode=None, error_type=type(exc).__name__)
        receipt["raw_sha256_after"] = {
            name: INPUT._sha(path) if path.is_file() else None for name, path in raw.items()}
        INPUT._new_text(out / "extract.stdout.txt", stdout)
        INPUT._new_text(out / "extract.stderr.txt", stderr)
        INPUT._new_json(out / "extract.receipt.json", receipt)
        raise
    INPUT._new_text(out / "extract.stdout.txt", stdout)
    INPUT._new_text(out / "extract.stderr.txt", stderr)
    after = {name: INPUT._sha(path) if path.is_file() else None for name, path in raw.items()}
    receipt["raw_sha256_after"] = after
    if completed.returncode != 0 or not telemetry.is_file():
        INPUT._new_json(out / "extract.receipt.json", receipt)
        raise RuntimeError("TensorBoard 추출 subprocess가 성공하지 않았다")
    _require(after == before, "TensorBoard 추출 도중 recovery 원문이 바뀌었다")
    telemetry_value = _json(telemetry)
    _require(isinstance(telemetry_value.get("summary"), dict), "telemetry summary가 없다")
    telemetry_sha = INPUT._sha(telemetry)
    receipt["telemetry_sha256"] = telemetry_sha
    INPUT._new_json(out / "extract.receipt.json", receipt)

    source_binding = {
        "manifest_fingerprint": TRAIN.manifest_fingerprint(manifest),
        **{f"{name}_sha256": digest for name, digest in before.items()},
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "telemetry_sha256": telemetry_sha,
        "collector_sha256_lf": receipt["collector_sha256_lf"],
        "wrapper_sha256_lf": receipt["wrapper_sha256_lf"],
        "extractor_sha256_lf": receipt["extractor_sha256_lf"],
    }
    source_map = {
        "lab": str(lab), "manifest": str(manifest_path),
        "evidence_dir": str(evidence_dir), "owned_result": str(owned_result_path),
        **{name: str(path) for name, path in raw.items() if name not in {
            "manifest", "owned_result"}},
        "event_copy": str(event_copy), "telemetry": str(telemetry),
        "extract_receipt": str((out / "extract.receipt.json").resolve()),
    }
    collected = {
        "schema": SCHEMA, "source_binding": source_binding,
        "private_source_map": source_map,
        "training": {
            "arm": manifest["arm"], "seed": manifest["seed"],
            "run_name": manifest["run_name"],
            "training_kind": "healthy_reference" if manifest["arm"] == "NONE"
            else "cumulative_correction",
            "original_faults": manifest["original_faults"],
            "applied_corrections": manifest["applied_corrections"],
            "remaining_faults": residual["remaining_faults"],
            "fresh_initialization": True, "checkpoint_loaded": False,
            "resume_requested": False,
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "source_checkpoint_role": "provenance_metadata_only_not_loaded",
        },
        "scope": {
            "activity": "recovery_training_artifact_cpu_materialization_only",
            "evidence_scope": "completed_wrapper_artifact_validation_not_training_execution",
            "approval": False, "execution_allowed": False,
            "excluded": ["gpu", "training", "model_packet", "epoch2",
                         "correction_approval", "model_call", "ledger_write"],
        },
    }
    INPUT._new_json(out / "collected.json", collected)
    return deepcopy(collected)


def checked_recovery(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """봉인된 recovery 입력과 모든 private 원문·코드 SHA를 다시 검증한다."""
    value = _json(Path(path).resolve())
    _require(value.get("schema") == SCHEMA, "recovery collected schema가 다르다")
    binding = value.get("source_binding")
    source = value.get("private_source_map")
    training = value.get("training")
    scope = value.get("scope")
    _require(all(isinstance(item, dict) for item in (binding, source, training, scope)),
             "recovery collected 하위 계약이 없다")
    assert isinstance(binding, dict) and isinstance(source, dict)
    assert isinstance(training, dict) and isinstance(scope, dict)
    manifest, evidence, _completion, residual = _check_training(
        Path(source["manifest"]), Path(source["evidence_dir"]),
        Path(source["owned_result"]), Path(source["lab"]))
    raw = _raw_inputs(Path(source["manifest"]), Path(source["evidence_dir"]),
                      Path(source["owned_result"]), manifest, evidence)
    for name, raw_path in raw.items():
        _require(_same_path(raw_path, source[name]), f"private source map {name} 경로가 다르다")
        _require(INPUT._sha(raw_path) == binding.get(f"{name}_sha256"),
                 f"recovery {name} 원문이 바뀌었다")
    _require(INPUT._sha(Path(source["event_copy"])) == binding["source_event_sha256"],
             "봉인 event 복사본이 바뀌었다")
    _require(TRAIN.manifest_fingerprint(manifest) == binding.get("manifest_fingerprint"),
             "manifest fingerprint가 다르다")
    _require(binding.get("source_checkpoint_sha256") == manifest["source_checkpoint_sha256"],
             "source checkpoint provenance가 다르다")
    code = {"collector_sha256_lf": Path(__file__),
            "wrapper_sha256_lf": Path(TRAIN.__file__),
            "extractor_sha256_lf": INPUT.EXTRACTOR}
    for field, code_path in code.items():
        _require(INPUT._sha(code_path, lf=True) == binding.get(field), f"{field} source가 바뀌었다")
    telemetry = _json(Path(source["telemetry"]))
    _require(INPUT._sha(Path(source["telemetry"])) == binding.get("telemetry_sha256"),
             "telemetry 원문이 바뀌었다")
    receipt = _json(Path(source["extract_receipt"]))
    expected_after = {name: binding[f"{name}_sha256"] for name in raw}
    _require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("cpu_only") is True
             and receipt.get("timeout_s") == INPUT.EXTRACT_TIMEOUT_S
             and receipt.get("returncode") == 0 and receipt.get("error_type") is None
             and receipt.get("event_sha256_before") == binding["source_event_sha256"]
             and receipt.get("raw_sha256_after") == expected_after
             and receipt.get("telemetry_sha256") == binding["telemetry_sha256"]
             and all(receipt.get(field) == binding[field] for field in code),
             "recovery extract receipt 결합이 다르다")
    expected_training = {
        "arm": manifest["arm"], "seed": manifest["seed"], "run_name": manifest["run_name"],
        "training_kind": "healthy_reference" if manifest["arm"] == "NONE"
        else "cumulative_correction",
        "original_faults": manifest["original_faults"],
        "applied_corrections": manifest["applied_corrections"],
        "remaining_faults": residual["remaining_faults"], "fresh_initialization": True,
        "checkpoint_loaded": False, "resume_requested": False,
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "source_checkpoint_role": "provenance_metadata_only_not_loaded",
    }
    _require(training == expected_training, "recovery training projection이 원문과 다르다")
    _require(scope.get("approval") is False and scope.get("execution_allowed") is False,
             "CPU materialization scope가 실행 또는 승인을 허용한다")
    return value, telemetry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    for name in ("manifest", "evidence-dir", "owned-result", "out", "isaac-python", "lab"):
        collect.add_argument(f"--{name}", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--collected", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect_recovery(args.manifest, args.evidence_dir, args.owned_result, args.out,
                         args.isaac_python, lab=args.lab)
    else:
        checked_recovery(args.collected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
