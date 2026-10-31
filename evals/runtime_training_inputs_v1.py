"""검증된 300-update 학습을 telemetry와 runtime fact 입력으로 봉인한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import lineage_recovery_inputs_v1 as RECOVERY
import lineage_recovery_training_v1 as TRAIN
import lineage_selection_inputs_v1 as INPUT
import optimizer_runtime_facts_v1 as FACT
import residual_selection_v2 as SELECTION

SCHEMA = "runtime_training_collected_v1"
METADATA_SCHEMA = "runtime_training_collected_metadata_v1"
PACKET_SCHEMA = "runtime_training_selection_packet_v1"
RECEIPT_SCHEMA = "runtime_training_extract_receipt_v1"
PURPOSE = "registered_training_observation"
COLLECTED = "collected.json"
METADATA = "metadata.json"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"비유한 JSON 숫자는 허용하지 않는다: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _new(path: Path, value: object) -> None:
    _require(not path.exists(), f"{path.name}은 새 파일이어야 한다")
    path.write_bytes(_canonical(value))


def _raw(
    manifest_path: Path, evidence_dir: Path, owned_result_path: Path,
    manifest: dict[str, Any], evidence: dict[str, Any],
) -> dict[str, Path]:
    return RECOVERY._raw_inputs(
        manifest_path, evidence_dir, owned_result_path, manifest, evidence)


def collect(
    manifest_path: Path,
    evidence_dir: Path,
    owned_result_path: Path,
    *,
    lab: Path,
    isaac_python: Path,
    out: Path,
) -> dict[str, Any]:
    """등록 학습 원문을 검증하고 새 목적의 CPU 입력을 write-once로 만든다."""
    manifest_path, evidence_dir = Path(manifest_path).resolve(), Path(evidence_dir).resolve()
    owned_result_path, lab = Path(owned_result_path).resolve(), Path(lab).resolve()
    isaac_python, out = Path(isaac_python).resolve(), Path(out).resolve()
    _require(not out.exists(), "out은 존재하지 않는 fresh 경로여야 한다")
    _require(isaac_python.is_file(), "Isaac Python 실행 파일이 없다")
    manifest, evidence, _completion, residual = FACT._check_registered_training(
        manifest_path, evidence_dir, owned_result_path, lab)
    raw = _raw(manifest_path, evidence_dir, owned_result_path, manifest, evidence)
    before = {name: _sha(path) for name, path in raw.items()}

    out.mkdir(parents=True, exist_ok=False)
    event_copy = out / "events.out.tfevents.raw"
    shutil.copyfile(raw["source_event"], event_copy)
    _require(_sha(event_copy) == before["source_event"], "event 사본 SHA가 원문과 다르다")
    telemetry_path = out / "telemetry.json"
    argv = [str(isaac_python), str(INPUT.EXTRACTOR),
            str(Path(evidence["log_dir"]).resolve()), str(telemetry_path)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    completed = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="strict", check=False,
        env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=INPUT.EXTRACT_TIMEOUT_S)
    (out / "extract.stdout.txt").write_text(completed.stdout, encoding="utf-8", newline="")
    (out / "extract.stderr.txt").write_text(completed.stderr, encoding="utf-8", newline="")
    after = {name: _sha(path) if path.is_file() else None for name, path in raw.items()}
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "argv": argv, "cpu_only": True,
        "timeout_s": INPUT.EXTRACT_TIMEOUT_S, "returncode": completed.returncode,
        "raw_sha256_before": before, "raw_sha256_after": after,
        "extractor_sha256_lf": _sha(INPUT.EXTRACTOR, lf=True),
        "collector_sha256_lf": _sha(Path(__file__), lf=True),
    }
    if completed.returncode != 0 or not telemetry_path.is_file():
        _new(out / "extract.receipt.json", receipt)
        raise RuntimeError("TensorBoard 추출 subprocess가 성공하지 않았다")
    _require(after == before, "telemetry 추출 중 학습 원문이 바뀌었다")
    telemetry = _json(telemetry_path)
    _require(isinstance(telemetry.get("summary"), dict), "telemetry summary가 없다")
    receipt["telemetry_sha256"] = _sha(telemetry_path)
    _new(out / "extract.receipt.json", receipt)

    checkpoint_sha = str(evidence["checkpoint"]["sha256"])
    binding = {
        "purpose": PURPOSE,
        "manifest_fingerprint": TRAIN.manifest_fingerprint(manifest),
        "checkpoint_sha256": checkpoint_sha,
        "telemetry_sha256": receipt["telemetry_sha256"],
        "raw_source_sha256": before,
        "collector_sha256_lf": receipt["collector_sha256_lf"],
        "extractor_sha256_lf": receipt["extractor_sha256_lf"],
    }
    metadata = {
        "schema": METADATA_SCHEMA, "purpose": PURPOSE,
        "lab": str(lab), "manifest": str(manifest_path), "evidence_dir": str(evidence_dir),
        "owned_result": str(owned_result_path), "telemetry": str(telemetry_path),
        "event_copy": str(event_copy), "extract_receipt": str(out / "extract.receipt.json"),
        "raw_sources": {name: str(path) for name, path in raw.items()},
        "binding": binding,
        "private_training": {
            "arm": manifest["arm"], "seed": manifest["seed"],
            "original_faults": manifest["original_faults"],
            "applied_corrections": manifest["applied_corrections"],
            "remaining_faults": residual["remaining_faults"],
        },
    }
    _new(out / METADATA, metadata)
    collected = {
        "schema": SCHEMA,
        "source_binding": {
            "purpose": PURPOSE, "checkpoint_sha256": checkpoint_sha,
            "telemetry_sha256": receipt["telemetry_sha256"],
            "binding_digest": hashlib.sha256(_canonical(binding)).hexdigest(),
        },
        "metadata_ref": {"file": METADATA, "sha256": _sha(out / METADATA)},
        "scope": {"activity": "cpu_training_input_materialization_only",
                  "approval": False, "execution_allowed": False},
    }
    _new(out / COLLECTED, collected)
    return deepcopy(collected)


def checked(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """봉인 입력, private metadata, 원문, event 사본과 telemetry를 다시 검증한다."""
    path = Path(path).resolve()
    value = _json(path)
    _require(value.get("schema") == SCHEMA and set(value) == {
        "schema", "source_binding", "metadata_ref", "scope"}, "collected shape가 다르다")
    ref = value.get("metadata_ref")
    _require(isinstance(ref, dict) and ref.get("file") == METADATA, "metadata ref가 다르다")
    ref = cast(dict[str, Any], ref)
    metadata_path = path.parent / METADATA
    _require(metadata_path.is_file() and _sha(metadata_path) == ref.get("sha256"),
             "metadata bytes가 ref와 다르다")
    metadata = _json(metadata_path)
    _require(metadata.get("schema") == METADATA_SCHEMA and metadata.get("purpose") == PURPOSE,
             "metadata schema 또는 purpose가 다르다")
    manifest, evidence, _completion, residual = FACT._check_registered_training(
        Path(metadata["manifest"]), Path(metadata["evidence_dir"]),
        Path(metadata["owned_result"]), Path(metadata["lab"]))
    raw = _raw(Path(metadata["manifest"]), Path(metadata["evidence_dir"]),
               Path(metadata["owned_result"]), manifest, evidence)
    _require(metadata.get("raw_sources") == {name: str(item) for name, item in raw.items()},
             "private raw source 경로가 다르다")
    binding = metadata.get("binding")
    _require(isinstance(binding, dict), "metadata binding이 없다")
    binding = cast(dict[str, Any], binding)
    expected_raw = {name: _sha(item) for name, item in raw.items()}
    _require(binding.get("raw_source_sha256") == expected_raw, "학습 원문 SHA가 바뀌었다")
    _require(_sha(Path(metadata["event_copy"])) == expected_raw["source_event"],
             "event 사본이 원문과 다르다")
    telemetry_path = Path(metadata["telemetry"])
    _json(telemetry_path)
    _require(_sha(telemetry_path) == binding.get("telemetry_sha256"), "telemetry가 바뀌었다")
    _require(_sha(Path(__file__), lf=True) == binding.get("collector_sha256_lf")
             and _sha(INPUT.EXTRACTOR, lf=True) == binding.get("extractor_sha256_lf"),
             "collector 또는 extractor source가 바뀌었다")
    receipt = _json(Path(metadata["extract_receipt"]))
    _require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("returncode") == 0
             and receipt.get("cpu_only") is True and receipt.get("raw_sha256_before") == expected_raw
             and receipt.get("raw_sha256_after") == expected_raw
             and receipt.get("telemetry_sha256") == binding.get("telemetry_sha256"),
             "extract receipt가 원문과 다르다")
    checkpoint_sha = str(evidence["checkpoint"]["sha256"])
    _require(binding.get("purpose") == PURPOSE
             and binding.get("checkpoint_sha256") == checkpoint_sha
             and binding.get("manifest_fingerprint") == TRAIN.manifest_fingerprint(manifest),
             "source binding이 현재 학습 원문과 다르다")
    _require(metadata.get("private_training") == {
        "arm": manifest["arm"], "seed": manifest["seed"],
        "original_faults": manifest["original_faults"],
        "applied_corrections": manifest["applied_corrections"],
        "remaining_faults": residual["remaining_faults"]}, "private training projection이 다르다")
    expected_public = {
        "purpose": PURPOSE, "checkpoint_sha256": checkpoint_sha,
        "telemetry_sha256": binding["telemetry_sha256"],
        "binding_digest": hashlib.sha256(_canonical(binding)).hexdigest(),
    }
    _require(value.get("source_binding") == expected_public, "공개 source binding이 다르다")
    _require(value.get("scope") == {"activity": "cpu_training_input_materialization_only",
                                    "approval": False, "execution_allowed": False},
             "collected가 승인 또는 실행 권한을 만들었다")
    return deepcopy(value), deepcopy(metadata)


def make_packet(
    case_path: Path, reference_path: Path, *, case_runtime_path: Path,
    reference_runtime_path: Path | None, budget_left: int = 3,
    remaining_candidates: list[str] | None = None,
    applied_corrections: list[str] | None = None, prior_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """telemetry와 checkpoint 결합 runtime fact로 모델 선택 packet을 만든다."""
    case, case_meta = checked(case_path)
    reference, reference_meta = checked(reference_path)
    case_telemetry = _json(Path(case_meta["telemetry"]))
    reference_telemetry = _json(Path(reference_meta["telemetry"]))
    case_facts, case_coverage = INPUT._telemetry_projection("case", case_telemetry)
    reference_facts, reference_coverage = INPUT._telemetry_projection("reference", reference_telemetry)
    case_cp = case["source_binding"]["checkpoint_sha256"]
    reference_cp = reference["source_binding"]["checkpoint_sha256"]
    _case_envelope, case_runtime = FACT.checked(
        case_runtime_path, expected_checkpoint_sha256=case_cp)
    if reference_runtime_path is None:
        reference_runtime = FACT.unavailable_facts("reference")
        reference_runtime_ref = None
    else:
        _reference_envelope, reference_runtime = FACT.checked(
            reference_runtime_path, expected_checkpoint_sha256=reference_cp)
        reference_runtime_ref = {"path": str(Path(reference_runtime_path).resolve()),
                                 "sha256": _sha(reference_runtime_path)}
    facts = {**case_facts, **reference_facts, **case_runtime, **reference_runtime}
    applied = list(applied_corrections or [])
    packet = SELECTION.build_packet(
        epoch=len(applied) + 1, applied_corrections=applied, prior_receipt=prior_receipt,
        facts=facts,
        input_bindings={
            "case": {"checkpoint_sha256": case_cp,
                     "telemetry_sha256": case["source_binding"]["telemetry_sha256"]},
            "reference": {"checkpoint_sha256": reference_cp,
                          "telemetry_sha256": reference["source_binding"]["telemetry_sha256"]},
        }, budget_left=budget_left)
    if remaining_candidates is not None:
        _require(packet["remaining_candidates"] == remaining_candidates,
                 "remaining_candidates는 등록 교정에서 재계산한 값과 같아야 한다")
    return {
        "schema": PACKET_SCHEMA, "packet": packet,
        "private_metadata": {
            "case": {"collected": str(Path(case_path).resolve()), "coverage": case_coverage,
                     "runtime": {"path": str(Path(case_runtime_path).resolve()),
                                 "sha256": _sha(case_runtime_path)}},
            "reference": {"collected": str(Path(reference_path).resolve()),
                          "coverage": reference_coverage, "runtime": reference_runtime_ref},
        },
        "scope": {"activity": "cpu_packet_materialization_only",
                  "approval": False, "execution_allowed": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("collect")
    for name in ("manifest", "evidence-dir", "owned-result", "lab", "isaac-python", "out"):
        create.add_argument(f"--{name}", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--collected", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.manifest, args.evidence_dir, args.owned_result, lab=args.lab,
                isaac_python=args.isaac_python, out=args.out)
    else:
        checked(args.collected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
