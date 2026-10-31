"""첫 교정 뒤 비회복 artifact를 둘째 진단 입력으로 봉인하는 CPU 어댑터."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import lineage_recovery_execution_v1 as EXEC
import lineage_recovery_inputs_v1 as RECOVERY
import lineage_selection_inputs_v1 as INPUT

SCHEMA = "lineage_post_correction_observation_v1"
LEDGER_EVENTS = ("proposed", "approved", "consumed", "receipt")
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "original_faults", "remaining_faults", "source_arm", "registered_correction",
    "correction_id", "readback", "receipt", "ledger_receipt",
})


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    _require(all(isinstance(row, dict) for row in values), "correction ledger 행 형식이 다르다")
    return values


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_ledger(path: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _rows(path)
    _require(len(rows) == 4 and tuple(row.get("event") for row in rows) == LEDGER_EVENTS,
             "correction ledger는 proposed→approved→consumed→receipt 네 행이어야 한다")
    request_ids = {row.get("request_id") for row in rows}
    _require(len(request_ids) == 1 and None not in request_ids,
             "correction ledger request_id 결합이 다르다")
    proposed, _approved, _consumed, ledger_receipt = rows
    _require(ledger_receipt == receipt.get("ledger_receipt"), "ledger receipt가 repair receipt와 다르다")
    _require(proposed.get("correction_id") == receipt.get("correction"),
             "proposed correction과 repair receipt가 다르다")
    _require(ledger_receipt.get("verdict") == "not_recovered"
             and ledger_receipt.get("exit_code") == 0,
             "ledger receipt가 유효한 비회복 완료 영수증이 아니다")
    return rows


def _validate_fixed_reports(receipt: dict[str, Any]) -> dict[str, Any]:
    corrected_ref, reference_ref = receipt.get("corrected_report"), receipt.get("reference_report")
    _require(isinstance(corrected_ref, dict) and isinstance(reference_ref, dict),
             "corrected/reference 고정 평가 참조가 없다")
    corrected_path = Path(corrected_ref.get("path", ""))
    reference_path = Path(reference_ref.get("path", ""))
    _require(corrected_path.is_file() and _sha(corrected_path) == corrected_ref.get("sha256"),
             "corrected 고정 평가 원문이 바뀌었다")
    _require(reference_path.is_file() and _sha(reference_path) == reference_ref.get("sha256"),
             "reference 고정 평가 원문이 바뀌었다")
    corrected, reference = _read(corrected_path), _read(reference_path)
    checkpoint = receipt["new_checkpoint"]
    EXEC.C.G.validate_fixed_report(corrected, {
        "name": corrected["name"], "checkpoint_file": Path(checkpoint["path"]).name,
        "checkpoint_sha256": checkpoint["sha256"], "action_scale": 0.25,
        "protocol_sha256_lf": receipt["fixed_protocol"]["sha256_lf"],
    })
    EXEC.C.G.validate_fixed_report(reference, {
        "name": reference["name"], "checkpoint_file": reference["checkpoint"]["file"],
        "checkpoint_sha256": receipt["reference_checkpoint_sha256"], "action_scale": 0.25,
        "protocol_sha256_lf": receipt["fixed_protocol"]["sha256_lf"],
    })
    bands_path = ROOT / receipt["bands"]["path"]
    _require(bands_path.is_file() and _sha(bands_path) == receipt["bands"]["sha256"],
             "동결 recovery bands가 바뀌었다")
    recomputed = EXEC.C.D1.recovery(corrected, reference, _read(bands_path)["bands"])
    _require(recomputed.get("outcome") == "not_recovered" and recomputed == receipt.get("recovery"),
             "고정 평가 재계산 결과가 not_recovered receipt와 다르다")
    return recomputed


def _validate_inputs(recovery_path: Path, receipt_path: Path, ledger_path: Path,
                     reference_path: Path, lab: Path) -> tuple[dict, dict, dict, dict, list[dict]]:
    receipt = _read(receipt_path)
    _require(receipt.get("schema") == EXEC.RECEIPT_SCHEMA, "repair receipt schema가 다르다")
    _require(receipt.get("status") == "not_recovered"
             and receipt.get("next_step") == "new_protocol_required"
             and receipt.get("second_epoch_registered") is False,
             "둘째 진단은 유효한 not_recovered receipt만 허용한다")
    recovery, telemetry = RECOVERY.checked_recovery(recovery_path)
    reference, reference_telemetry = INPUT._checked_collected(reference_path)
    _require(receipt.get("same_seed_pairing") is True
             and receipt.get("training_mode") == "fresh_restart",
             "repair receipt의 same-seed fresh restart 결합이 다르다")
    _require(Path(lab).resolve() == Path(recovery["private_source_map"]["lab"]).resolve(),
             "검증 Isaac Lab 경로가 recovery 입력과 다르다")
    training = recovery["training"]
    evidence = receipt.get("training")
    _require(isinstance(evidence, dict)
             and evidence.get("updates") == evidence.get("update_count") == 300
             and evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False,
             "repair receipt의 GUARD 300 fresh 증거가 다르다")
    _require(receipt.get("collected_recovery") == recovery,
             "repair receipt가 검증한 recovery collected와 다르다")
    checkpoint = receipt.get("new_checkpoint")
    _require(isinstance(checkpoint, dict) and Path(checkpoint.get("path", "")).is_file()
             and _sha(Path(checkpoint["path"])) == checkpoint.get("sha256")
             and checkpoint.get("sha256") == recovery["source_binding"]["checkpoint_sha256"],
             "repair receipt의 새 checkpoint 결합이 다르다")
    _require(receipt.get("seed") == training.get("seed")
             == reference["private_source_map"].get("seed")
             and reference["private_source_map"].get("arm") == "NONE",
             "same-seed NONE reference 결합이 다르다")
    _require(receipt.get("reference_checkpoint_sha256")
             == reference["source_binding"]["checkpoint_sha256"],
             "repair receipt와 NONE reference checkpoint가 다르다")
    _require(receipt.get("correction") in training.get("applied_corrections", [])
             and receipt.get("correction") in evidence.get("applied_corrections", [])
             and training.get("applied_corrections") == evidence.get("applied_corrections"),
             "등록 교정과 실제 applied correction이 다르다")
    expected_correction = INPUT.V.S.C1.CORRECTIONS.get(receipt.get("component"))
    _require(expected_correction == receipt.get("correction"), "등록 component/correction 결합이 다르다")
    ledger = _validate_ledger(ledger_path, receipt)
    _validate_fixed_reports(receipt)
    return recovery, telemetry, reference, reference_telemetry, ledger


def prepare_observation(recovery_collected_path: Path, repair_receipt_path: Path,
                        correction_ledger_path: Path, reference_collected_path: Path,
                        *, out: Path, lab: Path) -> dict:
    """검증한 첫 교정 결과를 실행 권한 없는 epoch-2 관측으로 봉인한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "post-correction out은 fresh 경로여야 한다")
    recovery, telemetry, _reference, _reference_telemetry, ledger = _validate_inputs(
        Path(recovery_collected_path).resolve(), Path(repair_receipt_path).resolve(),
        Path(correction_ledger_path).resolve(), Path(reference_collected_path).resolve(), Path(lab))
    receipt = _read(repair_receipt_path)
    facts, coverage = INPUT._telemetry_projection("case", telemetry)
    receipt_sha = _sha(repair_receipt_path)
    value = {
        "schema": SCHEMA,
        "source_binding": {
            "recovery_collected_sha256": _sha(recovery_collected_path),
            "repair_receipt_sha256": receipt_sha,
            "correction_ledger_sha256": _sha(correction_ledger_path),
            "reference_collected_sha256": _sha(reference_collected_path),
            "checkpoint_sha256": receipt["new_checkpoint"]["sha256"],
            "telemetry_sha256": recovery["source_binding"]["telemetry_sha256"],
        },
        "model_view": {"facts": facts},
        "private_metadata": {
            "sources": {
                "recovery_collected": str(Path(recovery_collected_path).resolve()),
                "repair_receipt": str(Path(repair_receipt_path).resolve()),
                "correction_ledger": str(Path(correction_ledger_path).resolve()),
                "reference_collected": str(Path(reference_collected_path).resolve()),
                "lab": str(Path(lab).resolve()),
            },
            "coverage": coverage,
            "host_only": {
                "original_faults": deepcopy(recovery["training"]["original_faults"]),
                "remaining_faults": deepcopy(recovery["training"]["remaining_faults"]),
                "applied_correction_ids": deepcopy(recovery["training"]["applied_corrections"]),
                "readback": recovery["private_source_map"]["readback"],
                "ledger_event_count": len(ledger),
            },
            "second_epoch_gate": {
                "prior_valid_not_recovered_receipt_sha": receipt_sha,
                "input_checkpoint_sha": receipt["new_checkpoint"]["sha256"],
                "epoch": 2,
                "max_different_corrections": 2,
            },
            "candidate_removal_provenance": (
                "validated_applied_correction_receipt; not hidden ground truth"),
        },
        "scope": {
            "activity": "cpu_post_correction_observation_materialization_only",
            "approval": False, "execution_allowed": False,
            "excluded": ["model_call", "probe_execution", "correction_approval", "training"],
        },
    }
    out.mkdir(parents=True)
    _write_new(out / "observation.json", value)
    return deepcopy(value)


def checked_observation(path: Path) -> tuple[dict, dict]:
    """봉인 관측과 모든 원문 결합을 다시 검증한다."""
    path = Path(path).resolve()
    value = _read(path)
    _require(value.get("schema") == SCHEMA, "post-correction observation schema가 다르다")
    source = value.get("private_metadata", {}).get("sources")
    _require(isinstance(source, dict), "post-correction private source map이 없다")
    recovery, telemetry, _reference, _reference_telemetry, ledger = _validate_inputs(
        Path(source["recovery_collected"]), Path(source["repair_receipt"]),
        Path(source["correction_ledger"]), Path(source["reference_collected"]), Path(source["lab"]))
    binding = value.get("source_binding")
    expected = {
        "recovery_collected_sha256": _sha(Path(source["recovery_collected"])),
        "repair_receipt_sha256": _sha(Path(source["repair_receipt"])),
        "correction_ledger_sha256": _sha(Path(source["correction_ledger"])),
        "reference_collected_sha256": _sha(Path(source["reference_collected"])),
        "checkpoint_sha256": recovery["source_binding"]["checkpoint_sha256"],
        "telemetry_sha256": recovery["source_binding"]["telemetry_sha256"],
    }
    _require(binding == expected, "post-correction source binding이 원문과 다르다")
    facts, coverage = INPUT._telemetry_projection("case", telemetry)
    _require(value.get("model_view") == {"facts": facts}, "공개 telemetry fact projection이 다르다")
    private = value["private_metadata"]
    receipt = _read(Path(source["repair_receipt"]))
    expected_host = {
        "original_faults": recovery["training"]["original_faults"],
        "remaining_faults": recovery["training"]["remaining_faults"],
        "applied_correction_ids": recovery["training"]["applied_corrections"],
        "readback": recovery["private_source_map"]["readback"],
        "ledger_event_count": len(ledger),
    }
    expected_gate = {
        "prior_valid_not_recovered_receipt_sha": binding["repair_receipt_sha256"],
        "input_checkpoint_sha": receipt["new_checkpoint"]["sha256"],
        "epoch": 2, "max_different_corrections": 2,
    }
    _require(private.get("coverage") == coverage and private.get("host_only") == expected_host,
             "private coverage/host 결합이 다르다")
    _require(private.get("second_epoch_gate") == expected_gate
             and private.get("candidate_removal_provenance")
             == "validated_applied_correction_receipt; not hidden ground truth",
             "second epoch gate 또는 후보 제거 provenance가 다르다")
    _require(value.get("scope", {}).get("approval") is False
             and value.get("scope", {}).get("execution_allowed") is False,
             "어댑터가 승인 또는 실행 권한을 만들었다")
    public_text = json.dumps(value["model_view"], ensure_ascii=False, sort_keys=True)
    _require(not any(key in public_text for key in FORBIDDEN_PUBLIC_KEYS),
             "공개 model view에 host-only 정답이 노출됐다")
    return value, telemetry


def make_selection_packet(post_path: Path, reference_path: Path, *, budget_left: int = 3,
                          execution_budget_s: float = 1200) -> dict:
    """검증한 새 telemetry와 같은-seed NONE을 epoch-2 selection packet으로 만든다."""
    post, _telemetry = checked_observation(post_path)
    reference, reference_telemetry = INPUT._checked_collected(reference_path)
    source = post["private_metadata"]["sources"]
    _require(Path(source["reference_collected"]).resolve() == Path(reference_path).resolve(),
             "관측과 packet의 NONE reference가 다르다")
    receipt = _read(Path(source["repair_receipt"]))
    reference_facts, reference_coverage = INPUT._telemetry_projection(
        "reference", reference_telemetry)
    facts = {**post["model_view"]["facts"], **reference_facts}
    prior = {
        "verdict": "not_recovered", "correction_id": receipt["correction"],
        "result_sha256": post["source_binding"]["repair_receipt_sha256"],
        "checkpoint_sha256": receipt["new_checkpoint"]["sha256"],
    }
    packet = INPUT.V.S.build_packet(
        epoch=2, applied_corrections=[receipt["correction"]], prior_receipt=prior, facts=facts,
        input_bindings={
            "case": {"checkpoint_sha256": receipt["new_checkpoint"]["sha256"],
                     "telemetry_sha256": post["source_binding"]["telemetry_sha256"]},
            "reference": {"checkpoint_sha256": reference["source_binding"]["checkpoint_sha256"],
                          "telemetry_sha256": reference["source_binding"]["telemetry_sha256"]},
        }, budget_left=budget_left, execution_budget_s=execution_budget_s)
    return {
        "schema": INPUT.PACKET_SCHEMA, "packet": packet,
        "private_metadata": {
            "post_observation": _ref(Path(post_path)),
            "reference_collected": _ref(Path(reference_path)),
            "reference_coverage": reference_coverage,
            "second_epoch_gate": deepcopy(post["private_metadata"]["second_epoch_gate"]),
            "candidate_removal_provenance": (
                post["private_metadata"]["candidate_removal_provenance"]),
            "authority": {"approval": False, "execution_allowed": False},
        },
        "scope": {"activity": "cpu_epoch2_packet_materialization_only",
                  "excluded": ["model_call", "probe_execution", "approval", "training"]},
    }
