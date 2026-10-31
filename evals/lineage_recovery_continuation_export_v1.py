"""완료된 evaluation-only recovery continuation을 공개 artifact로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_continuation_v4 as V4
import lineage_recovery_export_v1 as E1

V3 = V4.V3
V1 = V4.V1
SCHEMA = "lineage_recovery_continuation_public_export_v1"
RECEIPT_SCHEMA = "lineage_recovery_continuation_public_receipt_v1"
MANIFEST_SCHEMA = "lineage_recovery_continuation_public_artifact_manifest_v1"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _ref(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file(), f"원문 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": E1._sha(path)}


def _sha_lf(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _raw_receipt(embedded: dict) -> tuple[Path, dict]:
    path = Path(str(embedded.get("path", ""))).resolve()
    _require(path.is_file() and E1._sha(path) == embedded.get("sha256"),
             "continuation receipt 원문/SHA가 다르다")
    value = E1._read(path)
    expected = {key: deepcopy(item) for key, item in embedded.items()
                if key not in {"path", "sha256"}}
    _require(value == expected, "continuation receipt embedded 값이 원문과 다르다")
    return path, value


def _model_origin(parent_manifest: dict, row: dict) -> dict:
    batch_manifest = E1._read(Path(parent_manifest["batch_manifest"]["path"]))
    batch_protocol = E1._read(Path(batch_manifest["protocol_path"]))
    transport = batch_protocol["component_protocols"]["nim_transport"]
    path = ROOT / transport["path"]
    _require(path.is_file() and _sha_lf(path) == transport["sha256_lf"],
             "진단 model transport protocol SHA가 다르다")
    value = E1._read(path)
    _require(value.get("model") == "nvidia/nemotron-3.5-super-vl-120b-a12b",
             "진단 model origin이 등록 Nemotron 3.5 Super VL이 아니다")
    epoch = Path(row["epoch_result"]["path"])
    _require(epoch.is_file() and E1._sha(epoch) == row["epoch_result"]["sha256"],
             "진단 epoch origin SHA가 다르다")
    return {"model": value["model"], "transport_protocol_sha256": transport["sha256_lf"],
            "diagnosis_epoch_result_sha256": row["epoch_result"]["sha256"]}


def _hold(row: dict, receipt: dict) -> dict:
    expected = {
        "schema": V1.RECEIPT_SCHEMA, "status": "HOLD",
        "scope": "first_registered_correction_only", "case_id": row["case_id"],
        "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
        "approval": False, "model_calls": 0, "overall_goal_complete": False,
        "second_epoch_registered": False,
    }
    _require(row["status"] == "HOLD" and receipt == expected,
             "continuation HOLD receipt 또는 sdk_jobs=0 계약이 다르다")
    return {"schema": RECEIPT_SCHEMA, "case_id": row["case_id"], "policy": row["policy"],
            "arm": row["arm"], "seed": row["seed"], "status": "HOLD",
            "reason": row["hold_reason"], "sdk_jobs": 0, "new_training_jobs": 0,
            "fixed_evaluations": 0}


def _candidate(row: dict, receipt: dict, continuation: dict, prior: dict,
               result_root: Path) -> tuple[dict, dict[str, bytes]]:
    _require(row == prior["row"] and receipt.get("case_id") == row["case_id"]
             and receipt.get("policy") == row["policy"]
             and receipt.get("continuation") is True
             and receipt.get("training_reused") is True
             and receipt.get("new_training_jobs") == 0
             and receipt.get("model_calls") == 0
             and receipt.get("candidate") == row["candidate"]
             and receipt.get("new_checkpoint") == prior["evidence"]["checkpoint"]
             and receipt.get("training") == prior["evidence"]
             and receipt.get("collected_recovery") == prior["collected"]
             and receipt.get("prior_failed_execution_sha256")
             == continuation["failed_execution"]["failure"]["sha256"]
             and receipt.get("prior_training_stage") == _ref(
                 prior["stage"] / "train/manifest.json"),
             "continuation receipt의 old 300-update 재사용 결합이 다르다")
    _require(not any(path.is_symlink() for path in result_root.rglob("*"))
             and not any((result_root / name).exists() for name in ("train", "training", "collected", "runs")),
             "continuation output에 학습 사본·symlink·relabel stage가 있다")

    corrected_path = Path(receipt["corrected_report"]["path"])
    reference_path = Path(receipt["reference_report"]["path"])
    corrected = E1._fixed_report(corrected_path, receipt["corrected_report"]["sha256"])
    reference = E1._fixed_report(reference_path, receipt["reference_report"]["sha256"])
    _require(corrected_path.resolve() == (result_root / "evaluation/corrected.json").resolve()
             and reference_path.resolve() == Path(row["reference_report"]["path"]).resolve()
             and corrected["checkpoint"]["sha256"] == prior["evidence"]["checkpoint"]["sha256"]
             and reference["checkpoint"]["sha256"] == row["reference_checkpoint_sha256"],
             "새 평가/같은-seed reference report checkpoint가 다르다")
    recomputed = V1.C.D1.recovery(corrected, reference, E1._read(V1.BANDS)["bands"])
    _require(recomputed == receipt.get("recovery"), "continuation recovery 재계산이 다르다")
    status = {"recovered": "recovered", "not_recovered": "not_recovered",
              "unknown": "HOLD"}[recomputed["outcome"]]
    _require(receipt.get("status") == status, "unknown continuation 결과가 다른 상태로 승격됐다")
    _require(V1._owned_ok(receipt.get("evaluation_execution", {}), V1.WATCHDOG["eval"]),
             "새 fixed evaluation owned receipt가 완결되지 않았다")

    old_ledger = prior["ledger_path"].read_bytes()
    new_ledger = result_root / "ledger/correction.jsonl"
    _require(new_ledger.is_file() and new_ledger.read_bytes().startswith(old_ledger)
             and prior["ledger_path"].read_bytes() == old_ledger
             and E1._sha(prior["ledger_path"]) == continuation["ledger"]["source"]["sha256"],
             "새 ledger prefix 또는 이전 consumed ledger 원문이 바뀌었다")
    ledger = V1.C.D1.CorrectionLedger(new_ledger, V1.C.RC.correction_registry())
    events = ledger.events()
    _require(len(events) == 4 and events[:3] == prior["ledger_events"]
             and events[-1] == receipt.get("ledger_receipt"),
             "새 ledger 4-event 또는 기존 3-event prefix가 다르다")
    provenance = E1._read(result_root / "ledger/prefix-provenance.json")
    _require(provenance.get("source") == _ref(prior["ledger_path"])
             and provenance.get("copied_prefix_sha256") == hashlib.sha256(old_ledger).hexdigest()
             and provenance.get("copied_prefix_size") == len(old_ledger)
             and provenance.get("events") == 3,
             "ledger prefix provenance가 이전 원문과 다르다")

    stem = f"{row['case_id']}_{row['policy']}"
    files = {f"reports/{stem}-corrected.json": corrected_path.read_bytes(),
             f"reports/{stem}-same-seed-reference.json": reference_path.read_bytes()}
    public = {
        "schema": RECEIPT_SCHEMA, "case_id": row["case_id"], "policy": row["policy"],
        "arm": row["arm"], "seed": row["seed"], "status": status,
        "diagnosis_origin": _model_origin(prior["parent_manifest"], row),
        "correction": prior["candidate"]["correction"],
        "training": {"reused": True, "updates": 300, "new_training_jobs": 0,
                     "fresh_initialization": True, "seed_reused_for_pairing": True,
                     "checkpoint_sha256": prior["evidence"]["checkpoint"]["sha256"]},
        "evaluation": {"new_fixed_evaluations": 1, "conditions": 26,
                       "corrected_report": {"path": f"reports/{stem}-corrected.json",
                                            "sha256": E1._sha(corrected_path)},
                       "same_seed_reference_report": {
                           "path": f"reports/{stem}-same-seed-reference.json",
                           "sha256": E1._sha(reference_path)}},
        "recovery": deepcopy(recomputed),
        "scope": {"first_registered_correction_only": True, "model_vs_version_3": False,
                  "video_gain": False, "new_seed": False, "generalization_claim": False},
    }
    return public, files


def _validated(manifest_path: Path, result_path: Path) -> tuple[dict, dict, dict[str, bytes]]:
    with V4._v3_scope():
        continuation, parent_manifest, _approval, _smoke, prior = V3._load_continuation(manifest_path)
        prior["parent_manifest"] = parent_manifest
        result = E1._read(result_path)
        receipts = result.get("receipts")
        _require(result.get("schema") == V4.RESULT_SCHEMA and result.get("status") == "complete"
                 and result.get("scope") == "first_registered_correction_evaluation_continuation"
                 and result.get("planned_rows") == 4 and result.get("hold_rows") == 3
                 and result.get("evaluated_rows") == result.get("reused_training_rows") == 1
                 and result.get("new_training_jobs") == result.get("model_calls")
                 == result.get("new_correction_approvals") == 0
                 and result.get("evaluation_conditions") == 26
                 and result.get("same_seed_pairing") is True
                 and result.get("overall_goal_complete") is False
                 and isinstance(receipts, list) and len(receipts) == 4,
                 "완료 continuation result의 4행/평가-only counts가 다르다")
        result_root = Path(result_path).resolve().parent
        by_key = {(row["case_id"], row["policy"]): row for row in parent_manifest["rows"]}
        public_receipts, files, candidate_count = [], {}, 0
        candidate_origin = None
        for embedded in receipts:
            _path, receipt = _raw_receipt(embedded)
            key = (receipt.get("case_id"), receipt.get("policy"))
            _require(key in by_key, "continuation receipt가 등록 4행 밖이다")
            row = by_key.pop(key)
            if row["status"] == "HOLD":
                public = _hold(row, receipt)
            else:
                candidate_count += 1
                public, report_files = _candidate(row, receipt, continuation, prior, result_root)
                candidate_origin = public["diagnosis_origin"]
                files.update(report_files)
            relative = f"receipts/{row['case_id']}_{row['policy']}.json"
            raw = E1._bytes(public)
            files[relative] = raw
            public_receipts.append({"case_id": row["case_id"], "policy": row["policy"],
                                    "status": public["status"], "path": relative,
                                    "sha256": hashlib.sha256(raw).hexdigest()})
        _require(not by_key and candidate_count == 1 and candidate_origin is not None,
                 "candidate 1행/HOLD 3행이 완결되지 않았다")
        recovered = next(item for item in public_receipts if item["policy"] == "model_adaptive"
                         and item["case_id"] == "case-03")["status"]
        summary = {
            "schema": SCHEMA, "status": "complete", "planned_rows": 4,
            "completed_receipts": 4, "hold_input_rows": 3, "candidate_rows": 1,
            "reused_training_jobs": 1, "reused_training_updates": 300,
            "new_training_jobs": 0, "new_fixed_evaluations": 1,
            "evaluation_conditions": 26, "recovery_status": recovered,
            "diagnosis_model": candidate_origin["model"],
            "diagnosis_epoch_result_sha256": candidate_origin[
                "diagnosis_epoch_result_sha256"
            ],
            "source_anchor": continuation["anchor"], "parent_source_anchor": parent_manifest["anchor"],
            "receipts": public_receipts,
            "scope": {"same_seed_pairing": True, "new_independent_seed": False,
                      "first_registered_correction_only": True, "model_vs_version_3": False,
                      "video_gain": False, "generalization_claim": False,
                      "overall_goal_complete": False},
        }
        files["summary.json"] = E1._bytes(summary)
        provenance = {"schema": SCHEMA + "_provenance_v1",
                      "continuation_manifest_sha256": E1._sha(manifest_path),
                      "continuation_result_sha256": E1._sha(result_path),
                      "prior_failure_sha256": continuation["failed_execution"]["failure"]["sha256"],
                      "prior_training_evidence_sha256": continuation["training"]["evidence"]["sha256"],
                      "prior_consumed_ledger_sha256": continuation["ledger"]["source"]["sha256"],
                      "raw_paths_in_public_files": False}
        files["provenance.json"] = E1._bytes(provenance)
        return summary, provenance, files


def export_completed(manifest_path: Path, result_path: Path, *, out: Path,
                     repo: Path = ROOT) -> dict:
    out, repo = Path(out).resolve(), Path(repo).resolve()
    _require(not out.exists() and out.is_relative_to(repo), "공개 export는 repo 안의 새 경로여야 한다")
    summary, provenance, files = _validated(Path(manifest_path), Path(result_path))
    for name, raw in files.items():
        E1._privacy(name, raw)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in sorted(files.items()):
        E1._write_new(out / name, raw)
    artifact = {"schema": MANIFEST_SCHEMA,
                "files": {name: hashlib.sha256(raw).hexdigest()
                          for name, raw in sorted(files.items())},
                "manifest_self_sha256_recorded": False}
    E1._write_new(out / "artifact-manifest.json", E1._bytes(artifact))
    verify_export(out)
    return {"summary": summary, "provenance": provenance, "artifact_manifest": artifact}


def verify_export(out: Path) -> dict:
    root = Path(out).resolve()
    manifest = E1._read(root / "artifact-manifest.json")
    files = manifest.get("files")
    _require(manifest.get("schema") == MANIFEST_SCHEMA
             and manifest.get("manifest_self_sha256_recorded") is False
             and isinstance(files, dict) and "artifact-manifest.json" not in files,
             "continuation public manifest 계약이 다르다")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact-manifest.json"}, "공개 whitelist 파일 목록이 다르다")
    for name, expected in files.items():
        path = root / name
        _require(E1._sha(path) == expected, f"공개 파일 SHA가 다르다: {name}")
        E1._privacy(name, path.read_bytes())
    summary = E1._read(root / "summary.json")
    _require(summary.get("schema") == SCHEMA and summary.get("status") == "complete"
             and summary.get("planned_rows") == summary.get("completed_receipts") == 4
             and summary.get("hold_input_rows") == 3 and summary.get("candidate_rows") == 1
             and summary.get("reused_training_updates") == 300
             and summary.get("new_training_jobs") == 0
             and summary.get("new_fixed_evaluations") == 1
             and summary.get("evaluation_conditions") == 26
             and summary.get("diagnosis_model")
             == "nvidia/nemotron-3.5-super-vl-120b-a12b"
             and isinstance(summary.get("diagnosis_epoch_result_sha256"), str),
             "continuation 공개 summary counts가 다르다")
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--result", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export_completed(args.manifest, args.result, out=args.out)
    else:
        verify_export(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
