"""완료된 첫 교정 recovery 실행을 경로 없는 공개 artifact로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_execution_v2 as V2

V1 = V2.V1
SCHEMA = "lineage_recovery_public_export_v1"
RECEIPT_SCHEMA = "lineage_recovery_public_receipt_v1"
MANIFEST_SCHEMA = "lineage_recovery_public_artifact_manifest_v1"
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])[A-Z]:[\\/]")
PRIVATE_MARKERS = ("bench/private", "bench\\private", "/home/", "/mnt/c/users/", "c:\\users\\")
SECRET_PATTERNS = (
    re.compile(r"(?i)\bnvapi-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|apikey)\s*[:=]\s*[\"']?(?!null\b|none\b|redacted\b)[A-Za-z0-9._-]{8,}"),
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _privacy(name: str, raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"공개 파일은 UTF-8이어야 한다: {name}") from error
    normalized = text.replace("\\\\", "\\")
    lowered = normalized.lower()
    _require(not WINDOWS_ABSOLUTE.search(normalized), f"개인 Windows 절대경로가 있다: {name}")
    _require(not any(marker in lowered for marker in PRIVATE_MARKERS),
             f"private 경로 marker가 있다: {name}")
    _require(not any(pattern.search(text) for pattern in SECRET_PATTERNS),
             f"credential pattern이 있다: {name}")


def _raw_receipt(result_row: dict) -> tuple[Path, dict]:
    path = Path(str(result_row.get("path", ""))).resolve()
    _require(path.is_file() and _sha(path) == result_row.get("sha256"),
             "recovery receipt 원문/SHA가 다르다")
    raw = _read(path)
    embedded = {key: deepcopy(value) for key, value in result_row.items()
                if key not in {"path", "sha256"}}
    _require(raw == embedded, "recovery receipt embedded 값이 원문과 다르다")
    return path, raw


def _hold(row: dict, receipt: dict) -> dict:
    expected = {
        "schema": V1.RECEIPT_SCHEMA, "status": "HOLD",
        "scope": "first_registered_correction_only", "case_id": row["case_id"],
        "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
        "approval": False, "model_calls": 0, "overall_goal_complete": False,
        "second_epoch_registered": False,
    }
    _require(row["status"] == "HOLD" and receipt == expected,
             "HOLD receipt가 입력 행 또는 sdk_jobs=0 계약과 다르다")
    return {
        "schema": RECEIPT_SCHEMA, "case_id": row["case_id"], "policy": row["policy"],
        "arm": row["arm"], "seed": row["seed"], "status": "HOLD",
        "reason": row["hold_reason"], "sdk_jobs": 0, "training_updates": 0,
        "fixed_evaluations": 0,
    }


def _fixed_report(path: Path, expected_sha: str) -> dict:
    _require(path.is_file() and _sha(path) == expected_sha, "fixed report 원문/SHA가 다르다")
    report = _read(path)
    _require(report.get("protocol") == "fixed_eval_v1"
             and report.get("eval_seed") == 2026 and report.get("num_envs") == 1040
             and report.get("horizon_steps") == 1000 and report.get("action_scale") == 0.25
             and isinstance(report.get("by_condition"), list)
             and len(report["by_condition"]) == 26,
             "동결 26조건 fixed report 계약이 다르다")
    return report


def _candidate(row: dict, receipt: dict, manifest: dict, execution_root: Path) -> tuple[dict, dict[str, bytes]]:
    _require(row["status"] == "candidate", "실행 receipt의 입력 행이 candidate가 아니다")
    candidate = V1._candidate(row, manifest)
    _require(receipt.get("candidate") == row["candidate"]
             and receipt.get("case_id") == row["case_id"]
             and receipt.get("policy") == row["policy"]
             and receipt.get("seed") == row["seed"]
             and receipt.get("same_seed_pairing") is True
             and receipt.get("new_independent_seed") is False
             and receipt.get("training_mode") == "fresh_restart"
             and receipt.get("source_checkpoint_sha256") == row["source_checkpoint_sha256"]
             and receipt.get("reference_checkpoint_sha256") == row["reference_checkpoint_sha256"]
             and receipt.get("component") == candidate["component"]
             and receipt.get("correction") == candidate["correction"]
             and receipt.get("second_epoch_registered") is False
             and receipt.get("model_calls") == 0,
             "candidate recovery receipt의 입력·교정·same-seed 결합이 다르다")

    epoch_path = Path(row["epoch_result"]["path"])
    epoch = _read(epoch_path)
    _require(_sha(epoch_path) == row["epoch_result"]["sha256"]
             and epoch.get("status") == "complete" and epoch.get("end_state") == "confirmed"
             and epoch.get("policy") == row["policy"]
             and epoch.get("confirmed_subtype") == candidate["component"],
             "원래 비교 epoch의 모델 정책 확정과 candidate 결합이 다르다")

    stem = f"{row['case_id']}_{row['policy']}"
    stage = execution_root / "runs" / stem
    train_manifest_path = stage / "train/manifest.json"
    training_root = stage / "train/training"
    training_evidence_path = training_root / V1.GUARD.EVIDENCE
    train_manifest = _read(train_manifest_path)
    V1.TRAIN.verify_binding(train_manifest, lab=Path(manifest["lab"]))
    V1.GUARD.validate_completion(training_root, V1.TRAIN.manifest_fingerprint(train_manifest))
    evidence = _read(training_evidence_path)
    _require(evidence == receipt.get("training")
             and evidence.get("updates") == evidence.get("update_count") == 300
             and evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False,
             "guarded 300-update fresh training evidence가 다르다")
    checkpoint = evidence.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    _require(checkpoint_path.is_file() and _sha(checkpoint_path) == checkpoint.get("sha256")
             and checkpoint.get("iteration") == 299
             and receipt.get("new_checkpoint") == checkpoint,
             "300-update 최종 checkpoint 원문/SHA가 다르다")

    collected_path = stage / "collected/collected.json"
    collected, _telemetry = V1.RECOVERY_INPUTS.checked_recovery(collected_path)
    _require(collected == receipt.get("collected_recovery"),
             "봉인 recovery collected가 receipt와 다르다")

    corrected_path = Path(receipt.get("corrected_report", {}).get("path", ""))
    reference_path = Path(receipt.get("reference_report", {}).get("path", ""))
    corrected = _fixed_report(corrected_path, receipt["corrected_report"]["sha256"])
    reference = _fixed_report(reference_path, receipt["reference_report"]["sha256"])
    _require(reference_path.resolve() == Path(row["reference_report"]["path"]).resolve()
             and _sha(reference_path) == row["reference_report"]["sha256"]
             and reference["checkpoint"]["sha256"] == row["reference_checkpoint_sha256"]
             and corrected["checkpoint"]["sha256"] == checkpoint["sha256"],
             "교정/같은-seed reference fixed report checkpoint가 다르다")
    recomputed = V1.C.D1.recovery(corrected, reference, _read(V1.BANDS)["bands"])
    _require(recomputed == receipt.get("recovery"), "동결 recovery 재계산이 receipt와 다르다")
    expected_status = {"recovered": "recovered", "not_recovered": "not_recovered",
                       "unknown": "HOLD"}[recomputed["outcome"]]
    _require(receipt.get("status") == expected_status,
             "unknown 또는 recovery outcome이 다른 상태로 승격됐다")
    evaluation = receipt.get("evaluation_execution")
    _require(isinstance(evaluation, dict) and V1._owned_ok(evaluation, V1.WATCHDOG["eval"]),
             "fixed evaluation owned receipt가 완결되지 않았다")

    ledger_path = execution_root / "ledgers" / f"{stem}.jsonl"
    ledger = V1.C.D1.CorrectionLedger(ledger_path, V1.C.RC.correction_registry())
    events = ledger.events()
    _require([event["event"] for event in events] == ["proposed", "approved", "consumed", "receipt"]
             and events[-1] == receipt.get("ledger_receipt"),
             "후보 승인·소비·결과 ledger가 receipt와 다르다")

    files = {
        f"reports/{stem}-corrected.json": corrected_path.read_bytes(),
        f"reports/{stem}-same-seed-reference.json": reference_path.read_bytes(),
    }
    public = {
        "schema": RECEIPT_SCHEMA, "case_id": row["case_id"], "policy": row["policy"],
        "arm": row["arm"], "seed": row["seed"], "status": receipt["status"],
        "diagnosis": {"epoch_result_sha256": row["epoch_result"]["sha256"],
                      "end_state": epoch["end_state"],
                      "confirmed_subtype": epoch["confirmed_subtype"],
                      "candidate_sha256": row["candidate"]["sha256"]},
        "correction": candidate["correction"], "component": candidate["component"],
        "training": {"mode": "fresh_restart", "updates": 300,
                     "fresh_initialization": True, "seed_reused_for_pairing": True,
                     "source_checkpoint_provenance_sha256": row["source_checkpoint_sha256"],
                     "new_checkpoint_sha256": checkpoint["sha256"]},
        "evaluation": {"conditions": 26, "corrected_report": {
            "path": f"reports/{stem}-corrected.json", "sha256": _sha(corrected_path)},
            "same_seed_reference_report": {
                "path": f"reports/{stem}-same-seed-reference.json", "sha256": _sha(reference_path)},
            "reference_checkpoint_sha256": row["reference_checkpoint_sha256"]},
        "recovery": deepcopy(recomputed),
        "scope": {"first_registered_correction_only": True, "second_epoch": False,
                  "model_vs_version_3_comparison": False, "video_evidence": False},
    }
    return public, files


def _validated(manifest_path: Path, result_path: Path) -> tuple[dict, dict, dict[str, bytes]]:
    with V2._v1_scope():
        manifest, protocol = V1._load_manifest(manifest_path)
        result = _read(result_path)
        rows = manifest["rows"]
        receipts = result.get("receipts")
        _require(result.get("schema") == V2.SCHEMA + "_result_v1"
                 and result.get("status") == "complete"
                 and result.get("scope") == "first_registered_correction_only"
                 and result.get("overall_goal_complete") is False
                 and result.get("second_epoch_registered") is False
                 and result.get("model_calls") == 0
                 and result.get("automatic_retry") is False
                 and isinstance(receipts, list) and len(receipts) == len(rows) == 4,
                 "완료 recovery result의 4행/scope 계약이 다르다")
        by_key = {(row["case_id"], row["policy"]): row for row in rows}
        public_receipts, files = [], {}
        candidate_count = 0
        smoke_ref = None
        for embedded in receipts:
            _path, receipt = _raw_receipt(embedded)
            key = (receipt.get("case_id"), receipt.get("policy"))
            _require(key in by_key, "recovery receipt가 등록 4행 밖이다")
            row = by_key.pop(key)
            if row["status"] == "HOLD":
                public = _hold(row, receipt)
            else:
                candidate_count += 1
                public, report_files = _candidate(row, receipt, manifest, Path(result_path).resolve().parent)
                files.update(report_files)
                smoke_ref = receipt.get("smoke")
            relative = f"receipts/{row['case_id']}_{row['policy']}.json"
            raw = _bytes(public)
            files[relative] = raw
            public_receipts.append({"case_id": row["case_id"], "policy": row["policy"],
                                    "status": public["status"], "path": relative,
                                    "sha256": hashlib.sha256(raw).hexdigest()})
        _require(not by_key and candidate_count == 1 and smoke_ref is not None,
                 "등록 입력은 candidate 1행과 HOLD 3행이어야 한다")
        V1.verify_smoke(Path(smoke_ref["path"]), manifest, protocol)
        counts = {status: sum(row["status"] == status for row in public_receipts)
                  for status in ("HOLD", "recovered", "not_recovered")}
        summary = {
            "schema": SCHEMA, "status": "complete", "planned_rows": 4,
            "completed_receipts": 4, "receipt_counts": counts,
            "candidate_rows": 1, "hold_rows": 3, "training_jobs": 1,
            "training_updates": 300, "fixed_evaluations": 1, "smoke_jobs": 1,
            "smoke_updates": 3, "source_anchor": manifest["anchor"],
            "receipts": public_receipts,
            "scope": {"first_registered_correction_only": True,
                      "same_seed_pairing": True, "new_independent_seed": False,
                      "model_vs_version_3_comparison": False, "video_evidence": False,
                      "generalization_claim": False, "overall_goal_complete": False},
        }
        files["summary.json"] = _bytes(summary)
        provenance = {
            "schema": SCHEMA + "_provenance_v1",
            "source_manifest_sha256": _sha(manifest_path),
            "source_result_sha256": _sha(result_path),
            "source_protocol_sha256": manifest["protocol"]["sha256"],
            "source_anchor": manifest["anchor"],
            "raw_paths_in_public_files": False,
        }
        files["provenance.json"] = _bytes(provenance)
        return summary, provenance, files


def export_completed(manifest_path: Path, result_path: Path, *, out: Path,
                     repo: Path = ROOT) -> dict:
    """완료 원문을 재검증하고 새 공개 폴더에 허용 파일만 쓴다."""
    out, repo = Path(out).resolve(), Path(repo).resolve()
    _require(not out.exists() and (out == repo or out.is_relative_to(repo)),
             "공개 export는 repo 안의 새 경로여야 한다")
    summary, provenance, files = _validated(Path(manifest_path), Path(result_path))
    for name, raw in files.items():
        _privacy(name, raw)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in sorted(files.items()):
        _write_new(out / name, raw)
    artifact_manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())},
        "manifest_self_sha256_recorded": False,
    }
    _write_new(out / "artifact-manifest.json", _bytes(artifact_manifest))
    verify_export(out)
    return {"summary": summary, "provenance": provenance,
            "artifact_manifest": artifact_manifest}


def verify_export(out: Path) -> dict:
    root = Path(out).resolve()
    manifest = _read(root / "artifact-manifest.json")
    files = manifest.get("files")
    _require(manifest.get("schema") == MANIFEST_SCHEMA
             and manifest.get("manifest_self_sha256_recorded") is False
             and isinstance(files, dict) and "artifact-manifest.json" not in files,
             "공개 artifact manifest 계약이 다르다")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact-manifest.json"}, "공개 파일 목록이 manifest와 다르다")
    for name, expected in files.items():
        path = root / name
        _require(_sha(path) == expected, f"공개 파일 SHA가 다르다: {name}")
        _privacy(name, path.read_bytes())
    summary = _read(root / "summary.json")
    _require(summary.get("schema") == SCHEMA and summary.get("status") == "complete"
             and summary.get("planned_rows") == summary.get("completed_receipts") == 4
             and summary.get("candidate_rows") == summary.get("training_jobs")
             == summary.get("fixed_evaluations") == 1
             and summary.get("hold_rows") == 3 and summary.get("training_updates") == 300
             and summary.get("smoke_jobs") == 1 and summary.get("smoke_updates") == 3,
             "공개 summary counts가 다르다")
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
