"""완료 학습을 재사용해 중단된 첫 교정의 fixed evaluation만 이어간다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_execution_v2 as V2

V1 = V2.V1
SCHEMA = "lineage_recovery_continuation_protocol_v3"
MANIFEST_SCHEMA = "lineage_recovery_continuation_manifest_v3"
RESULT_SCHEMA = "lineage_recovery_continuation_result_v3"
SOURCE_FILES = (
    "evals/lineage_recovery_continuation_v3.py",
    "tests/test_lineage_recovery_continuation_v3.py",
    "evals/lineage_lease_context_v2.py",
    "tests/test_lineage_lease_context_v2.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
PERMISSION_PATH_SUFFIX = "CodexSignals\\gpu-lease.json"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"원문 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("공개 protocol component는 repo 내부여야 한다") from error


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git anchor/blob 결합을 확인하지 못했다")
    return result.stdout


def prepare_protocol(parent_protocol: Path) -> dict:
    parent = Path(parent_protocol).resolve()
    value = _read(parent)
    _require(value.get("schema") == V2.SCHEMA, "parent recovery protocol은 V2여야 한다")
    sources = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        _require(path.is_file(), f"continuation source가 아직 없다: {name}")
        sources[name] = _sha(path, lf=True)
    return {
        "schema": SCHEMA, "phase": "evaluation_only_after_completed_training",
        "parent_protocol": {"path": _relative(parent), "sha256_lf": _sha(parent, lf=True)},
        "source_sha256_lf": sources,
        "reuse": {"training_updates": 300, "fresh_initialization": True,
                  "new_training_jobs": 0, "new_correction_approvals": 0,
                  "new_model_calls": 0, "evaluation_conditions": 26,
                  "same_seed_reference": True, "first_correction_only": True},
        "failure_precondition": {"error_type": "PermissionError",
                                 "path_suffix": PERMISSION_PATH_SUFFIX,
                                 "evaluation_gpu_jobs_started": 0,
                                 "evaluation_owned_started": False},
        "lease": {"module": "lineage_lease_context_v2",
                  "callback": "lease_context", "bounded_permission_retry": True},
        "claims": {"new_training": False, "new_seed": False, "overall_goal_complete": False},
    }


def verify_protocol(protocol_path: Path, *, anchor: str, lab: Path) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "continuation anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    parent = value.get("parent_protocol")
    _require(isinstance(parent, dict) and set(parent) == {"path", "sha256_lf"},
             "parent protocol 결합이 없다")
    expected = prepare_protocol(ROOT / parent["path"])
    _require(value == expected, "continuation protocol 내용/source가 다르다")
    V2.verify_protocol(ROOT / parent["path"], anchor=anchor, lab=lab)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), parent["path"]: parent["sha256_lf"],
             **value["source_sha256_lf"]}
    for name, expected_sha in bound.items():
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 continuation source가 다르다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == expected_sha,
                 f"anchor continuation source가 다르다: {name}")
    return deepcopy(value)


def _failure_boundary(execution_root: Path) -> tuple[dict, dict]:
    failure_path = execution_root / "failure.json"
    failure = _read(failure_path)
    error_text = failure.get("error")
    normalized_error = error_text if isinstance(error_text, str) else ""
    while "\\\\" in normalized_error:
        normalized_error = normalized_error.replace("\\\\", "\\")
    _require(failure.get("status") == "invalid" and failure.get("error_type") == "PermissionError"
             and isinstance(error_text, str)
             and normalized_error.endswith(PERMISSION_PATH_SUFFIX + "'")
             and failure.get("automatic_retry") is False,
             "이전 실패는 등록된 metadata lease PermissionError가 아니다")
    completed = failure.get("completed_receipts")
    _require(isinstance(completed, list) and len(completed) == 1,
             "이전 실패 전 완료 receipt 수가 다르다")
    prior_hold = completed[0]
    hold_path = Path(str(prior_hold.get("path", "")))
    embedded = {key: value for key, value in prior_hold.items() if key not in {"path", "sha256"}}
    _require(hold_path.is_file() and _sha(hold_path) == prior_hold.get("sha256")
             and _read(hold_path) == embedded and prior_hold.get("case_id") == "case-03"
             and prior_hold.get("policy") == "registry_next"
             and prior_hold.get("status") == "HOLD" and prior_hold.get("sdk_jobs") == 0,
             "이전 실패 전 HOLD receipt 원문이 다르다")
    evaluation = execution_root / "runs/case-03_model_adaptive/evaluation"
    wait_result = _read(evaluation / "gpu-wait/result.json")
    _require(wait_result.get("status") == "error" and wait_result.get("reason") == "PermissionError"
             and wait_result.get("gpu_jobs_started") == 0,
             "이전 evaluation GPU 입장 실패가 다르다")
    for item in wait_result.get("samples", []):
        sample = evaluation / "gpu-wait" / str(item.get("path", ""))
        _require(sample.is_file() and _sha(sample) == item.get("sha256"),
                 "이전 evaluation GPU 입장 sample이 바뀌었다")
    forbidden = (evaluation / "owned", evaluation / "corrected.json", evaluation / "jobs.json",
                 evaluation / "owned-result.json")
    _require(not any(path.exists() for path in forbidden),
             "이전 실패에서 evaluation child 또는 report가 이미 시작됐다")
    return failure, wait_result


def _completed_training(manifest: dict, protocol: dict, execution_root: Path) -> dict:
    row = next(item for item in manifest["rows"] if item["status"] == "candidate")
    _require(sum(item["status"] == "candidate" for item in manifest["rows"]) == 1
             and sum(item["status"] == "HOLD" for item in manifest["rows"]) == 3,
             "continuation 입력은 candidate 1행/HOLD 3행이어야 한다")
    candidate = V1._candidate(row, manifest)
    stage = execution_root / "runs" / f"{row['case_id']}_{row['policy']}"
    train = stage / "train"
    fixed = _read(train / "manifest.json")
    expected = V1._training_manifest(Path(manifest["protocol"]["path"]), protocol,
                                     manifest["anchor"], row, train)
    _require(fixed == expected, "이전 300-update train manifest가 현재 등록 입력과 다르다")
    V1.TRAIN.verify_binding(fixed, lab=Path(manifest["lab"]))
    V1.GUARD.validate_completion(train / "training", V1.TRAIN.manifest_fingerprint(fixed))
    evidence = _read(train / "training" / V1.GUARD.EVIDENCE)
    owned = _read(train / "owned-result.json")
    _require(V1._owned_ok(owned, V1.WATCHDOG["train"])
             and evidence.get("updates") == evidence.get("update_count") == 300
             and evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False,
             "이전 guarded 300-update 학습이 완결되지 않았다")
    checkpoint = evidence.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    _require(checkpoint_path.is_file() and _sha(checkpoint_path) == checkpoint.get("sha256")
             and checkpoint.get("iteration") == 299,
             "이전 fresh checkpoint 299 원문/SHA가 다르다")
    collected_path = stage / "collected/collected.json"
    collected, _telemetry = V1.RECOVERY_INPUTS.checked_recovery(collected_path)
    ledger_path = execution_root / "ledgers" / f"{row['case_id']}_{row['policy']}.jsonl"
    ledger = V1.C.D1.CorrectionLedger(ledger_path, V1.C.RC.correction_registry())
    events = ledger.events()
    _require([event["event"] for event in events] == ["proposed", "approved", "consumed"],
             "이전 correction ledger는 consumed 3-event prefix여야 한다")
    request_id = events[0]["request_id"]
    _require(all(event["request_id"] == request_id for event in events),
             "이전 correction ledger request ID가 다르다")
    return {"row": row, "candidate": candidate, "fixed": fixed, "evidence": evidence,
            "owned": owned, "collected": collected, "collected_path": collected_path,
            "ledger_path": ledger_path, "ledger_events": events, "request_id": request_id,
            "stage": stage}


def _validate_sources(protocol_path: Path, manifest_path: Path, approval_path: Path,
                      smoke_root: Path, failed_execution: Path, *, anchor: str,
                      lab: Path) -> tuple[dict, dict, dict, dict]:
    protocol = verify_protocol(protocol_path, anchor=anchor, lab=lab)
    with V2._v1_scope():
        manifest, parent = V1._load_manifest(manifest_path)
        _require(manifest["anchor"] == anchor and Path(manifest["lab"]).resolve() == Path(lab).resolve(),
                 "parent manifest anchor/lab이 continuation과 다르다")
        _require(_sha(Path(manifest["protocol"]["path"]), lf=True)
                 == protocol["parent_protocol"]["sha256_lf"],
                 "manifest parent protocol이 continuation component와 다르다")
        smoke = V1.verify_smoke(smoke_root, manifest, parent)
        approval = V1._approval(approval_path, manifest["rows"])
        failure, _wait = _failure_boundary(Path(failed_execution).resolve())
        intent = _read(Path(failed_execution).resolve() / "intent.json")
        _require(intent.get("manifest") == _ref(manifest_path)
                 and intent.get("approval") == _ref(approval_path)
                 and intent.get("smoke") == smoke
                 and intent.get("automatic_retry") is False
                 and intent.get("model_calls") == 0,
                 "이전 execution intent의 manifest/approval/smoke 결합이 다르다")
        training = _completed_training(manifest, parent, Path(failed_execution).resolve())
    return manifest, approval, {"path": smoke["path"], "result_sha256": smoke["result_sha256"]}, {
        **training, "failure": failure,
    }


def prepare_inputs(protocol_path: Path, manifest_path: Path, approval_path: Path,
                   smoke_root: Path, failed_execution: Path, out: Path, *, anchor: str,
                   lab: Path) -> dict:
    out = Path(out).resolve()
    _require(not out.exists(), "continuation prepare-inputs out은 새 경로여야 한다")
    _manifest, _approval, smoke, prior = _validate_sources(
        protocol_path, manifest_path, approval_path, smoke_root, failed_execution,
        anchor=anchor, lab=lab)
    out.mkdir(parents=True, exist_ok=False)
    value = {
        "schema": MANIFEST_SCHEMA, "protocol": _ref(protocol_path), "anchor": anchor,
        "parent_manifest": _ref(manifest_path), "approval": _ref(approval_path),
        "smoke": {"root": str(Path(smoke_root).resolve()), **smoke},
        "failed_execution": {"root": str(Path(failed_execution).resolve()),
                             "failure": _ref(Path(failed_execution) / "failure.json")},
        "lab": str(Path(lab).resolve()), "row_key": {
            "case_id": prior["row"]["case_id"], "policy": prior["row"]["policy"]},
        "training": {"manifest": _ref(prior["stage"] / "train/manifest.json"),
                     "evidence": _ref(prior["stage"] / "train/training" / V1.GUARD.EVIDENCE),
                     "owned": _ref(prior["stage"] / "train/owned-result.json"),
                     "collected": _ref(prior["collected_path"]),
                     "checkpoint_sha256": prior["evidence"]["checkpoint"]["sha256"],
                     "updates": 300, "fresh_initialization": True},
        "ledger": {"source": _ref(prior["ledger_path"]), "events": 3,
                   "request_id": prior["request_id"]},
        "scope": {"new_training_jobs": 0, "new_model_calls": 0,
                  "new_correction_approvals": 0, "evaluation_only": True},
    }
    _write_new(out / "manifest.json", value)
    return deepcopy(value)


def _load_continuation(path: Path) -> tuple[dict, dict, dict, dict, dict]:
    value = _read(path)
    _require(value.get("schema") == MANIFEST_SCHEMA
             and value.get("scope") == {"new_training_jobs": 0, "new_model_calls": 0,
                                         "new_correction_approvals": 0, "evaluation_only": True},
             "continuation manifest schema/scope가 다르다")
    for name in ("protocol", "parent_manifest", "approval"):
        ref = value.get(name)
        _require(isinstance(ref, dict) and Path(ref.get("path", "")).is_file()
                 and _sha(Path(ref["path"])) == ref.get("sha256"),
                 f"continuation {name} 원문이 바뀌었다")
    manifest, approval, smoke, prior = _validate_sources(
        Path(value["protocol"]["path"]), Path(value["parent_manifest"]["path"]),
        Path(value["approval"]["path"]), Path(value["smoke"]["root"]),
        Path(value["failed_execution"]["root"]), anchor=value["anchor"], lab=Path(value["lab"]))
    _require(value["row_key"] == {"case_id": prior["row"]["case_id"],
                                  "policy": prior["row"]["policy"]}
             and value["smoke"] == {"root": str(Path(value["smoke"]["root"]).resolve()), **smoke}
             and value["failed_execution"]["failure"] == _ref(
                 Path(value["failed_execution"]["root"]) / "failure.json")
             and value["training"]["checkpoint_sha256"] == prior["evidence"]["checkpoint"]["sha256"]
             and value["training"]["manifest"] == _ref(prior["stage"] / "train/manifest.json")
             and value["training"]["evidence"] == _ref(
                 prior["stage"] / "train/training" / V1.GUARD.EVIDENCE)
             and value["training"]["owned"] == _ref(prior["stage"] / "train/owned-result.json")
             and value["training"]["collected"] == _ref(prior["collected_path"])
             and value["ledger"]["source"]["sha256"] == _sha(prior["ledger_path"])
             and value["ledger"]["request_id"] == prior["request_id"],
             "continuation manifest의 학습/ledger 결합이 다르다")
    return value, manifest, approval, smoke, prior


def _lease_context() -> dict:
    module = importlib.import_module("lineage_lease_context_v2")
    callback = getattr(module, "lease_context", None)
    _require(callable(callback), "lineage_lease_context_v2.lease_context가 없다")
    return callback()


def _wait(stage: Path, *, max_wait_s: float, interval_s: float, consecutive_clear: int) -> dict:
    return V1.GPU_WAIT.await_clear(
        stage, max_wait_s=max_wait_s, interval_s=interval_s,
        consecutive_clear=consecutive_clear, lease_check=_lease_context,
    )


def execute(manifest_path: Path, out: Path, *, runner=V2.OWNED.run_owned,
            wait: Callable = _wait) -> dict:
    out = Path(out).resolve()
    _require(not out.exists(), "continuation execute out은 새 경로여야 한다")
    value, manifest, _approval, smoke, prior = _load_continuation(manifest_path)
    out.mkdir(parents=True, exist_ok=False)
    old_ledger = prior["ledger_path"].read_bytes()
    ledger_path = out / "ledger/correction.jsonl"
    ledger_path.parent.mkdir(parents=True)
    with ledger_path.open("xb") as stream:
        stream.write(old_ledger)
        stream.flush()
        os.fsync(stream.fileno())
    _write_new(out / "ledger/prefix-provenance.json", {
        "source": _ref(prior["ledger_path"]), "copied_prefix_sha256": _sha(ledger_path),
        "copied_prefix_size": len(old_ledger), "events": 3,
    })
    receipts = []
    for row in manifest["rows"]:
        if row["status"] == "HOLD":
            receipt = {"schema": V1.RECEIPT_SCHEMA, "status": "HOLD",
                       "scope": "first_registered_correction_only", "case_id": row["case_id"],
                       "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
                       "approval": False, "model_calls": 0, "overall_goal_complete": False,
                       "second_epoch_registered": False}
            path = out / "receipts" / f"{row['case_id']}_{row['policy']}.json"
            _write_new(path, receipt)
            receipts.append({**receipt, **_ref(path)})
    row, evidence = prior["row"], prior["evidence"]
    stage = out / "evaluation"
    corrected, evaluation = V1._fixed_evaluate(
        stage, row, manifest, evidence, runner=V2._owned_runner(runner), wait=wait)
    reference_path = Path(row["reference_report"]["path"])
    _require(_sha(reference_path) == row["reference_report"]["sha256"],
             "same-seed reference report가 바뀌었다")
    verdict = V1.C.D1.recovery(corrected, _read(reference_path), _read(V1.BANDS)["bands"])
    bundle = {"evaluation_execution": evaluation, "corrected_report": _ref(stage / "corrected.json"),
              "reference_report": _ref(reference_path), "verdict": verdict,
              "reused_checkpoint_sha256": evidence["checkpoint"]["sha256"],
              "new_training_jobs": 0}
    _write_new(out / "result-bundle.json", bundle)
    ledger = V1.C.D1.CorrectionLedger(ledger_path, V1.C.RC.correction_registry())
    event = ledger.receipt(
        prior["request_id"], f"{row['case_id']}_{row['policy']}", verdict["outcome"],
        _sha(out / "result-bundle.json"), prior["owned"]["wall_s"] + evaluation["wall_s"], 0,
        reason="fixed_evaluation_undetermined" if verdict["outcome"] == "unknown" else None)
    _require(ledger_path.read_bytes().startswith(old_ledger)
             and prior["ledger_path"].read_bytes() == old_ledger,
             "ledger prefix 또는 이전 원문 ledger가 바뀌었다")
    receipt = V1._recovery_receipt(
        row, prior["candidate"], prior["ledger_events"][0]["evidence"]["context"],
        event, evidence, prior["collected"], stage / "corrected.json", reference_path,
        verdict, evaluation, smoke)
    receipt.update({"continuation": True, "training_reused": True, "new_training_jobs": 0,
                    "prior_failed_execution_sha256": value["failed_execution"]["failure"]["sha256"],
                    "prior_training_stage": _ref(prior["stage"] / "train/manifest.json")})
    receipt_path = out / "receipts" / f"{row['case_id']}_{row['policy']}.json"
    _write_new(receipt_path, receipt)
    receipts.append({**receipt, **_ref(receipt_path)})
    ordered = {(item["case_id"], item["policy"]): item for item in receipts}
    result = {"schema": RESULT_SCHEMA, "status": "complete",
              "scope": "first_registered_correction_evaluation_continuation",
              "receipts": [ordered[(row["case_id"], row["policy"])] for row in manifest["rows"]],
              "planned_rows": 4, "hold_rows": 3, "evaluated_rows": 1,
              "reused_training_rows": 1, "new_training_jobs": 0, "model_calls": 0,
              "new_correction_approvals": 0, "evaluation_conditions": 26,
              "reused_training_owned_wall_s": prior["owned"]["wall_s"],
              "same_seed_pairing": True, "overall_goal_complete": False,
              "prior_failure_sha256": value["failed_execution"]["failure"]["sha256"],
              "ledger_prefix_sha256": hashlib.sha256(old_ledger).hexdigest()}
    _write_new(out / "result.json", result)
    return deepcopy(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    public = sub.add_parser("prepare-protocol")
    public.add_argument("--parent-protocol", type=Path, required=True)
    public.add_argument("--out", type=Path, required=True)
    prep = sub.add_parser("prepare-inputs")
    for name in ("protocol", "manifest", "approval", "smoke-root", "failed-execution", "out", "lab"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--anchor", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.parent_protocol))
    elif args.command == "prepare-inputs":
        prepare_inputs(args.protocol, args.manifest, args.approval, args.smoke_root,
                       args.failed_execution, args.out, anchor=args.anchor, lab=args.lab)
    else:
        execute(args.manifest, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
