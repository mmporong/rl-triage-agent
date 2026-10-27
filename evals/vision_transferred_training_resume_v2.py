"""완료된 CORRECTED 학습을 다시 실행하지 않는 fixed-evaluation 전용 재개 경로."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    sys.path.insert(0, str(folder))

import vision_corrected_native_handoff_v1 as HANDOFF
import vision_transferred_e2e_v2 as E

from rl_triage.correction_ledger import CorrectionLedger

SCHEMA = "vision_transferred_evaluation_resume_protocol_v2"
RECEIPT_SCHEMA = "vision_transferred_evaluation_resume_receipt_v2"
INTENT_SCHEMA = "vision_transferred_evaluation_resume_intent_v2"
BRIDGE_RELATIVE = "evals/vision_transferred_training_resume_v2.py"
HANDOFF_RELATIVE = "evals/vision_corrected_native_handoff_v1.py"
RESUME_PROTOCOL_RELATIVE = "bench/protocols/vision_transferred_training_resume_20261027_v2.json"
GROUP = "source-01"
SEED = 2220
GROUPS = {"source-01": 2220, "source-02": 2221}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(repo: Path, path: Path, label: str, *, exists: bool = True) -> Path:
    repo = repo.resolve(strict=True)
    path = Path(path).absolute()
    resolved = path.resolve(strict=exists)
    require(repo == resolved or repo in resolved.parents, f"{label}가 저장소 경계 밖")
    if exists:
        require(resolved.is_file() and not resolved.is_symlink(), f"{label}는 regular file이어야 함")
    return resolved


def _ref(repo: Path, path: Path) -> dict[str, str]:
    path = _inside(repo, path, "portable reference")
    return {"path": path.relative_to(repo.resolve()).as_posix(), "sha256": E.ADAPTER.file_sha(path)}


def _open_ref(repo: Path, value: Mapping[str, Any], label: str) -> tuple[Path, Any]:
    path = _open_file_ref(repo, value, label)
    return path, _read(path)


def _open_file_ref(repo: Path, value: Mapping[str, Any], label: str) -> Path:
    require(set(value) == {"path", "sha256"}, f"{label} ref 형식 오류")
    path = _inside(repo, repo / str(value["path"]), label)
    require(E.ADAPTER.file_sha(path) == value["sha256"], f"{label} 원문 SHA 불일치")
    return path


def _old_unknown(ledger_path: Path, request_id: str, expected_sha: str) -> tuple[dict, dict]:
    require(E.ADAPTER.file_sha(ledger_path) == expected_sha, "기존 correction ledger SHA 변경")
    ledger = CorrectionLedger(ledger_path)
    request = ledger.requests().get(request_id)
    require(isinstance(request, dict) and request.get("state") == "done"
            and request.get("receipt", {}).get("verdict") == "unknown",
            "기존 correction 요청은 unknown 완료 상태여야 함")
    events = [row for row in ledger.events() if row.get("request_id") == request_id]
    require([row["event"] for row in events] == ["proposed", "approved", "consumed", "receipt"],
            "기존 correction 원장 순서 변경")
    return request, events[-1]


def _prepare_cached_group(repo: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    group, seed = str(raw["group"]), int(raw["seed"])
    require(group == "source-02" and seed == GROUPS[group], "cached group은 source-02만 허용")
    artifact_path, artifact = _open_ref(repo, _ref(repo, Path(raw["completed_artifact"])),
                                        "source-02 완료 artifact")
    checkpoint = Path(raw["external_checkpoint"]).resolve(strict=True)
    checkpoint_sha = str(raw["checkpoint_sha256"])
    require(artifact.get("arm") == "CORRECTED" and artifact.get("seed") == seed
            and artifact.get("checkpoint", {}).get("sha256") == checkpoint_sha
            and checkpoint.is_file() and not checkpoint.is_symlink()
            and E.ADAPTER.file_sha(checkpoint) == checkpoint_sha,
            "source-02 완료 artifact/checkpoint 불일치")
    old_ledger = _ref(repo, Path(raw["old_correction_ledger"]))
    request, receipt = _old_unknown(Path(raw["old_correction_ledger"]), str(raw["old_request_id"]),
                                    old_ledger["sha256"])
    fixed = {name: _ref(repo, Path(raw["fixed_evaluation"][name]))
             for name in ("job", "worker", "execution", "none_report", "corrected_report")}
    copy_relative, namespace_relative = str(raw["checkpoint_copy_relative"]), str(raw["output_namespace_relative"])
    copy_path = _inside(repo, repo / copy_relative, "source-02 checkpoint copy", exists=False)
    namespace = (repo / namespace_relative).resolve(strict=False)
    require(repo in namespace.parents and namespace not in copy_path.parents and copy_path not in namespace.parents
            and not copy_path.exists() and not namespace.exists(), "source-02 resume 출력 경계 불일치")
    return {"group": group, "seed": seed, "mode": "cached_fixed_evaluation",
        "corrected_job_id": artifact_path.parent.name,
        "completed_training": {"artifact": _ref(repo, artifact_path),
            "ledger": _ref(repo, Path(raw["training_ledger"])),
            "job": _ref(repo, artifact_path.parent / "job.json"),
            "worker": _ref(repo, artifact_path.parent / "worker.json"),
            "execution": _ref(repo, artifact_path.parent / "execution.json"),
            "checkpoint_sha256": checkpoint_sha},
        "old_failure": {"ledger": old_ledger, "request_id": str(raw["old_request_id"]),
            "unknown_receipt_sha256": E.ADAPTER.digest(receipt),
            "failure": _ref(repo, Path(raw["old_failure"])),
            "source_checkpoint_sha256": request["evidence"]["source_checkpoint_sha256"]},
        "fixed_evaluation": fixed, "checkpoint_copy_relative": copy_relative,
        "output_namespace_relative": namespace_relative,
        "observed_development_result": True}


def _validate_cached_group(repo: Path, value: object) -> dict[str, Any]:
    required = {"group", "seed", "mode", "corrected_job_id", "completed_training", "old_failure",
                "fixed_evaluation", "checkpoint_copy_relative", "output_namespace_relative",
                "observed_development_result"}
    require(isinstance(value, dict) and set(value) == required and value["group"] == "source-02"
            and value["seed"] == 2221 and value["mode"] == "cached_fixed_evaluation"
            and value["observed_development_result"] is True, "source-02 cached group schema 불일치")
    completed = value["completed_training"]
    require(isinstance(completed, dict)
            and set(completed) == {"artifact", "ledger", "job", "worker", "execution", "checkpoint_sha256"},
            "source-02 완료 학습 결합 오류")
    artifact_path, artifact = _open_ref(repo, completed["artifact"], "source-02 완료 artifact")
    _open_file_ref(repo, completed["ledger"], "source-02 training ledger")
    bodies = {}
    for name in ("job", "worker", "execution"):
        path, body = _open_ref(repo, completed[name], f"source-02 training {name}")
        require(path.parent == artifact_path.parent, f"source-02 training {name} stage 불일치")
        bodies[name] = body
    require(artifact.get("arm") == "CORRECTED" and artifact.get("seed") == 2221
            and artifact.get("checkpoint", {}).get("sha256") == completed["checkpoint_sha256"]
            and bodies["job"].get("mode") == "train" and bodies["worker"].get("completed") is True
            and bodies["execution"].get("completed") is True and bodies["execution"].get("exit_code") == 0
            and bodies["execution"].get("cleanup_verified") is True, "source-02 완료 학습 상태 불일치")
    old = value["old_failure"]
    old_path = _open_file_ref(repo, old["ledger"], "source-02 기존 correction ledger")
    request, receipt = _old_unknown(old_path, old["request_id"], old["ledger"]["sha256"])
    require(E.ADAPTER.digest(receipt) == old["unknown_receipt_sha256"]
            and request["evidence"]["source_checkpoint_sha256"] == old["source_checkpoint_sha256"],
            "source-02 기존 unknown receipt 변경")
    _open_ref(repo, old["failure"], "source-02 기존 실패 원문")
    fixed = value["fixed_evaluation"]
    require(isinstance(fixed, dict)
            and set(fixed) == {"job", "worker", "execution", "none_report", "corrected_report"},
            "source-02 fixed evaluation ref 오류")
    raw = {name: _open_ref(repo, ref, f"source-02 fixed {name}")[1] for name, ref in fixed.items()}
    descriptors = raw["worker"].get("reports")
    require(raw["job"].get("mode") == "evaluate" and raw["worker"].get("completed") is True
            and raw["execution"].get("completed") is True and raw["execution"].get("exit_code") == 0
            and raw["execution"].get("cleanup_verified") is True and isinstance(descriptors, Mapping)
            and raw["none_report"].get("name") == "NONE"
            and raw["corrected_report"].get("name") == "CORRECTED"
            and descriptors.get("NONE", {}).get("sha256") == fixed["none_report"]["sha256"]
            and descriptors.get("CORRECTED", {}).get("sha256") == fixed["corrected_report"]["sha256"],
            "source-02 fixed evaluation 완료/arm/SHA 불일치")
    copy_path = _inside(repo, repo / value["checkpoint_copy_relative"],
                        "source-02 checkpoint copy", exists=False)
    namespace = (repo / value["output_namespace_relative"]).resolve(strict=False)
    require(repo in namespace.parents and namespace not in copy_path.parents and copy_path not in namespace.parents,
            "source-02 resume namespace/checkpoint copy 경계 불일치")
    return value


def prepare_protocol(
    repo: Path,
    *,
    source_commit: str,
    source_sha256_lf: Mapping[str, str],
    execution_protocol: Path,
    completed_artifact: Path,
    training_ledger: Path,
    old_correction_ledger: Path,
    old_request_id: str,
    old_failure: Path,
    external_checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_copy_relative: str,
    output_namespace_relative: str,
    cached_group: Mapping[str, Any],
    deadline_utc: str,
    approver: str,
    output: Path,
    evidence_factory: Callable[[Path], Any] = E.ObjectEvidence,
) -> dict[str, Any]:
    """실행 전에 완료 학습·unknown 실패·새 평가 한도를 하나의 protocol로 고정한다."""
    repo = repo.resolve(strict=True)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "resume protocol 출력은 새 파일이어야 함")
    require(bool(approver.strip()) and len(source_commit) == 40, "승인자/source commit 형식 오류")
    required_sources = {BRIDGE_RELATIVE, HANDOFF_RELATIVE}
    require(required_sources <= set(source_sha256_lf), "resume/native handoff source closure 누락")
    evidence = evidence_factory(repo)
    for relative, sha in source_sha256_lf.items():
        current = E.ADAPTER.file_sha(repo / relative, lf=True)
        _, blob = evidence.file(source_commit, relative)
        committed = E.hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
        require(current == sha == committed, f"source closure 불일치: {relative}")
    artifact_path, artifact = _open_ref(repo, _ref(repo, completed_artifact), "완료 artifact")
    require(artifact.get("arm") == "CORRECTED" and artifact.get("seed") == SEED,
            "완료 artifact identity 오류")
    checkpoint = Path(external_checkpoint).resolve(strict=True)
    require(checkpoint.is_file() and not checkpoint.is_symlink()
            and E.ADAPTER.file_sha(checkpoint) == checkpoint_sha256
            and artifact.get("checkpoint", {}).get("sha256") == checkpoint_sha256,
            "외부 checkpoint 원문/SHA 불일치")
    ledger_ref = _ref(repo, training_ledger)
    old_ledger_ref = _ref(repo, old_correction_ledger)
    request, receipt = _old_unknown(Path(old_correction_ledger), old_request_id, old_ledger_ref["sha256"])
    failure_ref = _ref(repo, old_failure)
    copy_path = _inside(repo, repo / checkpoint_copy_relative, "checkpoint copy", exists=False)
    namespace = (repo / output_namespace_relative).resolve(strict=False)
    require(repo in copy_path.parents and repo in namespace.parents
            and namespace not in copy_path.parents and copy_path not in namespace.parents
            and not copy_path.exists() and not namespace.exists(), "resume 출력 경계는 분리된 새 repo 내부 경로여야 함")
    value = {
        "schema": SCHEMA,
        "protocol_id": "vision-transferred-evaluation-resume-source-01-v1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_utc": deadline_utc,
        "source_commit": source_commit,
        "source_sha256_lf": dict(source_sha256_lf),
        "execution_protocol": _ref(repo, execution_protocol),
        "group": GROUP,
        "seed": SEED,
        "corrected_job_id": artifact_path.parent.name,
        "completed_training": {"artifact": _ref(repo, artifact_path), "ledger": ledger_ref,
                               "job": _ref(repo, artifact_path.parent / "job.json"),
                               "worker": _ref(repo, artifact_path.parent / "worker.json"),
                               "execution": _ref(repo, artifact_path.parent / "execution.json"),
                               "checkpoint_sha256": checkpoint_sha256},
        "old_failure": {"ledger": old_ledger_ref, "request_id": old_request_id,
                        "unknown_receipt_sha256": E.ADAPTER.digest(receipt), "failure": failure_ref,
                        "source_checkpoint_sha256": request["evidence"]["source_checkpoint_sha256"]},
        "checkpoint_copy_relative": checkpoint_copy_relative,
        "output_namespace_relative": output_namespace_relative,
        "limits": {"training_runs": 0, "model_calls": 0,
                   "new_fixed_evaluations": {"source-01": 1, "source-02": 0}},
        "evaluation_approval": {"approved": True, "approver": approver,
                                "purpose": "completed_training_fixed_evaluation_only"},
        "cached_group": _prepare_cached_group(repo, cached_group),
    }
    E.ADAPTER._write_new(output, value)
    validate_protocol(output, repo=repo, require_current=True, evidence_factory=evidence_factory)
    return value


def validate_protocol(path: Path, *, repo: Path = ROOT, require_current: bool = True,
                      now: datetime | None = None,
                      evidence_factory: Callable[[Path], Any] = E.ObjectEvidence) -> dict[str, Any]:
    repo, path = repo.resolve(strict=True), Path(path).resolve(strict=True)
    value = _read(path)
    required = {"schema", "protocol_id", "registered_at_utc", "deadline_utc", "source_commit",
                "source_sha256_lf", "group", "seed", "corrected_job_id", "completed_training",
                "execution_protocol", "old_failure", "checkpoint_copy_relative", "output_namespace_relative", "limits",
                "evaluation_approval", "cached_group"}
    require(isinstance(value, dict) and set(value) == required and value["schema"] == SCHEMA
            and value["group"] == GROUP and value["seed"] == SEED
            and value["limits"] == {"training_runs": 0, "model_calls": 0,
                                    "new_fixed_evaluations": {"source-01": 1, "source-02": 0}}
            and value["evaluation_approval"].get("approved") is True,
            "evaluation resume protocol 형식/한도 불일치")
    registered = E.ADAPTER._time(value["registered_at_utc"], "resume registered")
    deadline = E.ADAPTER._time(value["deadline_utc"], "resume deadline")
    current = now or datetime.now(timezone.utc)
    require(registered < deadline and (not require_current or registered <= current < deadline),
            "evaluation resume protocol 시간 창 밖")
    evidence = evidence_factory(repo)
    sources = value["source_sha256_lf"]
    require(isinstance(sources, dict) and {BRIDGE_RELATIVE, HANDOFF_RELATIVE} <= set(sources),
            "evaluation resume source closure 누락")
    for relative, sha in sources.items():
        require(E.ADAPTER.file_sha(repo / relative, lf=True) == sha, f"현재 source SHA 변경: {relative}")
        _, blob = evidence.file(value["source_commit"], relative)
        require(E.hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == sha,
                f"commit source blob 변경: {relative}")
    _open_ref(repo, value["execution_protocol"], "등록 E2E execution protocol")
    completed = value["completed_training"]
    require(set(completed) == {"artifact", "ledger", "job", "worker", "execution", "checkpoint_sha256"},
            "완료 학습 사전등록 파일 결합 오류")
    artifact_path, artifact = _open_ref(repo, completed["artifact"], "완료 artifact")
    ledger_path = _open_file_ref(repo, value["completed_training"]["ledger"], "training ledger")
    completed_bodies: dict[str, Any] = {}
    for name in ("job", "worker", "execution"):
        bound, body = _open_ref(repo, completed[name], f"완료 training {name}")
        require(bound.parent == artifact_path.parent, f"완료 training {name} stage 불일치")
        completed_bodies[name] = body
    require(completed_bodies["job"].get("mode") == "train"
            and completed_bodies["worker"].get("completed") is True
            and completed_bodies["execution"].get("completed") is True
            and completed_bodies["execution"].get("exit_code") == 0
            and completed_bodies["execution"].get("cleanup_verified") is True,
            "완료 training job/worker/execution 상태 불일치")
    require(artifact.get("arm") == "CORRECTED" and artifact.get("seed") == SEED
            and artifact.get("checkpoint", {}).get("sha256") == value["completed_training"]["checkpoint_sha256"],
            "등록 완료 학습 identity 불일치")
    old_path = _open_file_ref(repo, value["old_failure"]["ledger"], "기존 correction ledger")
    request, receipt = _old_unknown(old_path, value["old_failure"]["request_id"],
                                    value["old_failure"]["ledger"]["sha256"])
    require(E.ADAPTER.digest(receipt) == value["old_failure"]["unknown_receipt_sha256"]
            and request["evidence"]["source_checkpoint_sha256"] == value["old_failure"]["source_checkpoint_sha256"],
            "기존 unknown receipt/source 결합 변경")
    _open_ref(repo, value["old_failure"]["failure"], "기존 실패 원문")
    del ledger_path
    copy_path = _inside(repo, repo / value["checkpoint_copy_relative"], "checkpoint copy", exists=False)
    namespace = (repo / value["output_namespace_relative"]).resolve(strict=False)
    require(repo in namespace.parents and namespace not in copy_path.parents and copy_path not in namespace.parents,
            "resume namespace/checkpoint copy 경계 불일치")
    _validate_cached_group(repo, value["cached_group"])
    return value


def _copy_checkpoint(repo: Path, source: Path, target_relative: str, expected_sha: str) -> Path:
    source = Path(source).resolve(strict=True)
    require(source.is_file() and not source.is_symlink() and E.ADAPTER.file_sha(source) == expected_sha,
            "완료 checkpoint 원문/SHA 불일치")
    target = _inside(repo, repo / target_relative, "checkpoint copy", exists=False)
    require(not target.exists() and not target.is_symlink(), "checkpoint copy는 새 경로여야 함")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush(); os.fsync(dst.fileno())
    require(E.ADAPTER.file_sha(target) == expected_sha, "checkpoint copy SHA 불일치")
    return target


def _training_evidence(repo: Path, backend: Any, job_id: str, artifact: Mapping[str, Any],
                       checkpoint_copy: Path) -> dict[str, Any]:
    events = backend._events()
    matches = [row for row in events if row.get("job_id") == job_id]
    require(matches and matches[-1].get("event") == "completed", "완료 training ledger event 누락")
    event, inventory = matches[-1], matches[-1].get("inventory")
    require(isinstance(inventory, Mapping), "완료 training inventory 누락")
    required = inventory.get("required_files")
    require(isinstance(required, Mapping) and set(required) == {"readback", "records", "execution"},
            "완료 training required files 불일치")
    files = {
        "artifact": _ref(repo, Path(str(event["artifact_path"]))),
        "checkpoint": _ref(repo, checkpoint_copy),
        "telemetry": _ref(repo, Path(str(artifact["telemetry"]["path"]))),
        "manifest": _ref(repo, Path(str(inventory["manifest"]["path"]))),
        **{name: _ref(repo, Path(str(required[name]["path"])))
           for name in ("readback", "records", "execution")},
    }
    return {"schema": "vision_selection_training_evidence_v1",
            "ledger": _ref(repo, backend.ledger_path),
            "completed_event_sha256": E.ADAPTER.digest(event), "files": files}


def fixed_labels_backend_names(context: Mapping[str, Any], reports: object,
                               none_sha: str, corrected_sha: str) -> tuple[dict[str, str], dict[str, Any]]:
    """backend 원문 arm 이름(NONE/CORRECTED)만 명시적으로 매핑한다."""
    require(isinstance(reports, Mapping) and set(reports) == {"none", "corrected"},
            "fixed evaluation reports arm이 다름")
    protocol_sha = context["protocol"]["execution_bindings"]["fixed_eval_protocol"]["sha256_lf"]
    checked: dict[str, Any] = {}
    for arm, backend_name, checkpoint_sha in (
        ("none", "NONE", none_sha), ("corrected", "CORRECTED", corrected_sha),
    ):
        report = reports[arm]
        require(isinstance(report, dict) and report.get("name") == backend_name,
                f"{arm} report의 backend arm 이름 불일치")
        checked[arm] = E.ADAPTER.GROUNDED.validate_fixed_report(report, {
            "name": backend_name, "checkpoint_file": report.get("checkpoint", {}).get("file"),
            "checkpoint_sha256": checkpoint_sha, "action_scale": 0.25,
            "protocol_sha256_lf": protocol_sha,
        })
    none_behavior = E.ADAPTER.GROUNDED._behavior(checked["none"])
    corrected_behavior = E.ADAPTER.GROUNDED._behavior(checked["corrected"])
    bands = E.ADAPTER.RECOVERY.derive_bands([
        E.ADAPTER.RECOVERY.compare(none_behavior, none_behavior),
    ])
    labels = {
        "none": E.ADAPTER.RECOVERY.verdict(
            E.ADAPTER.RECOVERY.compare(none_behavior, none_behavior), bands,
        )["label"],
        "corrected": E.ADAPTER.RECOVERY.verdict(
            E.ADAPTER.RECOVERY.compare(corrected_behavior, none_behavior), bands,
        )["label"],
    }
    return labels, checked


def validate_correction_result_backend_names(context: Mapping[str, Any], group: str,
                                             result: Mapping[str, Any]) -> dict[str, Any]:
    """기존 validator의 모든 경계를 유지하고 backend arm 이름만 대문자로 검증한다."""
    E.ADAPTER.refresh_context(context, require_active=False)
    required = {"schema", "status", "group", "job_id", "checkpoint_sha256", "training",
                "training_evidence", "evaluation", "fixed_eval", "gpu_s", "compute_usd", "exit_code"}
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == group)
    training, evaluation = result.get("training"), result.get("evaluation")
    checkpoint = training.get("checkpoint") if isinstance(training, Mapping) else None
    require(set(result) == required and result.get("schema") == "vision_selection_correction_result_v1"
            and result.get("status") == "complete" and result.get("group") == group
            and result.get("job_id") == entry["corrected_job_id"] and result.get("exit_code") == 0
            and isinstance(training, Mapping) and training.get("arm") == "CORRECTED"
            and training.get("seed") == entry["seed"]
            and training.get("source_checkpoint_sha256") == entry["checkpoints"]["F1"]
            and isinstance(checkpoint, Mapping)
            and result.get("checkpoint_sha256") == checkpoint.get("sha256")
            and isinstance(evaluation, Mapping) and set(evaluation) == {"reports", "execution", "evidence"},
            "cached correction identity/schema가 다름")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink()
            and E.ADAPTER.file_sha(checkpoint_path) == checkpoint.get("sha256"),
            "cached CORRECTED checkpoint 원문/SHA가 다름")
    E.ADAPTER._validate_training_evidence(Path(context["root"]), str(result["job_id"]), training,
                                          result.get("training_evidence"))
    E.ADAPTER._validate_evaluation_evidence(Path(context["root"]), evaluation,
        entry["checkpoints"]["NONE"], str(checkpoint["sha256"]))
    gpu_s = (E.ADAPTER._execution_gpu_s(training.get("execution"), "cached training", context,
                                        bind_registered_source=True)
             + E.ADAPTER._execution_gpu_s(evaluation.get("execution"), "cached fixed evaluation", context,
                                          bind_registered_source=True))
    labels, reports = fixed_labels_backend_names(context, evaluation.get("reports"),
        entry["checkpoints"]["NONE"], str(checkpoint["sha256"]))
    compute_usd = E.ADAPTER._action_cost(context, entry, "correction", gpu_s)
    require(evaluation.get("reports") == reports and result.get("fixed_eval") == labels
            and result.get("gpu_s") == gpu_s and result.get("compute_usd") == compute_usd,
            "cached correction 평가/비용 재계산이 다름")
    return dict(result)


def _group_protocol(protocol: Mapping[str, Any], group: str) -> Mapping[str, Any]:
    require(group in GROUPS, "resume group 오류")
    return protocol if group == "source-01" else protocol["cached_group"]


def _write_resume_result(repo: Path, protocol_path: Path, protocol: Mapping[str, Any],
                         group_protocol: Mapping[str, Any], context: Mapping[str, Any], group: str,
                         artifact: dict[str, Any], training_evidence: dict[str, Any], fixed: dict[str, Any],
                         checkpoint_copy: Path, namespace: Path, *, new_fixed_evaluations: int) -> dict[str, Any]:
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == group)
    train_gpu_s = E.ADAPTER._execution_gpu_s(artifact["execution"], "cached training", context,
                                              bind_registered_source=True)
    eval_gpu_s = E.ADAPTER._execution_gpu_s(fixed["execution"], "cached fixed evaluation", context,
                                             bind_registered_source=True)
    labels, reports = fixed_labels_backend_names(context, fixed["reports"], entry["checkpoints"]["NONE"],
                                                 artifact["checkpoint"]["sha256"])
    gpu_s = train_gpu_s + eval_gpu_s
    result = {"schema": "vision_selection_correction_result_v1", "status": "complete",
        "group": group, "job_id": group_protocol["corrected_job_id"],
        "checkpoint_sha256": artifact["checkpoint"]["sha256"], "training": artifact,
        "training_evidence": training_evidence,
        "evaluation": {"reports": reports, "execution": fixed["execution"], "evidence": fixed["evidence"]},
        "fixed_eval": labels, "gpu_s": gpu_s,
        "compute_usd": E.ADAPTER._action_cost(context, entry, "correction", gpu_s), "exit_code": 0}
    validate_correction_result_backend_names(context, group, result)
    result_output = namespace / "correction.result.json"
    E.ADAPTER._write_new(result_output, result)
    receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "group": group,
        "protocol": _ref(repo, protocol_path), "result": _ref(repo, result_output),
        "intent": _ref(repo, namespace / "evaluation.intent.json"),
        "evaluation_approval": _ref(repo, namespace / "evaluation.approval.json"),
        "old_unknown_receipt_sha256": group_protocol["old_failure"]["unknown_receipt_sha256"],
        "completed_training_artifact_sha256": group_protocol["completed_training"]["artifact"]["sha256"],
        "checkpoint_copy": _ref(repo, checkpoint_copy), "training_runs": 0, "model_calls": 0,
        "new_fixed_evaluations": new_fixed_evaluations,
        "evaluation_origin": "new" if new_fixed_evaluations else "registered_completed_cache",
        "evaluation_gpu_s": eval_gpu_s,
        "verdict": "recovered" if labels == {"none": "healthy", "corrected": "healthy"} else "not_recovered"}
    E.ADAPTER._write_new(namespace / "evaluation.receipt.json", receipt)
    return result


def resume_evaluation(protocol_path: Path, context: Mapping[str, Any], service: Any, *,
                      external_checkpoint: Path, result_output: Path, receipt_output: Path) -> dict[str, Any]:
    repo = Path(context["root"]).resolve(strict=True)
    protocol = validate_protocol(protocol_path, repo=repo)
    require(protocol["execution_protocol"] == context["protocol_reference"],
            "evaluation resume protocol과 현재 E2E execution protocol 불일치")
    namespace = (repo / protocol["output_namespace_relative"]).resolve(strict=False)
    result_output, receipt_output = Path(result_output).absolute(), Path(receipt_output).absolute()
    require(result_output == namespace / "correction.result.json"
            and receipt_output == namespace / "evaluation.receipt.json",
            "resume result/receipt 이름이 등록 계약과 다름")
    require(namespace in result_output.parents and namespace in receipt_output.parents,
            "resume result/receipt가 등록 namespace 밖")
    require(not namespace.exists() and not result_output.exists() and not receipt_output.exists(),
            "evaluation-only resume는 새 namespace에서 한 번만 실행")
    E.ADAPTER.verify_correction_service(context, service)
    require(service.preflight() is True, "등록 NONE 완료 cache가 필요함")
    backend, job_id = service.backend, protocol["corrected_job_id"]
    artifact = backend._completed(job_id)
    require(isinstance(artifact, dict), "완료 CORRECTED artifact가 없음")
    _, registered_artifact = _open_ref(repo, protocol["completed_training"]["artifact"], "등록 완료 artifact")
    require(artifact == registered_artifact, "backend 완료 artifact가 등록 원문과 다름")
    old_path = _open_file_ref(repo, protocol["old_failure"]["ledger"], "기존 correction ledger")
    _old_unknown(old_path, protocol["old_failure"]["request_id"], protocol["old_failure"]["ledger"]["sha256"])
    checkpoint_copy = _copy_checkpoint(repo, external_checkpoint, protocol["checkpoint_copy_relative"],
                                       protocol["completed_training"]["checkpoint_sha256"])
    training_evidence = _training_evidence(repo, backend, job_id, artifact, checkpoint_copy)
    namespace.mkdir(parents=True, exist_ok=False)
    approval_path = namespace / "evaluation.approval.json"
    E.ADAPTER._write_new(approval_path, {"schema": "vision_transferred_evaluation_resume_approval_v2",
        "approved": True, "protocol": _ref(repo, protocol_path),
        "approver": protocol["evaluation_approval"]["approver"],
        "purpose": protocol["evaluation_approval"]["purpose"],
        "training_runs": 0, "model_calls": 0, "new_fixed_evaluations": 1})
    intent = namespace / "evaluation.intent.json"
    E.ADAPTER._write_new(intent, {"schema": INTENT_SCHEMA, "protocol": _ref(repo, protocol_path), "group": GROUP,
        "completed_artifact_sha256": protocol["completed_training"]["artifact"]["sha256"],
        "checkpoint_copy": _ref(repo, checkpoint_copy), "training_runs": 0, "model_calls": 0,
        "new_fixed_evaluations_reserved": 1, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    entry = next(row for row in context["protocol"]["groups"] if row["group"] == GROUP)
    E.ADAPTER.require_action_window(context, entry, "correction")
    fixed = service.evaluate(artifact)
    require(isinstance(fixed, dict) and set(fixed) == {"reports", "execution", "evidence"},
            "fixed evaluation 결과 형식 오류")
    return _write_resume_result(repo, protocol_path, protocol, protocol, context, GROUP, artifact,
                                training_evidence, fixed, checkpoint_copy, namespace,
                                new_fixed_evaluations=1)


def evaluate_cached_group(protocol_path: Path, context: Mapping[str, Any], service: Any, *,
                          external_checkpoint: Path) -> dict[str, Any]:
    """source-02에서 이미 완료된 fixed-eval 원문을 GPU 호출 없이 재구성한다."""
    repo = Path(context["root"]).resolve(strict=True)
    protocol = validate_protocol(protocol_path, repo=repo)
    require(protocol["execution_protocol"] == context["protocol_reference"],
            "cached resume protocol과 현재 E2E execution protocol 불일치")
    group, bound = "source-02", protocol["cached_group"]
    namespace = (repo / bound["output_namespace_relative"]).resolve(strict=False)
    require(not namespace.exists(), "source-02 cached resume namespace는 새 경로여야 함")
    E.ADAPTER.verify_correction_service(context, service)
    require(service.preflight() is True, "등록 NONE 완료 cache가 필요함")
    artifact = service.backend._completed(bound["corrected_job_id"])
    require(isinstance(artifact, dict), "source-02 완료 CORRECTED artifact가 없음")
    _, registered = _open_ref(repo, bound["completed_training"]["artifact"], "source-02 등록 완료 artifact")
    require(artifact == registered, "source-02 backend 완료 artifact가 등록 원문과 다름")
    old_path = _open_file_ref(repo, bound["old_failure"]["ledger"], "source-02 correction ledger")
    _old_unknown(old_path, bound["old_failure"]["request_id"], bound["old_failure"]["ledger"]["sha256"])
    checkpoint_copy = _copy_checkpoint(repo, external_checkpoint, bound["checkpoint_copy_relative"],
                                       bound["completed_training"]["checkpoint_sha256"])
    training_evidence = _training_evidence(repo, service.backend, bound["corrected_job_id"],
                                           artifact, checkpoint_copy)
    fixed_raw = {name: _open_ref(repo, reference, f"source-02 fixed {name}")[1]
                 for name, reference in bound["fixed_evaluation"].items()}
    fixed = {"reports": {"none": fixed_raw["none_report"],
                          "corrected": fixed_raw["corrected_report"]},
             "execution": fixed_raw["execution"], "evidence": dict(bound["fixed_evaluation"])}
    namespace.mkdir(parents=True, exist_ok=False)
    E.ADAPTER._write_new(namespace / "evaluation.approval.json", {
        "schema": "vision_transferred_evaluation_resume_approval_v2", "approved": True,
        "protocol": _ref(repo, protocol_path), "approver": protocol["evaluation_approval"]["approver"],
        "purpose": "completed_training_completed_fixed_evaluation_cache_only",
        "training_runs": 0, "model_calls": 0, "new_fixed_evaluations": 0})
    E.ADAPTER._write_new(namespace / "evaluation.intent.json", {"schema": INTENT_SCHEMA,
        "protocol": _ref(repo, protocol_path), "group": group,
        "completed_artifact_sha256": bound["completed_training"]["artifact"]["sha256"],
        "checkpoint_copy": _ref(repo, checkpoint_copy), "training_runs": 0, "model_calls": 0,
        "new_fixed_evaluations_reserved": 0, "cached_fixed_evaluation": bound["fixed_evaluation"],
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return _write_resume_result(repo, protocol_path, protocol, bound, context, group, artifact,
                                training_evidence, fixed, checkpoint_copy, namespace,
                                new_fixed_evaluations=0)


def validate_resume_result(context: Mapping[str, Any], protocol_path: Path, group: str) -> tuple[Path, dict, dict]:
    repo = Path(context["root"]).resolve(strict=True)
    protocol = validate_protocol(protocol_path, repo=repo, require_current=False)
    bound = _group_protocol(protocol, group)
    namespace = repo / bound["output_namespace_relative"]
    result_path, receipt_path = namespace / "correction.result.json", namespace / "evaluation.receipt.json"
    result, receipt = _read(result_path), _read(receipt_path)
    receipt_keys = {"schema", "status", "protocol", "result", "intent", "evaluation_approval",
        "old_unknown_receipt_sha256", "completed_training_artifact_sha256", "checkpoint_copy",
        "training_runs", "model_calls", "new_fixed_evaluations", "evaluation_origin",
        "evaluation_gpu_s", "verdict", "group"}
    require(set(receipt) == receipt_keys and receipt.get("schema") == RECEIPT_SCHEMA
            and receipt.get("status") == "complete"
            and receipt.get("group") == group and receipt.get("training_runs") == 0
            and receipt.get("model_calls") == 0
            and receipt.get("new_fixed_evaluations") == (1 if group == "source-01" else 0)
            and receipt.get("evaluation_origin") == ("new" if group == "source-01" else "registered_completed_cache")
            and receipt.get("result") == _ref(repo, result_path)
            and receipt.get("old_unknown_receipt_sha256") == bound["old_failure"]["unknown_receipt_sha256"],
            "evaluation-only receipt 형식/결합 오류")
    _, approval = _open_ref(repo, receipt["evaluation_approval"], "evaluation-only approval")
    require(approval == {"schema": "vision_transferred_evaluation_resume_approval_v2", "approved": True,
        "protocol": _ref(repo, protocol_path), "approver": protocol["evaluation_approval"]["approver"],
        "purpose": (protocol["evaluation_approval"]["purpose"] if group == "source-01"
                    else "completed_training_completed_fixed_evaluation_cache_only"),
        "training_runs": 0, "model_calls": 0,
        "new_fixed_evaluations": 1 if group == "source-01" else 0}, "evaluation-only approval 변경")
    checked = validate_correction_result_backend_names(context, group, result)
    expected = "recovered" if checked["fixed_eval"] == {"none": "healthy", "corrected": "healthy"} else "not_recovered"
    require(receipt.get("verdict") == expected, "evaluation-only receipt verdict 불일치")
    return result_path, checked, receipt


class ResumeRuntime(E.Runtime):
    """두 seed의 새 evaluation-only receipt를 사용하는 기존 E2E runtime."""

    def _resume_protocol(self) -> Path:
        path = self.root / RESUME_PROTOCOL_RELATIVE
        protocol = validate_protocol(path, repo=self.root, require_current=False)
        require(protocol["execution_protocol"] == self.context["protocol_reference"],
                "resume protocol이 현재 E2E execution protocol을 가리키지 않음")
        return path.resolve(strict=True)

    def recording_bindings(self) -> dict:
        self.refresh()
        protocol_path = self._resume_protocol()
        protocol = validate_protocol(protocol_path, repo=self.root, require_current=False)
        verified = {group: validate_resume_result(self.context, protocol_path, group)
                    for group in GROUPS}
        require(all(receipt["verdict"] == "recovered" for _, _, receipt in verified.values()),
                "두 seed evaluation-only 결과가 모두 recovered여야 함")
        bindings = {}
        for group, seed in GROUPS.items():
            result_path, result, _receipt = verified[group]
            bound = _group_protocol(protocol, group)
            request, _ = _old_unknown(_open_file_ref(self.root, bound["old_failure"]["ledger"],
                                                      f"{group} correction ledger"),
                                      bound["old_failure"]["request_id"], bound["old_failure"]["ledger"]["sha256"])
            approval = self.path / "record-approvals-resume" / f"{group}.json"
            value = {"schema": "vision_corrected_record_approval_v1", "approved": True,
                "request_id": f"evaluation-resume-{bound['old_failure']['request_id']}",
                "corrected_job_id": result["job_id"], "seed": seed,
                "correction_id": request["correction_id"], "recovery_mode": request["recovery_mode"],
                "protocol_case_arm": request["protocol_case_arm"],
                "correction_result_sha256": E.ADAPTER.file_sha(result_path), "receipt_verdict": "recovered"}
            if approval.exists():
                require(E.ADAPTER._read(approval) == value, f"기존 {group} resume native 승인 변경")
            else:
                E.ADAPTER._write_new(approval, value)
            bindings[f"corrected-s{seed}"] = {"correction_result": result_path,
                                               "approval_receipt": approval}
        return bindings


def verify_native_resume_closure(repo: Path, bound_protocol: Path, *,
                                 evidence_factory: Callable[[Path], Any] = E.ObjectEvidence) -> dict:
    repo = repo.resolve(strict=True)
    protocol = _read(Path(bound_protocol).resolve(strict=True))
    sources, commit = protocol.get("source_sha256_lf"), protocol.get("source_commit")
    require(isinstance(sources, dict) and BRIDGE_RELATIVE in sources,
            "native protocol에 evaluation resume bridge가 결합되지 않음")
    require(sources[BRIDGE_RELATIVE] == E.ADAPTER.file_sha(repo / BRIDGE_RELATIVE, lf=True),
            "native protocol evaluation resume bridge SHA 불일치")
    _, blob = evidence_factory(repo).file(commit, BRIDGE_RELATIVE)
    require(E.hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == sources[BRIDGE_RELATIVE],
            "native protocol resume bridge commit blob 불일치")
    return protocol


def handoff_prepare(*args: Any, evidence_factory: Callable[[Path], Any] = E.ObjectEvidence,
                    **kwargs: Any) -> dict:
    repo, bound_protocol = Path(args[0]), Path(args[7])
    verify_native_resume_closure(repo, bound_protocol, evidence_factory=evidence_factory)
    kwargs["runtime_factory"] = ResumeRuntime
    return HANDOFF.prepare(*args, **kwargs)


def _runtime(args: argparse.Namespace, *, complete: Callable | None = None) -> tuple[ResumeRuntime, dict]:
    root = args.repo.resolve(strict=True)
    def portable(path: Path, sha: str) -> dict[str, str]:
        return {"path": path.resolve(strict=True).relative_to(root).as_posix(), "sha256": sha}
    context = E.ADAPTER.verify_context(root, portable(args.execution_protocol, args.execution_protocol_sha256),
                                       portable(args.lease, args.lease_sha256), require_active=False)
    runtime = ResumeRuntime(root, args.plan, context, args.scope, require_active=False)
    if args.command == "observe" and complete is None:
        _, scope = E.read_ref(runtime.scope)
        require(scope.get("model_relay_config") == E.ref(args.relay_config), "등록 relay config 불일치")
        complete = E.RelayComplete(E.ADAPTER._read(args.relay_config), runtime.path / "model-relay")
    payload = E.ADAPTER._read(args.payload)
    if args.command == "bind-recording":
        return runtime, runtime.bind_recording(payload)
    return runtime, runtime.observe(args.group, payload, complete)


def _evaluation_service(runtime: ResumeRuntime, payload: Mapping[str, Any], group: str) -> Any:
    bindings = runtime.context["protocol"]["execution_bindings"]
    registered = bindings["correction_backend"]
    require(registered == {"path": "evals/vision_transferred_e2e_v2.py",
            "sha256_lf": E.ADAPTER.file_sha(Path(E.__file__), lf=True),
            "class": "TransferredTrainingBackend"}, "transferred correction backend 등록 불일치")
    source_path = E.ADAPTER._file_ref(runtime.root, {"path": bindings["source_protocol"]["path"],
        "sha256": bindings["source_protocol"]["sha256_lf"]}, "source protocol", lf=True)
    source = E.ADAPTER._read(source_path)
    _, blueprint = E.read_ref(runtime.plan["transferred_blueprint"])
    backend = E.TransferredTrainingBackend.from_runtime(Path(payload["lab"]), Path(payload["python"]),
        {**source, "protocol_path": str(source_path),
         "protocol_sha256_lf": E.ADAPTER.file_sha(source_path, lf=True)},
        runtime.context["protocol"]["source_anchor"],
        runtime.root / ".triage-runtime/vision-brev" / source["protocol_id"],
        source["training_plan"], resource_proof=runtime.context["resource"],
        source_cache_reference=blueprint["source_cache"])
    jobs = E.target_jobs(source, {"seed_group": group, "source_seed": GROUPS[group]})
    bound = _group_protocol(validate_protocol(runtime._resume_protocol(), repo=runtime.root), group)
    require(jobs["CORRECTED"] == bound["corrected_job_id"],
            "resume corrected job과 source plan 불일치")
    return E.ADAPTER.CorrectionService(backend, backend.runtime_root, jobs["NONE"],
                                       f"e2e-{group}-corrected-resume")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", type=Path, default=ROOT)
    for name in ("execution-protocol", "completed-artifact", "training-ledger", "old-correction-ledger",
                 "old-failure", "external-checkpoint", "source-sha256-lf", "cached-group-input", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "old-request-id", "checkpoint-sha256", "checkpoint-copy-relative",
                 "output-namespace-relative", "deadline-utc", "approver"):
        prep.add_argument("--" + name, required=True)
    for command in ("bind-recording", "observe"):
        run = sub.add_parser(command)
        for name in ("repo", "plan", "scope", "execution-protocol", "lease", "payload"):
            run.add_argument("--" + name, type=Path, required=True)
        run.add_argument("--execution-protocol-sha256", required=True)
        run.add_argument("--lease-sha256", required=True)
        run.add_argument("--group", choices=tuple(E.SEEDS), required=True)
        run.add_argument("--relay-config", type=Path)
    evaluate = sub.add_parser("evaluate")
    for name in ("repo", "plan", "scope", "execution-protocol", "lease", "payload",
                 "external-checkpoint", "result-output", "receipt-output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    evaluate.add_argument("--execution-protocol-sha256", required=True)
    evaluate.add_argument("--lease-sha256", required=True)
    cached = sub.add_parser("evaluate-cached")
    for name in ("repo", "plan", "scope", "execution-protocol", "lease", "payload",
                 "external-checkpoint"):
        cached.add_argument("--" + name, type=Path, required=True)
    cached.add_argument("--execution-protocol-sha256", required=True)
    cached.add_argument("--lease-sha256", required=True)
    handoff = sub.add_parser("handoff-prepare")
    for name in ("repo", "plan", "scope", "execution-protocol", "lease", "bound-protocol",
                 "recording-registration-ref", "output"):
        handoff.add_argument("--" + name, type=Path, required=True)
    handoff.add_argument("--execution-protocol-sha256", required=True)
    handoff.add_argument("--lease-sha256", required=True)
    handoff.add_argument("--recording-registration-anchor", required=True)
    handoff.add_argument("--group", choices=tuple(E.SEEDS), required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare_protocol(args.repo, source_commit=args.source_commit,
            source_sha256_lf=_read(args.source_sha256_lf), execution_protocol=args.execution_protocol,
            completed_artifact=args.completed_artifact,
            training_ledger=args.training_ledger, old_correction_ledger=args.old_correction_ledger,
            old_request_id=args.old_request_id, old_failure=args.old_failure,
            external_checkpoint=args.external_checkpoint, checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_copy_relative=args.checkpoint_copy_relative,
            output_namespace_relative=args.output_namespace_relative,
            cached_group=_read(args.cached_group_input), deadline_utc=args.deadline_utc,
            approver=args.approver, output=args.output)
        print(json.dumps({"status": "evaluation_resume_protocol_ready_requires_commit", "protocol_id": value["protocol_id"]}))
    elif args.command == "handoff-prepare":
        value = handoff_prepare(args.repo, args.plan, args.scope, args.execution_protocol,
            args.execution_protocol_sha256, args.lease, args.lease_sha256, args.bound_protocol,
            args.recording_registration_ref, args.recording_registration_anchor, args.group,
            args.output)
        print(json.dumps({"status": "ticket_ready", "ticket_sha256": E.ADAPTER.file_sha(args.output),
                          "expires_at_utc": value["expires_at_utc"]}))
    elif args.command in {"evaluate", "evaluate-cached"}:
        root = args.repo.resolve(strict=True)
        def portable(path: Path, sha: str) -> dict[str, str]:
            return {"path": path.resolve(strict=True).relative_to(root).as_posix(), "sha256": sha}
        context = E.ADAPTER.verify_context(root,
            portable(args.execution_protocol, args.execution_protocol_sha256),
            portable(args.lease, args.lease_sha256), require_active=True)
        runtime = ResumeRuntime(root, args.plan, context, args.scope, require_active=True)
        payload = E.ADAPTER._read(args.payload)
        service = _evaluation_service(runtime, payload, "source-01" if args.command == "evaluate" else "source-02")
        if args.command == "evaluate":
            result = resume_evaluation(runtime._resume_protocol(), context, service,
                external_checkpoint=args.external_checkpoint, result_output=args.result_output,
                receipt_output=args.receipt_output)
        else:
            result = evaluate_cached_group(runtime._resume_protocol(), context, service,
                                           external_checkpoint=args.external_checkpoint)
        print(json.dumps({"status": result["status"], "fixed_eval": result["fixed_eval"],
                          "training_runs": 0, "model_calls": 0, "fixed_evaluations": 1}))
    else:
        _, value = _runtime(args)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
