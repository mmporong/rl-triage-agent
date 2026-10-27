"""검증된 CORRECTED checkpoint만 native annotator-pull RUN3로 녹화한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import run_vision_transferred_pull_record_v3 as RUN3
import vision_remote_gpu_v1 as REMOTE
import vision_transferred_pull_job_v3 as JOB3

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "vision_corrected_pull_protocol_v1"
JOB_SCHEMA = "vision_corrected_pull_job_v1"
RESOURCE_SCHEMA = JOB3.RESOURCE_SCHEMA
RESULT_SCHEMA = JOB3.RESULT_SCHEMA
FAILURE_SCHEMA = JOB3.FAILURE_SCHEMA
MODE = JOB3.MODE
LIMITS = JOB3.LIMITS
FIXED_FLAGS = JOB3.FIXED_FLAGS
CHECKPOINT_IDS = ("corrected-s2220", "corrected-s2221")
CATALOG = {
    "corrected-s2220": {"seed": 2220, "correction_job_id": "vision-s2220-corrected",
                         "source_f1_checkpoint_sha256": "8a7f43ec0f1cebf470f7e8116bf5f1905f9ae742fbec3afddf6720438d7e4474"},
    "corrected-s2221": {"seed": 2221, "correction_job_id": "vision-s2221-corrected",
                         "source_f1_checkpoint_sha256": "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731"},
}
REQUIRED_SOURCE_FILES = JOB3.REQUIRED_SOURCE_FILES | {
    "bench/run_vision_corrected_pull_record_v1.py",
    "evals/vision_corrected_record_service_v1.py",
    "tests/test_vision_corrected_pull_record_v1.py",
    "tests/test_vision_corrected_record_service_v1.py",
    "bench/protocols/vision_corrected_pull_record_template_v1.json",
}
COMMON, RECORD, SOURCE = JOB3.COMMON, JOB3.RECORD, JOB3.SOURCE
JOB_KEYS = JOB3.JOB_KEYS


def _sha(path: Path, *, lf: bool = False) -> str:
    value = Path(path).read_bytes()
    if lf:
        value = value.replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON이 regular file이 아니다: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON root가 객체가 아니다")
    return value


def _ref(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha(path)}


def _utc(value: Any, anchor: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{anchor} 시각이 문자열이 아니다")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{anchor} timezone이 없다")
    return parsed.astimezone(timezone.utc)


def _positive_decimal(value: Any, anchor: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{anchor} 금액이 잘못됐다") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{anchor} 금액은 양수여야 한다")
    return result


def validate_template(path: Path) -> dict[str, Any]:
    value = _read(path)
    required = {"schema", "template_status", "protocol_id", "source_lineage", "record_profile",
                "limits", "fixed_flags", "groups", "late_binding_contract", "post_observer"}
    if set(value) != required or value["schema"] != "vision_corrected_pull_protocol_template_v1":
        raise ValueError("CORRECTED protocol template 형식이 다르다")
    if (value["template_status"] != "before_training_requires_verified_late_binding"
            or value["record_profile"] != RECORD.fixed_profile() or value["limits"] != LIMITS
            or value["fixed_flags"] != FIXED_FLAGS
            or value["post_observer"] != {"used_for_decision": False}):
        raise ValueError("CORRECTED template 과학·승인 경계가 다르다")
    groups = value["groups"]
    if not isinstance(groups, list) or [row.get("logical_id") for row in groups] != list(CHECKPOINT_IDS):
        raise ValueError("CORRECTED template catalog가 다르다")
    for row in groups:
        expected = CATALOG[row["logical_id"]]
        if row != {"logical_id": row["logical_id"], **expected,
                   "correction_request_binding": {"correction_id": "restore_body_frame_reward",
                     "recovery_mode": "restart", "protocol_case_arm": {"case": "case_01", "arm": "corrected"}}}:
            raise ValueError("CORRECTED template correction binding이 다르다")
    contract = value["late_binding_contract"]
    if contract != {"checkpoint_from": "validated_correction.training.checkpoint",
                    "job_id_must_equal": "registered_correction_job_id",
                    "approval_receipt_required": True, "bind_before_aaba_and_record": True}:
        raise ValueError("CORRECTED late binding 계약이 다르다")
    return value


def _verified_binding(logical_id: str, correction_path: Path, approval_path: Path) -> dict[str, Any]:
    expected = CATALOG[logical_id]
    correction, approval = _read(correction_path), _read(approval_path)
    training = correction.get("training")
    checkpoint = training.get("checkpoint") if isinstance(training, Mapping) else None
    correction_keys = {"schema", "status", "group", "job_id", "checkpoint_sha256", "training",
                       "training_evidence", "evaluation", "fixed_eval", "gpu_s", "compute_usd", "exit_code"}
    expected_group = f"source-{1 if expected['seed'] == 2220 else 2:02d}"
    if (set(correction) != correction_keys
            or correction.get("schema") != "vision_selection_correction_result_v1"
            or correction.get("status") != "complete"
            or correction.get("group") != expected_group
            or correction.get("job_id") != expected["correction_job_id"]
            or correction.get("exit_code") != 0
            or not isinstance(training, Mapping) or training.get("arm") != "CORRECTED"
            or training.get("seed") != expected["seed"]
            or training.get("source_checkpoint_sha256") != expected["source_f1_checkpoint_sha256"]
            or correction.get("fixed_eval") != {"none": "healthy", "corrected": "healthy"}
            or not isinstance(checkpoint, Mapping)
            or correction.get("checkpoint_sha256") != checkpoint.get("sha256")):
        raise ValueError("CORRECTED training 결과가 등록 계보와 다르다")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve(strict=True)
    if checkpoint_path.is_symlink() or _sha(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("CORRECTED checkpoint 원문/SHA가 다르다")
    required = {"schema", "approved", "request_id", "corrected_job_id", "seed", "correction_id",
                "recovery_mode", "protocol_case_arm", "correction_result_sha256", "receipt_verdict"}
    if (set(approval) != required or approval.get("schema") != "vision_corrected_record_approval_v1"
            or approval.get("approved") is not True or approval.get("corrected_job_id") != expected["correction_job_id"]
            or approval.get("seed") != expected["seed"] or approval.get("correction_id") != "restore_body_frame_reward"
            or approval.get("recovery_mode") != "restart"
            or approval.get("protocol_case_arm") != {"case": "case_01", "arm": "corrected"}
            or approval.get("correction_result_sha256") != _sha(correction_path)
            or approval.get("receipt_verdict") != "recovered"
            or not isinstance(approval.get("request_id"), str) or not approval["request_id"].strip()):
        raise ValueError("CORRECTED approval receipt가 결과와 다르다")
    return {"logical_id": logical_id, **expected,
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint["sha256"]},
            "correction_result": _ref(correction_path), "approval_receipt": _ref(approval_path),
            "request_id": approval["request_id"]}


def bind_protocol(template_path: Path, bindings: Mapping[str, Mapping[str, Path]], *,
                  source_commit: str, source_sha256_lf: Mapping[str, str],
                  recording_target: Mapping[str, str], resources: Mapping[str, Any],
                  registered_at_utc: str, output: Path) -> dict[str, Any]:
    template = validate_template(template_path)
    if set(bindings) != set(CHECKPOINT_IDS):
        raise ValueError("CORRECTED 두 policy binding이 필요하다")
    checkpoints = [_verified_binding(logical, Path(bindings[logical]["correction_result"]),
                                     Path(bindings[logical]["approval_receipt"]))
                   for logical in CHECKPOINT_IDS]
    value = {"schema": PROTOCOL_SCHEMA, "protocol_id": template["protocol_id"],
             "registered_at_utc": registered_at_utc, "source_commit": source_commit,
             "source_sha256_lf": dict(source_sha256_lf), "source_lineage": template["source_lineage"],
             "recording_target": dict(recording_target), "resources": dict(resources),
             "record_profile": template["record_profile"], "limits": template["limits"],
             "fixed_flags": template["fixed_flags"], "checkpoints": checkpoints,
             "post_observer": template["post_observer"], "gpu_validation_status": "pending_live_gpu_execution"}
    JOB3.write_new(output, value)
    validate_protocol(output)
    return value


def validate_protocol(path: Path, *, repo: Path | None = None, now: datetime | None = None,
                      require_current: bool = True) -> dict[str, Any]:
    repo = Path(repo or ROOT).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    value = _read(path)
    required = {"schema", "protocol_id", "registered_at_utc", "source_commit", "source_sha256_lf",
                "source_lineage", "recording_target", "resources", "record_profile", "limits",
                "fixed_flags", "checkpoints", "post_observer", "gpu_validation_status"}
    if set(value) != required or value["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("CORRECTED bound protocol 형식이 다르다")
    if (value["record_profile"] != RECORD.fixed_profile() or value["limits"] != LIMITS
            or value["fixed_flags"] != FIXED_FLAGS or value["post_observer"] != {"used_for_decision": False}
            or value["gpu_validation_status"] != "pending_live_gpu_execution"):
        raise ValueError("CORRECTED bound protocol 과학 경계가 다르다")
    commit = value["source_commit"]
    if (not isinstance(commit, str) or COMMON.HEX40.fullmatch(commit) is None
            or COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode):
        raise ValueError("CORRECTED source commit이 현재 HEAD 조상이 아니다")
    sources = COMMON._source_map(value["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= sources.keys():
        raise ValueError("CORRECTED source closure가 부족하다")
    COMMON._verify_source_blobs(repo, commit, sources)
    lineage = value["source_lineage"]
    if lineage != {"source_protocol": JOB3.SOURCE_PROTOCOL, "base_record_protocol": JOB3.BASE_PROTOCOL}:
        raise ValueError("CORRECTED source lineage가 다르다")
    source_path = repo / JOB3.SOURCE_PROTOCOL["path"]
    if _sha(source_path, lf=True) != JOB3.SOURCE_PROTOCOL["sha256_lf"]:
        raise ValueError("CORRECTED source protocol SHA가 다르다")
    source = SOURCE.validate_protocol(_read(source_path))
    target = value["recording_target"]
    if (not isinstance(target, dict) or set(target) != JOB3.TARGET_KEYS
            or any(not isinstance(target[k], str) or not target[k] for k in target)
            or target["resource_id"] == source["resources"]["owned_instance_id"]):
        raise ValueError("CORRECTED recording target이 다르다")
    resources = value["resources"]
    if not isinstance(resources, dict) or set(resources) != {"compute_quote_usd_per_h", "batch_budget_usd",
            "deadline_utc", "max_owned_instances", "cost_scope"} or resources["max_owned_instances"] != 1:
        raise ValueError("CORRECTED recording resource 경계가 다르다")
    quote = _positive_decimal(resources["compute_quote_usd_per_h"], "compute quote")
    budget = _positive_decimal(resources["batch_budget_usd"], "batch budget")
    required_budget = quote * Decimal(LIMITS["max_policies"]) * Decimal(LIMITS["max_wall_s_per_policy"]) / Decimal(3600)
    if (budget < required_budget
            or resources["cost_scope"] != "new_corrected_native_recording_only"):
        raise ValueError("CORRECTED recording 예산 또는 비용 범위가 다르다")
    current = now or datetime.now(timezone.utc)
    registered, deadline = _utc(value["registered_at_utc"], "registered"), _utc(resources["deadline_utc"], "deadline")
    current = current.astimezone(timezone.utc)
    if (registered >= deadline
            or (require_current and (current < registered or current >= deadline))):
        raise ValueError("CORRECTED protocol deadline이 유효하지 않다")
    rows = value["checkpoints"]
    if not isinstance(rows, list) or [row.get("logical_id") for row in rows] != list(CHECKPOINT_IDS):
        raise ValueError("CORRECTED checkpoint catalog가 다르다")
    for row in rows:
        expected = CATALOG[row["logical_id"]]
        if any(row.get(key) != expected[key] for key in expected):
            raise ValueError("CORRECTED checkpoint 계보가 다르다")
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
            raise ValueError("CORRECTED checkpoint descriptor가 다르다")
        cp = Path(checkpoint["path"]).resolve(strict=True)
        if cp.is_symlink() or _sha(cp) != checkpoint["sha256"]:
            raise ValueError("CORRECTED checkpoint가 바뀌었다")
        for key in ("correction_result", "approval_receipt"):
            ref = row.get(key)
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or _sha(Path(ref["path"])) != ref["sha256"]:
                raise ValueError(f"CORRECTED {key} 원문이 바뀌었다")
        if _verified_binding(row["logical_id"], Path(row["correction_result"]["path"]),
                             Path(row["approval_receipt"]["path"])) != row:
            raise ValueError("CORRECTED late binding 재검증이 다르다")
    value["_source"] = source
    value["_recording_target"] = target
    return value


def validate_resource_proof(proof_path: Path, proof_sha: str, *, protocol: Mapping[str, Any],
                            protocol_sha: str, now: datetime, require_fresh: bool) -> dict[str, Any]:
    augmented = dict(protocol)
    augmented["source_lineage"] = {"historical_resource_id": protocol["_source"]["resources"]["owned_instance_id"]}
    return JOB3.validate_resource_proof(proof_path, proof_sha, protocol=augmented,
                                        protocol_sha=protocol_sha, now=now, require_fresh=require_fresh)


def build_resource_proof(
    protocol_path: Path,
    *,
    provider_inventory_path: Path,
    nvidia_gpu_csv_path: Path,
    nvidia_process_csv_path: Path,
    observed_at_utc: str,
    expires_at_utc: str,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """새 recording target의 fresh provider/GPU/idle 원문을 proof로 묶는다."""
    observed = _utc(observed_at_utc, "resource observed")
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol = validate_protocol(protocol_path, repo=repo, now=observed)
    target = protocol["_recording_target"]
    capture = {
        "resource_id": target["resource_id"],
        "gpu_uuid": Path(nvidia_gpu_csv_path).read_text(encoding="utf-8").strip(),
        "provider_inventory_path": str(Path(provider_inventory_path).resolve(strict=True)),
        "provider_inventory_sha256": _sha(provider_inventory_path),
        "nvidia_gpu_csv_path": str(Path(nvidia_gpu_csv_path).resolve(strict=True)),
        "nvidia_gpu_csv_sha256": _sha(nvidia_gpu_csv_path),
        "nvidia_process_csv_path": str(Path(nvidia_process_csv_path).resolve(strict=True)),
        "nvidia_process_csv_sha256": _sha(nvidia_process_csv_path),
    }
    REMOTE.verify_capture_proof(capture)
    provider = REMOTE.verify_provider_inventory(
        Path(capture["provider_inventory_path"]), capture["provider_inventory_sha256"],
        target["resource_id"],
    )
    if (provider["name"] != target["resource_name"]
            or provider["instance_type"] != target["instance_type"]):
        raise ValueError("CORRECTED provider target 이름·instance type이 다르다")
    expires = _utc(expires_at_utc, "resource expires")
    if observed >= expires or expires > _utc(protocol["resources"]["deadline_utc"], "deadline"):
        raise ValueError("CORRECTED resource proof 만료가 protocol deadline 밖이다")
    return {
        "schema": RESOURCE_SCHEMA,
        "status": "granted",
        "observed_at_utc": observed_at_utc,
        "expires_at_utc": expires_at_utc,
        "recording_protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_protocol_sha256_lf": JOB3.SOURCE_PROTOCOL["sha256_lf"],
        "recording_target": target,
        "source_resource_reused": False,
        "capture": capture,
    }


def validate_job(job_path: Path, *, now: datetime | None = None, require_fresh: bool = True) -> dict[str, Any]:
    job_path = Path(job_path).resolve(strict=True)
    job = _read(job_path)
    if set(job) != JOB_KEYS or job.get("schema") != JOB_SCHEMA or job.get("mode") != MODE:
        raise ValueError("CORRECTED record job 형식이 다르다")
    source = job["source"]
    repo = Path(source["repo_root"]).resolve(strict=True)
    protocol_path = Path(job["protocol"]["path"]).resolve(strict=True)
    protocol_sha = _sha(protocol_path, lf=True)
    if job["protocol"].get("sha256_lf") != protocol_sha:
        raise ValueError("CORRECTED protocol ref가 바뀌었다")
    current = now or datetime.now(timezone.utc)
    protocol = validate_protocol(protocol_path, repo=repo, now=current, require_current=require_fresh)
    if source != {"repo_root": str(repo), "commit": protocol["source_commit"],
                  "files_lf": protocol["source_sha256_lf"]}:
        raise ValueError("CORRECTED job source가 protocol과 다르다")
    if job["profile"] != RECORD.fixed_profile():
        raise ValueError("CORRECTED same Play profile이 다르다")
    checkpoint = job["checkpoint"]
    rows = [row for row in protocol["checkpoints"] if row["logical_id"] == checkpoint.get("logical_id")]
    if len(rows) != 1 or checkpoint != {"path": rows[0]["checkpoint"]["path"],
                                       "sha256": rows[0]["checkpoint"]["sha256"],
                                       "logical_id": rows[0]["logical_id"]}:
        raise ValueError("CORRECTED job checkpoint가 late binding과 다르다")
    if job["job_id"] != f"{protocol['protocol_id']}-{checkpoint['logical_id']}":
        raise ValueError("CORRECTED job ID가 다르다")
    resource = validate_resource_proof(Path(job["resource_proof"]["path"]), job["resource_proof"]["sha256"],
        protocol=protocol, protocol_sha=protocol_sha, now=current, require_fresh=require_fresh)
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, protocol["_source"])
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("CORRECTED output 경계가 다르다")
    if require_fresh and (output.exists() or output.is_symlink()):
        raise FileExistsError(output)
    return {"job_path": job_path, "job_sha256": _sha(job_path), "job": job,
            "protocol": protocol, "protocol_path": protocol_path, "protocol_sha256": protocol_sha,
            "output_directory": output, "repo": repo, "resource": resource,
            "checkpoint_path": Path(checkpoint["path"]), "checkpoint_sha256": checkpoint["sha256"]}


def create_job(protocol_path: Path, resource_path: Path, resource_sha: str, isaaclab: Path,
               python: Path, stage: Path, checkpoint_logical_id: str) -> Path:
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol = validate_protocol(protocol_path)
    rows = [row for row in protocol["checkpoints"] if row["logical_id"] == checkpoint_logical_id]
    if len(rows) != 1:
        raise ValueError("CORRECTED logical ID가 등록 catalog 밖이다")
    stage = Path(stage).absolute()
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    row = rows[0]
    job_path = stage / "job.json"
    JOB3.write_new(job_path, {"schema": JOB_SCHEMA, "mode": MODE,
        "job_id": f"{protocol['protocol_id']}-{checkpoint_logical_id}",
        "source": {"repo_root": str(ROOT), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path), "sha256_lf": _sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(Path(resource_path).absolute()), "sha256": resource_sha},
        "isaaclab": str(Path(isaaclab).absolute()), "python": str(Path(python).absolute()),
        "output_directory": str(stage / "output"), "profile": RECORD.fixed_profile(),
        "checkpoint": {"path": row["checkpoint"]["path"], "sha256": row["checkpoint"]["sha256"],
                       "logical_id": checkpoint_logical_id}})
    validate_job(job_path)
    return job_path


def _child_launcher(popen: Callable[..., Any]) -> Callable[..., Any]:
    old = str(Path(RUN3.PREVIOUS.__file__).resolve())
    current = str(Path(__file__).resolve())
    def launch(command: Sequence[str], **kwargs: Any) -> Any:
        rewritten = [current if value == old else value for value in command]
        if rewritten == list(command):
            raise ValueError("native RUN3 child 진입점이 없다")
        return popen(rewritten, **kwargs)
    return launch


@contextmanager
def corrected_contract() -> Iterator[None]:
    old_job, old_launcher = RUN3.JOB, RUN3._child_launcher
    RUN3.JOB, RUN3._child_launcher = sys.modules[__name__], _child_launcher
    try:
        yield
    finally:
        RUN3.JOB, RUN3._child_launcher = old_job, old_launcher


def aaba_child(job_path: Path, **kwargs: Any) -> int:
    with corrected_contract():
        return RUN3.aaba_child(job_path, **kwargs)


def sdk_child(job_path: Path, **kwargs: Any) -> int:
    with corrected_contract():
        return RUN3.sdk_child(job_path, **kwargs)


def run_job(job_path: Path, *, popen: Callable[..., Any] = subprocess.Popen,
            gpu_reader: Callable[[], list[int]] | None = None,
            identify: Callable[[int], Mapping[str, Any]] = RUN3.PREVIOUS.OWNED._process_identity,
            cleanup: Callable[..., bool] = RUN3.PREVIOUS.OWNED._safe_cleanup) -> dict[str, Any]:
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("CORRECTED native record는 소유 Linux group에서 실행한다")
    with corrected_contract():
        return RUN3.run_job(job_path, popen=popen, gpu_reader=gpu_reader, identify=identify, cleanup=cleanup)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--aaba-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--checkpoint-logical-id", choices=CHECKPOINT_IDS)
    args = parser.parse_args(argv)
    if args.sdk_child or args.aaba_child:
        if args.job is None:
            parser.error("child 실행에는 --job이 필요하다")
        return aaba_child(args.job) if args.aaba_child else sdk_child(args.job)
    if args.job is None:
        required = (args.protocol, args.resource_proof, args.resource_sha256, args.isaaclab,
                    args.python, args.stage, args.checkpoint_logical_id)
        if any(value is None for value in required):
            parser.error("job 생성 인자가 모두 필요하다")
        args.job = create_job(args.protocol, args.resource_proof, args.resource_sha256,
                              args.isaaclab, args.python, args.stage, args.checkpoint_logical_id)
    result = run_job(args.job)
    return 0 if result.get("completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
