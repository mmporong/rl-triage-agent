"""백업 checkpoint 계보와 새 recording GPU를 분리한 정책 녹화 계약."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import vision_pull_record_job_v5 as PREVIOUS
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]

COMMON = PREVIOUS.COMMON
SOURCE = PREVIOUS.SOURCE
RECORD = PREVIOUS.RECORD
RECORD_JOB = PREVIOUS.RECORD_JOB
ROOT = PREVIOUS.ROOT
MODE = PREVIOUS.MODE
JOB_SCHEMA = "vision_transferred_pull_job_v1"
PROTOCOL_SCHEMA = "vision_transferred_pull_protocol_v1"
RESOURCE_SCHEMA = "vision_transferred_recording_resource_v1"
RESULT_SCHEMA = PREVIOUS.RESULT_SCHEMA
FAILURE_SCHEMA = PREVIOUS.FAILURE_SCHEMA
CHECKPOINT_IDS = PREVIOUS.CHECKPOINT_IDS
JOB_KEYS = PREVIOUS.JOB_KEYS
SOURCE_PROTOCOL = {
    "path": "bench/protocols/vision_brev_source_20261026_v8.json",
    "sha256_lf": "10b2baa3b4529ff29f67ed552e6514bdc5db036e8afa24be005616144e77d02c",
}
BASE_PROTOCOL = {
    "path": "bench/protocols/vision_prepared_record_20261026_v4.json",
    "sha256_lf": "001bf823c2776d58b7fb7e47adf34e3f6071d0ade6fb3441ae4012be805d831e",
}
TECHNICAL_LIMITS = {
    "max_policies": 2,
    "max_wall_s_per_policy": 600,
    "max_child_s_per_policy": 400,
    "postflight_reserve_s_per_policy": 200,
    "frames_per_policy": 625,
    "fps": 50,
    "annotators_per_frame": 5,
    "raw_files_per_policy": 1250,
    "training_actions": 0,
    "model_api_requests": 0,
}
LIMITS = TECHNICAL_LIMITS
FIXED_FLAGS = PREVIOUS.FIXED_FLAGS
PROTOCOL_KEYS = {
    "schema", "protocol_id", "registered_before_measurement", "registered_at_utc",
    "source_commit", "source_sha256_lf", "source_lineage", "recording_target",
    "resources", "checkpoints", "record_profile", "limits", "preregistration",
    "gpu_validation_status",
}
TARGET_KEYS = {"resource_id", "resource_name", "instance_type"}
RESOURCE_KEYS = {
    "schema", "status", "observed_at_utc", "expires_at_utc",
    "recording_protocol_sha256_lf", "source_protocol_sha256_lf",
    "recording_target", "source_resource_reused", "capture",
}
REQUIRED_SOURCE_FILES = {
    "bench/vision_transferred_pull_job_v1.py",
    "bench/run_vision_transferred_pull_record_v1.py",
    "tests/test_vision_transferred_pull_job_v1.py",
    "tests/test_run_vision_transferred_pull_record_v1.py",
}
FRESH_CAPTURE_S = 30

_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
write_new = COMMON.write_new


def _utc(value: Any, anchor: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{anchor} 시각이 문자열이 아니다")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10
    except ValueError as error:
        raise ValueError(f"{anchor} 시각 형식이 다르다") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{anchor} 시각에 timezone이 없다")
    return result.astimezone(timezone.utc)  # noqa: UP017 - Isaac Python 3.10


def _positive_decimal(value: Any, anchor: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{anchor} 금액이 잘못됐다") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{anchor} 금액은 양수여야 한다")
    return result


def _reference(value: Any, anchor: str) -> dict[str, str]:
    row = COMMON._exact(value, {"path", "sha256_lf"}, anchor)
    return {
        "path": COMMON._relative(row["path"], f"{anchor} path"),
        "sha256_lf": COMMON._hash(row["sha256_lf"], f"{anchor} SHA"),
    }


def _target(value: Any, anchor: str) -> dict[str, str]:
    row = COMMON._exact(value, TARGET_KEYS, anchor)
    if any(not isinstance(row[key], str) or not row[key].strip() for key in TARGET_KEYS):
        raise ValueError(f"{anchor} 식별자가 비었다")
    return dict(row)


def _historical(repo: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    source_path = _regular(repo / SOURCE_PROTOCOL["path"], "historical source protocol")
    base_path = _regular(repo / BASE_PROTOCOL["path"], "historical base protocol")
    if (_sha(source_path, lf=True) != SOURCE_PROTOCOL["sha256_lf"]
            or _sha(base_path, lf=True) != BASE_PROTOCOL["sha256_lf"]):
        raise ValueError("historical source/base protocol SHA가 다르다")
    source = SOURCE.validate_protocol(_read(source_path))
    base = _read(base_path)
    if (base.get("schema") != RECORD_JOB.PROTOCOL_SCHEMA
            or base.get("parent_source_protocol") != SOURCE_PROTOCOL
            or base.get("profile_sha256") != RECORD.digest(RECORD.fixed_profile())):
        raise ValueError("historical base checkpoint 계보가 다르다")
    catalog = RECORD_JOB._checkpoints(base.get("checkpoints"))
    by_id = {row["logical_id"]: row for row in catalog}
    rows = [by_id[identity] for identity in CHECKPOINT_IDS]
    return source, base, rows


def build_protocol(
    *,
    protocol_id: str,
    registered_at_utc: str,
    source_commit: str,
    source_sha256_lf: Mapping[str, str],
    recording_target: Mapping[str, str],
    compute_quote_usd_per_h: str,
    batch_budget_usd: str,
    deadline_utc: str,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """새 live resource 값을 호출자가 제공하며 historical protocol은 읽기만 한다."""
    repo = Path(repo).resolve()
    source, _base, checkpoints = _historical(repo)
    target = _target(recording_target, "recording target")
    old_id = source["resources"]["owned_instance_id"]
    if target["resource_id"] == old_id:
        raise ValueError("새 recording target이 historical source resource ID와 같다")
    if "l40s" not in target["instance_type"].lower():
        raise ValueError("새 recording target이 L40S 계열이 아니다")
    registered, deadline = _utc(registered_at_utc, "registered"), _utc(deadline_utc, "deadline")
    if registered >= deadline:
        raise ValueError("새 recording deadline이 등록 시각 뒤가 아니다")
    _positive_decimal(compute_quote_usd_per_h, "compute quote")
    _positive_decimal(batch_budget_usd, "batch budget")
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": protocol_id,
        "registered_before_measurement": True,
        "registered_at_utc": registered_at_utc,
        "source_commit": source_commit,
        "source_sha256_lf": dict(source_sha256_lf),
        "source_lineage": {
            "source_protocol": SOURCE_PROTOCOL,
            "base_record_protocol": BASE_PROTOCOL,
            "historical_resource_id": old_id,
            "historical_resource_used_for_recording": False,
        },
        "recording_target": target,
        "resources": {
            "compute_quote_usd_per_h": str(compute_quote_usd_per_h),
            "batch_budget_usd": str(batch_budget_usd),
            "deadline_utc": deadline_utc,
            "max_owned_instances": 1,
            "cost_scope": "new_recording_instance_elapsed_wall_excludes_storage_v1",
        },
        "checkpoints": checkpoints,
        "record_profile": {
            "value": RECORD.fixed_profile(),
            "sha256": RECORD.digest(RECORD.fixed_profile()),
        },
        "limits": TECHNICAL_LIMITS,
        "preregistration": {
            "prediction": "백업 checkpoint 두 건을 새 idle GPU에서 같은 Play 설정으로 재생한다.",
            "outcomes": {
                "recording_complete": "두 정책의 기술 녹화가 완료된다.",
                "capture_hold": "과학·AABA·terminal 조건이 어긋나면 HOLD한다.",
                "infrastructure_failure": "새 자원·복원·실행 계층 실패를 분리 기록한다.",
            },
            "fixed_flags": FIXED_FLAGS,
        },
        "gpu_validation_status": "pending_live_gpu_execution",
    }


def validate_protocol(
    path: Path,
    *,
    repo: Path | None = None,
    now: datetime | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "transferred pull protocol")
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "transferred pull protocol")
    if (protocol["schema"] != PROTOCOL_SCHEMA
            or protocol["registered_before_measurement"] is not True
            or not isinstance(protocol["protocol_id"], str) or not protocol["protocol_id"].strip()
            or protocol["gpu_validation_status"] != "pending_live_gpu_execution"):
        raise ValueError("transferred pull protocol 기본 계약이 다르다")
    commit = protocol["source_commit"]
    if (not isinstance(commit, str) or COMMON.HEX40.fullmatch(commit) is None
            or COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode):
        raise ValueError("transferred pull source commit이 HEAD 조상이 아니다")
    sources = COMMON._source_map(protocol["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= sources.keys():
        raise ValueError("transferred pull source closure가 부족하다")
    COMMON._verify_source_blobs(repo, commit, sources)
    source, _base, checkpoints = _historical(repo)
    lineage = COMMON._exact(protocol["source_lineage"], {
        "source_protocol", "base_record_protocol", "historical_resource_id",
        "historical_resource_used_for_recording",
    }, "source lineage")
    old_id = source["resources"]["owned_instance_id"]
    if (lineage != {"source_protocol": SOURCE_PROTOCOL, "base_record_protocol": BASE_PROTOCOL,
                    "historical_resource_id": old_id,
                    "historical_resource_used_for_recording": False}):
        raise ValueError("historical source lineage가 다르다")
    target = _target(protocol["recording_target"], "recording target")
    if target["resource_id"] == old_id:
        raise ValueError("historical source resource를 live target으로 재사용할 수 없다")
    if "l40s" not in target["instance_type"].lower():
        raise ValueError("새 recording target이 L40S 계열이 아니다")
    resources = COMMON._exact(protocol["resources"], {
        "compute_quote_usd_per_h", "batch_budget_usd", "deadline_utc",
        "max_owned_instances", "cost_scope",
    }, "recording resources")
    if (resources["max_owned_instances"] != 1
            or resources["cost_scope"] != "new_recording_instance_elapsed_wall_excludes_storage_v1"):
        raise ValueError("새 recording resource 상한이 다르다")
    quote = _positive_decimal(resources["compute_quote_usd_per_h"], "compute quote")
    budget = _positive_decimal(resources["batch_budget_usd"], "batch budget")
    required = (quote * Decimal(TECHNICAL_LIMITS["max_policies"])
                * Decimal(TECHNICAL_LIMITS["max_wall_s_per_policy"]) / Decimal(3600))
    if budget < required:
        raise ValueError("새 recording batch budget이 등록 wall 상한보다 작다")
    registered = _utc(protocol["registered_at_utc"], "registered")
    deadline = _utc(resources["deadline_utc"], "deadline")
    current = now or datetime.now(timezone.utc)  # noqa: UP017
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    current = current.astimezone(timezone.utc)  # noqa: UP017
    if registered > current or registered >= deadline or (require_current and current >= deadline):
        raise ValueError("새 recording protocol 등록·deadline이 유효하지 않다")
    if protocol["checkpoints"] != checkpoints:
        raise ValueError("checkpoint catalog가 historical base와 다르다")
    profile = COMMON._exact(protocol["record_profile"], {"value", "sha256"}, "record profile")
    if (profile["value"] != RECORD.fixed_profile()
            or profile["sha256"] != RECORD.digest(RECORD.fixed_profile())):
        raise ValueError("same Play record profile이 다르다")
    if protocol["limits"] != TECHNICAL_LIMITS:
        raise ValueError("625 frame/50 fps/physics/PTS 기술 상한이 다르다")
    prereg = COMMON._exact(protocol["preregistration"],
                           {"prediction", "outcomes", "fixed_flags"}, "preregistration")
    if (not isinstance(prereg["prediction"], str) or not prereg["prediction"].strip()
            or not isinstance(prereg["outcomes"], dict) or set(prereg["outcomes"]) != {
                "recording_complete", "capture_hold", "infrastructure_failure"}
            or prereg["fixed_flags"] != FIXED_FLAGS):
        raise ValueError("transferred pull 사전등록이 다르다")
    protocol["checkpoints"] = checkpoints
    protocol["_source"] = source
    protocol["_recording_target"] = target
    return protocol


def validate_resource_proof(
    proof_path: Path,
    proof_sha: str,
    *,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    now: datetime,
    require_fresh: bool,
) -> dict[str, Any]:
    path = _regular(Path(proof_path).absolute(), "transferred recording resource proof")
    if _sha(path) != COMMON._hash(proof_sha, "resource proof SHA"):
        raise ValueError("transferred recording resource proof SHA가 다르다")
    proof = COMMON._exact(_read(path), RESOURCE_KEYS, "transferred recording resource proof")
    target = _target(proof["recording_target"], "proof recording target")
    if (proof["schema"] != RESOURCE_SCHEMA or proof["status"] != "granted"
            or proof["recording_protocol_sha256_lf"] != protocol_sha
            or proof["source_protocol_sha256_lf"] != SOURCE_PROTOCOL["sha256_lf"]
            or target != protocol["_recording_target"]
            or proof["source_resource_reused"] is not False
            or target["resource_id"] == protocol["source_lineage"]["historical_resource_id"]):
        raise ValueError("source lineage와 새 recording resource가 분리되지 않았다")
    observed = _utc(proof["observed_at_utc"], "resource observed")
    expires = _utc(proof["expires_at_utc"], "resource expires")
    deadline = _utc(protocol["resources"]["deadline_utc"], "protocol deadline")
    current = now.astimezone(timezone.utc)  # noqa: UP017
    age = (current - observed).total_seconds()
    if (observed > current or expires > deadline or observed >= expires
            or (require_fresh and (age > FRESH_CAPTURE_S or current >= expires))):
        raise ValueError("새 recording resource proof가 fresh하지 않다")
    capture = COMMON._exact(proof["capture"], REMOTE.CAPTURE_KEYS, "resource capture")
    if capture["resource_id"] != target["resource_id"]:
        raise ValueError("capture resource ID가 새 recording target과 다르다")
    REMOTE.verify_capture_proof(capture)
    provider = REMOTE.verify_provider_inventory(
        Path(capture["provider_inventory_path"]), capture["provider_inventory_sha256"],
        target["resource_id"],
    )
    if (provider["name"] != target["resource_name"]
            or provider["instance_type"] != target["instance_type"]):
        raise ValueError("provider target 이름·instance type이 등록값과 다르다")
    return {**proof, **capture, "_path": path}


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
    """새 VM의 provider/GPU/idle 원문으로 write 전 resource proof를 구성한다."""
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
        raise ValueError("provider target 이름·instance type이 등록값과 다르다")
    expires = _utc(expires_at_utc, "resource expires")
    if observed >= expires or expires > _utc(protocol["resources"]["deadline_utc"], "deadline"):
        raise ValueError("resource proof 만료가 새 protocol deadline 밖이다")
    return {
        "schema": RESOURCE_SCHEMA, "status": "granted",
        "observed_at_utc": observed_at_utc, "expires_at_utc": expires_at_utc,
        "recording_protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_protocol_sha256_lf": SOURCE_PROTOCOL["sha256_lf"],
        "recording_target": target, "source_resource_reused": False,
        "capture": capture,
    }


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    job_path = _regular(Path(job_path).absolute(), "transferred pull job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "transferred pull job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("transferred pull job schema/mode가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    ref = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol ref")
    protocol_path = _regular(Path(ref["path"]), "protocol path")
    protocol_sha = COMMON._hash(ref["sha256_lf"], "protocol SHA")
    if _sha(protocol_path, lf=True) != protocol_sha:
        raise ValueError("transferred pull protocol SHA가 다르다")
    current = now or datetime.now(timezone.utc)  # noqa: UP017
    protocol = validate_protocol(protocol_path, repo=repo, now=current,
                                 require_current=require_fresh)
    if source != {"repo_root": str(repo), "commit": protocol["source_commit"],
                  "files_lf": protocol["source_sha256_lf"]}:
        raise ValueError("transferred pull job source가 protocol과 다르다")
    if job["profile"] != RECORD.fixed_profile():
        raise ValueError("transferred pull job same Play profile이 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    checkpoint = COMMON._exact(job["checkpoint"], {"path", "sha256", "logical_id"},
                               "checkpoint")
    identity = checkpoint["logical_id"]
    if identity not in CHECKPOINT_IDS:
        raise ValueError("transferred pull checkpoint가 등록 두 건 밖이다")
    row, checkpoint_path = RECORD_JOB._checkpoint(protocol, identity, lab)
    if checkpoint != {"path": str(checkpoint_path), "sha256": row["sha256"],
                      "logical_id": identity}:
        raise ValueError("복원 checkpoint 경로·SHA가 catalog와 다르다")
    if job["job_id"] != f"{protocol['protocol_id']}-{identity}":
        raise ValueError("transferred pull job ID가 다르다")
    resource_ref = COMMON._exact(job["resource_proof"], {"path", "sha256"}, "resource ref")
    resource = validate_resource_proof(
        Path(resource_ref["path"]), resource_ref["sha256"], protocol=protocol,
        protocol_sha=protocol_sha, now=current, require_fresh=require_fresh,
    )
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, protocol["_source"])
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("transferred pull output 경계가 다르다")
    COMMON._check_ancestors(output.parent, "transferred pull output")
    present = output.exists() or output.is_symlink()
    if require_fresh and present:
        raise FileExistsError(output)
    if not require_fresh and present:
        output = COMMON._directory(output, "transferred pull output")
    return {"job_path": job_path, "job_sha256": _sha(job_path), "job": job,
            "protocol": protocol, "protocol_path": protocol_path,
            "protocol_sha256": protocol_sha, "output_directory": output, "repo": repo,
            "resource": resource, "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": row["sha256"]}


def create_job(
    protocol_path: Path, resource_path: Path, resource_sha: str, isaaclab: Path,
    python: Path, stage: Path, checkpoint_logical_id: str,
) -> Path:
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path)
    if checkpoint_logical_id not in CHECKPOINT_IDS:
        raise ValueError("transferred pull checkpoint가 등록값이 아니다")
    lab = COMMON._directory(Path(isaaclab).absolute(), "isaaclab")
    row, checkpoint = RECORD_JOB._checkpoint(protocol, checkpoint_logical_id, lab)
    stage = Path(stage).absolute()
    COMMON._check_ancestors(stage, "transferred pull stage")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    job_path = stage / "job.json"
    write_new(job_path, {
        "schema": JOB_SCHEMA, "mode": MODE,
        "job_id": f"{protocol['protocol_id']}-{checkpoint_logical_id}",
        "source": {"repo_root": str(ROOT), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path), "sha256_lf": _sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(Path(resource_path).absolute()),
                           "sha256": resource_sha},
        "isaaclab": str(lab), "python": str(Path(python).absolute()),
        "output_directory": str(stage / "output"), "profile": RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint), "sha256": row["sha256"],
                       "logical_id": row["logical_id"]},
    })
    validate_job(job_path)
    return job_path
