"""Native pull 정책 녹화의 protocol, checkpoint, resource 결합 계약."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import vision_annotator_pull_job_v2 as DIAGNOSTIC

COMMON = DIAGNOSTIC.COMMON
SOURCE = DIAGNOSTIC.SOURCE
RECORD = DIAGNOSTIC.RECORD
RECORD_JOB = DIAGNOSTIC.RECORD_JOB
PREPARED = DIAGNOSTIC.PREPARED
ROOT = DIAGNOSTIC.ROOT

MODE = RECORD.RECORD_MODE
PROTOCOL_SCHEMA = "vision_pull_record_protocol_v2"
JOB_SCHEMA = "vision_pull_record_job_v1"
RESULT_SCHEMA = "vision_pull_record_result_v1"
FAILURE_SCHEMA = "vision_pull_record_failure_v1"
PROTOCOL_ID = "native-policy-record-20261027ak"
CHECKPOINT_IDS = ("source-s2220-none", "source-s2220-f1")
PROTOCOL_KEYS = {
    "schema", "protocol_id", "mode", "registered_before_measurement",
    "source_commit", "source_sha256_lf", "predecessor", "base_record_protocol",
    "source_protocol", "checkpoints", "resources", "limits", "preregistration",
}
JOB_KEYS = RECORD_JOB.JOB_KEYS
LIMITS = {
    "max_policies": 2,
    "max_batch_compute_usd": "0.881793",
    "max_wall_s_per_policy": 600,
    "max_owned_instances": 1,
    "frames_per_policy": 625,
    "fps": 50,
    "annotators_per_frame": 5,
    "raw_files_per_policy": 1250,
    "training_actions": 0,
    "model_api_requests": 0,
}
FIXED_FLAGS = {
    "per_checkpoint_aaba_required": True,
    "writer_calls": 0,
    "atomicity_claimed": False,
    "capture_approval": False,
    "model_ready": False,
    "used_for_decision": False,
    "native_counter_source": "omni.kit.app.IApp.get_update_number",
    "native_pull_window_updates": 0,
    "orchestrator_steps_per_frame": 1,
    "physics_state_time_unchanged": True,
    "reference_time_source": "rep.annotators.get(ReferenceTime)",
    "clean_reserved_marker_count": 0,
    "actual_pts_required": True,
    "aaba_output_separate_from_record": True,
    "aaba_raw_receipts_required": True,
    "first_error_preserved_before_close": True,
}
REQUIRED_SOURCE_FILES = DIAGNOSTIC.REQUIRED_SOURCE_FILES | {
    "bench/vision_pull_record_capture_v1.py",
    "bench/vision_pull_record_job_v1.py",
    "bench/vision_pull_record_worker_v1.py",
    "bench/run_vision_pull_record_v1.py",
    "tests/test_vision_pull_record_capture_v1.py",
    "tests/test_vision_pull_record_job_v1.py",
    "tests/test_vision_pull_record_worker_v1.py",
    "tests/test_run_vision_pull_record_v1.py",
    "bench/vision_pull_record_job_v2.py",
    "tests/test_vision_pull_record_job_v2.py",
    "bench/run_vision_pull_record_v2.py",
    "tests/test_run_vision_pull_record_v2.py",
    "bench/vision_pull_record_checkpoint_v3.py",
    "tests/test_vision_pull_record_checkpoint_v3.py",
}
EXPECTED_PRIOR = Decimal("19.118207")
EXPECTED_FRESH = Decimal("0.881793")
EXPECTED_TOTAL = Decimal("20.000000")
EXPECTED_STOPS = (
    {
        "path": ".triage-runtime/vision-brev/lifecycle/annotator-pull-20261027ai-stop.json",
        "sha256": "e786d0b59dd2d8f325da00ed9f7172e718b061ec7804d1c651f6b032eb54ce17",
        "schema": "v04_owned_vm_stop_receipt_v1",
        "run_id": "annotator-pull-20261027ai",
        "cost": "0.150112",
    },
    {
        "path": ".triage-runtime/vision-brev/lifecycle/native-policy-record-20261027aj-infra-stop.json",
        "sha256": "07c53d3ee41aacf13d0370345ee669114ca2b759b80c1cc6e4a4f05f03a811f0",
        "schema": "vision_native_policy_record_pre_candidate_abort_metadata_v1",
        "run_id": None,
        "cost": "0.058680",
    },
    {
        "path": ".triage-runtime/vision-brev/lifecycle/native-policy-record-20261027aj-retry1-stop.json",
        "sha256": "b93ba86c984c9ce478bdadb9333cbf99f8e5e89c69e8a7003589ea662a9937b7",
        "schema": "v04_owned_vm_stop_receipt_v1",
        "run_id": "native-policy-record-20261027aj",
        "cost": "0.242750",
    },
    {
        "path": ".triage-runtime/vision-brev/lifecycle/native-policy-record-20261027aj-retry2-stop.json",
        "sha256": "d9b03cb83bc712753b4d82250fe74ec4021a470d71ad05f17880ba020985313e",
        "schema": "v04_owned_vm_stop_receipt_v1",
        "run_id": "native-policy-record-20261027aj",
        "cost": "0.336100",
    },
)
EXPECTED_DIAGNOSTIC = {
    "path": "bench/protocols/vision_annotator_pull_20261027_v2.json",
    "sha256_lf": "de692af3b0b1720f7106c5dc1c15990a9703f3441b4ec10992763690d263336b",
}
EXPECTED_BASE = {
    "path": "bench/protocols/vision_prepared_record_20261026_v4.json",
    "sha256_lf": "001bf823c2776d58b7fb7e47adf34e3f6071d0ade6fb3441ae4012be805d831e",
}
EXPECTED_SOURCE = {
    "path": "bench/protocols/vision_brev_source_20261026_v8.json",
    "sha256_lf": "10b2baa3b4529ff29f67ed552e6514bdc5db036e8afa24be005616144e77d02c",
}

_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
write_new = COMMON.write_new
validation_deadline = COMMON.validation_deadline


def _reference(value: Any, anchor: str) -> dict[str, Any]:
    row = COMMON._exact(value, {"path", "sha256_lf"}, anchor)
    COMMON._relative(row["path"], f"{anchor} path")
    COMMON._hash(row["sha256_lf"], f"{anchor} SHA")
    return row


def _budget(protocol: Mapping[str, Any], previous: Mapping[str, Any], repo: Path) -> None:
    resources = PREPARED._resource_shape(protocol["resources"])
    previous_resources = PREPARED._resource_shape(previous["resources"])
    predecessor = COMMON._exact(
        protocol["predecessor"], {"diagnostic", "appended_stops"}, "pull record predecessor"
    )
    diagnostic = _reference(predecessor["diagnostic"], "diagnostic")
    if diagnostic != EXPECTED_DIAGNOSTIC:
        raise ValueError("pull record predecessor diagnostic이 다르다")
    raw_stops = predecessor["appended_stops"]
    if not isinstance(raw_stops, list) or len(raw_stops) != len(EXPECTED_STOPS):
        raise ValueError("pull record appended STOP 수가 다르다")
    stop_refs: list[dict[str, Any]] = []
    stop_cost = Decimal(0)
    for index, (raw_ref, expected) in enumerate(zip(raw_stops, EXPECTED_STOPS, strict=True)):
        stop_ref = COMMON._exact(raw_ref, {"path", "sha256"}, f"appended STOP {index}")
        public_ref = {"path": expected["path"], "sha256": expected["sha256"]}
        if stop_ref != public_ref:
            raise ValueError("pull record appended STOP 순서·참조가 다르다")
        stop_path = _regular(repo / COMMON._relative(stop_ref["path"], "STOP path"), "STOP")
        if _sha(stop_path) != COMMON._hash(stop_ref["sha256"], "STOP SHA"):
            raise ValueError("pull record STOP SHA가 다르다")
        stop = _read(stop_path)
        cost = stop.get("cost_estimate")
        if stop.get("schema") != expected["schema"] or stop.get("status") != "STOPPED" or not isinstance(cost, Mapping):
            raise ValueError("pull record STOP schema/status가 다르다")
        if expected["run_id"] is None:
            if (
                stop.get("protocol_id") != "native-policy-record-20261027aj"
                or stop.get("source_protocol_sha256_lf") != EXPECTED_SOURCE["sha256_lf"]
                or stop.get("resource_id") != previous_resources["owned_instance_id"]
                or stop.get("resource_name") != previous_resources["owned_instance_name"]
                or stop.get("gpu_measurement_started") is not False
                or stop.get("resource_proof_created") is not False
                or stop.get("training_actions") != 0
                or stop.get("model_api_requests") != 0
                or stop.get("full_recordings") != 0
                or stop.get("capture_approved") is not False
                or stop.get("used_for_decision") is not False
            ):
                raise ValueError("pull record pre-candidate STOP 경계가 다르다")
        else:
            identity = stop.get("execution_identity")
            if (
                not isinstance(identity, Mapping)
                or identity.get("run_id") != expected["run_id"]
                or identity.get("protocol_sha256") != EXPECTED_SOURCE["sha256_lf"]
            ):
                raise ValueError("pull record STOP identity가 다르다")
        amount = PREPARED._amount(cost.get("estimated_compute_cost_usd"), "STOP cost")
        if amount != PREPARED._amount(expected["cost"], "expected STOP cost"):
            raise ValueError("pull record STOP 비용이 등록값과 다르다")
        stop_cost += amount
        stop_refs.append(stop_ref)
    previous_prior = PREPARED._amount(previous_resources["prior_compute_estimate_usd"], "previous prior")
    prior = PREPARED._amount(resources["prior_compute_estimate_usd"], "prior")
    fresh = PREPARED._amount(resources["fresh_lifecycle_budget_usd"], "fresh")
    total = PREPARED._amount(resources["total_program_budget_usd"], "total")
    if (
        prior != previous_prior + stop_cost
        or prior != EXPECTED_PRIOR
        or fresh != EXPECTED_FRESH
        or total != EXPECTED_TOTAL
        or prior + fresh != total
        or resources["max_owned_instances"] != 1
        or resources["prior_stop_receipts"]
        != [*previous_resources["prior_stop_receipts"], *stop_refs]
    ):
        raise ValueError("pull record 누적 비용 계약이 다르다")


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    COMMON._check_deadline()
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "pull record protocol")
    COMMON._verify_head_protocol(repo, path)
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "pull record protocol")
    if (
        protocol["schema"] != PROTOCOL_SCHEMA
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["mode"] != MODE
        or protocol["registered_before_measurement"] is not True
        or protocol["limits"] != LIMITS
    ):
        raise ValueError("pull record protocol 기본 계약이 다르다")
    commit = protocol["source_commit"]
    if (
        not isinstance(commit, str)
        or COMMON.HEX40.fullmatch(commit) is None
        or COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode
    ):
        raise ValueError("pull record source commit이 HEAD 조상이 아니다")
    sources = COMMON._source_map(protocol["source_sha256_lf"])
    if len(sources) < 365 or not REQUIRED_SOURCE_FILES <= sources.keys():
        raise ValueError("pull record source closure가 부족하다")
    COMMON._verify_source_blobs(repo, commit, sources)

    diagnostic_ref = _reference(protocol["predecessor"]["diagnostic"], "diagnostic")
    if diagnostic_ref != EXPECTED_DIAGNOSTIC:
        raise ValueError("pull record predecessor가 고정 AI diagnostic이 아니다")
    diagnostic_path = _regular(repo / diagnostic_ref["path"], "diagnostic protocol")
    if _sha(diagnostic_path, lf=True) != diagnostic_ref["sha256_lf"]:
        raise ValueError("diagnostic protocol SHA가 다르다")
    diagnostic = _read(diagnostic_path)
    if (
        diagnostic.get("schema") != DIAGNOSTIC.PROTOCOL_SCHEMA
        or diagnostic.get("protocol_id") != "annotator-pull-20261027ai"
        or diagnostic.get("source_protocol") != EXPECTED_SOURCE
        or diagnostic.get("base_record_protocol") != EXPECTED_BASE
        or len(COMMON._source_map(diagnostic.get("source_sha256_lf"))) != 351
        or PREPARED._amount(diagnostic.get("resources", {}).get("prior_compute_estimate_usd"), "AI prior")
        != Decimal("18.330565")
        or len(diagnostic.get("resources", {}).get("prior_stop_receipts", [])) != 21
    ):
        raise ValueError("고정 AI diagnostic 원문이 다르다")
    diagnostic["_protocol_path"] = diagnostic_path
    diagnostic["_protocol_sha256"] = diagnostic_ref["sha256_lf"]
    if any(sources.get(name) != digest for name, digest in diagnostic["source_sha256_lf"].items()):
        raise ValueError("pull record가 diagnostic source closure를 보존하지 않는다")

    base_ref = _reference(protocol["base_record_protocol"], "base record protocol")
    if base_ref != EXPECTED_BASE or base_ref != diagnostic["base_record_protocol"]:
        raise ValueError("pull record base catalog가 diagnostic과 다르다")
    base_path = _regular(repo / base_ref["path"], "base record protocol")
    if _sha(base_path, lf=True) != base_ref["sha256_lf"]:
        raise ValueError("base record protocol SHA가 다르다")
    base = _read(base_path)
    if (
        base.get("schema") != RECORD_JOB.PROTOCOL_SCHEMA
        or base.get("protocol_id") != "prepared-policy-record-20261026z"
        or base.get("profile_sha256") != RECORD.digest(RECORD.fixed_profile())
        or base.get("parent_source_protocol") != EXPECTED_SOURCE
    ):
        raise ValueError("고정 base checkpoint catalog 원문이 다르다")
    source_ref = _reference(protocol["source_protocol"], "source protocol")
    if source_ref != diagnostic["source_protocol"] or source_ref != base["parent_source_protocol"]:
        raise ValueError("pull record source protocol 계보가 다르다")
    if source_ref != EXPECTED_SOURCE:
        raise ValueError("pull record source S8 identity가 다르다")
    source_path = _regular(repo / source_ref["path"], "source protocol")
    if _sha(source_path, lf=True) != source_ref["sha256_lf"]:
        raise ValueError("source protocol SHA가 다르다")
    source_protocol = _read(source_path)
    if (
        source_protocol.get("schema") != "vision_brev_source_protocol_v1"
        or source_protocol.get("protocol_id") != "vision-source-20261026x"
        or source_protocol.get("record_profile_sha256")
        != RECORD.digest(RECORD.fixed_profile())
    ):
        raise ValueError("source S8 protocol identity/profile이 다르다")

    rows = RECORD_JOB._checkpoints(protocol["checkpoints"])
    if [row["logical_id"] for row in rows] != list(CHECKPOINT_IDS):
        raise ValueError("pull record checkpoint 순서가 정상·F1 두 건이 아니다")
    base_rows = {row["logical_id"]: row for row in base["checkpoints"]}
    if any(base_rows.get(row["logical_id"]) != row for row in rows):
        raise ValueError("pull record checkpoint가 고정 catalog와 다르다")
    prereg = COMMON._exact(
        protocol["preregistration"], {"prediction", "outcomes", "fixed_flags"}, "preregistration"
    )
    outcomes = prereg["outcomes"]
    if (
        not isinstance(prereg["prediction"], str)
        or not prereg["prediction"].strip()
        or not isinstance(outcomes, dict)
        or set(outcomes) != {"recording_complete", "capture_hold", "infrastructure_failure"}
        or any(not isinstance(value, str) or not value.strip() for value in outcomes.values())
        or prereg["fixed_flags"] != FIXED_FLAGS
    ):
        raise ValueError("pull record 사전등록이 다르다")
    _budget(protocol, diagnostic, repo)
    protocol["checkpoints"] = rows
    protocol["_base"] = base
    protocol["_source"] = source_protocol
    return protocol


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    job_path = _regular(Path(job_path).absolute(), "pull record job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "pull record job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("pull record job schema/mode가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    ref = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol ref")
    protocol_path = _regular(Path(ref["path"]), "protocol path")
    protocol_sha = COMMON._hash(ref["sha256_lf"], "protocol SHA")
    if _sha(protocol_path, lf=True) != protocol_sha:
        raise ValueError("pull record protocol SHA가 다르다")
    protocol = validate_protocol(protocol_path, repo=repo)
    if source != {
        "repo_root": str(repo), "commit": protocol["source_commit"],
        "files_lf": protocol["source_sha256_lf"],
    } or job["profile"] != RECORD.fixed_profile():
        raise ValueError("pull record job source/profile이 protocol과 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    checkpoint = COMMON._exact(job["checkpoint"], {"path", "sha256", "logical_id"}, "checkpoint")
    identity = checkpoint["logical_id"]
    if identity not in CHECKPOINT_IDS:
        raise ValueError("pull record checkpoint가 등록 두 건 밖이다")
    row, checkpoint_path = RECORD_JOB._checkpoint(protocol, identity, lab)
    if checkpoint != {"path": str(checkpoint_path), "sha256": row["sha256"], "logical_id": identity}:
        raise ValueError("pull record checkpoint 결합이 다르다")
    expected_id = f"{PROTOCOL_ID}-{identity}"
    if job["job_id"] != expected_id:
        raise ValueError("pull record job ID가 다르다")
    resource_ref = COMMON._exact(job["resource_proof"], {"path", "sha256"}, "resource ref")
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Isaac Python 3.10
    resource = COMMON._resource(
        _regular(Path(resource_ref["path"]), "resource proof"),
        COMMON._hash(resource_ref["sha256"], "resource SHA"),
        protocol["_source"], protocol["source_protocol"]["sha256_lf"], current, require_fresh,
    )
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, protocol["_source"])
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("pull record output 경계가 다르다")
    COMMON._check_ancestors(output.parent, "pull record output")
    if require_fresh and (output.exists() or output.is_symlink()):
        raise FileExistsError(output)
    if not require_fresh:
        output = COMMON._directory(output, "pull record output")
    return {
        "job_path": job_path, "job_sha256": _sha(job_path), "job": job,
        "protocol": protocol, "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha, "output_directory": output, "repo": repo,
        "resource": resource, "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": row["sha256"],
    }


def create_job(
    protocol_path: Path, resource_path: Path, resource_sha: str, isaaclab: Path,
    python: Path, stage: Path, checkpoint_logical_id: str,
) -> Path:
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path)
    if checkpoint_logical_id not in CHECKPOINT_IDS:
        raise ValueError("pull record checkpoint가 등록값이 아니다")
    lab = COMMON._directory(Path(isaaclab).absolute(), "isaaclab")
    row, checkpoint = RECORD_JOB._checkpoint(protocol, checkpoint_logical_id, lab)
    stage = Path(stage).absolute()
    COMMON._check_ancestors(stage, "pull record stage")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    job_path = stage / "job.json"
    write_new(job_path, {
        "schema": JOB_SCHEMA, "mode": MODE,
        "job_id": f"{PROTOCOL_ID}-{checkpoint_logical_id}",
        "source": {"repo_root": str(ROOT), "commit": protocol["source_commit"], "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path), "sha256_lf": _sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(Path(resource_path).absolute()), "sha256": resource_sha},
        "isaaclab": str(lab), "python": str(Path(python).absolute()),
        "output_directory": str(stage / "output"), "profile": RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint), "sha256": row["sha256"], "logical_id": row["logical_id"]},
    })
    validate_job(job_path)
    return job_path
