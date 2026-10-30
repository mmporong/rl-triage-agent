"""중단된 lineage source 13개를 검증해 재사용하고 등록된 나머지 5개만 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
import isaac_training_guard_v1 as GUARD
import lineage_gpu_admission_v1 as GPU
import lineage_source_batch_v1 as B

SCHEMA = "lineage_source_continuation_protocol_v1"
MANIFEST_SCHEMA = "lineage_source_continuation_manifest_v1"
SOURCE_ANCHOR = "10942fcdde672156ed751b26c3de2270c59c06bb"
PREFLIGHT_SHA256 = "eeb192a7e4b8c1af25bae98c0c95786183f86069791b3cb1432d02979b9fe756"
PREFIX_COUNT = 13
REMAINING_COUNT = 5
TOTAL_WALL_S = 9000
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FILES = (
    "evals/lineage_source_continuation_v1.py",
    "tests/test_lineage_source_continuation_v1.py",
    "bench/run_lineage_source_continuation_v1.ps1",
    "evals/lineage_source_batch_v1.py",
    "evals/lineage_gpu_admission_v1.py",
    "bench/lineage_source_training_v1.py",
    "bench/isaac_training_guard_v1.py",
    "bench/run_runtime_readback_smoke.py",
    "evals/preregistered_seed_guard.py",
)


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    GUARD.durable_new(Path(path), value)


def _jobs() -> tuple[list[dict], list[dict]]:
    training = B.jobs()[1]
    return training[:PREFIX_COUNT], training[PREFIX_COUNT:]


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("continuation registration Git 결합이 확인되지 않았다")
    return result.stdout


def _seed_guard(lab: Path) -> dict:
    from preregistered_seed_guard import SeedReuseError, assert_fresh_seeds
    try:
        assert_fresh_seeds([2308, 2309, 2310], lab, repo=ROOT)
    except SeedReuseError as error:
        marker = "이미 기록된 seed: "
        if not str(error).startswith(marker):
            raise
        evidence = json.loads(str(error)[len(marker):])
        used = {row.get("seed") for row in evidence if isinstance(row, dict)}
        if used != {2308}:
            raise ValueError("remaining target seed 2309/2310이 이미 사용됐다") from error
        return {"checked_seeds": [2308, 2309, 2310], "allowed_consumed": [2308],
                "required_unused": [2309, 2310], "evidence": evidence}
    raise ValueError("prefix NONE-s2308의 기존 seed 사용 근거가 사라졌다")


def _target_absence(lab: Path, remaining: list[dict]) -> dict:
    log_root = Path(lab).resolve() / "logs/rsl_rl/unitree_go2_flat"
    if not log_root.is_dir():
        raise ValueError("Isaac Lab unitree_go2_flat 로그 경로가 없다")
    found = []
    for directory in log_root.iterdir():
        if directory.is_dir():
            for row in remaining:
                if directory.name.endswith("_" + row["run_name"]):
                    found.append({"job_id": row["job_id"], "path": directory.name})
    if found:
        raise ValueError("remaining job의 기존 log_dir/run_name이 발견됐다: " + json.dumps(found))
    return {"remaining_job_ids": [row["job_id"] for row in remaining],
            "checked_log_root": "<isaaclab>/logs/rsl_rl/unitree_go2_flat", "found": []}


def _prior(prior_root: Path, lab: Path) -> dict:
    prior_root, lab = Path(prior_root).resolve(), Path(lab).resolve()
    value = B.verify(lab, SOURCE_ANCHOR)
    prefix, remaining = _jobs()
    intent, failure = _read(prior_root / "intent.json"), _read(prior_root / "failure.json")
    if (intent.get("anchor") != SOURCE_ANCHOR
            or intent.get("protocol_sha256_lf") != B.digest(ROOT / B.PROTOCOL, lf=True)
            or failure.get("status") != "invalid" or failure.get("error_type") != "RuntimeError"):
        raise ValueError("원 source intent/failure 결합이 다르다")
    records = failure.get("completed_jobs")
    if (not isinstance(records, list) or [row.get("job_id") for row in records]
            != [row["job_id"] for row in prefix]):
        raise ValueError("완료 13개가 원래 training job prefix 순서와 다르다")
    checked = []
    for job, record in zip(prefix, records, strict=True):
        stage = prior_root / job["job_id"]
        fixed = B.manifest(value, job, SOURCE_ANCHOR, purpose="source")
        B.CHILD.verify_binding(fixed, repo=ROOT, lab=lab)
        if _read(stage / "manifest.json") != fixed:
            raise ValueError(f"prior manifest가 등록 job과 다르다: {job['job_id']}")
        evidence = B.completed(stage, fixed, _read(stage / "owned-result.json"))
        accepted = _read(stage / "accepted.json")
        if (accepted != record or accepted.get("checkpoint") != evidence["checkpoint"]
                or accepted.get("evidence_sha256") != _sha(stage / "training" / GUARD.EVIDENCE)):
            raise ValueError(f"prior accepted/result/evidence가 다르다: {job['job_id']}")
        checkpoint = accepted["checkpoint"]
        checked.append({"job_id": job["job_id"], "arm": job["arm"], "seed": job["seed"],
                        "checkpoint": {"sha256": checkpoint["sha256"],
                                       "size_bytes": checkpoint.get("size_bytes"),
                                       "iteration": checkpoint.get("iteration")},
                        "manifest_sha256": _sha(stage / "manifest.json"),
                        "owned_result_sha256": _sha(stage / "owned-result.json"),
                        "evidence_sha256": accepted["evidence_sha256"],
                        "accepted_sha256": _sha(stage / "accepted.json"),
                        "owned_parent_wall_s": accepted["owned_parent_wall_s"]})
    failed_job = remaining[0]
    failed_stage = prior_root / failed_job["job_id"]
    files = sorted(path.name for path in failed_stage.iterdir()) if failed_stage.is_dir() else []
    if files != ["gpu_before.json"]:
        raise ValueError("중단 stage는 gpu_before만 있는 prelaunch 상태여야 한다")
    gpu_before = _read(failed_stage / "gpu_before.json")
    if (gpu_before.get("busy") is not True or not gpu_before.get("busy_reasons")
            or any((failed_stage / name).exists() for name in (
                "manifest.json", "launch.json", "owned-result.json", "accepted.json", "training"))):
        raise ValueError("14번째 stage가 infrastructure prelaunch-only가 아니다")
    for row in remaining[1:]:
        if (prior_root / row["job_id"]).exists():
            raise ValueError(f"미실행 target stage가 이미 존재한다: {row['job_id']}")
    preflight_root = prior_root.parents[1] / "recovery-preflight-20261031-v1"
    preflight_path, lease_path = preflight_root / "result.json", preflight_root / "lease-completed.json"
    if not preflight_path.is_file() or _sha(preflight_path) != PREFLIGHT_SHA256 or not lease_path.is_file():
        raise ValueError("immutable recovery preflight 원문이 다르다")
    preflight, lease = _read(preflight_path), _read(lease_path)
    if (preflight.get("prior_failure_sha256") != _sha(prior_root / "failure.json")
            or preflight.get("prior_completed_jobs") != PREFIX_COUNT
            or preflight.get("failed_stage_prelaunch_only") is not True
            or preflight.get("policy_unchanged") is not True or preflight.get("all_clear") is not True
            or preflight.get("new_gpu_jobs") != 0 or preflight.get("api") != 0
            or preflight.get("lease_sha256") != _sha(lease_path)
            or lease.get("state") != "failed" or not lease.get("completedUtc")):
        raise ValueError("recovery preflight/failure/lease 결합이 다르다")
    return {"source_protocol": value, "records": records, "prior_jobs": checked,
            "failure_sha256": _sha(prior_root / "failure.json"),
            "intent_sha256": _sha(prior_root / "intent.json"),
            "failed_job": failed_job["job_id"],
            "gpu_before_sha256": _sha(failed_stage / "gpu_before.json"),
            "preflight_sha256": PREFLIGHT_SHA256, "lease_completed_sha256": _sha(lease_path),
            "infrastructure_prelaunch_only": True,
            "seed_guard": _seed_guard(lab), "target_absence": _target_absence(lab, remaining)}


def prepare_protocol(prior_root: Path, lab: Path) -> dict:
    """개인 경로 없이 prior 13개와 remaining 5개를 고정한 공개 protocol을 만든다."""
    prior = _prior(prior_root, lab)
    prefix, remaining = _jobs()
    seed_guard = prior["seed_guard"]
    public_seed_guard = {
        "checked_seeds": seed_guard["checked_seeds"],
        "allowed_consumed": seed_guard["allowed_consumed"],
        "required_unused": seed_guard["required_unused"],
        "evidence_count": len(seed_guard["evidence"]),
        "evidence_sha256": _canonical_digest(seed_guard["evidence"]),
    }
    return {"schema": SCHEMA, "phase": "source_collection_continuation_once",
            "source_collection": {"protocol": B.PROTOCOL, "anchor": SOURCE_ANCHOR,
                                  "protocol_sha256_lf": B.digest(ROOT / B.PROTOCOL, lf=True)},
            "prior_prefix_count": len(prefix), "prior_jobs": prior["prior_jobs"],
            "prior_failure_sha256": prior["failure_sha256"],
            "prior_intent_sha256": prior["intent_sha256"],
            "failed_prelaunch": {"job_id": prior["failed_job"],
                                 "gpu_before_sha256": prior["gpu_before_sha256"],
                                 "infrastructure_prelaunch_only": True,
                                 "child_training_started": False},
            "recovery_preflight": {"result_sha256": prior["preflight_sha256"],
                                   "lease_completed_sha256": prior["lease_completed_sha256"],
                                   "gpu_policy_unchanged": True, "new_gpu_jobs": 0},
            "remaining_jobs": remaining, "seed_guard": public_seed_guard,
            "target_absence": prior["target_absence"],
            "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
            "gpu_admission": GPU.policy(),
            "budget": {"reused_source_jobs": 13, "new_source_jobs": 5,
                       "new_source_updates": 1500, "per_job_parent_wall_s": 1800,
                       "remaining_parent_wall_s": TOTAL_WALL_S, "automatic_retries": 0,
                       "model_api_requests": 0, "paid_cloud_gpu_starts": 0},
            "claims": {"single_uninterrupted_batch": False, "retraining_prior_13": False,
                       "fault_or_seed_reselection": False}}


def _verify_registration(protocol_path: Path, execution_anchor: str, prior_root: Path, lab: Path) -> dict:
    if not HEX40.fullmatch(execution_anchor):
        raise ValueError("execution anchor 형식이 잘못됐다")
    protocol_path = Path(protocol_path).resolve()
    value = _read(protocol_path)
    if value != prepare_protocol(prior_root, lab):
        raise ValueError("continuation protocol/prior/source closure가 다르다")
    try:
        relative = protocol_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("continuation protocol은 repo 내부여야 한다") from exc
    _git("merge-base", "--is-ancestor", execution_anchor, "HEAD")
    bound = {relative: _sha(protocol_path, lf=True), **value["source_sha256_lf"]}
    for name, expected in bound.items():
        current = ROOT / name
        committed = _git("show", f"{execution_anchor}:{name}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if not current.is_file() or _sha(current, lf=True) != expected \
                or hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f"execution anchor blob/current SHA가 다르다: {name}")
    return value


def prepare_inputs(prior_root: Path, protocol_path: Path, execution_anchor: str, out: Path,
                   lab: Path, isaac_python: Path) -> dict:
    prior_root, out = Path(prior_root).resolve(), Path(out).resolve()
    lab, isaac_python = Path(lab).resolve(), Path(isaac_python).resolve()
    if out.exists():
        raise FileExistsError("continuation output은 fresh 경로여야 한다")
    if not isaac_python.is_file():
        raise FileNotFoundError("Isaac Python launcher가 없다")
    protocol = _verify_registration(protocol_path, execution_anchor, prior_root, lab)
    prior = _prior(prior_root, lab)
    out.mkdir(parents=True)
    manifest = {"schema": MANIFEST_SCHEMA, "prior_root": str(prior_root),
                "prior_failure_sha256": protocol["prior_failure_sha256"],
                "protocol_path": str(Path(protocol_path).resolve()),
                "protocol_sha256_lf": _sha(protocol_path, lf=True),
                "source_anchor": SOURCE_ANCHOR, "execution_anchor": execution_anchor,
                "seed_guard_evidence": prior["seed_guard"]["evidence"],
                "seed_guard_evidence_sha256": _canonical_digest(prior["seed_guard"]["evidence"]),
                "lab": str(lab), "isaac_python": str(isaac_python), "output_root": str(out)}
    _write_new(out / "continuation-manifest.json", manifest)
    _write_new(out / "intent.json", {"mode": "all", "anchor": SOURCE_ANCHOR,
                                      "protocol_sha256_lf": B.digest(ROOT / B.PROTOCOL, lf=True),
                                      "execution_anchor": execution_anchor,
                                      "continuation_protocol_sha256_lf": manifest["protocol_sha256_lf"]})
    return manifest


def _manifest(path: Path) -> tuple[dict, dict, dict]:
    manifest = _read(path)
    required = {"schema", "prior_root", "prior_failure_sha256", "protocol_path",
                "protocol_sha256_lf", "source_anchor", "execution_anchor",
                "seed_guard_evidence", "seed_guard_evidence_sha256", "lab",
                "isaac_python", "output_root"}
    if set(manifest) != required or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("continuation private manifest 필드가 다르다")
    prior = _prior(Path(manifest["prior_root"]), Path(manifest["lab"]))
    protocol = _verify_registration(Path(manifest["protocol_path"]), manifest["execution_anchor"],
                                    Path(manifest["prior_root"]), Path(manifest["lab"]))
    if (manifest["source_anchor"] != SOURCE_ANCHOR
            or manifest["prior_failure_sha256"] != prior["failure_sha256"]
            or manifest["seed_guard_evidence"] != prior["seed_guard"]["evidence"]
            or manifest["seed_guard_evidence_sha256"]
            != _canonical_digest(prior["seed_guard"]["evidence"])
            or manifest["protocol_sha256_lf"] != _sha(Path(manifest["protocol_path"]), lf=True)):
        raise ValueError("continuation manifest의 prior/protocol anchor 결합이 다르다")
    return manifest, protocol, prior


def execute(manifest_path: Path, *, runner: Callable,
            check_lease: Callable = B.lease_context,
            check_gpu: Callable | None = GPU.snapshot,
            clock: Callable = time.monotonic) -> dict:
    manifest, protocol, prior = _manifest(Path(manifest_path).resolve())
    root, lab = Path(manifest["output_root"]), Path(manifest["lab"])
    if (root / "result.json").exists() or (root / "failure.json").exists():
        raise FileExistsError("continuation 결과를 재사용하지 않는다")
    original = prior["source_protocol"]
    remaining = protocol["remaining_jobs"]
    first_lease, start = check_lease(), clock()
    new_records = []
    try:
        for row in remaining:
            current_lease = check_lease()
            if current_lease.get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("continuation 도중 GPU lease ID가 바뀌었다")
            elapsed = clock() - start
            if elapsed >= TOTAL_WALL_S:
                raise TimeoutError("remaining5 전체 controller 예산이 소진됐다")
            stage = root / row["job_id"]
            stage.mkdir()
            if check_gpu is not None:
                check_gpu(stage)
            fixed = B.manifest(original, row, SOURCE_ANCHOR, purpose="source")
            B.CHILD.verify_binding(fixed, repo=ROOT, lab=lab)
            _write_new(stage / "manifest.json", fixed)
            _write_new(stage / "lease-before.json", current_lease)
            argv = [manifest["isaac_python"], str(ROOT / "bench/lineage_source_training_v1.py"),
                    "--manifest", str(stage / "manifest.json"), "--lab", str(lab),
                    "--out", str(stage / "training")]
            receipt = dict(runner(stage, argv, lab, min(1800, TOTAL_WALL_S - elapsed)))
            _write_new(stage / "owned-result.json", receipt)
            if check_lease().get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("child 종료 뒤 GPU lease ID가 바뀌었다")
            evidence = B.completed(stage, fixed, receipt)
            record = {"job_id": row["job_id"], "purpose": "source", "arm": row["arm"],
                      "seed": row["seed"], "checkpoint": evidence["checkpoint"],
                      "evidence_sha256": _sha(stage / "training" / GUARD.EVIDENCE),
                      "owned_parent_wall_s": receipt["wall_s"]}
            new_records.append(record)
            _write_new(stage / "accepted.json", record)
        records = [*prior["records"], *new_records]
        if [row["job_id"] for row in records] != [row["job_id"] for row in B.jobs()[1]]:
            raise AssertionError("결합 records가 원래 18job 순서와 다르다")
        grouped: dict[str, list[str]] = {}
        for row in records:
            grouped.setdefault(row["checkpoint"]["sha256"], []).append(row["job_id"])
        result = {"schema": B.SCHEMA + "_result", "status": "complete", "mode": "all",
                  "source_jobs_completed": 18, "records": records,
                  "duplicate_checkpoint_jobs": [ids for ids in grouped.values() if len(ids) > 1],
                  "model_api_requests": 0, "fixed_evaluation_executed": False,
                  "diagnosis_or_recovery_claim": False, "wall_s": clock() - start,
                  "continuation": {"reused_prior_jobs": 13, "new_jobs": 5,
                                   "prior_failure_sha256": prior["failure_sha256"],
                                   "prior_intent_sha256": prior["intent_sha256"],
                                   "prior_gpu_before_sha256": prior["gpu_before_sha256"],
                                   "source_anchor": SOURCE_ANCHOR,
                                   "execution_anchor": manifest["execution_anchor"],
                                   "protocol_sha256_lf": manifest["protocol_sha256_lf"],
                                   "single_uninterrupted_batch": False,
                                   "prior_owned_parent_wall_s": sum(
                                       row["owned_parent_wall_s"] for row in prior["records"]),
                                   "new_owned_parent_wall_s": sum(
                                       row["owned_parent_wall_s"] for row in new_records)}}
        _write_new(root / "result.json", result)
        return result
    except BaseException as error:
        _write_new(root / "failure.json", {"status": "invalid", "completed_new_jobs": new_records,
                                            "automatic_retry": False,
                                            "next_action": "manual_preregister_new_tag",
                                            "error_type": type(error).__name__, "error": str(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "prepare-inputs", "execute"))
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--execution-anchor")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.out is None:
            parser.error("prepare에는 --out이 필요하다")
        _write_new(args.out, prepare_protocol(args.prior_root, args.lab))
        return 0
    if args.command == "validate":
        _prior(args.prior_root, args.lab)
        print(json.dumps({"status": "valid", "reused": 13, "remaining": 5, "api": 0}))
        return 0
    if args.protocol is None or args.execution_anchor is None:
        parser.error("prepare-inputs/execute에는 --protocol --execution-anchor가 필요하다")
    if args.command == "prepare-inputs":
        if args.out is None or args.isaac_python is None:
            parser.error("prepare-inputs에는 --out --isaac-python이 필요하다")
        prepare_inputs(args.prior_root, args.protocol, args.execution_anchor, args.out,
                       args.lab, args.isaac_python)
        return 0
    if args.manifest is None:
        parser.error("execute에는 --manifest가 필요하다")
    import run_runtime_readback_smoke as OWNED
    result = execute(args.manifest, runner=OWNED.run_owned)
    print(json.dumps({"status": result["status"], "sources": result["source_jobs_completed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
