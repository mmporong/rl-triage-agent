"""두 중단 원문의 완료 16개를 재사용하고 원 등록의 마지막 2개만 실행한다."""
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
import lineage_gpu_wait_v1 as GPU_WAIT
import lineage_source_batch_v1 as B

SCHEMA = "lineage_source_continuation_protocol_v2"
MANIFEST_SCHEMA = "lineage_source_continuation_manifest_v2"
SOURCE_ANCHOR = "10942fcdde672156ed751b26c3de2270c59c06bb"
V1_EXECUTION_ANCHOR = "a5347e173ed475b8a17a80b90bbd0c71c8f0874f"
V1_PROTOCOL = "bench/protocols/lineage_source_continuation_20261031_v1.json"
V1_PROTOCOL_SHA256_LF = "2829c657637478883ae76e6eb9b4031ccc62e0c919be115353bf15b124b098a6"
ORIGIN_COUNT = 13
CONTINUATION_COUNT = 3
TOTAL_WALL_S = 5400
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_FILES = (
    "evals/lineage_source_continuation_v2.py",
    "tests/test_lineage_source_continuation_v2.py",
    "bench/run_lineage_source_continuation_v2.ps1",
    "evals/lineage_gpu_wait_v1.py",
    "tests/test_lineage_gpu_wait_v1.py",
    "evals/lineage_source_batch_v1.py",
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


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    GUARD.durable_new(Path(path), value)


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("continuation v2 Git blob/ancestor 결합이 확인되지 않았다")
    return result.stdout


def _normalized_git(anchor: str, relative: str) -> str:
    raw = _git("show", f"{anchor}:{relative}")
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _job_sets() -> tuple[list[dict], list[dict], list[dict]]:
    jobs = B.jobs()[1]
    return jobs[:ORIGIN_COUNT], jobs[ORIGIN_COUNT:ORIGIN_COUNT + CONTINUATION_COUNT], jobs[16:]


def _validate_stage(root: Path, job: dict, record: dict, source: dict) -> dict:
    stage = root / job["job_id"]
    fixed = B.manifest(source, job, SOURCE_ANCHOR, purpose="source")
    if _read(stage / "manifest.json") != fixed:
        raise ValueError(f"stage manifest가 original job과 다르다: {job['job_id']}")
    evidence = B.completed(stage, fixed, _read(stage / "owned-result.json"))
    accepted = _read(stage / "accepted.json")
    if (accepted != record or accepted.get("checkpoint") != evidence["checkpoint"]
            or accepted.get("evidence_sha256") != _sha(stage / "training" / GUARD.EVIDENCE)):
        raise ValueError(f"accepted/evidence/checkpoint가 다르다: {job['job_id']}")
    checkpoint = accepted["checkpoint"]
    return {"job_id": job["job_id"], "arm": job["arm"], "seed": job["seed"],
            "checkpoint": {"sha256": checkpoint["sha256"],
                           "size_bytes": checkpoint.get("size_bytes"),
                           "iteration": checkpoint.get("iteration")},
            "manifest_sha256": _sha(stage / "manifest.json"),
            "owned_result_sha256": _sha(stage / "owned-result.json"),
            "evidence_sha256": accepted["evidence_sha256"],
            "accepted_sha256": _sha(stage / "accepted.json"),
            "owned_parent_wall_s": accepted["owned_parent_wall_s"]}


def _prelaunch(stage: Path, expected_job: str) -> dict:
    files = sorted(path.name for path in stage.iterdir()) if stage.is_dir() else []
    if files != ["gpu_before.json"]:
        raise ValueError(f"{expected_job} failure stage는 gpu_before만 있어야 한다")
    gpu = _read(stage / "gpu_before.json")
    if gpu.get("busy") is not True or not gpu.get("busy_reasons"):
        raise ValueError(f"{expected_job}은 GPU infrastructure prelaunch failure여야 한다")
    return {"job_id": expected_job, "gpu_before_sha256": _sha(stage / "gpu_before.json"),
            "infrastructure_prelaunch_only": True, "child_training_started": False}


def _verify_v1_registration(manifest: dict, origin_root: Path) -> dict:
    protocol_path = ROOT / V1_PROTOCOL
    if (_sha(protocol_path, lf=True) != V1_PROTOCOL_SHA256_LF
            or _normalized_git(V1_EXECUTION_ANCHOR, V1_PROTOCOL) != V1_PROTOCOL_SHA256_LF):
        raise ValueError("v1 continuation 등록 protocol blob이 다르다")
    protocol = _read(protocol_path)
    for name, expected in protocol.get("source_sha256_lf", {}).items():
        if not (ROOT / name).is_file() or _sha(ROOT / name, lf=True) != expected \
                or _normalized_git(V1_EXECUTION_ANCHOR, name) != expected:
            raise ValueError(f"v1 continuation 등록 source blob이 다르다: {name}")
    required = {"schema", "prior_root", "prior_failure_sha256", "protocol_path",
                "protocol_sha256_lf", "source_anchor", "execution_anchor",
                "seed_guard_evidence", "seed_guard_evidence_sha256", "lab",
                "isaac_python", "output_root"}
    if (set(manifest) != required or manifest.get("schema") != "lineage_source_continuation_manifest_v1"
            or Path(manifest["prior_root"]).resolve() != origin_root
            or manifest.get("source_anchor") != SOURCE_ANCHOR
            or manifest.get("execution_anchor") != V1_EXECUTION_ANCHOR
            or manifest.get("protocol_sha256_lf") != V1_PROTOCOL_SHA256_LF
            or Path(manifest["protocol_path"]).resolve() != protocol_path.resolve()
            or manifest.get("seed_guard_evidence_sha256")
            != _canonical_sha(manifest.get("seed_guard_evidence"))):
        raise ValueError("v1 continuation private manifest/origin 결합이 다르다")
    return protocol


def _fresh_2310(lab: Path) -> dict:
    from preregistered_seed_guard import assert_fresh_seeds
    result = assert_fresh_seeds([2310], lab, repo=ROOT)
    return {"checked_seeds": [2310], "evidence_count": len(result["evidence"]),
            "evidence_sha256": _canonical_sha(result["evidence"]), "private_evidence": result["evidence"]}


def collect_prior(origin_root: Path, continuation_root: Path, lab: Path) -> dict:
    """공통 source/runtime 검증 1회 뒤 16개 완료 바이트와 두 prelaunch failure를 검사한다."""
    origin_root, continuation_root = Path(origin_root).resolve(), Path(continuation_root).resolve()
    lab = Path(lab).resolve()
    source = B.verify(lab, SOURCE_ANCHOR)
    origin_jobs, continued_jobs, remaining = _job_sets()
    origin_intent, origin_failure = _read(origin_root / "intent.json"), _read(origin_root / "failure.json")
    if (origin_intent.get("anchor") != SOURCE_ANCHOR
            or origin_intent.get("protocol_sha256_lf") != B.digest(ROOT / B.PROTOCOL, lf=True)
            or origin_failure.get("status") != "invalid"
            or [row.get("job_id") for row in origin_failure.get("completed_jobs", [])]
            != [row["job_id"] for row in origin_jobs]):
        raise ValueError("origin 13 failure/prefix/source intent가 다르다")
    origin_records = origin_failure["completed_jobs"]
    origin_public = [_validate_stage(origin_root, job, record, source)
                     for job, record in zip(origin_jobs, origin_records, strict=True)]
    origin_prelaunch = _prelaunch(origin_root / continued_jobs[0]["job_id"], continued_jobs[0]["job_id"])

    continuation_manifest = _read(continuation_root / "continuation-manifest.json")
    v1_protocol = _verify_v1_registration(continuation_manifest, origin_root)
    continuation_intent = _read(continuation_root / "intent.json")
    continuation_failure = _read(continuation_root / "failure.json")
    continued_records = continuation_failure.get("completed_new_jobs", [])
    if (continuation_intent.get("anchor") != SOURCE_ANCHOR
            or continuation_intent.get("protocol_sha256_lf") != B.digest(ROOT / B.PROTOCOL, lf=True)
            or continuation_intent.get("execution_anchor") != V1_EXECUTION_ANCHOR
            or continuation_intent.get("continuation_protocol_sha256_lf") != V1_PROTOCOL_SHA256_LF
            or continuation_failure.get("status") != "invalid"
            or continuation_failure.get("automatic_retry") is not False
            or [row.get("job_id") for row in continued_records]
            != [row["job_id"] for row in continued_jobs]):
        raise ValueError("continuation 3 failure/prefix/intent가 다르다")
    continued_public = [_validate_stage(continuation_root, job, record, source)
                        for job, record in zip(continued_jobs, continued_records, strict=True)]
    continued_prelaunch = _prelaunch(continuation_root / remaining[0]["job_id"], remaining[0]["job_id"])
    if (continuation_manifest.get("prior_failure_sha256") != _sha(origin_root / "failure.json")
            or Path(continuation_manifest["output_root"]).resolve() != continuation_root
            or v1_protocol.get("prior_failure_sha256") != _sha(origin_root / "failure.json")):
        raise ValueError("v1 manifest/protocol이 origin failure와 다르다")
    for job in remaining[1:]:
        if (continuation_root / job["job_id"]).exists():
            raise ValueError(f"17/18번째 child stage가 이미 존재한다: {job['job_id']}")
    log_root = lab / "logs/rsl_rl/unitree_go2_flat"
    if not log_root.is_dir():
        raise ValueError("Isaac training log root가 없다")
    found = [directory.name for directory in log_root.iterdir() if directory.is_dir()
             for job in remaining if directory.name.endswith("_" + job["run_name"])]
    if found:
        raise ValueError("remaining2 run_name/log가 이미 존재한다: " + json.dumps(found))
    fresh = _fresh_2310(lab)
    return {"source_protocol": source, "records": [*origin_records, *continued_records],
            "public_jobs": [*origin_public, *continued_public], "remaining_jobs": remaining,
            "origin_failure_sha256": _sha(origin_root / "failure.json"),
            "origin_intent_sha256": _sha(origin_root / "intent.json"),
            "origin_prelaunch": origin_prelaunch,
            "continuation_failure_sha256": _sha(continuation_root / "failure.json"),
            "continuation_intent_sha256": _sha(continuation_root / "intent.json"),
            "continuation_manifest_sha256": _sha(continuation_root / "continuation-manifest.json"),
            "continuation_prelaunch": continued_prelaunch, "seed_guard": fresh}


def protocol_from_prior(prior: dict) -> dict:
    """검증된 prior 객체만 받아 개인 경로 없는 공개 protocol을 만든다."""
    seed = prior["seed_guard"]
    return {"schema": SCHEMA, "phase": "source_collection_continuation_once_v2",
            "source_collection": {"protocol": B.PROTOCOL, "anchor": SOURCE_ANCHOR,
                                  "protocol_sha256_lf": B.digest(ROOT / B.PROTOCOL, lf=True)},
            "v1_registration": {"protocol": V1_PROTOCOL, "execution_anchor": V1_EXECUTION_ANCHOR,
                                "protocol_sha256_lf": V1_PROTOCOL_SHA256_LF},
            "reused_jobs": prior["public_jobs"],
            "origin": {"completed_jobs": 13, "failure_sha256": prior["origin_failure_sha256"],
                       "intent_sha256": prior["origin_intent_sha256"],
                       "prelaunch": prior["origin_prelaunch"]},
            "continuation_v1": {"completed_jobs": 3,
                                "failure_sha256": prior["continuation_failure_sha256"],
                                "intent_sha256": prior["continuation_intent_sha256"],
                                "manifest_sha256": prior["continuation_manifest_sha256"],
                                "prelaunch": prior["continuation_prelaunch"]},
            "remaining_jobs": prior["remaining_jobs"],
            "seed_guard": {"checked_seeds": seed["checked_seeds"],
                           "evidence_count": seed["evidence_count"],
                           "evidence_sha256": seed["evidence_sha256"]},
            "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
            "gpu_admission": GPU_WAIT.policy(),
            "budget": {"reused_source_jobs": 16, "new_source_jobs": 2,
                       "new_source_updates": 600, "per_job_parent_wall_s": 1800,
                       "per_job_admission_wait_s": 600, "controller_wall_s": TOTAL_WALL_S,
                       "automatic_retries": 0, "model_api_requests": 0,
                       "paid_cloud_gpu_starts": 0},
            "claims": {"single_uninterrupted_batch": False, "retraining_prior_16": False,
                       "fault_seed_or_band_change": False}}


def prepare_protocol(origin_root: Path, continuation_root: Path, lab: Path) -> dict:
    return protocol_from_prior(collect_prior(origin_root, continuation_root, lab))


def _verify_registration(protocol_path: Path, execution_anchor: str, value: dict) -> dict:
    if not HEX40.fullmatch(execution_anchor):
        raise ValueError("v2 execution anchor 형식이 잘못됐다")
    protocol_path = Path(protocol_path).resolve()
    if _read(protocol_path) != value:
        raise ValueError("v2 public protocol이 검증 prior에서 만든 값과 다르다")
    try:
        relative = protocol_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("v2 protocol은 repo 내부여야 한다") from exc
    _git("merge-base", "--is-ancestor", execution_anchor, "HEAD")
    bound = {relative: _sha(protocol_path, lf=True), **value["source_sha256_lf"]}
    for name, expected in bound.items():
        if not (ROOT / name).is_file() or _sha(ROOT / name, lf=True) != expected \
                or _normalized_git(execution_anchor, name) != expected:
            raise ValueError(f"v2 execution blob/current SHA가 다르다: {name}")
    return value


def prepare_inputs(origin_root: Path, continuation_root: Path, protocol_path: Path,
                   execution_anchor: str, out: Path, lab: Path, isaac_python: Path) -> dict:
    origin_root, continuation_root, out = (
        Path(path).resolve() for path in (origin_root, continuation_root, out))
    lab, isaac_python = Path(lab).resolve(), Path(isaac_python).resolve()
    if out.exists():
        raise FileExistsError("v2 continuation output은 fresh 경로여야 한다")
    if not isaac_python.is_file():
        raise FileNotFoundError("Isaac Python launcher가 없다")
    prior = collect_prior(origin_root, continuation_root, lab)
    _verify_registration(protocol_path, execution_anchor, protocol_from_prior(prior))
    out.mkdir(parents=True)
    manifest = {"schema": MANIFEST_SCHEMA, "origin_root": str(origin_root),
                "continuation_root": str(continuation_root),
                "origin_failure_sha256": prior["origin_failure_sha256"],
                "continuation_failure_sha256": prior["continuation_failure_sha256"],
                "seed_guard_evidence": prior["seed_guard"]["private_evidence"],
                "seed_guard_evidence_sha256": prior["seed_guard"]["evidence_sha256"],
                "protocol_path": str(Path(protocol_path).resolve()),
                "protocol_sha256_lf": _sha(protocol_path, lf=True),
                "source_anchor": SOURCE_ANCHOR, "execution_anchor": execution_anchor,
                "lab": str(lab), "isaac_python": str(isaac_python), "output_root": str(out)}
    _write_new(out / "continuation-manifest.json", manifest)
    _write_new(out / "intent.json", {"mode": "all", "anchor": SOURCE_ANCHOR,
                                      "protocol_sha256_lf": B.digest(ROOT / B.PROTOCOL, lf=True),
                                      "execution_anchor": execution_anchor,
                                      "continuation_v2_protocol_sha256_lf": manifest["protocol_sha256_lf"]})
    return manifest


def _private_manifest(path: Path) -> tuple[dict, dict, dict]:
    manifest = _read(path)
    required = {"schema", "origin_root", "continuation_root", "origin_failure_sha256",
                "continuation_failure_sha256", "seed_guard_evidence", "seed_guard_evidence_sha256",
                "protocol_path", "protocol_sha256_lf", "source_anchor", "execution_anchor",
                "lab", "isaac_python", "output_root"}
    if set(manifest) != required or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("v2 private manifest 필드가 다르다")
    prior = collect_prior(Path(manifest["origin_root"]), Path(manifest["continuation_root"]),
                          Path(manifest["lab"]))
    protocol = _verify_registration(Path(manifest["protocol_path"]), manifest["execution_anchor"],
                                    protocol_from_prior(prior))
    if (manifest["source_anchor"] != SOURCE_ANCHOR
            or manifest["origin_failure_sha256"] != prior["origin_failure_sha256"]
            or manifest["continuation_failure_sha256"] != prior["continuation_failure_sha256"]
            or manifest["seed_guard_evidence"] != prior["seed_guard"]["private_evidence"]
            or manifest["seed_guard_evidence_sha256"] != prior["seed_guard"]["evidence_sha256"]
            or manifest["protocol_sha256_lf"] != _sha(Path(manifest["protocol_path"]), lf=True)):
        raise ValueError("v2 private manifest prior/protocol 결합이 다르다")
    return manifest, protocol, prior


def execute(manifest_path: Path, *, runner: Callable,
            wait_gpu: Callable | None = None, check_lease: Callable = B.lease_context,
            clock: Callable = time.monotonic) -> dict:
    manifest, protocol, prior = _private_manifest(Path(manifest_path).resolve())
    root, lab = Path(manifest["output_root"]), Path(manifest["lab"])
    if (root / "result.json").exists() or (root / "failure.json").exists():
        raise FileExistsError("v2 continuation 결과를 재사용하지 않는다")
    if wait_gpu is None:
        wait_gpu = GPU_WAIT.await_clear
    first_lease, started = check_lease(), clock()
    new_records = []
    try:
        for job in protocol["remaining_jobs"]:
            current = check_lease()
            if current.get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("v2 continuation 도중 GPU lease ID가 바뀌었다")
            elapsed = clock() - started
            if elapsed >= TOTAL_WALL_S:
                raise TimeoutError("v2 controller 5400초 예산이 소진됐다")
            stage = root / job["job_id"]
            stage.mkdir()
            wait_receipt = wait_gpu(stage, max_wait_s=600, consecutive_clear=2)
            receipt_path = Path(wait_receipt.get("receipt_path", "")) if isinstance(wait_receipt, dict) else Path()
            if (not isinstance(wait_receipt, dict) or wait_receipt.get("status") != "clear"
                    or not receipt_path.is_file()
                    or receipt_path.resolve().parent != (stage / "gpu-wait").resolve()
                    or _sha(receipt_path) != wait_receipt.get("receipt_sha256")):
                raise ValueError("GPU wait 최종 clear 원문 영수증이 없다")
            if check_lease().get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("v2 GPU 대기 뒤 lease ID가 바뀌었다")
            if clock() - started >= TOTAL_WALL_S:
                raise TimeoutError("v2 GPU 대기 중 controller 예산이 소진됐다")
            fixed = B.manifest(prior["source_protocol"], job, SOURCE_ANCHOR, purpose="source")
            B.CHILD.verify_binding(fixed, repo=ROOT, lab=lab)
            _write_new(stage / "manifest.json", fixed)
            _write_new(stage / "lease-before.json", current)
            argv = [manifest["isaac_python"], str(ROOT / "bench/lineage_source_training_v1.py"),
                    "--manifest", str(stage / "manifest.json"), "--lab", str(lab),
                    "--out", str(stage / "training")]
            child_budget = min(1800, TOTAL_WALL_S - (clock() - started))
            if child_budget <= 0:
                raise TimeoutError("v2 학습 자식 시작 전 controller 예산이 소진됐다")
            receipt = dict(runner(stage, argv, lab, child_budget))
            _write_new(stage / "owned-result.json", receipt)
            if check_lease().get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("v2 child 종료 뒤 GPU lease ID가 바뀌었다")
            evidence = B.completed(stage, fixed, receipt)
            record = {"job_id": job["job_id"], "purpose": "source", "arm": job["arm"],
                      "seed": job["seed"], "checkpoint": evidence["checkpoint"],
                      "evidence_sha256": _sha(stage / "training" / GUARD.EVIDENCE),
                      "owned_parent_wall_s": receipt["wall_s"]}
            new_records.append(record)
            _write_new(stage / "accepted.json", record)
        records = [*prior["records"], *new_records]
        if [row["job_id"] for row in records] != [row["job_id"] for row in B.jobs()[1]]:
            raise AssertionError("v2 aggregate가 original 18job 순서와 다르다")
        grouped: dict[str, list[str]] = {}
        for row in records:
            grouped.setdefault(row["checkpoint"]["sha256"], []).append(row["job_id"])
        result = {"schema": B.SCHEMA + "_result", "status": "complete", "mode": "all",
                  "source_jobs_completed": 18, "records": records,
                  "duplicate_checkpoint_jobs": [ids for ids in grouped.values() if len(ids) > 1],
                  "model_api_requests": 0, "fixed_evaluation_executed": False,
                  "diagnosis_or_recovery_claim": False, "wall_s": clock() - started,
                  "continuation": {"reused_origin_jobs": 13, "reused_v1_jobs": 3,
                                   "new_v2_jobs": 2,
                                   "origin_failure_sha256": prior["origin_failure_sha256"],
                                   "continuation_v1_failure_sha256": prior["continuation_failure_sha256"],
                                   "source_anchor": SOURCE_ANCHOR,
                                   "v1_execution_anchor": V1_EXECUTION_ANCHOR,
                                   "v2_execution_anchor": manifest["execution_anchor"],
                                   "v2_protocol_sha256_lf": manifest["protocol_sha256_lf"],
                                   "single_uninterrupted_batch": False,
                                   "origin_owned_parent_wall_s": sum(
                                       row["owned_parent_wall_s"] for row in prior["records"][:13]),
                                   "v1_owned_parent_wall_s": sum(
                                       row["owned_parent_wall_s"] for row in prior["records"][13:]),
                                   "v2_owned_parent_wall_s": sum(
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
    parser.add_argument("--origin-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
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
        _write_new(args.out, prepare_protocol(args.origin_root, args.continuation_root, args.lab))
        return 0
    if args.command == "validate":
        prior = collect_prior(args.origin_root, args.continuation_root, args.lab)
        print(json.dumps({"status": "valid", "reused": len(prior["records"]),
                          "remaining": len(prior["remaining_jobs"]), "api": 0}))
        return 0
    if args.protocol is None or args.execution_anchor is None:
        parser.error("prepare-inputs/execute에는 --protocol --execution-anchor가 필요하다")
    if args.command == "prepare-inputs":
        if args.out is None or args.isaac_python is None:
            parser.error("prepare-inputs에는 --out --isaac-python이 필요하다")
        prepare_inputs(args.origin_root, args.continuation_root, args.protocol,
                       args.execution_anchor, args.out, args.lab, args.isaac_python)
        return 0
    if args.manifest is None:
        parser.error("execute에는 --manifest가 필요하다")
    import run_runtime_readback_smoke as OWNED
    result = execute(args.manifest, runner=OWNED.run_owned)
    print(json.dumps({"status": result["status"], "sources": result["source_jobs_completed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
