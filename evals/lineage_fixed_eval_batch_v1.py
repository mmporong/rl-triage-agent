"""수집된 lineage source 18개를 한 owned 고정 평가 작업으로 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import brev_recovery_worker_v2 as WORKER
import lineage_evaluation_gate_v1 as GATE
import lineage_gpu_admission_v1 as GPU
import lineage_source_batch_v1 as SOURCE

SCHEMA = "lineage_fixed_eval_protocol_v1"
MANIFEST_SCHEMA = "lineage_fixed_eval_manifest_v1"
SOURCE_PROTOCOL = "bench/protocols/lineage_source_collection_20261030_v2.json"
FIXED_PROTOCOL = GATE.FIXED_PROTOCOL
BANDS = "evals/results/p0c_fixed_eval_20261009/summary.json"
WATCHDOG_S = 3600
SOURCE_FILES = (
    "evals/lineage_fixed_eval_batch_v1.py",
    "tests/test_lineage_fixed_eval_batch_v1.py",
    "bench/run_lineage_fixed_eval_v1.ps1",
    "tests/test_lineage_fixed_eval_launcher_v1.py",
    "evals/lineage_evaluation_gate_v1.py",
    "tests/test_lineage_evaluation_gate_v1.py",
    "evals/lineage_source_batch_v1.py",
    "evals/lineage_gpu_admission_v1.py",
    "bench/brev_recovery_worker_v2.py",
    "bench/run_runtime_readback_smoke.py",
    "bench/protocols/fixed_eval_v1.json",
    "evals/fixed_eval.py",
    "evals/results/p0c_fixed_eval_20261009/summary.json",
    "src/rl_triage/grounded_recovery.py",
)
RSL_PACKAGE_FILES = (
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _tree(value: Mapping[str, str]) -> str:
    raw = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=False, capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git 결합 검증에 실패했다: {detail or ' '.join(args)}")
    return completed.stdout


def _git_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("eval protocol/source는 repo 내부 파일이어야 한다") from exc


def _verify_eval_git_binding(protocol_path: Path, value: dict, anchor: str) -> None:
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    paths = {**value["source_sha256_lf"], _git_path(protocol_path): _sha(protocol_path, lf=True)}
    for name, current_sha in paths.items():
        committed = _git("show", f"{anchor}:{name}")
        normalized = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() != current_sha:
            raise ValueError(f"eval anchor blob과 현재 source SHA가 다르다: {name}")


def _verify_ancestry(source_anchor: str, eval_anchor: str) -> None:
    if HEX40.fullmatch(source_anchor) is None:
        raise ValueError("source collection anchor 형식이 잘못됐다")
    _git("merge-base", "--is-ancestor", source_anchor, eval_anchor)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_protocol() -> dict:
    value = SOURCE.validate_protocol(_read(ROOT / SOURCE_PROTOCOL))
    if value.get("next_stage", {}).get("registered") is not False:
        raise ValueError("source protocol은 다음 평가 단계를 등록하지 않은 수집 판이어야 한다")
    return value


def prepare_protocol(lab: Path) -> dict:
    """절대/private 경로가 없는 eval-only 공개 프로토콜을 만든다."""
    lab = Path(lab).resolve()
    source_protocol = _source_protocol()
    if (_sha(ROOT / FIXED_PROTOCOL, lf=True) != GATE.FIXED_PROTOCOL_SHA256_LF
            or _sha(ROOT / BANDS) != GATE.BANDS_SHA256):
        raise ValueError("동결 fixed_eval protocol 또는 원래 bands SHA가 다르다")
    source_hashes = {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES}
    runtime = {name: source_protocol["runtime_source_sha256_lf"][name]
               for name in source_protocol["runtime_source_sha256_lf"] if name not in RSL_PACKAGE_FILES}
    packages = {name: source_protocol["runtime_source_sha256_lf"][name] for name in RSL_PACKAGE_FILES}
    for name, digest in {**runtime, **packages}.items():
        if not (lab / name).is_file() or _sha(lab / name, lf=True) != digest:
            raise ValueError(f"Isaac runtime/RSL pin이 source collection과 다르다: {name}")
    value = {
        "schema": SCHEMA, "phase": "eval_only", "task": GATE.TASK,
        "source_collection_protocol": {"path": SOURCE_PROTOCOL,
                                         "sha256_lf": _sha(ROOT / SOURCE_PROTOCOL, lf=True),
                                         "source_tree_sha256": _tree(source_protocol["source_sha256_lf"])},
        "source_sha256_lf": source_hashes, "source_tree_sha256": _tree(source_hashes),
        "runtime_source_sha256_lf": runtime, "python_package_source_sha256_lf": packages,
        "fixed_eval": {"path": FIXED_PROTOCOL, "sha256_lf": GATE.FIXED_PROTOCOL_SHA256_LF,
                       "eval_seed": 2026, "conditions": 26},
        "bands": {"path": BANDS, "sha256": GATE.BANDS_SHA256},
        "budget": {"owned_workers": 1, "evaluation_jobs": 18, "training_updates": 0,
                   "model_api_requests": 0, "automatic_retries": 0,
                   "whole_controller_wall_s": WATCHDOG_S,
                   "basis": "기존 동일 fixed_eval 보고서 약 13초×18보다 큰 단일 owned 상한"},
        "gpu_admission_profile": GPU.GPU_PROFILE,
        "source_next_stage_was_registered": False,
        "claims": {"training": False, "diagnosis": False, "recovery": False,
                   "heldout": False, "generalization": False},
    }
    if any(Path(text).is_absolute() for text in (SOURCE_PROTOCOL, FIXED_PROTOCOL, BANDS, *SOURCE_FILES)):
        raise AssertionError("공개 프로토콜에 절대경로를 넣을 수 없다")
    return value


def verify(protocol_path: Path, *, lab: Path, anchor: str) -> dict:
    if HEX40.fullmatch(anchor) is None:
        raise ValueError("source anchor 형식이 잘못됐다")
    value = _read(protocol_path)
    if value != prepare_protocol(lab):
        raise ValueError("eval-only 프로토콜·source closure·runtime pin 결합이 다르다")
    _verify_eval_git_binding(Path(protocol_path), value, anchor)
    return value


def _source_records(source_root: Path, source_protocol: dict,
                    source_anchor: str) -> tuple[dict, dict[str, dict]]:
    result_path, intent_path = source_root / "result.json", source_root / "intent.json"
    result, intent = _read(result_path), _read(intent_path)
    jobs = {row["job_id"]: row for row in source_protocol["training_jobs"]}
    records = [row for row in result.get("records", [])
               if isinstance(row, dict) and row.get("purpose") == "source"]
    by_job = {row.get("job_id"): row for row in records}
    if (result.get("status") != "complete" or result.get("source_jobs_completed") != 18
            or len(records) != 18 or set(by_job) != set(jobs)
            or intent.get("anchor") != source_anchor
            or intent.get("protocol_sha256_lf") != _sha(ROOT / SOURCE_PROTOCOL, lf=True)):
        raise ValueError("수집 완료 18개·anchor·source protocol 결합이 다르다")
    artifacts = {}
    for job_id, job in jobs.items():
        row, checkpoint = by_job[job_id], by_job[job_id].get("checkpoint")
        path = Path(checkpoint.get("path", "")) if isinstance(checkpoint, dict) else Path()
        if (row.get("arm") != job["arm"] or row.get("seed") != job["seed"]
                or not path.is_absolute() or not path.is_file()
                or HEX64.fullmatch(str(checkpoint.get("sha256", ""))) is None
                or _sha(path) != checkpoint["sha256"]):
            raise ValueError(f"source checkpoint 바이트·job 결합이 다르다: {job_id}")
        artifacts[job_id] = {"checkpoint": {"path": str(path.resolve()),
                                             "sha256": checkpoint["sha256"]},
                             "training_seed": job["seed"], "arm": job["arm"]}
    return result, artifacts


def prepare_inputs(source_root: Path, protocol_path: Path, anchor: str, out: Path,
                   lab: Path, isaac_python: Path) -> dict:
    """수집 receipt를 실제 worker evaluate job과 controller manifest로 고정한다."""
    source_root, protocol_path = Path(source_root).resolve(), Path(protocol_path).resolve()
    out, lab, isaac_python = Path(out).resolve(), Path(lab).resolve(), Path(isaac_python).resolve()
    if out.exists():
        raise FileExistsError("평가 output은 fresh 경로여야 한다")
    if not isaac_python.is_file():
        raise FileNotFoundError("Isaac Python launcher가 없다")
    protocol = verify(protocol_path, lab=lab, anchor=anchor)
    source_protocol = _source_protocol()
    source_intent = _read(source_root / "intent.json")
    source_anchor = source_intent.get("anchor")
    if not isinstance(source_anchor, str) or HEX40.fullmatch(source_anchor) is None:
        raise ValueError("source collection intent anchor 형식이 잘못됐다")
    if SOURCE.verify(lab, source_anchor) != source_protocol:
        raise ValueError("source collection protocol/current blob 재검증이 다르다")
    _verify_ancestry(source_anchor, anchor)
    _, artifacts = _source_records(source_root, source_protocol, source_anchor)
    out.mkdir(parents=True)
    stage = out / "owned-evaluate"
    stage.mkdir()
    job = {
        "schema": "brev_recovery_worker_job_v1", "mode": "evaluate",
        "stage": str(stage), "lab": str(lab), "repo": str(ROOT.resolve()),
        "protocol_path": str(protocol_path), "protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_commit_sha": anchor, "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": protocol["python_package_source_sha256_lf"],
        "artifacts": artifacts, "fixed_eval_protocol": str((ROOT / FIXED_PROTOCOL).resolve()),
        "reports_dir": str(stage / "reports"), "jobs_path": str(stage / "jobs.json"),
    }
    WORKER.validate_job(job)
    manifest = {
        "schema": MANIFEST_SCHEMA, "protocol_path": str(protocol_path),
        "protocol_sha256_lf": _sha(protocol_path, lf=True), "source_root": str(source_root),
        "source_result_path": str(source_root / "result.json"),
        "source_result_sha256": _sha(source_root / "result.json"),
        "source_protocol_sha256_lf": _sha(ROOT / SOURCE_PROTOCOL, lf=True),
        "eval_execution_anchor": anchor, "source_collection_anchor": source_anchor,
        "lab": str(lab), "isaac_python": str(isaac_python),
        "output_root": str(out), "worker_job": job,
        "source_artifact_count": len(artifacts),
        "training_seed_scope": "source protocol jobs; fixed evaluation uses eval_seed 2026",
    }
    _write_new(stage / "job.json", job)
    _write_new(out / "manifest.json", manifest)
    _write_new(out / "intent.json", {"mode": "eval_only", "api": 0, "training": False,
                                      "source_result_sha256": manifest["source_result_sha256"]})
    return manifest


def _validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = _read(path)
    required = {"schema", "protocol_path", "protocol_sha256_lf", "source_root",
                "source_result_path", "source_result_sha256", "source_protocol_sha256_lf",
                "eval_execution_anchor", "source_collection_anchor", "lab", "isaac_python",
                "output_root", "worker_job",
                "source_artifact_count", "training_seed_scope"}
    if set(manifest) != required or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("actual evaluation manifest 필드가 다르다")
    if _sha(Path(manifest["source_result_path"])) != manifest["source_result_sha256"]:
        raise ValueError("source result가 준비 뒤 바뀌었다")
    protocol = verify(Path(manifest["protocol_path"]), lab=Path(manifest["lab"]),
                      anchor=manifest["eval_execution_anchor"])
    _verify_ancestry(manifest["source_collection_anchor"], manifest["eval_execution_anchor"])
    if (manifest["protocol_sha256_lf"] != _sha(Path(manifest["protocol_path"]), lf=True)
            or manifest["source_protocol_sha256_lf"] != _sha(ROOT / SOURCE_PROTOCOL, lf=True)
            or manifest["source_artifact_count"] != 18):
        raise ValueError("actual manifest protocol/source count 결합이 다르다")
    job = WORKER.validate_job(manifest["worker_job"])
    if _read(Path(job["stage"]) / "job.json") != job:
        raise ValueError("저장 worker job이 manifest와 다르다")
    for artifact in job["artifacts"].values():
        checkpoint = artifact["checkpoint"]
        if _sha(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("평가 입력 checkpoint가 준비 뒤 바뀌었다")
    return manifest, protocol


def execute(manifest_path: Path, *, runner: Callable,
            check_lease: Callable = SOURCE.lease_context,
            check_gpu: Callable | None = GPU.snapshot) -> dict:
    """동일 lease에서 단일 owned worker를 실행하고 gate 결과까지 소비한다."""
    manifest_path = Path(manifest_path).resolve()
    manifest, _ = _validate_manifest(manifest_path)
    root = Path(manifest["output_root"])
    if (root / "result.json").exists() or (root / "failure.json").exists():
        raise FileExistsError("평가 controller 결과를 재사용하지 않는다")
    stage, job = Path(manifest["worker_job"]["stage"]), manifest["worker_job"]
    first_lease = check_lease()
    started = time.monotonic()
    try:
        if check_gpu is not None:
            check_gpu(stage)
        before = check_lease()
        if before.get("leaseId") != first_lease.get("leaseId"):
            raise ValueError("평가 시작 전 GPU lease ID가 바뀌었다")
        argv = [manifest["isaac_python"], str(ROOT / "bench/brev_recovery_worker_v2.py"),
                "--job", str(stage / "job.json"), "--output", str(stage / "worker.json")]
        receipt = dict(runner(stage, argv, Path(manifest["lab"]), WATCHDOG_S))
        _write_new(stage / "owned-result.json", receipt)
        after = check_lease()
        if after.get("leaseId") != first_lease.get("leaseId"):
            raise ValueError("평가 종료 뒤 GPU lease ID가 바뀌었다")
        if (receipt.get("exit_code") != 0 or receipt.get("natural_exit") is not True
                or receipt.get("cleanup_verified") is not True
                or receipt.get("job_assigned_before_release") is not True
                or receipt.get("timed_out") is not False or receipt.get("error_type") is not None
                or not isinstance(receipt.get("wall_s"), (int, float))
                or isinstance(receipt.get("wall_s"), bool)
                or not 0 <= receipt["wall_s"] <= WATCHDOG_S):
            raise RuntimeError("owned 평가 child의 exit0·자연 종료·정리가 확인되지 않았다")
        worker = _read(stage / "worker.json")
        reports = worker.get("reports")
        if worker.get("completed") is not True or not isinstance(reports, dict) or set(reports) != set(job["artifacts"]):
            raise ValueError("worker의 18개 고정 평가 완료 receipt가 없다")
        index = {}
        for job_id, artifact in job["artifacts"].items():
            report = reports[job_id]
            path = Path(report.get("path", "")) if isinstance(report, dict) else Path()
            if (not path.is_file() or _sha(path) != report.get("sha256")
                    or _sha(Path(artifact["checkpoint"]["path"])) != artifact["checkpoint"]["sha256"]):
                raise ValueError("평가 report 또는 source checkpoint가 실행 중 바뀌었다")
            index[job_id] = {"report_path": str(path.resolve()), "report_sha256": report["sha256"],
                             "checkpoint_sha256": artifact["checkpoint"]["sha256"]}
        _write_new(root / "report-index.json", index)
        gate = GATE.consume_gate(manifest["source_result_path"], ROOT / SOURCE_PROTOCOL, index,
                                 repo=ROOT, bands_path=ROOT / BANDS, out=root / "gate.json")
        result = {"schema": SCHEMA + "_result", "status": "complete", "evaluation_jobs": 18,
                  "owned_workers": 1, "model_api_requests": 0, "training_updates": 0,
                  "report_index_path": str(root / "report-index.json"), "gate": gate,
                  "owned_parent_wall_s": receipt["wall_s"], "controller_wall_s": time.monotonic() - started}
        _write_new(root / "result.json", result)
        return result
    except BaseException as error:
        _write_new(root / "failure.json", {"status": "invalid", "error_type": type(error).__name__,
                                           "error": str(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "prepare-inputs", "execute"))
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.out is None:
            parser.error("prepare에는 --out이 필요하다")
        _write_new(args.out, prepare_protocol(args.lab))
        return 0
    if args.anchor is None:
        parser.error("verify/prepare-inputs/execute에는 --anchor가 필요하다")
    verify(args.protocol, lab=args.lab, anchor=args.anchor)
    if args.command == "verify":
        return 0
    if args.command == "prepare-inputs":
        if args.source_root is None or args.out is None or args.isaac_python is None:
            parser.error("prepare-inputs에는 --source-root --out --isaac-python이 필요하다")
        prepare_inputs(args.source_root, args.protocol, args.anchor, args.out, args.lab, args.isaac_python)
        return 0
    if args.manifest is None:
        parser.error("execute에는 --manifest가 필요하다")
    import run_runtime_readback_smoke as OWNED
    execute(args.manifest, runner=OWNED.run_owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
