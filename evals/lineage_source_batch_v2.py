# pyright: reportMissingImports=false
"""10 case·10 reference의 새 source와 fixed26을 순차 수집한다."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import core_seed_guard_v1 as SEEDS
import isaac_training_guard_v1 as GUARD
import lineage_source_batch_v1 as V1
import lineage_source_training_v2 as CHILD

SCHEMA = "lineage_source_collection_protocol_v2"
RESULT_SCHEMA = "lineage_source_collection_result_v2"
OWNER = V1.OWNER
ARMS = ("NONE", "NONE", "F1", "F1", "F2", "F2", "COMPOUND", "COMPOUND", "F5", "F5")
SOURCE_FILES = (
    "evals/lineage_source_batch_v2.py", "tests/test_lineage_source_batch_v2.py",
    "bench/lineage_source_training_v2.py", "tests/test_lineage_source_training_v2.py",
    "bench/run_lineage_source_batch_v2.py",
    "evals/core_seed_guard_v1.py", "tests/test_core_seed_guard_v1.py",
)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def source_files() -> tuple[str, ...]:
    return tuple(sorted(set(V1.source_files()) | set(CHILD.SOURCE_FILES) | set(SOURCE_FILES)))


def jobs(case_seeds: list[int], normal_reference_seeds: list[int], smoke_seed: int
         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if (len(case_seeds) != 10 or len(set(case_seeds)) != 10
            or len(normal_reference_seeds) != 2 or len(set(normal_reference_seeds)) != 2
            or len({*case_seeds, *normal_reference_seeds, smoke_seed}) != 13
            or any(type(seed) is not int or seed < 0
                   for seed in [*case_seeds, *normal_reference_seeds, smoke_seed])):
        raise ValueError("case10·정상 reference2·smoke1 seed는 서로 다른 음이 아닌 정수여야 한다")
    cases: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(job_id: str, case_id: str, arm: str, seed: int, role: str) -> None:
        if job_id in seen:
            return
        seen.add(job_id)
        training.append({"job_id": job_id, "case_id": case_id, "arm": arm, "seed": seed,
                         "role": role, "run_name": f"lineage_v2_s{seed}_{arm.lower()}_{role}",
                         "task": CHILD.TASK, "num_envs": 4096, "max_iterations": 300})

    for index, (seed, arm) in enumerate(zip(case_seeds, ARMS, strict=True)):
        case_id = f"case-{index + 1:02d}"
        if arm == "NONE":
            reference_seed = normal_reference_seeds[index]
            case_job, reference_job = f"none-case-s{seed}", f"none-reference-s{reference_seed}"
            relation = "different_seed_normal_reference"
            add(case_job, case_id, "NONE", seed, "case")
            add(reference_job, case_id, "NONE", reference_seed, "reference")
        else:
            case_job, reference_job = f"{arm.lower()}-s{seed}", f"none-s{seed}"
            relation = "same_seed_none_reference"
            add(reference_job, case_id, "NONE", seed, "reference")
            add(case_job, case_id, arm, seed, "case")
        cases.append({"case_id": case_id, "seed": seed, "arm": arm,
                      "source_job": case_job, "reference_job": reference_job,
                      "reference_relation": relation})
    if len(training) != 20:
        raise ValueError("고유 source training job은 20개여야 한다")
    smoke = {"job_id": f"smoke-none-s{smoke_seed}", "case_id": "preflight", "arm": "NONE",
             "seed": smoke_seed, "role": "smoke", "run_name": f"lineage_v2_smoke_s{smoke_seed}",
             "task": CHILD.TASK, "num_envs": 4096, "max_iterations": 1}
    return cases, training, smoke


def protocol_value(lab: Path, *, case_seeds: list[int], normal_reference_seeds: list[int],
                   smoke_seed: int) -> dict[str, Any]:
    cases, training, smoke = jobs(case_seeds, normal_reference_seeds, smoke_seed)
    return {
        "schema": SCHEMA, "phase": "source_training_and_fixed26_only",
        "cases": cases, "training_jobs": training, "smoke_job": smoke,
        "seed_plan": {"case": case_seeds, "normal_reference": normal_reference_seeds,
                      "smoke": smoke_seed, "replacement_seeds": 0},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in source_files()},
        "runtime_source_sha256_lf": {name: _sha(Path(lab) / name, lf=True)
                                     for name in V1.RUNTIME_FILES},
        "lease_helper_sha256_lf": _sha(Path.home() / "CodexSignals/Use-GpuLease.ps1", lf=True),
        "budget": {"smoke_training": 1, "source_training": 20, "fixed26": 20,
                   "maximum_owned_sdk_jobs": 41, "model_api": 0, "repairs": 0,
                   "automatic_retries": 0, "per_job_parent_wall_s": 1800,
                   "total_parent_wall_s": 73800},
        "runtime_facts": {"required_for_source_completion": False,
                          "fabricated_when_unavailable": False,
                          "v1_child_compatible": False},
        "outcomes": {"source_failure_absent": "보존하고 대체 seed를 뽑지 않는다.",
                     "not_applicable": "보존하고 대체 seed를 뽑지 않는다."},
        "pre_registration_prediction": {
            "NONE": "정상 reference와 case의 고정 평가가 등록 밴드를 충족할 것으로 예측한다.",
            "F1_F2_COMPOUND": "등록 결함이 고정 평가에서 source failure를 만들 것으로 예측한다.",
            "F5": "contact friction을 0.25배로 낮춘 physics 결함은 source failure가 없을 수 있고 등록 교정은 미지원이다."},
        "outcome_wording": {
            "complete": "계획된 source 20개와 각 고정 26조건 평가를 수집했다.",
            "source_failure_absent": "등록 source에서 예상한 실패가 없어 대체 seed 없이 보존했다.",
            "not_applicable": "등록 결함이 이 source에 적용되지 않아 대체 seed 없이 보존했다.",
            "invalid": "실행 결합 또는 완료 검증이 중단되어 후속 비교를 시작하지 않았다."},
        "claims": {"heldout": False, "new_fault_generalization": False,
                   "diagnosis": False, "recovery": False, "model_advantage": False},
    }


def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
    plan = value.get("seed_plan", {})
    cases, training, smoke = jobs(plan.get("case", []), plan.get("normal_reference", []),
                                  plan.get("smoke"))
    if (value.get("schema") != SCHEMA or value.get("phase") != "source_training_and_fixed26_only"
            or value.get("cases") != cases or value.get("training_jobs") != training
            or value.get("smoke_job") != smoke):
        raise ValueError("v2 case·reference·training 순서가 고정 계약과 다르다")
    if value.get("budget") != {"smoke_training": 1, "source_training": 20, "fixed26": 20,
                               "maximum_owned_sdk_jobs": 41, "model_api": 0, "repairs": 0,
                               "automatic_retries": 0, "per_job_parent_wall_s": 1800,
                               "total_parent_wall_s": 73800}:
        raise ValueError("v2 SDK·시간·API 예산이 다르다")
    if set(value.get("source_sha256_lf", {})) != set(source_files()):
        raise ValueError("v2 source closure가 다르다")
    if set(value.get("runtime_source_sha256_lf", {})) != set(V1.RUNTIME_FILES):
        raise ValueError("runtime pin 목록이 다르다")
    return deepcopy(value)


def verify(protocol_path: Path, lab: Path, anchor: str) -> dict[str, Any]:
    """등록 protocol, Git blob, runtime, lease helper를 실행 전에 함께 확인한다."""
    protocol_path = Path(protocol_path).resolve()
    try:
        relative = protocol_path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("protocol은 저장소 안에 있어야 한다") from exc
    value = validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")))
    if len(anchor) != 40 or any(character not in "0123456789abcdef" for character in anchor):
        raise ValueError("source anchor 형식이 잘못됐다")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", anchor, "HEAD"],
        capture_output=True, check=False)
    if ancestor.returncode:
        raise ValueError("source anchor가 현재 HEAD의 조상이 아니다")
    commitments = {relative: _sha(protocol_path, lf=True), **value["source_sha256_lf"]}
    for name, expected in commitments.items():
        path = ROOT / name
        if not path.is_file() or _sha(path, lf=True) != expected:
            raise ValueError(f"현재 source binding이 다르다: {name}")
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{anchor}:{name}"],
            capture_output=True, check=True).stdout
        normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"anchor source blob이 다르다: {name}")
    for name, expected in value["runtime_source_sha256_lf"].items():
        if _sha(Path(lab) / name, lf=True) != expected:
            raise ValueError(f"runtime source binding이 다르다: {name}")
    helper = Path.home() / "CodexSignals/Use-GpuLease.ps1"
    if _sha(helper, lf=True) != value["lease_helper_sha256_lf"]:
        raise ValueError("공용 GPU lease helper가 등록본과 다르다")
    CHILD.verify_binding(manifest(protocol_path, value, value["smoke_job"], anchor,
                                  purpose="smoke"), repo=ROOT, lab=lab)
    for row in value["training_jobs"]:
        CHILD.validate_manifest(manifest(protocol_path, value, row, anchor, purpose="source"))
    return value


def manifest(protocol_path: Path, protocol: dict[str, Any], row: dict[str, Any], anchor: str,
             *, purpose: str) -> dict[str, Any]:
    fields = {key: row[key] for key in CHILD.JOB_FIELDS}
    return CHILD.validate_manifest({
        "schema": CHILD.SCHEMA, "purpose": purpose, **fields, "source_commit": anchor,
        "protocol_relative": Path(protocol_path).resolve().relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
    })


def fresh(protocol: dict[str, Any], lab: Path, *, include_smoke: bool = True) -> dict[str, Any]:
    seeds = [row["seed"] for row in protocol["training_jobs"]]
    if include_smoke:
        seeds.append(protocol["smoke_job"]["seed"])
    return SEEDS.assert_fresh_seeds(seeds, lab, repo=ROOT)


def evidence_abi(evidence: dict[str, Any]) -> dict[str, Any]:
    checkpoint = evidence.get("checkpoint")
    if (not isinstance(checkpoint, dict) or set(checkpoint) < {"path", "sha256", "log_dir"}
            or evidence.get("learning_completed") is not True
            or evidence.get("recorded_before_app_close") is not True):
        raise ValueError("fixed26에 넘길 source evidence ABI가 부족하다")
    return deepcopy(evidence)


def completed(stage: Path, fixed: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    if (execution.get("exit_code") != 0 or execution.get("natural_exit") is not True
            or execution.get("cleanup_verified") is not True
            or execution.get("job_assigned_before_release") is not True
            or execution.get("timed_out") is not False or execution.get("error_type") is not None):
        raise RuntimeError("owned child의 자연 종료·종료 코드·정리가 확인되지 않았다")
    output = Path(stage) / "training"
    GUARD.validate_completion(output, CHILD.manifest_fingerprint(fixed))
    evidence = json.loads((output / GUARD.EVIDENCE).read_text(encoding="utf-8"))
    if ((output / "failure.json").exists() or evidence.get("arm") != fixed["arm"]
            or evidence.get("seed") != fixed["seed"] or evidence.get("learn_returned") is not True
            or evidence.get("learning_completed") is not True
            or evidence.get("recorded_before_app_close") is not True
            or evidence.get("update_count") != fixed["max_iterations"]):
        raise ValueError("학습 완료 evidence가 등록 job과 다르다")
    checkpoint = evidence.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if (not checkpoint_path.is_file() or _sha(checkpoint_path) != checkpoint.get("sha256")
            or checkpoint_path.resolve().parent != Path(evidence.get("log_dir", "")).resolve()):
        raise ValueError("학습 checkpoint SHA·로그 계보 결합이 다르다")
    return evidence_abi(evidence)


def verify_smoke(root: Path, protocol_path: Path, protocol: dict[str, Any],
                 anchor: str) -> dict[str, str]:
    result = json.loads((Path(root) / "result.json").read_text(encoding="utf-8"))
    intent = json.loads((Path(root) / "intent.json").read_text(encoding="utf-8"))
    if (result.get("status") != "complete" or result.get("mode") != "smoke"
            or result.get("source_jobs_completed") != 0 or len(result.get("records", [])) != 1
            or intent.get("anchor") != anchor
            or intent.get("protocol_sha256_lf") != _sha(protocol_path, lf=True)):
        raise ValueError("별도 smoke 완료 결과·등록 결합이 다르다")
    row = protocol["smoke_job"]
    stage = Path(root) / row["job_id"]
    fixed = manifest(protocol_path, protocol, row, anchor, purpose="smoke")
    if json.loads((stage / "manifest.json").read_text(encoding="utf-8")) != fixed:
        raise ValueError("smoke manifest가 현재 등록 입력과 다르다")
    evidence = completed(stage, fixed, json.loads((stage / "owned-result.json").read_text(encoding="utf-8")))
    if (result["records"][0].get("evidence_sha256") != _sha(stage / "training" / GUARD.EVIDENCE)
            or result["records"][0].get("checkpoint") != evidence["checkpoint"]):
        raise ValueError("smoke 학습 원문과 결과 참조가 다르다")
    return {"path": str(Path(root).resolve()), "result_sha256": _sha(Path(root) / "result.json"),
            "evidence_sha256": _sha(stage / "training" / GUARD.EVIDENCE)}


def run_sequence(root: Path, protocol_path: Path, protocol: dict[str, Any], anchor: str,
                 lab: Path, python: Path, *, mode: str, runner: Callable,
                 fixed_runner: Callable, check_lease: Callable = V1.lease_context,
                 check_gpu: Callable | None = None, clock: Callable = time.monotonic,
                 smoke_reference: dict[str, Any] | None = None,
                 wait: Callable | None = None) -> dict[str, Any]:
    """smoke 또는 20 training+20 fixed26을 자동 재시도 없이 순차 실행한다."""
    if mode not in {"smoke", "all"}:
        raise ValueError("mode는 smoke 또는 all이어야 한다")
    value = validate_protocol(protocol)
    if mode == "all":
        if not isinstance(smoke_reference, dict) or "path" not in smoke_reference:
            raise ValueError("전체 실행 전 별도 smoke 완료 참조가 필요하다")
        if verify_smoke(Path(smoke_reference["path"]), protocol_path, value, anchor) != smoke_reference:
            raise ValueError("smoke 재사용 참조가 원문과 다르다")
    first_lease = check_lease()
    if first_lease.get("owner") != OWNER:
        raise ValueError("공용 GPU lease owner가 다르다")
    root.mkdir(parents=True, exist_ok=False)
    _write_new(root / "intent.json", {"mode": mode, "anchor": anchor,
                                       "protocol_sha256_lf": _sha(protocol_path, lf=True),
                                       "smoke_reference": smoke_reference})
    start, records = clock(), []
    selected = [("smoke", value["smoke_job"])] if mode == "smoke" else [
        ("source", row) for row in value["training_jobs"]]
    try:
        for purpose, row in selected:
            lease = check_lease()
            if lease.get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("배치 도중 GPU lease ID가 바뀌었다")
            remaining = value["budget"]["total_parent_wall_s"] - (clock() - start)
            if remaining <= 0:
                raise TimeoutError("v2 batch 전체 wall 상한이 끝났다")
            stage = root / row["job_id"]
            stage.mkdir()
            if check_gpu is not None:
                check_gpu(stage)
            fixed = manifest(protocol_path, value, row, anchor, purpose=purpose)
            CHILD.verify_binding(fixed, repo=ROOT, lab=lab)
            _write_new(stage / "manifest.json", fixed)
            argv = [str(python), str(ROOT / "bench/lineage_source_training_v2.py"),
                    "--manifest", str(stage / "manifest.json"), "--lab", str(lab),
                    "--out", str(stage / "training")]
            execution = dict(runner(stage, argv, lab, min(
                remaining, value["budget"]["per_job_parent_wall_s"])))
            _write_new(stage / "owned-result.json", execution)
            evidence = completed(stage, fixed, execution)
            row_result: dict[str, Any] = {
                "job_id": row["job_id"], "purpose": purpose, "arm": row["arm"],
                "seed": row["seed"], "role": row["role"], "checkpoint": evidence["checkpoint"],
                "evidence_sha256": _sha(stage / "training" / GUARD.EVIDENCE),
                "records_sha256": _sha(stage / "training" / GUARD.RECORDS),
                "owned_parent_wall_s": execution["wall_s"], "runtime_facts": None,
            }
            if purpose == "source":
                if wait is None:
                    raise ValueError("fixed26 실행에는 GPU clear wait callback이 필요하다")
                report, fixed_execution = fixed_runner(
                    stage / "fixed", evidence, name=f"source_{row['job_id'].replace('-', '_')}",
                    lab=lab, isaac_python=python, runner=runner, wait=wait)
                _write_new(stage / "fixed-result.json", {
                    "report": report, "execution": fixed_execution})
                row_result["fixed26"] = {
                    "result_sha256": _sha(stage / "fixed-result.json"),
                    "conditions": 26, "execution": fixed_execution}
            _write_new(stage / "accepted.json", row_result)
            records.append(row_result)
        result = {"schema": RESULT_SCHEMA, "status": "complete", "mode": mode,
                  "records": records, "source_jobs_completed": sum(
                      row["purpose"] == "source" for row in records),
                  "fixed26_completed": sum("fixed26" in row for row in records),
                  "model_api": 0, "repairs": 0, "runtime_facts_fabricated": False,
                  "wall_s": clock() - start}
        _write_new(root / "result.json", result)
        return result
    except BaseException as error:
        _write_new(root / "failure.json", {"status": "invalid", "records": records,
                                             "error_type": type(error).__name__, "error": str(error)})
        raise

