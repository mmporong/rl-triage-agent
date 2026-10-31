"""새 seed의 등록 결함 범주를 기존 guarded 학습·fixed26 경계로 실행한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_training_v1 as TRAIN
import runtime_observation_training_v1 as RUNTIME

SCHEMA = "prospective_diagnosis_training_protocol_v1"
RESULT_SCHEMA = "prospective_diagnosis_training_result_v1"
CATEGORIES = {
    "NONE": {"arm": "NONE", "original_faults": []},
    "F1": {"arm": "F1", "original_faults": ["F1"]},
    "R1_ONLY": {"arm": "R1", "original_faults": ["R1"]},
    "COMPOUND": {"arm": "COMPOUND", "original_faults": ["F1", "R1"]},
}
SOURCE_FILES = tuple(sorted({
    *TRAIN.BOUND_SOURCES, *RUNTIME.EXEC2.SOURCE_FILES,
    "evals/runtime_observation_training_v1.py",
    "tests/test_runtime_observation_training_v1.py",
    "evals/runtime_training_inputs_v1.py",
    "tests/test_runtime_training_inputs_v1.py",
    "evals/optimizer_runtime_facts_v1.py",
    "tests/test_optimizer_runtime_facts_v1.py",
    "evals/lineage_recovery_inputs_v1.py",
    "tests/test_lineage_recovery_inputs_v1.py",
    "evals/lineage_selection_inputs_v1.py",
    "tests/test_lineage_selection_inputs_v1.py",
    "evals/residual_selection_v2.py",
    "tests/test_residual_selection_v2.py",
    "bench/prospective_diagnosis_training_v1.py",
    "tests/test_prospective_diagnosis_training_v1.py",
    "bench/run_prospective_diagnosis_training_v1.py",
    "tests/test_run_prospective_diagnosis_training_v1.py",
    "evals/prospective_diagnosis_inputs_v1.py",
    "tests/test_prospective_diagnosis_inputs_v1.py",
}))
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    _require(result.returncode == 0, "Git anchor/blob 결합을 확인하지 못했다")
    return result.stdout


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 내부여야 한다") from error


def prepare_protocol(
    seeds: list[int], categories: list[str], runtime_source_sha256_lf: dict[str, str],
) -> dict[str, Any]:
    """root가 seed 소비 확인 뒤 커밋할 protocol 본문을 생성한다."""
    _require(len(seeds) >= 2 and len(set(seeds)) == len(seeds)
             and all(type(seed) is int and seed >= 0 for seed in seeds),
             "서로 다른 새 seed가 최소 2개 필요하다")
    _require(bool(categories) and len(set(categories)) == len(categories)
             and set(categories) <= set(CATEGORIES) and {"NONE", "R1_ONLY"} <= set(categories),
             "범주는 NONE/R1_ONLY를 포함한 등록 집합이어야 한다")
    _require(isinstance(runtime_source_sha256_lf, dict) and len(runtime_source_sha256_lf) == 9
             and all(isinstance(name, str) and HEX64.fullmatch(digest)
                     for name, digest in runtime_source_sha256_lf.items()),
             "runtime source pin은 9개 SHA여야 한다")
    return {
        "schema": SCHEMA, "purpose": "prospective_registered_diagnosis_training",
        "seeds": list(seeds), "categories": list(categories),
        "category_contract": deepcopy(CATEGORIES),
        "training": {"task": TRAIN.RESIDUAL.TASK, "num_envs": 4096, "updates": 300,
                     "configured_learning_epochs": 5, "configured_mini_batches": 4,
                     "runtime_updates_to_observe": 3, "fresh_initialization": True,
                     "checkpoint_loaded": False},
        "evaluation": {"fixed_conditions": 26, "unchanged_existing_evaluator": True},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "runtime_source_sha256_lf": deepcopy(runtime_source_sha256_lf),
        "scope": {"registered_fault_families_only": True, "model_calls": 0,
                  "source_checkpoint_provenance_only": True,
                  "fault_labels_in_model_view": False, "physical_paths_in_model_view": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value == prepare_protocol(
        value.get("seeds", []), value.get("categories", []),
        value.get("runtime_source_sha256_lf", {})), "prospective protocol 내용 또는 source가 다르다")
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "anchor 형식이 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _require(subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", anchor, "HEAD"],
                            capture_output=True, check=False).returncode == 0,
             "anchor가 현재 HEAD의 조상이 아니다")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for name, expected in bound.items():
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == expected,
                 f"현재 source SHA가 다르다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == expected,
                 f"anchor source blob이 다르다: {name}")
    for name, expected in value["runtime_source_sha256_lf"].items():
        target = Path(lab).resolve() / name
        _require(target.is_file() and _sha(target, lf=True) == expected,
                 f"runtime source SHA가 다르다: {name}")
    return deepcopy(value)


def build_manifest(
    protocol_path: Path, protocol: Mapping[str, Any], anchor: str, *, stage: Path,
    seed: int, category: str, run_name: str, mode: str = "train",
    source_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """범주명을 residual 실제 arm으로 명시 변환해 고정 manifest를 만든다."""
    _require(category in CATEGORIES and category in protocol.get("categories", []),
             "등록되지 않은 학습 범주")
    _require(seed in protocol.get("seeds", []), "등록되지 않은 seed")
    _require(mode in {"train", "inspection"}, "mode 형식이 다르다")
    if mode == "train":
        _require(source_checkpoint_sha256 is None, "새 학습은 checkpoint weight를 불러오지 않는다")
    else:
        _require(category == "NONE" and isinstance(source_checkpoint_sha256, str)
                 and HEX64.fullmatch(source_checkpoint_sha256) is not None,
                 "inspection은 NONE checkpoint provenance가 필요하다")
    contract = CATEGORIES[category]
    manifest = {
        "schema": TRAIN.SCHEMA, "mode": mode, "arm": contract["arm"], "seed": seed,
        "run_name": run_name, "task": TRAIN.RESIDUAL.TASK, "num_envs": 4096,
        "max_iterations": 300 if mode == "train" else 3,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": anchor,
        "protocol_path": _relative(protocol_path),
        "protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_sha256_lf": deepcopy(cast(dict[str, str], protocol["source_sha256_lf"])),
        "runtime_source_sha256_lf": deepcopy(
            cast(dict[str, str], protocol["runtime_source_sha256_lf"])),
        "records_path": str(Path(stage).resolve() / "observer-records.jsonl"),
        "readback_path": str(Path(stage).resolve() / "readback.json"),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "original_faults": deepcopy(contract["original_faults"]),
        "applied_corrections": [],
    }
    return TRAIN.validate_manifest(manifest)


def run_case(
    stage: Path, fixed: dict[str, Any], *, category: str, lab: Path,
    isaac_python: Path, runner: Callable, wait: Callable,
) -> dict[str, Any]:
    """guarded 학습·수집 뒤 결과를 봉인하고 같은 checkpoint를 fixed26으로 평가한다."""
    stage = Path(stage).resolve()
    _require(not stage.exists(), "case stage는 fresh 경로여야 한다")
    expected = CATEGORIES.get(category)
    _require(expected is not None and fixed.get("arm") == expected["arm"]
             and fixed.get("original_faults") == expected["original_faults"]
             and fixed.get("applied_corrections") == [], "category와 manifest 의미가 다르다")
    training = RUNTIME.run_training(
        stage / "train", fixed, lab=lab, isaac_python=isaac_python, runner=runner, wait=wait)
    stage.mkdir(parents=True, exist_ok=True)
    training_path = stage / "training-result.json"
    training_path.write_text(json.dumps(training, ensure_ascii=False, sort_keys=True, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    evidence = json.loads(Path(training["evidence"]["path"]).read_text(encoding="utf-8"))
    report, execution = RUNTIME.run_fixed(
        stage / "fixed", evidence, name=f"pd_{category.lower()}_s{fixed['seed']}",
        lab=lab, isaac_python=isaac_python, runner=runner, wait=wait)
    report_path = stage / "fixed/corrected.json"
    _require(report_path.is_file(), "fixed26 report가 없다")
    result = {
        "schema": RESULT_SCHEMA, "status": "complete", "seed": fixed["seed"],
        "category": category, "runtime_arm": fixed["arm"],
        "training": {"path": str(training_path), "sha256": _sha(training_path)},
        "checkpoint": deepcopy(training["checkpoint"]),
        "fixed_report": {"path": str(report_path), "sha256": _sha(report_path)},
        "fixed_execution": deepcopy(execution), "report": deepcopy(report),
        "claims": {"recovery": False, "diagnosis": False, "model_advantage": False},
    }
    result_path = stage / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return result


__all__ = ["CATEGORIES", "SCHEMA", "SOURCE_FILES", "build_manifest", "prepare_protocol",
           "run_case", "verify_protocol"]
