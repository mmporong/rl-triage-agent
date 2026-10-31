"""등록된 새 seed 학습과 고정 평가를 기존 guarded 실행기로 연결한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_recovery_execution_v1 as EXEC
import lineage_recovery_execution_v2 as EXEC2
import lineage_recovery_training_v1 as TRAIN
import optimizer_runtime_facts_v1 as FACT
import runtime_training_inputs_v1 as INPUT

HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ARMS = {"NONE", "COMPOUND", "FIRST", "BOTH"}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("protocol은 저장소 내부 파일이어야 한다") from exc


def _git(*args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if completed.returncode:
        raise ValueError("Git anchor/blob 결합을 확인하지 못했다")
    return completed.stdout


def _registered_sources(protocol_path: Path, protocol: Mapping[str, Any], anchor: str) -> None:
    _require(HEX40.fullmatch(anchor) is not None, "anchor 형식이 잘못됐다")
    source = protocol.get("source_sha256_lf")
    runtime = protocol.get("runtime_source_sha256_lf")
    _require(isinstance(source, dict) and TRAIN.BOUND_SOURCES <= set(source),
             "protocol에 guarded training source closure가 없다")
    _require(isinstance(runtime, dict) and len(runtime) == 9,
             "protocol runtime source closure는 9개여야 한다")
    source = cast(dict[str, str], source)
    relative = _relative(protocol_path)
    _require(_sha(protocol_path, lf=True) == protocol.get("protocol_sha256_lf", _sha(protocol_path, lf=True)),
             "protocol self SHA가 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", anchor, "HEAD"],
        capture_output=True, check=False)
    _require(ancestor.returncode == 0, "anchor가 현재 HEAD의 조상이 아니다")
    bound = {relative: _sha(protocol_path, lf=True), **source}
    for name, expected in bound.items():
        _require(isinstance(expected, str) and HEX64.fullmatch(expected) is not None,
                 f"source SHA 형식이 잘못됐다: {name}")
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == expected,
                 f"현재 source가 protocol과 다르다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == expected,
                 f"anchor blob이 protocol과 다르다: {name}")


def build_manifest(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    anchor: str,
    *,
    stage: Path,
    seed: int,
    arm: str,
    original_faults: list[str],
    applied_corrections: list[str],
    source_checkpoint_sha256: str | None,
    run_name: str,
    mode: str = "train",
) -> dict[str, Any]:
    """등록 protocol에서 residual manifest를 만들고 고정 계약으로 검증한다."""
    protocol_path, stage = Path(protocol_path).resolve(), Path(stage).resolve()
    _registered_sources(protocol_path, protocol, anchor)
    _require(arm in ARMS, "이 실행기는 NONE/COMPOUND/FIRST/BOTH arm만 허용한다")
    _require(mode in {"train", "inspection"}, "mode는 train 또는 inspection이어야 한다")
    if mode == "inspection":
        _require(arm == "NONE", "inspection smoke는 NONE arm만 허용한다")
    source = protocol["source_sha256_lf"]
    runtime = protocol["runtime_source_sha256_lf"]
    manifest = {
        "schema": TRAIN.SCHEMA, "mode": mode, "arm": arm, "seed": seed,
        "run_name": run_name, "task": TRAIN.RESIDUAL.TASK, "num_envs": 4096,
        "max_iterations": 300 if mode == "train" else 3,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": anchor,
        "protocol_path": _relative(protocol_path), "protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_sha256_lf": deepcopy(source), "runtime_source_sha256_lf": deepcopy(runtime),
        "records_path": str((stage / "observer-records.jsonl").resolve()),
        "readback_path": str((stage / "readback.json").resolve()),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "original_faults": list(original_faults),
        "applied_corrections": list(applied_corrections),
    }
    return TRAIN.validate_manifest(manifest)


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def run_training(
    stage: Path,
    fixed: dict[str, Any],
    *,
    lab: Path,
    isaac_python: Path,
    runner: Callable,
    wait: Callable,
) -> dict[str, Any]:
    """기존 guarded 실행기로 학습하고 300 update 결과만 입력과 fact로 봉인한다."""
    stage, lab, isaac_python = Path(stage).resolve(), Path(lab).resolve(), Path(isaac_python).resolve()
    _require(not stage.exists(), "training stage는 fresh 경로여야 한다")
    _require(not Path(fixed["records_path"]).exists() and not Path(fixed["readback_path"]).exists(),
             "observer/readback 출력은 실행 전에 없어야 한다")
    TRAIN.verify_binding(fixed, lab=lab)
    outer = {"lab": str(lab), "isaac_python": str(isaac_python)}
    watchdog = EXEC.WATCHDOG["train"] if fixed["mode"] == "train" else EXEC.WATCHDOG["smoke"]
    execution, evidence = EXEC._run_training(
        stage, fixed, outer, {}, runner=EXEC2._owned_runner(runner), wait=wait, watchdog=watchdog)
    result: dict[str, Any] = {
        "manifest": _ref(stage / "manifest.json"),
        "evidence": _ref(stage / "training" / EXEC.GUARD.EVIDENCE),
        "owned_result": _ref(stage / "owned-result.json"),
        "checkpoint": deepcopy(evidence["checkpoint"]),
        "execution": deepcopy(execution),
        "collected": None,
        "runtime_facts": None,
    }
    if fixed["mode"] == "train":
        collected_dir = stage / "collected"
        INPUT.collect(
            stage / "manifest.json", stage / "training", stage / "owned-result.json",
            lab=lab, isaac_python=isaac_python, out=collected_dir)
        facts: dict[str, dict[str, str]] = {}
        for role in ("case", "reference"):
            target = stage / "runtime-facts" / role
            FACT.prepare(
                stage / "manifest.json", stage / "training", stage / "owned-result.json",
                role=role, lab=lab,
                expected_checkpoint_sha256=evidence["checkpoint"]["sha256"], out=target)
            facts[role] = _ref(target / FACT.ARTIFACT_NAME)
        result["collected"] = _ref(collected_dir / INPUT.COLLECTED)
        result["runtime_facts"] = facts
    return result


def run_fixed(
    stage: Path,
    evidence: dict[str, Any],
    *,
    name: str,
    lab: Path,
    isaac_python: Path,
    runner: Callable,
    wait: Callable,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """기존 고정 26조건 평가기를 고유 stage/name으로 실행한다."""
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,49}", name) is not None,
             "fixed evaluation name 형식이 잘못됐다")
    outer = {"lab": str(Path(lab).resolve()), "isaac_python": str(Path(isaac_python).resolve())}
    row = {"case_id": name, "policy": "runtime"}
    return EXEC._fixed_evaluate(
        Path(stage).resolve(), row, outer, evidence,
        runner=EXEC2._owned_runner(runner), wait=wait)
