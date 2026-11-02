# pyright: reportMissingImports=false
"""Continuation FIRST와 V9 SECOND를 검증한 뒤 기존 same-Play 녹화기를 재사용한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_five_policy_recovery_v1 as EXPORT1
import export_five_policy_recovery_v2 as EXPORT2
import recovery_epoch2_training_boundary_v1 as TRAIN_BOUNDARY
import run_five_policy_recovery_resume_v2 as CONT
import run_five_policy_recovery_resume_v9 as V9
import run_recovery_same_play_capture_v1 as V1

SCHEMA = "recovery_same_play_capture_continuation_adapter_v1"
CONFIG_SCHEMA = "recovery_same_play_capture_config_v2"
SOURCE_FILES = tuple(sorted({
    "bench/run_recovery_same_play_capture_v2.py",
    "tests/test_recovery_same_play_capture_v2.py",
    *CONT.SOURCE_FILES, *EXPORT2.SOURCE_FILES, *V9.SOURCE_FILES,
}))
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_LOCK = RLock()
_ORIGINAL_TARGETS = V1.validated_targets
_ORIGINAL_CONFIG_SCHEMA = V1.CONFIG_SCHEMA


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object가 아니다")
    return cast(dict[str, Any], value)


def relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("adapter 파일은 repo 내부여야 한다") from error


def ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    return {"path": str(target), "sha256": sha(target)}


def public_ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    return {"path": relative(target), "sha256_lf": sha(target, lf=True)}


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    require(result.returncode == 0, "Git adapter 결합을 확인하지 못했다")
    return result.stdout


def protocol_value(parent_design: Path, parent_anchor: str) -> dict[str, Any]:
    require(_HEX40.fullmatch(parent_anchor) is not None, "parent design anchor 형식")
    V1.CAPTURE.validate_design(parent_design)
    head = _git("rev-parse", "HEAD").decode("ascii").strip()
    require(_HEX40.fullmatch(head) is not None, "현재 HEAD 형식")
    V1.CAPTURE.validate_source_anchor(parent_design, parent_anchor, head)
    return {
        "schema": SCHEMA, "status": "preregistered_conditional_adapter",
        "parent_design": {**public_ref(parent_design), "anchor": parent_anchor},
        "sources_lf": {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "authority": {"gpu": False, "api": False, "capture_approved": False,
                      "model_ready": False, "used_for_decision": False},
        "implementation_coupling": "created_after_second_recovered_result_for_registered_media_consumption",
    }


def verify_protocol(path: Path, anchor: str) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    require(_HEX40.fullmatch(anchor) is not None, "adapter anchor 형식")
    value = read(target)
    parent = value.get("parent_design")
    require(isinstance(parent, Mapping), "parent design ref가 없다")
    parent = cast(Mapping[str, Any], parent)
    parent_path = ROOT / str(parent.get("path", ""))
    expected = protocol_value(parent_path, str(parent.get("anchor", "")))
    require(value == expected, "adapter protocol 값이 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    for name, digest in {relative(target): sha(target, lf=True), **value["sources_lf"]}.items():
        current = ROOT / name
        require(current.is_file() and sha(current, lf=True) == digest, f"adapter current source: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(blob).hexdigest() == digest, f"adapter anchor source: {name}")
    return deepcopy(value)


def _facade(module: ModuleType, name: str) -> ModuleType:
    clone = ModuleType(name)
    clone.__dict__.update(module.__dict__)
    clone.__name__ = name
    return clone


def _isolated_v1() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_recovery_capture_v1_continuation", V1.__file__)
    require(spec is not None and spec.loader is not None, "v1 capture clone loader")
    clone = importlib.util.module_from_spec(cast(Any, spec))
    cast(Any, spec).loader.exec_module(clone)
    export = _facade(EXPORT1, "_capture_export_continuation")
    parent = _facade(EXPORT1.PARENT, "_capture_parent_continuation")
    parent.__dict__["FIRST_RESULT_SCHEMA"] = CONT.FIRST_CONTINUATION_RESULT_SCHEMA
    export.__dict__["PARENT"] = parent
    clone.__dict__["EXPORT"] = export
    return clone


def _export_binding(folder: Path, first_path: Path, second_path: Path) -> None:
    manifest = read(folder / "manifest.json")
    require(manifest.get("schema") == EXPORT2.MANIFEST_SCHEMA and manifest.get("status") == "complete",
            "V2 public manifest")
    for name in ("summary.json", "cases.json", "provenance.json"):
        path = folder / name
        row = manifest.get("files", {}).get(name)
        require(path.is_file() and isinstance(row, Mapping)
                and row.get("sha256") == sha(path) and row.get("size_bytes") == path.stat().st_size,
                f"V2 public {name}")
    provenance = read(folder / "provenance.json")
    require(provenance.get("first_result_schema") == CONT.FIRST_CONTINUATION_RESULT_SCHEMA
            and provenance.get("first_result", {}).get("sha256") == sha(first_path)
            and provenance.get("second_result", {}).get("sha256") == sha(second_path),
            "V2 public/raw result 결합")


def _verify_selected_second(second_path: Path, *, v9_plan: Mapping[str, Any],
                            lab: Path) -> dict[str, Any]:
    """선택된 SECOND receipt의 training·fixed26·recovery를 원 검증기로 재계산한다."""
    second = read(second_path)
    rows = second.get("records")
    require(isinstance(rows, list), "SECOND records가 없다")
    rows = cast(list[dict[str, Any]], rows)
    matches = [row for row in rows if row.get("case_id") == "case-07"
               and row.get("policy") == "logs_only"]
    require(len(matches) == 1 and matches[0].get("outcome") == "recovered",
            "선택 SECOND recovered row가 하나가 아니다")
    row = cast(dict[str, Any], matches[0])
    result_path = V1.bound(row.get("result"), "selected SECOND result")
    result = read(result_path)
    receipt = read(V1.bound(result.get("receipt"), "selected SECOND receipt"))
    candidate_path = V1.bound(row.get("candidate"), "selected SECOND candidate")
    candidate = read(candidate_path)
    require(receipt.get("candidate") == V1.B._ref(candidate_path), "SECOND receipt/candidate")
    parent_path = V1.bound(v9_plan["parent_protocol"], "V9 parent protocol")
    parent_private = V1.bound(v9_plan["parent_private_plan"], "V9 parent private plan")
    parent = V9.V8.verify_protocol(parent_path, anchor=v9_plan["parent_anchor"],
                                   private_plan=parent_private)
    _v7, _v6, _v5, second_plan_path, _sources = V9.V8._parents(parent)
    second_plan = read(second_plan_path)
    loop_path = V1.bound(second_plan["loop_protocol"], "SECOND loop protocol")
    with TRAIN_BOUNDARY.training_boundary_scope(
        loop_protocol=loop_path, loop_anchor=second_plan["anchor"],
        training_protocol=V1.bound(v9_plan["training_protocol"], "training protocol"),
        training_anchor=v9_plan["training_anchor"], lab=lab,
    ):
        return EXPORT2._training(
            receipt, candidate, result_path=result_path, lab=lab,
            second=True, allow_fixture=False)


def validated_targets(design_path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """원 continuation/V9/V2 검증 뒤 v1의 고정 media 선택 조건을 재사용한다."""
    first_path = V1.bound(config.get("first_result"), "first raw result")
    second_path = V1.bound(config.get("second_result"), "second raw result")
    first = read(first_path)
    require(first.get("schema") == CONT.FIRST_CONTINUATION_RESULT_SCHEMA
            and first.get("status") == "complete", "FIRST continuation result")
    EXPORT2._continuation(first_path, first, lab=Path(str(config["lab"])),
                          verify=EXPORT2._default_verifiers())
    v9_protocol = V1.bound(config.get("v9_protocol"), "V9 protocol")
    v9_plan = V1.bound(config.get("v9_private_plan"), "V9 private plan")
    v9 = V9.verify_protocol(v9_protocol, anchor=str(config.get("v9_anchor", "")), private_plan=v9_plan)
    second = read(second_path)
    require(second.get("schema") == EXPORT1.PARENT.SECOND_RESULT_SCHEMA
            and second.get("status") == "complete", "V9 SECOND result")
    _export_binding(Path(str(config["recovery_export"])).resolve(strict=True), first_path, second_path)
    selected_training = _verify_selected_second(second_path, v9_plan=v9,
                                                lab=Path(str(config["lab"])).resolve())
    require(selected_training.get("fixed_conditions") == 26
            and selected_training.get("fresh_initialization") is True,
            "selected SECOND training/fixed26 검증")
    clone = _isolated_v1()
    targets = clone.validated_targets(design_path, config)
    require(sum(row.get("status") == "registered_recovered" for row in targets) == 1
            and any(row.get("key") == "case-07:logs_only"
                    and row.get("registered_seed") == 5557
                    and row.get("final_epoch") == "second"
                    for row in targets)
            and any(row.get("key") == "case-03:logs_only" and row.get("status") == "HOLD"
                    for row in targets), "등록 media 조건은 case07 second recovered 단일 target이어야 한다")
    return cast(list[dict[str, Any]], targets)


@contextmanager
def target_scope(design_path: Path, config: Mapping[str, Any]) -> Iterator[list[dict[str, Any]]]:
    with _LOCK:
        targets = validated_targets(design_path, config)
        original = V1.validated_targets
        original_schema = V1.CONFIG_SCHEMA
        require(original is _ORIGINAL_TARGETS and original_schema == _ORIGINAL_CONFIG_SCHEMA,
                "v1 media target/config verifier가 이미 바뀌었다")

        def scoped(path: Path, value: Mapping[str, Any]) -> list[dict[str, Any]]:
            require(Path(path).resolve() == Path(design_path).resolve() and dict(value) == dict(config),
                    "다른 media config 요청")
            return deepcopy(targets)

        V1.__dict__["validated_targets"] = scoped
        V1.__dict__["CONFIG_SCHEMA"] = CONFIG_SCHEMA
        try:
            yield deepcopy(targets)
        finally:
            current = (V1.validated_targets, V1.CONFIG_SCHEMA)
            V1.__dict__["validated_targets"] = original
            V1.__dict__["CONFIG_SCHEMA"] = original_schema
            if current != (scoped, CONFIG_SCHEMA):
                raise RuntimeError("media target/config scope가 외부에서 바뀌었다")


def prepare_config(adapter_protocol: Path, adapter_anchor: str, design_path: Path,
                   first_result: Path, second_result: Path, recovery_export: Path,
                   input_capture_batch: Path, input_source_index: Path, lab: Path,
                   isaac_python: Path, v9_protocol: Path, v9_anchor: str,
                   v9_private_plan: Path, out: Path) -> dict[str, Any]:
    verify_protocol(adapter_protocol, adapter_anchor)
    require(not Path(out).exists() and Path(isaac_python).is_file() and Path(lab).is_dir(),
            "fresh config/Isaac launcher/lab")
    config = {
        "schema": CONFIG_SCHEMA, "adapter_protocol": ref(adapter_protocol),
        "adapter_anchor": adapter_anchor, "design": V1.B._ref(design_path),
        "first_result": V1.B._ref(first_result), "second_result": V1.B._ref(second_result),
        "recovery_export": str(Path(recovery_export).resolve(strict=True)),
        "input_capture_batch": V1.B._ref(input_capture_batch),
        "input_source_index": V1.B._ref(input_source_index),
        "lab": str(Path(lab).resolve()), "isaac_python": str(Path(isaac_python).resolve()),
        "v9_protocol": V1.B._ref(v9_protocol), "v9_anchor": v9_anchor,
        "v9_private_plan": V1.B._ref(v9_private_plan),
    }
    targets = validated_targets(design_path, config)
    config["input_pairs"] = V1.validate_input_pairs(config, targets)
    V1.B._write_new(out, config)
    return config


def preflight(adapter_protocol: Path, adapter_anchor: str, design_path: Path,
              config_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    verify_protocol(adapter_protocol, adapter_anchor)
    config = read(config_path)
    require(config.get("schema") == CONFIG_SCHEMA and config.get("adapter_protocol") == ref(adapter_protocol)
            and config.get("adapter_anchor") == adapter_anchor, "adapter config/protocol")
    targets = validated_targets(design_path, config)
    pairs = V1.validate_input_pairs(config, targets)
    require(config.get("input_pairs") == pairs, "adapter input pairs")
    design = V1.CAPTURE.validate_design(design_path)
    recovered = [row for row in targets if row["status"] == "registered_recovered"]
    require(len(recovered) == 1 and 1 + len(recovered) <= design["budget"]["sdk_cap"], "media SDK cap")
    return design, config, targets


def execute(adapter_protocol: Path, adapter_anchor: str, design_path: Path,
            config_path: Path, source_anchor: str, execution_anchor: str,
            out: Path) -> dict[str, Any]:
    _design, config, targets = preflight(adapter_protocol, adapter_anchor, design_path, config_path)
    with target_scope(design_path, config) as scoped:
        require(scoped == targets, "preflight/scope targets")
        return V1.execute(design_path, config_path, source_anchor, execution_anchor, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("adapter-protocol", "design", "first-result", "second-result",
                 "recovery-export", "input-capture-batch", "input-source-index",
                 "lab", "isaac-python", "v9-protocol", "v9-private-plan", "out"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--adapter-anchor", required=True)
    prep.add_argument("--v9-anchor", required=True)
    for command in ("preflight", "execute"):
        child = sub.add_parser(command)
        child.add_argument("--adapter-protocol", type=Path, required=True)
        child.add_argument("--adapter-anchor", required=True)
        child.add_argument("--design", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
        if command == "execute":
            child.add_argument("--source-anchor", required=True)
            child.add_argument("--execution-anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_config(
            args.adapter_protocol, args.adapter_anchor, args.design,
            args.first_result, args.second_result, args.recovery_export,
            args.input_capture_batch, args.input_source_index, args.lab,
            args.isaac_python, args.v9_protocol, args.v9_anchor,
            args.v9_private_plan, args.out)
    elif args.command == "preflight":
        _design, _config, targets = preflight(
            args.adapter_protocol, args.adapter_anchor, args.design, args.config)
        V1.B._write_new(args.out, {"status": "complete", "gpu": 0, "api": 0, "targets": targets})
    else:
        execute(args.adapter_protocol, args.adapter_anchor, args.design, args.config,
                args.source_anchor, args.execution_anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
