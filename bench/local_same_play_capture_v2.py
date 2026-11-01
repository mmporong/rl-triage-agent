"""로컬 same-Play v1 실행기를 겹치지 않는 sidecar sibling layout에 결합한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import local_same_play_capture_v1 as V1

SCHEMA = "local_same_play_capture_protocol_v2"
RESULT_SCHEMA = "local_same_play_capture_execution_v2"
SOURCE_FILES = ("bench/local_same_play_capture_v2.py", "tests/test_local_same_play_capture_v2.py")
LAYOUT = {"native_and_job": "fresh_stage", "sidecar": "unique_sibling_outside_stage",
          "sidecar_suffix": ".sidecar-v2", "v1_check_destination_unchanged": True}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _source_map(base: Mapping[str, Any]) -> dict[str, str]:
    inherited = base.get("source_files_lf")
    _require(isinstance(inherited, dict), "base v1 source closure")
    result = dict(cast(dict[str, str], inherited))
    result.update({name: V1._sha(ROOT / name, lf=True) for name in SOURCE_FILES})
    return dict(sorted(result.items()))


def prepare_protocol(base_protocol_path: Path, inspection_path: Path, lab: Path, out: Path,
                     *, source_anchor: str | None = None, repo: Path = ROOT) -> dict[str, Any]:
    base = V1.validate_protocol(base_protocol_path, lab, inspection_path, verify_anchor=False)
    if source_anchor is not None:
        _require(len(source_anchor) == 40
                 and all(char in "0123456789abcdef" for char in source_anchor), "source anchor")
    relative = Path(base_protocol_path).resolve().relative_to(repo.resolve()).as_posix()
    value = {
        "schema": SCHEMA,
        "status": "registered_before_local_gpu" if source_anchor else "candidate_not_registered",
        "source_anchor": source_anchor,
        "base_v1_protocol": {"path": relative, "sha256": V1._sha(base_protocol_path)},
        "source_files_lf": _source_map(base),
        "runtime_pins": dict(base["runtime_pins"]),
        "training_runtime_pins": dict(base["training_runtime_pins"]),
        "source_batch": dict(base["source_batch"]),
        "checkpoint": dict(base["checkpoint"]),
        "platform": dict(base["platform"]),
        "gpu": dict(base["gpu"]),
        "play": dict(base["play"]),
        "shutdown": dict(base["shutdown"]),
        "layout": dict(LAYOUT),
    }
    V1._write_new(out, value)
    return value


def validate_protocol(path: Path, inspection_path: Path, lab: Path, *,
                      execution_anchor: str | None = None, verify_anchor: bool = True,
                      repo: Path = ROOT) -> dict[str, Any]:
    value = V1._read(path)
    expected = {"schema", "status", "source_anchor", "base_v1_protocol", "source_files_lf",
                "runtime_pins", "training_runtime_pins", "source_batch", "checkpoint",
                "platform", "gpu", "play", "shutdown", "layout"}
    _require(set(value) == expected and value.get("schema") == SCHEMA
             and value.get("status") == "registered_before_local_gpu", "v2 protocol schema/status")
    _require(value.get("layout") == LAYOUT, "v2 sibling layout")
    base_ref = value.get("base_v1_protocol")
    _require(isinstance(base_ref, Mapping) and set(base_ref) == {"path", "sha256"}, "base v1 ref")
    checked_base_ref = cast(Mapping[str, Any], base_ref)
    base_path = (repo / cast(str, checked_base_ref["path"])).resolve()
    _require(base_path.is_relative_to(repo.resolve()) and V1._sha(base_path) == checked_base_ref.get("sha256"),
             "base v1 protocol ref")
    base = V1.validate_protocol(
        base_path, lab, inspection_path, repo=repo,
        execution_anchor=execution_anchor, verify_anchor=verify_anchor)
    _require(value["source_files_lf"] == _source_map(base)
             and value["runtime_pins"] == base["runtime_pins"]
             and value["training_runtime_pins"] == base["training_runtime_pins"]
             and value["source_batch"] == base["source_batch"]
             and value["checkpoint"] == base["checkpoint"]
             and value["platform"] == base["platform"]
             and value["gpu"] == base["gpu"]
             and value["play"] == base["play"]
             and value["shutdown"] == base["shutdown"], "v2/base v1 binding")
    anchor = value.get("source_anchor")
    _require(isinstance(anchor, str) and len(anchor) == 40, "v2 source anchor")
    if verify_anchor:
        _require(isinstance(execution_anchor, str), "v2 execution anchor")
        for relative, expected_sha in value["source_files_lf"].items():
            _require(V1._sha(repo / relative, lf=True) == expected_sha, f"v2 source changed: {relative}")
        _verify_v2_source_blobs(repo, cast(str, anchor), value["source_files_lf"])
        V1._verify_protocol_blob(path, repo, cast(str, anchor), cast(str, execution_anchor))
    return value


def sibling_sidecar(stage: Path) -> Path:
    stage = Path(stage).resolve()
    return stage.parent / f"{stage.name}{LAYOUT['sidecar_suffix']}"


def _verify_v2_source_blobs(repo: Path, source_anchor: str,
                            source_map: Mapping[str, str]) -> None:
    """v1 base anchor가 다루지 않는 v2 source/test를 새 anchor blob에 결합한다."""
    for relative in SOURCE_FILES:
        expected = source_map.get(relative)
        _require(isinstance(expected, str) and V1._sha(repo / relative, lf=True) == expected,
                 f"v2 current source changed: {relative}")
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{source_anchor}:{relative}"],
            check=True, capture_output=True).stdout
        actual = hashlib.sha256(shown.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        _require(actual == expected, f"v2 source anchor blob changed: {relative}")


def build_job(protocol_path: Path, inspection_path: Path, lab: Path, stage: Path, *,
              mode: str, execution_anchor: str, verify_anchor: bool = True) -> dict[str, Any]:
    _require(mode in {"smoke", "full"}, "mode")
    protocol = validate_protocol(protocol_path, inspection_path, lab,
                                 execution_anchor=execution_anchor,
                                 verify_anchor=verify_anchor)
    _require(not stage.exists(), "stage는 fresh 경로여야 한다")
    sidecar = sibling_sidecar(stage)
    _require(not sidecar.exists() and not sidecar.is_symlink(), "sidecar sibling은 fresh 경로여야 한다")
    inspection = V1._read(inspection_path)
    stage.mkdir(parents=True)
    frames = 1 if mode == "smoke" else V1.FULL_FRAMES
    job = {
        "schema": V1.JOB_SCHEMA, "mode": mode, "frames": frames, "fps": V1.FPS,
        "protocol": V1._ref(protocol_path), "source_anchor": protocol["source_anchor"],
        "execution_anchor": execution_anchor, "checkpoint_path": inspection["checkpoint"]["path"],
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "checkpoint_logical_id": protocol["checkpoint"]["logical_id"],
        "source": {"commit": protocol["source_anchor"],
                   "files_lf": dict(protocol["source_files_lf"])},
        "output_directory": str((stage / "capture").resolve()),
        "sidecar_directory": str(sidecar), "job_path": str((stage / "job.json").resolve()),
        "source_id": f"local-{protocol['checkpoint']['logical_id']}-{mode}-v2",
    }
    V1._write_new(stage / "job.json", job)
    return {**job, "job_sha256": V1._sha(stage / "job.json")}


def execute(protocol_path: Path, inspection_path: Path, execution_anchor: str, lab: Path,
            stage: Path, isaac_python: Path, *, mode: str,
            runner: Callable[[Path, list[str], Path, float], Mapping[str, Any]] | None = None,
            lease_reader: Callable[[Path | None], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    _require(Path(isaac_python).is_file(), "Isaac Python launcher")
    run_owned = runner or importlib.import_module("run_runtime_readback_smoke").run_owned
    read_lease = lease_reader or importlib.import_module("lineage_lease_context_v2").lease_context
    before = V1._lease_identity(dict(read_lease(stage.parent / f"{stage.name}-lease-before")))
    job = build_job(protocol_path, inspection_path, lab, stage, mode=mode,
                    execution_anchor=execution_anchor)
    owned = stage / "owned"
    owned.mkdir()
    command = [str(isaac_python), str(Path(__file__).resolve()), "sdk-child", str(stage / "job.json"),
               *V1.KIT_ARGS]
    execution = dict(run_owned(owned, command, ROOT, 600.0 if mode == "smoke" else 2400.0))
    failure_path = Path(job["output_directory"]) / "failure.json"
    if failure_path.is_file():
        failure = V1._read(failure_path)
        raise RuntimeError(
            f"owned child failure: {failure.get('error_type')}: {failure.get('error_message')}")
    V1.verify_owned(execution)
    after = V1._lease_identity(dict(read_lease(stage / "lease-after")))
    _require(after == before, "실행 중 공용 GPU lease가 바뀌었다")
    if mode == "full":
        child = V1.verify_result(stage, job, execution)
    else:
        sidecar = importlib.import_module("record_agentic_state_sidecar_v2")
        child = sidecar.verify_shutdown_evidence(
            Path(job["output_directory"]), checkpoint_sha256=job["checkpoint_sha256"],
            source_id=job["source_id"], frames=1)
    result = {"schema": RESULT_SCHEMA, "status": "technical_complete", "mode": mode,
              "protocol": V1._ref(protocol_path), "source_anchor": job["source_anchor"],
              "execution_anchor": execution_anchor, "checkpoint_sha256": job["checkpoint_sha256"],
              "layout": {"stage": str(stage.resolve()), "sidecar": job["sidecar_directory"]},
              "lease": {"owner": before[0], "lease_id": before[1]}, "owned": execution,
              "child": child, "capture_approved": False, "model_ready": False,
              "used_for_decision": False}
    V1._write_new(stage / "execution-result-v2.json", result)
    return result


def sdk_child(job_path: Path) -> int:
    return V1.sdk_child(job_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--base-protocol", required=True, type=Path)
    prepare.add_argument("--inspection", required=True, type=Path)
    prepare.add_argument("--lab", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--source-anchor")
    run = sub.add_parser("execute")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--inspection", required=True, type=Path)
    run.add_argument("--anchor", required=True)
    run.add_argument("--lab", required=True, type=Path)
    run.add_argument("--stage", required=True, type=Path)
    run.add_argument("--isaac-python", required=True, type=Path)
    run.add_argument("--mode", required=True, choices=("smoke", "full"))
    child = sub.add_parser("sdk-child")
    child.add_argument("job", type=Path)
    args, extras = parser.parse_known_args(argv)
    _require(extras == (V1.KIT_ARGS if args.command == "sdk-child" else []), "허용되지 않은 CLI 인자")
    if args.command == "prepare-protocol":
        value = prepare_protocol(args.base_protocol, args.inspection, args.lab, args.out,
                                 source_anchor=args.source_anchor)
    elif args.command == "execute":
        value = execute(args.protocol, args.inspection, args.anchor, args.lab, args.stage,
                        args.isaac_python, mode=args.mode)
    else:
        return sdk_child(args.job)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
