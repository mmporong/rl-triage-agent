"""로컬 same-Play v2 layout을 production record worker 계약에 결합한다."""
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
import local_same_play_capture_v2 as V2

SCHEMA = "local_same_play_capture_protocol_v3"
RESULT_SCHEMA = "local_same_play_capture_execution_v3"
SOURCE_FILES = ("bench/local_same_play_capture_v3.py", "tests/test_local_same_play_capture_v3.py")
WORKER_KEYS = ("job_path", "job_sha256", "checkpoint_path", "checkpoint_sha256",
               "output_directory", "sidecar_directory", "source_id", "frames", "job")
JOB_KEYS = ("mode", "job_id", "profile", "protocol", "checkpoint", "source")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _record() -> Any:
    return importlib.import_module("record_vision_evidence_v1")


def _source_map(base: Mapping[str, Any]) -> dict[str, str]:
    inherited = base.get("source_files_lf")
    _require(isinstance(inherited, dict), "base v2 source closure")
    result = dict(cast(dict[str, str], inherited))
    result.update({name: V1._sha(ROOT / name, lf=True) for name in SOURCE_FILES})
    return dict(sorted(result.items()))


def prepare_protocol(base_protocol_path: Path, inspection_path: Path, lab: Path, out: Path, *,
                     source_anchor: str | None = None, repo: Path = ROOT) -> dict[str, Any]:
    base = V2.validate_protocol(base_protocol_path, inspection_path, lab, verify_anchor=False,
                                repo=repo)
    if source_anchor is not None:
        _require(len(source_anchor) == 40 and all(c in "0123456789abcdef" for c in source_anchor),
                 "source anchor")
    relative = Path(base_protocol_path).resolve().relative_to(repo.resolve()).as_posix()
    record = _record()
    value = {
        "schema": SCHEMA,
        "status": "registered_before_local_gpu" if source_anchor else "candidate_not_registered",
        "source_anchor": source_anchor,
        "base_v2_protocol": {"path": relative, "sha256": V1._sha(base_protocol_path)},
        "source_files_lf": _source_map(base),
        "runtime_pins": dict(base["runtime_pins"]),
        "training_runtime_pins": dict(base["training_runtime_pins"]),
        "source_batch": dict(base["source_batch"]), "checkpoint": dict(base["checkpoint"]),
        "platform": dict(base["platform"]), "gpu": dict(base["gpu"]),
        "play": dict(base["play"]), "shutdown": dict(base["shutdown"]),
        "layout": dict(base["layout"]),
        "worker_contract": {
            "mode": record.RECORD_MODE, "profile": record.fixed_profile(),
            "profile_sha256": record.digest(record.fixed_profile()),
            "validated_keys": list(WORKER_KEYS), "job_keys": list(JOB_KEYS),
            "resource_scope": "local_common_gpu_lease",
            "cloud_resource_claimed": False,
        },
    }
    V1._write_new(out, value)
    return value


def validate_protocol(path: Path, inspection_path: Path, lab: Path, *,
                      execution_anchor: str | None = None, verify_anchor: bool = True,
                      repo: Path = ROOT) -> dict[str, Any]:
    value = V1._read(path)
    expected = {"schema", "status", "source_anchor", "base_v2_protocol", "source_files_lf",
                "runtime_pins", "training_runtime_pins", "source_batch", "checkpoint",
                "platform", "gpu", "play", "shutdown", "layout", "worker_contract"}
    _require(set(value) == expected and value.get("schema") == SCHEMA
             and value.get("status") == "registered_before_local_gpu", "v3 protocol schema/status")
    ref = value.get("base_v2_protocol")
    _require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}, "base v2 ref")
    checked_ref = cast(Mapping[str, Any], ref)
    base_path = (repo / cast(str, checked_ref["path"])).resolve()
    _require(base_path.is_relative_to(repo.resolve())
             and V1._sha(base_path) == checked_ref.get("sha256"),
             "base v2 protocol ref")
    base = V2.validate_protocol(base_path, inspection_path, lab, execution_anchor=execution_anchor,
                                verify_anchor=verify_anchor, repo=repo)
    record = _record()
    contract = {"mode": record.RECORD_MODE, "profile": record.fixed_profile(),
                "profile_sha256": record.digest(record.fixed_profile()),
                "validated_keys": list(WORKER_KEYS), "job_keys": list(JOB_KEYS),
                "resource_scope": "local_common_gpu_lease", "cloud_resource_claimed": False}
    for key in ("runtime_pins", "training_runtime_pins", "source_batch", "checkpoint",
                "platform", "gpu", "play", "shutdown", "layout"):
        _require(value[key] == base[key], f"v3/base v2 {key} binding")
    _require(value["source_files_lf"] == _source_map(base)
             and value["worker_contract"] == contract, "v3 worker/source contract")
    anchor = value.get("source_anchor")
    _require(isinstance(anchor, str) and len(anchor) == 40, "v3 source anchor")
    if verify_anchor:
        _require(isinstance(execution_anchor, str), "v3 execution anchor")
        _verify_source_blobs(repo, cast(str, anchor), value["source_files_lf"])
        V1._verify_protocol_blob(path, repo, cast(str, anchor), cast(str, execution_anchor))
    return value


def _verify_source_blobs(repo: Path, anchor: str, source_map: Mapping[str, str]) -> None:
    for relative in SOURCE_FILES:
        expected = source_map.get(relative)
        _require(isinstance(expected, str) and V1._sha(repo / relative, lf=True) == expected,
                 f"v3 current source changed: {relative}")
        shown = subprocess.run(["git", "-C", str(repo), "show", f"{anchor}:{relative}"],
                               check=True, capture_output=True).stdout
        actual = hashlib.sha256(shown.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        _require(actual == expected, f"v3 source anchor blob changed: {relative}")


def build_job(protocol_path: Path, inspection_path: Path, lab: Path, stage: Path, *, mode: str,
              execution_anchor: str, verify_anchor: bool = True) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path, inspection_path, lab, execution_anchor=execution_anchor,
                                 verify_anchor=verify_anchor)
    _require(mode in {"smoke", "full"} and not stage.exists(), "mode/fresh stage")
    sidecar = V2.sibling_sidecar(stage)
    _require(not sidecar.exists() and not sidecar.is_symlink(), "sidecar sibling은 fresh여야 한다")
    inspection = V1._read(inspection_path)
    stage.mkdir(parents=True)
    frames = 1 if mode == "smoke" else V1.FULL_FRAMES
    job = {"schema": V1.JOB_SCHEMA, "mode": mode, "frames": frames, "fps": V1.FPS,
           "protocol": V1._ref(protocol_path), "source_anchor": protocol["source_anchor"],
           "execution_anchor": execution_anchor, "checkpoint_path": inspection["checkpoint"]["path"],
           "checkpoint_sha256": protocol["checkpoint"]["sha256"],
           "checkpoint_logical_id": protocol["checkpoint"]["logical_id"],
           "source": {"commit": protocol["source_anchor"],
                      "files_lf": dict(protocol["source_files_lf"])},
           "output_directory": str((stage / "capture").resolve()),
           "sidecar_directory": str(sidecar), "job_path": str((stage / "job.json").resolve()),
           "source_id": f"local-{protocol['checkpoint']['logical_id']}-{mode}-v3"}
    V1._write_new(stage / "job.json", job)
    return {**job, "job_sha256": V1._sha(stage / "job.json")}


def _validated_job(job_path: Path) -> dict[str, Any]:
    job = V1._read(job_path)
    _require(job.get("schema") == V1.JOB_SCHEMA
             and V1._sha(Path(job["checkpoint_path"])) == job.get("checkpoint_sha256"),
             "child job/checkpoint")
    record = _record()
    nested = {"mode": record.RECORD_MODE, "job_id": job["source_id"],
              "profile": record.fixed_profile(), "protocol": job["protocol"],
              "checkpoint": {"path": job["checkpoint_path"], "sha256": job["checkpoint_sha256"],
                             "logical_id": job["checkpoint_logical_id"]},
              "source": job["source"]}
    value = {**job, "job_path": Path(job_path).resolve(), "job_sha256": V1._sha(job_path),
             "checkpoint_path": Path(job["checkpoint_path"]),
             "output_directory": Path(job["output_directory"]),
             "sidecar_directory": Path(job["sidecar_directory"]), "job": nested}
    _require(all(key in value for key in WORKER_KEYS) and set(nested) == set(JOB_KEYS),
             "production worker key contract")
    return value


def sdk_child(job_path: Path) -> int:
    validated = _validated_job(job_path)
    if validated["mode"] != "full":
        sidecar2 = importlib.import_module("record_agentic_state_sidecar_v2")
        return int(sidecar2.run_state_dry_run(
            checkpoint_path=Path(validated["checkpoint_path"]),
            checkpoint_sha256=validated["checkpoint_sha256"], source_id=validated["source_id"],
            out_dir=Path(validated["output_directory"]), frames=1))
    modules = {name: importlib.import_module(module) for name, module in {
        "worker": "vision_pull_record_worker_v3", "capture": "vision_pull_record_capture_v2",
        "render": "run_agentic_play_record_v4", "sidecar": "record_agentic_state_sidecar_v1",
        "shutdown": "vision_shutdown_policy_v1"}.items()}
    stage = Path(job_path).resolve().parent
    app = V1._full_app(validated, stage, modules["shutdown"])
    report_path = Path(validated["output_directory"]) / modules["worker"].REPORT_NAME

    def run_worker() -> Mapping[str, Any]:
        return V1.compose_full_worker(
            validated, app, worker=modules["worker"], capture=modules["capture"],
            render=modules["render"], sidecar=modules["sidecar"],
            seal_report=lambda value: V1._write_new(report_path, value),
            reader_factory=modules["sidecar"]._default_reader_factory)

    exit_code, _ = V1.run_full_terminal(
        validated, app, run_worker,
        before_close=lambda value: V1._seal_preclose_proof(validated, value, modules))
    return exit_code


def verify_result(stage: Path, job: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    V1.verify_owned(execution)
    validated = _validated_job(stage / "job.json")
    modules = {name: importlib.import_module(module) for name, module in {
        "worker": "vision_pull_record_worker_v3", "sidecar": "record_agentic_state_sidecar_v1",
        "render": "run_agentic_play_record_v4", "shutdown": "vision_shutdown_policy_v1"}.items()}
    proof = V1._verify_preclose_proof(validated, modules)
    modules["shutdown"].verify(stage / "job.json", validated,
                               Path(validated["output_directory"]) / "result.json")
    entered = V1._read(stage / "local-app-close-entered.json")
    _require(entered == {"schema": "local_same_play_app_close_v1", "status": "entered",
                         "job_sha256": validated["job_sha256"],
                         "checkpoint_sha256": validated["checkpoint_sha256"]},
             "app close entered proof")
    returned = stage / "local-app-close-returned.json"
    _require(not (stage / "local-app-close-error.json").exists(), "app close error proof")
    if returned.exists():
        raw_returned = V1._read(returned)
        _require(raw_returned.get("schema") == "local_same_play_app_close_v1"
                 and raw_returned.get("status") == "returned"
                 and raw_returned.get("entered_sha256")
                 == V1._sha(stage / "local-app-close-entered.json"), "app close returned proof")
    refs = {**dict(proof["proofs"]),
            "shutdown": V1._ref(stage / modules["shutdown"].SIDECAR),
            "app_close_entered": V1._ref(stage / "local-app-close-entered.json")}
    if returned.exists():
        refs["app_close_returned"] = V1._ref(returned)
    final = {"schema": V1.RESULT_SCHEMA, "status": "technical_complete", "mode": "full",
             "frames": V1.FULL_FRAMES, "fps": V1.FPS,
             "checkpoint_sha256": job["checkpoint_sha256"],
             "source_anchor": job["source_anchor"], "proofs": refs, "same_pts": True,
             "app_close_returned": returned.exists(), "capture_approved": False,
             "model_ready": False, "used_for_decision": False}
    V1._write_new(stage / "parent-finalized-result-v3.json", final)
    return final


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
    failure = Path(job["output_directory"]) / "failure.json"
    if failure.is_file():
        raw = V1._read(failure)
        raise RuntimeError(f"owned child failure: {raw.get('error_type')}: {raw.get('error_message')}")
    V1.verify_owned(execution)
    after = V1._lease_identity(dict(read_lease(stage / "lease-after")))
    _require(after == before, "실행 중 공용 GPU lease가 바뀌었다")
    if mode == "full":
        child = verify_result(stage, job, execution)
    else:
        sidecar = importlib.import_module("record_agentic_state_sidecar_v2")
        child = sidecar.verify_shutdown_evidence(Path(job["output_directory"]),
                                                  checkpoint_sha256=job["checkpoint_sha256"],
                                                  source_id=job["source_id"], frames=1)
    result = {"schema": RESULT_SCHEMA, "status": "technical_complete", "mode": mode,
              "protocol": V1._ref(protocol_path), "source_anchor": job["source_anchor"],
              "execution_anchor": execution_anchor, "checkpoint_sha256": job["checkpoint_sha256"],
              "layout": {"stage": str(stage.resolve()), "sidecar": job["sidecar_directory"]},
              "lease": {"owner": before[0], "lease_id": before[1]}, "owned": execution,
              "child": child, "capture_approved": False, "model_ready": False,
              "used_for_decision": False}
    V1._write_new(stage / "execution-result-v3.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-protocol")
    for name in ("base-protocol", "inspection", "lab", "out"):
        prep.add_argument(f"--{name}", required=True, type=Path)
    prep.add_argument("--source-anchor")
    run = sub.add_parser("execute")
    for name in ("protocol", "inspection", "lab", "stage", "isaac-python"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--anchor", required=True)
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
