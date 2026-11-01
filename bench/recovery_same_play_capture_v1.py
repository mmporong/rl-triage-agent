# pyright: reportMissingImports=false
"""R4의 검증된 최종 회복 checkpoint를 기존 same-Play recorder로 녹화한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import local_same_play_capture_v1 as V1
import local_same_play_capture_v2 as V2
import local_same_play_capture_v3 as V3

SCHEMA = "five_type_recovery_media_conditional_v1"
RESULT_SCHEMA = "recovery_same_play_capture_execution_v1"
SOURCE_FILES = (
    "bench/recovery_same_play_capture_v1.py",
    "bench/run_recovery_same_play_capture_v1.py",
    "tests/test_recovery_same_play_capture_v1.py",
    "tests/test_run_recovery_same_play_capture_v1.py",
)
TARGET_KEYS = ("case-03:logs_only", "case-07:logs_only")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    return V1._read(Path(path))


def source_files() -> tuple[str, ...]:
    return tuple(sorted(set(V1.source_files()) | set(V2.SOURCE_FILES) | set(V3.SOURCE_FILES) |
                        set(SOURCE_FILES)))


def _source_map() -> dict[str, str]:
    return {name: V1._sha(ROOT / name, lf=True) for name in source_files()}


def _source_digest() -> str:
    raw = json.dumps(_source_map(), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_source_anchor(design_path: Path, source_anchor: str, execution_anchor: str) -> None:
    require(all(isinstance(value, str) and len(value) == 40
                and set(value) <= set("0123456789abcdef")
                for value in (source_anchor, execution_anchor)), "source/execution anchor")
    ancestry = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                               source_anchor, execution_anchor], check=False)
    require(ancestry.returncode == 0, "source anchor ancestry")
    for relative, expected in _source_map().items():
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{source_anchor}:{relative}"],
                              check=True, capture_output=True).stdout
        normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(normalized).hexdigest() == expected,
                f"source anchor blob 변경: {relative}")
    V1._verify_protocol_blob(design_path, ROOT, source_anchor, execution_anchor)


def validate_design(path: Path, *, verify_sources: bool = True) -> dict[str, Any]:
    value = read(path)
    require(value.get("schema") == SCHEMA and value.get("status") == "preregistered_conditional",
            "recovery media conditional schema/status")
    require(value.get("target_order") == list(TARGET_KEYS), "고정 target 순서")
    targets = value.get("targets")
    require(isinstance(targets, dict) and set(targets) == set(TARGET_KEYS), "exact target2")
    expected = {
        "case-03:logs_only": {"case_id": "case-03", "policy": "logs_only", "source_arm": "F1",
                              "source_pair": {"reference": "source-05", "case": "source-06"},
                              "first_seed": 4553, "second_seed": 5553},
        "case-07:logs_only": {"case_id": "case-07", "policy": "logs_only", "source_arm": "COMPOUND",
                              "source_pair": {"reference": "source-13", "case": "source-14"},
                              "first_seed": 4557, "second_seed": 5557},
    }
    require(targets == expected, "target/source pair/seed 등록")
    require(value.get("selection") == {
        "required_final_outcome": "recovered", "first_precedes_second": True,
        "hold_without_replacement": True, "replacement_target": False}, "conditional selection")
    require(value.get("budget") == {"api_cap": 0, "sdk_cap": 3, "smoke": 1,
                                     "full_per_recovered_target": 1, "wall_cap_s": 9000,
                                     "automatic_retries": 0}, "capture budget")
    require(value.get("play") == {
        "task": V1.TASK, "num_envs": 1, "frames": 625, "fps": 50,
        "duration_s": 12.5, "width": 640, "height": 480, "controller_seed": 42,
        "command_mps": 1.0, "actor_inference_only": True, "training": False,
        "camera_profile": "same_play_fixed_v1", "render_sync": True,
        "sidecar": True, "same_pts": True, "shutdown": "vision_shutdown_policy_v1"},
        "same-Play 설정")
    require(value.get("runtime") == {"fault_injection": False, "actuator_change": False,
                                      "owned_process": True, "exclusive_gpu_lease": True},
            "recovery runtime 불변")
    require(value.get("authority") == {"gpu": False, "api": False, "capture_approved": False,
                                        "model_ready": False, "used_for_decision": False},
            "기술 완료와 운영 권한 분리")
    if verify_sources:
        require(value.get("source_closure") == {"count": len(source_files()),
                                                 "sha256_lf_map_digest": _source_digest()},
                "capture source closure 변경")
    return value


def build_job(design_path: Path, target: Mapping[str, Any], stage: Path, *, mode: str,
              source_anchor: str, execution_anchor: str) -> dict[str, Any]:
    design = validate_design(design_path)
    validate_source_anchor(design_path, source_anchor, execution_anchor)
    key = target.get("key")
    require(key in TARGET_KEYS and target.get("status") == "registered_recovered", "recovered target")
    registered = design["targets"][key]
    require(target.get("case_id") == registered["case_id"] and target.get("policy") == "logs_only"
            and target.get("source_arm") == registered["source_arm"], "target identity")
    require(target.get("final_epoch") in {"first", "second"}, "final epoch")
    expected_seed = registered[f"{target['final_epoch']}_seed"]
    require(target.get("registered_seed") == expected_seed, "final seed")
    checkpoint = Path(str(target.get("checkpoint_path", ""))).resolve(strict=True)
    checkpoint_sha = target.get("checkpoint_sha256")
    require(checkpoint.is_file() and not checkpoint.is_symlink()
            and V1._sha(checkpoint) == checkpoint_sha
            and checkpoint_sha != target.get("original_checkpoint_sha256"), "final checkpoint raw/SHA")
    require(mode in {"smoke", "full"} and not stage.exists(), "mode/fresh stage")
    sidecar = V2.sibling_sidecar(stage)
    require(not sidecar.exists() and not sidecar.is_symlink(), "fresh sidecar")
    stage.mkdir(parents=True)
    frames = 1 if mode == "smoke" else V1.FULL_FRAMES
    job = {
        "schema": V1.JOB_SCHEMA, "mode": mode, "frames": frames, "fps": V1.FPS,
        "protocol": V1._ref(design_path), "source_anchor": source_anchor,
        "execution_anchor": execution_anchor, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_logical_id": f"r4-{key.replace(':', '-')}-{target['final_epoch']}",
        "source": {"commit": source_anchor, "files_lf": _source_map()},
        "output_directory": str((stage / "capture").resolve()),
        "sidecar_directory": str(sidecar), "job_path": str((stage / "job.json").resolve()),
        "source_id": f"recovery-{key.replace(':', '-')}-{mode}-v1",
        "recovery_binding": dict(target), "fault_injection": False, "actuator_change": False,
    }
    V1._write_new(stage / "job.json", job)
    return {**job, "job_sha256": V1._sha(stage / "job.json")}


def sdk_child(job_path: Path) -> int:
    job = read(job_path)
    require(job.get("fault_injection") is False and job.get("actuator_change") is False,
            "recovery playback/actuator 변경 금지")
    protocol_ref = job.get("protocol")
    require(isinstance(protocol_ref, Mapping) and set(protocol_ref) == {"path", "sha256"},
            "child protocol ref")
    protocol_path = Path(str(protocol_ref["path"])).resolve(strict=True)
    require(V1._sha(protocol_path) == protocol_ref["sha256"], "child protocol SHA")
    validate_design(protocol_path)
    require(job.get("source") == {"commit": job.get("source_anchor"),
                                  "files_lf": _source_map()}, "child source closure")
    return int(V3.sdk_child(job_path))


def execute(design_path: Path, target: Mapping[str, Any], source_anchor: str,
            execution_anchor: str, stage: Path, isaac_python: Path, *, mode: str,
            runner: Callable[[Path, list[str], Path, float], Mapping[str, Any]] | None = None,
            lease_reader: Callable[[Path | None], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    require(Path(isaac_python).is_file(), "Isaac Python launcher")
    run_owned = runner or importlib.import_module("run_runtime_readback_smoke").run_owned
    read_lease = lease_reader or importlib.import_module("lineage_lease_context_v2").lease_context
    before = V1._lease_identity(dict(read_lease(stage.parent / f"{stage.name}-lease-before")))
    job = build_job(design_path, target, stage, mode=mode, source_anchor=source_anchor,
                    execution_anchor=execution_anchor)
    owned = stage / "owned"
    owned.mkdir()
    command = [str(Path(isaac_python)), str(Path(__file__).resolve()), "sdk-child",
               str(stage / "job.json"), *V1.KIT_ARGS]
    execution = dict(run_owned(owned, command, ROOT, 600.0 if mode == "smoke" else 2400.0))
    V1.verify_owned(execution)
    after = V1._lease_identity(dict(read_lease(stage / "lease-after")))
    require(after == before, "실행 중 공용 GPU lease 변경")
    if mode == "full":
        child = V3.verify_result(stage, job, execution)
    else:
        sidecar = importlib.import_module("record_agentic_state_sidecar_v2")
        child = sidecar.verify_shutdown_evidence(Path(job["output_directory"]),
            checkpoint_sha256=job["checkpoint_sha256"], source_id=job["source_id"], frames=1)
    result = {"schema": RESULT_SCHEMA, "status": "technical_complete", "mode": mode,
              "target": target["key"], "checkpoint_sha256": job["checkpoint_sha256"],
              "frames": job["frames"], "fps": 50, "owned": execution, "child": child,
              "lease": {"owner": before[0], "lease_id": before[1]},
              "fault_injection": False, "actuator_change": False,
              "capture_approved": False, "model_ready": False, "used_for_decision": False}
    V1._write_new(stage / "execution-result-recovery-v1.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("sdk-child",))
    parser.add_argument("job", type=Path)
    args, extras = parser.parse_known_args(argv)
    require(extras == V1.KIT_ARGS, "허용되지 않은 CLI 인자")
    return sdk_child(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
