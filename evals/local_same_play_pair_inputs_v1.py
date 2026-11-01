"""실제 학습 telemetry와 local same-Play 녹화를 승인 전 paired 입력으로 결합한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_state_facts_v1 as STATE
import local_same_play_capture_v1 as V1  # pyright: ignore[reportMissingImports]
import local_same_play_capture_v3 as V3  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]

SCHEMA = "local_same_play_pair_input_spec_v1"
PENDING_SCHEMA = "local_same_play_pair_inputs_pending_v1"
ROLES = ("case", "reference")
PROOF_KEYS = {"app_close_entered", "native_manifest", "raw_trace", "render_sync",
              "same_pts", "shutdown", "sidecar_manifest", "sidecar_records", "worker_report"}
SOURCE_FILES = ("evals/local_same_play_pair_inputs_v1.py",
                "tests/test_local_same_play_pair_inputs_v1.py",
                "evals/agentic_state_facts_v1.py",
                "tests/test_agentic_state_facts_v1.py")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")))
    require(isinstance(value, dict), f"JSON object required: {Path(path).name}")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_protocol(base_phase: Path, out: Path, *, source_anchor: str | None = None) -> dict[str, Any]:
    phase = read(base_phase)
    require(phase.get("schema") == "local_same_play_pair_capture_phase_v1"
            and phase.get("status") == "registered_before_new_three_condition_gpu",
            "base capture phase")
    if source_anchor is not None:
        require(len(source_anchor) == 40 and all(char in "0123456789abcdef" for char in source_anchor),
                "source anchor")
    value = {
        "schema": "local_same_play_pair_inputs_protocol_v1",
        "status": "registered_before_pair_input_use" if source_anchor else "candidate_not_registered",
        "source_anchor": source_anchor,
        "sources_lf": {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                                                   .replace(b"\r", b"\n")).hexdigest()
                       for name in SOURCE_FILES},
        "base_capture_phase": {"path": "bench/protocols/local_same_play_pairs_s2430_s2431_20261101_v1.json",
                               "sha256": sha(base_phase)},
        "split": "development", "training_fact_count": 48,
        "same_play_numeric_fact_count": 54, "planned_image_counts": [0, 6],
        "approval": {"main_only": True, "required_after_technical_acceptance": True,
                     "ready_before_approval": False},
        "claims": {"heldout": False, "recovery": False, "video_benefit": False,
                   "model_use_authorized": False},
    }
    write_new(out, value)
    return value


def _git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True)


def verify_protocol(path: Path, base_phase: Path, execution_anchor: str, *,
                    repo: Path = ROOT) -> dict[str, Any]:
    value = read(path)
    require(value.get("schema") == "local_same_play_pair_inputs_protocol_v1"
            and value.get("status") in {"candidate_not_registered", "registered_before_pair_input_use"}
            and value.get("split") == "development", "pair input protocol")
    require(value.get("base_capture_phase") == {
        "path": "bench/protocols/local_same_play_pairs_s2430_s2431_20261101_v1.json",
        "sha256": sha(base_phase)}, "base capture phase binding")
    current = {name: hashlib.sha256((repo / name).read_bytes().replace(b"\r\n", b"\n")
                                             .replace(b"\r", b"\n")).hexdigest()
               for name in SOURCE_FILES}
    require(value.get("sources_lf") == current, "pair input source closure")
    require(value.get("training_fact_count") == 48
            and value.get("same_play_numeric_fact_count") == 54
            and value.get("planned_image_counts") == [0, 6]
            and value.get("approval", {}).get("ready_before_approval") is False,
            "pair input registered shape")
    source_anchor = value.get("source_anchor")
    require(value.get("status") == "registered_before_pair_input_use"
            and isinstance(source_anchor, str) and len(source_anchor) == 40
            and all(char in "0123456789abcdef" for char in source_anchor),
            "registered source anchor")
    require(len(execution_anchor) == 40
            and all(char in "0123456789abcdef" for char in execution_anchor),
            "execution anchor")
    checked_source_anchor = cast(str, source_anchor)
    require(_git(repo, ["merge-base", "--is-ancestor", checked_source_anchor, execution_anchor],
                 check=False).returncode == 0
            and _git(repo, ["merge-base", "--is-ancestor", execution_anchor, "HEAD"],
                     check=False).returncode == 0,
            "protocol anchor ancestry")
    for relative, expected in current.items():
        blob = _git(repo, ["show", f"{checked_source_anchor}:{relative}"]).stdout
        require(hashlib.sha256(blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
                == expected, f"source anchor blob: {relative}")
    relative_protocol = Path(path).resolve().relative_to(repo.resolve()).as_posix()
    protocol_blob = _git(repo, ["show", f"{execution_anchor}:{relative_protocol}"]).stdout
    require(protocol_blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            == Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"),
            "execution anchor protocol blob")
    return value


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"unsafe source: {path.name}")
    return {"path": str(path), "sha256": sha(path)}


def _checked_ref(value: object, reason: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, Mapping) and set(value) >= {"path", "sha256"}, reason)
    ref = cast(Mapping[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and sha(path) == ref["sha256"], reason)
    return path, read(path)


def _stage(batch: Mapping[str, Any], logical_id: str) -> Mapping[str, Any]:
    rows = batch.get("stages")
    require(isinstance(rows, list), "source stages")
    found = [row for row in cast(list[object], rows)
             if isinstance(row, Mapping) and row.get("stage") == logical_id]
    require(len(found) == 1, f"source stage count: {logical_id}")
    return cast(Mapping[str, Any], found[0])


def _training(batch_path: Path, logical_id: str) -> dict[str, Any]:
    batch = read(batch_path)
    require(batch.get("schema") == "prospective_diagnosis_batch_result_v1"
            and batch.get("status") == "complete", "source batch completion")
    row = _stage(batch, logical_id)
    result_path, result = _checked_ref(row.get("result"), "stage result")
    training_path, training = _checked_ref(result.get("training"), "training result")
    manifest_path, manifest = _checked_ref(training.get("manifest"), "training manifest")
    collected_path, collected = _checked_ref(training.get("collected"), "collected input")
    metadata_ref = collected.get("metadata_ref")
    require(isinstance(metadata_ref, Mapping) and set(metadata_ref) == {"file", "sha256"},
            "collected metadata ref")
    checked_metadata_ref = cast(Mapping[str, Any], metadata_ref)
    metadata_path = (collected_path.parent / cast(str, checked_metadata_ref["file"])).resolve(strict=True)
    require(metadata_path.parent == collected_path.parent
            and sha(metadata_path) == checked_metadata_ref["sha256"],
            "collected metadata SHA")
    metadata = read(metadata_path)
    telemetry_path = Path(cast(str, metadata.get("telemetry"))).resolve(strict=True)
    telemetry = read(telemetry_path)
    private = metadata.get("private_training")
    binding = collected.get("source_binding")
    checkpoint = row.get("checkpoint")
    require(isinstance(private, Mapping) and isinstance(binding, Mapping)
            and isinstance(checkpoint, Mapping), "training metadata shape")
    checked_private = cast(Mapping[str, Any], private)
    checked_binding = cast(Mapping[str, Any], binding)
    checked_cp = cast(Mapping[str, Any], checkpoint)
    cp_path = Path(cast(str, checked_cp["path"])).resolve(strict=True)
    require(cp_path.is_file() and sha(cp_path) == checked_cp.get("sha256")
            and checked_cp.get("iteration") == 299
            and cp_path.stat().st_size == checked_cp.get("size_bytes"), "checkpoint 299")
    require(result.get("status") == "complete" and result.get("seed") == manifest.get("seed")
            and result.get("runtime_arm") == manifest.get("arm"), "result/manifest binding")
    require(manifest.get("schema") == "residual_training_manifest_v1"
            and checked_private.get("seed") == manifest.get("seed")
            and checked_private.get("arm") == manifest.get("arm")
            and checked_private.get("original_faults") == manifest.get("original_faults")
            and checked_private.get("applied_corrections") == manifest.get("applied_corrections") == [],
            "training lineage/no-correction")
    require(checked_binding.get("checkpoint_sha256") == checked_cp["sha256"]
            and checked_binding.get("telemetry_sha256") == sha(telemetry_path),
            "training source binding")
    summary = telemetry.get("summary")
    require(isinstance(summary, Mapping) and len(summary) == 24, "training telemetry24")
    facts = {}
    for metric, value in cast(Mapping[str, Any], summary).items():
        require(isinstance(value, Mapping) and type(value.get("mean_last_20pct")) in (int, float),
                f"telemetry metric: {metric}")
        facts[metric] = value["mean_last_20pct"]
    return {"logical_id": logical_id, "seed": manifest["seed"], "arm": manifest["arm"],
            "original_faults": manifest["original_faults"], "checkpoint": _ref(cp_path),
            "telemetry": _ref(telemetry_path), "facts": facts,
            "runtime_source_sha256_lf": dict(manifest["runtime_source_sha256_lf"]),
            "refs": {"stage_result": _ref(result_path), "training": _ref(training_path),
                     "manifest": _ref(manifest_path), "collected": _ref(collected_path),
                     "metadata": _ref(metadata_path)}}


def _proof_ref(value: object, key: str) -> Path:
    require(isinstance(value, Mapping) and set(value) >= {"path", "sha256"},
            f"capture proof: {key}")
    ref = cast(Mapping[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and sha(path) == ref["sha256"],
            f"capture proof: {key}")
    return path


def _capture(row: Mapping[str, Any], training: Mapping[str, Any], lab: Path,
             *, allow_fixture: bool) -> dict[str, Any]:
    stage = Path(cast(str, row.get("stage"))).resolve()
    require(stage.is_dir(), "capture stage missing")
    protocol = Path(cast(str, row.get("protocol"))).resolve(strict=True)
    inspection = Path(cast(str, row.get("inspection"))).resolve(strict=True)
    execution = read(stage / "execution-result-v3.json")
    checked_protocol: Mapping[str, Any] | None = None
    if not allow_fixture:
        capture_anchor = execution.get("execution_anchor")
        require(isinstance(capture_anchor, str), "capture execution anchor")
        checked_protocol = V3.validate_protocol(protocol, inspection, lab,
                                                execution_anchor=capture_anchor, verify_anchor=True)
        checked_protocol_map = cast(Mapping[str, Any], checked_protocol)
        require(checked_protocol_map["checkpoint"]["sha256"] == training["checkpoint"]["sha256"],
                "protocol/training checkpoint")
    job = read(stage / "job.json")
    final = read(stage / "parent-finalized-result-v3.json")
    require(job.get("checkpoint_sha256") == training["checkpoint"]["sha256"]
            and execution.get("checkpoint_sha256") == training["checkpoint"]["sha256"]
            and final.get("checkpoint_sha256") == training["checkpoint"]["sha256"],
            "capture/training checkpoint")
    protocol_ref = _ref(protocol)
    require(job.get("protocol") == protocol_ref and execution.get("protocol") == protocol_ref,
            "job/execution protocol binding")
    if not allow_fixture:
        require(checked_protocol is not None
                and job.get("source_anchor") == execution.get("source_anchor")
                == final.get("source_anchor") == checked_protocol.get("source_anchor"),
                "capture source anchor binding")
    owned = execution.get("owned")
    require(isinstance(owned, Mapping) and owned.get("exit_code") == 0
            and owned.get("natural_exit") is True and owned.get("cleanup_verified") is True
            and owned.get("timed_out") is False, "owned execution completion")
    require(final.get("schema") == V1.RESULT_SCHEMA and final.get("status") == "technical_complete"
            and final.get("mode") == "full" and final.get("frames") == 625
            and final.get("fps") == 50 and final.get("same_pts") is True
            and final.get("capture_approved") is False and final.get("model_ready") is False
            and final.get("used_for_decision") is False, "parent technical result")
    proofs = final.get("proofs")
    require(isinstance(proofs, Mapping) and set(proofs) >= PROOF_KEYS, "capture proof set")
    checked_proofs = cast(Mapping[str, Any], proofs)
    proof_paths = {key: _proof_ref(checked_proofs[key], key) for key in PROOF_KEYS}
    render = read(proof_paths["render_sync"])
    same_pts = read(proof_paths["same_pts"])
    native = read(proof_paths["native_manifest"])
    worker = read(proof_paths["worker_report"])
    require(render.get("schema") == "agentic_render_sync_v1" and render.get("status") == "closed"
            and render.get("calls") == render.get("frames_expected") == 625
            and render.get("physics_unchanged") is True and render.get("first_error") is None,
            "render sync 625")
    require(same_pts.get("schema") == "agentic_state_sidecar_join_v1"
            and same_pts.get("joined") is True and same_pts.get("frame_count") == 625
            and same_pts.get("checkpoint_sha256") == training["checkpoint"]["sha256"]
            and same_pts.get("used_for_decision") is False, "same PTS join")
    require(native.get("frame_count") == 625 and native.get("fps") == 50
            and worker.get("completed") is True and worker.get("fps") == 50,
            "native/worker 625")
    if not allow_fixture:
        require(worker.get("job_sha256") == sha(stage / "job.json"), "worker/job binding")
        trace = read(proof_paths["raw_trace"])
        require(trace.get("job_id") == job.get("source_id")
                and trace.get("checkpoint_sha256") == training["checkpoint"]["sha256"],
                "trace/job/checkpoint binding")
        require((stage / "preclose-proof.json").is_file()
                and (stage / "same-pts.json").is_file(), "sealed preclose proofs")
        modules = {name: importlib.import_module(module) for name, module in {
            "worker": "vision_pull_record_worker_v3", "sidecar": "record_agentic_state_sidecar_v1",
            "render": "run_agentic_play_record_v4", "shutdown": "vision_shutdown_policy_v1"}.items()}
        validated = V3._validated_job(stage / "job.json")
        preclose_before = _ref(stage / "preclose-proof.json")
        recomputed = V1._verify_preclose_proof(validated, modules)
        require(_ref(stage / "preclose-proof.json") == preclose_before
                and recomputed.get("proofs", {}).get("raw_trace") == final["proofs"]["raw_trace"],
                "preclose proof recomputation")
        terminal = Path(cast(str, validated["output_directory"])) / "result.json"
        modules["shutdown"].verify(stage / "job.json", validated, terminal)
    evidence = proof_paths["raw_trace"].parent
    specs = [evidence / f"window_{number}_v01_spec.json" for number in (1, 2)]
    require(all(path.is_file() and not path.is_symlink() for path in specs), "V01 specs missing")
    state = STATE.build_state_facts(proof_paths["sidecar_manifest"].parent,
                                    proof_paths["raw_trace"], proof_paths["native_manifest"])
    require(state["public"]["binding"]["frame_count"] == 625
            and state["public"]["binding"]["fps"] == 50, "state facts binding")
    return {"stage": str(stage), "protocol": _ref(protocol), "inspection": _ref(inspection),
            "job": _ref(stage / "job.json"), "execution": _ref(stage / "execution-result-v3.json"),
            "parent_result": _ref(stage / "parent-finalized-result-v3.json"),
            "proofs": {key: _ref(path) for key, path in sorted(proof_paths.items())},
            "v01_specs": [_ref(path) for path in specs], "state_facts": state["public"]}


def _play_facts(role: str, spec_ref: Mapping[str, str],
                raw_trace_ref: Mapping[str, str]) -> dict[str, Any]:
    spec_path = Path(spec_ref["path"]).resolve(strict=True)
    require(sha(spec_path) == spec_ref["sha256"], "V01 spec source")
    spec = read(spec_path)
    measurement_path = Path(cast(str, spec["measurement"]["path"])).resolve(strict=True)
    require(measurement_path.parent == spec_path.parent and not measurement_path.is_symlink(),
            "measurement source boundary")
    measurement = read(measurement_path)
    raw_trace_path = Path(raw_trace_ref["path"]).resolve(strict=True)
    require(sha(raw_trace_path) == raw_trace_ref["sha256"], "raw trace source")
    trace = read(raw_trace_path)
    records = trace.get("records")
    require(isinstance(records, list) and len(records) == 625, "raw trace records625")
    checked_records = cast(list[Mapping[str, Any]], records)
    samples = measurement.get("samples")
    require(isinstance(samples, list) and len(samples) == 6, "same-Play six samples")
    checked_samples = cast(list[object], samples)
    result = {}
    for out_index, source_index in enumerate((0, 2, 5), 1):
        sample = cast(Mapping[str, Any], checked_samples[source_index])
        raw = RECORD._at_timestamp(checked_records, float(sample["timestamp_s"]))
        expected = {"timestamp_s": raw["actual_timestamp_s"],
                    "command": dict(zip(("linear_x_mps", "linear_y_mps", "yaw_rate_rps"),
                                        raw["state"]["command"], strict=True)),
                    "motion": dict(zip(("linear_x_mps", "linear_y_mps", "yaw_rate_rps"),
                                       raw["state"]["motion"], strict=True)),
                    "pose": dict(raw["state"]["pose"])}
        require(dict(sample) == expected, "V01 measurement/raw trace mismatch")
        frame_id = f"f{out_index if role == 'case' else out_index + 3}"
        result[f"vision:play:{frame_id}:timestamp_s"] = sample["timestamp_s"]
        for group in ("command", "motion", "pose"):
            values = cast(Mapping[str, Any], sample[group])
            for name, value in values.items():
                result[f"vision:play:{frame_id}:{group}:{name}"] = value
    require(len(result) == 27, f"same-Play facts27: {role}")
    return result


def inspect(spec_path: Path, lab: Path, *, execution_anchor: str | None = None,
            allow_fixture: bool = False) -> dict[str, Any]:
    spec = read(spec_path)
    require(spec.get("schema") == SCHEMA and spec.get("split") == "development"
            and set(spec.get("roles", {})) == set(ROLES), "pair spec")
    phase_path = Path(cast(str, spec.get("phase_protocol"))).resolve(strict=True)
    phase = read(phase_path)
    require((allow_fixture and phase.get("schema") == "fixture_phase_protocol")
            or (phase.get("schema") == "local_same_play_pair_capture_phase_v1"
                and phase.get("status") == "registered_before_new_three_condition_gpu"
                and phase.get("claims", {}).get("capture_approved") is False
                and phase.get("claims", {}).get("new_image_model_exposure") is False
                and phase.get("exposure", {}).get("same_source_development_capture") is True
                and phase.get("exposure", {}).get("heldout_or_new_model_performance_claim") is False),
            "registered local capture phase")
    input_protocol_path = Path(cast(str, spec.get("input_protocol"))).resolve(strict=True)
    if allow_fixture:
        input_protocol = read(input_protocol_path)
        require(input_protocol.get("schema") == "local_same_play_pair_inputs_protocol_v1",
                "fixture pair input protocol")
    else:
        require(isinstance(execution_anchor, str), "pair execution anchor")
        input_protocol = verify_protocol(input_protocol_path, phase_path, cast(str, execution_anchor))
    batch_path = Path(cast(str, spec.get("source_batch"))).resolve(strict=True)
    missing = [role for role in ROLES
               if not Path(cast(str, spec["roles"][role]["stage"])).resolve().is_dir()]
    if missing:
        return {"schema": PENDING_SCHEMA, "status": "HOLD", "reason": "capture_stage_missing",
                "missing_roles": missing, "prepared": False, "model_use_authorized": False}
    training = {role: _training(batch_path, spec["roles"][role]["logical_id"]) for role in ROLES}
    require(training["case"]["seed"] == training["reference"]["seed"], "same-seed pair")
    require(training["case"]["arm"] == "F1" and training["reference"]["arm"] == "NONE",
            "case F1/reference NONE")
    captures = {role: _capture(spec["roles"][role], training[role], lab,
                               allow_fixture=allow_fixture) for role in ROLES}
    telemetry = {f"telemetry:{role}:{key}": value for role in ROLES
                 for key, value in training[role]["facts"].items()}
    play = {key: value for role in ROLES
            for key, value in _play_facts(role, captures[role]["v01_specs"][0],
                                          captures[role]["proofs"]["raw_trace"]).items()}
    require(len(telemetry) == 48 and len(play) == 54, "paired fact counts")
    source = {"source_batch": _ref(batch_path), "phase_protocol": _ref(phase_path),
              "input_protocol": _ref(input_protocol_path),
              "training": training, "captures": captures,
              "training_facts": telemetry, "same_play_facts": play}
    binding = digest(source)
    return {"schema": PENDING_SCHEMA, "status": "local_prepared_pending_main_approval",
            "split": "development", "already_seen": True, "source": source,
            "binding_sha256": binding, "training_fact_count": 48,
            "same_play_numeric_fact_count": 54, "planned_image_count": 6,
            "same_snapshot_required": True, "raw_frames_reused": True,
            "fresh_recording_created": False, "ready_for_registered_calls": False,
            "approval_granted": False, "model_use_authorized": False,
            "capture_approved": False, "heldout": False, "recovery": False,
            "video_benefit": False, "model_calls": 0, "gpu_executions": 0}


def prepare(spec_path: Path, lab: Path, output: Path, *, execution_anchor: str | None = None,
            allow_fixture: bool = False) -> dict[str, Any]:
    value = inspect(spec_path, lab, execution_anchor=execution_anchor, allow_fixture=allow_fixture)
    if value["status"] == "HOLD":
        return value
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "pending.json", value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("prepare-protocol")
    protocol.add_argument("--base-phase", required=True, type=Path)
    protocol.add_argument("--output", required=True, type=Path)
    protocol.add_argument("--source-anchor")
    run = sub.add_parser("prepare")
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--lab", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--execution-anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        result = prepare_protocol(args.base_phase, args.output, source_anchor=args.source_anchor)
        print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))
        return 0
    result = prepare(args.spec, args.lab, args.output, execution_anchor=args.execution_anchor)
    print(json.dumps({key: result[key] for key in ("schema", "status", "prepared",
                     "ready_for_registered_calls") if key in result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
