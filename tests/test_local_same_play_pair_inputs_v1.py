from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import local_same_play_pair_inputs_v1 as M  # pyright: ignore[reportMissingImports]


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_jsonl(path: Path, values: list[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
                    encoding="utf-8")
    return path


def training(tmp: Path, logical: str, seed: int, arm: str, cp: Path) -> tuple[dict, Path]:
    root = tmp / "source" / logical
    telemetry = write(root / "telemetry.json", {"summary": {
        f"metric-{index:02d}": {"mean_last_20pct": float(index)} for index in range(24)}})
    manifest = write(root / "manifest.json", {
        "schema": "residual_training_manifest_v1", "seed": seed, "arm": arm,
        "original_faults": [] if arm == "NONE" else ["F1"], "applied_corrections": [],
        "runtime_source_sha256_lf": {"source.py": "a" * 64}})
    metadata = write(root / "metadata.json", {
        "telemetry": str(telemetry.resolve()), "private_training": {
            "seed": seed, "arm": arm, "original_faults": [] if arm == "NONE" else ["F1"],
            "applied_corrections": []}})
    collected = write(root / "collected.json", {
        "metadata_ref": {"file": "metadata.json", "sha256": ref(metadata)["sha256"]},
        "source_binding": {"checkpoint_sha256": ref(cp)["sha256"],
                           "telemetry_sha256": ref(telemetry)["sha256"]}})
    train = write(root / "training.json", {"manifest": ref(manifest), "collected": ref(collected)})
    result = write(root / "result.json", {"status": "complete", "seed": seed,
                                          "runtime_arm": arm, "training": ref(train)})
    row = {"stage": logical, "result": ref(result), "checkpoint": {
        **ref(cp), "iteration": 299, "size_bytes": cp.stat().st_size}}
    return row, telemetry


def capture(tmp: Path, name: str, cp: Path, *, render_calls: int = 625) -> dict[str, str]:
    stage = tmp / "captures" / name
    evidence = stage / "capture" / "evidence"
    sidecar = tmp / "captures" / f"{name}.sidecar"
    protocol = write(tmp / "protocols" / f"{name}.json", {"schema": "fixture_protocol"})
    inspection = write(tmp / "inspections" / f"{name}.json", {"schema": "fixture_inspection"})
    write(stage / "job.json", {"checkpoint_sha256": ref(cp)["sha256"],
                                "protocol": ref(protocol)})
    records = [{"actual_timestamp_s": float(index), "state": {
        "command": [1, 2, 3], "motion": [4, 5, 6], "pose": {"x": 7, "y": 8}}}
        for index in range(625)]
    raw = write(evidence / "raw_trace.json", {"records": records})
    native = write(stage / "capture" / "native_capture_manifest.json",
                   {"frame_count": 625, "fps": 50})
    render = write(stage / "render-sync.json", {"schema": "agentic_render_sync_v1",
        "status": "closed", "calls": render_calls, "frames_expected": 625,
        "physics_unchanged": True, "first_error": None})
    same = write(stage / "same-pts.json", {"schema": "agentic_state_sidecar_join_v1",
        "joined": True, "frame_count": 625, "checkpoint_sha256": ref(cp)["sha256"],
        "used_for_decision": False})
    worker = write(stage / "capture" / "native-policy-record.report.json",
                   {"completed": True, "fps": 50})
    supporting = {
        "app_close_entered": write(stage / "close.json", {"entered": True}),
        "shutdown": write(stage / "shutdown.json", {"status": "complete"}),
        "sidecar_manifest": write(sidecar / "sidecar_manifest.json", {"fixture": True}),
        "sidecar_records": write_jsonl(sidecar / "records.jsonl",
                                        [{"frame": 0}, {"frame": 1}]),
    }
    proofs = {"native_manifest": ref(native), "raw_trace": ref(raw), "render_sync": ref(render),
              "same_pts": ref(same), "worker_report": ref(worker),
              **{key: ref(path) for key, path in supporting.items()}}
    final = write(stage / "parent-finalized-result-v3.json", {
        "schema": M.V1.RESULT_SCHEMA, "status": "technical_complete", "mode": "full",
        "frames": 625, "fps": 50, "same_pts": True, "checkpoint_sha256": ref(cp)["sha256"],
        "capture_approved": False, "model_ready": False, "used_for_decision": False,
        "proofs": proofs})
    write(stage / "execution-result-v3.json", {"checkpoint_sha256": ref(cp)["sha256"],
        "protocol": ref(protocol),
        "owned": {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                  "timed_out": False}})
    samples = [{"timestamp_s": float(i),
                "command": {"linear_x_mps": 1, "linear_y_mps": 2, "yaw_rate_rps": 3},
                "motion": {"linear_x_mps": 4, "linear_y_mps": 5, "yaw_rate_rps": 6},
                "pose": {"x": 7, "y": 8}}
               for i in range(6)]
    measurement = write(evidence / "window_1_measurements.json", {"samples": samples})
    for number in (1, 2):
        write(evidence / f"window_{number}_v01_spec.json", {
            "measurement": {"path": str(measurement.resolve())}})
    return {"stage": str(stage), "protocol": str(protocol), "inspection": str(inspection),
            "final": str(final), "raw": str(raw), "render": str(render)}


def world(tmp: Path, monkeypatch: pytest.MonkeyPatch, *, case_seed: int = 2430,
          reference_seed: int = 2430) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(M.STATE, "build_state_facts", lambda *_args, **_kwargs: {
        "public": {"binding": {"frame_count": 625, "fps": 50}, "facts": {"fixture": True}}})
    case_cp = tmp / "case.pt"; case_cp.write_bytes(b"case-checkpoint")
    ref_cp = tmp / "reference.pt"; ref_cp.write_bytes(b"reference-checkpoint")
    case_row, _ = training(tmp, "s2430-f1", case_seed, "F1", case_cp)
    ref_row, _ = training(tmp, "s2430-none", reference_seed, "NONE", ref_cp)
    batch = write(tmp / "batch.json", {"schema": "prospective_diagnosis_batch_result_v1",
        "status": "complete", "stages": [case_row, ref_row]})
    phase = write(tmp / "phase.json", {
        "schema": "local_same_play_pair_capture_phase_v1",
        "status": "registered_before_new_three_condition_gpu",
        "claims": {"capture_approved": False, "new_image_model_exposure": False},
        "exposure": {"same_source_development_capture": True,
                     "heldout_or_new_model_performance_claim": False}})
    input_protocol = write(tmp / "input-protocol.json", {
        "schema": "local_same_play_pair_inputs_protocol_v1", "status": "candidate_not_registered",
        "source_anchor": None,
        "sources_lf": {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                                                   .replace(b"\r", b"\n")).hexdigest()
                       for name in M.SOURCE_FILES},
        "base_capture_phase": {"path": "bench/protocols/local_same_play_pairs_s2430_s2431_20261101_v1.json",
                               "sha256": ref(phase)["sha256"]},
        "split": "development", "training_fact_count": 48,
        "same_play_numeric_fact_count": 54, "planned_image_counts": [0, 6],
        "approval": {"ready_before_approval": False},
        "claims": {}})
    roles = {
        "case": {"logical_id": "s2430-f1", **capture(tmp, "case", case_cp)},
        "reference": {"logical_id": "s2430-none", **capture(tmp, "reference", ref_cp)},
    }
    spec = write(tmp / "spec.json", {"schema": M.SCHEMA, "split": "development",
        "source_batch": str(batch), "phase_protocol": str(phase),
        "input_protocol": str(input_protocol), "roles": roles})
    return spec, roles


def test_prepares_pending_pair_with_48_and_54_facts(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _ = world(tmp_path, monkeypatch)
    result = M.prepare(spec, tmp_path, tmp_path / "out", allow_fixture=True)
    assert result["status"] == "local_prepared_pending_main_approval"
    assert (result["training_fact_count"], result["same_play_numeric_fact_count"],
            result["planned_image_count"]) == (48, 54, 6)
    assert result["ready_for_registered_calls"] is False
    assert result["capture_approved"] is result["heldout"] is result["recovery"] is False


def test_missing_capture_is_hold_without_output(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    spec, roles = world(tmp_path, monkeypatch)
    roles["case"]["stage"] = str(tmp_path / "missing")
    write(tmp_path / "missing-spec.json", {"schema": M.SCHEMA, "split": "development",
        "source_batch": M.read(spec)["source_batch"],
        "phase_protocol": M.read(spec)["phase_protocol"],
        "input_protocol": M.read(spec)["input_protocol"], "roles": roles})
    out = tmp_path / "out"
    result = M.prepare(tmp_path / "missing-spec.json", tmp_path, out, allow_fixture=True)
    assert result["status"] == "HOLD" and not out.exists()


def test_rejects_different_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _ = world(tmp_path, monkeypatch, reference_seed=2431)
    with pytest.raises(ValueError, match="same-seed"):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_rejects_missing_raw_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, roles = world(tmp_path, monkeypatch)
    Path(roles["case"]["raw"]).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_rejects_stale_render_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, roles = world(tmp_path, monkeypatch)
    render = Path(roles["case"]["render"])
    write(tmp_path / "changed-render.json", {"schema": "agentic_render_sync_v1"})
    render.write_text((tmp_path / "changed-render.json").read_text(), encoding="utf-8")
    final = Path(roles["case"]["final"])
    body = M.read(final); body["proofs"]["render_sync"] = ref(render); write(tmp_path / "new-final.json", body)
    final.write_text((tmp_path / "new-final.json").read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="render sync"):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_rejects_resealed_checkpoint_swap(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    spec, roles = world(tmp_path, monkeypatch)
    job = Path(roles["case"]["stage"]) / "job.json"
    body = M.read(job); body["checkpoint_sha256"] = "0" * 64; job.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="capture/training checkpoint"):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_rejects_other_protocol_binding(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    spec, roles = world(tmp_path, monkeypatch)
    job = Path(roles["case"]["stage"]) / "job.json"
    body = M.read(job)
    body["protocol"] = ref(Path(roles["reference"]["protocol"]))
    job.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol binding"):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_rejects_changed_training_telemetry(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _ = world(tmp_path, monkeypatch)
    telemetry = tmp_path / "source" / "s2430-f1" / "telemetry.json"
    telemetry.write_text('{"summary":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="training source binding"):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_rejects_measurement_value_changed_from_raw_trace(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, roles = world(tmp_path, monkeypatch)
    measurement = Path(roles["case"]["stage"]) / "capture/evidence/window_1_measurements.json"
    body = M.read(measurement)
    body["samples"][0]["motion"]["linear_x_mps"] = 999
    measurement.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement/raw trace"):
        M.inspect(spec, tmp_path, allow_fixture=True)


def test_proof_ref_accepts_jsonl_and_still_checks_sha(tmp_path: Path) -> None:
    records = write_jsonl(tmp_path / "records.jsonl", [{"frame": 0}, {"frame": 1}])
    checked = ref(records)
    assert M._proof_ref(checked, "sidecar_records") == records.resolve()
    records.write_text('{"frame":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar_records"):
        M._proof_ref(checked, "sidecar_records")


def test_registered_protocol_rejects_dirty_source_and_protocol_blob(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    source = repo / M.SOURCE_FILES[0]
    for relative in M.SOURCE_FILES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    original_source = (ROOT / M.SOURCE_FILES[0]).read_bytes()
    phase = write(repo / "bench/protocols/local_same_play_pairs_s2430_s2431_20261101_v1.json", {
        "schema": "local_same_play_pair_capture_phase_v1",
        "status": "registered_before_new_three_condition_gpu"})
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "fixture"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", *M.SOURCE_FILES,
                    "bench/protocols/local_same_play_pairs_s2430_s2431_20261101_v1.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture source"], check=True,
                   capture_output=True)
    source_anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                   capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(M, "ROOT", repo)
    protocol = repo / "bench/protocols/local_same_play_pair_inputs_s2430_s2431_20261101_v1.json"
    M.prepare_protocol(phase, protocol, source_anchor=source_anchor)
    subprocess.run(["git", "-C", str(repo), "add", protocol.relative_to(repo).as_posix()], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture protocol"], check=True,
                   capture_output=True)
    execution_anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                      capture_output=True, text=True).stdout.strip()
    assert M.verify_protocol(protocol, phase, execution_anchor, repo=repo)["source_anchor"] == source_anchor
    source.write_bytes(original_source + b"\n")
    with pytest.raises(ValueError, match="source closure"):
        M.verify_protocol(protocol, phase, execution_anchor, repo=repo)
    source.write_bytes(original_source)
    state_source = repo / "evals/agentic_state_facts_v1.py"
    original_state_source = state_source.read_bytes()
    state_source.write_bytes(original_state_source + b"\n")
    with pytest.raises(ValueError, match="source closure"):
        M.verify_protocol(protocol, phase, execution_anchor, repo=repo)
    state_source.write_bytes(original_state_source)
    protocol.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="protocol blob"):
        M.verify_protocol(protocol, phase, execution_anchor, repo=repo)
