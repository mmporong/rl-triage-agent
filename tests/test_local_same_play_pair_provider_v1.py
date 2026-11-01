from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]
import local_same_play_pair_provider_v1 as M  # pyright: ignore[reportMissingImports]


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def file_ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    facts = {f"telemetry:{role}:{metric}": float(index)
             for role in ("case", "reference")
             for index, metric in enumerate(M.C.V.TELEMETRY_FIELDS)}
    bindings = {"case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
                "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64}}
    packet = M.C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None, facts={
        **facts, **{f"runtime:{role}:{name}": None for role in ("case", "reference") for name in (
            "available", "updates_observed", "expected_optimizer_steps", "optimizer_steps_min",
            "optimizer_steps_max", "expected_minibatches", "minibatches_min", "minibatches_max")}},
        input_bindings=bindings, execution_budget_s=1200)
    state = write(tmp / "state.json", {"initial_packet": packet})
    input_protocol = write(tmp / "input-protocol.json", {"schema": "fixture-input"})
    play = {f"vision:play:f{number}:{field}": float(number)
            for number in range(1, 7) for field in (
                "timestamp_s", "command:linear_x_mps", "command:linear_y_mps",
                "command:yaw_rate_rps", "motion:linear_x_mps", "motion:linear_y_mps",
                "motion:yaw_rate_rps", "pose:base_height_m", "pose:up_axis_z")}
    source_body = {"input_protocol": file_ref(input_protocol), "training_facts": facts,
                   "same_play_facts": play, "captures": {}, "training": {
                       role: {"checkpoint": {"sha256": binding["checkpoint_sha256"]},
                              "telemetry": {"sha256": binding["telemetry_sha256"]}}
                       for role, binding in bindings.items()}}
    pending_binding = M.INPUT.digest(source_body)
    pending = write(tmp / "pending.json", {
        "schema": M.INPUT.PENDING_SCHEMA, "status": "local_prepared_pending_main_approval",
        "binding_sha256": pending_binding, "training_fact_count": 48,
        "same_play_numeric_fact_count": 54, "planned_image_count": 6,
        "ready_for_registered_calls": False, "model_use_authorized": False,
        "capture_approved": False, "heldout": False, "recovery": False, "video_benefit": False,
        "source": source_body,
    })
    protocol = write(tmp / "provider-protocol.json", {
        "schema": M.SCHEMA, "status": "candidate_not_registered", "source_anchor": None,
        "sources_lf": {}, "input_protocol": {"path": "fixture", "sha256": file_ref(input_protocol)["sha256"],
                                                "execution_anchor": "a" * 40},
        "frame_selection": M.FRAME_SELECTION, "projection": M.PROJECTION,
        "authority": M.AUTHORITY})
    approval = write(tmp / "approval.json", {
        "schema": "local_pair_provider_preparation_approval_v1",
        "actor": "fixture_operator", "scope": "CPU_provider_input_materialization_only",
        "pending": file_ref(pending), "pending_binding_sha256": pending_binding,
        "input_execution_anchor": "a" * 40, "expected_image_counts": [0, 6],
        "frame_role_mapping": {"case": ["f1", "f2", "f3"],
                               "reference": ["f4", "f5", "f6"]},
        "window": 1, "source_indices_per_role": [0, 2, 5],
        "input_preparation_authorized": True, "api_authorized": False,
        "probe_approval": False, "gpu_authorized": False, "correction_authorized": False,
        "recovery_claim": False, "capture_flags_mutated": False,
    })
    input_spec = write(tmp / "input-spec.json", {"schema": "fixture-input-spec"})
    lab = tmp / "lab"; lab.mkdir()
    return pending, approval, input_spec, lab, state, protocol


def inspector(pending: Path):
    return lambda *_args, **_kwargs: M.read(pending)


def materializer(_pending: dict[str, Any], _output: Path) -> tuple[list[dict[str, Any]], dict]:
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
             base64.b64encode(b"\xff\xd8fixture\xff\xd9").decode(), "detail": "low"}}
    rows = []
    for index in range(1, 7):
        role = "case" if index <= 3 else "reference"
        rows.append({"role": role, "frame_id": f"f{index}", "source_frame_id": f"f0{index}",
                     "timestamp_s": float(index), "label": {"type": "text", "text": json.dumps(
                         {"frame_id": f"f{index}", "timestamp_s": float(index)})},
                     "image": deepcopy(image)})
    return rows, {"fixture": True}


def test_prepares_equal_information_pair_without_authority(tmp_path: Path) -> None:
    pending, approval, input_spec, lab, state, protocol = fixture(tmp_path)
    result = M.prepare(pending, approval, input_spec, lab, state, protocol, "b" * 40,
                       tmp_path / "out", allow_fixture=True, materializer=materializer,
                       input_inspector=inspector(pending))
    assert result["image_counts"] == {"logs_only": 0, "logs_plus_frames": 6}
    assert result["training_fact_count"] == 48 and result["same_play_numeric_fact_count"] == 54
    assert result["runtime_facts_excluded_equally"] == 16
    assert all(result[key] is False for key in ("api_authorized", "gpu_authorized",
                                                 "probe_authorized", "correction_authorized",
                                                 "heldout", "recovery", "video_benefit"))
    pair = M.PAIRED.read(tmp_path / "out/paired/pair.json")
    assert pair["differences"]["common_telemetry_fact_count"] == 0  # snapshot owns the 48
    assert pair["differences"]["common_same_play_numeric_fact_count"] == 54
    assert pair["status"] == "prepared_registered_cpu_inputs_not_model_result"
    assert pair["frame_role_mapping"] == {"case": ["f1", "f2", "f3"],
                                           "reference": ["f4", "f5", "f6"]}
    for arm in M.PAIRED.ARMS:
        provider = M.PAIRED.read(Path(pair["arms"][arm]["provider_input"]["path"]))
        assert provider["context"]["goal"] == M.PAIR_GOAL
    comparison = M.PAIRED.compare(tmp_path / "out/paired/pair.json", tmp_path / "comparison")
    assert comparison["status"] == "inconclusive"


def test_rejects_approval_or_packet_binding_swap(tmp_path: Path) -> None:
    pending, approval, input_spec, lab, state, protocol = fixture(tmp_path)
    body = M.read(approval); body["pending_binding_sha256"] = "0" * 64
    write(tmp_path / "bad-approval.json", body)
    with pytest.raises(ValueError, match="approval"):
        M.prepare(pending, tmp_path / "bad-approval.json", input_spec, lab, state, protocol,
                  "b" * 40, tmp_path / "bad", allow_fixture=True, materializer=materializer,
                  input_inspector=inspector(pending))
    packet = M.read(state); packet["initial_packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "9" * 64
    write(tmp_path / "bad-state.json", packet)
    with pytest.raises(ValueError, match="CP telemetry"):
        M.prepare(pending, approval, input_spec, lab, tmp_path / "bad-state.json", protocol,
                  "b" * 40, tmp_path / "bad2", allow_fixture=True, materializer=materializer,
                  input_inspector=inspector(pending))


def test_rejects_training_fact_or_frame_role_swap(tmp_path: Path) -> None:
    pending, approval, input_spec, lab, state, protocol = fixture(tmp_path)
    changed = M.read(pending); changed["source"]["training_facts"][next(iter(
        changed["source"]["training_facts"]))] = 999
    changed["binding_sha256"] = M.INPUT.digest(changed["source"])
    write(tmp_path / "changed-pending.json", changed)
    changed_approval = M.read(approval); changed_approval["pending"] = file_ref(tmp_path / "changed-pending.json")
    changed_approval["pending_binding_sha256"] = changed["binding_sha256"]
    write(tmp_path / "changed-approval.json", changed_approval)
    with pytest.raises(ValueError, match="training fact"):
        M.prepare(tmp_path / "changed-pending.json", tmp_path / "changed-approval.json",
                  input_spec, lab, state, protocol, "b" * 40, tmp_path / "bad",
                  allow_fixture=True, materializer=materializer,
                  input_inspector=inspector(tmp_path / "changed-pending.json"))

    def swapped(value: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict]:
        rows, refs = materializer(value, out)
        rows[0]["role"] = "reference"
        return rows, refs

    with pytest.raises(ValueError, match="role/frame mapping"):
        M.prepare(pending, approval, input_spec, lab, state, protocol, "b" * 40,
                  tmp_path / "bad2", allow_fixture=True, materializer=swapped,
                  input_inspector=inspector(pending))


def test_rejects_recomputed_pending_or_registered_policy_change(tmp_path: Path) -> None:
    pending, approval, input_spec, lab, state, protocol = fixture(tmp_path)
    changed = M.read(pending)
    changed["source"]["same_play_facts"][next(iter(changed["source"]["same_play_facts"]))] += 1
    with pytest.raises(ValueError, match="pending/current"):
        M.prepare(pending, approval, input_spec, lab, state, protocol, "b" * 40,
                  tmp_path / "bad", allow_fixture=True, materializer=materializer,
                  input_inspector=lambda *_args, **_kwargs: changed)
    changed_protocol = M.read(protocol); changed_protocol["projection"]["image_counts"] = [0, 5]
    write(tmp_path / "bad-protocol.json", changed_protocol)
    with pytest.raises(ValueError, match="protocol policy"):
        M.prepare(pending, approval, input_spec, lab, state, tmp_path / "bad-protocol.json",
                  "b" * 40, tmp_path / "bad2", allow_fixture=True, materializer=materializer,
                  input_inspector=inspector(pending))


def test_protocol_candidate_pins_transitive_sources(tmp_path: Path) -> None:
    source = write(tmp_path / "input.json", {"schema": "input"})
    protocol = M.prepare_protocol(source, "a" * 40, tmp_path / "protocol.json")
    assert protocol["status"] == "candidate_not_registered"
    assert set(protocol["sources_lf"]) == set(M.SOURCE_FILES)
    assert protocol["authority"] == {"cpu_input_preparation": True, "api": False,
                                     "gpu": False, "probe": False, "correction": False}


def test_registered_protocol_rejects_source_or_protocol_blob_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    for relative in M.SOURCE_FILES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    input_protocol = write(
        repo / "bench/protocols/local_same_play_pair_inputs_s2430_s2431_20261101_v1.json",
        {"schema": "registered-input"})
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "fixture"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", *M.SOURCE_FILES,
                    input_protocol.relative_to(repo).as_posix()], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture source"], check=True,
                   capture_output=True)
    source_anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                   capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(M, "ROOT", repo)
    monkeypatch.setattr(M.INPUT, "verify_protocol", lambda *_args, **_kwargs: {})
    protocol_path = repo / "bench/protocols/local_same_play_pair_provider_s2430_s2431_20261101_v1.json"
    M.prepare_protocol(input_protocol, source_anchor, protocol_path, source_anchor=source_anchor)
    subprocess.run(["git", "-C", str(repo), "add", protocol_path.relative_to(repo).as_posix()], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture protocol"], check=True,
                   capture_output=True)
    execution_anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                      capture_output=True, text=True).stdout.strip()
    assert M.verify_protocol(protocol_path, input_protocol, execution_anchor, repo=repo)["source_anchor"]
    source = repo / M.SOURCE_FILES[0]
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="source closure"):
        M.verify_protocol(protocol_path, input_protocol, execution_anchor, repo=repo)
    source.write_bytes(original)
    protocol_path.write_bytes(protocol_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="protocol blob"):
        M.verify_protocol(protocol_path, input_protocol, execution_anchor, repo=repo)
