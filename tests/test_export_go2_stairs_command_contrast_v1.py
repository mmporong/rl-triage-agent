# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evals/export_go2_stairs_command_contrast_v1.py"
SPEC = importlib.util.spec_from_file_location("export_go2_stairs_command_contrast_v1", SOURCE)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, incomplete: bool = False):
    repo = tmp_path / "repo"
    adapter = repo / "bench/run_go2_stairs_velocity_v1.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# fixture\n", encoding="utf-8")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("# fixture\n", encoding="utf-8")
    protocol = {
        "schema": M.PROTOCOL_SCHEMA, "status": "registered_only_after_commit",
        "source_anchor": "1" * 40,
        "clips": [{"clip_id": clip, "direction": direction, "command_x_m_s": command}
                  for clip, direction, command in M.CLIPS],
        "condition": {"checkpoint_sha256": "a" * 64, "seed": 200,
                      "step_height_m": 0.05, "step_width_m": 0.3},
        "isaaclab": {"commit": "i" * 40, "files_lf": {"a.py": "d" * 64}},
        "source_sha256_lf": {"bench/run_go2_stairs_velocity_v1.py": "b" * 64},
        "private_source_sha256_lf": {"exact_writer_shutdown_wrapper": "c" * 64},
        "preregistered_prediction": {"speed_effect_confirmed_before_run": False},
        "acceptance": {"body_up_world_z_negative_count": 0},
        "result_language": {"positive": "all", "partial": "some",
                            "not_supported": "none", "technical_incomplete": "technical"},
    }
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    run = tmp_path / "run"
    rows = []
    accepted_values = [False, False, True, False]
    for index, ((clip, direction, command), accepted) in enumerate(zip(M.CLIPS, accepted_values, strict=True)):
        spec_path = run / "specs" / f"{clip}.json"
        clip_out = run / "clips" / clip
        spec = {"clip_id": clip, "direction": direction,
                "capture": {"command_x_m_s": command}}
        _write(spec_path, spec)
        clip_out.mkdir(parents=True)
        initial = {"frame": 0, "sim_time_s": 0.0, "root_pos_w": [0.0, 0.0, 0.05],
                   "root_quat_w": [1.0, 0.0, 0.0, 0.0], "terminated": False}
        trace_raw = json.dumps(initial, separators=(",", ":")) + "\n"
        (clip_out / "state-trace.jsonl").write_text(trace_raw, encoding="utf-8")
        trace_sha = M._sha(clip_out / "state-trace.jsonl")
        _write(clip_out / "manifest.json", {
            "checkpoint_sha256": protocol["condition"]["checkpoint_sha256"],
            "source": {"isaaclab_commit": "i" * 40,
                       "isaaclab_files_lf": {"a.py": "d" * 64}},
            "terrain": {"terrain_seed": 200, "step_height_m": 0.05, "step_width_m": 0.3,
                        "terrain_id": f"stairs-{direction}",
                        "command": {"lin_vel_x": command}},
            "record_video_contract": {"video_length": 625},
            "render_sync": {"calls": 625, "physics_state_unchanged": True},
            "state_trace": {"path": "state-trace.jsonl", "sha256": trace_sha},
        })
        pid = 1000 + index
        shutdown = {"child_pid": pid, "shutdown_verified": True}
        execution = {"job_assigned_before_release": True, "natural_exit": True,
                     "timed_out": False, "error_type": None, "exit_code": 0,
                     "cleanup_verified": True, "owned_process_identities": {str(pid): 123 + index},
                     "wall_s": 10.0 + index}
        _write(run / "owned" / clip / "launch.json", {
            "args": ["python.bat", "-B", str(wrapper), "--runner", str(adapter),
                     "--spec", str(spec_path), "--out", str(clip_out)],
            "cwd": str(tmp_path / "IsaacLab"),
        })
        (run / "owned" / clip / "stdout.log").write_text("ok\n", encoding="utf-8")
        (run / "owned" / clip / "stderr.log").write_text("", encoding="utf-8")
        assessment = {"outcome": "completed" if accepted else "failed",
                      "root_dx_m": 1.0 + index, "root_dz_m": 0.1 * index,
                      "progress_condition_pass": accepted, "height_condition_pass": accepted,
                      "termination_seen": False}
        up = {"min": 0.8, "last": 0.9, "negative_count": 0 if index != 1 else 2}
        rows.append({
            "clip_id": clip, "direction": direction, "command_x_m_s": command,
            "execution": execution, "shutdown": shutdown,
            "technical": {"complete": True, "manifest_sha256": f"{index + 1}" * 64,
                          "trace_sha256": f"{index + 2}" * 64,
                          "video_sha256": f"{index + 3}" * 64},
            "motion": {"accepted": accepted, "failure_preserved": not accepted,
                       "assessment": assessment},
            "poseguard": {"body_up_world_z": up,
                          "integrity_result": "no_inversion_observed_not_full_stability_proof"},
            "registered_outcome": "positive" if accepted else "not_supported",
        })
    result = {"schema": M.RESULT_SCHEMA, "status": "complete", "technical_complete": True,
              "registered_outcome": "partial", "condition_rows": rows, "failure": None}
    if incomplete:
        result.update(status="failed", technical_complete=False,
                      registered_outcome="technical_incomplete", failure={"clip_id": M.CLIPS[1][0]})
        result["condition_rows"] = rows[:1]
    result_path = run / "result.json"
    _write(result_path, result)

    by_clip = {row["clip_id"]: row for row in rows}

    def adapter_verify(spec_path: Path, _out: Path, *, technical_receipt: dict, repo: Path):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        row = by_clip[spec["clip_id"]]
        return {"motion_accepted": row["motion"]["accepted"],
                "motion_failure_preserved": row["motion"]["failure_preserved"],
                "assessment": row["motion"]["assessment"],
                "body_up_world_z": row["poseguard"]["body_up_world_z"],
                "integrity_result": row["poseguard"]["integrity_result"],
                "manifest_sha256": row["technical"]["manifest_sha256"],
                "trace_sha256": row["technical"]["trace_sha256"],
                "video_sha256": row["technical"]["video_sha256"]}

    def shutdown_verify(_out: Path, spec: dict, *, child_exit_code: int):
        assert child_exit_code == 0
        return by_clip[spec["clip_id"]]["shutdown"]

    return repo, wrapper, protocol_path, result_path, {
        "adapter_verify": adapter_verify, "shutdown_verify": shutdown_verify}


def _export(tmp_path: Path, *, incomplete: bool = False):
    repo, wrapper, protocol, result, verifiers = _fixture(tmp_path, incomplete=incomplete)
    out = tmp_path / "public"
    value = M.export(result, protocol, "0" * 40, repo, wrapper, out,
                     allow_fixture=True, verifiers=verifiers)
    return value, out, result


def test_complete_four_condition_export_keeps_one_seed_and_registered_partial(tmp_path: Path) -> None:
    summary, out, _result = _export(tmp_path)
    assert summary["status"] == "complete"
    assert summary["registered_outcome"] == "partial"
    assert summary["conditions"] == 4
    assert summary["independent_source_seeds"] == 1
    assert summary["motion_accepted_conditions"] == 1
    assert summary["claims"]["four_conditions_are_four_independent_seeds"] is False
    assert summary["claims"]["new_training"] is False
    assert len(json.loads((out / "cases.json").read_text(encoding="utf-8"))["rows"]) == 4
    assert not (out / "media").exists()


def test_speed_contrast_and_owned_wall_are_recomputed(tmp_path: Path) -> None:
    summary, out, _result = _export(tmp_path)
    assert summary["contrasts"] == [
        {"direction": "up", "root_dx_delta_m_0p75_minus_0p25": 2.0,
         "initial_state_exact_match": True,
         "checkpoint_terrain_runtime_exact_match": True,
         "root_dz_delta_m_0p75_minus_0p25": pytest.approx(0.2),
         "motion_accepted_0p25": False, "motion_accepted_0p75": True,
         "negative_count_delta_0p75_minus_0p25": 0,
         "owned_child_wall_delta_s_0p75_minus_0p25": 2.0},
        {"direction": "down", "root_dx_delta_m_0p75_minus_0p25": 2.0,
         "initial_state_exact_match": True,
         "checkpoint_terrain_runtime_exact_match": True,
         "root_dz_delta_m_0p75_minus_0p25": pytest.approx(0.2),
         "motion_accepted_0p25": False, "motion_accepted_0p75": False,
         "negative_count_delta_0p75_minus_0p25": -2,
         "owned_child_wall_delta_s_0p75_minus_0p25": 2.0},
    ]
    cases = json.loads((out / "cases.json").read_text(encoding="utf-8"))["rows"]
    assert [row["owned_child_wall_s"] for row in cases] == [10.0, 11.0, 12.0, 13.0]


def test_tampered_execution_identity_or_stored_motion_is_rejected(tmp_path: Path) -> None:
    repo, wrapper, protocol, result, verifiers = _fixture(tmp_path)
    value = json.loads(result.read_text(encoding="utf-8"))
    value["condition_rows"][0]["execution"]["owned_process_identities"] = {}
    _write(result, value)
    with pytest.raises(ValueError, match="shutdown SDK pid"):
        M.export(result, protocol, "0" * 40, repo, wrapper, tmp_path / "out1",
                 allow_fixture=True, verifiers=verifiers)


def test_incomplete_result_is_sealed_without_public_summary(tmp_path: Path) -> None:
    value, out, _result = _export(tmp_path, incomplete=True)
    assert value["status"] == "incomplete_not_public_result"
    assert value["claims"]["public_complete_result"] is False
    assert (out / "incomplete.json").is_file()
    assert not (out / "summary.json").exists()
    assert not (out / "manifest.json").exists()


def test_output_is_fresh_and_public_payload_has_no_absolute_or_private_paths(tmp_path: Path) -> None:
    _summary, out, result = _export(tmp_path)
    payload = "\n".join(path.read_text(encoding="utf-8") for path in out.glob("*.json"))
    assert "bench/private" not in payload
    assert str(tmp_path) not in payload
    repo, wrapper, protocol, _ignored, verifiers = _fixture(tmp_path / "second")
    with pytest.raises(FileExistsError):
        M.export(result, protocol, "0" * 40, repo, wrapper, out,
                 allow_fixture=True, verifiers=verifiers)
