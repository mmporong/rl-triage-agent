"""rough play 상태 어댑터 CPU 회귀. 모든 clip은 synthetic_cpu fixture이며 실제 Isaac 계단 녹화가 아니다."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import rough_play_state_adapter_v1 as A  # pyright: ignore[reportMissingImports]

from rl_triage import correction_registry as CR  # pyright: ignore[reportMissingImports]
from rl_triage import probe_loop as L  # pyright: ignore[reportMissingImports]

CONTRACT = A.load_contract()
IFACE = CONTRACT["recorder_interface"]
CHECKPOINTS = {"biped": "a" * 64, "quadruped": "b" * 64}
BASE_HEIGHT = {"biped": 1.0, "quadruped": 0.4}  # synthetic_cpu 값. 실제 robot root 높이가 아니다.
CLIPS = [(row["robot_role"], row["direction"]) for row in CONTRACT["clips"]]
FRAMES = IFACE["frames"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def _seal(clip: Path) -> None:
    """녹화기 execute()의 shutdown marker 순서를 synthetic_cpu로 흉내 낸다."""
    folder = clip / "shutdown"
    if folder.exists():
        shutil.rmtree(folder)
    for number, event in enumerate(IFACE["shutdown_events"], 1):
        extra: dict[str, Any] = {}
        if event == "app-startup":
            extra = {"enable_cameras": True, "headless": True}
        elif event == "result-ready":
            extra = {"manifest_sha256": _sha(clip / "manifest.json")}
        _write(folder / f"{number:02d}-{event}.json", {"schema": IFACE["shutdown_marker_schema"], "event": event,
                                                        "pid": 4242, "sequence": number, "monotonic_ns": 1000 + number,
                                                        **extra})


def _edit_manifest(clip: Path, mutate: Callable[[dict[str, Any]], None], *, reseal: bool = True) -> None:
    path = clip / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    _write(path, manifest)
    if reseal:
        _seal(clip)


def make_clip(tmp: Path, role: str, direction: str, *, done_at: int | None = None,
              checkpoint: str | None = None) -> tuple[Path, Path]:
    """녹화기가 쓰는 spec·intent·manifest·trace·video·marker 형식의 synthetic_cpu clip."""
    name = f"synthetic_cpu-{role}-{direction}"
    checkpoint = checkpoint or CHECKPOINTS[role]
    tasks = IFACE["robot_tasks"][role]
    files = [*IFACE["common_isaaclab_sources"], IFACE["role_isaaclab_source"][role]]
    source = {"isaaclab_commit": "c" * 40, "isaaclab_files_lf": {path: "d" * 64 for path in files},
              "runner_sha256_lf": IFACE["sha256_lf"]}
    spec = {"schema": IFACE["spec_schema"], "clip_id": name, "robot_role": role, "direction": direction, **tasks,
            "seed": 7, "checkpoint": {"path": str(tmp / f"{name}.pt"), "sha256": checkpoint}, "source": source,
            "capture": dict(IFACE["capture"])}
    spec_path = tmp / f"{name}.spec.json"
    _write(spec_path, spec)
    clip = tmp / name
    _write(clip / "intent.json", {"spec_sha256": _sha(spec_path), "checkpoint_sha256": checkpoint, "new_training": 0,
                                  "model_api": 0, "diagnosis_or_recovery_claim": False})
    sign = 1.0 if direction == "up" else -1.0
    base = BASE_HEIGHT[role]
    last = FRAMES if done_at is None else done_at
    rows = [{"frame": i, "sim_time_s": i / IFACE["fps"], "root_pos_w": [0.004 * i, 0.0, base + sign * 0.0008 * i],
             "root_quat_w": [1.0, 0.0, 0.0, 0.0], "root_lin_vel_w": [0.2, 0.0, sign * 0.04],
             "terrain_origin_w": [0.0, 0.0, 0.0], "terminated": i == done_at} for i in range(last + 1)]
    if done_at is not None:
        rows[-1]["root_pos_w"] = [0.0, 0.0, base]  # done 행은 reset 위치처럼 둔다(synthetic_cpu)
    trace = clip / IFACE["trace_path"]
    _write_rows(trace, rows)
    video = clip / "video" / "rl-video-step-0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(f"synthetic_cpu mp4 placeholder {name}".encode())
    first, final = rows[0]["root_pos_w"], rows[-1]["root_pos_w"]
    manifest = {
        "schema": IFACE["manifest_schema"], "clip_id": name, "robot_role": role, "direction": direction, **tasks,
        "checkpoint_sha256": checkpoint, "source": source,
        "terrain": {"terrain_id": IFACE["terrain_for_direction"][direction], "terrain_seed": 7,
                    "step_height_m": IFACE["capture"]["step_height_m"],
                    "step_width_m": IFACE["capture"]["step_width_m"],
                    **IFACE["terrain_fixed"], "command": dict(IFACE["configured_command"]),
                    "direction_geometry": IFACE["direction_geometry"][direction]},
        "isaaclab_source": {"commit": source["isaaclab_commit"], "files_lf": source["isaaclab_files_lf"]},
        "assessment": {"outcome": "fall" if done_at is not None else "completed",
                       "root_dx_m": final[0] - first[0], "root_dz_m": final[2] - first[2],
                       "height_condition_pass": done_at is None, "progress_condition_pass": done_at is None,
                       "termination_seen": done_at is not None, "trace_rows": len(rows)},
        "video": {"path": "video/rl-video-step-0.mp4", "sha256": _sha(video)},
        "record_video_contract": {"gymnasium_version": "1.2.0", "source": "synthetic_cpu",
                                  "step_trigger": "step_id == 0",
                                  "video_length": FRAMES, "executed_steps": FRAMES, "frames_before_close": FRAMES,
                                  "recording_before_close": True, "close_finalizes_recording": True},
        "state_trace": {"path": IFACE["trace_path"], "sha256": _sha(trace)},
        "render_sync": {"method": IFACE["render_sync_method"], "calls": len(rows), "physics_state_unchanged": True},
        "camera": dict(IFACE["camera"]),
    }
    _write(clip / "manifest.json", manifest)
    _seal(clip)
    return clip, spec_path


def make_set(tmp: Path, **overrides: dict[str, Any]) -> list[tuple[Path, Path]]:
    return [make_clip(tmp, role, direction, **overrides.get(f"{role}-{direction}", {})) for role, direction in CLIPS]


def _clip(result: dict[str, Any], clip_id: str) -> dict[str, Any]:
    return next(row for row in result["public"]["clips"] if row["clip_id"] == clip_id)


def _tree(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): _sha(path) for path in sorted(folder.rglob("*")) if path.is_file()}


# --- 가용 수치와 unavailable -------------------------------------------------------------------


def test_complete_clips_yield_world_facts_only(tmp_path: Path) -> None:
    result = A.build_rough_play_facts(make_set(tmp_path))
    clip = _clip(result, "synthetic_cpu-biped-up")
    facts = clip["facts"]
    assert list(facts) == CONTRACT["fact_ids"]
    move = facts["root.world_displacement"]
    assert move["frame"] == "world"
    assert move["rows_used"] == {"first_frame": 0, "last_frame": FRAMES, "count": FRAMES + 1}
    assert move["x"] == pytest.approx(2.5, abs=1e-12) and move["z"] == pytest.approx(0.5, abs=1e-12)
    assert facts["root.world_xy_path_length"]["value"] == pytest.approx(2.5, abs=1e-12)
    height = facts["root.height_above_terrain_origin"]
    assert height["first"] == 1.0 and height["last"] == pytest.approx(1.5, abs=1e-12)
    assert facts["root.world_horizontal_speed"]["mean"] == pytest.approx(0.2, abs=1e-15)
    down = _clip(result, "synthetic_cpu-quadruped-down")["facts"]
    assert down["root.world_vertical_velocity"]["max"] == pytest.approx(-0.04, abs=1e-15)
    assert down["root.world_displacement"]["z"] == pytest.approx(-0.5, abs=1e-12)
    assert clip["coordinate_frames"]["command"] is None and clip["coordinate_frames"]["root_lin_vel_w"] == "world"
    assert clip["inputs"]["runner_sha256_lf"] == IFACE["sha256_lf"]
    assert facts["episode.done"]["seen"] is False and facts["trace.coverage"]["rows"] == FRAMES + 1


def test_missing_fields_are_unavailable_with_reason_not_zero_filled(tmp_path: Path) -> None:
    clip = _clip(A.build_rough_play_facts(make_set(tmp_path)), "synthetic_cpu-biped-down")
    availability = clip["field_availability"]
    for name in ("joint_state", "contact", "training_telemetry", "command_frame", "command_per_frame",
                 "termination_cause", "video_content"):
        assert availability[name].startswith("unavailable: ")
    tracking = clip["facts"]["command.tracking_error"]
    assert tracking["value"] is None and tracking["status"] == "unavailable"
    assert set(tracking["reason"]) == {"command_frame", "command_per_frame", "root_quat_w"}
    configured = clip["facts"]["command.configured"]
    assert configured["value"] == IFACE["configured_command"] and configured["frame"] is None
    assert not any(key.startswith(("joint", "contact", "training")) for key in clip["facts"])


def test_done_is_not_fall_and_done_row_is_excluded(tmp_path: Path) -> None:
    result = A.build_rough_play_facts(make_set(tmp_path, **{"biped-up": {"done_at": 300}}))
    clip = _clip(result, "synthetic_cpu-biped-up")
    done = clip["facts"]["episode.done"]
    assert done == {**done, "seen": True, "first_done_frame": 300, "cause": None, "done_row_in_motion_facts": False}
    assert done["cause_status"].startswith("unavailable")
    move = clip["facts"]["root.world_displacement"]
    assert move["rows_used"] == {"first_frame": 0, "last_frame": 299, "count": 300}
    assert move["x"] == pytest.approx(0.004 * 299, abs=1e-12)
    recorder = clip["recorder_interpretation"]
    assert recorder["outcome"] == "fall" and recorder["used_for_decision"] is False and recorder["done_row_included"]
    assert not any("fall" in key for key in clip["facts"])
    assert "fall_from_done" in {row["claim"] for row in result["public"]["unsupported_claims"]}


def test_done_on_first_step_leaves_motion_unavailable(tmp_path: Path) -> None:
    clip = _clip(A.build_rough_play_facts(make_set(tmp_path, **{"quadruped-up": {"done_at": 1}})),
                 "synthetic_cpu-quadruped-up")
    for key in ("root.world_displacement", "root.world_horizontal_speed"):
        assert clip["facts"][key]["value"] is None and clip["facts"][key]["status"] == "unavailable"
    assert list(clip["facts"]) == CONTRACT["fact_ids"]


def test_height_gain_is_not_stair_or_correction_success(tmp_path: Path) -> None:
    public = A.build_rough_play_facts(make_set(tmp_path))["public"]
    clip = next(row for row in public["clips"] if row["direction"] == "up")
    assert clip["facts"]["root.world_displacement"]["z"] > 0
    assert clip["recorder_interpretation"]["outcome"] == "completed"
    assert clip["recorder_interpretation"]["used_for_decision"] is False
    assert public["behavior_label"] is None and public["used_for_decision"] is False
    assert public["thresholds_created"] == 0
    claims = {row["claim"] for row in public["unsupported_claims"]}
    assert {"stair_success_from_height", "correction_success", "recorder_label_as_judgement",
            "model_interpretation_as_fact"} <= claims
    assert public["evidence_accounting"]["model_interpretation_source_count"] == 0
    text = json.dumps(public["clips"], ensure_ascii=False)
    assert "success" not in text.replace("stair_success", "")


# --- 결합 거부 -------------------------------------------------------------------------------


def _swap_spec(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    (clip, _), (_, other_spec) = pairs[0], pairs[1]
    return [(clip, other_spec), *pairs[1:]]


def _swap_file(pairs: list[tuple[Path, Path]], name: str) -> list[tuple[Path, Path]]:
    target, source = pairs[0][0], pairs[2][0]  # biped-up ← quadruped-up
    shutil.copyfile(source / name, target / name)
    return pairs


@pytest.mark.parametrize(("case", "match"), [
    ("spec", "다른 spec"),
    ("robot", "robot_role"),
    ("direction", "direction"),
    ("terrain", "terrain"),
    ("checkpoint", "checkpoint"),
    ("video", "video SHA"),
    ("trace", "state trace SHA"),
    ("runner", "runner"),
])
def test_swapped_inputs_are_rejected(tmp_path: Path, case: str, match: str) -> None:
    pairs = make_set(tmp_path)
    clip = pairs[0][0]
    if case == "spec":
        pairs = _swap_spec(pairs)
    elif case == "robot":
        _edit_manifest(clip, lambda m: m.update(robot_role="quadruped"))
    elif case == "direction":
        _edit_manifest(clip, lambda m: m.update(direction="down"))
    elif case == "terrain":
        _edit_manifest(clip, lambda m: m["terrain"].update(terrain_id=IFACE["terrain_for_direction"]["down"]))
    elif case == "checkpoint":
        _edit_manifest(clip, lambda m: m.update(checkpoint_sha256=CHECKPOINTS["quadruped"]))
    elif case == "video":
        _swap_file(pairs, "video/rl-video-step-0.mp4")
    elif case == "trace":
        _swap_file(pairs, IFACE["trace_path"])
    else:
        spec = json.loads(pairs[0][1].read_text(encoding="utf-8"))
        spec["source"]["runner_sha256_lf"] = "e" * 64
        _write(pairs[0][1], spec)
    with pytest.raises(A.FactsError, match=match):
        A.build_rough_play_facts(pairs)


def test_checkpoint_swap_across_robots_is_rejected(tmp_path: Path) -> None:
    pairs = make_set(tmp_path / "cross", **{"biped-down": {"checkpoint": CHECKPOINTS["quadruped"]}})
    with pytest.raises(A.FactsError, match="checkpoint"):
        A.build_rough_play_facts(pairs)
    pairs = make_set(tmp_path / "same", **{"biped-down": {"checkpoint": CHECKPOINTS["quadruped"]},
                                           "biped-up": {"checkpoint": CHECKPOINTS["quadruped"]}})
    with pytest.raises(A.FactsError, match="같은 checkpoint"):
        A.build_rough_play_facts(pairs)
    contract = json.loads(A.CONTRACT.read_text(encoding="utf-8"))
    contract["registered_checkpoints"]["biped"] = "f" * 64
    registered = tmp_path / "registered.json"
    _write(registered, contract)
    with pytest.raises(A.FactsError, match="등록 SHA"):
        A.build_rough_play_facts(make_set(tmp_path / "reg"), contract_path=registered)


def test_duplicate_clip_in_set_is_rejected(tmp_path: Path) -> None:
    pairs = make_set(tmp_path)
    with pytest.raises(A.FactsError, match="중복"):
        A.build_rough_play_facts([*pairs, pairs[0]])


@pytest.mark.parametrize("case", ["failure", "marker_missing", "manifest_unsealed", "truncated", "nan", "row_field",
                                  "assessment"])
def test_failed_partial_or_tampered_recording_is_rejected(tmp_path: Path, case: str) -> None:
    pairs = make_set(tmp_path)
    clip = pairs[0][0]
    trace = clip / IFACE["trace_path"]
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    match = {"failure": "실패", "marker_missing": "shutdown", "manifest_unsealed": "result-ready",
             "truncated": "길이", "nan": "비유한", "row_field": "필드", "assessment": "assessment"}[case]
    if case == "failure":
        _write(clip / "failure.json", {"error_type": "synthetic_cpu", "partial_evidence_preserved": True})
    elif case == "marker_missing":
        (clip / "shutdown" / "04-app-close-returned.json").unlink()
    elif case == "manifest_unsealed":
        _edit_manifest(clip, lambda m: m["render_sync"].update(calls=1), reseal=False)
    elif case in {"truncated", "nan", "row_field"}:
        if case == "truncated":
            rows = rows[:-1]
        elif case == "row_field":
            del rows[10]["root_lin_vel_w"]
        _write_rows(trace, rows)
        if case == "nan":
            text = trace.read_text(encoding="utf-8").replace('"root_lin_vel_w":[0.2', '"root_lin_vel_w":[NaN', 1)
            trace.write_text(text, encoding="utf-8", newline="\n")
        _edit_manifest(clip, lambda m: m["state_trace"].update(sha256=_sha(trace)))
    else:
        _edit_manifest(clip, lambda m: m["assessment"].update(root_dz_m=0.4))
    with pytest.raises(A.FactsError, match=match):
        A.build_rough_play_facts(pairs)


# --- 적용성·관측 전용·외부 호출 0 -------------------------------------------------------------


def test_flat_probe_and_repair_are_hold_for_rough_task(tmp_path: Path) -> None:
    public = A.build_rough_play_facts(make_set(tmp_path))["public"]
    rules = public["applicability"]
    assert rules["diagnosis_readiness"]["status"] == "observation_only"
    assert set(rules["probe_applicability"]) == set(L.PROBES)
    assert {row["status"] for row in rules["probe_applicability"].values()} == {CR.HOLD}
    assert {row["correction_id"] for row in rules["repair_applicability"].values()} == {
        row["correction_id"] for row in CR.REGISTRY.values()}
    assert {row["status"] for row in rules["repair_applicability"].values()} == {CR.HOLD}
    assert rules["requests_created"] == 0
    before = _tree(tmp_path)
    flat = A.check_repair(public, "restore_body_frame_reward")
    assert flat == {"action": CR.HOLD, "correction_id": "restore_body_frame_reward",
                    "rule": "flat_registered_correction_not_registered_for_rough_task",
                    "requests_created": 0, "ledger_written": False}
    assert A.check_repair(public, "restore_stair_gait")["rule"] == "unregistered_correction"
    assert _tree(tmp_path) == before and not list(tmp_path.rglob("*ledger*"))
    with pytest.raises(A.FactsError, match="rough play facts"):
        A.check_repair({"schema": "agentic_state_facts_v1"}, "restore_body_frame_reward")


def test_cli_is_observation_only_without_external_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pairs = make_set(inputs)
    before = _tree(inputs)

    def deny(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("외부 호출이 시도됐다")

    for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection"),
                        (subprocess, "run"), (subprocess, "Popen"), (os, "system")):
        monkeypatch.setattr(owner, name, deny)
    args = [item for clip, spec in pairs for item in ("--clip", str(clip), str(spec))]
    out = tmp_path / "facts"
    assert A.main([*args, "--out", str(out)]) == 0
    public = json.loads((out / A.PUBLIC_NAME).read_text(encoding="utf-8"))
    metadata = json.loads((out / A.METADATA_NAME).read_text(encoding="utf-8"))
    assert public["schema"] == A.SCHEMA and len(public["clips"]) == len(CLIPS)
    assert public["contract"]["status"] == "unexecuted_preregistration_candidate"
    assert metadata["public_sha256"] == A.R.digest(public)
    assert metadata["inputs"][0] == {"clip_dir": str(pairs[0][0]), "spec": str(pairs[0][1])}
    text = json.dumps(public, ensure_ascii=False)
    assert json.dumps(str(tmp_path))[1:-1] not in text and tmp_path.as_posix() not in text
    assert _tree(inputs) == before
    with pytest.raises(FileExistsError):
        A.main([*args, "--out", str(out)])


def test_protocol_is_unexecuted_candidate_bound_to_interface() -> None:
    assert CONTRACT["status"] == "unexecuted_preregistration_candidate"
    assert IFACE["sha256"] == IFACE["sha256_lf"] == "6858d73783ce02fecee6f901033660fd115c17f139d2937e07728fc25c075b5a"
    assert CONTRACT["thresholds"]["created"] == 0
    assert all(CONTRACT["thresholds"][key] is None for key in ("stair_success", "fall", "command_tracking_error"))
    assert set(CONTRACT["registered_checkpoints"].values()) == {None}
    assert sorted(CLIPS) == sorted((role, d) for role in IFACE["robot_tasks"] for d in IFACE["terrain_for_direction"])
    assert CONTRACT["validation"]["fixture"] == "synthetic_cpu" and CONTRACT["validation"]["real_trace"] == "pending"


def test_cpu_import_does_not_load_isaac_or_torch() -> None:
    script = (
        "import sys; sys.dont_write_bytecode = True; "
        f"[sys.path.insert(0, p) for p in {[str(ROOT / name) for name in ('src', 'bench', 'evals')]!r}]; "
        "import rough_play_state_adapter_v1; "
        "heavy = ('isaaclab', 'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'gymnasium', 'rsl_rl', 'openai'); "
        "print(sorted(name for name in sys.modules if name.split('.')[0] in heavy))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            check=True, timeout=120)
    assert result.stdout.strip() == "[]"
