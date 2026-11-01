"""Go2 계단 명령 속도 4조건의 검증된 결과를 공개 JSON으로 투영한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

SCHEMA = "go2_stairs_command_contrast_public_v1"
PROTOCOL_SCHEMA = "go2_stairs_velocity_execution_v1"
RESULT_SCHEMA = "go2_stairs_velocity_execution_result_v1"
CLIPS = (
    ("quadruped-stairs-up-step-0p05-cmd-0p25", "up", 0.25),
    ("quadruped-stairs-down-step-0p05-cmd-0p25", "down", 0.25),
    ("quadruped-stairs-up-step-0p05-cmd-0p75", "up", 0.75),
    ("quadruped-stairs-down-step-0p05-cmd-0p75", "down", 0.75),
)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve(strict=True))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}은 유한 수여야 한다")
    return float(value)


def _json_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _protocol_proof(protocol_path: Path, anchor: str, repo: Path) -> dict[str, str]:
    actual_repo = Path(repo).resolve()
    protocol = Path(protocol_path).resolve(strict=True)
    try:
        relative = protocol.relative_to(actual_repo).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 안 파일이어야 한다") from error
    if relative.startswith("bench/private/"):
        raise ValueError("공개 exporter는 private protocol을 받지 않는다")
    head = _git(actual_repo, "rev-parse", "HEAD").decode().strip()
    if subprocess.run(["git", "-C", str(actual_repo), "merge-base", "--is-ancestor", anchor, head],
                      capture_output=True, check=False).returncode:
        raise ValueError("protocol anchor가 HEAD의 조상이 아니다")
    blob = _git(actual_repo, "show", f"{anchor}:{relative}")
    observed = hashlib.sha256(blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    current = _sha(protocol, lf=True)
    if observed != current:
        raise ValueError("protocol anchor blob과 현재 파일이 다르다")
    return {"path": relative, "sha256_lf": current, "anchor": anchor}


def _strict_execution(value: Any) -> float:
    if (not isinstance(value, dict)
            or value.get("job_assigned_before_release") is not True
            or value.get("natural_exit") is not True
            or value.get("timed_out") is not False
            or value.get("error_type") is not None
            or value.get("exit_code") != 0
            or value.get("cleanup_verified") is not True
            or not isinstance(value.get("owned_process_identities"), dict)):
        raise ValueError("owned SDK 실행의 exit0·natural exit·cleanup이 확인되지 않았다")
    return _finite(value.get("wall_s"), "owned child wall_s")


def _relative_file(root: Path, relative: Any) -> Path:
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if pure is None or pure.is_absolute() or ".." in pure.parts:
        raise ValueError("상대 경로가 다르다")
    path = (root / pure.as_posix()).resolve(strict=True)
    if root.resolve() not in path.parents:
        raise ValueError("파일이 clip 폴더 밖이다")
    return path


def _verifiers(repo: Path, wrapper_path: Path, protocol: dict[str, Any], supplied: Any):
    if supplied is not None:
        return supplied["adapter_verify"], supplied["shutdown_verify"]
    adapter = _load("go2_contrast_adapter", Path(repo) / "bench/run_go2_stairs_velocity_v1.py")
    wrapper = _load("go2_contrast_wrapper", wrapper_path)
    if _sha(wrapper_path, lf=True) != protocol["private_source_sha256_lf"]["exact_writer_shutdown_wrapper"]:
        raise ValueError("RI197 wrapper LF SHA가 protocol과 다르다")
    return adapter.verify_record, wrapper.verify_shutdown_evidence


def _incomplete(result: dict[str, Any], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    output = Path(out)
    output.mkdir(parents=True)
    value = {
        "schema": "go2_stairs_command_contrast_incomplete_v1",
        "status": "incomplete_not_public_result",
        "registered_outcome": "technical_incomplete",
        "completed_conditions": len(result.get("condition_rows", []))
        if isinstance(result.get("condition_rows"), list) else 0,
        "planned_conditions": len(protocol.get("clips", []))
        if isinstance(protocol.get("clips"), list) else None,
        "failure_present": result.get("failure") is not None,
        "claims": {"speed_effect": False, "heldout": False, "new_training": False,
                   "model_api": False, "recovery": False, "public_complete_result": False},
    }
    _write_new(output / "incomplete.json", value)
    return value


def _verify_launch(run_root: Path, clip_id: str, wrapper_path: Path, adapter_path: Path,
                   spec_path: Path, clip_out: Path, execution: dict[str, Any],
                   shutdown: dict[str, Any]) -> dict[str, str]:
    owned = run_root / "owned" / clip_id
    launch_path = owned / "launch.json"
    stdout_path = owned / "stdout.log"
    stderr_path = owned / "stderr.log"
    for path in (launch_path, stdout_path, stderr_path):
        if not path.is_file():
            raise ValueError(f"owned 실행 증거가 없다: {path.name}")
    launch = _read(launch_path)
    args = launch.get("args")
    if (not isinstance(args, list) or len(args) != 9 or args[1:] != [
            "-B", str(wrapper_path), "--runner", str(adapter_path), "--spec", str(spec_path),
            "--out", str(clip_out)] or not isinstance(launch.get("cwd"), str)):
        raise ValueError("run_owned launch argv/cwd가 등록 실행과 다르다")
    child_pid = shutdown.get("child_pid")
    identities = execution["owned_process_identities"]
    if not isinstance(child_pid, int) or str(child_pid) not in identities:
        raise ValueError("shutdown SDK pid가 owned process identity에 없다")
    return {"launch_sha256": _sha(launch_path), "stdout_sha256": _sha(stdout_path),
            "stderr_sha256": _sha(stderr_path)}


def _record_binding(clip_out: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = _read(clip_out / "manifest.json")
    trace_ref = manifest.get("state_trace")
    if not isinstance(trace_ref, dict):
        raise TypeError("manifest state_trace가 없다")
    trace = _relative_file(clip_out, trace_ref.get("path"))
    if _sha(trace) != trace_ref.get("sha256"):
        raise ValueError("manifest state_trace SHA가 다르다")
    first_line = next((line for line in trace.read_text(encoding="utf-8").splitlines()
                       if line.strip()), None)
    if first_line is None:
        raise ValueError("state trace가 비었다")
    first = json.loads(first_line)
    if not isinstance(first, dict) or first.get("frame") != 0 or first.get("sim_time_s") != 0.0:
        raise ValueError("state trace 첫 행이 frame0/sim_time0이 아니다")
    condition = protocol["condition"]
    terrain = manifest.get("terrain")
    source = manifest.get("source")
    if (manifest.get("checkpoint_sha256") != condition["checkpoint_sha256"]
            or not isinstance(terrain, dict) or not isinstance(source, dict)
            or terrain.get("terrain_seed") != condition["seed"]
            or terrain.get("step_height_m") != condition["step_height_m"]
            or terrain.get("step_width_m") != condition["step_width_m"]
            or source.get("isaaclab_commit") != protocol["isaaclab"]["commit"]
            or source.get("isaaclab_files_lf") != protocol["isaaclab"]["files_lf"]):
        raise ValueError("checkpoint·terrain·runtime source 결합이 protocol과 다르다")
    terrain_without_command = {key: value for key, value in terrain.items() if key != "command"}
    return {"initial_state_sha256": _json_digest(first),
            "pair_binding_sha256": _json_digest({
                "checkpoint_sha256": manifest["checkpoint_sha256"], "source": source,
                "terrain_without_command": terrain_without_command,
                "record_video_contract": manifest.get("record_video_contract"),
                "render_sync": manifest.get("render_sync"),
            })}


def export(result_path: Path, protocol_path: Path, protocol_anchor: str, repo: Path,
           wrapper_path: Path, out: Path, *, copy_videos: bool = False,
           allow_fixture: bool = False, verifiers: Any = None) -> dict[str, Any]:
    result_file = Path(result_path).resolve(strict=True)
    protocol_file = Path(protocol_path).resolve(strict=True)
    output = Path(out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result = _read(result_file)
    protocol = _read(protocol_file)
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "registered_only_after_commit":
        raise ValueError("Go2 command contrast protocol이 아니다")
    expected_clips = [{"clip_id": clip, "direction": direction, "command_x_m_s": command}
                      for clip, direction, command in CLIPS]
    if protocol.get("clips") != expected_clips:
        raise ValueError("등록 4조건 순서가 다르다")
    proof = ({"path": protocol_file.name, "sha256_lf": _sha(protocol_file, lf=True),
              "anchor": "cpu_fixture"}
             if allow_fixture else _protocol_proof(protocol_file, protocol_anchor, repo))
    if (result.get("schema") != RESULT_SCHEMA or result.get("status") != "complete"
            or result.get("technical_complete") is not True or result.get("failure") is not None):
        return _incomplete(result, protocol, output)
    rows = result.get("condition_rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("완료 결과는 4조건 전건이어야 한다")
    wrapper_path = Path(wrapper_path).resolve(strict=True)
    adapter_verify, shutdown_verify = _verifiers(repo, wrapper_path, protocol, verifiers)
    run_root = result_file.parent
    adapter_path = Path(repo).resolve() / "bench/run_go2_stairs_velocity_v1.py"
    public_rows: list[dict[str, Any]] = []
    media: list[dict[str, str]] = []
    for expected, stored in zip(expected_clips, rows, strict=True):
        if not isinstance(stored, dict) or any(stored.get(key) != expected[key] for key in expected):
            raise ValueError("result 조건 순서/값이 protocol과 다르다")
        wall_s = _strict_execution(stored.get("execution"))
        clip_id = expected["clip_id"]
        spec_path = run_root / "specs" / f"{clip_id}.json"
        clip_out = run_root / "clips" / clip_id
        spec = _read(spec_path)
        if (spec.get("clip_id") != clip_id or spec.get("direction") != expected["direction"]
                or spec.get("capture", {}).get("command_x_m_s") != expected["command_x_m_s"]):
            raise ValueError("spec과 protocol 조건이 다르다")
        shutdown = shutdown_verify(clip_out, spec, child_exit_code=0)
        if shutdown != stored.get("shutdown"):
            raise ValueError("shutdown 재검증과 result가 다르다")
        owned_artifacts = _verify_launch(
            run_root, clip_id, wrapper_path, adapter_path, spec_path, clip_out,
            stored["execution"], shutdown)
        verified = adapter_verify(
            spec_path, clip_out,
            technical_receipt={"child_exit_code": 0, "shutdown_verified": True,
                               "exact_encoder_verified": True}, repo=repo)
        technical, motion, poseguard = stored.get("technical"), stored.get("motion"), stored.get("poseguard")
        if (technical != {"complete": True, "manifest_sha256": verified["manifest_sha256"],
                          "trace_sha256": verified["trace_sha256"],
                          "video_sha256": verified["video_sha256"]}
                or motion != {"accepted": verified["motion_accepted"],
                              "failure_preserved": verified["motion_failure_preserved"],
                              "assessment": verified["assessment"]}
                or poseguard != {"body_up_world_z": verified["body_up_world_z"],
                                 "integrity_result": verified["integrity_result"]}):
            raise ValueError("adapter 재검증과 저장 result가 다르다")
        registered = "positive" if verified["motion_accepted"] else "not_supported"
        if stored.get("registered_outcome") != registered:
            raise ValueError("조건별 등록 결과 문구가 다르다")
        assessment, up = verified["assessment"], verified["body_up_world_z"]
        binding = _record_binding(clip_out, protocol)
        public_rows.append({
            "condition_id": clip_id, "seed": 200, "direction": expected["direction"],
            "command_x_m_s": expected["command_x_m_s"], "technical_complete": True,
            "motion_accepted": verified["motion_accepted"], "registered_outcome": registered,
            "root_dx_m": _finite(assessment.get("root_dx_m"), "root_dx_m"),
            "root_dz_m": _finite(assessment.get("root_dz_m"), "root_dz_m"),
            "progress_condition_pass": assessment.get("progress_condition_pass"),
            "height_condition_pass": assessment.get("height_condition_pass"),
            "termination_seen": assessment.get("termination_seen"),
            "body_up_world_z": up, "integrity_result": verified["integrity_result"],
            "owned_child_wall_s": wall_s, "manifest_sha256": verified["manifest_sha256"],
            "trace_sha256": verified["trace_sha256"], "video_sha256": verified["video_sha256"],
            "initial_state_sha256": binding["initial_state_sha256"],
            "pair_binding_sha256": binding["pair_binding_sha256"],
            "owned_artifacts": owned_artifacts,
        })
        if copy_videos:
            manifest = _read(clip_out / "manifest.json")
            video = _relative_file(clip_out, manifest.get("video", {}).get("path"))
            if _sha(video) != verified["video_sha256"]:
                raise ValueError("복사 전 video SHA가 다르다")
            target = output / "media" / f"{clip_id}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(video.read_bytes())
            media.append({"condition_id": clip_id, "file": f"media/{target.name}",
                          "sha256": _sha(target)})
    accepted = sum(row["motion_accepted"] is True for row in public_rows)
    registered_outcome = "positive" if accepted == 4 else "partial" if accepted else "not_supported"
    if result.get("registered_outcome") != registered_outcome:
        raise ValueError("전체 등록 결과 문구가 재계산과 다르다")
    by_key = {(row["direction"], row["command_x_m_s"]): row for row in public_rows}
    contrasts = []
    for direction in ("up", "down"):
        low, high = by_key[(direction, 0.25)], by_key[(direction, 0.75)]
        low_up, high_up = low["body_up_world_z"], high["body_up_world_z"]
        initial_state_exact_match = low["initial_state_sha256"] == high["initial_state_sha256"]
        pair_binding_exact_match = low["pair_binding_sha256"] == high["pair_binding_sha256"]
        if not initial_state_exact_match or not pair_binding_exact_match:
            raise ValueError("같은 방향의 속도 대조가 동일 초기상태·실행 결합이 아니다")
        contrasts.append({
            "direction": direction,
            "initial_state_exact_match": initial_state_exact_match,
            "checkpoint_terrain_runtime_exact_match": pair_binding_exact_match,
            "root_dx_delta_m_0p75_minus_0p25": high["root_dx_m"] - low["root_dx_m"],
            "root_dz_delta_m_0p75_minus_0p25": high["root_dz_m"] - low["root_dz_m"],
            "motion_accepted_0p25": low["motion_accepted"],
            "motion_accepted_0p75": high["motion_accepted"],
            "negative_count_delta_0p75_minus_0p25": (
                None if not isinstance(low_up, dict) or not isinstance(high_up, dict)
                else high_up["negative_count"] - low_up["negative_count"]),
            "owned_child_wall_delta_s_0p75_minus_0p25": high["owned_child_wall_s"] - low["owned_child_wall_s"],
        })
    summary = {
        "schema": SCHEMA, "status": "complete", "registered_outcome": registered_outcome,
        "conditions": 4, "independent_source_seeds": 1,
        "motion_accepted_conditions": accepted, "technical_complete_conditions": 4,
        "contrasts": contrasts, "preregistered_prediction": protocol["preregistered_prediction"],
        "acceptance": protocol["acceptance"], "result_language": protocol["result_language"],
        "claims": {"development_seen_source": True, "heldout": False,
                   "four_conditions_are_four_independent_seeds": False, "new_training": False,
                   "model_api": False, "recovery": False, "generalization": False,
                   "video_benefit": False},
        "media_copied": copy_videos,
    }
    provenance = {
        "schema": "go2_stairs_command_contrast_provenance_v1", "protocol": proof,
        "raw_result_sha256": _sha(result_file),
        "checkpoint_sha256": protocol["condition"]["checkpoint_sha256"],
        "source_anchor": protocol["source_anchor"],
        "source_sha256_lf": protocol["source_sha256_lf"],
        "private_source_role_sha256_lf": protocol["private_source_sha256_lf"],
        "exporter_sha256_lf": _sha(Path(__file__), lf=True),
        "consumer_reexecution": False, "gpu_or_model_reexecution": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output / "summary.json", summary)
    _write_new(output / "cases.json", {"schema": "go2_stairs_command_contrast_cases_v1",
                                        "rows": public_rows})
    _write_new(output / "provenance.json", provenance)
    if copy_videos:
        _write_new(output / "media.json", {"schema": "go2_stairs_command_contrast_media_v1",
                                            "items": media})
    files = sorted(path for path in output.rglob("*") if path.is_file())
    _write_new(output / "manifest.json", {
        "schema": "go2_stairs_command_contrast_manifest_v1",
        "files": [{"path": path.relative_to(output).as_posix(), "sha256": _sha(path)} for path in files],
        "private_or_user_paths_in_payload": False, "raw_videos_copied": copy_videos})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-anchor", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copy-videos", action="store_true")
    args = parser.parse_args()
    value = export(args.result, args.protocol, args.protocol_anchor, args.repo, args.wrapper, args.out,
                   copy_videos=args.copy_videos)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0 if value.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
