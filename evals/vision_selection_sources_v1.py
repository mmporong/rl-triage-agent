"""비교 입력을 원본 Play·렌더 확인 기록·동결 선택 입력과 대조한다."""
from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
from datetime import timedelta
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import run_vision_brev_batch_v1 as SOURCE  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE
import vision_render_context_v1 as CONTEXT  # pyright: ignore[reportMissingImports]
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMPARISON
import vision_selection_contract_v1 as CONTRACT

LOCATIONS_SCHEMA = "vision_selection_input_locations_v1"
INPUT_SCHEMA = "vision_selection_source_manifest_v1"
VERIFIED_SCHEMA = "vision_selection_verified_inputs_v1"
CAMERA_SCHEMA = "vision_camera_pose_audit_v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
READINESS_SCHEMA = "vision_selection_model_readiness_v1"
RESULT_KEYS = {
    "schema", "completed", "job_sha256", "protocol_sha256", "checkpoint_sha256",
    "request_binding", "resource", "resource_parent_sha256", "live_lease_proof_sha256",
    "admission_sha256", "admission_issued_at_utc", "admission_expires_at_utc",
    "profile_sha256", "mode", "primary_artifact_sha256", "video_sha256", "artifacts",
    "fixed_evaluation_used_as_behavior_ground_truth", "independent_visual_evidence_source_count",
    "zero_update_pixel_lag_confirmed", "gpu_smoke_required_before_model_use",
}
_COMMON_REFS = {"job", "protocol", "admission", "result", "execution", "context", "attestation",
                "gpu_observation_before", "gpu_observation_after"}
_RECORD_ARTIFACTS = {"trace", "measurement_1", "measurement_2", "spec_1", "spec_2"}
_SMOKE_ARTIFACTS = {"render_sync_smoke", "audit_rgb_1", "audit_rgb_2", "audit_rgb_3"}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _same(actual: Any, expected: Any) -> bool:
    return RECORD.canonical(actual) == RECORD.canonical(expected)


def _exact(value: Any, keys: set[str], label: str) -> dict:
    _require(isinstance(value, dict) and set(value) == keys, f"{label} 필드가 다르다")
    return value


def _regular(path: Path) -> Path:
    _require(path.is_absolute() and path.is_file() and path == path.resolve(strict=True)
             and not path.is_symlink() and not any(p.is_symlink() for p in path.parents),
             "원본 파일은 canonical regular path여야 한다")
    info = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "원본 hardlink를 거부한다")
    return path


def _relative(repo: Path, name: Any) -> Path:
    _require(isinstance(name, str) and bool(name) and "\\" not in name
             and ":" not in name and not name.startswith("/")
             and all(part not in {"", ".", ".."} for part in name.split("/")),
             "원본 위치는 저장소 상대 경로여야 한다")
    path = repo / name
    _require(path.relative_to(repo).as_posix() == name, "원본 상대 경로가 다르다")
    return _regular(path)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _ref(repo: Path, value: Any, *, frozen: bool = False) -> Path:
    keys = {"path", "sha256", "status"} if frozen else {"path", "sha256"}
    row = _exact(value, keys, "원본 reference")
    _require(isinstance(row["sha256"], str) and HEX64.fullmatch(row["sha256"]) is not None,
             "원본 SHA 형식이 다르다")
    if frozen:
        _require(row["status"] == "frozen", "원본 reference가 frozen 상태가 아니다")
    path = _relative(repo, row["path"])
    _require(_sha(path) == row["sha256"], "원본 파일 SHA가 다르다")
    return path


def _read(path: Path) -> dict:
    def pairs(values: list[tuple[str, Any]]) -> dict:
        value = {}
        for key, item in values:
            _require(key not in value, "원본 JSON 중복 key를 거부한다")
            value[key] = item
        return value
    value = json.loads(_regular(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("원본 JSON 비유한 수를 거부한다")))
    _require(isinstance(value, dict), "원본 JSON은 object여야 한다")
    return value


def _verify_record_anchor(repo: Path, capture: dict) -> None:
    """녹화 당시 source commit의 결합 blob을 현재 Git 바이트와 대조한다."""
    anchor = capture["job"]["source"]["commit"]
    _require(isinstance(anchor, str) and re.fullmatch(r"[0-9a-f]{40}", anchor) is not None,
             "녹화 source commit 형식이 다르다")
    names = [*capture["job"]["source"]["files_lf"], capture["source_snapshot"]["protocol"]["path"]]
    for name in names:
        _relative(repo, name)
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              encoding="utf-8", errors="strict", check=False)
    ancestry = git("merge-base", "--is-ancestor", anchor, "HEAD")
    _require(ancestry.returncode == 0, "녹화 source commit이 현재 HEAD의 조상이 아니다")
    tracked = git("ls-files", "--error-unmatch", "--", *names)
    clean = git("status", "--porcelain", "--", *names)
    diff = git("diff", "--quiet", anchor, "HEAD", "--", *names)
    _require(tracked.returncode == clean.returncode == diff.returncode == 0 and not clean.stdout.strip(),
             "녹화 source commit의 결합 blob 또는 현재 파일이 다르다")


def camera_pose_audit(trace_path: Path) -> dict:
    """625개 state·시각·camera 행을 대조한 감사값을 계산한다. 파일을 쓰지 않는다."""
    trace_path = _regular(trace_path)
    trace = _read(trace_path)
    _require(trace.get("schema") == RECORD.TRACE_SCHEMA and trace.get("method") == RECORD.METHOD
             and trace.get("profile") == RECORD.fixed_profile()
             and trace.get("profile_sha256") == RECORD.digest(RECORD.fixed_profile())
             and trace.get("frame_count") == RECORD.FRAME_COUNT,
             "원본 trace profile이 다르다")
    records = trace.get("records")
    if not isinstance(records, list) or len(records) != RECORD.FRAME_COUNT:
        raise ValueError("원본 trace의 capture 수가 다르다")
    camera_rows = []
    camera_profile = RECORD.fixed_profile()["camera"]
    for index, row in enumerate(records):
        _exact(row, {"frame_index", "capture_ordinal", "simulation_step", "state", "camera",
                     "render_timing", "rgb_sha256", "actual_timestamp_s"}, "trace record")
        _require(type(row["frame_index"]) is int and type(row["capture_ordinal"]) is int
                 and row["frame_index"] == row["capture_ordinal"] == index,
                 "trace capture 순서가 다르다")
        _require(isinstance(row["rgb_sha256"], str) and HEX64.fullmatch(row["rgb_sha256"]) is not None,
                 "trace RGB SHA 형식이 다르다")
        state = _exact(row["state"], {"command", "motion", "pose", "root_position_world_m",
                                     "simulation_time_s", "common_step_counter"}, "trace state")
        _require(type(row["simulation_step"]) is int and row["simulation_step"] >= 0
                 and type(state["common_step_counter"]) is int
                 and state["common_step_counter"] == row["simulation_step"], "trace 실제 step이 다르다")
        _require(RECORD._vector3(state["command"], "command") == [1.0, 0.0, 0.0],
                 "trace 실제 명령이 다르다")
        RECORD._vector3(state["motion"], "motion")
        pose = _exact(state["pose"], {"base_height_m", "up_axis_z"}, "pose")
        for key, value in pose.items():
            RECORD._finite(value, key)
        root = RECORD._vector3(state["root_position_world_m"], "root")
        sim_time = RECORD._finite(state["simulation_time_s"], "simulation time")
        RECORD.validate_render_timing(row["render_timing"], sim_time)
        camera = _exact(row["camera"], {"eye_world_m", "lookat_world_m"}, "camera")
        for key, offset_key in (("eye_world_m", "eye_offset_m"), ("lookat_world_m", "lookat_offset_m")):
            actual = RECORD._vector3(camera[key], key)
            expected = [root[i] + camera_profile[offset_key][i] for i in range(3)]
            _require(all(abs(a - b) <= 1e-9 for a, b in zip(actual, expected, strict=True)),
                     "trace camera가 root 평행 추적과 다르다")
        timestamp = RECORD._finite(row["actual_timestamp_s"], "actual PTS")
        _require(abs(timestamp - index * RECORD.FRAME_PERIOD_S) <= RECORD.PTS_TOLERANCE_S,
                 "trace 실제 PTS가 고정 간격과 다르다")
        camera_rows.append({"frame_index": index, "actual_timestamp_s": timestamp,
                            "root_position_world_m": root, "camera": camera})
    _require(RECORD.bind_actual_pts(records, [row["actual_timestamp_s"] for row in records]) == records,
             "trace 실제 physics step과 video PTS가 다르다")
    return {"schema": CAMERA_SCHEMA, "status": "pass", "trace_sha256": _sha(trace_path),
            "profile_sha256": trace["profile_sha256"], "frame_count": len(records),
            "camera_rows_sha256": RECORD.digest(camera_rows),
            "pixel_content_recomputed": False}


def _verify_window(contract: dict, trace: dict, window: str) -> None:
    _require(window in COMPARISON.WINDOWS, "등록되지 않은 Play window")
    number = COMPARISON.WINDOWS.index(window) + 1
    timestamps = list(RECORD.WINDOWS[number - 1])
    provenance = contract["provenance"]
    _require(provenance["play_run_id"] == f"{trace['job_id']}-w{number}"
             and provenance["source_lineage_id"] == f"{trace['job_id']}-same-play"
             and provenance["parent_source_lineage_id"] is None
             and provenance["measurement"] is not None,
             "V01 구간이 원본 Play 실행과 다르다")
    expected_play = {"task": RECORD.TASK, "camera": RECORD.fixed_profile()["camera"]["identifier"],
                     "command_name": "base_velocity", "command_frame": "body", "motion_frame": "body",
                     "settings_sha256": RECORD.digest(RECORD.fixed_profile())}
    _require(provenance["play"] == expected_play
             and [row["timestamp_s"] for row in contract["frames"]] == timestamps,
             "V01 Play 설정·시각이 고정 구간과 다르다")
    _require(contract["synchronized_samples"] == _samples(trace, timestamps),
             "V01 운동 측정이 원본 trace와 다르다")


def _samples(trace: dict, timestamps: list[float]) -> list[dict]:
    samples = []
    for timestamp in timestamps:
        matches = [row for row in trace["records"] if row["actual_timestamp_s"] == timestamp]
        _require(len(matches) == 1, "원본 trace의 구간 시각이 유일하지 않다")
        state = matches[0]["state"]
        keys = ("linear_x_mps", "linear_y_mps", "yaw_rate_rps")
        samples.append({"timestamp_s": timestamp,
                        "command": dict(zip(keys, state["command"], strict=True)),
                        "motion": dict(zip(keys, state["motion"], strict=True)), "pose": state["pose"]})
    return samples


def _verify_snapshot(snapshot: dict, protocol: dict, job: dict) -> None:
    _require(snapshot.get("lab_head") == protocol["environment"]["isaaclab_sha"]
             and snapshot.get("lab_tracked_clean") is True
             and snapshot.get("source_anchor") == job["source"]["commit"], "실행 source snapshot 결합이 다르다")
    _require(snapshot.get("protocol", {}).get("sha256_lf") == protocol["_verified_sha256_lf"],
             "실행 snapshot의 source protocol SHA가 다르다")
    expected_environment = protocol["environment"]
    actual_environment = snapshot.get("environment", {})
    for key in ("python", "isaacsim", "isaaclab_sha", "isaaclab", "isaaclab_rl", "torch",
                "rsl-rl-lib", "gymnasium"):
        _require(actual_environment.get(key) == expected_environment.get(key),
                 f"실행 SDK가 source protocol과 다르다: {key}")
    _require(actual_environment.get("cuda_available") is True
             and type(actual_environment.get("cuda_device_count")) is int
             and actual_environment["cuda_device_count"] == 1, "실행 CUDA 장치 수가 다르다")
    for key in ("source_sha256_lf", "runtime_source_sha256_lf", "python_package_source_sha256_lf"):
        _require(snapshot.get(key) == protocol[key], f"실행 source map이 다르다: {key}")


def _result_files(repo: Path, refs: dict, result: dict, artifact_keys: set[str]) -> dict[str, Path]:
    artifacts = _exact(result.get("artifacts"), artifact_keys, "실행 artifact 목록")
    paths: dict[str, Path] = {}
    names: set[str] = set()
    for key in artifact_keys:
        descriptor = _exact(artifacts[key], {"file", "sha256", "bytes"}, "실행 artifact")
        path = _ref(repo, refs[key])
        _require(path.name == descriptor["file"] and path.name not in names and path not in paths.values()
                 and _sha(path) == descriptor["sha256"] and type(descriptor["bytes"]) is int
                 and descriptor["bytes"] > 0 and path.stat().st_size == descriptor["bytes"],
                 "실행 artifact 이름·바이트·크기가 다르다")
        names.add(path.name)
        paths[key] = path
    return paths


def _verify_capture(repo: Path, refs: dict, mode: str, source_protocol: dict,
                    resource_parent_sha: str, live_proof_sha: str, resource: dict) -> dict:
    artifact_keys = _RECORD_ARTIFACTS if mode == RECORD.RECORD_MODE else _SMOKE_ARTIFACTS
    extra = {"video"} if mode == RECORD.RECORD_MODE else set()
    _exact(refs, _COMMON_REFS | artifact_keys | extra, "capture 원본 위치")
    paths = {key: _ref(repo, ref) for key, ref in refs.items()}
    job = _exact(_read(paths["job"]), {"schema", "method", "mode", "job_id", "profile", "source",
                                      "protocol", "checkpoint", "resource", "remote_gpu_admission",
                                      "output_directory"}, "capture job")
    _require(job["schema"] == RECORD.JOB_SCHEMA and job["method"] == RECORD.METHOD
             and job["mode"] == mode and _same(job["profile"], RECORD.fixed_profile()), "capture job mode/profile이 다르다")
    _exact(job["source"], {"repo_root", "commit", "files_lf"}, "capture source")
    _exact(job["resource"], {"resource_id", "gpu_uuid"}, "capture resource")
    _exact(job["checkpoint"], {"path", "sha256", "logical_id"}, "capture checkpoint")
    _exact(job["protocol"], {"path", "sha256"}, "capture protocol reference")
    _exact(job["remote_gpu_admission"], {"path", "sha256", "resource_parent_sha256", "live_lease_proof_sha256"},
           "capture remote admission")
    for value in (job["output_directory"], job["checkpoint"]["path"]):
        _require(isinstance(value, str) and value.startswith("/") and "\\" not in value
                 and ":" not in value and all(part not in {".", ".."} for part in PurePosixPath(value).parts)
                 and str(PurePosixPath(value)) == value, "원격 provenance 경로가 POSIX 정규형이 아니다")
    _require(job["source"]["files_lf"] == source_protocol["recording"]["source_files_lf"],
             "capture source가 source protocol과 다르다")
    _require({"bench/record_vision_evidence_v1.py", "bench/vision_play_worker_v1.py",
              "bench/vision_render_context_v1.py", "bench/vision_render_attestation_v1.py"}
             <= set(job["source"]["files_lf"]), "capture 실행 source가 빠졌다")
    _require(job["resource"]["resource_id"] == source_protocol["resources"]["owned_instance_id"],
             "capture 자원이 등록 VM과 다르다")
    protocol = _read(paths["protocol"])
    _require(job["protocol"]["sha256"] == _sha(paths["protocol"])
             and _same(protocol, {"schema": RECORD.PROTOCOL_SCHEMA, "method": RECORD.METHOD,
                              "mode": mode, "profile": RECORD.fixed_profile(),
                              "source_commit": job["source"]["commit"],
                              "source_files_lf": job["source"]["files_lf"],
                              "checkpoint_sha256": job["checkpoint"]["sha256"], "resource": job["resource"]}),
             "capture 개별 protocol이 job과 다르다")
    admission = _exact(_read(paths["admission"]), {"schema", "status", "job_id", "resource_kind", "lease_id",
                                                  "request_binding", "issued_at_utc", "expires_at_utc",
                                                  "resource_id", "gpu_uuid", "resource_parent_sha256",
                                                  "live_lease_proof_sha256"}, "capture admission")
    remote = job["remote_gpu_admission"]
    _require(remote["sha256"] == _sha(paths["admission"])
             and remote["resource_parent_sha256"] == resource_parent_sha
             and remote["live_lease_proof_sha256"] == live_proof_sha, "capture admission 원본 결합이 다르다")
    for key, expected in {"schema": RECORD.ADMISSION_SCHEMA, "status": "granted", "job_id": job["job_id"],
                          "resource_kind": "remote_gpu", "request_binding": RECORD.request_binding(job),
                          **job["resource"], "lease_id": resource["lease_id"], "resource_parent_sha256": resource_parent_sha,
                          "live_lease_proof_sha256": live_proof_sha}.items():
        _require(_same(admission.get(key), expected), "capture admission 의미 결합이 다르다")
    issued, expires = SOURCE._stamp(admission["issued_at_utc"]), SOURCE._stamp(admission["expires_at_utc"])
    _require(SOURCE._stamp(resource["issued_at_utc"]) <= issued < expires
             and expires == SOURCE._stamp(resource["expires_at_utc"])
             and issued + timedelta(seconds=RECORD.EXECUTION_AUTHORIZATION_S) <= expires,
             "capture admission 시각이 등록 lease 범위 밖이다")
    result = _exact(_read(paths["result"]), RESULT_KEYS, "capture result")
    expected = {"schema": RECORD.RESULT_SCHEMA, "completed": True, "mode": mode,
                "job_sha256": _sha(paths["job"]), "protocol_sha256": _sha(paths["protocol"]),
                "checkpoint_sha256": job["checkpoint"]["sha256"],
                "request_binding": RECORD.request_binding(job), "resource": job["resource"],
                "resource_parent_sha256": resource_parent_sha, "live_lease_proof_sha256": live_proof_sha,
                "admission_sha256": _sha(paths["admission"]),
                "admission_issued_at_utc": admission.get("issued_at_utc"),
                "admission_expires_at_utc": admission.get("expires_at_utc"),
                "profile_sha256": RECORD.digest(RECORD.fixed_profile()),
                "fixed_evaluation_used_as_behavior_ground_truth": False, "independent_visual_evidence_source_count": 1,
                "zero_update_pixel_lag_confirmed": False, "gpu_smoke_required_before_model_use": mode == RECORD.RECORD_MODE}
    _require(_same({key: result[key] for key in expected}, expected), "capture result가 job/admission과 다르다")
    execution = _read(paths["execution"])
    _require(execution.get("schema") == "vision_brev_record_execution_v1"
             and execution.get("completed") is True and type(execution.get("exit_code")) is int
             and execution["exit_code"] == 0 and execution.get("timed_out") is False
             and execution.get("cleanup_verified") is True
             and execution.get("job_sha256") == _sha(paths["job"])
             and execution.get("result_sha256") == _sha(paths["result"])
             and execution.get("gpu_before") == execution.get("gpu_after") == []
             and isinstance(execution.get("source_before"), dict)
             and execution["source_before"] == execution.get("source_after"), "capture 실행 영수증이 완료 상태가 아니다")
    snapshot = execution["source_before"]
    _verify_snapshot(snapshot, source_protocol, job)
    for side in ("before", "after"):
        observation = paths[f"gpu_observation_{side}"]
        _require(execution.get(f"gpu_{side}_observation", {}).get("sha256") == _sha(observation),
                 "capture GPU 관측 SHA가 다르다")
        REMOTE.verify_observation_reference({"path": str(observation), "sha256": _sha(observation)},
                                            job["resource"]["resource_id"], job["resource"]["gpu_uuid"])
    fingerprint = CONTEXT.read_context(paths["context"], job, _sha(paths["job"]), snapshot)
    context_body = _read(paths["context"])
    _require(issued <= SOURCE._stamp(context_body["captured_at_utc"])
             <= SOURCE._stamp(context_body["finished_at_utc"]) <= expires,
             "renderer context 수집 시각이 admission 범위 밖이다")
    import vision_render_attestation_v1 as ATTEST  # pyright: ignore[reportMissingImports]
    attestation = ATTEST.verify_attestation(paths["attestation"], job, _sha(paths["job"]), snapshot,
                                          paths["context"], paths["execution"], paths["result"],
                                          protocol_path=paths["protocol"])
    _require(SOURCE._stamp(context_body["finished_at_utc"])
             <= SOURCE._stamp(attestation["execute_started_at_utc"])
             <= SOURCE._stamp(attestation["execute_finished_at_utc"])
             <= SOURCE._stamp(attestation["attested_at_utc"])
             <= min(expires, SOURCE._stamp(source_protocol["resources"]["deadline_utc"])),
             "renderer 실행 순서·시각이 등록 범위 밖이다")
    artifact_paths = _result_files(repo, refs, result, artifact_keys)
    return {"job": job, "result": result, "paths": paths, "artifact_paths": artifact_paths,
            "source_snapshot": snapshot, "fingerprint": fingerprint}


def _verify_smoke(capture: dict) -> dict:
    paths, result = capture["artifact_paths"], capture["result"]
    report_path = paths["render_sync_smoke"]
    report = _exact(_read(report_path), {"schema", "status", "capture_order", "policy_steps",
                                       "capture_phase_physics_steps", "environment_reset_performed", "delta_time_s",
                                       "rt_subframes", "pause_timeline", "renderer_combination_only",
                                       "universal_zero_lag_claim", "captures"}, "render smoke")
    expected = {"schema": "vision_render_sync_smoke_v1", "status": "pass",
                "capture_order": ["warmup_discard", "A", "B", "A"], "policy_steps": 0,
                "capture_phase_physics_steps": 0, "environment_reset_performed": True,
                "delta_time_s": 0.0, "rt_subframes": 1, "pause_timeline": False,
                "renderer_combination_only": True, "universal_zero_lag_claim": False}
    _require(_same({key: report[key] for key in expected}, expected), "render smoke 상태·범위가 다르다")
    _require(result["video_sha256"] is None and result["primary_artifact_sha256"] == _sha(report_path),
             "render smoke primary 결과가 다르다")
    captures = report["captures"]
    _require(isinstance(captures, list) and len(captures) == 4
             and [row.get("label") for row in captures] == ["warmup", "A", "B", "A"], "render smoke capture 순서가 다르다")
    for index, row in enumerate(captures):
        _exact(row, {"label", "rgb_sha256", "simulation_time_s", "read_times", "reference_time", "camera"},
               "render smoke capture")
        RECORD.validate_render_timing({"read_times": row["read_times"], "reference_time": row["reference_time"]},
                                     RECORD._finite(row["simulation_time_s"], "smoke time"))
        _require(isinstance(row["rgb_sha256"], str) and HEX64.fullmatch(row["rgb_sha256"]) is not None,
                 "render smoke RGB SHA 형식이 다르다")
        camera = _exact(row["camera"], {"eye_world_m", "lookat_world_m"}, "smoke camera")
        for key, value in camera.items():
            RECORD._vector3(value, key)
        root = [camera["lookat_world_m"][i] - RECORD.fixed_profile()["camera"]["lookat_offset_m"][i]
                for i in range(3)]
        offset = [-3.0, 3.0, 2.0] if index == 2 else [3.0, 3.0, 2.0]
        _require(all(abs(camera["eye_world_m"][i] - (root[i] + offset[i])) <= 1e-9 for i in range(3)),
                 "render smoke camera가 고정 A/B offset과 다르다")
        if index:
            rgb = paths[f"audit_rgb_{index}"]
            _require(rgb.stat().st_size == RECORD.WIDTH * RECORD.HEIGHT * 3
                     and _sha(rgb) == row["rgb_sha256"], "render smoke RGB 원문이 report와 다르다")
    _require(captures[1]["rgb_sha256"] == captures[3]["rgb_sha256"] != captures[2]["rgb_sha256"]
             and captures[1]["camera"] == captures[3]["camera"] != captures[2]["camera"]
             and len({row["simulation_time_s"] for row in captures}) == 1,
             "render smoke A-B-A가 같은 정지 시각에서 재현되지 않았다")
    return report


def _verify_record(capture: dict) -> dict:
    paths, result, job = capture["artifact_paths"], capture["result"], capture["job"]
    video = capture["paths"]["video"]
    _require(_sha(video) == result["video_sha256"] == result["primary_artifact_sha256"], "record video SHA가 다르다")
    trace = _read(paths["trace"])
    camera_pose_audit(paths["trace"])
    _require(trace.get("job_id") == job["job_id"] and trace.get("video_sha256") == _sha(video)
             and trace.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
             and trace.get("time_basis") == RECORD.fixed_profile()["time_basis"], "record trace provenance가 다르다")
    expected_alignment = {"camera_pose_applied_before_render": True,
                          "capture_primitive": RECORD.fixed_profile()["camera"]["capture_primitive"],
                          "rt_subframes": 1, "pause_timeline": False, "delta_time_s": 0.0,
                          "robot_state_unchanged_across_render_enforced": True,
                          "rgb_timing_matches_state_enforced": True, "annotator_cpu_copy": True,
                          "zero_update_pixel_lag_confirmed": False, "gpu_smoke_required_before_model_use": True}
    _require(trace.get("render_alignment") == expected_alignment, "record render alignment가 다르다")
    encoder = trace.get("encoder_receipt", {})
    _require(encoder.get("natural_exit") is True and encoder.get("returncode") == 0
             and encoder.get("frames_written") == RECORD.FRAME_COUNT
             and encoder.get("bytes_written") == RECORD.FRAME_COUNT * RECORD.WIDTH * RECORD.HEIGHT * 3,
             "record encoder 완료 증거가 다르다")
    actual_timing = RECORD.probe_video_timing(video)
    _require(trace.get("timing_receipt") == actual_timing
             and actual_timing["relative_actual_pts_s"] == [row["actual_timestamp_s"] for row in trace["records"]],
             "record 실제 video PTS가 trace와 다르다")
    for number in (1, 2):
        measurement = _read(paths[f"measurement_{number}"])
        play_id = f"{job['job_id']}-w{number}"
        _require(measurement == {"schema": RECORD.MEASUREMENT_SCHEMA, "play_run_id": play_id,
                                "video_sha256": _sha(video), "checkpoint_sha256": job["checkpoint"]["sha256"],
                                "command_name": "base_velocity", "coordinate_frames": {"command": "body", "motion": "body"},
                                "samples": _samples(trace, list(RECORD.WINDOWS[number - 1]))},
                 "record 측정값이 trace에서 재계산한 수치와 다르다")
        spec = _read(paths[f"spec_{number}"])
        _require(spec == {"schema": RECORD.PREPARE_SCHEMA, "play_run_id": play_id,
                         "source_lineage_id": f"{job['job_id']}-same-play", "parent_source_lineage_id": None,
                         "video": {"path": str(PurePosixPath(job["output_directory"]) / "same_play.mp4"),
                                   "logical_id": "same_play_video", "play_run_id": play_id},
                         "checkpoint": {"path": job["checkpoint"]["path"], "logical_id": job["checkpoint"]["logical_id"],
                                        "play_run_id": play_id},
                         "frames": [{"timestamp_s": value, "play_run_id": play_id, "video_sha256": _sha(video)}
                                    for value in RECORD.WINDOWS[number - 1]],
                         "measurement": {"path": str(PurePosixPath(job["output_directory"]) / "evidence" / paths[f"measurement_{number}"].name),
                                         "logical_id": f"same_play_measurements_w{number}", "play_run_id": play_id},
                         "play": {"task": RECORD.TASK, "camera": RECORD.fixed_profile()["camera"]["identifier"],
                                  "command_name": "base_velocity", "command_frame": "body", "motion_frame": "body",
                                  "settings_sha256": RECORD.digest(RECORD.fixed_profile())}},
                 "record V01 spec의 원격 provenance가 다르다")
    return trace


def _verify_resource(repo: Path, manifest: dict, protocol: dict) -> dict:
    """보관된 proof의 발급 시점 유효성을 검사한다. 새 실행 권한을 발급하지 않는다."""
    proof = _read(_ref(repo, manifest["resource_parent"]))
    local = dict(proof)
    mappings = {"raw_gpu_receipt": "live_lease_proof", "budget_admission": "budget_admission",
                "provider_inventory": "provider_inventory", "nvidia_gpu_csv": "gpu_csv",
                "nvidia_process_csv": "process_csv"}
    for field, reference_name in mappings.items():
        actual = _ref(repo, manifest[reference_name])
        _require(proof.get(f"{field}_sha256") == _sha(actual), "보관 자원 proof의 원문 SHA가 다르다")
        local[f"{field}_path"] = str(actual)
    # 원문 proof를 덮어쓰지 않고 portable 위치 표만 기존 읽기 전용 검증기에 전달한다.
    context = SimpleNamespace(protocol=protocol, protocol_sha=protocol["_verified_sha256_lf"],
                              now=lambda: SOURCE._stamp(proof["issued_at_utc"]))
    SOURCE.SourceBatch._verify_resource(context, local)  # type: ignore[arg-type]
    return proof


def _verify_training_action(repo: Path, references: dict, protocol: dict, resource_path: Path,
                            artifact: dict, training: dict, capture: dict) -> str:
    _exact(references, {"intent", "raw", "receipt", "runtime_manifest", "freshness"}, "source 학습 action 위치")
    paths = {key: _ref(repo, value) for key, value in references.items()}
    actions = [(i, row) for i, row in enumerate(protocol["source_action_order"])
               if row["kind"] == "train" and row["job_id"] == training["job_id"]]
    _require(len(actions) == 1, "source 학습 action이 유일하지 않다")
    ordinal, action = actions[0]
    intent = _exact(_read(paths["intent"]), {"schema", "action_id", "kind", "ordinal", "protocol_sha256_lf",
                                           "resource_sha256", "cost_before_usd", "started_at_utc", "reserved_max_wall_s",
                                           "reserved_max_compute_usd", "cost_scope"}, "source 학습 intent")
    limit = protocol["resources"]["action_limits"]["train"]
    expected = {"schema": SOURCE.INTENT_SCHEMA, "action_id": action["action_id"], "kind": "train", "ordinal": ordinal,
                "protocol_sha256_lf": protocol["_verified_sha256_lf"], "resource_sha256": _sha(resource_path),
                "reserved_max_wall_s": limit["max_wall_s"], "reserved_max_compute_usd": limit["max_compute_usd"],
                "cost_scope": protocol["resources"]["cost_scope"]}
    _require(_same({key: intent[key] for key in expected}, expected), "source 학습 intent가 사전등록과 다르다")
    raw = _exact(_read(paths["raw"]), {"kind", "action_id", "artifact", "job_id"}, "source 학습 raw")
    _require(raw["kind"] == "train" and raw["action_id"] == action["action_id"]
             and raw["job_id"] == training["job_id"] and _same(raw["artifact"], artifact), "source 학습 artifact와 raw가 다르다")
    _exact(artifact, {"checkpoint", "telemetry", "seed", "arm", "original_faults", "applied_corrections",
                      "readback", "execution", "manifest_path", "manifest_sha256", "manifest",
                      "source_checkpoint_sha256"}, "source 학습 artifact")
    _require(isinstance(artifact["manifest"], dict) and isinstance(artifact["readback"], dict)
             and isinstance(artifact["execution"], dict)
             and artifact["execution"].get("completed") is True
             and artifact["execution"].get("exit_code") == 0
             and artifact["execution"].get("timed_out") is False
             and artifact["execution"].get("cleanup_verified") is True,
             "source 학습 artifact의 실행 완료 증거가 없다")
    receipt = _exact(_read(paths["receipt"]), {"schema", "action_id", "intent_sha256", "raw_sha256",
                                              "cost_after_usd", "finished_at_utc"}, "source 학습 receipt")
    _require(receipt["schema"] == SOURCE.RECEIPT_SCHEMA and receipt["action_id"] == action["action_id"]
             and receipt["intent_sha256"] == _sha(paths["intent"]) and receipt["raw_sha256"] == _sha(paths["raw"]),
             "source 학습 receipt의 바이트 결합이 다르다")
    resource = _read(resource_path)
    started, finished = SOURCE._stamp(intent["started_at_utc"]), SOURCE._stamp(receipt["finished_at_utc"])
    compute_started = SOURCE._stamp(resource["compute_started_at_utc"])
    quote = protocol["resources"]["compute_quote_usd_per_h"]
    cost_before = (started - compute_started).total_seconds() * quote / 3600
    cost_after = (finished - compute_started).total_seconds() * quote / 3600
    _require(0 <= (finished - started).total_seconds() <= limit["max_wall_s"]
             and finished <= SOURCE._stamp(protocol["resources"]["deadline_utc"])
             and 0 <= cost_before <= cost_after <= protocol["resources"]["whole_batch_budget_usd"]
             and abs(RECORD._finite(intent["cost_before_usd"], "cost before") - cost_before) <= 1e-9
             and abs(RECORD._finite(receipt["cost_after_usd"], "cost after") - cost_after) <= 1e-9,
             "source 학습 시각·비용 영수증이 다르다")
    runtime = _exact(_read(paths["runtime_manifest"]), {"schema", "protocol_id", "protocol_path", "protocol_sha256_lf",
                                                       "source_anchor", "resource_path_sha256", "training_plan_sha256",
                                                       "source_action_order_sha256", "freshness_sha256"}, "source runtime manifest")
    expected_runtime = {"schema": SOURCE.MANIFEST_SCHEMA, "protocol_id": protocol["protocol_id"],
                        "protocol_path": capture["source_snapshot"]["protocol"]["path"],
                        "protocol_sha256_lf": protocol["_verified_sha256_lf"], "source_anchor": capture["job"]["source"]["commit"],
                        "resource_path_sha256": _sha(resource_path), "training_plan_sha256": SOURCE.digest(protocol["training_plan"]),
                        "source_action_order_sha256": SOURCE.digest(protocol["source_action_order"])}
    _require(_same({key: runtime[key] for key in expected_runtime}, expected_runtime)
             and isinstance(runtime["freshness_sha256"], str) and HEX64.fullmatch(runtime["freshness_sha256"]) is not None,
             "source runtime manifest가 실제 학습 action과 다르다")
    freshness = _exact(_read(paths["freshness"]), {"schema", "status", "seeds", "checked_at_utc", "checked_roots",
                                                 "checked_file_count", "evidence", "scope"}, "source seed freshness")
    seeds = sorted({row["seed"] for row in protocol["training_plan"]})
    _require(runtime["freshness_sha256"] == _sha(paths["freshness"])
             and freshness["schema"] == "preregistered_seed_freshness_v1" and freshness["status"] == "fresh"
             and _same(freshness["seeds"], seeds) and freshness["evidence"] == []
             and SOURCE._stamp(freshness["checked_at_utc"]) <= started
             and type(freshness["checked_file_count"]) is int and freshness["checked_file_count"] >= 0
             and isinstance(freshness["checked_roots"], list) and bool(freshness["checked_roots"])
             and all(isinstance(value, str) and value for value in freshness["checked_roots"])
             and isinstance(freshness["scope"], str) and bool(freshness["scope"]),
             "source seed freshness 원문이 학습 사전등록과 다르다")
    return _sha(paths["receipt"])


def verify_model_readiness(repo: Path, path: Path, source_protocol: dict) -> dict:
    """제한된 renderer 조합의 readiness를 실제 기록에서 재계산한다."""
    manifest = _exact(_read(path), {"schema", "source_training_artifact", "training_action", "resource_parent",
                                  "live_lease_proof", "budget_admission", "provider_inventory", "gpu_csv", "process_csv",
                                  "record", "smoke", "receipt"}, "model readiness")
    _require(manifest["schema"] == READINESS_SCHEMA, "model readiness schema가 다르다")
    resource_parent = _ref(repo, manifest["resource_parent"])
    live_proof = _ref(repo, manifest["live_lease_proof"])
    resource = _verify_resource(repo, manifest, source_protocol)
    record = _verify_capture(repo, manifest["record"], RECORD.RECORD_MODE, source_protocol, _sha(resource_parent), _sha(live_proof), resource)
    smoke = _verify_capture(repo, manifest["smoke"], RECORD.RENDER_SYNC_SMOKE_MODE, source_protocol, _sha(resource_parent), _sha(live_proof), resource)
    job = record["job"]
    _verify_record_anchor(repo, record)
    _require(job["source"] == smoke["job"]["source"] and job["resource"] == smoke["job"]["resource"]
             and job["checkpoint"] == smoke["job"]["checkpoint"]
             and record["source_snapshot"] == smoke["source_snapshot"]
             and record["fingerprint"] == smoke["fingerprint"], "record/smoke의 source·자원·runtime 조합이 다르다")
    trace = _verify_record(record)
    _verify_smoke(smoke)
    artifact_path = _ref(repo, manifest["source_training_artifact"])
    artifact = _read(artifact_path)
    actions = [row for row in source_protocol["source_action_order"]
               if row["kind"] == "record" and row["job_id"] == job["job_id"]]
    _require(len(actions) == 1, "record가 등록 source action이 아니다")
    plan = [row for row in source_protocol["training_plan"] if row["job_id"] == actions[0]["source_job_id"]]
    _require(len(plan) == 1 and plan[0]["arm"] in {"NONE", "F1"}, "record source가 등록 source 학습이 아니다")
    training = plan[0]
    _require(artifact.get("arm") == training["arm"] and type(artifact.get("seed")) is int
             and artifact["seed"] == training["seed"]
             and artifact.get("checkpoint", {}).get("sha256") == job["checkpoint"]["sha256"]
             and job["checkpoint"]["logical_id"] == actions[0]["checkpoint_logical_id"],
             "source 학습 seed·arm·checkpoint 결합이 다르다")
    training_receipt = _verify_training_action(repo, manifest["training_action"], source_protocol, resource_parent,
                                              artifact, training, record)
    expected_receipt = {"schema": "vision_selection_readiness_receipt_v1",
                        "scope": "registered_renderer_combination_only", "model_use_ready": True,
                        "record_result_sha256": _sha(record["paths"]["result"]),
                        "smoke_result_sha256": _sha(smoke["paths"]["result"]),
                        "smoke_report_sha256": _sha(smoke["artifact_paths"]["render_sync_smoke"]),
                        "record_context_sha256": _sha(record["paths"]["context"]),
                        "smoke_context_sha256": _sha(smoke["paths"]["context"]),
                        "record_attestation_sha256": _sha(record["paths"]["attestation"]),
                        "smoke_attestation_sha256": _sha(smoke["paths"]["attestation"]),
                        "record_job_sha256": _sha(record["paths"]["job"]),
                        "smoke_job_sha256": _sha(smoke["paths"]["job"]),
                        "resource_proof_sha256": _sha(resource_parent), "live_lease_proof_sha256": _sha(live_proof),
                        "training_action_receipt_sha256": training_receipt,
                        "source_protocol_sha256_lf": source_protocol["_verified_sha256_lf"],
                        "universal_zero_lag_claim": False, "smoke_raw_state_recomputed": False}
    _require(_same(manifest["receipt"], expected_receipt), "readiness receipt가 재계산한 결합과 다르다")
    return {"trace_sha256": _sha(record["artifact_paths"]["trace"]), "checkpoint_sha256": trace["checkpoint_sha256"],
            "video_sha256": trace["video_sha256"], "source_seed": training["seed"],
            "measurement_sha256": {f"window-{i:02d}": _sha(record["artifact_paths"][f"measurement_{i}"]) for i in (1, 2)},
            "source_role": "normal" if training["arm"] == "NONE" else "registered-f1"}


def verify_inputs(repo_root: Path, execution_protocol: dict[str, Any], input_locations_path: Path) -> dict:
    """공개 24행을 실제 원본·렌더 확인·동결 payload와 대조한다."""
    repo = Path(repo_root).resolve(strict=True)
    locations_path = _regular(Path(input_locations_path))
    locations_path.relative_to(repo)
    locations = _exact(_read(locations_path), {"schema", "execution_id", "rows"}, "입력 위치")
    execution_id = execution_protocol["execution_id"]
    _require(locations["schema"] == LOCATIONS_SCHEMA and locations["execution_id"] == execution_id,
             "입력 위치의 실행 결합이 다르다")
    _require(_sha(locations_path) == execution_protocol["input_locations_sha256"],
             "입력 위치 SHA가 공개 사전등록과 다르다")
    method_sha = execution_protocol["method_protocol"]["sha256_lf"]
    source_sha = execution_protocol["source_protocol"]["sha256_lf"]
    public_rows = COMPARISON.validate_inputs(execution_protocol["rows"],
                                           method_protocol_sha256=method_sha,
                                           source_protocol_sha256=source_sha)
    _require(isinstance(locations["rows"], list) and len(locations["rows"]) == len(public_rows),
             "입력 위치의 row 수가 다르다")
    source_protocol = _read(_relative(repo, execution_protocol["source_protocol"]["path"]))
    source_protocol = SOURCE.validate_protocol(source_protocol)
    _require(source_protocol.get("schema") == "vision_brev_source_protocol_v1", "source protocol schema가 다르다")
    source_protocol_path = _relative(repo, execution_protocol["source_protocol"]["path"])
    _require(hashlib.sha256(source_protocol_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == source_sha,
             "source protocol 실제 LF SHA가 다르다")
    source_protocol["_verified_sha256_lf"] = source_sha
    rows, manifest_hashes = [], {}
    proof_cache: dict[str, dict] = {}
    for public, location in zip(public_rows, locations["rows"], strict=True):
        _exact(location, {"call_id", "input_manifest", "capsule"}, "입력 위치 row")
        _require(location["call_id"] == public["call_id"], "입력 위치 순서가 다르다")
        manifest_path = _ref(repo, location["input_manifest"], frozen=True)
        manifest = _exact(_read(manifest_path), {"schema", "snapshot", "contract", "trace",
                                               "camera_pose_audit", "model_readiness"}, "입력 원본 manifest")
        _require(manifest["schema"] == INPUT_SCHEMA, "입력 원본 schema가 다르다")
        paths = {key: _ref(repo, manifest[key]) for key in manifest if key != "schema"}
        for key, public_key in (("snapshot", "snapshot_sha256"), ("contract", "contract_sha256"),
                                ("trace", "trace_sha256"), ("camera_pose_audit", "camera_pose_audit_sha256")):
            _require(_sha(paths[key]) == public[public_key], f"{key} 공개 SHA가 다르다")
        proof_key = _sha(paths["model_readiness"])
        if proof_key not in proof_cache:
            proof_cache[proof_key] = verify_model_readiness(repo, paths["model_readiness"], source_protocol)
        proof = proof_cache[proof_key]
        _require(proof["trace_sha256"] == public["trace_sha256"]
                 and proof["checkpoint_sha256"] == public["checkpoint_sha256"]
                 and proof["video_sha256"] == public["video_sha256"]
                 and proof["source_seed"] == public["source_seed"]
                 and proof["source_role"] == public["source_role"], "원본 readiness와 공개 row가 다르다")
        audit_key = public["trace_sha256"]
        if audit_key not in proof_cache:
            proof_cache[audit_key] = camera_pose_audit(paths["trace"])
        _require(_read(paths["camera_pose_audit"]) == proof_cache[audit_key], "camera 감사가 원본 trace와 다르다")
        snapshot = _read(paths["snapshot"])
        contract = CONTRACT.VISION.verify(paths["contract"])
        provenance = contract["provenance"]
        _require(provenance["source_lineage_id"] == public["source_lineage_id"]
                 and provenance["parent_source_lineage_id"] == public["parent_lineage_id"]
                 and provenance["checkpoint"]["sha256"] == public["checkpoint_sha256"]
                 and provenance["video"]["sha256"] == public["video_sha256"], "V01 공개 계보가 다르다")
        _require(provenance["measurement"] is not None
                 and provenance["measurement"]["sha256"] == proof["measurement_sha256"][public["window"]],
                 "V01 measurement 원본 SHA가 record artifact와 다르다")
        _verify_window(contract, _read(paths["trace"]), public["window"])
        capsule_path = _ref(repo, location["capsule"], frozen=True)
        _require(_sha(capsule_path) == public["capsule_sha256"], "공개 capsule SHA가 다르다")
        capsule_ref = {"path": str(capsule_path), "sha256": public["capsule_sha256"], "status": "frozen"}
        frozen = CALL.prepared(capsule_ref)
        _require(frozen["capsule"]["messages_sha256"] == public["messages_sha256"]
                 and frozen["capsule"]["method_sha256"] == public["capsule_method_raw_sha256"]
                 and frozen["view_kind"] == public["view_kind"], "공개 capsule method/view 결합이 다르다")
        for relative, sha in public["capsule_method_raw_sha256"].items():
            _require(_sha(_relative(repo, relative)) == sha, "현재 capsule method bytes가 다르다")
        binding = {"contract_sha256": public["contract_sha256"], "source_lineage_id": public["source_lineage_id"]}
        CONTRACT.verify(frozen["bundle"], snapshot, paths["contract"], binding)
        _require(CONTRACT.messages(frozen["bundle"], snapshot, paths["contract"], binding) == frozen["messages"],
                 "동결 messages가 원본에서 재계산한 payload와 다르다")
        rows.append({"call_id": public["call_id"], "run_id": f"vision-selection-{execution_id}-{public['call_id']}",
                     "capsule": capsule_ref})
        manifest_hashes[public["call_id"]] = _sha(manifest_path)
    return {"schema": VERIFIED_SCHEMA, "execution_id": execution_id, "rows": rows,
            "input_manifest_sha256": manifest_hashes,
            "source_sha256_lf": execution_protocol["source_sha256_lf"],
            "evidence_sha256_lf": execution_protocol["evidence_sha256_lf"],
            "input_locations_sha256": _sha(locations_path)}
