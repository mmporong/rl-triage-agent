"""고정 평가 뒤 VL 보조 관찰에 쓸 Play 클립 하나를 비공개 경로에 기록한다."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import record_evidence_media as M


PLAY_RELATIVE = Path("scripts/reinforcement_learning/rsl_rl/play.py")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FIXED_PLAYBACK = {
    "schema": "vl_aux_play_recording_contract_v1",
    "task": "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    "seed": 42,
    "num_envs": 16,
    "video_length": 300,
    "kit_args": "--/app/vulkan=false --/app/window/hideUi=true",
    "action_scale": 0.25,
    "overrides": [
        "env.actions.joint_pos.scale=0.25",
        "env.viewer.eye=[3.0,3.0,2.0]",
        "env.viewer.lookat=[0.0,0.0,0.35]",
    ],
    "video_contract": M.EXPECTED_VIDEO_CONTRACT,
    "gpu_admission": M.EXPECTED_GPU_ADMISSION,
    "timeout_s": 900,
    "scope": "auxiliary_visual_observation_only",
    "changes_fixed_evaluation": False,
    "used_for_recovery_decision": False,
}


class RecordingContractError(RuntimeError):
    """입력 결합 또는 실행 경계가 계약과 다름."""


def _sha(path: Path) -> str:
    return M.sha256_file(path)


def _write(path: Path, value: dict[str, Any]) -> None:
    M._write_new_json(path, value)


def build_recording_contract(isaaclab: Path) -> dict[str, Any]:
    """Isaac Lab Play 소스 SHA를 포함한 CPU 검증 가능 계약을 만든다."""
    isaaclab = Path(isaaclab).resolve()
    play = isaaclab / PLAY_RELATIVE
    if not play.is_file():
        raise RecordingContractError("Play source is missing")
    contract = deepcopy(FIXED_PLAYBACK)
    contract["play_source"] = {
        "relative_path": PLAY_RELATIVE.as_posix(),
        "sha256": _sha(play),
    }
    return contract


def _path_binding(artifact: dict[str, Any], key: str) -> tuple[Path, str]:
    binding = artifact.get(key)
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise RecordingContractError(f"artifact {key} binding is invalid")
    path = Path(binding["path"]).resolve() if isinstance(binding["path"], (str, Path)) else None
    expected = binding["sha256"]
    if path is None or not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        raise RecordingContractError(f"artifact {key} binding is invalid")
    if not path.is_file() or _sha(path) != expected:
        raise RecordingContractError(f"artifact {key} digest differs")
    return path, expected


def _fixed_checkpoint_sha(report: dict[str, Any]) -> str:
    checkpoint = report.get("checkpoint")
    if isinstance(checkpoint, dict):
        value = checkpoint.get("sha256")
    else:
        value = report.get("checkpoint_sha256")
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RecordingContractError("fixed evaluation checkpoint binding is invalid")
    return value


def _verify_contract(protocol: dict[str, Any], isaaclab: Path) -> None:
    expected = build_recording_contract(isaaclab)
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RecordingContractError(f"recording contract differs at {key}")
    if set(protocol) - set(expected) != {"fixed_evaluation_label"}:
        raise RecordingContractError("recording contract has unregistered fields")
    if protocol["fixed_evaluation_label"] not in {"healthy", "unhealthy", "unknown"}:
        raise RecordingContractError("fixed evaluation label is invalid")


def record_clip(
    artifact: dict[str, Any],
    fixed_report: Path,
    output: Path,
    isaaclab: Path,
    protocol: dict[str, Any],
) -> dict[str, str]:
    """검증된 정책 한 개를 기록한다. 실패한 출력 경로는 재사용하지 않는다."""
    fixed_report = Path(fixed_report).resolve()
    output = Path(output).resolve()
    isaaclab = Path(isaaclab).resolve()
    _verify_contract(protocol, isaaclab)

    checkpoint, checkpoint_sha = _path_binding(artifact, "checkpoint")
    environment, environment_sha = _path_binding(artifact, "environment")
    agent, agent_sha = _path_binding(artifact, "agent")
    params = environment.parent
    if agent.parent != params or params.name != "params" or checkpoint.parent != params.parent:
        raise RecordingContractError("checkpoint and parameter files do not share one run directory")
    if environment.name != "env.yaml" or agent.name != "agent.yaml":
        raise RecordingContractError("parameter file names differ")
    if not fixed_report.is_file():
        raise RecordingContractError("fixed evaluation report is missing")
    report = json.loads(fixed_report.read_text(encoding="utf-8"))
    if _fixed_checkpoint_sha(report) != checkpoint_sha:
        raise RecordingContractError("fixed evaluation and artifact checkpoint differ")
    if report.get("action_scale") != protocol["action_scale"]:
        raise RecordingContractError("fixed evaluation action scale differs")

    source_hashes = {
        "checkpoint": checkpoint_sha,
        "environment": environment_sha,
        "agent": agent_sha,
        "fixed_report": _sha(fixed_report),
        "play_source": protocol["play_source"]["sha256"],
    }
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "recording_contract.json", protocol)

    before = M.query_gpu(protocol["gpu_admission"]["policy"])
    _write(output / "gpu_before.json", before)
    if before.get("busy") is not False:
        _write(output / "failure.json", {"stage": "gpu_before", "retry_requires_new_output": True})
        raise RecordingContractError("GPU admission failed before recording")

    item = {
        "clip": {
            "id": "policy_play",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "params_env_sha256": environment_sha,
            "params_agent_sha256": agent_sha,
            "overrides": list(protocol["overrides"]),
        },
        "checkpoint": checkpoint,
        "params_dir": params,
    }
    recording_error: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        # 평가 라벨은 item에 넣지 않아 Play 또는 후속 영상 모델 입력으로 전달되지 않는다.
        result = M._record_clip(protocol, item, isaaclab, output, protocol["timeout_s"])
    except Exception as error:
        recording_error = error

    after_error: BaseException | None = None
    try:
        after = M.query_gpu(protocol["gpu_admission"]["policy"])
        _write(output / "gpu_after.json", after)
        if after.get("busy") is not False:
            raise RecordingContractError("GPU admission failed after recording")
    except Exception as error:
        after_error = error

    unchanged = {
        "checkpoint": _sha(checkpoint),
        "environment": _sha(environment),
        "agent": _sha(agent),
        "fixed_report": _sha(fixed_report),
        "play_source": _sha(isaaclab / PLAY_RELATIVE),
    }
    if unchanged != source_hashes:
        _write(output / "failure.json", {"stage": "source_recheck", "retry_requires_new_output": True})
        raise RecordingContractError("recording input changed during execution")
    if recording_error is not None:
        _write(output / "failure.json", {"stage": "recording", "retry_requires_new_output": True})
        raise recording_error
    if after_error is not None:
        _write(output / "failure.json", {"stage": "gpu_after", "retry_requires_new_output": True})
        raise after_error
    assert result is not None
    if result.get("own_pid_cleanup") is not True or result.get("residual_processes_observed") != 0:
        _write(output / "failure.json", {"stage": "cleanup", "retry_requires_new_output": True})
        raise RecordingContractError("owned process cleanup was not verified")

    video = Path(result["video"]).resolve()
    if not video.is_file() or output not in video.parents:
        raise RecordingContractError("recorded video path escaped output")
    manifest_path = output / "result_manifest.json"
    manifest = {
        "schema": "vl_aux_play_recording_result_v1",
        "scope": protocol["scope"],
        "fixed_evaluation_label": protocol["fixed_evaluation_label"],
        "label_forwarded_to_player_or_model": False,
        "used_for_recovery_decision": False,
        "source_sha256": source_hashes,
        "playback": {
            "task": protocol["task"],
            "seed": protocol["seed"],
            "num_envs": protocol["num_envs"],
            "video_length": protocol["video_length"],
            "action_scale": protocol["action_scale"],
            "overrides": protocol["overrides"],
        },
        "video": {
            "file": video.relative_to(output).as_posix(),
            "sha256": _sha(video),
            "probe": result["probe"],
        },
        "gpu_admission": {"before_busy": False, "after_busy": False},
        "owned_process_cleanup": True,
        "retry_requires_new_output": True,
    }
    _write(manifest_path, manifest)
    return {
        "video_path": str(video),
        "sha256": manifest["video"]["sha256"],
        "manifest_path": str(manifest_path),
    }
