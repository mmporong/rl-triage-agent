"""회수된 기술 녹화를 기존 V01/V02/V03 CPU 계약으로 준비한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(folder))

import grounded_vision_observer_v1 as OBSERVER
import vision_evidence_contract_v1 as V01
import vision_pull_record_source_readiness_v5 as LOCAL
import vision_pull_record_worker_v3 as WORKER
import vision_selection_contract_v1 as SELECT
import vision_selection_sources_v1 as SOURCES

FLAGS = ("capture_approved", "model_ready", "used_for_decision")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": V01.sha256(path)}


def transfer_gate(receipt: dict) -> None:
    require(receipt.get("all_match") is True and type(receipt.get("files_verified")) is int
            and receipt["files_verified"] == 3865, "회수 SHA 검증 게이트 미통과")


def original_flags(value: dict) -> dict:
    require(all(value.get(key) is False for key in FLAGS), "원본 사용 승인 경계가 다르다")
    return {key: value[key] for key in FLAGS}


def local_spec(original: dict, video: Path, checkpoint: Path, measurement: Path) -> dict:
    translated = LOCAL._translated_spec(original, video, checkpoint, measurement)
    restored = deepcopy(translated)
    for key in ("video", "checkpoint", "measurement"):
        restored[key]["path"] = original[key]["path"]
    require(restored == original, "경로 외 spec 내용이 달라졌다")
    values = V01.read_object(measurement)
    V01._measurement(values, run_id=translated["play_run_id"],
                     video_sha=V01.sha256(video), checkpoint_sha=V01.sha256(checkpoint),
                     play=translated["play"],
                     frame_timestamps=[row["timestamp_s"] for row in translated["frames"]])
    return translated


def motion_summary(trace: dict) -> dict:
    """동일 Play 운동을 기술 통계로 남기며 행동 임계값은 만들지 않는다."""
    rows = [r for r in trace["records"] if 1.0 <= r["actual_timestamp_s"] <= 12.0]
    states = [r["state"] for r in rows]
    count = len(states)
    start, end = states[0]["root_position_world_m"], states[-1]["root_position_world_m"]
    return {
        "source": "same_play_simulator_state", "samples": count,
        "interval_s": [rows[0]["actual_timestamp_s"], rows[-1]["actual_timestamp_s"]],
        "command_x_mps": sorted({s["command"][0] for s in states}),
        "mean_body_velocity_x_mps": sum(s["motion"][0] for s in states) / count,
        "velocity_xy_rmse_mps": math.sqrt(sum(
            sum((s["motion"][i] - s["command"][i]) ** 2 for i in (0, 1)) for s in states) / count),
        "world_xy_displacement_m": math.hypot(end[0] - start[0], end[1] - start[1]),
        "minimum_up_axis_z": min(s["pose"]["up_axis_z"] for s in states),
        "height_range_m": [min(s["pose"]["base_height_m"] for s in states),
                           max(s["pose"]["base_height_m"] for s in states)],
        "behavior_label": None,
        "label_hold_reason": "같은 Play 행동 정답의 등록 임계값을 확인하지 못함",
        "fixed_evaluation_used_as_behavior_ground_truth": False,
    }


def snapshot(trace: dict, reference_trace: dict, ledger: Path) -> dict:
    controller = SELECT.CONTROLLER
    binding = lambda value: {"checkpoint_sha256": value["checkpoint_sha256"],
                             "telemetry_sha256": V01.digest(value)}
    packet = controller.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": trace["profile"]["action_scale"]},
        input_bindings={"case": binding(trace), "reference": binding(reference_trace)},
        execution_budget_s=1200)
    protocol = controller.V.PROTOCOL
    protocol_sha = hashlib.sha256(protocol.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return controller.V.build_snapshot(packet, "capture_dev", "model_adaptive", protocol_sha, ledger)


def prepare_pair(retrieved: Path, checkpoints: Path, output: Path, *, owned_root: Path) -> dict:
    output = output.resolve()
    output.relative_to(owned_root.resolve())
    retrieved = retrieved.resolve(strict=True)
    checkpoints = checkpoints.resolve(strict=True)
    gate_path = retrieved / "local-copy-verification.json"
    gate = V01.read_object(gate_path)
    transfer_gate(gate)
    protocol_path = ROOT / "bench/protocols/vision_transferred_pull_record_20261027b_v3.json"
    protocol = V01.read_object(protocol_path)
    policies = retrieved / "rl-triage-transferred-record-state-20261027b-v3/policies"
    reference_trace = V01.read_object(policies / "source-s2220-none/output/evidence/raw_trace.json")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "vision_transferred_cpu_input_preparation_v1", "task_id": "RI_CAPTURE_ACCEPT_01",
               "attempt": 1, "contract_revision": 1, "status": "HOLD",
               "scope": "development_cpu_input_preparation_only", "split": "development",
               "source_seed": 2220, "transfer_verification": reference(gate_path),
               "protocol": reference(protocol_path), "policies": [], "model_api_requests": 0,
               "gpu_executions": 0, "training_actions": 0, "used_for_decision": False,
               "model_use_authorized": False, "independent_acceptance": False,
               "snapshot_binding_scope": "same_play_trace_digest; training telemetry not supplied",
               "hold_reasons": ["별도 capture/model 사용 승인 미완료", "같은 Play 행동 정답 임계값 미확정",
                                "정식 비교의 독립 seed 두 쌍 중 seed2220 한 쌍만 준비됨"],
               "next_fix": "총괄이 이 CPU receipt와 원본 AABA·RGB 근거를 검수해 capture 사용 범위를 결정한다"}
    for cp in protocol["checkpoints"]:
        logical = cp["logical_id"]
        stage = policies / logical
        destination = output / logical
        destination.mkdir()
        checkpoint = checkpoints / cp["isaaclab_relative"]
        require(V01.sha256(checkpoint) == cp["sha256"], "checkpoint SHA 불일치")
        files = {"video": stage / "output/same_play.mp4", "raw_trace": stage / "output/evidence/raw_trace.json"}
        for i in (1, 2):
            files[f"measurement_{i}"] = stage / f"output/evidence/window_{i}_measurements.json"
            files[f"spec_{i}"] = stage / f"output/evidence/window_{i}_v01_spec.json"
        for key in ("encoder_receipt", "timing_receipt"):
            files[key] = stage / f"output/evidence/{key}.json"
        source_files = {**files, "checkpoint": checkpoint, "job": stage / "job.json",
                        "execution": stage / "execution.json", "result": stage / "output/result.json",
                        "manifest": stage / "output/native_capture_manifest.json",
                        "aaba": stage / "aaba-output/native-aaba-preflight.report.json"}
        before = {key: reference(path) for key, path in source_files.items()}
        trace = V01.read_object(files["raw_trace"])
        require(trace["checkpoint_sha256"] == cp["sha256"] and
                trace["video_sha256"] == V01.sha256(files["video"]), "trace source SHA 불일치")
        require(trace["profile"] == protocol["record_profile"]["value"], "Play profile 불일치")
        execution = V01.read_object(source_files["execution"])
        require(execution["completed"] is True and execution["cleanup_verified"] is True
                and execution["exit_code"] == 0, "원본 실행 미완료")
        flags = {key: original_flags(V01.read_object(source_files[key]))
                 for key in ("execution", "result", "manifest")}
        context = {"job_path": source_files["job"], "checkpoint_sha256": cp["sha256"],
                   "job": V01.read_object(source_files["job"])}
        require(execution["result_sha256"] == V01.sha256(source_files["result"])
                and execution["job_sha256"] == V01.sha256(source_files["job"]), "원본 실행 SHA 결합 불일치")
        require(WORKER.PREVIOUS.RECORD_VERIFY._record_artifacts(context, files),
                "기존 8개 record artifact 검증 실패")
        require(WORKER._manifest(context, files) == V01.read_object(source_files["manifest"]),
                "기존 native manifest 재검증 불일치")
        camera = SOURCES.camera_pose_audit(files["raw_trace"])
        V01.write_new(destination / "camera_pose_audit.json", camera)
        aaba = V01.read_object(source_files["aaba"])
        aaba_raw = LOCAL._aaba_raw(ROOT, source_files["aaba"], aaba)
        require(aaba["worker"]["validation"]["approved"] is True
                and aaba["worker"]["policy_probe"]["checkpoint_sha256"] == cp["sha256"], "AABA 결합 불일치")
        for index, record in enumerate(trace["records"]):
            with np.load(stage / f"job.pull-record-{index:06d}.npz", allow_pickle=False) as archive:
                rgb = archive["rgb"]
                require(rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
                        and hashlib.sha256(rgb.tobytes()).hexdigest() == record["rgb_sha256"], "raw RGB 결합 불일치")
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
                                "-of", "json", str(files["video"])], check=True, capture_output=True, text=True)
        timing = json.loads(probe.stdout)
        stream = timing["streams"][0]
        require(stream["width"] == 640 and stream["height"] == 480 and stream["r_frame_rate"] == "50/1"
                and stream["nb_read_frames"] == "625" and float(timing["format"]["duration"]) == 12.5,
                "영상 규격 불일치")
        points = V01._source_frame_pts(files["video"])
        require(points == [r["actual_timestamp_s"] for r in trace["records"]], "실제 영상 PTS/trace 불일치")
        state = snapshot(trace, reference_trace, destination / "unexecuted-probes.jsonl")
        V01.write_new(destination / "snapshot.json", state)
        row = {"logical_id": logical, "source_inputs": before, "original_flags": flags,
               "raw_rgb_frames_verified": len(trace["records"]), "record_artifacts_verified": True,
               "video_probe": timing,
               "aaba_raw_verified": aaba_raw, "same_play_motion": motion_summary(trace), "windows": []}
        for i in (1, 2):
            spec = local_spec(V01.read_object(files[f"spec_{i}"]), files["video"], checkpoint,
                              files[f"measurement_{i}"])
            spec_path = destination / f"window_{i}_local_spec.json"
            V01.write_new(spec_path, spec)
            contract_path = V01.prepare(spec_path, destination / f"window_{i}")
            contract = V01.verify(contract_path)
            SOURCES._verify_window(contract, trace, f"window-0{i}")
            window = {"window": i, "contract": reference(contract_path), "split": "development",
                      "source_lineage_id": contract["provenance"]["source_lineage_id"], "views": {}}
            binding = {"contract_sha256": V01.sha256(contract_path),
                       "source_lineage_id": contract["provenance"]["source_lineage_id"]}
            observer = OBSERVER.request_view(contract_path)
            LOCAL._assert_public_view_safe(observer["messages"])
            observer_path = destination / f"window_{i}/v02_request.json"
            V01.write_new(observer_path, observer)
            window["v02_request"] = reference(observer_path)
            base_digests = set()
            for view in sorted(SELECT.VIEW_KINDS):
                bundle = SELECT.prepare(state, contract_path, binding, view)
                base_digests.add(bundle["operational_base_digest"])
                messages = SELECT.messages(bundle, state, contract_path, binding)
                LOCAL._assert_public_view_safe(messages)
                bundle_path = destination / f"window_{i}/v03_{view}_bundle.json"
                messages_path = destination / f"window_{i}/v03_{view}_messages.json"
                V01.write_new(bundle_path, bundle)
                V01.write_new(messages_path, messages)
                window["views"][view] = {"bundle": reference(bundle_path), "messages": reference(messages_path),
                                         "image_count": sum(p.get("type") == "image_url"
                                             for m in messages for p in m["content"] if isinstance(p, dict))}
                require(window["views"][view]["image_count"] == (0 if view == "text_only" else 6),
                        "view 이미지 수가 계약과 다르다")
            require(len(base_digests) == 1, "세 view의 운영 상태가 다르다")
            row["windows"].append(window)
        require(before == {key: reference(path) for key, path in source_files.items()}, "원본 변경 감지")
        row["source_inputs_unchanged"] = True
        receipt["policies"].append(row)
        V01.write_new(destination / "cpu_receipt.json", row)
    receipt["cpu_preparation"] = "PASS"
    receipt["v01_contracts"] = 4
    receipt["v02_requests"] = 4
    receipt["v03_payloads"] = 12
    V01.write_new(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieved", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--owned-root", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_pair(args.retrieved, args.checkpoints, args.output, owned_root=args.owned_root)
    print(json.dumps({k: result[k] for k in ("status", "cpu_preparation", "v01_contracts", "v02_requests", "v03_payloads")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
