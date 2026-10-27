"""여러 native-pull protocol의 capture·학습 입력을 한 계약으로 엄격히 수용한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import vision_transferred_input_acceptance_v1 as V1
import vision_transferred_input_acceptance_v2 as V2

V01, SELECT, WORKER = V1.V01, V1.SELECT, V1.WORKER
VIEWS = tuple(sorted(SELECT.VIEW_KINDS))
USE_SCOPE = ("registered_native_pull_combination_only", "first_probe_comparison_only")
MECHANICAL_DOMAINS = ("source_refs", "capture_artifacts", "raw_rgb", "actual_pts",
                      "training_archive", "telemetry")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_ref(value: Any, anchor: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{anchor} ref 형식")
    path = Path(value["path"]).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and V01.sha256(path) == value["sha256"],
            f"{anchor} 원문 SHA 불일치")
    return path, V01.read_object(path)


def _read_list_ref(value: Any, anchor: str) -> tuple[Path, list[Any]]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{anchor} ref 형식")
    path = Path(value["path"]).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and V01.sha256(path) == value["sha256"],
            f"{anchor} 원문 SHA 불일치")
    body = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(body, list), f"{anchor} JSON list가 아니다")
    return path, body


def _protocol(path: Path, seed: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    protocol = V01.read_object(path.resolve(strict=True))
    rows = protocol.get("checkpoints")
    require(isinstance(rows, list) and len(rows) == 2, "protocol checkpoint catalog가 두 건이 아니다")
    by_id = {row["logical_id"]: row for row in rows}
    require(len(by_id) == 2 and all(identity.startswith(f"source-s{seed}-") for identity in by_id),
            "protocol seed/catalog가 입력 seed와 다르다")
    profile = protocol.get("record_profile", {}).get("value")
    require(isinstance(profile, dict) and profile.get("frame_count") == 625 and profile.get("fps") == 50,
            "protocol 625 frame/50 fps 계약이 다르다")
    return protocol, by_id


def _refs_unchanged(refs: dict[str, Any], anchor: str) -> dict[str, dict[str, Any]]:
    checked = {}
    for name, value in refs.items():
        require(isinstance(value, dict) and set(value) == {"path", "sha256"},
                f"{anchor} {name} ref 형식")
        path = Path(value["path"]).resolve(strict=True)
        require(path.is_file() and not path.is_symlink() and V01.sha256(path) == value["sha256"],
                f"{anchor} {name} 원문 SHA 불일치")
        checked[name] = {"path": str(path), "sha256": V01.sha256(path)}
    return checked


def _capture_copy_gate(copy_verification_path: Path, capture_manifest_path: Path,
                       state_root: Path, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gate = V01.read_object(copy_verification_path.resolve(strict=True))
    manifest_path = capture_manifest_path.resolve(strict=True)
    manifest = V01.read_object(manifest_path)
    require(gate.get("all_match") is True and type(gate.get("files_verified")) is int,
            "capture copy verification이 완료되지 않았다")
    require(gate["files_verified"] == len(manifest.get("files", []))
            and gate.get("manifest_sha256") == V01.sha256(manifest_path),
            "capture copy manifest 결합이 다르다")
    require(manifest.get("root_basename") == state_root.name, "capture manifest root가 다르다")
    seen: set[Path] = set()
    for row in manifest["files"]:
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"},
                "capture manifest file row 형식이 다르다")
        path = (state_root / row["path"]).resolve(strict=True)
        path.relative_to(state_root)
        require(path not in seen and path.is_file() and not path.is_symlink()
                and path.stat().st_size == row["bytes"] and V01.sha256(path) == row["sha256"],
                "capture copy file SHA/크기가 다르다")
        seen.add(path)
    policy_rows = gate.get("policies")
    require(isinstance(policy_rows, list)
            and {row.get("logical_id") for row in policy_rows} == set(catalog),
            "capture copy policy catalog가 다르다")
    for row in policy_rows:
        require(row.get("raw_receipts") == 625 and row.get("npz_frames") == 625
                and row.get("completed") is True and row.get("cleanup_verified") is True
                and row.get("capture_approved") is False and row.get("model_ready") is False,
                "capture copy policy gate가 다르다")
    return {"copy_verification": {"path": str(copy_verification_path.resolve()),
                                  "sha256": V01.sha256(copy_verification_path)},
            "manifest": {"path": str(manifest_path), "sha256": V01.sha256(manifest_path)},
            "files_verified": len(seen), "all_match": True}


def _archive_material(archive_root: Path, summary: dict, catalog: dict[str, dict[str, Any]]) -> tuple[Path, dict[str, Path]]:
    tree = (archive_root / "tree").resolve(strict=True)
    inventory = V01.read_object((archive_root / "inventory.json").resolve(strict=True))
    rows = inventory.get("files")
    require(isinstance(rows, list), "archive inventory files가 없다")
    checkpoints: dict[str, Path] = {}
    for logical, checkpoint in catalog.items():
        matches = []
        suffix = checkpoint["isaaclab_relative"].replace("\\", "/")
        for row in rows:
            if row.get("path", "").replace("\\", "/").endswith(suffix) \
                    and row.get("sha256") == checkpoint["sha256"]:
                candidate = (tree / row["path"]).resolve(strict=True)
                candidate.relative_to(tree)
                matches.append(candidate)
        require(len(matches) == 1 and V01.sha256(matches[0]) == checkpoint["sha256"],
                f"archive checkpoint가 유일하지 않다: {logical}")
        checkpoints[logical] = matches[0]
    roots = []
    jobs = {logical.replace("source-", "vision-", 1) for logical in catalog}
    for row in rows:
        if not row.get("path", "").replace("\\", "/").endswith("/training_plan.json"):
            continue
        plan_path = (tree / row["path"]).resolve(strict=True)
        plan_path.relative_to(tree)
        candidate = plan_path.parent
        manifest_path, ledger_path = candidate / "manifest.json", candidate / "training.ledger.jsonl"
        if not manifest_path.is_file() or not ledger_path.is_file():
            continue
        manifest = V01.read_object(manifest_path)
        plan = V01.read_object(candidate / "training_plan.json")
        if (manifest.get("source_anchor") == summary.get("source_commit")
                and manifest.get("protocol_sha256_lf") == summary.get("source_protocol", {}).get("sha256_lf")
                and jobs <= {item.get("job_id") for item in plan.get("rows", [])}):
            roots.append(candidate)
    require(len(roots) == 1, "archive training source root가 유일하지 않다")
    return roots[0], checkpoints


def _prepare_capture(capture_state_root: Path, protocol_path: Path, catalog: dict[str, dict[str, Any]],
                     checkpoints: dict[str, Path], transfer: dict[str, Any], output: Path,
                     *, seed: int) -> dict[str, Any]:
    protocol = V01.read_object(protocol_path)
    policies_root = capture_state_root / "policies"
    require(policies_root.is_dir() and not policies_root.is_symlink(), "capture policies root가 없다")
    traces = {logical: V01.read_object(policies_root / logical / "output/evidence/raw_trace.json")
              for logical in catalog}
    baseline = next((logical for logical in catalog if logical.endswith("-none")), None)
    require(baseline is not None, "NONE baseline checkpoint가 없다")
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "vision_transferred_cpu_input_preparation_v3",
        "task_id": "RI_CAPTURE_ACCEPT_01", "status": "HOLD", "cpu_preparation": "PASS",
        "scope": "development_cpu_input_preparation_only", "split": "development",
        "source_seed": seed, "transfer_verification": transfer,
        "protocol": {"path": str(protocol_path.resolve()), "sha256": V01.sha256(protocol_path)},
        "policies": [], "model_api_requests": 0, "gpu_executions": 0,
        "training_actions": 0, "used_for_decision": False, "model_use_authorized": False,
        "independent_acceptance": False,
        "snapshot_binding_scope": "same_play_trace_digest; training telemetry supplied in paired receipt",
    }
    for logical, cp in catalog.items():
        stage = policies_root / logical
        destination = output / logical
        destination.mkdir()
        checkpoint = checkpoints[logical]
        files = {"video": stage / "output/same_play.mp4",
                 "raw_trace": stage / "output/evidence/raw_trace.json"}
        for index in (1, 2):
            files[f"measurement_{index}"] = stage / f"output/evidence/window_{index}_measurements.json"
            files[f"spec_{index}"] = stage / f"output/evidence/window_{index}_v01_spec.json"
        for key in ("encoder_receipt", "timing_receipt"):
            files[key] = stage / f"output/evidence/{key}.json"
        source_files = {**files, "checkpoint": checkpoint, "job": stage / "job.json",
                        "execution": stage / "execution.json", "result": stage / "output/result.json",
                        "manifest": stage / "output/native_capture_manifest.json",
                        "aaba": stage / "aaba-output/native-aaba-preflight.report.json"}
        before = {key: V1.reference(path) for key, path in source_files.items()}
        trace = traces[logical]
        require(len(trace.get("records", [])) == 625 and trace.get("profile") == protocol["record_profile"]["value"],
                "capture trace frame/profile이 다르다")
        require(trace.get("checkpoint_sha256") == cp["sha256"]
                and trace.get("video_sha256") == V01.sha256(files["video"]),
                "capture trace source 결합이 다르다")
        execution = V01.read_object(source_files["execution"])
        require(execution.get("completed") is True and execution.get("cleanup_verified") is True
                and execution.get("exit_code") == 0
                and execution.get("result_sha256") == V01.sha256(source_files["result"])
                and execution.get("job_sha256") == V01.sha256(source_files["job"]),
                "capture execution 결합이 다르다")
        flags = {name: V1.original_flags(V01.read_object(source_files[name]))
                 for name in ("execution", "result", "manifest")}
        context = {"job_path": source_files["job"], "checkpoint_sha256": cp["sha256"],
                   "job": V01.read_object(source_files["job"])}
        require(WORKER.PREVIOUS.RECORD_VERIFY._record_artifacts(context, files), "capture 8 artifact 검증 실패")
        require(WORKER._manifest(context, files) == V01.read_object(source_files["manifest"]),
                "capture manifest 재검증 실패")
        aaba = V01.read_object(source_files["aaba"])
        aaba_raw = V1.LOCAL._aaba_raw(ROOT, source_files["aaba"], aaba)
        require(aaba.get("worker", {}).get("validation", {}).get("approved") is True
                and aaba.get("worker", {}).get("policy_probe", {}).get("checkpoint_sha256") == cp["sha256"],
                "capture AABA 결합이 다르다")
        for index, record in enumerate(trace["records"]):
            with np.load(stage / f"job.pull-record-{index:06d}.npz", allow_pickle=False) as archive:
                rgb = archive["rgb"]
                require(rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
                        and hashlib.sha256(rgb.tobytes()).hexdigest() == record["rgb_sha256"],
                        "capture raw RGB 결합이 다르다")
        points = V01._source_frame_pts(files["video"])
        require(points == [row["actual_timestamp_s"] for row in trace["records"]],
                "capture 실제 PTS/trace가 다르다")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
             "-of", "json", str(files["video"])], check=True, capture_output=True, text=True,
        )
        timing = json.loads(probe.stdout)
        stream = timing["streams"][0]
        require(stream["width"] == 640 and stream["height"] == 480
                and stream["r_frame_rate"] == "50/1" and stream["nb_read_frames"] == "625"
                and float(timing["format"]["duration"]) == 12.5, "capture video 규격이 다르다")
        state = V1.snapshot(trace, traces[baseline], destination / "unexecuted-probes.jsonl")
        V01.write_new(destination / "snapshot.json", state)
        row: dict[str, Any] = {
            "logical_id": logical, "source_inputs": before, "original_flags": flags,
            "raw_rgb_frames_verified": 625, "record_artifacts_verified": True,
            "video_probe": timing, "aaba_raw_verified": aaba_raw,
            "same_play_motion": V1.motion_summary(trace), "windows": [],
        }
        for index in (1, 2):
            spec = V1.local_spec(V01.read_object(files[f"spec_{index}"]), files["video"], checkpoint,
                                 files[f"measurement_{index}"])
            spec_path = destination / f"window_{index}_local_spec.json"
            V01.write_new(spec_path, spec)
            contract_path = V01.prepare(spec_path, destination / f"window_{index}")
            contract = V01.verify(contract_path)
            V1.SOURCES._verify_window(contract, trace, f"window-0{index}")
            row["windows"].append({"window": index, "contract": V1.reference(contract_path),
                "split": "development", "source_lineage_id": contract["provenance"]["source_lineage_id"]})
        require(before == {key: V1.reference(path) for key, path in source_files.items()},
                "capture source가 생성 중 바뀌었다")
        receipt["policies"].append(row)
    receipt["v01_contracts"] = 4
    receipt["v03_payloads"] = 0
    V01.write_new(output / "receipt.json", receipt)
    return receipt


def prepare_pair(archive_root: Path, capture_state_root: Path, protocol_path: Path,
                 summary_path: Path, copy_verification_path: Path, capture_manifest_path: Path,
                 output: Path, *, seed: int, owned_root: Path) -> dict[str, Any]:
    output = output.resolve()
    output.relative_to(owned_root.resolve())
    require(not output.exists(), "producer output은 새 경로여야 한다")
    archive_root, state_root = archive_root.resolve(strict=True), capture_state_root.resolve(strict=True)
    protocol_path, summary_path = protocol_path.resolve(strict=True), summary_path.resolve(strict=True)
    _protocol_body, catalog = _protocol(protocol_path, seed)
    transfer = _capture_copy_gate(copy_verification_path, capture_manifest_path, state_root, catalog)
    summary = V01.read_object(summary_path)
    archive_proof = V2.verify_archive(archive_root, summary)
    source_root, checkpoints = _archive_material(archive_root, summary, catalog)
    output.mkdir(parents=True, exist_ok=False)
    capture_root, training_root = output / "capture", output / "training"
    _prepare_capture(state_root, protocol_path, catalog, checkpoints, transfer, capture_root, seed=seed)
    V2.prepare(source_root, archive_root, summary_path, capture_root / "receipt.json", training_root,
               owned_root=output)
    accepted = accept_pair(protocol_path, capture_root / "receipt.json", training_root / "receipt.json",
                           seed=seed, verify_raw_rgb=True)
    receipt = {"schema": "vision_transferred_input_producer_v3", "status": "HOLD",
               "cpu_preparation": "PASS", "seed": seed, "archive_verification": archive_proof,
               "capture_receipt": V1.reference(capture_root / "receipt.json"),
               "training_receipt": V1.reference(training_root / "receipt.json"),
               "accepted_pair": accepted, "policy_count": len(accepted["policies"]),
               "payload_count": sum(len(row["payloads"]) for row in accepted["policies"]),
               "behavior_labels_created": False, "model_use_authorized": False,
               "used_for_decision": False, "model_api_requests": 0, "gpu_executions": 0,
               "new_training_actions": 0}
    V01.write_new(output / "receipt.json", receipt)
    return receipt


def _capture_policy(policy: dict[str, Any], checkpoint: dict[str, Any], profile: dict[str, Any],
                    *, verify_raw_rgb: bool) -> dict[str, Any]:
    logical = policy.get("logical_id")
    require(logical == checkpoint["logical_id"], "capture logical ID가 protocol catalog와 다르다")
    inputs = policy.get("source_inputs")
    require(isinstance(inputs, dict), "capture source_inputs가 없다")
    paths = {}
    bodies = {}
    for name, value in inputs.items():
        require(isinstance(value, dict) and set(value) == {"path", "sha256"},
                f"capture {logical} {name} ref 형식")
        path = Path(value["path"]).resolve(strict=True)
        require(path.is_file() and not path.is_symlink() and V01.sha256(path) == value["sha256"],
                f"capture {logical} {name} 원문 SHA 불일치")
        paths[name] = path
        if name not in {"video", "checkpoint"}:
            bodies[name] = V01.read_object(path)
    required = {"video", "raw_trace", "measurement_1", "spec_1", "measurement_2", "spec_2",
                "encoder_receipt", "timing_receipt", "checkpoint", "job", "execution", "result",
                "manifest", "aaba"}
    require(required <= paths.keys(), "capture source input이 부족하다")
    require(V01.sha256(paths["checkpoint"]) == checkpoint["sha256"], "capture checkpoint SHA 불일치")
    trace = bodies["raw_trace"]
    records = trace.get("records")
    require(isinstance(records, list) and len(records) == 625 and trace.get("profile") == profile,
            "capture trace frame/profile 불일치")
    require(trace.get("checkpoint_sha256") == checkpoint["sha256"]
            and trace.get("video_sha256") == V01.sha256(paths["video"]), "capture trace source 결합 불일치")
    execution = bodies["execution"]
    require(execution.get("completed") is True and execution.get("cleanup_verified") is True
            and execution.get("exit_code") == 0 and execution.get("job_sha256") == V01.sha256(paths["job"])
            and execution.get("result_sha256") == V01.sha256(paths["result"]), "capture 실행 receipt 불일치")
    for name in ("execution", "result", "manifest"):
        V1.original_flags(bodies[name])
    files = {name: paths[name] for name in ("video", "raw_trace", "encoder_receipt", "timing_receipt",
             "measurement_1", "spec_1", "measurement_2", "spec_2")}
    context = {"job_path": paths["job"], "checkpoint_sha256": checkpoint["sha256"],
               "job": bodies["job"]}
    require(WORKER.PREVIOUS.RECORD_VERIFY._record_artifacts(context, files), "capture 8 artifact 검증 실패")
    require(WORKER._manifest(context, files) == bodies["manifest"], "capture native manifest 재검증 실패")
    aaba = bodies["aaba"]
    require(aaba.get("worker", {}).get("validation", {}).get("approved") is True
            and aaba.get("worker", {}).get("policy_probe", {}).get("checkpoint_sha256") == checkpoint["sha256"],
            "capture AABA/checkpoint 결합 불일치")
    if verify_raw_rgb:
        stage = paths["job"].parent
        for index, record in enumerate(records):
            archive = stage / f"job.pull-record-{index:06d}.npz"
            with np.load(archive, allow_pickle=False) as values:
                rgb = values["rgb"]
                require(rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
                        and hashlib.sha256(rgb.tobytes()).hexdigest() == record["rgb_sha256"],
                        "capture raw RGB 결합 불일치")
    points = V01._source_frame_pts(paths["video"])
    require(points == [row["actual_timestamp_s"] for row in records], "capture 실제 PTS/trace 불일치")
    motion = policy.get("same_play_motion")
    require(isinstance(motion, dict) and motion.get("behavior_label") is None
            and motion.get("fixed_evaluation_used_as_behavior_ground_truth") is False,
            "capture 행동 label 경계가 다르다")
    return {"logical_id": logical, "trace_sha256": V01.sha256(paths["raw_trace"]),
            "video_sha256": V01.sha256(paths["video"]), "checkpoint_sha256": checkpoint["sha256"],
            "source_inputs": _refs_unchanged(inputs, f"capture {logical}"), "raw_rgb_frames": 625,
            "behavior_label": None}


def _training_policy(row: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    logical = row.get("logical_id")
    require(logical == capture["logical_id"] and row.get("checkpoint_sha256") == capture["checkpoint_sha256"],
            "capture/training policy checkpoint 불일치")
    require(row.get("iterations") == 300 and row.get("training_fact_count") == 48
            and row.get("config_fact_count") == 1, "training iteration/fact 계약 불일치")
    sources = _refs_unchanged(row.get("training_sources", {}), f"training {logical}")
    telemetry_ref = row["training_sources"].get("telemetry.json")
    require(isinstance(telemetry_ref, dict) and telemetry_ref["sha256"] == row.get("telemetry_sha256"),
            "training telemetry ref 불일치")
    _telemetry_path, telemetry = _read_ref(telemetry_ref, f"training {logical} telemetry")
    for metric in SELECT.CONTROLLER.V.TELEMETRY_FIELDS:
        series = telemetry.get("series", {}).get(metric)
        summary = telemetry.get("summary", {}).get(metric)
        require(isinstance(series, list) and len(series) == 300 and isinstance(summary, dict)
                and summary.get("n") == 300, "training 300회 telemetry 불일치")
    snapshot_path, snapshot = _read_ref(row.get("snapshot"), f"training {logical} snapshot")
    packet = snapshot.get("inputs", {}).get("packet", {})
    require(len(packet.get("facts", {})) == 49 and snapshot.get("execution_authorized") is False,
            "training snapshot fact/실행 승인 경계 불일치")
    windows = row.get("windows")
    require(isinstance(windows, list) and [item.get("window") for item in windows] == [1, 2],
            "training window 두 건이 아니다")
    payloads = []
    lineages = []
    for window in windows:
        contract_path, contract = _read_ref(window.get("contract"), f"training {logical} contract")
        verified = V01.verify(contract_path)
        lineage = verified["provenance"]["source_lineage_id"]
        require(lineage == window.get("source_lineage_id") and window.get("split") == "development"
                and verified["provenance"]["checkpoint"]["sha256"] == capture["checkpoint_sha256"],
                "training contract lineage/checkpoint 불일치")
        views = window.get("views")
        require(isinstance(views, dict) and set(views) == set(VIEWS), "training view 세 건이 아니다")
        for view in VIEWS:
            bundle_path, _bundle = _read_ref(views[view].get("bundle"), f"training {logical} {view} bundle")
            messages_path, messages = _read_list_ref(
                views[view].get("messages"), f"training {logical} {view} messages")
            V2.check_messages(messages, [logical, lineage, capture["checkpoint_sha256"], row["telemetry_sha256"]])
            image_count = sum(part.get("type") == "image_url" for message in messages
                              for part in message["content"] if isinstance(part, dict))
            require(image_count == (0 if view == "text_only" else 6), "training view image count 불일치")
            payloads.append({"window": window["window"], "view": view,
                             "bundle": {"path": str(bundle_path), "sha256": V01.sha256(bundle_path)},
                             "messages": {"path": str(messages_path), "sha256": V01.sha256(messages_path)},
                             "image_count": image_count})
        lineages.append({"window": window["window"], "source_lineage_id": lineage,
                         "parent_lineage_id": contract["provenance"].get("parent_source_lineage_id")})
    require(len(payloads) == 6, "정책별 2 window × 3 view가 아니다")
    return {"logical_id": logical, "source_role": row.get("source_role"),
            "checkpoint_sha256": capture["checkpoint_sha256"], "telemetry_sha256": row["telemetry_sha256"],
            "snapshot": {"path": str(snapshot_path), "sha256": V01.sha256(snapshot_path)},
            "training_sources": sources, "nested_lineage": lineages, "payloads": payloads}


def accept_pair(protocol_path: Path, capture_path: Path, training_path: Path, *, seed: int,
                verify_raw_rgb: bool = True) -> dict[str, Any]:
    protocol, catalog = _protocol(protocol_path, seed)
    capture = V01.read_object(capture_path.resolve(strict=True))
    training = V01.read_object(training_path.resolve(strict=True))
    require(capture.get("cpu_preparation") == "PASS" and capture.get("source_seed") == seed,
            "capture receipt seed/PASS 불일치")
    require(training.get("cpu_preparation") == "PASS" and training.get("behavior_labels_created") is False
            and training.get("model_use_authorized") is False and training.get("used_for_decision") is False,
            "training receipt 승인/label 경계 불일치")
    capture_ref = training.get("source_inputs", {}).get("capture")
    require(isinstance(capture_ref, dict) and Path(capture_ref["path"]).resolve() == capture_path.resolve()
            and capture_ref["sha256"] == V01.sha256(capture_path), "training/capture receipt 결합 불일치")
    archive = training.get("archive_verification", {})
    require(archive.get("all_match") is True and archive.get("files_verified") == 437,
            "실제 학습 437-file archive 게이트 불일치")
    _refs_unchanged({**training.get("shared_training_sources", {}),
                     "archive_transfer": archive["transfer"], "archive_inventory": archive["inventory"]},
                    f"seed {seed} training archive")
    captures = {row["logical_id"]: _capture_policy(row, catalog[row["logical_id"]],
                protocol["record_profile"]["value"], verify_raw_rgb=verify_raw_rgb)
                for row in capture.get("policies", [])}
    require(set(captures) == set(catalog), "capture policy catalog가 protocol과 다르다")
    trained = {row["logical_id"]: _training_policy(row, captures[row["logical_id"]])
               for row in training.get("policies", [])}
    require(set(trained) == set(catalog), "training policy catalog가 protocol과 다르다")
    return {"seed": seed, "protocol": {"path": str(protocol_path.resolve()),
            "sha256": V01.sha256(protocol_path)}, "capture_receipt": {"path": str(capture_path.resolve()),
            "sha256": V01.sha256(capture_path)}, "training_receipt": {"path": str(training_path.resolve()),
            "sha256": V01.sha256(training_path)},
            "mechanical_verification": {"source_refs": True, "capture_artifacts": True,
                "raw_rgb": verify_raw_rgb, "actual_pts": True, "training_archive": True,
                "telemetry": True},
            "policies": [trained[key] for key in catalog]}


def attest(pairs: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    policies = [policy for pair in pairs for policy in pair["policies"]]
    require(len({pair["seed"] for pair in pairs}) == len(pairs), "중복 seed pair가 있다")
    verified_domains = {name: all(pair.get("mechanical_verification", {}).get(name) is True
                                  for pair in pairs) for name in MECHANICAL_DOMAINS}
    payload_count = sum(len(row["payloads"]) for row in policies)
    value = {"schema": "vision_transferred_input_acceptance_v3", "status": "HOLD",
             "cpu_preparation": "PASS", "pairs": pairs, "policy_count": len(policies),
             "payload_count": payload_count, "expected_complete_payload_count": 24,
             "complete_four_policy_set": len(policies) == 4 and payload_count == 24,
             "mechanical_verification": verified_domains,
             "use_scope_attestation": {"scopes": list(USE_SCOPE),
                 "behavior_labels_created": False, "behavior_ground_truth_assigned": False,
                 "model_use_authorized": False, "used_for_decision": False,
                 "final_approval": "leader_pending"},
             "model_api_requests": 0, "gpu_executions": 0, "new_training_actions": 0}
    V01.write_new(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, action="append",
                        help="seed/protocol/capture/training 경로를 담은 JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-spec", type=Path,
                        help="archive/capture/protocol/summary/copy manifest/output을 담은 JSON")
    parser.add_argument("--skip-raw-rgb", action="store_true")
    args = parser.parse_args()
    if args.prepare_spec is not None:
        require(args.pair is None and args.output is None and not args.skip_raw_rgb,
                "prepare-spec과 pair 입력을 함께 쓸 수 없다")
        spec = V01.read_object(args.prepare_spec)
        result = prepare_pair(
            Path(spec["archive_root"]), Path(spec["capture_state_root"]), Path(spec["protocol"]),
            Path(spec["summary"]), Path(spec["copy_verification"]), Path(spec["capture_manifest"]),
            Path(spec["output"]), seed=spec["seed"], owned_root=Path(spec["owned_root"]),
        )
        print(json.dumps({"policy_count": result["policy_count"],
                          "payload_count": result["payload_count"], "status": result["status"]}))
        return 0
    require(args.pair is not None and args.output is not None, "pair와 output이 필요하다")
    pairs = []
    for path in args.pair:
        row = V01.read_object(path)
        pairs.append(accept_pair(Path(row["protocol"]), Path(row["capture"]), Path(row["training"]),
                                 seed=row["seed"], verify_raw_rgb=not args.skip_raw_rgb))
    result = attest(pairs, args.output)
    print(json.dumps({"policy_count": result["policy_count"], "payload_count": result["payload_count"],
                      "complete_four_policy_set": result["complete_four_policy_set"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
