"""실제 학습 telemetry와 기존 영상 계약을 결합하는 CPU 개발 입력."""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import residual_demo_v2 as DEMO
import vision_selection_materialize_v1 as MATERIAL
import vision_transferred_input_acceptance_v1 as BASE

V01, SELECT = BASE.V01, BASE.SELECT
require, reference = BASE.require, BASE.reference


def verify_archive(archive: Path, summary: dict) -> dict:
    transfer_path, inventory_path = archive / "transfer-receipt.json", archive / "inventory.json"
    transfer, inventory = V01.read_object(transfer_path), V01.read_object(inventory_path)
    require(V01.sha256(transfer_path) == summary["artifact_transfer"]["sha256"], "공개 보관 receipt SHA 불일치")
    require(transfer["status"] == "complete" and transfer["inventory_sha256"] == V01.sha256(inventory_path),
            "보관 inventory 결합 불일치")
    rows = inventory["files"]
    require(len(rows) == transfer["files"] == summary["artifact_transfer"]["files"], "보관 파일 수 불일치")
    seen = set()
    for row in rows:
        path = (archive / "tree" / row["path"]).resolve(strict=True)
        path.relative_to((archive / "tree").resolve())
        require(path not in seen and path.stat().st_size == row["bytes"]
                and V01.sha256(path) == row["sha256"], "보관 원문 SHA/크기 불일치")
        seen.add(path)
    return {"transfer": reference(transfer_path), "inventory": reference(inventory_path),
            "files_verified": len(rows), "all_match": True}


def verify_training(stage: Path, raw_action: Path, checkpoint: Path, public: dict, summary: dict) -> dict:
    """공개 raw SHA부터 checkpoint·학습 JSON·300회 곡선까지 결합한다."""
    paths = {name: stage / name for name in (
        "adapter_artifact.json", "telemetry.json", "training.manifest.json",
        "training.readback.json", "worker.json", "execution.json")}
    before = {key: reference(path) for key, path in paths.items()}
    before["raw_action"] = reference(raw_action)
    before["checkpoint"] = reference(checkpoint)
    require(V01.sha256(raw_action) == public["raw_sha256"], "공개 학습 raw SHA 불일치")
    raw = V01.read_object(raw_action)
    artifact = V01.read_object(paths["adapter_artifact.json"])
    telemetry = V01.read_object(paths["telemetry.json"])
    manifest = V01.read_object(paths["training.manifest.json"])
    readback = V01.read_object(paths["training.readback.json"])
    worker, execution = V01.read_object(paths["worker.json"]), V01.read_object(paths["execution.json"])
    require(raw["kind"] == "train" and raw["job_id"] == public["job_id"] and raw["artifact"] == artifact,
            "학습 action/artifact 결합 불일치")
    cp_sha, telemetry_sha = V01.sha256(checkpoint), V01.sha256(paths["telemetry.json"])
    require(cp_sha == public["checkpoint_sha256"] == artifact["checkpoint"]["sha256"]
            == worker["checkpoint"]["sha256"], "학습 checkpoint 불일치")
    require(telemetry_sha == artifact["telemetry"]["sha256"] == worker["telemetry"]["sha256"],
            "실제 학습 telemetry SHA 불일치")
    require(telemetry["summary"] == artifact["telemetry"]["summary"] == worker["telemetry"]["summary"],
            "학습 summary 결합 불일치")
    require(artifact["manifest"] == manifest and artifact["readback"] == readback
            and worker["manifest_sha256"] == artifact["manifest_sha256"]
            == V01.sha256(paths["training.manifest.json"])
            and worker["readback_sha256"] == V01.sha256(paths["training.readback.json"]),
            "학습 manifest/readback 결합 불일치")
    require(all(value["arm"] == public["arm"] and value["seed"] == public["seed"]
                for value in (artifact, manifest, worker)), "학습 역할·seed 불일치")
    require(manifest["max_iterations"] == worker["iterations"] == public["iterations"] == 300
            and manifest["source_commit_sha"] == summary["source_commit"]
            and manifest["protocol_sha256_lf"] == summary["source_protocol"]["sha256_lf"],
            "동결 학습 source/iteration 결합 불일치")
    require(worker["completed"] is True and execution["completed"] is True
            and execution["cleanup_verified"] is True and execution["exit_code"] == 0, "학습 실행 미완료")
    for metric in SELECT.CONTROLLER.V.TELEMETRY_FIELDS:
        row, series = telemetry["summary"][metric], telemetry["series"][metric]
        require(row["n"] == len(series) == 300 and all(SELECT.CONTROLLER.V.number(n) for n in series)
                and SELECT.CONTROLLER.V.number(row["mean_last_20pct"]), "300회 학습 지표가 누락되거나 무효함")
    adapted = deepcopy(artifact)
    adapted["checkpoint"]["path"] = str(checkpoint)
    adapted["telemetry"]["path"] = str(paths["telemetry.json"])
    require(before == {**{key: reference(path) for key, path in paths.items()},
                        "raw_action": reference(raw_action), "checkpoint": reference(checkpoint)},
            "학습 source 재조회 불일치")
    return {"artifact": adapted, "files": before, "checkpoint_sha256": cp_sha,
            "telemetry_sha256": telemetry_sha, "metric_count": 24, "iterations": 300}


def training_snapshot(case: dict, baseline: dict, old_snapshot: dict, ledger: Path) -> dict:
    legacy = DEMO.D1.diagnosis_packet(case["artifact"], baseline["artifact"], 1, [], None)
    allowed = SELECT.CONTROLLER.V.BASE_FACT_IDS
    facts = {key: value for key, value in legacy["facts"].items() if key in allowed}
    facts["config:action_scale"] = old_snapshot["inputs"]["packet"]["facts"]["config:action_scale"]
    require(len(facts) == 49, "등록 학습 fact48개와 설정 fact1개가 필요함")
    old_packet = old_snapshot["inputs"]["packet"]
    packet = SELECT.CONTROLLER.S.build_packet(
        epoch=old_packet["epoch"], applied_corrections=old_packet["applied_corrections"],
        prior_receipt=old_packet["prior_receipt"], facts=facts,
        budget_left=old_packet["budget_left"], execution_budget_s=old_packet["execution_budget_s"],
        input_bindings={"case": DEMO.input_binding(case["artifact"]),
                        "reference": DEMO.input_binding(baseline["artifact"])})
    return SELECT.CONTROLLER.V.build_snapshot(
        packet, old_snapshot["inputs"]["run_id"], old_snapshot["inputs"]["policy"],
        old_snapshot["inputs"]["protocol_sha256"], ledger)


def contract_binding(contract: dict, original_window: dict, training: dict) -> dict:
    provenance = contract["provenance"]
    require(provenance["checkpoint"]["sha256"] == training["checkpoint_sha256"], "영상/학습 checkpoint 불일치")
    require(provenance["source_lineage_id"] == original_window["source_lineage_id"]
            and original_window["split"] == "development", "원 영상 계보 또는 개발 split 불일치")
    return {"contract_sha256": original_window["contract"]["sha256"],
            "source_lineage_id": original_window["source_lineage_id"]}


def check_messages(messages: list, hidden: list) -> None:
    BASE.LOCAL._assert_public_view_safe(messages)
    MATERIAL._reject_leak(messages, hidden)
    forbidden = {"fault", "faults", "runid", "runname", "filename", "answerkey", "correctionname",
                 "originalfaults", "appliedcorrections", "checkpointsha256", "telemetrysha256", "sourcelineageid"}

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            require(not forbidden.intersection(re.sub(r"[^a-z0-9]", "", str(k).lower()) for k in value),
                    "모델 payload에 비공개 source 필드가 있음")
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    for value in MATERIAL._visible_json(messages):
        inspect(value)


def parity(bundles: dict, messages: dict) -> dict:
    require(set(bundles) == set(SELECT.VIEW_KINDS), "세 view 누락")
    require(len({b["operational_base_digest"] for b in bundles.values()}) == 1, "운영 상태 parity 불일치")
    facts = {view: bundle["selection_view"]["evidence_facts"] for view, bundle in bundles.items()}
    training = lambda view: {k: v for k, v in facts[view].items() if k.startswith("telemetry:")}
    require(training("text_only") == training("text_and_video") and len(training("text_only")) == 48,
            "텍스트/결합 view 학습 정보 불일치")
    require(not training("video_only"), "video_only에 학습 지표가 노출됨")
    common = lambda view: {k: v for k, v in facts[view].items() if k.startswith("vision:play:")}
    require(common("text_only") == common("video_only") == common("text_and_video"), "같은 Play 수치 parity 불일치")
    counts = {}
    for view, payload in messages.items():
        counts[view] = sum(part.get("type") == "image_url" for message in payload
                           for part in message["content"] if isinstance(part, dict))
        require(counts[view] == (0 if view == "text_only" else 6), "view 이미지 수 불일치")
    return {"training_fact_count": 48, "same_play_fact_count": len(common("text_only")), "image_counts": counts}


def prepare(source_root: Path, archive_root: Path, summary_path: Path, capture_path: Path,
            output: Path, *, owned_root: Path) -> dict:
    output = output.resolve()
    output.relative_to(owned_root.resolve())
    source_root, archive_root = source_root.resolve(strict=True), archive_root.resolve(strict=True)
    source_root.relative_to(archive_root / "tree")
    summary, capture = V01.read_object(summary_path), V01.read_object(capture_path)
    require(capture["cpu_preparation"] == "PASS" and capture["task_id"] == "RI_CAPTURE_ACCEPT_01", "01 입력 미검증")
    seal = {"summary": reference(summary_path), "capture": reference(capture_path)}
    archive_proof = verify_archive(archive_root, summary)
    plan_path, manifest_path, ledger_path = (source_root / name for name in ("training_plan.json", "manifest.json", "training.ledger.jsonl"))
    plan, manifest = V01.read_object(plan_path), V01.read_object(manifest_path)
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(manifest["protocol_sha256_lf"] == summary["source_protocol"]["sha256_lf"]
            and manifest["source_anchor"] == summary["source_commit"], "보관 source manifest 불일치")
    shared = {path.name: reference(path) for path in (plan_path, manifest_path, ledger_path)}
    source_materials, proofs = {}, {}
    actions = {V01.read_object(p)["job_id"]: p for p in (source_root / "actions").glob("*/raw.json")
               if V01.read_object(p).get("kind") == "train"}
    for policy in capture["policies"]:
        logical = policy["logical_id"]
        job = logical.replace("source-", "vision-", 1)
        public_rows = [row for row in summary["training"] if row["job_id"] == job]
        require(len(public_rows) == 1, "공개 학습 행이 유일하지 않음")
        public = public_rows[0]
        require(any(row["job_id"] == job and row["arm"] == public["arm"] and row["seed"] == public["seed"]
                    and row["iterations"] == 300 for row in plan["rows"]), "학습 plan 불일치")
        stage = source_root / "training" / job
        checkpoint = Path(policy["source_inputs"]["checkpoint"]["path"])
        material = verify_training(stage, actions[job], checkpoint, public, summary)
        completed = [row for row in ledger if row["job_id"] == job and row["event"] == "completed"]
        require(len(completed) == 1 and completed[0]["artifact_sha256"] == V01.sha256(stage / "adapter_artifact.json"),
                "학습 완료 ledger 결합 불일치")
        source_materials[logical] = {"stage": stage, "raw_action": actions[job], "checkpoint": checkpoint,
                                     "public": public, "summary": summary}
        proofs[logical] = material
    baseline = next(p["logical_id"] for p in capture["policies"]
                    if source_materials[p["logical_id"]]["public"]["arm"] == "NONE")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "vision_transferred_training_inputs_v2", "task_id": "RI_CAPTURE_TELEMETRY_02",
              "attempt": 1, "contract_revision": 1, "status": "HOLD", "cpu_preparation": "PASS",
              "acceptance": "leader_review_pending", "split": "development", "source_inputs": seal,
              "archive_verification": archive_proof, "shared_training_sources": shared, "policies": [],
              "model_api_requests": 0, "gpu_executions": 0, "new_training_actions": 0,
              "model_use_authorized": False, "used_for_decision": False,
              "behavior_labels_created": False, "independent_acceptance": False,
              "remaining_gates": ["capture/model 사용 승인", "행동 정답 임계값", "정식8입력 비교와 독립 계보 배정"]}
    for policy in capture["policies"]:
        logical, training = policy["logical_id"], proofs[policy["logical_id"]]
        target = output / logical
        target.mkdir()
        old_snapshot = V01.read_object(capture_path.parent / logical / "snapshot.json")
        state = training_snapshot(training, proofs[baseline], old_snapshot, target / "unexecuted-probes.jsonl")
        V01.write_new(target / "snapshot.json", state)
        row = {"logical_id": logical, "source_role": source_materials[logical]["public"]["arm"],
               "training_sources": training["files"], "checkpoint_sha256": training["checkpoint_sha256"],
               "telemetry_sha256": training["telemetry_sha256"], "snapshot": reference(target / "snapshot.json"),
               "iterations": 300, "training_fact_count": 48, "config_fact_count": 1, "windows": []}
        trace_path = Path(policy["source_inputs"]["raw_trace"]["path"])
        require(V01.sha256(trace_path) == policy["source_inputs"]["raw_trace"]["sha256"], "원 trace SHA 불일치")
        trace = V01.read_object(trace_path)
        for original in policy["windows"]:
            contract_path = Path(original["contract"]["path"])
            require(V01.sha256(contract_path) == original["contract"]["sha256"], "원 V01 계약 SHA 불일치")
            contract = V01.verify(contract_path)
            binding = contract_binding(contract, original, training)
            BASE.SOURCES._verify_window(contract, trace, f"window-0{original['window']}")
            bundles, payloads, views = {}, {}, {}
            for view in sorted(SELECT.VIEW_KINDS):
                bundle = SELECT.prepare(state, contract_path, binding, view)
                messages = SELECT.messages(bundle, state, contract_path, binding)
                check_messages(messages, [logical, source_materials[logical]["public"]["job_id"],
                    contract["provenance"]["source_lineage_id"], training["checkpoint_sha256"], training["telemetry_sha256"],
                    training["artifact"]["manifest"]["run_name"]])
                bundles[view], payloads[view] = bundle, messages
                bundle_path, messages_path = target / f"w{original['window']}_{view}_bundle.json", target / f"w{original['window']}_{view}_messages.json"
                V01.write_new(bundle_path, bundle)
                V01.write_new(messages_path, messages)
                views[view] = {"bundle": reference(bundle_path), "messages": reference(messages_path)}
            row["windows"].append({"window": original["window"], "contract": original["contract"],
                "source_lineage_id": original["source_lineage_id"], "split": "development", "views": views,
                "parity": parity(bundles, payloads)})
        require(verify_training(**source_materials[logical])["files"] == training["files"], "입력 생성 후 학습 source 변경")
        result["policies"].append(row)
    require(seal == {"summary": reference(summary_path), "capture": reference(capture_path)}
            and shared == {path.name: reference(path) for path in (plan_path, manifest_path, ledger_path)}, "공유 source 변경")
    result["source_unchanged"] = True
    result["v01_contracts_reused"] = 4
    result["v03_payloads"] = 12
    V01.write_new(output / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "archive-root", "summary", "capture", "output", "owned-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.source_root, args.archive_root, args.summary, args.capture, args.output, owned_root=args.owned_root)
    print(json.dumps({"cpu_preparation": result["cpu_preparation"], "status": result["status"],
                      "v03_payloads": result["v03_payloads"], "training_facts_per_snapshot": 48}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
