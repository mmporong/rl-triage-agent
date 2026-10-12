"""에이전트 진입과 분리해 등록된 콜백 교정의 행동 효과를 한 번 측정한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import run_grounded_recovery as H  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/recovery_feasibility_v3.json"
PRIVATE = ROOT / "bench/private/recovery_feasibility"
METADATA = ROOT / "bench/private/e01_s2028.meta.json"
CODE = {"evals/run_recovery_feasibility.py"}
ARM_ARTIFACTS = (
    "manifest.json", "body.json", "fixed_report.json", "configuration.yaml",
    "execution.json", "launch.json", "stdout.log", "stderr.log", "child_started.json",
    "effect_consumed.json", "gpu_before.json", "gpu_after.json",
)
FIXED = {
    "schema": "recovery_feasibility_protocol_v1",
    "scope": "known_development_input_host_selected_callback_feasibility",
    "effect_order": list(H.C.ARMS), "callbacks": H.CALLBACKS,
    "execution_limit": 3, "semantic_retries": 0, "model_calls": 0,
    "training_seed": 2028, "training_num_envs": 4096, "training_iterations": 300,
    "checkpoint_sha256": H.CHECKPOINT_SHA256,
    "retraining": False, "additional_resets": 0,
    "agent_entry_or_approval_evidence": False, "judgment_influence_proven": False,
    "generalization_proven": False, "policy_weights_repaired": False,
    "stop_rule": "first_invalid_arm_or_one_complete_triplet_no_semantic_retry",
    "result_wording": {
        "recovery_conditions_met": "알려진 개발 입력의 동일 정책에서 등록 콜백 교정의 행동 회복 조건을 관찰했다. 에이전트 선택·판단 변화·학습된 가중치 교정·일반화의 증거는 아니다.",
        "no_recovery": "콜백 교정 조건에서 고정 보행 기준을 충족하지 못했다. 이 조건의 즉시 회복 경로는 채택하지 않는다.",
        "no_failure_observed": "이 고정 평가에서 결함 유지와 콜백 교정 사이에 사전등록한 실패 차이가 없었다. 회복 시연 경로로 채택하지 않는다.",
        "control_invalid": "무변경 반복의 동등성 또는 실패 지표 일치 조건을 충족하지 못했다. 효과의 유무를 확정하지 않는다.",
        "invalid": "실행 또는 출처 검증이 실패했다. 행동 효과로 집계하지 않고 실패 기록을 보존한다.",
    },
}


def source_input() -> tuple[dict, dict]:
    metadata, _, configuration = H.R.original_source_metadata(METADATA)
    if (metadata["fault"] != "X1" or metadata["seed"] != FIXED["training_seed"]
            or metadata["num_envs"] != FIXED["training_num_envs"]
            or metadata["max_iterations"] != FIXED["training_iterations"]):
        raise ValueError("development source identity changed")
    checkpoint = Path(metadata["run_dir"]) / "model_299.pt"
    if H.W.file_sha(checkpoint) != H.CHECKPOINT_SHA256:
        raise ValueError("frozen checkpoint changed")
    scale = H.B._load_configuration_yaml(configuration)["actions"]["joint_pos"]["scale"]
    if type(scale) not in (int, float) or scale != 0.25:
        raise ValueError("source action scale changed")
    binding = {"metadata_sha256": H.W.file_sha(METADATA),
               "configuration_sha256": H.W.file_sha(configuration),
               "agent_configuration_sha256": H.W.file_sha(configuration.with_name("agent.yaml")),
               "checkpoint_sha256": H.W.file_sha(checkpoint), "action_scale": scale}
    return binding, {"checkpoint_path": str(checkpoint.resolve()), "action_scale": scale}


def prepare(isaaclab: Path) -> dict:
    H.validate_protocol(H.read(H.PROTOCOL), isaaclab)
    binding, _ = source_input()
    protocol = {**deepcopy(FIXED), "input_binding": binding,
                "recovery_protocol_sha256": H.W.file_sha(H.PROTOCOL),
                "source_files": {name: H.R.sha_lf(ROOT / name) for name in sorted(CODE)}}
    H.write_new(PROTOCOL, protocol)
    return {"status": "prepared_commit_before_execution", "protocol": str(PROTOCOL),
            "model_calls": 0, "gpu_runs": 0}


def verify_source(isaaclab: Path) -> tuple[dict, dict, str, dict]:
    _, source, _ = H.verified_protocol(isaaclab)
    binding, entry = source_input()
    expected = {**deepcopy(FIXED), "input_binding": binding,
                "recovery_protocol_sha256": H.W.file_sha(H.PROTOCOL),
                "source_files": {name: H.R.sha_lf(ROOT / name) for name in sorted(CODE)}}
    protocol = H.read(PROTOCOL)
    if H.A._sha(protocol) != H.A._sha(expected):
        raise ValueError("feasibility protocol or source changed")
    commit = H.P.verify_committed(ROOT, CODE | {PROTOCOL.relative_to(ROOT).as_posix()})
    return protocol, {**source, "feasibility_source_files": protocol["source_files"]}, commit, entry


def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) is None:
        raise ValueError("invalid feasibility tag")
    return PRIVATE / tag


def request_document(isaaclab: Path) -> tuple[dict, dict]:
    protocol, source, commit, entry = verify_source(isaaclab)
    document = {"schema": "recovery_feasibility_request_v1", "scope": FIXED["scope"],
                "source_commit_sha": commit, "protocol_sha256": H.W.file_sha(PROTOCOL),
                "input_binding": protocol["input_binding"], "entry": entry,
                "effects": [{"arm": arm, "callback": H.CALLBACKS[arm]} for arm in H.C.ARMS],
                "execution_limit": 3, "model_calls": 0,
                "authorization": "user_requested_local_development_measurement",
                "identity_attested": False, "agent_entry_or_approval_evidence": False}
    # 태그나 HEAD만 바꿔 같은 실험을 다시 소비할 수 없다.
    document["bundle_key"] = H.A._sha({"protocol_sha256": document["protocol_sha256"],
                                      "input_binding": document["input_binding"]})
    return document, source


def make_manifest(folder: Path, document: dict, arm: str) -> dict:
    arm_folder = folder / "arms" / arm
    return {"schema": "recovery_feasibility_manifest_v1", "scope": FIXED["scope"],
            "arm": arm, "callback": H.CALLBACKS[arm], "bundle_key": document["bundle_key"],
            "request_sha256": H.W.file_sha(folder / "request.json"),
            "effect_consumed_sha256": H.W.file_sha(arm_folder / "effect_consumed.json"),
            "project_commit_sha": document["source_commit_sha"],
            "protocol_sha256": document["protocol_sha256"],
            "checkpoint_sha256": H.CHECKPOINT_SHA256, **document["entry"],
            "report_path": str(arm_folder / "fixed_report.json"),
            "body_path": str(arm_folder / "body.json"),
            "configuration_path": str(arm_folder / "configuration.yaml")}


def validate_manifest(path: Path, isaaclab: Path) -> dict:
    if not path.is_absolute() or path.name != "manifest.json":
        raise ValueError("fixed absolute manifest path required")
    path = path.resolve()
    arm, folder = path.parent.name, path.parent.parent.parent
    if (arm not in H.C.ARMS or path.parent.parent.name != "arms"
            or folder.parent != PRIVATE.resolve() or folder != folder_for(folder.name).resolve()):
        raise ValueError("manifest outside feasibility folder")
    expected, source = request_document(isaaclab)
    document = H.read(folder / "request.json")
    consumed = {"bundle_key": document["bundle_key"],
                "request_sha256": H.W.file_sha(folder / "request.json"), "execution_limit": 3}
    if (H.A._sha(document) != H.A._sha(expected)
            or H.read(PRIVATE / "consumed" / f"{document['bundle_key']}.json") != consumed
            or H.read(folder / "bundle_consumed.json") != consumed
            or H.read(path.parent / "effect_consumed.json") != {**consumed, "arm": arm}
            or H.A._sha(H.read(path)) != H.A._sha(make_manifest(folder, document, arm))):
        raise ValueError("feasibility source or consumption binding changed")
    manifest = H.read(path)
    return {"manifest": manifest,
            "paths": {key: Path(manifest[key]) for key in ("report_path", "body_path", "configuration_path")},
            "protocol": {**H.read(H.PROTOCOL), "fixed_protocol_path": H.FIXED_EVAL},
            "protocol_sha256": document["protocol_sha256"],
            "manifest_sha256": H.W.file_sha(path), "project_commit_sha": document["source_commit_sha"],
            "source_receipt": source}


def child(path: Path, isaaclab: Path) -> int:
    validated = validate_manifest(path, isaaclab)
    paths = H.D._owned_paths(validated)
    if any(value.exists() for value in paths.values()):
        raise FileExistsError("feasibility child outputs already exist")
    identity = H.S.process_identity()
    # 공통 SDK 검증기가 소비하는 저수준 실행 기록이며 에이전트 승인 기록이 아니다.
    H.write_new(paths["child_started"], {"schema": "grounded_recovery_child_started_v1", **identity,
                                       "manifest_sha256": validated["manifest_sha256"]})
    app = None
    try:
        result, app = H.D.run_sdk(validated)
        manifest = validated["manifest"]
        body = {"schema": H.D.BODY_SCHEMA, "completed": True,
                "arm": manifest["arm"], "callback": manifest["callback"], **result,
                "process_identity": identity, "project_commit_sha": validated["project_commit_sha"],
                "protocol_sha256": validated["protocol_sha256"],
                "manifest_sha256": validated["manifest_sha256"], "source_receipt": validated["source_receipt"],
                "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True}
        H.write_new(paths["body"], body)
    except BaseException as error:
        if isinstance(error, H.D.SdkExecutionError):
            app, error = error.app, error.cause
        H.write_new(paths["child_failure"], {"schema": "recovery_feasibility_child_failure_v1",
                                            "error_type": type(error).__name__, **identity})
        traceback.print_exception(type(error), error, error.__traceback__)
        sys.stdout.flush()
        sys.stderr.flush()
        raise error
    finally:
        if app is not None:
            app.close()
    return 0


def verify_arm(folder: Path, arm: str, execution: dict, isaaclab: Path) -> tuple[dict, dict]:
    if not H.owned_execution_valid(execution):
        raise ValueError("SDK execution or owned process cleanup failed")
    arm_folder = folder / "arms" / arm
    if (H.A._sha(H.read(arm_folder / "execution.json")) != H.A._sha(execution)
            or H.read(arm_folder / "launch.json") != launch_document(arm_folder, isaaclab)
            or H.read(arm_folder / "gpu_before.json").get("busy") is not False):
        raise ValueError("raw execution, launch or GPU admission changed")
    validated = validate_manifest((arm_folder / "manifest.json").resolve(), isaaclab)
    body = H.read(arm_folder / "body.json")
    return H.verify_body(body, validated, execution), body


def launch_document(arm_folder: Path, isaaclab: Path) -> dict:
    return {"args": [str(isaaclab / "_isaac_sim/python.bat"),
                     str(ROOT / "evals/run_recovery_feasibility.py"), "child", "--manifest",
                     str(arm_folder / "manifest.json"), "--isaaclab", str(isaaclab)],
            "cwd": str(isaaclab)}


def verify_triplet(folder: Path, isaaclab: Path) -> dict:
    reports, bodies, inventory = {}, {}, {}
    for arm in H.C.ARMS:
        arm_folder = folder / "arms" / arm
        parent = H.read(arm_folder / "parent_receipt.json")
        if parent.get("status") != "pass" or parent.get("scope") != FIXED["scope"]:
            raise ValueError("feasibility parent is invalid")
        reports[arm], bodies[arm] = verify_arm(folder, arm, parent["execution"], isaaclab)
        inventory[arm] = {name: H.W.file_sha(arm_folder / name) for name in ARM_ARTIFACTS}
        if inventory[arm] != parent["artifact_sha256"]:
            raise ValueError("saved feasibility arm changed")
        if H.read(arm_folder / "gpu_after.json") != parent["gpu_after"]:
            raise ValueError("saved GPU postcheck changed")
    if (len({body["configuration_sha256"] for body in bodies.values()}) != 1
            or len({body["observation_configuration_sha256"] for body in bodies.values()}) != 1):
        raise ValueError("resolved settings differ across arms")
    return {"summary": H.C.summarize_triplet(reports), "arm_artifacts": inventory,
            "execution_provenance_verified": True}


def execute(tag: str, isaaclab: Path) -> dict:
    document, _ = request_document(isaaclab)
    folder = folder_for(tag)
    consume_path = PRIVATE / "consumed" / f"{document['bundle_key']}.json"
    if folder.exists() or consume_path.exists():
        raise FileExistsError("feasibility tag or experiment already consumed")
    for value in (str(ROOT), str(isaaclab), document["entry"]["checkpoint_path"]):
        if any(char in value for char in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    before = H.M.query_gpu(H.FIXED["gpu_admission_profile"])
    if before["busy"]:
        return {"status": "gpu_busy_no_launch", "model_calls": 0, "sdk_jobs_launched": 0}
    folder.mkdir(parents=True, exist_ok=False)
    H.write_new(folder / "request.json", document)
    H.write_new(folder / "gpu_before.json", before)
    consumed = {"bundle_key": document["bundle_key"],
                "request_sha256": H.W.file_sha(folder / "request.json"), "execution_limit": 3}
    H.write_new(consume_path, consumed)
    H.write_new(folder / "bundle_consumed.json", consumed)
    parents, reasons, verified = {}, [], None
    for arm in H.C.ARMS:
        try:
            admission = H.M.query_gpu(H.FIXED["gpu_admission_profile"])
            if admission["busy"]:
                raise RuntimeError("GPU busy before next arm")
            arm_folder = folder / "arms" / arm
            arm_folder.mkdir(parents=True, exist_ok=False)
            H.write_new(arm_folder / "gpu_before.json", admission)
            H.write_new(arm_folder / "effect_consumed.json", {**consumed, "arm": arm})
            H.write_new(arm_folder / "manifest.json", make_manifest(folder, document, arm))
            validate_manifest((arm_folder / "manifest.json").resolve(), isaaclab)
            execution = H.P.run_owned(arm_folder, launch_document(arm_folder, isaaclab)["args"], isaaclab,
                H.FIXED["launcher_watchdog_s"])
            H.write_new(arm_folder / "execution.json", execution)
            verify_arm(folder, arm, execution, isaaclab)
            after = H.M.query_gpu(H.FIXED["gpu_admission_profile"])
            H.write_new(arm_folder / "gpu_after.json", after)
            if any(str(row["pid"]) in execution["owned_process_identities"] for row in after["processes"]):
                raise ValueError("owned GPU process remains")
            parent = {"schema": "recovery_feasibility_parent_v1", "scope": FIXED["scope"],
                      "status": "pass", "execution": execution, "gpu_after": after,
                      "artifact_sha256": {name: H.W.file_sha(arm_folder / name) for name in ARM_ARTIFACTS}}
            H.write_new(arm_folder / "parent_receipt.json", parent)
            parents[arm] = parent
            print(json.dumps({"arm": arm, "status": "pass", "wall_s": execution["wall_s"]}), flush=True)
        except Exception as error:
            reasons.append({"arm": arm, "error_type": type(error).__name__, "error": str(error)})
            break
    if len(parents) == 3:
        try:
            verified = verify_triplet(folder, isaaclab)
        except Exception as error:
            reasons.append({"stage": "triplet", "error_type": type(error).__name__, "error": str(error)})
    status = "invalid" if reasons or verified is None else verified["summary"]["status"]
    result = {"schema": "recovery_feasibility_result_v1", "scope": FIXED["scope"],
              "status": status, "source_commit_sha": document["source_commit_sha"],
              "protocol_sha256": document["protocol_sha256"], "checkpoint_sha256": H.CHECKPOINT_SHA256,
              "execution_provenance_verified": verified is not None, "evidence": verified,
              "validation_reasons": reasons, "model_calls": 0,
              "sdk_jobs_launched": sum((folder / "arms" / arm / "launch.json").exists() for arm in H.C.ARMS),
              "agent_entry_or_approval_evidence": False, "judgment_influence_proven": False,
              "claims": deepcopy(H.C.CLAIMS), "semantic_retries": 0,
              "result_wording": FIXED["result_wording"][status]}
    H.write_new(folder / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "child", "verify"))
    parser.add_argument("--tag")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args()
    isaaclab = args.isaaclab.resolve()
    if args.command == "prepare":
        outcome = prepare(isaaclab)
    elif args.command == "child":
        if args.manifest is None:
            parser.error("child requires --manifest")
        raise SystemExit(child(args.manifest.resolve(), isaaclab))
    else:
        if args.tag is None:
            parser.error("execute/verify requires --tag")
        outcome = execute(args.tag, isaaclab) if args.command == "execute" else verify_triplet(folder_for(args.tag), isaaclab)
    print(json.dumps(outcome, ensure_ascii=False, allow_nan=False))
    raise SystemExit(1 if outcome.get("status") == "invalid" else 0)
