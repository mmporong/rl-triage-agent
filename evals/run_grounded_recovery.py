"""유효한 모델 선택 검사 뒤 동일 T1 체크포인트의 등록 콜백 교정을 평가한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import run_grounded_active as R  # noqa: E402
import grounded_recovery_sdk as D  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402

B, A, G, W, S, P, M = R.B, R.A, R.G, R.W, R.B.S, R.B.P, R.B.M
PROTOCOL = ROOT / "bench/protocols/grounded_recovery_v4.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1.json"
PRIVATE_ROOT = ROOT / "bench/private/grounded_recovery"
CALLBACKS = {"fault_before": "X1", "fault_sham": "X1", "callback_fixed": "X1_REF"}
CHECKPOINT_SHA256 = "ed28e874db3e984b6946aea7fe2a193f5f0661cb3799e0f6aacc04cbecc9f647"
CODE = R.CODE | {
    "evals/run_grounded_recovery.py", "bench/grounded_recovery_sdk.py",
    "src/rl_triage/grounded_recovery.py", "src/rl_triage/recovery.py",
    "evals/fixed_eval.py", "evals/t1_calibration.py",
}
SDK_FILES = B.ISAACLAB_SOURCE_FILES | {
    "source/isaaclab/isaaclab/managers/action_manager.py",
    "source/isaaclab/isaaclab/managers/observation_manager.py",
    "source/isaaclab/isaaclab/managers/termination_manager.py",
    "source/isaaclab/isaaclab/managers/command_manager.py",
    "source/isaaclab/isaaclab/envs/mdp/observations.py",
    "source/isaaclab/isaaclab/envs/mdp/terminations.py",
    "source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py",
    "source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py",
    "source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py",
    "source/isaaclab/isaaclab/utils/dict.py",
}
FIXED = {
    "schema": "grounded_recovery_protocol_v1",
    "scope": "development_same_frozen_policy_registered_environment_callback_intervention",
    "task": C.TASK, "device": "cuda:0", "eval_seed": 2026,
    "num_envs": 1040, "horizon_steps": 1000, "step_dt": 0.02,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "source_training_seed": 2028, "source_training_num_envs": 4096,
    "source_training_iterations": 300,
    "effect_order": list(C.ARMS), "callbacks": CALLBACKS,
    "execution_limit": 3, "semantic_retries": 0,
    "additional_resets": 0, "retraining": False,
    "launcher_watchdog_s": 360,
    "gpu_admission_profile": M.BASE_COM_RESET_GPU_ADMISSION["policy"],
    "isaaclab_commit": B.ISAACLAB_COMMIT, "isaacsim_version": "4.5.0",
    "rsl_rl_version": "2.3.3",
    "entry_max_delta_strict_gt": 0.0501, "entry_outside_fraction_strict_gt": 0.0,
    "require_judgment_change_for_recovery_entry": False,
    "sham_equivalence": "bidirectional_reference_self_bands",
    "relative_failure": "both_unhealthy_same_nonempty_failed_metrics",
    "walk_fraction_strict_lt": 0.5, "fall_max_inclusive": 0.05,
    "aggregation_relative_tolerance": 3e-6, "aggregation_absolute_tolerance": 1e-6,
    "claims": C.CLAIMS,
    "result_wording": {
        "recovery_observed": "동일 동결 정책이 이 고정 task·seed·명령 격자에서 등록된 콜백 교정 후 T1 행동 기준을 통과했다. 정책 가중치 수정·원래 학습 실패 원인·모델의 교정 선택·일반화·우위는 미검증이다.",
        "no_recovery": "등록된 콜백 교정이 같은 동결 정책의 행동 회복 조건을 충족하지 못했다. 효과로 집계하지 않으며 크기·reset·기준을 바꿔 재시도하지 않는다.",
        "no_failure_observed": "교정 조건 대비 사전등록한 실패 차이를 결함 유지와 무변경 반복 대조에서 관찰하지 못했다. 회복 효과로 집계하지 않는다.",
        "control_invalid": "무변경 반복 대조의 동등성 또는 실패 지표 일치 조건을 충족하지 못했다. 교정 효과의 유무는 보류한다.",
        "invalid": "출처·승인·실행·계측·집계·종료 중 한 계약을 충족하지 못했다. 회복 효과로 집계하지 않고 기존 기록을 보존한다.",
    },
}
write_new, read = R.write_new, R.read


def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) is None:
        raise ValueError("tag must use the fixed lowercase filename alphabet")
    return PRIVATE_ROOT / tag


def rsl_root(isaaclab: Path) -> Path:
    return (isaaclab / "_isaac_sim/kit/python/Lib/site-packages/rsl_rl").resolve()


def rsl_paths(isaaclab: Path) -> set[str]:
    root = rsl_root(isaaclab)
    paths = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    if not {"runners/on_policy_runner.py", "modules/actor_critic.py", "algorithms/ppo.py"} <= paths:
        raise FileNotFoundError("pinned RSL-RL distribution source is unavailable")
    return paths


def prepare(isaaclab: Path) -> dict:
    # 모델/GPU 호출 없이 결과별 문구와 실행 소스를 먼저 동결한다.
    protocol = deepcopy(FIXED)
    protocol.update(
        source_files={name: R.sha_lf(ROOT / name) for name in sorted(CODE)},
        isaaclab_source_files={name: W.file_sha(isaaclab / name) for name in sorted(SDK_FILES)},
        rsl_rl_source_files={name: W.file_sha(rsl_root(isaaclab) / name)
                             for name in sorted(rsl_paths(isaaclab))},
        fixed_eval_protocol_sha256_lf=R.sha_lf(FIXED_EVAL),
        grounded_active_protocol_sha256=W.file_sha(R.PROTOCOL),
    )
    validate_protocol(protocol, isaaclab)
    write_new(PROTOCOL, protocol)
    return {"status": "prepared", "path": str(PROTOCOL), "gpu_runs": 0, "model_calls": 0}


def validate_protocol(protocol: dict, isaaclab: Path) -> dict:
    fields = set(FIXED) | {"source_files", "isaaclab_source_files", "rsl_rl_source_files",
                           "fixed_eval_protocol_sha256_lf", "grounded_active_protocol_sha256"}
    if set(protocol) != fields:
        raise ValueError("recovery protocol fields changed")
    for name, value in FIXED.items():
        if type(protocol[name]) is not type(value) or protocol[name] != value:
            raise ValueError(f"fixed recovery protocol field changed: {name}")
    if not isinstance(protocol["source_files"], dict) or set(protocol["source_files"]) != CODE:
        raise ValueError("recovery project source set changed")
    for name, sha in protocol["source_files"].items():
        if S._require_sha(sha, name) != R.sha_lf(S._relative_source_path(ROOT, name, name)):
            raise ValueError(f"recovery source hash changed: {name}")
    sdk = S._verify_source_map(isaaclab, protocol["isaaclab_source_files"], "SDK",
                               exact_paths=SDK_FILES)
    rsl = S._verify_source_map(rsl_root(isaaclab), protocol["rsl_rl_source_files"], "RSL-RL",
                               exact_paths=rsl_paths(isaaclab))
    if S._git_head(isaaclab) != B.ISAACLAB_COMMIT or not S._git_tracked_clean(isaaclab):
        raise ValueError("SDK checkout changed")
    if (protocol["fixed_eval_protocol_sha256_lf"] != R.sha_lf(FIXED_EVAL)
            or protocol["grounded_active_protocol_sha256"] != W.file_sha(R.PROTOCOL)):
        raise ValueError("prior frozen protocol changed")
    return {"project_source_files": protocol["source_files"], "isaaclab_source_files": sdk,
            "rsl_rl_source_files": rsl, "task_binding_sha256": A._sha({"task": C.TASK, "sdk": sdk}),
            "observation_source_binding_sha256": A._sha({"sdk": sdk, "rsl": rsl})}


def verified_protocol(isaaclab: Path) -> tuple[dict, dict, str]:
    protocol = read(PROTOCOL)
    source = validate_protocol(protocol, isaaclab.resolve())
    R.verify_protocol()
    commit = P.verify_committed(ROOT, CODE | {
        PROTOCOL.relative_to(ROOT).as_posix(), FIXED_EVAL.relative_to(ROOT).as_posix(),
    })
    return protocol, source, commit


def fresh_record(folder: Path, stage: str, packet: dict) -> dict:
    record_path = folder / f"{stage}.model.json"
    outcome = read(folder / f"{stage}.outcome.json")
    record = read(record_path)
    valid = (outcome.get("valid") is True and outcome.get("worker_cleanup_verified") is True
             and outcome.get("stage") == stage and outcome.get("model_record_sha256") == W.file_sha(record_path)
             and type(outcome.get("launch", {}).get("returncode")) is int
             and outcome["launch"]["returncode"] == 0
             and outcome.get("launch", {}).get("timed_out") is False
             and record.get("schema") == "grounded_active_model_call_v1"
             and record.get("valid") is True and record.get("stage") == stage
             and record.get("packet_sha256") == W.file_sha(folder / f"{stage}.packet.json")
             and read(folder / f"{stage}.packet.json") == packet
             and record.get("model") == W.MODEL and type(record.get("temperature")) in (int, float)
             and record["temperature"] == 0
             and type(record.get("semantic_retries")) is int and record["semantic_retries"] == 0
             and type(record.get("transport_retries")) is int and record["transport_retries"] == 0
             and record.get("client_cache_used") is False and record.get("saved_response_cached") is False)
    if stage == "initial":
        W.validate_initial_execution_record(record_path, record)
    else:
        valid = (valid and record.get("messages_sha256") == W.canonical_sha(W.post_messages(packet))
                 and type(record.get("llm_calls_started")) is int
                 and type(record.get("llm_calls_finished")) is int
                 and record["llm_calls_started"] == record["llm_calls_finished"] == 1
                 and isinstance(record.get("provider_response_id"), str) and bool(record["provider_response_id"])
                 and isinstance(record.get("provider_model"), str) and bool(record["provider_model"].strip())
                 and record.get("max_output_tokens") == W.POST_MAX_TOKENS
                 and record.get("reasoning_effort") == W.POST_REASONING_EFFORT
                 and record.get("provider_cache_status") == "not_attested")
    if not valid:
        raise ValueError(f"fresh {stage} call is not bound and valid")
    return record


def validate_entry(grounded_tag: str, isaaclab: Path) -> dict:
    folder, spec, binding = R.load_run(grounded_tag)
    if spec["arm"] != "X1" or spec["source_seed"] != FIXED["source_training_seed"]:
        raise ValueError("only the original declared X1 development input is supported")
    case, effect = spec["case_id"], read(folder / "consumed_binding.json")["effect_key"]
    assessment = A._load_bound("grounded_active_assessments", f"{case}.json")
    receipt = A._load_bound("active_receipts", f"{case}__{effect}.json")
    initial_packet = {"schema": "grounded_active_initial_packet_v1", "case_id": case,
                      "catalog_sha256": G.evidence_catalog(case)["catalog_sha256"],
                      "workflow_config_sha256_lf": R.sha_lf(W.CONFIG)}
    initial = fresh_record(folder, "initial", initial_packet)
    actual_packet = A._post_packet(case, effect, receipt, False)
    actual = fresh_record(folder, "actual", actual_packet)
    decision = A.validate_final_decision(actual_packet, actual["decision"])
    if (initial.get("assessment_sha256") != A._sha(assessment)
            or assessment["next_check"]["registry_id"] != "reset_base_com_stability"
            or receipt.get("execution_valid") is not True
            or receipt["case_source_binding_sha256"] != binding["case_source_binding_sha256"]):
        raise ValueError("initial selection or latest normalized receipt is invalid")
    consumed = A._load_bound("active_consumed", f"{case}__{effect}.json")
    A._validate_consumed_document(case, effect, binding["case_source_binding_sha256"], consumed)
    if read(folder / "consumed_binding.json") != {
            **{key: consumed[key] for key in ("effect_key", "request_sha256", "approval_sha256",
                                              "case_source_binding_sha256")}, "consumed_sha256": A._sha(consumed)}:
        raise ValueError("grounded effect consume binding changed")
    sdk_folder = ROOT / "bench/private/base_com_reset" / f"ga_{grounded_tag}"
    parent = read(sdk_folder / "parent_receipt.json")
    validated = B.validate_manifest((sdk_folder / "manifest.json").resolve(), isaaclab_root=isaaclab.resolve())
    body = read(sdk_folder / "body.json")
    if (parent["status"] != "pass" or parent["arm"] != "X1"
            or parent["project_commit_sha"] != spec["source_commit_sha"]
            or parent["body_sha256"] != W.file_sha(sdk_folder / "body.json")
            or parent["manifest_sha256"] != W.file_sha(sdk_folder / "manifest.json")
            or parent["protocol_sha256"] != W.file_sha(B.PROTOCOL)
            or B.verify_body(body, validated, parent["execution"], spec["source_commit_sha"])):
        raise ValueError("model-selected SDK measurement provenance is invalid")
    measured = {row["measurement_id"]: row["value"] for row in receipt["measurements"]}
    raw = body["observer_report"]["measurements"]
    if set(measured) != {"base_com_max_abs_delta_xy", "base_com_outside_range_fraction"}:
        raise ValueError("receipt measurement set changed")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or value != raw[name]["value"] for name, value in measured.items()):
        raise ValueError("receipt values differ from the verified raw SDK values")
    predictions = {row["hypothesis_id"]: row["predictions"]
                   for row in assessment["next_check"]["hypothesis_predictions"]}
    if (not all(G._relation_holds(measured[row["measurement_id"]], row["relation"], row["operand"]["value"])
                for row in predictions[decision["ranking"][0]])
            or measured["base_com_max_abs_delta_xy"] <= 0.0501
            or measured["base_com_outside_range_fraction"] <= 0):
        raise ValueError("actual decision predictions or registered mechanism entry gate failed")
    metadata, _, config_path = R.original_source_metadata(Path(spec["metadata_path"]))
    if metadata["num_envs"] != 4096 or metadata.get("max_iterations") != 300:
        raise ValueError("original checkpoint training identity changed")
    checkpoint = Path(metadata["run_dir"]) / "model_299.pt"
    if W.file_sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("original frozen policy checkpoint changed")
    scale = B._load_configuration_yaml(config_path)["actions"]["joint_pos"]["scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("original action scale is invalid")
    attribution = {"judgment_influence_proven": False, "reason": "masked_pair_not_verified"}
    try:
        masked_packet = A._post_packet(case, effect, receipt, True)
        masked = fresh_record(folder, "masked", masked_packet)
        if (actual["provider_response_id"] == masked["provider_response_id"]
                or actual["provider_model"] != masked["provider_model"]):
            raise ValueError("post pair is not distinct and matched")
        attribution = A.build_judgment_influence_proof(assessment, actual_packet, decision,
                                                      masked_packet, masked["decision"])
    except (OSError, KeyError, ValueError):
        pass  # 판단 변화와 행동 회복의 진입 조건은 별도로 보존한다.
    artifacts = [folder / name for name in (
        "host_spec.json", "case_binding.json", "consumed_binding.json",
        "initial.packet.json", "initial.model.json", "initial.outcome.json",
        "actual.packet.json", "actual.model.json", "actual.outcome.json",
    )] + [sdk_folder / name for name in ("manifest.json", "body.json", "parent_receipt.json")]
    return {"case_id": case, "effect_key": effect,
            "source_binding_sha256": binding["case_source_binding_sha256"],
            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_configuration_path": str(config_path.resolve()),
            "source_configuration_sha256": W.file_sha(config_path), "action_scale": scale,
            "grounded_artifacts": {str(path.relative_to(ROOT)): W.file_sha(path) for path in artifacts},
            "latest_receipt_sha256": A._sha(receipt), "actual_top": decision["ranking"][0],
            "mechanism_values": measured, "judgment_influence": attribution,
            "model_selected_check_executed": True, "model_selected_intervention": False,
            "intervention_registration": "host_registered_X1_to_X1_REF_callback_only",
            "source_relationship": R.FIXED["case_and_probe_relationship"]}


def request(tag: str, grounded_tag: str, isaaclab: Path) -> dict:
    _, _, commit = verified_protocol(isaaclab)
    entry = validate_entry(grounded_tag, isaaclab)
    core = {"grounded_tag": grounded_tag, "source_commit_sha": commit,
            "protocol_sha256": W.file_sha(PROTOCOL), "entry": entry,
            "effects": [{"arm": arm, "callback": CALLBACKS[arm]} for arm in C.ARMS],
            "execution_limit": 3}
    key = A._sha(core)
    if (PRIVATE_ROOT / "consumed" / f"{key}.json").exists():
        raise FileExistsError("this frozen recovery bundle was already consumed")
    folder = folder_for(tag)
    folder.mkdir(parents=True, exist_ok=False)
    document = {"schema": "grounded_recovery_request_v1", "bundle_key": key, **core}
    write_new(folder / "request.json", document)
    return {"status": "approval_pending", "bundle_key": key, "execution_limit": 3,
            "model_selected_intervention": False, "gpu_runs": 0}


def load_request(tag: str, isaaclab: Path) -> tuple[Path, dict, dict, dict]:
    protocol, source, commit = verified_protocol(isaaclab)
    folder = folder_for(tag)
    document = read(folder / "request.json")
    keys = {"schema", "bundle_key", "grounded_tag", "source_commit_sha", "protocol_sha256",
            "entry", "effects", "execution_limit"}
    if set(document) != keys or document["schema"] != "grounded_recovery_request_v1":
        raise ValueError("recovery request fields changed")
    core = {name: value for name, value in document.items() if name not in {"schema", "bundle_key"}}
    if (document["bundle_key"] != A._sha(core) or document["source_commit_sha"] != commit
            or document["protocol_sha256"] != W.file_sha(PROTOCOL)
            or type(document["execution_limit"]) is not int or document["execution_limit"] != 3
            or document["effects"] != [{"arm": arm, "callback": CALLBACKS[arm]} for arm in C.ARMS]
            or document["entry"] != validate_entry(document["grounded_tag"], isaaclab)):
        raise ValueError("recovery request/source/actual decision binding changed")
    return folder, document, protocol, source


def approve(tag: str, actor: str, isaaclab: Path) -> dict:
    folder, document, _, _ = load_request(tag, isaaclab)
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
        raise ValueError("approval actor must be a nonempty bounded label")
    if (folder / "bundle_consumed.json").exists() or (folder / "result.json").exists():
        raise FileExistsError("consumed or finished bundles cannot be approved again")
    approval = {"schema": "grounded_recovery_approval_v1", "bundle_key": document["bundle_key"],
                "request_sha256": W.file_sha(folder / "request.json"), "actor": actor.strip(),
                "identity_attested": False, "approved": True, "execution_limit": 3}
    write_new(folder / "approval.json", approval)
    return approval


def validate_approval(folder: Path, document: dict) -> dict:
    approval = read(folder / "approval.json")
    expected = {"schema": "grounded_recovery_approval_v1", "bundle_key": document["bundle_key"],
                "request_sha256": W.file_sha(folder / "request.json"),
                "actor": approval.get("actor"), "identity_attested": False,
                "approved": True, "execution_limit": 3}
    if (A._sha(approval) != A._sha(expected) or set(approval) != set(expected)
            or approval["approved"] is not True or approval["identity_attested"] is not False
            or type(approval["execution_limit"]) is not int
            or not isinstance(approval["actor"], str) or not approval["actor"].strip()
            or approval["actor"] != approval["actor"].strip() or len(approval["actor"]) > 128):
        raise ValueError("bundle approval binding changed")
    return approval


def bundle_consumed(folder: Path, document: dict) -> dict:
    approval = validate_approval(folder, document)
    return {"schema": "grounded_recovery_bundle_consumed_v1", "bundle_key": document["bundle_key"],
            "request_sha256": approval["request_sha256"],
            "approval_sha256": W.file_sha(folder / "approval.json"),
            "execution_limit": 3, "side_effect_started": False}


def effect_consumed(folder: Path, document: dict, arm: str) -> dict:
    return {"schema": "grounded_recovery_effect_consumed_v1", "bundle_key": document["bundle_key"],
            "bundle_consumed_sha256": W.file_sha(folder / "bundle_consumed.json"),
            "arm": arm, "callback": CALLBACKS[arm], "ordinal": C.ARMS.index(arm) + 1,
            "side_effect_started": False}


def make_manifest(folder: Path, document: dict, arm: str) -> dict:
    arm_folder = folder / "arms" / arm
    return {"schema": "grounded_recovery_manifest_v1", "arm": arm, "callback": CALLBACKS[arm],
            "bundle_key": document["bundle_key"], "request_sha256": W.file_sha(folder / "request.json"),
            "bundle_consumed_sha256": W.file_sha(folder / "bundle_consumed.json"),
            "effect_consumed_sha256": W.file_sha(arm_folder / "effect_consumed.json"),
            "project_commit_sha": document["source_commit_sha"], "protocol_path": str(PROTOCOL),
            "protocol_sha256": document["protocol_sha256"],
            "checkpoint_path": document["entry"]["checkpoint_path"],
            "checkpoint_sha256": CHECKPOINT_SHA256, "action_scale": document["entry"]["action_scale"],
            "report_path": str(arm_folder / "fixed_report.json"), "body_path": str(arm_folder / "body.json"),
            "configuration_path": str(arm_folder / "configuration.yaml")}


def validate_manifest(manifest_path: Path, isaaclab: Path | None = None) -> dict:
    isaaclab = (Path.home() / "IsaacLab" if isaaclab is None else isaaclab).resolve()
    if not manifest_path.is_absolute() or manifest_path.name != "manifest.json":
        raise ValueError("child manifest requires its fixed absolute path")
    manifest_path = manifest_path.resolve()
    arm = manifest_path.parent.name
    folder = manifest_path.parent.parent.parent
    if (arm not in C.ARMS or manifest_path.parent.parent.name != "arms"
            or folder.parent != PRIVATE_ROOT.resolve() or folder != folder_for(folder.name).resolve()):
        raise ValueError("child manifest is outside its registered arm folder")
    _, document, protocol, source = load_request(folder.name, isaaclab)
    consumed = bundle_consumed(folder, document)
    global_path = PRIVATE_ROOT / "consumed" / f"{document['bundle_key']}.json"
    if (A._sha(read(folder / "bundle_consumed.json")) != A._sha(consumed)
            or A._sha(read(global_path)) != A._sha(consumed)
            or A._sha(read(manifest_path.parent / "effect_consumed.json"))
                != A._sha(effect_consumed(folder, document, arm))):
        raise ValueError("bundle/effect approval consumption changed")
    manifest = read(manifest_path)
    if A._sha(manifest) != A._sha(make_manifest(folder, document, arm)):
        raise ValueError("child manifest differs from its approved effect")
    return {"manifest": manifest, "paths": {key: Path(manifest[key]) for key in (
                "report_path", "body_path", "configuration_path")},
            "protocol": {**protocol, "fixed_protocol_path": FIXED_EVAL},
            "protocol_sha256": document["protocol_sha256"],
            "manifest_sha256": W.file_sha(manifest_path),
            "project_commit_sha": document["source_commit_sha"], "source_receipt": source}


def report_expected(manifest: dict) -> dict:
    return {"name": manifest["arm"], "checkpoint_file": Path(manifest["checkpoint_path"]).name,
            "checkpoint_sha256": CHECKPOINT_SHA256, "action_scale": manifest["action_scale"],
            "protocol_sha256_lf": R.sha_lf(FIXED_EVAL)}


def validate_configuration(path: Path, manifest: dict) -> dict:
    config = B._load_configuration_yaml(path)
    if set(config) != {"environment", "observation_configuration", "injection_readback",
                       "resolved_event_readback", "target_body_readback"}:
        raise ValueError("resolved configuration fields changed")
    env = config["environment"]
    expected = [(env["seed"], 2026), (env["scene"]["num_envs"], 1040),
                (env["decimation"], 4), (env["sim"]["dt"], 0.005),
                (env["sim"]["device"], "cuda:0"), (env["episode_length_s"], 20.0),
                (env["actions"]["joint_pos"]["scale"], manifest["action_scale"]),
                (env["commands"]["base_velocity"]["heading_command"], False),
                (env["commands"]["base_velocity"]["rel_standing_envs"], 0.0),
                (env["commands"]["base_velocity"]["resampling_time_range"], [1e9, 1e9])]
    if any(type(value) is not type(wanted) or value != wanted for value, wanted in expected):
        raise ValueError("fixed evaluation resolved settings changed")
    if (env["events"]["add_base_mass"] is not None
            or env["observations"]["policy"]["enable_corruption"] is not False
            or config["observation_configuration"] != env["observations"]):
        raise ValueError("mass or observation settings changed")
    resolved = config["resolved_event_readback"]
    if set(resolved) != {"source", "base_com"} or resolved["source"] != "base.event_manager.get_term_cfg('base_com')":
        raise ValueError("resolved event source changed")
    D.validate_event_readback(config["injection_readback"], resolved["base_com"])
    D.validate_event_readback(config["injection_readback"], env["events"]["base_com"])
    original = env["events"]["base_com"]["params"]["asset_cfg"]
    entity = resolved["base_com"]["params"]["asset_cfg"]
    target = config["target_body_readback"]
    if (not isinstance(target, dict) or set(target) != {
            "source", "asset_name", "body_names", "base_body_index"}
            or target["source"] != "base.scene['robot'].body_names" or target["asset_name"] != "robot"):
        raise ValueError("runtime target body readback source changed")
    names, index, ids = target["body_names"], target["base_body_index"], entity.get("body_ids")
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or names.count("base") != 1
            or type(index) is not int or not 0 <= index < len(names) or names[index] != "base"
            or original["name"] != "robot" or original["body_names"] != "base"
            or original.get("body_ids") != [None, None, None]
            or entity["name"] != "robot" or entity["body_names"] != ["base"]
            or not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int or ids[0] != index):
        raise ValueError("base COM original/resolved body binding differs from the runtime robot")
    return config


def verify_body(body: dict, validated: dict, execution: dict) -> dict:
    manifest, paths = validated["manifest"], validated["paths"]
    expected = {
        "schema": D.BODY_SCHEMA, "completed": True, "arm": manifest["arm"],
        "isaacsim_version": FIXED["isaacsim_version"],
        "callback": manifest["callback"], "checkpoint_sha256": CHECKPOINT_SHA256,
        "configuration_sha256": W.file_sha(paths["configuration_path"]),
        "callback_code_identity_verified": True,
        "project_commit_sha": validated["project_commit_sha"],
        "protocol_sha256": validated["protocol_sha256"],
        "manifest_sha256": validated["manifest_sha256"],
        "source_receipt": validated["source_receipt"],
        "report_sha256": W.file_sha(paths["report_path"]),
        "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True,
        "env_create_count": 1,
    }
    additional = {"observation_configuration_sha256", "injection_readback", "resolved_event_readback",
                  "target_body_readback", "clocks", "process_identity"}
    if set(body) != set(expected) | additional:
        raise ValueError("SDK body fields changed")
    for name, wanted in expected.items():
        if type(body[name]) is not type(wanted) or body[name] != wanted:
            raise ValueError(f"SDK body binding changed: {name}")
    identity = body["process_identity"]
    identity_keys = {"process_id", "process_creation_time_filetime"}
    if (not isinstance(identity, dict) or set(identity) != identity_keys
            or any(type(identity[key]) is not int or identity[key] <= 0 for key in identity_keys)
            or execution["owned_process_identities"].get(str(identity["process_id"]))
                != identity["process_creation_time_filetime"]):
        raise ValueError("SDK child is not identified in the owned Windows Job")
    started = read(paths["body_path"].parent / "child_started.json")
    if started != {"schema": "grounded_recovery_child_started_v1", **identity,
                   "manifest_sha256": validated["manifest_sha256"]}:
        raise ValueError("child start identity or manifest changed")
    clocks = body["clocks"]
    if (not isinstance(clocks, dict) or set(clocks) != {"rows", "decimation", "reset_calls"}
            or type(clocks["decimation"]) is not int or clocks["decimation"] != 4):
        raise ValueError("SDK clock evidence schema changed")
    D.validate_clock_trace(clocks["rows"], reset_calls=clocks["reset_calls"],
                           env_create_count=body["env_create_count"])
    config = validate_configuration(paths["configuration_path"], manifest)
    if (body["injection_readback"] != config["injection_readback"]
            or body["resolved_event_readback"] != config["resolved_event_readback"]
            or A._sha(body["target_body_readback"]) != A._sha(config["target_body_readback"])
            or body["observation_configuration_sha256"] != A._sha(config["observation_configuration"])):
        raise ValueError("SDK event or observation configuration differs from its saved snapshot")
    report = C.validate_fixed_report(read(paths["report_path"]), report_expected(manifest))
    if report["versions"]["rsl-rl-lib"] != FIXED["rsl_rl_version"]:
        raise ValueError("runtime RSL-RL version differs from its pinned source")
    return report


def owned_execution_valid(execution: dict) -> bool:
    return (execution.get("job_assigned_before_release") is True
            and execution.get("natural_exit") is True and type(execution.get("exit_code")) is int
            and execution["exit_code"] == 0 and execution.get("cleanup_verified") is True
            and execution.get("timed_out") is False and execution.get("error_type") is None)


def run_effect(folder: Path, document: dict, arm: str, isaaclab: Path) -> tuple[dict, dict | None, dict | None]:
    arm_folder = folder / "arms" / arm
    arm_folder.mkdir(parents=True, exist_ok=False)
    write_new(arm_folder / "effect_consumed.json", effect_consumed(folder, document, arm))
    write_new(arm_folder / "manifest.json", make_manifest(folder, document, arm))
    validated = validate_manifest((arm_folder / "manifest.json").resolve(), isaaclab)
    execution = P.run_owned(arm_folder, [str(isaaclab / "_isaac_sim/python.bat"),
                             str(ROOT / "bench/grounded_recovery_sdk.py"),
                             str(arm_folder / "manifest.json")], isaaclab, FIXED["launcher_watchdog_s"])
    reasons, report, body = [], None, None
    try:
        if not owned_execution_valid(execution):
            raise ValueError("owned SDK execution did not complete naturally and cleanly")
        body = read(arm_folder / "body.json")
        # 출처가 실행 중 바뀌었는지도 같은 source/consume 계약으로 재검증한다.
        after = validate_manifest((arm_folder / "manifest.json").resolve(), isaaclab)
        if after != validated:
            raise ValueError("SDK manifest/source changed during evaluation")
        report = verify_body(body, validated, execution)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        reasons.append("artifact_or_execution_" + type(error).__name__)
    gpu_after = None
    try:
        gpu_after = M.query_gpu(FIXED["gpu_admission_profile"])
        if any(str(row["pid"]) in execution["owned_process_identities"] for row in gpu_after["processes"]):
            reasons.append("owned_gpu_process_remaining")
    except Exception as error:
        reasons.append("post_gpu_query_" + type(error).__name__)
    parent = {"schema": "grounded_recovery_parent_v1", "arm": arm, "callback": CALLBACKS[arm],
              "status": "invalid" if reasons else "pass", "validation_reasons": reasons,
              "execution": execution, "manifest_sha256": W.file_sha(arm_folder / "manifest.json"),
              "body_sha256": W.file_sha(arm_folder / "body.json") if (arm_folder / "body.json").exists() else None,
              "report_sha256": W.file_sha(arm_folder / "fixed_report.json")
                  if (arm_folder / "fixed_report.json").exists() else None,
              "source_commit_sha": document["source_commit_sha"],
              "protocol_sha256": document["protocol_sha256"], "gpu_after": gpu_after,
              "claims": deepcopy(C.CLAIMS)}
    write_new(arm_folder / "parent_receipt.json", parent)
    if read(arm_folder / "parent_receipt.json") != parent:
        raise ValueError("saved and returned SDK parent receipt differ")
    return parent, report if not reasons else None, body if not reasons else None


def arm_artifact_inventory(folder: Path, parents: dict) -> dict:
    inventory = {}
    for arm in C.ARMS:
        arm_folder = folder / "arms" / arm
        paths = {"parent_receipt_sha256": "parent_receipt.json", "body_sha256": "body.json",
                 "report_sha256": "fixed_report.json", "configuration_sha256": "configuration.yaml",
                 "manifest_sha256": "manifest.json"}
        digests = {}
        for key, name in paths.items():
            try:
                digests[key] = W.file_sha(arm_folder / name) if (arm_folder / name).is_file() else None
            except OSError:
                digests[key] = None
        inventory[arm] = digests
        inventory[arm]["parent_status"] = parents.get(arm, {}).get(
            "status", "receipt_missing" if arm_folder.exists() else "not_launched")
    return inventory


def verify_saved_triplet(folder: Path, parents: dict, reports: dict, bodies: dict, isaaclab: Path) -> dict:
    if any(set(value) != set(C.ARMS) for value in (parents, reports, bodies)):
        raise ValueError("complete source-verified three-arm evidence is required")
    inventory = arm_artifact_inventory(folder, parents)
    for arm in C.ARMS:
        arm_folder = folder / "arms" / arm
        parent, body, report = (read(arm_folder / name) for name in (
            "parent_receipt.json", "body.json", "fixed_report.json"))
        if (A._sha(parent) != A._sha(parents[arm]) or A._sha(body) != A._sha(bodies[arm])
                or A._sha(report) != A._sha(reports[arm]) or parent["status"] != "pass"
                or not owned_execution_valid(parent["execution"])
                or parent["body_sha256"] != inventory[arm]["body_sha256"]
                or parent["report_sha256"] != inventory[arm]["report_sha256"]
                or parent["manifest_sha256"] != inventory[arm]["manifest_sha256"]
                or body["configuration_sha256"] != inventory[arm]["configuration_sha256"]):
            raise ValueError("saved arm artifacts changed after their parent verification")
        validated = validate_manifest((arm_folder / "manifest.json").resolve(), isaaclab)
        verify_body(body, validated, parent["execution"])
    return inventory


def execute(tag: str, isaaclab: Path) -> dict:
    isaaclab = isaaclab.resolve()
    folder, document, _, _ = load_request(tag, isaaclab)
    consumed = bundle_consumed(folder, document)
    if any((folder / name).exists() for name in ("bundle_consumed.json", "result.json", "arms")):
        raise FileExistsError("consumed, orphaned, or finished bundles are never rerun")
    for value in (str(ROOT), str(isaaclab), document["entry"]["checkpoint_path"]):
        if any(char in value for char in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    first_gpu = M.query_gpu(FIXED["gpu_admission_profile"])
    if first_gpu["busy"]:
        raise RuntimeError("another or unclassified GPU workload is present; approval not consumed")
    # 전체 원장과 arm별 원장을 모두 launch 전에 소비한다. 고아 기록은 재실행하지 않는다.
    write_new(PRIVATE_ROOT / "consumed" / f"{document['bundle_key']}.json", consumed)
    write_new(folder / "bundle_consumed.json", consumed)
    parents, reports, bodies, reasons = {}, {}, {}, []
    for arm in C.ARMS:
        try:
            gpu = M.query_gpu(FIXED["gpu_admission_profile"])
            if gpu["busy"]:
                raise RuntimeError("GPU became busy before the next effect; bundle stays consumed")
            parent, report, body = run_effect(folder, document, arm, isaaclab)
            parents[arm] = parent
            if parent["status"] != "pass":
                reasons.append(f"{arm}_invalid")
                break
            reports[arm], bodies[arm] = report, body
        except Exception as error:
            reasons.append(f"{arm}_launch_or_validation_{type(error).__name__}")
            write_new(folder / f"{arm}.failure.json", {"error_type": type(error).__name__,
                                                       "side_effect_retry": False})
            break
    summary, artifacts = None, arm_artifact_inventory(folder, parents)
    provenance_verified = False
    if len(reports) == 3 and not reasons:
        try:
            if (len({body["configuration_sha256"] for body in bodies.values()}) != 1
                    or len({body["observation_configuration_sha256"] for body in bodies.values()}) != 1):
                raise ValueError("resolved settings differ across arms")
            artifacts = verify_saved_triplet(folder, parents, reports, bodies, isaaclab)
            summary = C.summarize_triplet(reports)
            provenance_verified = True
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            reasons.append("triplet_" + type(error).__name__)
    status = "invalid" if reasons or summary is None else (
        "recovery_observed" if summary["status"] == "recovery_conditions_met" else summary["status"])
    result = {"schema": "grounded_recovery_result_v1", "status": status,
              "execution_provenance_verified": provenance_verified,
              "recovery_proven": bool(provenance_verified and summary
                                      and summary["recovery_conditions_met"] and not reasons),
              "bundle_key": document["bundle_key"], "source_commit_sha": document["source_commit_sha"],
              "protocol_sha256": document["protocol_sha256"], "checkpoint_sha256": CHECKPOINT_SHA256,
              "entry": document["entry"], "effect_order": list(parents), "validation_reasons": reasons,
              "summary": summary, "summary_scope": "numeric_conditions_only",
              "arm_artifacts": artifacts, "claims": deepcopy(C.CLAIMS),
              "judgment_influence": document["entry"]["judgment_influence"],
              "cpu_verification_is_recovery_evidence": False, "new_model_calls": 0,
              "sdk_jobs_launched": sum((folder / "arms" / arm / "launch.json").exists() for arm in C.ARMS),
              "parent_wall_s": sum(parent["execution"]["wall_s"] for parent in parents.values()),
              "wall_scope": "owned_parent_launch_exit_cleanup_sum_not_gpu_time",
              "gpu_exclusive": False, "performance_measurement": False,
              "gpu_before": first_gpu, "result_wording": FIXED["result_wording"][status]}
    write_new(folder / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "request", "approve", "execute"))
    parser.add_argument("--tag")
    parser.add_argument("--grounded-tag")
    parser.add_argument("--actor")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args()
    if args.command != "prepare" and args.tag is None:
        parser.error("this command requires --tag")
    if args.command == "prepare":
        result = prepare(args.isaaclab.resolve())
    elif args.command == "request":
        if args.grounded_tag is None:
            parser.error("request requires --grounded-tag")
        result = request(args.tag, args.grounded_tag, args.isaaclab.resolve())
    elif args.command == "approve":
        if args.actor is None:
            parser.error("approve requires --actor")
        result = approve(args.tag, args.actor, args.isaaclab.resolve())
    else:
        result = execute(args.tag, args.isaaclab.resolve())
    print(__import__("json").dumps(result, ensure_ascii=False, allow_nan=False))
    return 1 if result.get("status") == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
