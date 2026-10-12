"""사전 커밋한 grounded 모델 선택·승인·SDK 검사·값 가림 판단 비교를 연결한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(folder))
from rl_triage import grounded_active_profile as A, grounded_evidence as G, triage_tools as T  # noqa: E402
from rl_triage.process_diagnostics import launch_diagnostics  # noqa: E402
import run_base_com_reset as B  # noqa: E402
import grounded_active_model_call as W  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/grounded_active_v3.json"
PRIVATE_ROOT = ROOT / "bench/private/grounded_active"
CODE = {
    "src/rl_triage/grounded_active_profile.py", "src/rl_triage/grounded_evidence.py",
    "src/rl_triage/triage_tools.py", "src/rl_triage/single_run.py",
    "src/rl_triage/nat_functions.py", "src/rl_triage/nat_single_run.py",
    "src/rl_triage/process_diagnostics.py", "evals/grounded_active_model_call.py",
    "evals/run_grounded_active.py", "configs/triage_single_run_grounded_active.yml",
} | B.PROJECT_SOURCE_FILES
FIXED = {
    "schema": "grounded_active_protocol_v1",
    "scope": "development_model_measurement_integration_not_causal_or_policy_recovery_evaluation",
    "model": W.MODEL, "temperature": 0, "semantic_retries": 0, "transport_retries": 0,
    "nat_version": "1.9.0", "post_max_output_tokens": W.POST_MAX_TOKENS,
    "post_reasoning_effort": W.POST_REASONING_EFFORT,
    "provider_cache_status": "not_attested",
    "initial_llm_call_limit": 7, "post_llm_calls": 2,
    "initial_usage_source": W.NAT_USAGE_SOURCE,
    "initial_event_artifact": "type_uuid_token_usage_only_write_once",
    "initial_callback_binding": "explicit_openai_config_callback_before_workflow_build",
    "semantic_tool_error": "domain_writer_rejection_trace_then_exception_stop",
    "schema_tool_error": "strict_input_rejection_before_domain_trace_no_retry",
    "initial_parent_validation": "ordered_events_and_raw_tool_trace_reaggregation",
    "initial_tool_limits": W.TOOL_LIMITS,
    "registry_id": "grounded_active_v1",
    "executable_registry_ids": ["reset_base_com_stability"],
    "unsupported_check_result": "HOLD_without_execution",
    "execution_limit": 1, "execution_arguments": A.FIXED_EXECUTION_ARGUMENTS,
    "supported_probe_task": B.TASK,
    "case_and_probe_relationship": "separate_reset_mechanism_replay_not_original_training_run",
    "post_call_order": "case_id_canonical_sha_first_hex_digit_parity",
    "post_masking": "measurement_values_only_original_receipt_digest_preserved",
    "post_previous_rank_or_decision_included": False,
    "controls": ["same_receipt_preregistered_rule_update", "fixed_initial_ranking"],
    "control_scope": "ranking_update_with_same_initial_hypotheses_and_receipt_not_full_diagnosis",
    "policy_recovery_verified": False,
    "result_wording": {
        "judgment_influence": "측정값을 전달한 새 모델 판단에서만 사전 가설 순위가 바뀌었다. 원인·정확성·정책 회복·우위는 미검증이다.",
        "no_judgment_influence": "측정값에 의한 판단 변화 조건을 충족하지 못했다. 원인·정확성·정책 회복·우위는 미검증이다.",
        "invalid": "호출·승인·실행·출처·종료·판단 계약을 충족하지 못했다. 효과로 집계하지 않는다.",
    },
}


def read(path: Path) -> dict:
    return B.S._read_json(path)


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    B.S._publish_bytes_no_overwrite(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n").encode())


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def original_source_metadata(path: Path) -> tuple[dict, Path, Path]:
    path = path.resolve()
    metadata_root = ROOT / "bench/private"
    if path.parent != metadata_root.resolve() or not path.name.endswith(".meta.json"):
        raise ValueError("original benchmark metadata must be in bench/private")
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(metadata, dict) or metadata.get("task") != B.TASK
            or metadata.get("fault") not in B.ARMS or type(metadata.get("train_exit_code")) is not int
            or metadata["train_exit_code"] != 0 or type(metadata.get("num_envs")) is not int
            or metadata["num_envs"] <= 0
            or type(metadata.get("seed")) is not int
            or not isinstance(metadata.get("case_id"), str)
            or re.fullmatch(r"[a-zA-Z0-9_-]+", metadata["case_id"]) is None):
        raise ValueError("original task/seed/arm metadata is unsupported")
    run_name = f"{metadata['case_id']}_s{metadata['seed']}"
    if path.name != f"{run_name}.meta.json":
        raise ValueError("source metadata name differs from its run identity")
    run_dir = Path(metadata["run_dir"]).resolve()
    allowed = (Path.home() / "IsaacLab/logs/rsl_rl").resolve()
    if allowed not in run_dir.parents or not run_dir.name.endswith("_" + run_name):
        raise ValueError("source run folder does not match its recorded identity")
    telemetry = ROOT / "bench/runs_t1" / f"{run_name}.telemetry.json"
    configuration = run_dir / "params/env.yaml"
    if not telemetry.is_file() or not configuration.is_file():
        raise FileNotFoundError("original telemetry/configuration is missing")
    return metadata, telemetry, configuration


def neutral_configuration(metadata: dict, source: dict) -> tuple[dict, list[dict]]:
    if (source.get("seed") != metadata["seed"]
            or source.get("scene", {}).get("num_envs") != metadata["num_envs"]):
        raise ValueError("saved environment configuration differs from original run metadata")
    configuration = {"task": {"name": metadata["task"], "seed": metadata["seed"]}}
    for name in ("rewards", "observations", "scene", "actions", "terminations", "commands", "curriculum", "events"):
        if name in source:
            configuration[name] = source[name]
    if "sim" in source:
        configuration["simulation"] = source["sim"]
    facts = []
    excluded = {"func", "class_type", "usd_path", "prim_path", "_target_"}

    def selected(value, path):
        if any(str(part) in excluded for part in path):
            return
        if isinstance(value, dict):
            for key in sorted(value):
                selected(value[key], path + [key])
        elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
            for index, child in enumerate(value):
                selected(child, path + [index])
        elif (value is None or type(value) in (bool, int, float)
              or isinstance(value, list) and all(item is None or type(item) in (bool, int, float) for item in value)
              or isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value)
              and "fault" not in value.lower() and value not in B.ARMS):
            facts.append({"id": f"cfg{len(facts):04d}", "path": path})

    selected(configuration, [])
    return configuration, facts


def prepare_inputs(tag: str, metadata_path: Path, case_id: str) -> dict:
    if re.fullmatch(r"case_[a-z0-9_-]{1,32}", case_id) is None:
        raise ValueError("use a neutral case_ alias")
    metadata, telemetry_path, configuration_path = original_source_metadata(metadata_path)
    source_configuration = yaml.load(configuration_path.read_text(encoding="utf-8"), Loader=T._TolerantLoader)
    configuration, facts = neutral_configuration(metadata, source_configuration)
    workspace = run_folder("inputs_" + tag)
    workspace.mkdir(parents=True, exist_ok=False)
    case_dir = workspace / "cases" / case_id
    case_dir.mkdir(parents=True)
    telemetry_raw = telemetry_path.read_bytes()
    B.S._publish_bytes_no_overwrite(case_dir / "telemetry.json", telemetry_raw)
    write_new(case_dir / "configuration.json", configuration)
    write_new(case_dir / "context.json", {
        "schema": "single_run_context_v1", "telemetry_sha256": W.file_sha(telemetry_path),
        "configuration_sha256": W.file_sha(case_dir / "configuration.json"), "facts": facts,
    })
    write_new(workspace / "reference/telemetry.json", {"reference_status": "absent", "summary": {}, "series": {}})
    lineage = {"schema": "grounded_active_input_lineage_v1", "case_id": case_id,
               "metadata_path": str(metadata_path.resolve()), "metadata_sha256": W.file_sha(metadata_path),
               "original_telemetry_path": str(telemetry_path), "original_telemetry_sha256": W.file_sha(telemetry_path),
               "original_configuration_path": str(configuration_path),
               "original_configuration_sha256": W.file_sha(configuration_path),
               "derived_configuration_sha256": W.file_sha(case_dir / "configuration.json"),
               "case_input_context_sha256": W.file_sha(case_dir / "context.json"),
               "input_source_seed": metadata["seed"], "probe_seed": A.FIXED_EXECUTION_ARGUMENTS["seed"],
               "original_source_code_attestation": "not_recorded_by_legacy_training_harness",
               "source_relationship": FIXED["case_and_probe_relationship"], "policy_recovery_verified": False}
    write_new(workspace / "host_input_lineage.json", lineage)
    T.WORKSPACE = workspace
    catalog = G.evidence_catalog(case_id)
    return {"workspace": str(workspace), "case_id": case_id, "facts": len(facts),
            "catalog_sha256": catalog["catalog_sha256"], "model_calls": 0, "gpu_runs": 0}


def validate_input_lineage(workspace: Path, case_id: str, metadata_path: Path) -> dict:
    lineage = read(workspace / "host_input_lineage.json")
    metadata, telemetry, configuration_path = original_source_metadata(metadata_path)
    expected_lineage = {
        "schema": "grounded_active_input_lineage_v1", "case_id": case_id,
        "metadata_path": str(metadata_path.resolve()), "metadata_sha256": W.file_sha(metadata_path),
        "original_telemetry_path": str(telemetry), "original_telemetry_sha256": W.file_sha(telemetry),
        "original_configuration_path": str(configuration_path),
        "original_configuration_sha256": W.file_sha(configuration_path),
        "derived_configuration_sha256": W.file_sha(workspace / f"cases/{case_id}/configuration.json"),
        "case_input_context_sha256": W.file_sha(workspace / f"cases/{case_id}/context.json"),
        "input_source_seed": metadata["seed"], "probe_seed": A.FIXED_EXECUTION_ARGUMENTS["seed"],
        "original_source_code_attestation": "not_recorded_by_legacy_training_harness",
        "source_relationship": FIXED["case_and_probe_relationship"], "policy_recovery_verified": False,
    }
    if lineage != expected_lineage:
        raise ValueError("case lineage identity, seed roles or source binding differs")
    paths = {"metadata": metadata_path, "original_telemetry": telemetry,
             "original_configuration": configuration_path,
             "derived_configuration": workspace / f"cases/{case_id}/configuration.json",
             "case_input_context": workspace / f"cases/{case_id}/context.json"}
    for name, path in paths.items():
        if lineage[f"{name}_sha256"] != W.file_sha(path):
            raise ValueError("original input lineage source changed")
    if W.file_sha(workspace / f"cases/{case_id}/telemetry.json") != W.file_sha(telemetry):
        raise ValueError("case telemetry differs from original source run")
    source = yaml.load(configuration_path.read_text(encoding="utf-8"), Loader=T._TolerantLoader)
    expected, facts = neutral_configuration(metadata, source)
    if read(paths["derived_configuration"]) != expected or read(paths["case_input_context"])["facts"] != facts:
        raise ValueError("case configuration/context differs from registered neutral conversion")
    return metadata


def protocol_path(path: Path) -> Path:
    candidate = path.resolve()
    if candidate != PROTOCOL.resolve() or path.is_symlink():
        raise ValueError("only the current repository grounded-active protocol is supported")
    return candidate


def prepare(path: Path = PROTOCOL) -> dict:
    target = protocol_path(path)
    doc = {**deepcopy(FIXED), "registry_sha256": A._sha(A.ACTIVE_CHECKS),
           "base_com_protocol": {"path": B.PROTOCOL.relative_to(ROOT).as_posix(),
                                 "sha256": W.file_sha(B.PROTOCOL)},
           "source_files": {name: sha_lf(ROOT / name) for name in sorted(CODE)}}
    write_new(target, doc)
    return doc


def verify_protocol(path: Path = PROTOCOL) -> tuple[dict, str]:
    path = protocol_path(path)
    doc = read(path)
    if set(doc) != set(FIXED) | {"registry_sha256", "base_com_protocol", "source_files"}:
        raise ValueError("protocol fields changed")
    if any(doc[key] != value for key, value in FIXED.items()):
        raise ValueError("fixed grounded active protocol changed")
    if doc["registry_sha256"] != A._sha(A.ACTIVE_CHECKS):
        raise ValueError("active registry digest changed")
    if doc["base_com_protocol"] != {
            "path": B.PROTOCOL.relative_to(ROOT).as_posix(), "sha256": W.file_sha(B.PROTOCOL)}:
        raise ValueError("base COM protocol binding changed")
    if doc["source_files"] != {name: sha_lf(ROOT / name) for name in sorted(CODE)}:
        raise ValueError("source file binding changed")
    commit = B.P.verify_committed(ROOT, CODE | {path.relative_to(ROOT).as_posix(),
                                              B.PROTOCOL.relative_to(ROOT).as_posix()})
    return doc, commit


def run_folder(tag: str) -> Path:
    if not isinstance(tag, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", tag) is None:
        raise ValueError("tag must be a bounded lowercase token")
    folder = PRIVATE_ROOT / tag
    if folder.is_symlink() or folder.resolve().parent != PRIVATE_ROOT.resolve():
        raise ValueError("private run folder escaped its root")
    return folder


def wsl_path(path: Path) -> str:
    absolute = path.resolve().as_posix()
    if not re.fullmatch(r"[A-Za-z]:/.*", absolute) or any(char in absolute for char in "\r\n\x00"):
        raise ValueError("native Windows path required")
    return f"/mnt/{absolute[0].lower()}/{absolute[3:]}"


def stop_owned_worker(output: Path, packet: Path) -> bool:
    identity_path = output.with_suffix(".pid.json")
    if not identity_path.exists():
        return False
    identity = read(identity_path)
    if (type(identity.get("pid")) is not int or identity["pid"] <= 0
            or type(identity.get("start_ticks")) is not int
            or identity.get("packet_path") != wsl_path(packet)
            or identity.get("output_path") != wsl_path(output)
            or identity.get("packet_sha256") != W.file_sha(packet)):
        raise RuntimeError("owned model worker identity mismatch")
    script = """import os,sys,signal,time
from pathlib import Path
p,start=int(sys.argv[1]),int(sys.argv[2])
base=Path('/proc')/str(p)
if not base.exists(): sys.exit(0)
raw=(base/'stat').read_text(); observed=int(raw[raw.rfind(')')+2:].split()[19])
argv=(base/'cmdline').read_bytes().split(b'\\x00')
if observed!=start or sys.argv[3].encode() not in argv or sys.argv[4].encode() not in argv or not any(x.endswith(b'grounded_active_model_call.py') for x in argv): sys.exit(4)
os.kill(p,signal.SIGTERM)
for _ in range(20):
    if not base.exists(): sys.exit(0)
    raw=(base/'stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    if int(fields[19])!=start or fields[0]=='Z': sys.exit(0)
    time.sleep(0.1)
sys.exit(5)
"""
    command = shlex.join(["python3", "-c", script, str(identity["pid"]), str(identity["start_ticks"]),
                          wsl_path(packet), wsl_path(output)])
    subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
                   check=True, capture_output=True, timeout=15)
    identity_path.unlink(missing_ok=True)
    return True


def fresh_model(folder: Path, stage: str, packet: dict, workspace: Path) -> dict:
    packet_path, output = folder / f"{stage}.packet.json", folder / f"{stage}.model.json"
    if output.exists() or output.with_suffix(".pid.json").exists():
        raise FileExistsError("fresh model stage has existing artifacts")
    write_new(packet_path, packet)
    argv = [wsl_path(ROOT / "evals/grounded_active_model_call.py"), "--stage", stage,
            "--packet", wsl_path(packet_path), "--output", wsl_path(output),
            "--workspace", wsl_path(workspace)]
    command = '. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" ' + shlex.join(argv)
    timeout = 1830 if stage == "initial" else 330
    launch, cleanup_ok, proc = None, True, None
    started = time.perf_counter()
    try:
        proc = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
                              capture_output=True, cwd=ROOT, timeout=timeout)
        launch = launch_diagnostics(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as error:
        launch = launch_diagnostics(None, error.stdout, error.stderr, timed_out=True)
    except OSError:
        launch = launch_diagnostics(None)
    if proc is None or output.with_suffix(".pid.json").exists():
        try:
            cleanup_ok = stop_owned_worker(output, packet_path)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            cleanup_ok = False
    record, record_error = None, None
    if output.exists():
        try:
            record = read(output)
        except (OSError, ValueError):
            record_error = "invalid_model_record"
    expected_messages = W.canonical_sha(W.post_messages(packet)) if stage != "initial" else None
    valid = (proc is not None and proc.returncode == 0 and cleanup_ok and record is not None
             and record.get("schema") == "grounded_active_model_call_v1"
             and record.get("packet_sha256") == W.file_sha(packet_path)
             and record.get("stage") == stage and record.get("valid") is True
             and record.get("model") == W.MODEL and record.get("temperature") == 0
             and record.get("semantic_retries") == 0 and record.get("transport_retries") == 0
             and record.get("client_cache_used") is False and record.get("saved_response_cached") is False)
    if stage != "initial":
        valid = bool(valid and record.get("messages_sha256") == expected_messages
                     and record.get("provider_response_id") and record.get("llm_calls_started") == 1
                     and record.get("llm_calls_finished") == 1
                     and isinstance(record.get("provider_model"), str) and bool(record["provider_model"].strip())
                     and record.get("max_output_tokens") == W.POST_MAX_TOKENS
                     and record.get("reasoning_effort") == W.POST_REASONING_EFFORT
                     and record.get("provider_cache_status") == "not_attested")
    elif valid:
        try:
            W.validate_initial_execution_record(output, record)
        except (OSError, KeyError, TypeError, ValueError):
            valid, record_error = False, "invalid_initial_execution_artifact"
    outcome = {"schema": "grounded_active_model_outcome_v1", "stage": stage, "valid": bool(valid),
               "worker_cleanup_verified": cleanup_ok, "launch": launch,
               "model_record_sha256": W.file_sha(output) if output.exists() else None,
               "record_error": record_error,
               "parent_wall_s": time.perf_counter() - started}
    write_new(folder / f"{stage}.outcome.json", outcome)
    return {"record": record, "outcome": outcome}


def initial(tag: str, workspace: Path, case_id: str, metadata_path: Path) -> dict:
    _, commit = verify_protocol()
    workspace = workspace.resolve()
    if ROOT not in workspace.parents or workspace.is_symlink():
        raise ValueError("case workspace must be inside the repository")
    T.WORKSPACE = workspace
    metadata = validate_input_lineage(workspace, case_id, metadata_path)
    task, source_seed, arm = metadata["task"], metadata["seed"], metadata["fault"]
    catalog = G.evidence_catalog(case_id)
    if catalog["context_status"] != "provided":
        raise ValueError("source-bound configuration/context is required")
    if task != B.TASK or arm not in B.ARMS or type(source_seed) is not int:
        raise ValueError("only the declared stock task development replay is supported")
    folder = run_folder(tag)
    folder.mkdir(parents=True, exist_ok=False)
    sources = {name: {"path": f"cases/{case_id}/{name}.json",
                      "sha256": W.file_sha(T._safe(workspace / f"cases/{case_id}/{name}.json"))}
               for name in ("telemetry", "configuration", "context")}
    spec = {"schema": "grounded_active_host_spec_v1", "case_id": case_id,
            "workspace": str(workspace), "task": task, "source_seed": source_seed, "arm": arm,
            "sources": sources, "source_commit_sha": commit,
            "protocol_sha256": W.file_sha(PROTOCOL),
            "metadata_path": str(metadata_path.resolve()), "metadata_sha256": W.file_sha(metadata_path),
            "input_lineage_sha256": W.file_sha(workspace / "host_input_lineage.json"),
            "source_relationship": FIXED["case_and_probe_relationship"],
            "probe_seed": A.FIXED_EXECUTION_ARGUMENTS["seed"],
            "source_task_and_seed_attestation": "original_training_metadata_and_configuration_not_original_code"}
    write_new(folder / "host_spec.json", spec)
    packet = {"schema": "grounded_active_initial_packet_v1", "case_id": case_id,
              "catalog_sha256": catalog["catalog_sha256"], "workflow_config_sha256_lf": sha_lf(W.CONFIG)}
    result = fresh_model(folder, "initial", packet, workspace)
    if not result["outcome"]["valid"]:
        result = {"schema": "grounded_active_development_result_v1", "case_id": case_id,
                  "source_commit_sha": commit, "protocol_sha256": spec["protocol_sha256"],
                  "status": "HOLD", "reason": "initial_model_invalid", "gpu_runs": 0,
                  "policy_recovery_verified": False, "cause_confirmed": False,
                  "cost": cost_summary(folder, {}, None),
                  "result_wording": FIXED["result_wording"]["invalid"]}
        write_new(folder / "result.json", result)
        return result
    assessment = A._load_bound("grounded_active_assessments", f"{case_id}.json")
    if result["record"]["assessment_sha256"] != A._sha(assessment):
        raise ValueError("fresh model assessment binding mismatch")
    manifest = {"schema": "grounded_active_case_manifest_v1", "case_id": case_id,
                "public_case_alias": case_id, "sources": sources,
                "assessment_sha256": A._sha(assessment), "task": task, "seed": source_seed,
                "protocol_sha256": W.file_sha(PROTOCOL), "source_commit_sha": commit,
                "private_executor_binding_sha256": A._sha(spec)}
    binding = A.validate_case_manifest(case_id, manifest)
    write_new(folder / "case_binding.json", binding)
    check = assessment["next_check"]["registry_id"]
    if check != "reset_base_com_stability":
        result = {"schema": "grounded_active_development_result_v1", "case_id": case_id,
                  "source_commit_sha": commit, "protocol_sha256": spec["protocol_sha256"],
                  "status": "HOLD", "reason": "selected_check_execution_unsupported", "gpu_runs": 0,
                  "policy_recovery_verified": False, "cause_confirmed": False,
                  "cost": cost_summary(folder, {}, None),
                  "result_wording": "새 모델이 선택한 검사의 실행 adapter를 지원하지 않아 HOLD로 종료했다. 측정 효과로 집계하지 않는다."}
        write_new(folder / "result.json", result)
        return result
    request = A.request_execution(case_id, binding["case_source_binding_sha256"], check,
                                  deepcopy(A.FIXED_EXECUTION_ARGUMENTS))
    write_new(folder / "request_binding.json", request)
    return {"status": "approval_pending", "case_id": case_id, "effect_key": request["effect_key"], "gpu_runs": 0}


def load_run(tag: str) -> tuple[Path, dict, dict]:
    _, commit = verify_protocol()
    folder = run_folder(tag)
    spec, binding = read(folder / "host_spec.json"), read(folder / "case_binding.json")
    if (spec["source_commit_sha"] != commit or binding["source_commit_sha"] != commit
            or spec["protocol_sha256"] != W.file_sha(PROTOCOL)
            or binding["private_executor_binding_sha256"] != A._sha(spec)
            or binding["protocol_sha256"] != spec["protocol_sha256"]
            or binding["task"] != spec["task"] or binding["seed"] != spec["source_seed"]
            or binding["case_id"] != spec["case_id"] or binding["sources"] != spec["sources"]):
        raise ValueError("host spec/source/protocol binding changed")
    T.WORKSPACE = Path(spec["workspace"])
    if (spec["metadata_sha256"] != W.file_sha(Path(spec["metadata_path"]))
            or spec["input_lineage_sha256"] != W.file_sha(T.WORKSPACE / "host_input_lineage.json")):
        raise ValueError("original source metadata/lineage changed")
    metadata = validate_input_lineage(T.WORKSPACE, spec["case_id"], Path(spec["metadata_path"]))
    if (metadata["task"] != spec["task"] or metadata["seed"] != spec["source_seed"]
            or metadata["fault"] != spec["arm"] or spec["probe_seed"] != A.FIXED_EXECUTION_ARGUMENTS["seed"]):
        raise ValueError("original task/seed and replay arm binding changed")
    # 원본 입력 검증은 승인을 소비하기 전에도 반복한다.
    A.validate_case_manifest(spec["case_id"], {key: value for key, value in binding.items()
                                             if key != "case_source_binding_sha256"})
    return folder, spec, binding


def approve(tag: str, actor: str) -> dict:
    folder, spec, _ = load_run(tag)
    approved = A.approve_execution(spec["case_id"], read(folder / "request_binding.json"), actor)
    write_new(folder / "approval_binding.json", approved)
    return approved


def ranking_controls(assessment: dict, receipt: dict) -> dict:
    prior = [row["id"] for row in sorted(assessment["hypotheses"], key=lambda row: row["rank"])]
    measurements = {row["measurement_id"]: row["value"] for row in receipt["measurements"]}
    predictions = {row["hypothesis_id"]: row["predictions"]
                   for row in assessment["next_check"]["hypothesis_predictions"]}
    survivors = []
    if receipt["execution_valid"]:
        survivors = [identifier for identifier in prior if all(
            G._relation_holds(measurements[row["measurement_id"]], row["relation"], row["operand"]["value"])
            for row in predictions[identifier])]
    return {"schema": "grounded_active_ranking_controls_v1",
            "scope": FIXED["control_scope"], "receipt_sha256": A._sha(receipt),
            "rule_update_ranking": survivors + [identifier for identifier in prior if identifier not in survivors],
            "fixed_no_update_ranking": prior, "additional_gpu_runs": 0, "policy_recovery_verified": False}


def cost_summary(folder: Path, calls: dict, parent: dict | None) -> dict:
    records = {}
    initial_path = folder / "initial.model.json"
    if initial_path.exists():
        try:
            records["initial"] = read(initial_path)
        except (OSError, ValueError):
            records["initial"] = None
    else:
        records["initial"] = None
    records.update({stage: call["record"] for stage, call in calls.items()})
    stages = {}
    for stage in ("initial", "actual", "masked"):
        record = records.get(stage)
        if record is None:
            stages[stage] = {"record_status": "absent_or_invalid", "input_tokens": None,
                             "output_tokens": None, "total_tokens": None, "wall_s": None,
                             "cpu_s": None, "calls_started": None, "calls_finished": None}
            continue
        stages[stage] = {
            "record_status": "valid" if record.get("valid") is True else "invalid",
            **{key: record.get(key) for key in ("input_tokens", "output_tokens", "total_tokens", "wall_s", "cpu_s")},
            "calls_started": record.get("llm_calls_started"),
            "calls_finished": record.get("llm_calls_finished"),
        }
    token_keys = ("input_tokens", "output_tokens", "total_tokens")
    complete = all(
        all(type(stage[key]) is int and stage[key] >= 0 for key in token_keys)
        and stage["input_tokens"] > 0
        and stage["total_tokens"] >= stage["input_tokens"] + stage["output_tokens"]
        for stage in stages.values())
    execution = parent.get("execution", {}) if parent is not None else {}
    return {
        "schema": "grounded_active_cost_v1", "model_stages": stages,
        "usage_coverage": "complete" if complete else "partial",
        **{key: sum(row[key] for row in stages.values()) if complete else None for key in token_keys},
        "sdk_parent_wall_s": execution.get("wall_s"),
        "sdk_wall_scope": "launch_to_exit_and_owned_job_empty_verification",
        "sdk_gpu_time_s": None, "gpu_time_status": "not_measured",
        "gpu_exclusive": False, "performance_measurement": False,
        "money_cost": None, "money_cost_status": "not_measured",
        "controls_additional_model_calls": 0, "controls_additional_gpu_runs": 0,
        "controls_cost_scope": "same_model_initial_hypotheses_and_receipt_shared_not_independent_initial_diagnosis",
    }


def execute(tag: str, isaaclab: Path) -> dict:
    folder, spec, binding = load_run(tag)
    if (folder / "result.json").exists():
        raise FileExistsError("finished runs are never overwritten or rerun")
    if B.M.query_gpu(B.FIXED_PROTOCOL_VALUES["gpu_admission_profile"])["busy"]:
        raise RuntimeError("GPU busy; approval not consumed")
    case_id = spec["case_id"]
    approved = read(folder / "approval_binding.json")
    consumed = A.consume_approval(case_id, approved)
    write_new(folder / "consumed_binding.json", consumed)
    sdk_tag = f"ga_{tag}"
    raw_parent, body, verified = None, None, False
    sdk_folder = ROOT / "bench/private/base_com_reset" / sdk_tag
    parent_path = sdk_folder / "parent_receipt.json"
    try:
        raw_parent = B.run(sdk_tag, spec["arm"], isaaclab)
        if raw_parent != read(parent_path):
            raise ValueError("SDK returned receipt differs from its saved artifact")
        validated = B.validate_manifest(sdk_folder / "manifest.json", repo_root=ROOT,
                                        isaaclab_root=isaaclab.resolve())
        body = read(sdk_folder / "body.json")
        verified = (raw_parent["status"] == "pass" and raw_parent["arm"] == spec["arm"]
                    and raw_parent["project_commit_sha"] == spec["source_commit_sha"]
                    and raw_parent["protocol_sha256"] == W.file_sha(B.PROTOCOL)
                    and raw_parent["body_sha256"] == W.file_sha(sdk_folder / "body.json")
                    and raw_parent["manifest_sha256"] == W.file_sha(sdk_folder / "manifest.json")
                    and not B.verify_body(body, validated, raw_parent["execution"], spec["source_commit_sha"]))
    except Exception as error:
        write_new(folder / "execution_failure.json", {"error_type": type(error).__name__, "side_effect_retry": False})
    measurements = []
    for measurement_id, definition in sorted(A.ACTIVE_CHECKS["reset_base_com_stability"]["measurements"].items()):
        row = body["observer_report"]["measurements"][measurement_id] if verified else None
        measurements.append({"measurement_id": measurement_id, "unit": definition["unit"],
                             "state": "present" if verified else "unavailable",
                             "value": row["value"] if row is not None else None})
    execution = raw_parent["execution"] if raw_parent is not None else {}
    receipt = {"schema": "grounded_active_execution_receipt_v1", "case_id": case_id,
               "effect_key": consumed["effect_key"],
               "case_source_binding_sha256": binding["case_source_binding_sha256"],
               "request_sha256": consumed["request_sha256"], "approval_sha256": consumed["approval_sha256"],
               "consumed_sha256": consumed["consumed_sha256"], "registry_id": "reset_base_com_stability",
               "check_definition_sha256": A._sha(A.ACTIVE_CHECKS["reset_base_com_stability"]),
               "parent_status": "pass" if verified else "invalid",
               "parent_receipt_sha256": W.file_sha(parent_path) if parent_path.exists() else None,
               "body_sha256": raw_parent["body_sha256"] if raw_parent else None,
               "natural_exit": execution.get("natural_exit", False),
               "cleanup_verified": execution.get("cleanup_verified", False), "source_verified": verified,
               "coverage_status": "complete" if verified else "unavailable", "measurements": measurements,
               "model_selected": False, "diagnosis_confirmed": False, "cause_confirmed": False,
               "recovery_confirmed": False, "policy_recovery": False}
    normalized = A.normalize_execution_receipt(case_id, binding["case_source_binding_sha256"],
                                              consumed["effect_key"], receipt)
    write_new(folder / "execution_receipt.json", receipt)
    assessment = A._load_bound("grounded_active_assessments", f"{case_id}.json")
    stored_receipt = {key: value for key, value in normalized.items() if key != "normalized_receipt_sha256"}
    controls = ranking_controls(assessment, stored_receipt)
    write_new(folder / "ranking_controls.json", controls)
    if not normalized["execution_valid"]:
        result = {"schema": "grounded_active_development_result_v1", "case_id": case_id,
                  "status": "HOLD", "execution_valid": False, "post_model_calls": 0,
                  "judgment_influence": {"judgment_influence_proven": False, "reason": "invalid_execution"},
                  "policy_recovery_verified": False, "cause_confirmed": False,
                  "cost": cost_summary(folder, {}, raw_parent),
                  "controls": controls, "result_wording": FIXED["result_wording"]["invalid"]}
        write_new(folder / "result.json", result)
        return result
    packets = {stage: A.build_post_measurement_packet(case_id, normalized, value_masked=stage == "masked")
               for stage in ("actual", "masked")}
    order = ["actual", "masked"] if int(A._sha(case_id)[0], 16) % 2 == 0 else ["masked", "actual"]
    calls = {}
    for stage in order:
        calls[stage] = fresh_model(folder, stage, packets[stage], Path(spec["workspace"]))
        if not calls[stage]["outcome"]["worker_cleanup_verified"]:
            break
    valid_pair = all(stage in calls and calls[stage]["outcome"]["valid"] for stage in packets)
    if valid_pair:
        actual_record, masked_record = calls["actual"]["record"], calls["masked"]["record"]
        valid_pair = (
            actual_record["provider_response_id"] != masked_record["provider_response_id"]
            and isinstance(actual_record.get("provider_model"), str)
            and bool(actual_record["provider_model"].strip())
            and actual_record["provider_model"] == masked_record.get("provider_model")
            and actual_record.get("llm_calls_finished") == masked_record.get("llm_calls_finished") == 1)
    proof = {"judgment_influence_proven": False, "reason": "invalid_model_pair"}
    if valid_pair:
        proof = A.build_judgment_influence_proof(assessment, packets["actual"], calls["actual"]["record"]["decision"],
                                                packets["masked"], calls["masked"]["record"]["decision"])
    result = {"schema": "grounded_active_development_result_v1", "case_id": case_id,
              "effect_key": consumed["effect_key"], "source_commit_sha": spec["source_commit_sha"],
              "sdk_tag": sdk_tag, "execution_valid": normalized["execution_valid"],
              "fresh_initial_model_verified": True, "model_selected_check_executed": bool(verified),
              "valid_post_model_pair": valid_pair, "call_order": order, "judgment_influence": proof,
              "controls": controls, "policy_recovery_verified": False, "cause_confirmed": False,
              "correctness_verified": False, "superiority_verified": False,
              "cost": cost_summary(folder, calls, raw_parent),
              "source_relationship": spec["source_relationship"],
              "input_source_seed": spec["source_seed"], "probe_seed": spec["probe_seed"],
              "original_case_source_task_seed_attestation": spec["source_task_and_seed_attestation"],
              "status": "HOLD", "result_wording": FIXED["result_wording"][
                  "judgment_influence" if proof["judgment_influence_proven"] else
                  "no_judgment_influence" if verified and valid_pair else "invalid"]}
    write_new(folder / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    inputs = commands.add_parser("inputs")
    inputs.add_argument("--tag", required=True)
    inputs.add_argument("--source-metadata", type=Path, required=True)
    inputs.add_argument("--case", required=True)
    start = commands.add_parser("initial")
    start.add_argument("--tag", required=True)
    start.add_argument("--workspace", type=Path, required=True)
    start.add_argument("--case", required=True)
    start.add_argument("--source-metadata", type=Path, required=True)
    permit = commands.add_parser("approve")
    permit.add_argument("--tag", required=True)
    permit.add_argument("--actor", required=True)
    run = commands.add_parser("execute")
    run.add_argument("--tag", required=True)
    run.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        result = {"status": "protocol_written_commit_before_model_or_gpu", "path": str(PROTOCOL)}
    elif args.command == "inputs":
        result = prepare_inputs(args.tag, args.source_metadata, args.case)
    elif args.command == "initial":
        result = initial(args.tag, args.workspace, args.case, args.source_metadata)
    elif args.command == "approve":
        result = {"status": "approved", "effect_key": approve(args.tag, args.actor)["effect_key"]}
    else:
        record = execute(args.tag, args.isaaclab)
        result = {"status": record["status"], "judgment_influence": record["judgment_influence"],
                  "policy_recovery_verified": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
