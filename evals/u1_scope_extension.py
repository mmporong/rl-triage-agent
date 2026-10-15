"""U1 범위 부족 뒤 전체 보상 함수군 교정을 선택하고 재시험 입력을 검증한다.

이미 본 U1 결과를 본 뒤 설계한 개발용 확장이다. 새 seed 2051에서 원 결함을 확인한
뒤 seed 3051로 처음부터 학습한다. NONE에는 같은 교정을 적용해 300회 학습과 고정
평가에서 healthy를 유지해야 한다. 결과는 아직 측정하지 않았다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import reward_family_correction as F  # noqa: E402
import run_training_correction as V3  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import demo_correction_u1 as STRICT  # noqa: E402
import record_evidence_media as MEDIA  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/u1_scope_extension_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task5"
TAG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
FAULT_SEED = 2051
RESTART_SEED = 3051
GPU_PROFILE = "base_com_reset_readback_v2"
ARMS = {"none_family_control": {"fault": "NONE", "seed": FAULT_SEED},
        "u1_family_restart": {"fault": "U1", "seed": RESTART_SEED}}
CODE = {
    "bench/reward_family_correction.py", "bench/demo_correction_u1_family.py",
    "evals/u1_scope_extension.py", "bench/protocols/u1_scope_extension_v1.json",
    "bench/train_with_recovery_mode.py", "bench/upstream_faults.py", "bench/upstream_isaac.py",
    "evals/fixed_eval.py", "evals/u1_scope_source.py",
    "bench/protocols/u1_scope_extension_source/e2e_demo_u1_v1.json",
}
CODE |= set(__import__("e2e_demo_u1").CODE)

sha = V3.sha
write_new = V3.write_new
load_config = V3.load_config


def choose_correction(fault: str, scope_verdict: str, uncovered_terms: list[str], family: str) -> str:
    """확정 U1 함수군 전체와 일치할 때만 확장 교정을 선택한다."""
    if fault != "U1" or family != F.REGISTRY[F.CORRECTION_ID]["fault_family"]:
        raise ValueError("unregistered fault family")
    if scope_verdict != "scope_insufficient":
        raise ValueError("full-family correction requires scope_insufficient")
    if set(uncovered_terms or ()) != set(F.FUNCTIONS) - {"track_lin_vel_xy_exp"}:
        raise ValueError("scope contains an unregistered reward function")
    return F.CORRECTION_ID


def fixed_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict, isaaclab: Path | None = None) -> dict:
    if protocol.get("schema") != "u1_scope_extension_protocol_v1":
        raise ValueError("protocol schema differs")
    expected = {"fault_seed": FAULT_SEED, "restart_seed": RESTART_SEED, "num_envs": 4096,
                "max_iterations": 300, "num_steps_per_env": 24, "correction_id": F.CORRECTION_ID,
                "fault_functions": list(F.FUNCTIONS), "action_scale": 0.25, "rsl_rl_version": "2.3.3",
                "isaaclab_commit": "90b79bb2d44feb8d833f260f2bf37da3487180ba",
                "gpu_admission_profile": GPU_PROFILE, "api_calls": 0}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("protocol fixed fields differ")
    selection = protocol.get("selection_contract", {})
    choose_correction("U1", selection.get("required_scope_verdict"), selection.get("registered_uncovered_terms"),
                      selection.get("fault_family"))
    evidence = protocol.get("scope_evidence_contract", {})
    if (evidence.get("probe_observed_terms") != ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"]
            or evidence.get("source_registered_terms") != list(F.FUNCTIONS)
            or evidence.get("model_loop") != "inactive"
            or evidence.get("source_only_enabled_steps") != ["begin", "train_NONE", "train_FAULT", "alarm",
                                                                "evaluate", "reference", "diagnose-scope"]):
        raise ValueError("scope evidence contract differs")
    if isaaclab is not None and protocol.get("correction_source_contract") != F.source_contract(isaaclab):
        raise ValueError("protocol source contract differs from Isaac Lab")
    return protocol


def validate_readback(row: dict, fault: str, correction: str | None) -> None:
    if row.get("schema") != "reward_family_correction_readback_v1" or correction != F.CORRECTION_ID:
        raise ValueError("reward family correction readback selection differs")
    if row.get("selected_correction") != correction or row.get("registered_functions") != list(F.FUNCTIONS):
        raise ValueError("reward family readback target set differs")
    clean, injected, after = (row.get(key, {}) for key in ("before_fault", "after_fault", "after_correction"))
    contract = fixed_protocol()["correction_source_contract"]
    if clean != contract["functions"] or after != clean:
        raise ValueError("reward family was not fully restored")
    if any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None for stage in (clean, injected, after)
           for value in stage.values()):
        raise ValueError("reward family fingerprint is invalid")
    binding = row.get("source_binding")
    expected_binding = {"module": contract["module"], "source": contract["source"],
                        "source_sha256": contract["source_sha256"]}
    if (not isinstance(binding, dict) or any(binding.get(key) != value for key, value in expected_binding.items())
            or not str(binding.get("live_source_path", "")).replace("\\", "/").endswith(contract["source"])):
        raise ValueError("reward family source binding differs")
    changed_by_fault = {name for name in F.FUNCTIONS if injected.get(name) != clean[name]}
    expected = set() if fault == "NONE" else set(F.FUNCTIONS)
    if changed_by_fault != expected:
        raise ValueError("fault did not change the exact registered family")
    expected_restored = [] if fault == "NONE" else list(F.FUNCTIONS)
    if row.get("restored_targets") != expected_restored:
        raise ValueError("restored target list differs")
    if row.get("unrelated_changed_after_fault") or row.get("unrelated_changed_after_correction"):
        raise ValueError("unrelated reward function changed")


def validate_authorization(scope_path: Path, approval_path: Path, isaaclab: Path) -> tuple[dict, dict]:
    import u1_scope_source as SOURCE

    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    result = scope.get("result", {})
    uncovered = result.get("uncovered_terms")
    correction = choose_correction(scope.get("fault"), result.get("verdict"), uncovered,
                                   scope.get("fault_family"))
    if scope.get("schema") != "u1_scope_extension_scope_v1" or scope.get("seed") != FAULT_SEED:
        raise ValueError("scope record is not the preregistered new-seed U1 check")
    SOURCE.validate_scope_record(scope_path, isaaclab)
    if (scope.get("observed_probe_terms") != ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"]
            or scope.get("source_registered_terms") != list(F.FUNCTIONS)):
        raise ValueError("scope record does not separate measured and source-registered terms")
    expected = {"schema": "human_correction_approval_v1", "decision": "approve", "correction_id": correction,
                "scope_sha256": sha(scope_path), "identity_attested": False}
    if approval != expected:
        raise ValueError("approval is not exactly bound to the scope record and correction")
    return scope, approval


def _folder(tag: str) -> Path:
    if not TAG_RE.fullmatch(tag):
        raise ValueError("invalid tag")
    path = PRIVATE / "runs" / tag
    if path.is_symlink() or path.resolve().parent != (PRIVATE / "runs").resolve():
        raise ValueError("output escapes task5 run root")
    return path


def manifest_for(tag: str, arm: str, isaaclab: Path, source_env: Path, source_agent: Path,
                 source_commit: str) -> dict:
    if arm not in ARMS:
        raise ValueError("unregistered arm")
    spec = ARMS[arm]
    folder = _folder(tag) / arm / "train"
    return {
        "schema": "training_correction_manifest_v2", "tag": tag, "seed": spec["seed"], "arm": arm,
        "recovery_mode": "restart", "source_commit_sha": source_commit,
        "protocol_sha256_lf": sha(PROTOCOL, lf=True), "isaaclab_root": str(isaaclab.resolve()),
        "protocol_path": str(PROTOCOL.resolve()), "fault": spec["fault"], "correction_id": F.CORRECTION_ID,
        "num_envs": 4096, "max_iterations": 300, "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "run_name": f"{tag}_{arm}", "output_dir": str(folder.resolve()), "final_iteration": 299,
        "resume_checkpoint_path": None, "resume_checkpoint_sha256": None, "resume_iteration": None,
        "load_run_regex": None, "checkpoint_regex": None,
        "source_env_config_path": str(source_env.resolve()), "source_agent_config_path": str(source_agent.resolve()),
        "config_source_fault": spec["fault"], "init_noise_std": None,
    }


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != R.MANIFEST_FIELDS or manifest.get("arm") not in ARMS:
        raise ValueError("manifest schema or arm differs")
    expected_path = _folder(manifest["tag"]) / manifest["arm"] / "train/manifest.json"
    if path.is_symlink() or path.resolve() != expected_path.resolve():
        raise ValueError("manifest is outside its registered arm")
    expected = manifest_for(manifest["tag"], manifest["arm"], Path(manifest["isaaclab_root"]),
                            Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"]),
                            manifest["source_commit_sha"])
    if manifest != expected:
        raise ValueError("manifest differs from its fixed arm")
    folder = _folder(manifest["tag"])
    validate_authorization(folder / "scope.json", folder / "approval.json", Path(manifest["isaaclab_root"]))
    consumed = json.loads((folder / "consumed.json").read_text(encoding="utf-8"))
    if consumed != {"approval_sha256": sha(folder / "approval.json"), "arms": list(ARMS),
                    "train_limit_per_arm": 1, "eval_limit": 1}:
        raise ValueError("approval consumption contract differs")
    scope, _ = _validate_source_inputs(json.loads((folder / "prereg.json").read_text(encoding="utf-8")),
                                       Path(manifest["isaaclab_root"]))
    inputs = _source_inputs(scope)
    if (manifest["source_env_config_path"] != inputs["source_env"]["path"]
            or manifest["source_agent_config_path"] != inputs["source_agent"]["path"]
            or manifest["source_commit_sha"] != scope["bindings"]["source_commit_sha"]):
        raise ValueError("manifest source inputs differ from the validated scope")
    if (sha(Path(manifest["source_env_config_path"])) != inputs["source_env"]["sha256"]
            or sha(Path(manifest["source_agent_config_path"])) != inputs["source_agent"]["sha256"]):
        raise ValueError("manifest source configuration changed")
    if manifest["source_commit_sha"] != P.verify_committed(ROOT, CODE):
        raise ValueError("manifest source commit differs from the committed implementation")
    return manifest, validate_protocol(fixed_protocol(), Path(manifest["isaaclab_root"]))


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    source_env, source_agent = Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"])
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env, current_agent = load_config(log_dir / "params/env.yaml"), load_config(log_dir / "params/agent.yaml")
    if {k: v for k, v in original_env.items() if k != "seed"} != {k: v for k, v in current_env.items() if k != "seed"}:
        raise ValueError("environment configuration changed outside seed and registered code correction")
    allowed = {"run_name", "seed"}
    if {k: v for k, v in original_agent.items() if k not in allowed} != {k: v for k, v in current_agent.items() if k not in allowed}:
        raise ValueError("agent configuration changed outside restart arguments")
    if current_env.get("seed") != manifest["seed"] or current_agent.get("seed") != manifest["seed"]:
        raise ValueError("restart seed differs")
    if current_agent.get("run_name") != manifest["run_name"] or current_agent.get("num_steps_per_env") != 24:
        raise ValueError("runner configuration differs")
    if current_env.get("scene", {}).get("num_envs") != 4096:
        raise ValueError("environment count differs")


def _source_inputs(scope: dict) -> dict:
    bindings = scope["bindings"]
    return {
        "source_env": {"path": bindings["source_env_config_path"], "sha256": bindings["source_env_config_sha256"]},
        "source_agent": {"path": bindings["source_agent_config_path"], "sha256": bindings["source_agent_config_sha256"]},
        "reference_report": {"path": bindings["none_fixed_report_path"], "sha256": bindings["none_fixed_report_sha256"]},
        "reference_checkpoint": {"path": bindings["none_checkpoint_path"], "sha256": bindings["none_checkpoint_sha256"]},
    }


def _validate_source_inputs(prereg: dict, isaaclab: Path) -> tuple[dict, dict]:
    scope, _ = validate_authorization(_folder(prereg["tag"]) / "scope.json",
                                      _folder(prereg["tag"]) / "approval.json", isaaclab)
    expected = _source_inputs(scope)
    if prereg.get("source_inputs") != expected:
        raise ValueError("preregistered source inputs differ from the validated scope")
    for entry in expected.values():
        path = Path(entry["path"])
        if path.is_symlink() or sha(path) != entry["sha256"]:
            raise ValueError("a source input changed after preregistration")
    reference = C.validate_fixed_report(json.loads(Path(expected["reference_report"]["path"]).read_text(encoding="utf-8")), {
        "name": f"baseline_p0c_s{FAULT_SEED}", "checkpoint_file": "model_299.pt",
        "checkpoint_sha256": expected["reference_checkpoint"]["sha256"], "action_scale": 0.25,
        "protocol_sha256_lf": sha(ROOT / "bench/protocols/fixed_eval_v1.json", lf=True),
    })
    if not __import__("e2e_reward_demo_v2").walks(reference):
        raise ValueError("source NONE reference does not satisfy the walking gate")
    return scope, reference


def write_manifests(tag: str, isaaclab: Path, scope_path: Path, approval_path: Path) -> dict:
    protocol = validate_protocol(fixed_protocol(), isaaclab)
    scope, approval = validate_authorization(scope_path, approval_path, isaaclab)
    inputs = _source_inputs(scope)
    source_env = Path(inputs["source_env"]["path"])
    source_agent = Path(inputs["source_agent"]["path"])
    source_commit = scope["bindings"]["source_commit_sha"]
    if source_commit != P.verify_committed(ROOT, CODE):
        raise ValueError("source commit is not the committed task5 implementation")
    folder = _folder(tag)
    if folder.exists():
        raise FileExistsError("task5 run tag already exists")
    reference_report = Path(inputs["reference_report"]["path"])
    reference = C.validate_fixed_report(json.loads(reference_report.read_text(encoding="utf-8")), {
        "name": f"baseline_p0c_s{FAULT_SEED}",
        "checkpoint_file": "model_299.pt",
        "checkpoint_sha256": inputs["reference_checkpoint"]["sha256"],
        "action_scale": 0.25,
        "protocol_sha256_lf": sha(ROOT / "bench/protocols/fixed_eval_v1.json", lf=True),
    })
    outputs = {}
    for arm in ARMS:
        path = folder / arm / "train/manifest.json"
        write_new(path, manifest_for(tag, arm, isaaclab, source_env, source_agent, source_commit))
        outputs[arm] = str(path)
    for source, target in ((scope_path, folder / "scope.json"), (approval_path, folder / "approval.json")):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
    write_new(folder / "consumed.json", {"approval_sha256": sha(folder / "approval.json"),
                                           "arms": list(ARMS), "train_limit_per_arm": 1, "eval_limit": 1})
    write_new(folder / "prereg.json", {"schema": "u1_scope_extension_run_v1", "tag": tag,
                                        "protocol_sha256_lf": sha(PROTOCOL, lf=True), "arms": outputs,
                                        "source_inputs": inputs,
                                        "reference_report": inputs["reference_report"],
                                        "reference_checkpoint": reference["checkpoint"],
                                        "outcome": "unmeasured", "source_commit_sha": source_commit})
    _validate_source_inputs(json.loads((folder / "prereg.json").read_text(encoding="utf-8")), isaaclab)
    return {"protocol": protocol["schema"], "manifests": outputs}


def _expected_environment() -> dict:
    return {"schema": "reward_family_environment_readback_v1",
            "clean": dict.fromkeys(F.FUNCTIONS, True),
            "bound_to_reward_terms": dict.fromkeys(F.FUNCTIONS, True), "ok": True}


def _training_attestation(folder: Path, arm: str, protocol: dict, versions: dict, *, create: bool) -> dict:
    training = folder / arm / "train"
    manifest_path = training / "manifest.json"
    manifest, _ = validate_manifest(manifest_path)
    execution_path = training / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    receipt = _strict_verify(folder, arm, execution, protocol, versions)
    correction_path = training / "correction.json"
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    validate_readback(correction, manifest["fault"], manifest["correction_id"])
    environment_path = training / "family_environment.json"
    if json.loads(environment_path.read_text(encoding="utf-8")) != _expected_environment():
        raise ValueError("environment reward-family binding differs")
    checkpoint = Path(receipt["checkpoint"]["path"])
    if checkpoint.is_symlink() or sha(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise ValueError("training checkpoint changed after verification")
    attestation = {
        "schema": "u1_scope_training_attestation_v1", "arm": arm,
        "manifest_sha256": sha(manifest_path), "execution_sha256": sha(execution_path),
        "receipt_sha256": sha(training / "receipt.json"), "correction_sha256": sha(correction_path),
        "family_environment_sha256": sha(environment_path), "checkpoint": receipt["checkpoint"],
        "scope_sha256": sha(folder / "scope.json"), "approval_sha256": sha(folder / "approval.json"),
    }
    path = training / "training_attestation.json"
    if create:
        write_new(path, attestation)
    elif not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != attestation:
        raise ValueError("training attestation differs from current artifacts")
    return attestation


def write_eval_jobs(tag: str) -> dict:
    folder = _folder(tag)
    prereg = json.loads((folder / "prereg.json").read_text(encoding="utf-8"))
    reference_path = Path(prereg["reference_report"]["path"])
    if sha(reference_path) != prereg["reference_report"]["sha256"]:
        raise ValueError("reference report changed before evaluation")
    versions = json.loads(reference_path.read_text(encoding="utf-8"))["versions"]
    protocol = fixed_protocol()
    jobs = []
    for arm in ARMS:
        attestation = _training_attestation(folder, arm, protocol, versions, create=False)
        checkpoint = attestation["checkpoint"]
        jobs.append({"name": arm, "checkpoint": checkpoint["path"], "output": str(folder / "eval" / f"{arm}.json")})
    path = folder / "eval/jobs.json"
    write_new(path, {"action_scale": 0.25, "jobs": jobs, "protocol_sha256_lf": prereg["protocol_sha256_lf"]})
    return {"jobs": str(path), "command": (f"python evals/fixed_eval.py --protocol bench/protocols/fixed_eval_v1.json "
                                             f"--jobs {path} --headless")}


def _strict_verify(folder: Path, arm: str, execution: dict, protocol: dict, versions: dict) -> dict:
    """기존 U1 v4 검증기를 새 manifest/readback에 연결한다."""
    original = (STRICT.validate_manifest, STRICT.validate_readback, STRICT.validate_training_configuration,
                STRICT.UPSTREAM_TARGETS)
    STRICT.validate_manifest = validate_manifest
    STRICT.validate_readback = validate_readback
    STRICT.validate_training_configuration = validate_training_configuration
    STRICT.UPSTREAM_TARGETS = {}
    try:
        bridge = deepcopy(protocol)
        bridge["correction_source_contract"]["targets"] = {
            name: {"clean_fingerprint": fingerprint}
            for name, fingerprint in protocol["correction_source_contract"]["functions"].items()
        }
        return STRICT.verify_training(folder, arm, execution, bridge, versions)
    finally:
        (STRICT.validate_manifest, STRICT.validate_readback, STRICT.validate_training_configuration,
         STRICT.UPSTREAM_TARGETS) = original


def run_training(tag: str, isaaclab: Path) -> dict:
    folder = _folder(tag)
    prereg = json.loads((folder / "prereg.json").read_text(encoding="utf-8"))
    protocol = validate_protocol(fixed_protocol(), isaaclab)
    _validate_source_inputs(prereg, isaaclab)
    reference_path = Path(prereg["reference_report"]["path"])
    if sha(reference_path) != prereg["reference_report"]["sha256"]:
        raise ValueError("reference report changed before training")
    versions = json.loads(reference_path.read_text(encoding="utf-8"))["versions"]
    results = {}
    for arm in ARMS:
        training = folder / arm / "train"
        execution_path = training / "execution.json"
        if execution_path.exists():
            raise FileExistsError("training arm execution was already consumed")
        manifest = training / "manifest.json"
        argv = [str((isaaclab / "_isaac_sim/python.bat").resolve()),
                str((ROOT / "bench/demo_correction_u1_family.py").resolve()), "--manifest", str(manifest.resolve())]
        execution = P.run_owned(training, argv, isaaclab, protocol["watchdogs_s"]["train"])
        write_new(execution_path, execution)
        attestation = _training_attestation(folder, arm, protocol, versions, create=True)
        results[arm] = {"execution": execution, "checkpoint": attestation["checkpoint"],
                        "attestation_sha256": sha(training / "training_attestation.json")}
    return results


def judge(tag: str) -> dict:
    folder = _folder(tag)
    prereg = json.loads((folder / "prereg.json").read_text(encoding="utf-8"))
    execution_path = folder / "eval/execution.json"
    if not execution_path.is_file() or not V3.execution_valid(json.loads(execution_path.read_text(encoding="utf-8"))):
        raise ValueError("fixed evaluation execution receipt is absent or invalid")
    first_manifest = json.loads((folder / "none_family_control/train/manifest.json").read_text(encoding="utf-8"))
    isaaclab = Path(first_manifest["isaaclab_root"])
    protocol = validate_protocol(fixed_protocol(), isaaclab)
    _validate_source_inputs(prereg, isaaclab)
    reference_path = Path(prereg["reference_report"]["path"])
    if sha(reference_path) != prereg["reference_report"]["sha256"]:
        raise ValueError("reference report changed after preregistration")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    evaluation = folder / "eval"
    expected_argv = [str((isaaclab / "_isaac_sim/python.bat").resolve()), str((ROOT / "evals/fixed_eval.py").resolve()),
                     "--protocol", str((ROOT / "bench/protocols/fixed_eval_v1.json").resolve()),
                     "--jobs", str((evaluation / "jobs.json").resolve()), "--headless"]
    launch = json.loads((evaluation / "launch.json").read_text(encoding="utf-8"))
    if launch != {"args": expected_argv, "cwd": str(isaaclab)}:
        raise ValueError("fixed evaluation launch differs")
    gpu_before = json.loads((evaluation / "gpu_before.json").read_text(encoding="utf-8"))
    gpu_after = json.loads((evaluation / "gpu_after.json").read_text(encoding="utf-8"))
    for snapshot in (gpu_before, gpu_after):
        if (snapshot.get("admission_policy") != GPU_PROFILE or type(snapshot.get("busy")) is not bool
                or not isinstance(snapshot.get("processes"), list)):
            raise ValueError("fixed evaluation GPU receipt differs")
    if gpu_before["busy"] is not False:
        raise ValueError("GPU was busy before fixed evaluation")
    owned = set(json.loads(execution_path.read_text(encoding="utf-8"))["owned_process_identities"])
    if any(str(row.get("pid")) in owned for row in gpu_after["processes"]):
        raise ValueError("owned fixed-evaluation process remained after execution")
    bands = json.loads((ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json").read_text(encoding="utf-8"))["bands"]
    labels, reports = {}, {}
    expected_jobs = []
    for arm in ARMS:
        attestation = _training_attestation(folder, arm, protocol, reference["versions"], create=False)
        expected_jobs.append({"name": arm, "checkpoint": attestation["checkpoint"]["path"],
                              "output": str(folder / "eval" / f"{arm}.json")})
        report_path = folder / "eval" / f"{arm}.json"
        reports[arm] = C.validate_fixed_report(json.loads(report_path.read_text(encoding="utf-8")), {
            "name": arm, "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": attestation["checkpoint"]["sha256"],
            "action_scale": 0.25, "protocol_sha256_lf": sha(ROOT / "bench/protocols/fixed_eval_v1.json", lf=True)})
        if reports[arm]["versions"] != reference["versions"]:
            raise ValueError("fixed evaluation runtime differs from the preregistered reference")
        verdict = RC.verdict(RC.compare(C._behavior(reports[arm]), C._behavior(reference)), bands)
        labels[arm] = {"label": verdict["label"], "verdict": verdict}
    jobs = json.loads((folder / "eval/jobs.json").read_text(encoding="utf-8"))
    if jobs != {"action_scale": 0.25, "jobs": expected_jobs,
                "protocol_sha256_lf": prereg["protocol_sha256_lf"]}:
        raise ValueError("fixed evaluation jobs differ from verified training attestations")
    none_walks = __import__("e2e_reward_demo_v2").walks(reports["none_family_control"])
    none_healthy = labels["none_family_control"]["label"] == "healthy" and none_walks
    u1_label = {"healthy": "recovered", "unhealthy": "not_recovered"}.get(labels["u1_family_restart"]["label"], "unknown")
    outcome = u1_label if none_healthy else "invalid"
    wording_key = "none_failed" if not none_healthy else u1_label
    if wording_key not in fixed_protocol()["wording"]:
        wording_key = "invalid"
    result = {"schema": "u1_scope_extension_result_v1", "tag": tag, "outcome": outcome,
              "none_control_healthy": none_healthy, "none_control_walks": none_walks, "arms": labels,
              "wording": fixed_protocol()["wording"][wording_key], "generalization_claim": False,
              "protocol_sha256_lf": sha(PROTOCOL, lf=True), "source_commit_sha": prereg["source_commit_sha"]}
    write_new(folder / "result.json", result)
    return result


def run_evaluation(tag: str, isaaclab: Path) -> dict:
    folder = _folder(tag)
    evaluation = folder / "eval"
    if (evaluation / "execution.json").exists() or (folder / "result.json").exists():
        raise FileExistsError("fixed evaluation was already consumed")
    protocol = fixed_protocol()
    gpu_before = MEDIA.query_gpu(protocol["gpu_admission_profile"])
    if gpu_before.get("busy") is not False:
        raise ValueError("GPU admission is busy before fixed evaluation")
    write_eval_jobs(tag)
    write_new(evaluation / "gpu_before.json", gpu_before)
    argv = [str((isaaclab / "_isaac_sim/python.bat").resolve()), str((ROOT / "evals/fixed_eval.py").resolve()),
            "--protocol", str((ROOT / "bench/protocols/fixed_eval_v1.json").resolve()),
            "--jobs", str((evaluation / "jobs.json").resolve()), "--headless"]
    execution = P.run_owned(evaluation, argv, isaaclab, protocol["watchdogs_s"]["fixed_eval"])
    write_new(evaluation / "execution.json", execution)
    if not V3.execution_valid(execution):
        raise ValueError("fixed evaluation did not complete and clean up")
    owned = set(execution["owned_process_identities"])
    gpu = MEDIA.query_gpu(protocol["gpu_admission_profile"])
    if any(str(row["pid"]) in owned for row in gpu["processes"]):
        raise ValueError("an owned fixed-evaluation process remains on GPU")
    write_new(evaluation / "gpu_after.json", gpu)
    return judge(tag)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--isaaclab", type=Path)
    make = sub.add_parser("write-manifests")
    make.add_argument("--tag", required=True)
    make.add_argument("--isaaclab", required=True, type=Path)
    make.add_argument("--scope-record", required=True, type=Path)
    make.add_argument("--approval-record", required=True, type=Path)
    train = sub.add_parser("run-training")
    train.add_argument("--tag", required=True)
    train.add_argument("--isaaclab", required=True, type=Path)
    jobs = sub.add_parser("run-evaluation")
    jobs.add_argument("--tag", required=True)
    jobs.add_argument("--isaaclab", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = validate_protocol(fixed_protocol(), args.isaaclab)
    elif args.command == "write-manifests":
        result = write_manifests(args.tag, args.isaaclab, args.scope_record, args.approval_record)
    elif args.command == "run-training":
        result = run_training(args.tag, args.isaaclab)
    elif args.command == "run-evaluation":
        result = run_evaluation(args.tag, args.isaaclab)
    else:
        result = judge(args.tag)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
