"""U1 결과 뒤 만든 네 항 확장 교정을 처음 쓰는 seed에서 확인한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import u1_scope_extension as BASE  # noqa: E402
import novel_u1_scope_source as SOURCE  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/novel_u1_scope_confirmation_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task11/u1"
FAULT_SEED = 2062
RESTART_SEED = 3062
ARMS = {
    "none_family_control": {"fault": "NONE", "seed": FAULT_SEED},
    "u1_family_restart": {"fault": "U1", "seed": RESTART_SEED},
}
CODE = set(BASE.CODE) | {
    "evals/novel_u1_scope_source.py",
    "evals/novel_u1_scope_confirmation.py",
    "bench/novel_u1_family_training.py",
    "bench/protocols/novel_u1_scope_confirmation_v1.json",
    "bench/protocols/novel_u1_scope_source/e2e_demo_u1_v1.json",
    "evals/preregistered_seed_guard.py",
}
_BASE_VALIDATE_PROTOCOL = BASE.validate_protocol
_VERIFY_COMMITTED = BASE.P.verify_committed


def fixed_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict, isaaclab: Path | None = None) -> dict:
    configure()
    if protocol.get("schema") != "novel_u1_scope_confirmation_protocol_v1":
        raise ValueError("protocol schema differs")
    bridge = deepcopy(protocol)
    bridge["schema"] = "u1_scope_extension_protocol_v1"
    _BASE_VALIDATE_PROTOCOL(bridge, isaaclab)
    novelty = protocol.get("novel_seed_contract", {})
    if (novelty.get("source_seed") != FAULT_SEED or novelty.get("restart_seed") != RESTART_SEED
            or novelty.get("freshness_check") != "required_once_before_first_gpu_step"
            or novelty.get("previous_result_observed") is not True):
        raise ValueError("novel-seed contract differs")
    return protocol


def validate_authorization(scope_path: Path, approval_path: Path, isaaclab: Path) -> tuple[dict, dict]:
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    result = scope.get("result", {})
    correction = BASE.choose_correction(scope.get("fault"), result.get("verdict"),
                                        result.get("uncovered_terms"), scope.get("fault_family"))
    if scope.get("schema") != "novel_u1_scope_confirmation_scope_v1" or scope.get("seed") != FAULT_SEED:
        raise ValueError("scope record is not the preregistered novel-seed U1 check")
    SOURCE.validate_scope_record(scope_path, isaaclab)
    if (scope.get("observed_probe_terms") != ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"]
            or scope.get("source_registered_terms") != list(BASE.F.FUNCTIONS)):
        raise ValueError("scope record does not separate measured and source-registered terms")
    expected = {"schema": "human_correction_approval_v1", "decision": "approve",
                "correction_id": correction, "scope_sha256": BASE.sha(scope_path), "identity_attested": False}
    if approval != expected:
        raise ValueError("approval differs from the scope-bound correction")
    return scope, approval


def verify_source_anchor(source_commit: str) -> str:
    """현재 HEAD가 앞서도 결합 파일 blob이 같을 때만 source commit을 유지한다."""
    head = _VERIFY_COMMITTED(ROOT, CODE)
    return V2.frozen_commit(source_commit, head, CODE)


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != BASE.R.MANIFEST_FIELDS or manifest.get("arm") not in ARMS:
        raise ValueError("manifest schema or arm differs")
    expected_path = BASE._folder(manifest["tag"]) / manifest["arm"] / "train/manifest.json"
    if path.is_symlink() or path.resolve() != expected_path.resolve():
        raise ValueError("manifest is outside its registered arm")
    expected = BASE.manifest_for(manifest["tag"], manifest["arm"], Path(manifest["isaaclab_root"]),
                                 Path(manifest["source_env_config_path"]),
                                 Path(manifest["source_agent_config_path"]), manifest["source_commit_sha"])
    if manifest != expected:
        raise ValueError("manifest differs from its fixed arm")
    folder = BASE._folder(manifest["tag"])
    validate_authorization(folder / "scope.json", folder / "approval.json", Path(manifest["isaaclab_root"]))
    consumed = json.loads((folder / "consumed.json").read_text(encoding="utf-8"))
    if consumed != {"approval_sha256": BASE.sha(folder / "approval.json"), "arms": list(ARMS),
                    "train_limit_per_arm": 1, "eval_limit": 1}:
        raise ValueError("approval consumption contract differs")
    scope, _ = BASE._validate_source_inputs(json.loads((folder / "prereg.json").read_text(encoding="utf-8")),
                                            Path(manifest["isaaclab_root"]))
    inputs = BASE._source_inputs(scope)
    if (manifest["source_env_config_path"] != inputs["source_env"]["path"]
            or manifest["source_agent_config_path"] != inputs["source_agent"]["path"]
            or manifest["source_commit_sha"] != scope["bindings"]["source_commit_sha"]):
        raise ValueError("manifest source inputs differ from the validated scope")
    if (BASE.sha(Path(manifest["source_env_config_path"])) != inputs["source_env"]["sha256"]
            or BASE.sha(Path(manifest["source_agent_config_path"])) != inputs["source_agent"]["sha256"]):
        raise ValueError("manifest source configuration changed")
    if verify_source_anchor(manifest["source_commit_sha"]) != manifest["source_commit_sha"]:
        raise ValueError("manifest source anchor differs")
    return manifest, validate_protocol(fixed_protocol(), Path(manifest["isaaclab_root"]))


def write_manifests(tag: str, isaaclab: Path, scope_path: Path, approval_path: Path) -> dict:
    protocol = validate_protocol(fixed_protocol(), isaaclab)
    scope, approval = validate_authorization(scope_path, approval_path, isaaclab)
    inputs = BASE._source_inputs(scope)
    source_env, source_agent = Path(inputs["source_env"]["path"]), Path(inputs["source_agent"]["path"])
    source_commit = scope["bindings"]["source_commit_sha"]
    verify_source_anchor(source_commit)
    folder = BASE._folder(tag)
    if folder.exists():
        raise FileExistsError("task11 U1 run tag already exists")
    reference_report = Path(inputs["reference_report"]["path"])
    reference = BASE.C.validate_fixed_report(json.loads(reference_report.read_text(encoding="utf-8")), {
        "name": f"baseline_p0c_s{FAULT_SEED}", "checkpoint_file": "model_299.pt",
        "checkpoint_sha256": inputs["reference_checkpoint"]["sha256"], "action_scale": 0.25,
        "protocol_sha256_lf": BASE.sha(ROOT / "bench/protocols/fixed_eval_v1.json", lf=True),
    })
    outputs = {}
    for arm in ARMS:
        path = folder / arm / "train/manifest.json"
        BASE.write_new(path, BASE.manifest_for(tag, arm, isaaclab, source_env, source_agent, source_commit))
        outputs[arm] = str(path)
    for source, target in ((scope_path, folder / "scope.json"), (approval_path, folder / "approval.json")):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
    BASE.write_new(folder / "consumed.json", {"approval_sha256": BASE.sha(folder / "approval.json"),
                                               "arms": list(ARMS), "train_limit_per_arm": 1, "eval_limit": 1})
    BASE.write_new(folder / "prereg.json", {
        "schema": "novel_u1_scope_confirmation_run_v1", "tag": tag,
        "protocol_sha256_lf": BASE.sha(PROTOCOL, lf=True), "arms": outputs, "source_inputs": inputs,
        "reference_report": inputs["reference_report"], "reference_checkpoint": reference["checkpoint"],
        "outcome": "unmeasured", "source_commit_sha": source_commit,
    })
    BASE._validate_source_inputs(json.loads((folder / "prereg.json").read_text(encoding="utf-8")), isaaclab)
    return {"protocol": protocol["schema"], "manifests": outputs}


def configure() -> None:
    SOURCE.configure()
    SOURCE.BASE.configure = SOURCE.configure
    SOURCE.BASE._scope_bindings = SOURCE.scope_bindings
    BASE.PROTOCOL = PROTOCOL
    BASE.PRIVATE = PRIVATE
    BASE.FAULT_SEED = FAULT_SEED
    BASE.RESTART_SEED = RESTART_SEED
    BASE.ARMS = ARMS
    BASE.CODE = CODE
    BASE.fixed_protocol = fixed_protocol
    BASE.validate_protocol = validate_protocol
    BASE.validate_authorization = validate_authorization
    BASE.validate_manifest = validate_manifest


def run_training(tag: str, isaaclab: Path) -> dict:
    configure()
    folder = BASE._folder(tag)
    prereg = json.loads((folder / "prereg.json").read_text(encoding="utf-8"))
    protocol = validate_protocol(fixed_protocol(), isaaclab)
    BASE._validate_source_inputs(prereg, isaaclab)
    reference_path = Path(prereg["reference_report"]["path"])
    if BASE.sha(reference_path) != prereg["reference_report"]["sha256"]:
        raise ValueError("reference report changed before training")
    versions = json.loads(reference_path.read_text(encoding="utf-8"))["versions"]
    results = {}
    for arm in ARMS:
        training = folder / arm / "train"
        execution_path = training / "execution.json"
        if execution_path.exists():
            raise FileExistsError("training arm execution was already consumed")
        argv = [str((isaaclab / "_isaac_sim/python.bat").resolve()),
                str((ROOT / "bench/novel_u1_family_training.py").resolve()),
                "--manifest", str((training / "manifest.json").resolve())]
        execution = BASE.P.run_owned(training, argv, isaaclab, protocol["watchdogs_s"]["train"])
        BASE.write_new(execution_path, execution)
        attestation = BASE._training_attestation(folder, arm, protocol, versions, create=True)
        results[arm] = {"execution": execution, "checkpoint": attestation["checkpoint"],
                        "attestation_sha256": BASE.sha(training / "training_attestation.json")}
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check"); check.add_argument("--isaaclab", type=Path)
    make = sub.add_parser("write-manifests"); make.add_argument("--tag", required=True); make.add_argument("--isaaclab", required=True, type=Path); make.add_argument("--scope-record", required=True, type=Path); make.add_argument("--approval-record", required=True, type=Path)
    train = sub.add_parser("run-training"); train.add_argument("--tag", required=True); train.add_argument("--isaaclab", required=True, type=Path)
    evaluate = sub.add_parser("run-evaluation"); evaluate.add_argument("--tag", required=True); evaluate.add_argument("--isaaclab", required=True, type=Path)
    args = parser.parse_args(argv)
    configure()
    if args.command == "check": result = validate_protocol(fixed_protocol(), args.isaaclab)
    elif args.command == "write-manifests": result = write_manifests(args.tag, args.isaaclab, args.scope_record, args.approval_record)
    elif args.command == "run-training": result = run_training(args.tag, args.isaaclab)
    else: result = BASE.run_evaluation(args.tag, args.isaaclab)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
