"""교정 후 DCMotor mapping만 보충 확인하는 등록 후보 계층.

이 모듈은 SDK를 시작하지 않는다. 기존 actuator envelope producer가 만든 결과를
기존 validator로 다시 검사하고, 기존 recovery observation의 actuator check만 갱신한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


def _find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "bench/actuator_envelope_runtime.py").is_file() and (parent / "src/rl_triage").is_dir():
            return parent
    raise RuntimeError("repository root is unavailable")


ROOT = _find_repo()
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import registered_mechanism_observer_v1 as OBSERVER
from bench import run_actuator_envelope_runtime as PRODUCER
from rl_triage import intervention_governor as governor
from rl_triage import investigation_recovery as recovery

PROFILE_ID = "registered_post_actuator_confirmation_v1"
REQUEST_SCHEMA = "registered_post_actuator_confirmation_request_v1"
PLAN_SCHEMA = "registered_post_actuator_confirmation_plan_v1"
RECEIPT_SCHEMA = "registered_post_actuator_confirmation_receipt_v1"
DECISION_SCHEMA = "registered_post_actuator_supplementary_decision_v1"
PROTOCOL_SCHEMA = "registered_post_actuator_confirmation_protocol_v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
CASES = frozenset({"case-05"})
FORBIDDEN_REF_FRAGMENT = "ri_codex2_tokenfactory_live_160"
RECEIPT_CORE_FIELDS = {
    "plan_id", "case_id", "checkpoint", "outcome", "measurement_present", "joint_count",
    "state_restored", "policy_free_runtime_config_readback", "actual_checkpoint_execution",
    "trained_policy_validated", "source_refs", "validator_reasons", "api_calls",
    "approval_granted", "effect_created",
}
SOURCE_FILES = (
    "bench/actuator_envelope_runtime.py",
    "bench/run_actuator_envelope_runtime.py",
    "bench/registered_mechanism_observer_v1.py",
    "evals/registered_recovery_confirmation_v1.py",
    "src/rl_triage/investigation_recovery.py",
    "src/rl_triage/intervention_governor.py",
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
            _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON input unavailable: {path.name}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _write_new(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))


def _ref(value: object, label: str, *, require_file: bool = True) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path, digest = value.get("path"), value.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(digest, str) or SHA.fullmatch(digest) is None:
        raise ValueError(f"{label} reference values differ")
    normalized = path.replace("\\", "/").casefold()
    if FORBIDDEN_REF_FRAGMENT in normalized:
        raise ValueError(f"{label} points into a forbidden private subtree")
    resolved = Path(path).resolve()
    if require_file and (not resolved.is_file() or resolved.is_symlink() or _sha(resolved) != digest):
        raise ValueError(f"{label} reference changed")
    return {"path": str(resolved), "sha256": digest}


def _binding(value: object, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} fields differ")
    result = dict(value)
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) or item == ""
           for item in result.values()):
        raise ValueError(f"{label} values differ")
    return result


def _protocol(value: object) -> dict[str, Any]:
    fields = {"schema", "status", "profile_id", "max_sdk_launches", "cases", "condition",
              "producer", "validators", "source_sha256", "limitations", "predictions"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("protocol fields differ")
    if (value.get("schema") != PROTOCOL_SCHEMA or value.get("profile_id") != PROFILE_ID
            or value.get("status") not in {"not_registered_not_executable", "registered"}
            or value.get("max_sdk_launches") != 1 or value.get("cases") != ["case-05"]
            or value.get("condition") != "NONE"):
        raise ValueError("protocol fixed contract differs")
    expected_producer = {"callable": "bench.actuator_envelope_runtime.run_sdk",
                         "semantics": "policy_free_runtime_config_readback",
                         "checkpoint_consumed": False, "runtime_updates": 0}
    expected_validators = ["bench.run_actuator_envelope_runtime.validate_manifest",
                           "bench.run_actuator_envelope_runtime.verify_body"]
    if value.get("producer") != expected_producer or value.get("validators") != expected_validators:
        raise ValueError("protocol producer contract differs")
    sources = value.get("source_sha256")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_FILES):
        raise ValueError("protocol source closure differs")
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file() or sources[relative] != _sha(path):
            raise ValueError(f"protocol source changed: {relative}")
    if not isinstance(value.get("limitations"), list) or not isinstance(value.get("predictions"), Mapping):
        raise TypeError("protocol limitations or predictions differ")
    return deepcopy(dict(value))


def _request(value: object) -> dict[str, Any]:
    fields = {"schema", "profile_id", "case_id", "source_case_id", "seed", "task", "condition",
              "checkpoint", "fixed_report", "training_receipt", "correction_readback",
              "recovery_receipt", "source_protocol", "runtime_closure", "lease", "effect",
              "request", "sdk_budget", "result_seen_design"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("request fields differ")
    if (value.get("schema") != REQUEST_SCHEMA or value.get("profile_id") != PROFILE_ID
            or value.get("case_id") not in CASES or not isinstance(value.get("source_case_id"), str)
            or type(value.get("seed")) is not int or value["seed"] < 0
            or not isinstance(value.get("task"), str) or not value["task"]
            or value.get("condition") != "NONE" or value.get("sdk_budget") != 1
            or value.get("result_seen_design") is not True):
        raise ValueError("request fixed values differ")
    refs: dict[str, Any] = {name: _ref(value[name], name) for name in (
        "checkpoint", "fixed_report", "training_receipt", "correction_readback",
        "recovery_receipt", "source_protocol")}
    closure = value.get("runtime_closure")
    if not isinstance(closure, Mapping) or not closure:
        raise ValueError("runtime closure is empty")
    refs["runtime_closure"] = {name: _ref(ref, f"runtime_closure.{name}")
                               for name, ref in sorted(closure.items())}
    lease = _binding(value["lease"], "lease", {"profile_id", "lease_id", "caller_pid"})
    effect = _binding(value["effect"], "effect", {"effect_key", "request_id"})
    request = _binding(value["request"], "request", {"approval_id", "request_id"})
    if lease["profile_id"] != "registered_desktop_shared_baseline_v1":
        raise ValueError("shared no-arg GPU profile binding differs")
    return {**{key: deepcopy(value[key]) for key in fields - set(refs) - {"lease", "effect", "request"}},
            **refs, "lease": lease, "effect": effect, "request": request}


def prepare(request: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """CPU에서 등록 입력과 기존 producer 진입점을 결합하며 SDK는 시작하지 않는다."""
    checked_protocol, checked_request = _protocol(protocol), _request(request)
    if checked_request["case_id"] not in checked_protocol["cases"]:
        raise ValueError("case is outside the registered one-case budget")
    entry = OBSERVER.production_entrypoints(ROOT)["actuator"]
    if (entry["checkpoint_mode"] != "unsupported"
            or OBSERVER.component_catalog()["actuator_mapping"]["checkpoint_consumed"] is not False):
        raise ValueError("producer checkpoint semantics changed")
    core = {"protocol": checked_protocol, "protocol_sha256": _digest(checked_protocol),
            "request": checked_request,
            "component_entrypoint": entry, "condition": "NONE", "sdk_launches": 1,
            "execution_authority": False, "approval_granted": False, "effect_created": False,
            "api_calls": 0, "status": checked_protocol["status"]}
    return {"schema": PLAN_SCHEMA, **core, "plan_id": _digest(core),
            "component_invocation": {
                "callable": "bench.actuator_envelope_runtime.run_sdk",
                "validators": list(checked_protocol["validators"]),
                "condition": "NONE", "checkpoint_consumed": False,
                "requires_existing_owned_windows_runner": True,
                "requires_use_gpu_lease": True,
                "requires_no_arg_shared_gpu_profile": "registered_desktop_shared_baseline_v1",
            },
            "interpretation": {
                "checkpoint_association": checked_request["checkpoint"],
                "actual_checkpoint_execution": False,
                "trained_policy_validated": False,
                "measurement_scope": "live_stock_go2_dcmotor_12joint_envelope_and_state_restoration",
            }}


def _validate_plan(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan schema differs")
    core_keys = {"protocol", "protocol_sha256", "request", "component_entrypoint", "condition", "sdk_launches",
                 "execution_authority", "approval_granted", "effect_created", "api_calls", "status"}
    core = {key: deepcopy(value.get(key)) for key in core_keys}
    if (value.get("plan_id") != _digest(core) or value.get("condition") != "NONE"
            or value.get("sdk_launches") != 1 or value.get("execution_authority") is not False
            or value.get("approval_granted") is not False or value.get("effect_created") is not False
            or value.get("api_calls") != 0):
        raise ValueError("plan binding differs")
    checked_protocol = _protocol(value.get("protocol"))
    if value.get("protocol_sha256") != _digest(checked_protocol):
        raise ValueError("plan protocol digest differs")
    expected_invocation = {
        "callable": "bench.actuator_envelope_runtime.run_sdk",
        "validators": ["bench.run_actuator_envelope_runtime.validate_manifest",
                       "bench.run_actuator_envelope_runtime.verify_body"],
        "condition": "NONE", "checkpoint_consumed": False,
        "requires_existing_owned_windows_runner": True,
        "requires_use_gpu_lease": True,
        "requires_no_arg_shared_gpu_profile": "registered_desktop_shared_baseline_v1",
    }
    expected_interpretation = {
        "checkpoint_association": value.get("request", {}).get("checkpoint")
        if isinstance(value.get("request"), Mapping) else None,
        "actual_checkpoint_execution": False,
        "trained_policy_validated": False,
        "measurement_scope": "live_stock_go2_dcmotor_12joint_envelope_and_state_restoration",
    }
    if (value.get("component_invocation") != expected_invocation
            or value.get("interpretation") != expected_interpretation):
        raise ValueError("plan component invocation or interpretation differs")
    _request(value.get("request"))
    return deepcopy(dict(value))


def verify_component_receipt(
    plan: Mapping[str, Any], manifest_path: Path, body_path: Path, execution_path: Path, *,
    repo_root: Path = ROOT, isaaclab_root: Path,
) -> dict[str, Any]:
    """기존 producer validator로 실제 DCMotor body와 owned execution을 재검증한다."""
    checked = _validate_plan(plan)
    manifest = Path(manifest_path).resolve(strict=True)
    body_ref = _ref({"path": str(Path(body_path).resolve(strict=True)),
                     "sha256": _sha(Path(body_path).resolve(strict=True))}, "body")
    execution_ref = _ref({"path": str(Path(execution_path).resolve(strict=True)),
                          "sha256": _sha(Path(execution_path).resolve(strict=True))}, "execution")
    body, execution = _read(Path(body_ref["path"])), _read(Path(execution_ref["path"]))
    validated = PRODUCER.validate_manifest(manifest, repo_root=Path(repo_root).resolve(),
                                           isaaclab_root=Path(isaaclab_root).resolve(),
                                           completed_output=True)
    current_entry = OBSERVER.production_entrypoints(Path(repo_root).resolve())["actuator"]
    if checked["component_entrypoint"] != current_entry:
        raise ValueError("component producer source binding changed")
    if validated["manifest"].get("condition") != "NONE" or Path(validated["paths"]["output_path"]).resolve() != Path(body_ref["path"]):
        raise ValueError("component manifest is not the registered NONE body")
    reasons = PRODUCER.verify_body(body, validated, execution)
    reasons.extend(PRODUCER._owned_execution_reasons(execution))
    if reasons:
        raise ValueError("component receipt validation failed: " + ",".join(sorted(set(reasons))))
    helper = body.get("helper_result")
    measurement_present = (body.get("measurement_status") == "valid" and isinstance(helper, Mapping)
                           and helper.get("status") == "valid")
    if not measurement_present:
        raise ValueError("valid actuator measurements are required before normal or abnormal")
    outcome = "normal" if body.get("curve_assessment") == "normal" else (
        "abnormal" if body.get("curve_assessment") == "curve_difference" else "unknown")
    if outcome == "unknown":
        raise ValueError("validated component did not produce a registered mapping outcome")
    restoration = body.get("state_restoration")
    if not isinstance(restoration, Mapping) or restoration.get("all_restored") is not True:
        raise ValueError("actuator state restoration is not confirmed")
    refs = {"manifest": {"path": str(manifest), "sha256": _sha(manifest)},
            "body": body_ref, "execution": execution_ref}
    core = {"plan_id": checked["plan_id"], "case_id": checked["request"]["case_id"],
            "checkpoint": deepcopy(checked["request"]["checkpoint"]), "outcome": outcome,
            "measurement_present": True, "joint_count": 12, "state_restored": True,
            "policy_free_runtime_config_readback": True, "actual_checkpoint_execution": False,
            "trained_policy_validated": False, "source_refs": refs,
            "validator_reasons": [], "api_calls": 0, "approval_granted": False,
            "effect_created": False}
    return {"schema": RECEIPT_SCHEMA, **core, "receipt_sha256": _digest(core)}


def unknown_receipt(plan: Mapping[str, Any], reason: str) -> dict[str, Any]:
    checked = _validate_plan(plan)
    if not isinstance(reason, str) or not reason:
        raise ValueError("unknown reason is required")
    core = {"plan_id": checked["plan_id"], "case_id": checked["request"]["case_id"],
            "checkpoint": deepcopy(checked["request"]["checkpoint"]), "outcome": "unknown",
            "measurement_present": False, "joint_count": None, "state_restored": None,
            "policy_free_runtime_config_readback": True, "actual_checkpoint_execution": False,
            "trained_policy_validated": False, "source_refs": {},
            "validator_reasons": [reason], "api_calls": 0, "approval_granted": False,
            "effect_created": False}
    return {"schema": RECEIPT_SCHEMA, **core, "receipt_sha256": _digest(core)}


def _component_receipt(value: object, plan: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != RECEIPT_CORE_FIELDS | {"schema", "receipt_sha256"}
            or value.get("schema") != RECEIPT_SCHEMA):
        raise ValueError("component receipt schema differs")
    core = {key: deepcopy(value[key]) for key in value if key not in {"schema", "receipt_sha256"}}
    if value.get("receipt_sha256") != _digest(core) or value.get("plan_id") != plan["plan_id"]:
        raise ValueError("component receipt binding differs")
    if value.get("checkpoint") != plan["request"]["checkpoint"] or value.get("case_id") != plan["request"]["case_id"]:
        raise ValueError("component checkpoint or case binding differs")
    outcome = value.get("outcome")
    if outcome not in {"normal", "abnormal", "unknown"}:
        raise ValueError("component outcome differs")
    if outcome in {"normal", "abnormal"} and (value.get("measurement_present") is not True
            or value.get("joint_count") != 12 or value.get("state_restored") is not True):
        raise ValueError("measurement-free mapping verdict is forbidden")
    if (value.get("policy_free_runtime_config_readback") is not True
            or value.get("actual_checkpoint_execution") is not False
            or value.get("trained_policy_validated") is not False):
        raise ValueError("component interpretation differs")
    refs = value.get("source_refs")
    if not isinstance(refs, Mapping):
        raise TypeError("component source refs differ")
    expected_ref_names = {"manifest", "body", "execution"} if outcome in {"normal", "abnormal"} else set()
    if set(refs) != expected_ref_names:
        raise ValueError("component source ref set differs")
    reasons = value.get("validator_reasons")
    if (outcome in {"normal", "abnormal"} and reasons != []) or (
            outcome == "unknown" and (not isinstance(reasons, list) or not reasons)):
        raise ValueError("component validator reasons differ")
    for name, ref in refs.items():
        _ref(ref, f"component.{name}")
    return deepcopy(dict(value))


def reverify_and_supplement(
    plan: Mapping[str, Any], source_observation: governor.VerifiedObservation,
    recovery_folder: Path, recovery_request_id: str,
    component_receipt: Mapping[str, Any], host_gate: governor.VerifiedHostGate,
    *, repo_root: Path = ROOT, isaaclab_root: Path,
) -> dict[str, Any]:
    """기존 actual recovery를 재검증하고 actuator check 하나만 보충한다."""
    checked_plan, checked_receipt = _validate_plan(plan), _component_receipt(component_receipt, plan)
    if checked_receipt["outcome"] in {"normal", "abnormal"}:
        refs = checked_receipt["source_refs"]
        reverified_receipt = verify_component_receipt(
            checked_plan,
            Path(refs["manifest"]["path"]),
            Path(refs["body"]["path"]),
            Path(refs["execution"]["path"]),
            repo_root=Path(repo_root),
            isaaclab_root=Path(isaaclab_root),
        )
        if reverified_receipt != checked_receipt:
            raise ValueError("component receipt differs from fresh producer validation")
    # build_model_packet은 opaque seal과 payload digest를 다시 검사한다.
    governor.build_model_packet(source_observation)
    observed = recovery.sealed_post_observation(
        Path(recovery_folder), recovery_request_id, source_observation,
    )
    payload = governor.build_model_packet(observed)["observation"]
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("checkpoint_sha256") != checked_plan["request"]["checkpoint"]["sha256"]:
        raise ValueError("reverified recovery checkpoint differs from the post-check plan")
    checks = payload.get("confirmation_checks")
    if not isinstance(checks, list):
        raise TypeError("reverified recovery confirmation checks are unavailable")
    actuator_rows = [row for row in checks if isinstance(row, Mapping) and row.get("check_id") == "actuator_mapping"]
    if len(actuator_rows) != 1 or actuator_rows[0].get("outcome") != "unknown":
        raise ValueError("source recovery must contain exactly one unknown actuator mapping check")
    refs = [f"sha256:{ref['sha256']}" for ref in checked_receipt["source_refs"].values()]
    if not refs:
        refs = [f"sha256:{checked_receipt['receipt_sha256']}"]
    replacement = {"check_id": "actuator_mapping", "outcome": checked_receipt["outcome"], "refs": refs}
    updated_checks = [replacement if row.get("check_id") == "actuator_mapping" else deepcopy(row)
                      for row in checks]
    updated = deepcopy(payload)
    updated.update({"receipt_id": f"{payload['receipt_id']}:post-actuator",
                    "verifier_id": "registered_post_actuator_confirmation_v1",
                    "verification_scope": "registered_recovery_plus_policy_free_actuator_mapping",
                    "confirmation_checks": updated_checks,
                    "post_confirmation_scope": "actuator_mapping_only",
                    "confirmation_catalog_complete": False,
                    "unobserved_confirmation_components": sorted(
                        set(payload.get("unobserved_confirmation_components", []))
                        - ({"actuator_mapping"} if checked_receipt["outcome"] != "unknown" else set())),
                    "supplementary_actuator_interpretation": {
                        "checkpoint_associated": True, "actual_checkpoint_execution": False,
                        "trained_policy_validated": False,
                        "policy_free_runtime_config_readback": True,
                    }})
    updated.setdefault("source_refs", {})["post_actuator_confirmation"] = {
        "sha256": checked_receipt["receipt_sha256"], "path": "registered-post-check-receipt"}
    sealed = governor._verified_observation(updated)
    governed = governor.decide_intervention(sealed, None, host_gate)
    conservative_hold = checked_receipt["outcome"] in {"abnormal", "unknown"}
    effective = "hold" if conservative_hold else governed["effective_action"]
    return {"schema": DECISION_SCHEMA, "status": "hold" if conservative_hold else governed["status"],
            "effective_action": effective, "mapping_outcome": checked_receipt["outcome"],
            "governor_decision": governed, "observation": governor.build_model_packet(sealed)["observation"],
            "approval_granted": False, "authority_created": False, "api_calls": 0,
            "sdk_launches_consumed": 1 if checked_receipt["measurement_present"] else 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--request", required=True, type=Path)
    prep.add_argument("--protocol", required=True, type=Path)
    prep.add_argument("--out", required=True, type=Path)
    verify = modes.add_parser("verify-component")
    verify.add_argument("--plan", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--body", required=True, type=Path)
    verify.add_argument("--execution", required=True, type=Path)
    verify.add_argument("--repo-root", type=Path, default=ROOT)
    verify.add_argument("--isaaclab", required=True, type=Path)
    verify.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(_read(args.request.resolve()), _read(args.protocol.resolve()))
        else:
            result = verify_component_receipt(
                _read(args.plan.resolve()), args.manifest, args.body, args.execution,
                repo_root=args.repo_root, isaaclab_root=args.isaaclab,
            )
        _write_new(args.out, result)
        print(json.dumps({"status": "prepared", "output": str(args.out.resolve())}, sort_keys=True))
        return 0
    except (ValueError, TypeError, OSError) as error:
        print(json.dumps({"status": "rejected", "reason": type(error).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
