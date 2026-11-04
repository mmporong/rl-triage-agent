"""Resume-safe controller for preregistered diagnostic-to-recovery comparison arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

_ENTRY_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ENTRY_ROOT), *(str(_ENTRY_ROOT / name) for name in ("src", "bench", "evals"))]

from evals import registered_recovery_resources_v1 as recovery_resources
from evals import registered_recovery_timing_v1 as recovery_timing
from evals.diagnostic_terminal_status_v1 import terminal_aware_status
from rl_triage import investigate, registered_diagnostic_backend
from rl_triage.correction_registry_v2 import REGISTRY, registry_digest
from rl_triage.investigation_session import (
    InvestigationSession,
    file_sha,
    reference,
    write_new,
)

PLAN_SCHEMA = "registered_recovery_comparison_plan_v1"
PLAN_SCHEMA_V2 = "registered_recovery_comparison_plan_v2"
JOB_SCHEMA = "registered_recovery_comparison_job_v1"
RESULT_SCHEMA = "registered_recovery_comparison_result_v1"
METHODS = ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent")
SUPPORTED_CASES = ("case-05", "case-06", "case-07", "case-08")
BACKEND = investigate.REGISTERED_BACKEND
REVIEW_POLICY_RELATIVE = "bench/protocols/registered_recovery_review_host_policy_v1.json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_HIDDEN = {"scoring", "original_faults", "remaining_faults", "hidden_label", "hidden_labels",
           "answer_key", "correct_any_family"}
_TERMINAL_WITHOUT_RECOVERY = {
    "host_no_intervention": "normal_or_host_gate",
    "hold_observation_unqualified": "hold",
    "none_supported": "unsupported",
    "unidentifiable": "hold",
    "hold_unknown": "hold",
    "hold_budget_exhausted": "hold",
    "hold_no_registered_probe": "hold",
    "model_hold": "hold",
}


class ComparisonContractError(ValueError):
    """A preregistration, immutable artifact, or state transition is invalid."""


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _reject_hidden(value: object) -> None:
    if isinstance(value, dict):
        if _HIDDEN & {str(key).casefold() for key in value}:
            raise ComparisonContractError("hidden/scoring authority is forbidden")
        for child in value.values():
            _reject_hidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_hidden(child)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ComparisonContractError("JSON object required")
    canonical(value)
    _reject_hidden(value)
    return value


def _ref(value: object, label: str, *, require_now: bool = True) -> tuple[Path, dict | None]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ComparisonContractError(f"{label} reference fields differ")
    if not isinstance(value["sha256"], str) or _SHA.fullmatch(value["sha256"]) is None:
        raise ComparisonContractError(f"{label} reference SHA differs")
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink():
        raise ComparisonContractError(f"{label} reference path differs")
    if not require_now and not path.exists():
        return path.resolve(), None
    path = path.resolve(strict=True)
    if file_sha(path) != value["sha256"]:
        raise ComparisonContractError(f"{label} reference changed")
    return path, _read(path)


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ComparisonContractError(f"{label} must be POSIX relative")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ComparisonContractError(f"{label} escapes repo")
    return path


def _validate_plan(plan: dict) -> None:
    fields = {"schema", "source_anchor", "source_closure", "methods", "cases", "limits", "inputs",
              "backend", "authorization", "recovery_policy", "template_commitments"}
    schema = plan.get("schema")
    if schema == PLAN_SCHEMA_V2:
        fields.add("continuation")
    if set(plan) != fields or schema not in {PLAN_SCHEMA, PLAN_SCHEMA_V2}:
        raise ComparisonContractError("plan fields differ")
    if _COMMIT.fullmatch(str(plan["source_anchor"])) is None:
        raise ComparisonContractError("source anchor differs")
    closure = plan["source_closure"]
    if not isinstance(closure, dict) or not closure:
        raise ComparisonContractError("source closure missing")
    for path, sha in closure.items():
        _relative(path, "source closure")
        if not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
            raise ComparisonContractError("source closure SHA differs")
    if plan["methods"] != list(METHODS) or plan["backend"] != BACKEND:
        raise ComparisonContractError("method/backend differs")
    cases = plan["cases"]
    case_ids = {row.get("case_id") for row in cases if isinstance(row, dict)} if isinstance(cases, list) else set()
    if not case_ids or not case_ids <= set(SUPPORTED_CASES) or len(case_ids) != len(cases):
        raise ComparisonContractError("explicit supported case list required")
    for row in cases:
        if (set(row) != {"case_id", "rotation"} or not isinstance(row["rotation"], list)
                or len(row["rotation"]) != 4 or set(row["rotation"]) != set(METHODS)):
            raise ComparisonContractError("four-arm case rotation differs")
    limits = plan["limits"]
    expected_sdk_cap = len(case_ids) * 8 if schema == PLAN_SCHEMA else len(case_ids) * 8 - 1
    expected_total_cap = len(case_ids) * 16 if schema == PLAN_SCHEMA else len(case_ids) * 16 - 1
    if limits != {"max_recoveries": len(case_ids) * 4,
                  "gpu_sdk_launch_cap": expected_sdk_cap,
                  "cpu_checkpoint_launch_cap": len(case_ids) * 4,
                  "cpu_telemetry_launch_cap": len(case_ids) * 4,
                  "total_owned_launch_cap": expected_total_cap,
                  "max_parent_wall_s_per_recovery": 5000, "api_cap": 0}:
        raise ComparisonContractError("recovery/GPU/API cap differs")
    if schema == PLAN_SCHEMA_V2:
        continuation = plan["continuation"]
        required = {
            "gpu_admission_profile", "prior_training", "prior_consumed_gpu_sdk_launches",
            "prior_training_gpu_s", "expected_new_gpu_sdk_launches", "continuation_tag",
            "prior_failed_attempt_preserved",
        }
        if (
            not isinstance(continuation, dict)
            or set(continuation) != required
            or continuation["gpu_admission_profile"] != "registered_desktop_shared_v2"
            or continuation["prior_consumed_gpu_sdk_launches"] != 1
            or continuation["prior_training_gpu_s"] != 515.3279999999795
            or continuation["expected_new_gpu_sdk_launches"] != 29
            or continuation["continuation_tag"] != "v5_fresh"
            or continuation["prior_failed_attempt_preserved"] is not True
            or case_ids != set(SUPPORTED_CASES)
        ):
            raise ComparisonContractError("continuation accounting/profile differs")
        prior = continuation["prior_training"]
        if (
            not isinstance(prior, dict)
            or set(prior) != {
                "logical_id", "proof_sha256", "source_checkpoint_sha256",
                "completed_checkpoint_sha256", "seed", "task", "correction_id",
                "diagnostic_manifest_sha256",
            }
            or prior["logical_id"] != "case-05/fixed_sequence/case_02/CORRECTED"
            or any(
                not isinstance(prior[key], str) or _SHA.fullmatch(prior[key]) is None
                for key in (
                    "proof_sha256", "source_checkpoint_sha256", "completed_checkpoint_sha256",
                    "diagnostic_manifest_sha256",
                )
            )
            or type(prior["seed"]) is not int
            or prior["correction_id"] != "restore_motor_torque_speed"
            or not isinstance(prior["task"], str)
            or not prior["task"]
        ):
            raise ComparisonContractError("prior training logical identity differs")
    policy = plan["recovery_policy"]
    if (not isinstance(policy, dict)
            or set(policy) != {"training_iterations", "seeds", "same_case_same_template", "source_cost_policy"}
            or policy["training_iterations"] != 300
            or set(policy["seeds"]) != case_ids
            or any(not isinstance(seeds, dict) or set(seeds) != {"resume", "restart"}
                   or any(type(seed) is not int or seed < 0 for seed in seeds.values())
                   for seeds in policy["seeds"].values())
            or policy["same_case_same_template"] is not True
            or policy["source_cost_policy"] != "phase_a_shared_count_once"):
        raise ComparisonContractError("recovery seed/template/source-cost policy differs")
    commitments = plan["template_commitments"]
    mechanisms = {"actuator", "optimizer_update_count", "optimizer_surrogate"}
    if not isinstance(commitments, dict) or set(commitments) != case_ids:
        raise ComparisonContractError("template commitment cases differ")
    for case_rows in commitments.values():
        if not isinstance(case_rows, dict) or set(case_rows) != mechanisms:
            raise ComparisonContractError("template commitment mechanisms differ")
        for row in case_rows.values():
            if (not isinstance(row, dict)
                    or set(row) != {"training_binding_sha256", "fixed_binding_sha256",
                                    "source_checkpoint_sha256"}
                    or any(not isinstance(value, str) or _SHA.fullmatch(value) is None
                           for value in row.values())):
                raise ComparisonContractError("template commitment SHA differs")
    authorization = plan["authorization"]
    if (not isinstance(authorization, dict)
            or set(authorization) != {"actor", "ambiguous_consumed_policy"}
            or not isinstance(authorization["actor"], str) or not authorization["actor"]
            or authorization["ambiguous_consumed_policy"] != "settle_unknown_without_reexecution"):
        raise ComparisonContractError("authorization policy differs")
    inputs = plan["inputs"]
    roles = set()
    if not isinstance(inputs, list):
        raise ComparisonContractError("plan inputs missing")
    for row in inputs:
        if (not isinstance(row, dict) or set(row) != {"role", "path", "sha256"}
                or row["role"] in roles or not isinstance(row["role"], str)
                or _SHA.fullmatch(str(row["sha256"])) is None):
            raise ComparisonContractError("plan input differs")
        _relative(row["path"], "plan input")
        roles.add(row["role"])
    if not {"diagnostic_plan", "recovery_review_policy",
            *(f"diagnostic_protocol:{case}" for case in case_ids)} <= roles:
        raise ComparisonContractError("diagnostic preregistration inputs missing")


def load_plan(plan_path: Path, *, repo_root: Path, registration_anchor: str,
              require_committed: bool = True) -> dict:
    repo = Path(repo_root).resolve(strict=True)
    path = Path(plan_path).resolve(strict=True)
    plan = _read(path)
    _validate_plan(plan)
    for row in plan["inputs"]:
        candidate = (repo / _relative(row["path"], "plan input")).resolve(strict=True)
        if _lf_sha(candidate) != row["sha256"]:
            raise ComparisonContractError(f"plan input changed: {row['role']}")
    for relative, sha in plan["source_closure"].items():
        candidate = (repo / _relative(relative, "source closure")).resolve(strict=True)
        if _lf_sha(candidate) != sha:
            raise ComparisonContractError("source closure changed")
    if require_committed:
        if _COMMIT.fullmatch(registration_anchor) is None:
            raise ComparisonContractError("registration anchor differs")
        for ancestor, descendant in ((plan["source_anchor"], registration_anchor),
                                     (registration_anchor, "HEAD")):
            result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                                     ancestor, descendant], check=False, capture_output=True)
            if result.returncode != 0:
                raise ComparisonContractError("source/registration ancestry differs")
        registered = [path, *((repo / _relative(row["path"], "input")).resolve() for row in plan["inputs"])]
        for item in registered:
            relative = item.relative_to(repo).as_posix()
            blob = subprocess.run(["git", "-C", str(repo), "show", f"{registration_anchor}:{relative}"],
                                  check=True, capture_output=True).stdout
            if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != _lf_sha(item):
                raise ComparisonContractError("registration blob differs")
        for relative, sha in plan["source_closure"].items():
            blob = subprocess.run(["git", "-C", str(repo), "show", f"{plan['source_anchor']}:{relative}"],
                                  check=True, capture_output=True).stdout
            if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != sha:
                raise ComparisonContractError("source closure blob differs")
    return plan


def _validate_job(job: dict, plan_ref: dict, plan: dict, job_path: Path) -> list[dict]:
    if set(job) != {"schema", "plan", "arms"} or job["schema"] != JOB_SCHEMA or job["plan"] != plan_ref:
        raise ComparisonContractError("job fields/plan binding differ")
    rows = job["arms"]
    case_ids = tuple(row["case_id"] for row in plan["cases"])
    if not isinstance(rows, list) or len(rows) != len(case_ids) * 4:
        raise ComparisonContractError("complete recovery arm matrix required")
    expected = {(case, method) for case in case_ids for method in METHODS}
    seen = set()
    normalized = []
    seeds = plan["recovery_policy"]["seeds"]
    for row in rows:
        fields = {"case_id", "method", "diagnostic_manifest", "recovery_folder",
                  "recovery_payload", "approval_ref"}
        if not isinstance(row, dict) or set(row) != fields:
            raise ComparisonContractError("job arm fields differ")
        key = (row["case_id"], row["method"])
        if key not in expected or key in seen:
            raise ComparisonContractError("job arm identity differs")
        seen.add(key)
        manifest_path, manifest = _ref(row["diagnostic_manifest"], "diagnostic manifest")
        if not isinstance(manifest, dict) or manifest.get("method") != row["method"]:
            raise ComparisonContractError("diagnostic method binding differs")
        verification = manifest.get("source_verification", {})
        _, source = _ref(manifest.get("source"), "diagnostic source")
        if (not isinstance(source, dict) or source.get("case_id") != row["case_id"]
                or verification.get("case_id") != row["case_id"]):
            raise ComparisonContractError("diagnostic source case binding differs")
        source_checkpoint = verification.get("checkpoint_sha256")
        if not isinstance(source_checkpoint, str) or _SHA.fullmatch(source_checkpoint) is None:
            raise ComparisonContractError("diagnostic source checkpoint binding missing")
        session = manifest_path.parent
        optional = (row["recovery_folder"], row["recovery_payload"], row["approval_ref"])
        if any(value is None for value in optional):
            if not all(value is None for value in optional):
                raise ComparisonContractError("recovery folder/payload/approval must be jointly null")
            normalized.append({**deepcopy(row), "session_path": str(session)})
            continue
        folder = Path(row["recovery_folder"])
        expected_folder = (job_path.parent / "prepared-recoveries" / row["case_id"]
                           / row["method"]).resolve()
        if not folder.is_absolute() or folder != expected_folder:
            raise ComparisonContractError("recovery folder layout differs")
        payload = row["recovery_payload"]
        if not isinstance(payload, dict) or set(payload) != {
                "protocol", "artifact_resolver", "isaaclab_root", "output_root"}:
            raise ComparisonContractError("recovery payload fields differ")
        _, protocol = _ref(payload["protocol"], "recovery protocol")
        diagnostic = protocol.get("diagnostic", {}) if isinstance(protocol, dict) else {}
        operation = protocol.get("operation", {}) if isinstance(protocol, dict) else {}
        mechanism = diagnostic.get("mechanism")
        entry = REGISTRY.get(mechanism) if isinstance(mechanism, str) else None
        mode = operation.get("recovery_mode")
        expected_seed = seeds[row["case_id"]].get(mode) if mode in {"resume", "restart"} else None
        commitment = (plan["template_commitments"].get(row["case_id"], {}).get(mechanism)
                      if isinstance(mechanism, str) else None)
        operation_v2_fields = {
            "correction_id", "recovery_mode", "runner_id", "training_seed",
            "training_iterations", "gpu_budget_s", "source_checkpoint_consumed",
            "training_binding_artifact_id", "fixed_binding_artifact_id",
            "gpu_admission_profile", "prior_training_artifact_id",
        }
        if (not isinstance(protocol, dict)
                or (plan["schema"] == PLAN_SCHEMA_V2
                    and protocol.get("schema") != "registered_recovery_protocol_v2")
                or (plan["schema"] == PLAN_SCHEMA_V2 and set(operation) != operation_v2_fields)
                or diagnostic.get("case_id") != row["case_id"]
                or entry is None or operation.get("correction_id") != entry["correction_id"]
                or mode not in entry["recovery_modes"]
                or operation.get("training_seed") != expected_seed
                or operation.get("training_iterations") != 300
                or protocol.get("actors") != [plan["authorization"]["actor"]]
                or protocol.get("source_anchor") != plan["source_anchor"]
                or protocol.get("source_closure") != plan["source_closure"]
                or protocol.get("registry_sha256") != registry_digest()
                or diagnostic.get("checkpoint_sha256") != source_checkpoint
                or not isinstance(commitment, dict)
                or commitment.get("source_checkpoint_sha256") != source_checkpoint):
            raise ComparisonContractError("per-arm recovery protocol differs from parent policy")
        prior_id = operation.get("prior_training_artifact_id")
        if plan["schema"] == PLAN_SCHEMA_V2:
            continuation = plan["continuation"]
            prior = continuation["prior_training"]
            expected_prior_id = (
                "prior_training"
                if row["case_id"] == "case-05"
                and row["method"] == "fixed_sequence"
                and mechanism == "actuator"
                else None
            )
            if (
                operation.get("gpu_admission_profile") != continuation["gpu_admission_profile"]
                or prior_id != expected_prior_id
                or (
                    expected_prior_id is not None
                    and (
                        diagnostic.get("session_manifest_sha256")
                        != prior["diagnostic_manifest_sha256"]
                        or source_checkpoint != prior["source_checkpoint_sha256"]
                        or operation.get("training_seed") != prior["seed"]
                        or operation.get("correction_id") != prior["correction_id"]
                    )
                )
            ):
                raise ComparisonContractError("prior training arm/profile binding differs")
        _, resolver = _ref(payload["artifact_resolver"], "artifact resolver")
        training_id = operation.get("training_binding_artifact_id")
        fixed_id = operation.get("fixed_binding_artifact_id")
        artifacts = resolver.get("artifacts") if isinstance(resolver, dict) else None
        if (not isinstance(artifacts, dict) or training_id not in artifacts or fixed_id not in artifacts
                or artifacts[training_id].get("sha256") != commitment["training_binding_sha256"]
                or artifacts[fixed_id].get("sha256") != commitment["fixed_binding_sha256"]):
            raise ComparisonContractError("artifact resolver differs from parent template commitment")
        _ref(artifacts[training_id], "training binding")
        _ref(artifacts[fixed_id], "fixed binding")
        if prior_id is not None:
            if set(artifacts) != {training_id, fixed_id, prior_id}:
                raise ComparisonContractError("prior training resolver artifacts differ")
            prior_path, _prior_proof = _ref(artifacts[prior_id], "prior training proof")
            if file_sha(prior_path) != plan["continuation"]["prior_training"]["proof_sha256"]:
                raise ComparisonContractError("prior training proof differs from plan")
        elif set(artifacts) != {training_id, fixed_id}:
            raise ComparisonContractError("unexpected resolver artifact")
        approval_path, _ = _ref(row["approval_ref"], "approval")
        normalized.append({**deepcopy(row), "session_path": str(session),
                           "approval_ref": {"path": str(approval_path), "sha256": row["approval_ref"]["sha256"]}})
    if seen != expected:
        raise ComparisonContractError("job arm matrix incomplete")
    return normalized


def prepare(plan_path: Path, job_path: Path, output: Path, *, repo_root: Path,
            registration_anchor: str, require_committed: bool = True) -> dict:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("comparison output must be fresh")
    plan = load_plan(plan_path, repo_root=repo_root, registration_anchor=registration_anchor,
                     require_committed=require_committed)
    resolved_job = Path(job_path).resolve(strict=True)
    job = _read(resolved_job)
    plan_ref = {"path": str(Path(plan_path).resolve()), "sha256": file_sha(Path(plan_path))}
    job_ref = {"path": str(resolved_job), "sha256": file_sha(resolved_job)}
    arms = _validate_job(job, plan_ref, plan, resolved_job)
    target.mkdir(parents=True)
    manifest = {"schema": RESULT_SCHEMA + "_manifest", "plan": plan_ref, "job": job_ref,
                "registration_anchor": registration_anchor, "arms": arms,
                "limits": deepcopy(plan["limits"]), "source_cost_policy": "phase_a_shared_count_once"}
    write_new(target / "manifest.json", manifest)
    write_new(target / "anchor.json", {"schema": RESULT_SCHEMA + "_anchor",
                                        "manifest_sha256": file_sha(target / "manifest.json")})
    return status(target)


def _load(folder: Path) -> dict:
    root = Path(folder).resolve(strict=True)
    manifest = _read(root / "manifest.json")
    anchor = _read(root / "anchor.json")
    if anchor != {"schema": RESULT_SCHEMA + "_anchor", "manifest_sha256": file_sha(root / "manifest.json")}:
        raise ComparisonContractError("comparison anchor changed")
    _ref(manifest["plan"], "plan")
    _ref(manifest["job"], "job")
    return manifest


def _arm_result_path(root: Path, arm: dict) -> Path:
    return root / "results" / arm["case_id"] / f"{arm['method']}.json"


def _record(root: Path, arm: dict, value: dict) -> None:
    value = {"schema": RESULT_SCHEMA + "_arm", "case_id": arm["case_id"],
             "method": arm["method"], "source_cost_accounting": "phase_a_shared_not_repeated", **value}
    write_new(_arm_result_path(root, arm), value)


def _request_id(state: dict) -> str | None:
    requests = state.get("requests")
    if not isinstance(requests, dict) or not requests:
        return None
    live = [row for row in requests.values() if row.get("state") in {"pending", "approved", "running"}]
    done = [row for row in requests.values() if row.get("state") == "done"]
    rows = live or done
    return rows[-1].get("request_id") if rows else None


def _verified_diagnostic_status(session_path: Path) -> dict:
    session = InvestigationSession(session_path)
    session._verify_operation(registered_diagnostic_backend.verify_source)
    if session.manifest.get("execution_class") != "actual":
        raise ComparisonContractError("phase-A session is not an actual registered execution")
    return terminal_aware_status(session)


def _ambiguous_costs(recovery_folder: Path, request_id: str) -> dict:
    manifest = _read(recovery_folder / "manifest.json")
    output = Path(manifest["recovery_context"]["output_root"])
    result = {"prepare_s": manifest.get("prepare_s"), "authority_s": None, "queue_s": None,
              "training_s": None, "fixed_s": None, "postcheck_s": None, "api_calls": 0,
              "status": "unknown_partial_cost", "refs": []}
    authority = recovery_folder / "authority_costs" / f"{request_id}.json"
    if authority.is_file():
        row = _read(authority)
        result["authority_s"] = row.get("authority_s")
        result["refs"].append({"path": str(authority.resolve()), "sha256": file_sha(authority)})
    for field, path in (("training_s", output / "training/execution.json"),
                        ("fixed_s", output / "fixed/execution.json")):
        if path.is_file():
            row = _read(path)
            wall = row.get("wall_s")
            if isinstance(wall, (int, float)) and not isinstance(wall, bool) \
                    and math.isfinite(float(wall)) and float(wall) >= 0:
                result[field] = float(wall)
            result["refs"].append({"path": str(path.resolve()), "sha256": file_sha(path)})
    return result


def _resource_cost(root: Path, arm: dict, manifest: dict) -> dict | None:
    path = root / "resources" / arm["case_id"] / f"{arm['method']}.json"
    if not path.exists():
        return None
    row = _read(path)
    fields = {"schema", "case_id", "method", "plan_sha256", "status", "phase_wall_s",
              "lease_await_s", "locked_wall_s", "owned_gpu_s", "owned_gpu_s_complete",
              "simulator_sdk_launches", "cpu_checkpoint_launches", "cpu_telemetry_launches", "total_owned_children",
              "api_calls", "sampler_ref", "admission_ref", "recovery_resource_ref",
              "execution_refs", "error_type"}
    if (set(row) != fields or row["schema"] != "registered_recovery_resource_receipt_v1"
            or row["case_id"] != arm["case_id"] or row["method"] != arm["method"]
            or row["plan_sha256"] != manifest["plan"]["sha256"]
            or row["status"] not in {"complete", "failed", "unknown"}
            or type(row["simulator_sdk_launches"]) is not int
            or not 0 <= row["simulator_sdk_launches"] <= 2
            or type(row["cpu_checkpoint_launches"]) is not int
            or not 0 <= row["cpu_checkpoint_launches"] <= 1
            or type(row["cpu_telemetry_launches"]) is not int
            or not 0 <= row["cpu_telemetry_launches"] <= 1
            or type(row["total_owned_children"]) is not int
            or not 0 <= row["total_owned_children"] <= 4
            or row["total_owned_children"] != (row["simulator_sdk_launches"]
                                                + row["cpu_checkpoint_launches"]
                                                + row["cpu_telemetry_launches"])
            or row["api_calls"] != 0):
        raise ComparisonContractError("resource receipt binding differs")
    for field in ("phase_wall_s", "lease_await_s", "locked_wall_s", "owned_gpu_s"):
        value = row[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ComparisonContractError("resource cost differs")
    if row["phase_wall_s"] is not None and row["phase_wall_s"] > manifest["limits"]["max_parent_wall_s_per_recovery"]:
        raise ComparisonContractError("recovery parent wall cap exceeded")
    if row["sampler_ref"] is not None:
        _ref(row["sampler_ref"], "sampler")
    _ref(row["admission_ref"], "lease admission")
    admission = _read(Path(row["admission_ref"]["path"]))
    measured = recovery_resources.verify(row["recovery_resource_ref"],
                                         request_id=admission["request_id"],
                                         effect_key=admission["effect_key"])
    projected = {"status": measured["status"], "phase_wall_s": measured["phase_wall_s"],
                 "lease_await_s": measured["lease_await_s"], "locked_wall_s": measured["locked_wall_s"],
                 "owned_gpu_s": measured["owned_gpu_s"],
                 "owned_gpu_s_complete": measured["owned_gpu_s_complete"],
                 "simulator_sdk_launches": measured["simulator_sdk_launches"],
                 "cpu_checkpoint_launches": measured["cpu_checkpoint_launches"],
                 "cpu_telemetry_launches": measured["cpu_telemetry_launches"],
                 "total_owned_children": measured["total_owned_children"],
                 "sampler_ref": measured["sampler_ref"], "admission_ref": measured["admission_ref"],
                 "execution_refs": measured["execution_refs"],
                 "error_type": measured["error_type"] or measured["sampler_error_type"]}
    if any(row[key] != value for key, value in projected.items()):
        raise ComparisonContractError("projected resource proof differs from recovery evidence")
    return {**row, "receipt_ref": {"path": str(path.resolve()), "sha256": file_sha(path)}}


def _seal_resource(root: Path, manifest: dict, arm: dict, request: dict) -> dict:
    destination = root / "resources" / arm["case_id"] / f"{arm['method']}.json"
    if destination.exists():
        cost = _resource_cost(root, arm, manifest)
        if cost is None:
            raise ComparisonContractError("sealed resource receipt missing")
        return cost
    source = Path(arm["recovery_folder"]) / "resources" / f"{request['request_id']}.json"
    if not source.is_file():
        raise ComparisonContractError("dispatch resource evidence missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    recovery_resources.project({"path": str(source.resolve()), "sha256": file_sha(source)}, destination,
                               case_id=arm["case_id"], method=arm["method"],
                               plan_sha256=manifest["plan"]["sha256"],
                               request_id=request["request_id"], effect_key=request["effect_key"])
    cost = _resource_cost(root, arm, manifest)
    if cost is None:
        raise ComparisonContractError("sealed resource receipt missing")
    return cost


def _halt(root: Path, arm: dict, reason: str, *, error_type: str | None = None,
          detail: str | None = None) -> None:
    path = root / "halt.json"
    value = {"schema": RESULT_SCHEMA + "_halt", "case_id": arm["case_id"], "method": arm["method"],
             "reason": reason, "error_type": error_type, "detail": detail, "automatic_retry": False}
    if path.exists():
        if _read(path) != value:
            raise ComparisonContractError("halt record changed")
    else:
        write_new(path, value)


def _advance_once(folder: Path) -> dict:
    """Advance at most one arm transition; consumed operations are never dispatched again."""
    root = Path(folder).resolve(strict=True)
    manifest = _load(root)
    actor = _read(Path(manifest["plan"]["path"]))["authorization"]["actor"]
    for arm in manifest["arms"]:
        result_path = _arm_result_path(root, arm)
        if result_path.exists():
            continue
        session = Path(arm["session_path"])
        diagnostic = _verified_diagnostic_status(session)
        diagnostic_status = diagnostic.get("status")
        recovery_prepared = arm["recovery_payload"] is not None
        if diagnostic_status != "confirmed":
            if recovery_prepared:
                raise ComparisonContractError(
                    "unconfirmed/terminal diagnosis must not carry recovery authority")
            outcome = _TERMINAL_WITHOUT_RECOVERY.get(diagnostic_status) if isinstance(diagnostic_status, str) else None
            if outcome is None:
                return {**status(root), "resume": {"required": True, "reason": "diagnostic_not_terminal",
                                                   "case_id": arm["case_id"], "method": arm["method"]}}
            _record(root, arm, {"outcome": outcome, "diagnostic_status": diagnostic_status,
                                "recovery_started": False, "costs": None})
            return status(root)
        mechanism = diagnostic.get("confirmed_mechanism")
        if mechanism not in REGISTRY or mechanism not in {"optimizer_update_count", "optimizer_surrogate", "actuator"}:
            if recovery_prepared:
                raise ComparisonContractError(
                    "unsupported diagnosis must not carry recovery authority")
            _record(root, arm, {"outcome": "unsupported", "diagnostic_status": diagnostic_status,
                                "confirmed_mechanism": mechanism, "recovery_started": False, "costs": None})
            return status(root)
        if not recovery_prepared:
            return {**status(root), "resume": {"required": True,
                                               "reason": "recovery_artifacts_not_prepared",
                                               "case_id": arm["case_id"],
                                               "method": arm["method"],
                                               "confirmed_mechanism": mechanism}}
        recovery_folder = Path(arm["recovery_folder"])
        if not recovery_folder.exists():
            result = investigate.run("recovery-prepare", session,
                                     {"recovery_payload": arm["recovery_payload"],
                                      "folder": str(recovery_folder)}, backend_name=BACKEND, fixture=False)
            if result.get("mechanism") != mechanism:
                raise ComparisonContractError("recovery prepare mechanism differs from verified diagnosis")
            return status(root)
        recovery_state = investigate.run("recovery-status", recovery_folder, {},
                                         backend_name=BACKEND, fixture=False)
        request_id = _request_id(recovery_state)
        if request_id is None:
            investigate.run("recovery-propose", recovery_folder, {}, backend_name=BACKEND, fixture=False)
            return status(root)
        request = recovery_state["requests"][request_id]
        if request["state"] == "pending":
            investigate.run("recovery-approve", recovery_folder,
                            {"request_id": request_id, "actor": actor, "approval_ref": arm["approval_ref"]},
                            backend_name=BACKEND, fixture=False)
            return status(root)
        if request["state"] == "approved":
            investigate.run("recovery-dispatch", recovery_folder, {"request_id": request_id},
                            backend_name=BACKEND, fixture=False)
            refreshed = investigate.run("recovery-status", recovery_folder, {},
                                        backend_name=BACKEND, fixture=False)
            refreshed_request = refreshed["requests"][request_id]
            resource_path = recovery_folder / "resources" / f"{request_id}.json"
            if resource_path.is_file():
                _seal_resource(root, manifest, arm, refreshed_request)
            elif refreshed_request["state"] != "running":
                raise ComparisonContractError("dispatch returned without resource evidence")
            return status(root)
        if request["state"] == "running":
            # The fixed plan chooses unknown closure, not a second launch, for ambiguous consumption.
            resource_path = recovery_folder / "resources" / f"{request_id}.json"
            if resource_path.is_file():
                _seal_resource(root, manifest, arm, request)
            investigate.run("recovery-settle", recovery_folder, {"request_id": request_id, "receipt": None},
                            backend_name=BACKEND, fixture=False)
            _record(root, arm, {"outcome": "unknown", "diagnostic_status": diagnostic_status,
                                "confirmed_mechanism": mechanism, "recovery_started": True,
                                "failure_reason": "ambiguous_consumed_side_effect",
                                "costs": _ambiguous_costs(recovery_folder, request_id)})
            _halt(root, arm, "ambiguous_consumed_side_effect")
            return status(root)
        ledger_receipt = request.get("receipt", {})
        resource = _seal_resource(root, manifest, arm, request)
        if resource["status"] != "complete":
            _halt(root, arm, "infrastructure_or_integrity_failure",
                  error_type=resource.get("error_type") or "ResourceMeasurementFailure",
                  detail="registered recovery resource measurement did not complete")
            return status(root)
        if ledger_receipt.get("verdict") == "unknown" and not ledger_receipt.get("result_sha"):
            _record(root, arm, {"outcome": "unknown", "diagnostic_status": diagnostic_status,
                                "confirmed_mechanism": mechanism, "recovery_started": True,
                                "failure_reason": ledger_receipt.get("reason"),
                                "costs": _ambiguous_costs(recovery_folder, request_id)})
            return status(root)
        post = investigate.run("recovery-post", recovery_folder, {"request_id": request_id},
                               backend_name=BACKEND, fixture=False)
        plan = _read(Path(manifest["plan"]["path"]))
        policy = next(row for row in plan["inputs"] if row["role"] == "recovery_review_policy")
        policy_path = Path(__file__).resolve().parents[1] / REVIEW_POLICY_RELATIVE
        if policy["path"] != REVIEW_POLICY_RELATIVE or _lf_sha(policy_path) != policy["sha256"]:
            raise ComparisonContractError("post review host policy changed")
        review = investigate.run("recovery-review", recovery_folder,
            {"request_id": request_id, "host_gate_sources": {
                "schema": "intervention_host_gate_source_v1", "receipt": reference(policy_path)}},
            backend_name=BACKEND, fixture=False)
        outcome = ledger_receipt.get("verdict", "unknown")
        _record(root, arm, {"outcome": outcome, "diagnostic_status": diagnostic_status,
                            "confirmed_mechanism": mechanism, "recovery_started": True,
                            "new_checkpoint": post.get("new_checkpoint"), "fixed_report": post.get("fixed_report"),
                            "behavior_outcome": post.get("behavior_outcome"), "costs": post.get("costs"),
                            "post_review": review})
        return status(root)
    return status(root)


def advance(folder: Path) -> dict:
    """Advance once; integrity/infrastructure failures halt the remaining batch without retry."""
    root = Path(folder).resolve(strict=True)
    current = status(root)
    if current.get("halted"):
        return current
    manifest = _load(root)
    arm = next((row for row in manifest["arms"] if not _arm_result_path(root, row).exists()), None)
    if arm is None:
        return current
    try:
        with recovery_timing.transition(root, case_id=arm["case_id"], method=arm["method"]):
            return _advance_once(root)
    except Exception as exc:  # noqa: BLE001 - preserve the first infrastructure/integrity failure and stop.
        _halt(root, arm, "infrastructure_or_integrity_failure",
              error_type=type(exc).__name__, detail=str(exc))
        return status(root)


def status(folder: Path) -> dict:
    root = Path(folder).resolve(strict=True)
    manifest = _load(root)
    rows = []
    for arm in manifest["arms"]:
        path = _arm_result_path(root, arm)
        row = (_read(path) if path.exists() else {"case_id": arm["case_id"],
                                                  "method": arm["method"], "outcome": "not_started"})
        row["resource_cost"] = _resource_cost(root, arm, manifest)
        rows.append(row)
    completed = sum(row["outcome"] != "not_started"
                    and (row.get("recovery_started") is not True or row["resource_cost"] is not None)
                    for row in rows)
    total = len(manifest["arms"])
    halt_path = root / "halt.json"
    halt = _read(halt_path) if halt_path.exists() else None
    return {"schema": RESULT_SCHEMA, "completed": completed == total, "completed_arms": completed,
            "total_arms": total, "arms": rows,
            "halted": halt is not None, "halt": halt,
            "resume": {"required": completed != total and halt is None,
                       "remaining_arms": total - completed},
            "limits": deepcopy(manifest["limits"]), "api_calls": 0,
            "source_cost_accounting": "phase_a_shared_count_once"}


def run_to_boundary(folder: Path, *, max_transitions: int = 128) -> dict:
    if type(max_transitions) is not int or not 1 <= max_transitions <= 256:
        raise ValueError("transition bound differs")
    for _ in range(max_transitions):
        current = status(folder)
        if current["completed"] or current["halted"]:
            return current
        after = advance(folder)
        if after.get("resume", {}).get("reason") == "diagnostic_not_terminal":
            return after
    return {**status(folder), "resume": {"required": True, "reason": "transition_limit"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--plan", required=True, type=Path)
    prepare_parser.add_argument("--job", required=True, type=Path)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    prepare_parser.add_argument("--registration-anchor", required=True)
    for command in ("advance", "run", "status"):
        child = sub.add_parser(command)
        child.add_argument("--prepared", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.plan, args.job, args.out, repo_root=args.repo_root,
                             registration_anchor=args.registration_anchor)
        elif args.command == "advance":
            result = advance(args.prepared)
        elif args.command == "run":
            result = run_to_boundary(args.prepared)
        else:
            result = status(args.prepared)
    except Exception as exc:  # noqa: BLE001 - controller boundary records fail-closed JSON.
        result = {"schema": RESULT_SCHEMA, "completed": False, "status": "rejected",
                  "error_type": type(exc).__name__, "reason": str(exc),
                  "resume": {"required": True, "reason": "controller_error"}}
    print(canonical(result).decode())
    return 0 if result.get("status") != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
