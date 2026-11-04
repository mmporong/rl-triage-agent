"""U2 post-actuator NONE 확인을 전역 단회 cap과 canonical primary 상태로 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any


def _find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "bench/actuator_envelope_runtime.py").is_file():
            return parent
    raise RuntimeError("repository root is unavailable")


ROOT = _find_repo()
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import actuator_envelope_runtime as BODY
from bench import run_actuator_envelope_runtime as PRODUCER
from bench import run_runtime_readback_smoke as OWNED
from bench import runtime_readback_smoke as JSON_IO
from evals import lineage_source_batch_v1 as LEASE
from evals import registered_desktop_gpu_shared_v1 as GPU
from evals import registered_post_actuator_confirmation_v1 as POST
from evals import run_registered_recovery_comparison_v1 as PRIMARY
from rl_triage import intervention_governor as GOVERNOR
from rl_triage import investigation_recovery as RECOVERY
from rl_triage import registered_diagnostic_backend as DIAGNOSTIC
from rl_triage.investigation_session import InvestigationSession

SOURCE_SCHEMA = "post_actuator_operational_source_v4"
OUTER_SCHEMA = "post_actuator_operational_protocol_v5"
AUTHORITY_SCHEMA = "post_actuator_operational_authority_v5"
RESULT_SCHEMA = "post_actuator_operational_result_v5"
RUNNER_REL = "evals/run_registered_post_actuator_confirmation_v5.py"
CHILD_REL = "bench/post_actuator_child_v4.py"
TEST_REL = "tests/test_run_registered_post_actuator_confirmation_v5.py"
PY310_TEST_REL = "tests/post_actuator_child_py310_preflight_v4.py"
INNER_REL = "bench/protocols/registered_post_actuator_confirmation_v4.json"
OUTER_REL = "bench/protocols/registered_post_actuator_operational_v5.json"
PRIMARY_PLAN_SCHEMA = PRIMARY.PLAN_SCHEMA_V7
SOURCE_RELATIONS = (
    RUNNER_REL, CHILD_REL, TEST_REL, PY310_TEST_REL, INNER_REL,
    "evals/registered_post_actuator_confirmation_v1.py",
    "bench/actuator_envelope_runtime.py",
    "bench/run_actuator_envelope_runtime.py",
    "bench/run_runtime_readback_smoke.py",
    "bench/runtime_readback_smoke.py",
    "bench/registered_mechanism_observer_v1.py",
    "evals/lineage_source_batch_v1.py",
    "evals/registered_desktop_gpu_shared_v1.py",
    "evals/run_registered_recovery_comparison_v1.py",
    "evals/registered_epoch_namespace_failure_v1.py",
    "tests/test_registered_epoch_namespace_failure_v1.py",
    "bench/protocols/diagnostic_recovery_comparison_20261105_v13.md",
    "evals/registered_recovery_confirmation_v1.py",
    "src/rl_triage/investigation_recovery.py",
    "src/rl_triage/intervention_governor.py",
)
OP_ROOT = ROOT / "bench/private/post_actuator_confirmation"
RUN_ROOT = OP_ROOT / "runs"
CONSUMED_ROOT = OP_ROOT / "consumed"
FORBIDDEN = "ri_codex2_tokenfactory_live_160"
SHA = re.compile(r"[0-9a-f]{64}\Z")
TAG = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
PRIMARY_SDK_CAP = 32
POST_SDK_CAP = 1
STUDY_SDK_CAP = 33
EXPECTED_PRIOR_SDK = 18
EXPECTED_PRIMARY_NEW_SDK = 14
EXPECTED_IMPORTED_ARMS = 8
EXPECTED_FRESH_ARMS = 7
EXPECTED_HOLD_ARMS = 1
V11_FAILURE_PROOF_SHA256 = "91aaef7cc486b66d2ce01a1ca13fa27767466033a229eba1692ae5ebe44c5073"
V12_NAMESPACE_CONTEXT_SHA256 = "83d46f5e1d2a640cf24d46b127db946b77ee8fe7e65960e2f082d1240fe480f2"
MATRIX = tuple((f"case-{case:02d}", method) for case in range(5, 9)
               for method in ("fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent"))


class OperationalError(RuntimeError):
    """Fail-closed operational contract error."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _lexical(raw: object, label: str, roots: tuple[Path, ...]) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError(f"{label} path differs")
    text = os.fspath(raw)
    normalized = text.replace("\\", "/")
    pure = PurePath(text)
    if (not text or FORBIDDEN in normalized.casefold() or not pure.is_absolute()
            or ".." in pure.parts):
        raise ValueError(f"{label} path is forbidden")
    candidate = Path(os.path.abspath(text))
    folded = str(candidate).casefold()
    if not any(folded == str(root).casefold()
               or folded.startswith(str(root).rstrip("\\/").casefold() + os.sep.casefold())
               for root in roots):
        raise ValueError(f"{label} path is outside the allowed root")
    return candidate


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))


def _ref(value: object, label: str, roots: tuple[Path, ...]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path = _lexical(value.get("path"), label, roots)
    digest = value.get("sha256")
    if not isinstance(digest, str) or SHA.fullmatch(digest) is None:
        raise ValueError(f"{label} SHA differs")
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or _sha(path) != digest:
        raise ValueError(f"{label} reference changed")
    return {"path": str(path), "sha256": digest}


def _precheck_cli(source: Path, outer: Path, authority: Path, isaaclab: Path,
                  out_dir: Path, tag: str) -> tuple[Path, Path, Path, Path, Path]:
    if TAG.fullmatch(tag) is None:
        raise ValueError("tag differs")
    source = _lexical(source, "source inputs", (ROOT,))
    outer = _lexical(outer, "outer protocol", (ROOT,))
    authority = _lexical(authority, "authority", (ROOT,))
    isaaclab = _lexical(isaaclab, "IsaacLab", (Path.home(),))
    out_dir = _lexical(out_dir, "output", (RUN_ROOT,))
    if out_dir.parent != RUN_ROOT or out_dir.name != tag:
        raise ValueError("output must be the fixed post run tag folder")
    return source, outer, authority, isaaclab, out_dir


def _source(value: object) -> dict[str, Any]:
    fields = {
        "schema", "case_id", "source_case_id", "seed", "task", "condition",
        "recovery_request_id", "recovery_folder", "checkpoint", "fixed_report",
        "training_receipt", "correction_readback", "recovery_receipt", "source_protocol",
        "recovery_manifest", "recovery_approval", "host_gate_policy", "runtime_closure",
        "result_seen_design", "interpretation",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("source input fields differ")
    if (value.get("schema") != SOURCE_SCHEMA or value.get("case_id") != "case-05"
            or value.get("source_case_id") != "source-10" or value.get("seed") != 2664
            or value.get("task") != "Isaac-Velocity-Flat-Unitree-Go2-v0"
            or value.get("condition") != "NONE" or value.get("recovery_request_id") != "fb51e2eb0752"
            or value.get("result_seen_design") is not True):
        raise ValueError("source input fixed values differ")
    expected_interpretation = {
        "checkpoint_association_only": True, "producer_loads_checkpoint": False,
        "trained_policy_validated": False,
        "measurement_scope": "stock_go2_dcmotor_runtime_envelope_and_state_restoration",
    }
    if value.get("interpretation") != expected_interpretation:
        raise ValueError("source interpretation differs")
    folder = _lexical(value.get("recovery_folder"), "recovery folder", (ROOT,)).resolve(strict=True)
    if folder.name != "fixed_sequence" or not (folder / "recovery.jsonl").is_file():
        raise ValueError("recovery folder differs")
    roots = (ROOT, Path.home() / "IsaacLab")
    refs: dict[str, Any] = {name: _ref(value[name], name, roots) for name in (
        "checkpoint", "fixed_report", "training_receipt", "correction_readback",
        "recovery_receipt", "source_protocol", "recovery_manifest", "recovery_approval",
        "host_gate_policy",
    )}
    closure = value.get("runtime_closure")
    if not isinstance(closure, Mapping) or set(closure) != set(POST.SOURCE_FILES):
        raise ValueError("runtime closure differs")
    refs["runtime_closure"] = {name: _ref(ref, f"runtime_closure.{name}", (ROOT,))
                               for name, ref in sorted(closure.items())}
    receipt = _read(Path(refs["recovery_receipt"]["path"]))
    training = receipt.get("training_evidence")
    if (receipt.get("schema") != "registered_recovery_receipt_v1"
            or receipt.get("request_id") != value["recovery_request_id"]
            or receipt.get("new_checkpoint") != refs["checkpoint"]
            or receipt.get("fixed_report") != refs["fixed_report"]
            or not isinstance(training, Mapping)
            or training.get("training_receipt") != refs["training_receipt"]
            or training.get("readback") != refs["correction_readback"]):
        raise ValueError("recovery receipt lineage differs")
    approval = _read(Path(refs["recovery_approval"]["path"]))
    if (approval.get("schema") != "registered_recovery_approval_v1"
            or approval.get("request_id") != value["recovery_request_id"]
            or approval.get("effect_key") != receipt.get("effect_key")
            or approval.get("protocol_sha256") != refs["source_protocol"]["sha256"]):
        raise ValueError("recovery approval lineage differs")
    return {**deepcopy(dict(value)), **refs, "recovery_folder": str(folder)}


def _source_observation(source: Mapping[str, Any]):
    manifest = _read(Path(source["recovery_manifest"]["path"]))
    session = InvestigationSession(Path(manifest["session_path"]))
    binding = session.source["binding"]
    protocol, _ = DIAGNOSTIC._protocol(binding["protocol"])
    resolver = DIAGNOSTIC._resolver(binding["artifact_resolver"], protocol)
    observed = DIAGNOSTIC._source_observation(session.source["case_id"], protocol, resolver)
    candidate = GOVERNOR.build_model_packet(observed)["observation"].get("candidate", {})
    if (session.source["case_id"] != source["case_id"]
            or candidate.get("source_id") != source["source_case_id"]
            or candidate.get("task") != source["task"]):
        raise ValueError("source observation binding differs")
    return observed


def _host_gate(source: Mapping[str, Any]):
    return GOVERNOR.verify_host_gate_receipt({
        "schema": GOVERNOR.HOST_SOURCE_SCHEMA, "receipt": source["host_gate_policy"],
    })


def validate_outer(value: object, *, repo_root: Path = ROOT) -> dict[str, Any]:
    fields = {
        "schema", "status", "profile_id", "condition", "max_sdk_launches",
        "child_entrypoint", "inner_confirmation_protocol", "primary_plan",
        "primary_execution", "source_sha256_lf", "result_seen_design", "automatic_retry",
        "post_budget", "interpretation",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("outer protocol fields differ")
    if (value.get("schema") != OUTER_SCHEMA or value.get("status") != "registered"
            or value.get("profile_id") != POST.PROFILE_ID or value.get("condition") != "NONE"
            or value.get("max_sdk_launches") != POST_SDK_CAP
            or value.get("child_entrypoint") != CHILD_REL
            or value.get("result_seen_design") is not True or value.get("automatic_retry") is not False
            or value.get("interpretation") != {
                "checkpoint_association_only": True, "producer_loads_checkpoint": False,
                "trained_policy_validated": False,
            }):
        raise ValueError("outer protocol fixed values differ")
    sources = value.get("source_sha256_lf")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_RELATIONS):
        raise ValueError("outer operational closure filenames differ")
    for relative in SOURCE_RELATIONS:
        expected = sources.get(relative)
        path = repo_root / relative
        if (not isinstance(expected, str) or SHA.fullmatch(expected) is None
                or not path.is_file() or _sha_lf(path) != expected):
            raise ValueError(f"outer operational source changed: {relative}")
    inner = value.get("inner_confirmation_protocol")
    primary = value.get("primary_plan")
    if inner != {"path": INNER_REL, "sha256": _sha(repo_root / INNER_REL)}:
        raise ValueError("inner confirmation protocol differs")
    if (not isinstance(primary, Mapping) or set(primary) != {"path", "sha256"}
            or not isinstance(primary.get("path"), str)
            or primary["path"] != PurePath(primary["path"]).as_posix()
            or PurePath(primary["path"]).is_absolute() or ".." in PurePath(primary["path"]).parts
            or not primary["path"].endswith("_v13.json")
            or SHA.fullmatch(str(primary.get("sha256"))) is None
            or not (repo_root / primary["path"]).is_file()
            or primary["sha256"] != _sha(repo_root / primary["path"])):
        raise ValueError("primary plan differs")
    _primary_execution(value.get("primary_execution"))
    _post_budget(value.get("post_budget"))
    POST._protocol(_read(repo_root / INNER_REL))
    return deepcopy(dict(value))


def _primary_execution(value: object) -> dict[str, Any]:
    fields = {
        "registration_anchor", "source_anchor", "manifest_sha256", "run_anchor_sha256",
        "supervisor_sha256", "supervisor_mode", "supervisor_config_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("primary execution binding fields differ")
    for name in ("registration_anchor", "source_anchor"):
        if re.fullmatch(r"[0-9a-f]{40}", str(value.get(name))) is None:
            raise ValueError(f"primary execution {name} differs")
    for name in ("manifest_sha256", "run_anchor_sha256", "supervisor_sha256",
                 "supervisor_config_sha256"):
        if SHA.fullmatch(str(value.get(name))) is None:
            raise ValueError(f"primary execution {name} differs")
    if value.get("supervisor_mode") != "recovery_v13":
        raise ValueError("primary supervisor mode differs")
    return deepcopy(dict(value))


def _post_budget(value: object) -> dict[str, Any]:
    expected = {
        "schema": "post_actuator_separate_budget_v1",
        "primary_sdk_cap": PRIMARY_SDK_CAP,
        "post_sdk_cap": POST_SDK_CAP,
        "study_sdk_cap": STUDY_SDK_CAP,
        "api_cap": 0,
        "basis": "result_seen_post_actuator_preregistration",
    }
    if value != expected:
        raise ValueError("separate post budget differs")
    return deepcopy(expected)


def _git_anchor(anchor: str, outer: Mapping[str, Any], outer_path: Path, *, repo_root: Path = ROOT) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", anchor) is None:
        raise ValueError("registration anchor differs")
    subprocess.run(["git", "-C", str(repo_root), "merge-base", "--is-ancestor", anchor, "HEAD"],
                   check=True, capture_output=True)
    for relative, expected in outer["source_sha256_lf"].items():
        blob = subprocess.run(["git", "-C", str(repo_root), "show", f"{anchor}:{relative}"],
                              check=True, capture_output=True).stdout
        normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"registration anchor blob differs: {relative}")
    outer_relative = outer_path.relative_to(repo_root).as_posix()
    outer_blob = subprocess.run(["git", "-C", str(repo_root), "show", f"{anchor}:{outer_relative}"],
                                check=True, capture_output=True).stdout
    if hashlib.sha256(outer_blob).hexdigest() != _sha(outer_path):
        raise ValueError("registration anchor outer protocol blob differs")


def _authority(value: object, source_path: Path, outer_path: Path,
               outer: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema", "status", "profile_id", "case_id", "condition", "sdk_launches_approved",
        "approval_id", "request_id", "effect_key", "approved_by", "source_inputs",
        "outer_protocol", "primary_result", "primary_manifest", "primary_plan",
        "primary_supervisor",
        "source_sha256_lf", "registration_anchor", "result_seen_design", "automatic_retry",
        "approval_basis",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("authority fields differ")
    if (value.get("schema") != AUTHORITY_SCHEMA or value.get("status") != "approved"
            or value.get("profile_id") != POST.PROFILE_ID or value.get("case_id") != "case-05"
            or value.get("condition") != "NONE" or value.get("sdk_launches_approved") != 1
            or value.get("result_seen_design") is not True or value.get("automatic_retry") is not False
            or value.get("approval_basis") != "post_primary_completion_result_seen_development"):
        raise ValueError("authority fixed values differ")
    for name in ("approval_id", "request_id", "effect_key", "approved_by"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"authority {name} differs")
    if (_ref(value["source_inputs"], "authority source", (ROOT,))
            != {"path": str(source_path), "sha256": _sha(source_path)}
            or _ref(value["outer_protocol"], "authority protocol", (ROOT,))
            != {"path": str(outer_path), "sha256": _sha(outer_path)}):
        raise ValueError("authority source or protocol binding differs")
    if value.get("source_sha256_lf") != outer["source_sha256_lf"]:
        raise ValueError("authority operational closure differs")
    _git_anchor(value["registration_anchor"], outer, outer_path)
    return deepcopy(dict(value))


def _primary_budget(result_ref: object, manifest_ref: object, plan_ref: object,
                    supervisor_ref: object,
                    outer: Mapping[str, Any], *, repo_root: Path = ROOT,
                    status_fn: Callable[[Path], dict[str, Any]] | None = None) -> dict[str, Any]:
    result = _ref(result_ref, "primary result", (repo_root,))
    manifest = _ref(manifest_ref, "primary manifest", (repo_root,))
    plan = _ref(plan_ref, "primary plan", (repo_root,))
    supervisor = _ref(supervisor_ref, "primary supervisor", (repo_root,))
    result_path, manifest_path, plan_path = (Path(row["path"]) for row in (result, manifest, plan))
    run_root = manifest_path.parent
    anchor_path = run_root / "anchor.json"
    execution = _primary_execution(outer.get("primary_execution"))
    _post_budget(outer.get("post_budget"))
    if (result_path != run_root / "final-controller-result.json"
            or manifest_path != run_root / "manifest.json"
            or manifest["sha256"] != execution["manifest_sha256"]
            or not anchor_path.is_file() or _sha(anchor_path) != execution["run_anchor_sha256"]
            or plan_path != repo_root / outer["primary_plan"]["path"]
            or plan != {"path": str(plan_path), "sha256": outer["primary_plan"]["sha256"]}):
        raise ValueError("primary path contract differs")
    _verify_terminal_supervisor(supervisor, execution)
    plan_body, manifest_body, supplied = _read(plan_path), _read(manifest_path), _read(result_path)
    cases = plan_body.get("cases")
    methods = plan_body.get("methods")
    if (plan_body.get("schema") != PRIMARY_PLAN_SCHEMA
            or plan["sha256"] != outer["primary_plan"]["sha256"]
            or plan_body.get("source_anchor") != execution["source_anchor"]
            or not isinstance(cases, list) or len(cases) != 4
            or [row.get("case_id") for row in cases]
            != [f"case-{case:02d}" for case in range(5, 9)]
            or not isinstance(methods, list) or len(methods) != 4
            or len(set(methods)) != 4):
        raise ValueError("primary plan contract differs")
    expected_matrix: list[tuple[str, str]] = []
    for case in cases:
        rotation = case.get("rotation")
        if (not isinstance(rotation, list) or len(rotation) != 4
                or len(set(rotation)) != 4 or set(rotation) != set(methods)):
            raise ValueError("primary plan rotation differs")
        expected_matrix.extend((str(case["case_id"]), str(method)) for method in rotation)
    expected_matrix_tuple = tuple(expected_matrix)
    _verify_v13_plan_accounting(plan_body)
    PRIMARY.verify_prepared_manifest(
        run_root, repo_root=repo_root,
        registration_anchor=execution["registration_anchor"],
    )
    manifest_matrix = tuple((row.get("case_id"), row.get("method"))
                            for row in manifest_body.get("arms", []))
    if (manifest_body.get("schema") != "registered_recovery_comparison_result_v1_manifest"
            or manifest_body.get("registration_anchor") != execution["registration_anchor"]
            or manifest_body.get("plan") != plan
            or manifest_matrix != expected_matrix_tuple):
        raise ValueError("primary registered manifest differs")
    canonical = (status_fn or PRIMARY.status)(run_root)
    if canonical != supplied:
        raise ValueError("supplied primary result differs from canonical status")
    arms = supplied.get("arms")
    if (supplied.get("schema") != "registered_recovery_comparison_result_v1"
            or supplied.get("completed") is not True or supplied.get("halted") is not False
            or supplied.get("completed_arms") != 16 or supplied.get("total_arms") != 16
            or not isinstance(arms, list) or len(arms) != 16
            or tuple((row.get("case_id"), row.get("method")) for row in arms)
            != expected_matrix_tuple):
        raise ValueError("canonical primary result is incomplete")
    launches = 0
    imported = fresh = terminal_holds = 0
    for arm, row in zip(manifest_body["arms"], arms, strict=True):
        cost = row.get("resource_cost")
        if isinstance(arm.get("continuation_import"), Mapping):
            _verify_imported_arm(plan_body, arm, row, cost, repo_root)
            imported += 1
            continue
        if arm.get("recovery_payload") is None:
            _verify_terminal_hold_arm(arm, row, cost)
            terminal_holds += 1
            continue
        launches += _verify_fresh_arm(arm, row, cost)
        fresh += 1
    if (imported, fresh, terminal_holds, launches) != (
            EXPECTED_IMPORTED_ARMS, EXPECTED_FRESH_ARMS, EXPECTED_HOLD_ARMS,
            EXPECTED_PRIMARY_NEW_SDK):
        raise ValueError("primary 16-arm resource allocation differs")
    before_post = EXPECTED_PRIOR_SDK + launches
    remaining = STUDY_SDK_CAP - before_post
    if before_post != PRIMARY_SDK_CAP or remaining != POST_SDK_CAP:
        raise ValueError("separate post SDK budget differs")
    return {
        "prior_sdk_launches": EXPECTED_PRIOR_SDK, "primary_sdk_launches": launches,
        "actual_before_post": before_post, "primary_sdk_cap": PRIMARY_SDK_CAP,
        "study_sdk_cap": STUDY_SDK_CAP, "remaining_before_post": remaining,
        "post_sdk_cap": POST_SDK_CAP,
        "primary_result": result, "primary_manifest": manifest, "primary_plan": plan,
        "primary_supervisor": supervisor,
        "canonical_status_sha256": _digest(canonical),
        "arm_accounting": {"verified_imports": imported, "fresh_recoveries": fresh,
                           "verified_terminal_holds": terminal_holds},
    }


def _verify_v13_plan_accounting(plan: Mapping[str, Any]) -> None:
    continuation = plan.get("continuation")
    expected_seeds = {
        "case-05": {"resume": 2664, "restart": 5664},
        "case-06": {"resume": 2665, "restart": 5665},
        "case-07": {"resume": 2666, "restart": 6266},
        "case-08": {"resume": 2667, "restart": 6267},
    }
    if (not isinstance(continuation, Mapping)
            or continuation.get("prior_consumed_gpu_sdk_launches") != EXPECTED_PRIOR_SDK
            or continuation.get("expected_new_gpu_sdk_launches") != EXPECTED_PRIMARY_NEW_SDK
            or continuation.get("new_gpu_sdk_launch_cap") != EXPECTED_PRIMARY_NEW_SDK
            or continuation.get("cumulative_gpu_sdk_launches") != PRIMARY_SDK_CAP
            or continuation.get("completed_import_profile")
            != "motor_completed_epoch_namespace_retry_v1"
            or continuation.get("failure_proof_sha256") != V11_FAILURE_PROOF_SHA256
            or continuation.get("namespace_failure_context_sha256")
            != V12_NAMESPACE_CONTEXT_SHA256
            or continuation.get("v11_failed_epoch_sdk_launches") != 1
            or plan.get("recovery_policy", {}).get("seeds") != expected_seeds):
        raise ValueError("V13 primary accounting/profile differs")


def _verify_terminal_supervisor(supervisor: Mapping[str, Any],
                                execution: Mapping[str, Any]) -> None:
    if (supervisor.get("sha256") != execution["supervisor_sha256"]
            or Path(str(supervisor.get("path"))).name != "supervisor-exit.json"):
        raise ValueError("primary supervisor reference differs")
    body = _read(Path(supervisor["path"]))
    required = {
        "job_assigned_before_release": True,
        "natural_exit": True,
        "timed_out": False,
        "error_type": None,
        "exit_code": 0,
        "cleanup_verified": True,
        "mode": execution["supervisor_mode"],
        "config_sha256": execution["supervisor_config_sha256"],
        "role": "outer_controller_not_a_simulator_sdk_launch",
    }
    if any(body.get(name) != expected for name, expected in required.items()):
        raise ValueError("primary terminal supervisor is not successful")


def _verify_imported_arm(plan: Mapping[str, Any], arm: Mapping[str, Any],
                         row: Mapping[str, Any], cost: object, repo_root: Path) -> None:
    if (row.get("execution_origin") != "verified_prior_effect"
            or row.get("recovery_started") is not True
            or not isinstance(cost, Mapping)
            or cost.get("schema") != "registered_recovery_imported_resource_v1"
            or cost.get("execution_origin") != "verified_prior_effect"):
        raise ValueError("primary imported arm differs")
    current = cost.get("new_dispatch_costs")
    historical = cost.get("historical_effect_costs")
    if (not isinstance(current, Mapping) or current.get("simulator_sdk_launches") != 0
            or not isinstance(historical, Mapping)
            or historical.get("simulator_sdk_launches") != 2):
        raise ValueError("primary imported arm cost differs")
    binding = PRIMARY._stage_import(dict(plan), dict(arm))
    verified = PRIMARY._continuation_module(dict(plan)).verify_completed_effect(
        arm["continuation_import"]["proof"], expected_import=binding,
        repo=repo_root.resolve(strict=True),
    )
    if (not isinstance(verified, Mapping)
            or verified.get("execution_origin") != "verified_prior_effect"
            or verified.get("proof_ref") != arm["continuation_import"]["proof"]):
        raise ValueError("primary continuation proof differs")


def _verify_terminal_hold_arm(arm: Mapping[str, Any], row: Mapping[str, Any], cost: object) -> None:
    if (arm.get("continuation_import") is not None or arm.get("recovery_payload") is not None
            or arm.get("method") not in {"one_shot_llm", "iterative_agent"}
            or cost is not None or row.get("outcome") != "hold"
            or row.get("diagnostic_status") != "model_hold"
            or row.get("recovery_started") is not False):
        raise ValueError("primary terminal HOLD arm differs")
    diagnostic = PRIMARY._verified_diagnostic_status(Path(arm["session_path"]))
    if (diagnostic.get("status") != "model_hold"
            or diagnostic.get("new_model_call") is not False
            or not isinstance(diagnostic.get("decision_terminal"), Mapping)):
        raise ValueError("primary terminal HOLD replay differs")


def _verify_fresh_arm(arm: Mapping[str, Any], row: Mapping[str, Any], cost: object) -> int:
    if (arm.get("continuation_import") is not None or not isinstance(arm.get("recovery_payload"), Mapping)
            or row.get("recovery_started") is not True or not isinstance(cost, Mapping)
            or cost.get("schema") != "registered_recovery_resource_receipt_v1"
            or cost.get("status") != "complete" or cost.get("simulator_sdk_launches") != 2
            or not isinstance(cost.get("admission_ref"), Mapping)
            or not isinstance(cost.get("recovery_resource_ref"), Mapping)
            or not isinstance(cost.get("execution_refs"), Mapping)
            or set(cost["execution_refs"]) != {"checkpoint", "fixed", "telemetry", "training"}
            or not isinstance(cost["execution_refs"].get("fixed"), Mapping)
            or not isinstance(cost["execution_refs"].get("training"), Mapping)):
        raise ValueError("primary fresh recovery cost differs")
    state = RECOVERY.status(Path(arm["recovery_folder"]))
    requests = state.get("requests")
    if not isinstance(requests, Mapping) or len(requests) != 1:
        raise ValueError("primary recovery ledger differs")
    request = next(iter(requests.values()))
    receipt = request.get("receipt") if isinstance(request, Mapping) else None
    if (request.get("state") != "done" or not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("result_sha"), str) or not receipt["result_sha"]
            or receipt.get("verdict") != row.get("outcome")):
        raise ValueError("primary recovery ledger is not a completed effect")
    return 2


def _build_post_request(source: Mapping[str, Any], authority: Mapping[str, Any],
                        lease: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": POST.REQUEST_SCHEMA, "profile_id": POST.PROFILE_ID,
        "case_id": source["case_id"], "source_case_id": source["source_case_id"],
        "seed": source["seed"], "task": source["task"], "condition": "NONE",
        "checkpoint": source["checkpoint"], "fixed_report": source["fixed_report"],
        "training_receipt": source["training_receipt"],
        "correction_readback": source["correction_readback"],
        "recovery_receipt": source["recovery_receipt"],
        "source_protocol": source["source_protocol"], "runtime_closure": source["runtime_closure"],
        "lease": {"profile_id": GPU.PROFILE_ID, "lease_id": lease["leaseId"],
                  "caller_pid": os.getpid()},
        "effect": {"effect_key": authority["effect_key"], "request_id": authority["request_id"]},
        "request": {"approval_id": authority["approval_id"], "request_id": authority["request_id"]},
        "sdk_budget": 1, "result_seen_design": True,
    }


def _producer_manifest(tag: str, isaaclab: Path) -> tuple[Path, dict[str, Any]]:
    protocol_path = PRODUCER.protocol_location(PRODUCER.PROTOCOL)
    protocol = _read(protocol_path)
    source = PRODUCER.validate_protocol(protocol, repo_root=ROOT, isaaclab_root=isaaclab)
    folder = PRODUCER.PRIVATE_ROOT / tag
    if folder.exists():
        raise FileExistsError("producer tag already exists")
    folder.mkdir(parents=True, exist_ok=False)
    request = PRODUCER._request(tag, protocol_path, protocol)
    OWNED.write_new(folder / "request.json", request)
    consumption = {
        "schema": "actuator_envelope_runtime_consumption_v1",
        "request_id": request["request_id"], "tag": tag,
        "request_sha256": PRODUCER.S._canonical_sha(request),
    }
    OWNED.write_new(folder / "consumed.json", consumption)
    condition = folder / "NONE"
    condition.mkdir()
    manifest_path = condition / "manifest.json"
    manifest = {
        "schema": "actuator_envelope_runtime_manifest_v1",
        "protocol_path": str(protocol_path), "protocol_sha256": _sha(protocol_path),
        "condition": "NONE", "output_path": str(condition / "body.json"),
        "request_path": str(folder / "request.json"),
        "consumption_path": str(folder / "consumed.json"),
    }
    OWNED.write_new(manifest_path, manifest)
    validated = PRODUCER.validate_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab)
    if validated["source_receipt"] != source:
        raise ValueError("producer source changed after preflight")
    return manifest_path, validated


def _scope(source: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    core = {
        "profile_id": POST.PROFILE_ID, "case_id": source["case_id"],
        "checkpoint_sha256": source["checkpoint"]["sha256"],
        "recovery_receipt_sha256": source["recovery_receipt"]["sha256"],
    }
    return _digest(core), core


def _failure(out_dir: Path, *, phase: str, error: BaseException, sdk_attempted: bool,
             consumption: Mapping[str, str] | None, execution: Mapping[str, str] | None) -> dict[str, Any]:
    body = {
        "schema": "post_actuator_operational_failure_v2", "status": "invalid",
        "phase": phase, "error_type": type(error).__name__, "sdk_attempted": sdk_attempted,
        "consumption": deepcopy(consumption), "execution": deepcopy(execution),
        "automatic_retry": False,
    }
    _write_new(out_dir / "failure.json", body)
    return body


def _admit_and_run(query: Callable[[], dict[str, Any]],
                   launch: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu = query()
    if gpu.get("busy"):
        raise OperationalError("fresh shared GPU admission became busy after consumption")
    return gpu, launch()


def run(source_path: Path, outer_path: Path, authority_path: Path, isaaclab: Path,
        tag: str, out_dir: Path) -> dict[str, Any]:
    source_path, outer_path, authority_path, isaaclab, out_dir = _precheck_cli(
        source_path, outer_path, authority_path, isaaclab, out_dir, tag,
    )
    if out_dir.exists():
        raise FileExistsError("output tag folder already exists")
    source_path, outer_path, authority_path = (
        source_path.resolve(strict=True), outer_path.resolve(strict=True), authority_path.resolve(strict=True))
    isaaclab = isaaclab.resolve(strict=True)
    source = _source(_read(source_path))
    outer = validate_outer(_read(outer_path))
    authority = _authority(_read(authority_path), source_path, outer_path, outer)
    budget = _primary_budget(
        authority["primary_result"], authority["primary_manifest"],
        authority["primary_plan"], authority["primary_supervisor"], outer,
    )
    scope_key, scope_core = _scope(source)
    consumed_path = CONSUMED_ROOT / f"{scope_key}.json"
    if consumed_path.exists():
        raise FileExistsError("post-actuator scope was already consumed")
    inner = POST._protocol(_read(ROOT / INNER_REL))
    lease_before = LEASE.lease_context()
    if lease_before.get("owner") != LEASE.OWNER:
        raise ValueError("lease owner differs")
    gpu_initial = GPU.query_registered_desktop_gpu_shared()
    if gpu_initial.get("busy"):
        return {"schema": RESULT_SCHEMA, "status": "waiting_gpu_idle", "sdk_launches": 0,
                "trained_policy_validated": False, "budget": budget}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)
    operation_id = _digest({"scope_key": scope_key, "authority_sha256": _sha(authority_path), "tag": tag})
    outer_ref = {"path": str(outer_path), "sha256": _sha(outer_path)}
    authority_ref = {"path": str(authority_path), "sha256": _sha(authority_path)}
    source_ref = {"path": str(source_path), "sha256": _sha(source_path)}
    intent = {
        "schema": "post_actuator_operation_intent_v2", "status": "reserved_pre_sdk",
        "operation_id": operation_id, "scope_key": scope_key, "condition": "NONE",
        "checkpoint_sha256": scope_core["checkpoint_sha256"],
        "recovery_receipt_sha256": scope_core["recovery_receipt_sha256"],
        "authority_sha256": authority_ref["sha256"], "effect_key": authority["effect_key"],
        "outer_protocol": outer_ref, "authority": authority_ref, "source_inputs": source_ref,
        "tag": tag, "sdk_cap": 1,
    }
    intent_path = out_dir / "intent.json"
    _write_new(intent_path, intent)
    intent_ref = {"path": str(intent_path), "sha256": _sha(intent_path)}
    phase, sdk_attempted = "manifest_preparation", False
    consumption_ref: dict[str, str] | None = None
    execution_ref: dict[str, str] | None = None
    try:
        plan = POST.prepare(_build_post_request(source, authority, lease_before), inner)
        manifest_path, validated = _producer_manifest(tag, isaaclab)
        manifest_ref = {"path": str(manifest_path), "sha256": _sha(manifest_path)}
        if LEASE.lease_context().get("leaseId") != lease_before.get("leaseId"):
            raise ValueError("lease changed after manifest validation")
        phase = "fresh_gpu_admission"
        gpu_fresh = GPU.query_registered_desktop_gpu_shared()
        if gpu_fresh.get("busy"):
            waiting = {"schema": RESULT_SCHEMA, "status": "waiting_gpu_idle_after_manifest",
                       "sdk_launches": 0, "intent": intent_ref, "budget": budget,
                       "trained_policy_validated": False}
            _write_new(out_dir / "waiting.json", waiting)
            return waiting
        phase = "global_consumption"
        consumption = {
            "schema": "post_actuator_global_consumption_v2", "status": "consumed_pre_sdk",
            "operation_id": operation_id, "scope_key": scope_key, "condition": "NONE",
            "checkpoint_sha256": scope_core["checkpoint_sha256"],
            "recovery_receipt_sha256": scope_core["recovery_receipt_sha256"],
            "authority_sha256": authority_ref["sha256"], "effect_key": authority["effect_key"],
            "outer_protocol": outer_ref, "intent": intent_ref,
            "approval_id": authority["approval_id"], "request_id": authority["request_id"],
        }
        _write_new(consumed_path, consumption)
        consumption_ref = {"path": str(consumed_path), "sha256": _sha(consumed_path)}
        context = {
            "schema": "post_actuator_child_context_v1", "operation_id": operation_id,
            "scope_key": scope_key, "condition": "NONE", "outer_protocol": outer_ref,
            "intent": intent_ref, "consumption": consumption_ref, "producer_manifest": manifest_ref,
            "checkpoint_sha256": scope_core["checkpoint_sha256"],
            "recovery_receipt_sha256": scope_core["recovery_receipt_sha256"],
            "authority_sha256": authority_ref["sha256"], "effect_key": authority["effect_key"],
            "body_entrypoint": "bench.actuator_envelope_runtime.child_main",
        }
        context_path = out_dir / "child-context.json"
        _write_new(context_path, context)
        if LEASE.lease_context().get("leaseId") != lease_before.get("leaseId"):
            raise ValueError("lease changed before final GPU admission")
        phase = "final_gpu_admission"
        condition_folder = manifest_path.parent
        def launch_owned() -> dict[str, Any]:
            nonlocal phase, sdk_attempted
            phase, sdk_attempted = "owned_sdk", True
            return OWNED.run_owned(
                condition_folder,
                [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / CHILD_REL),
                 "--context", str(context_path)],
                isaaclab, validated["protocol"]["launcher_watchdog_s"],
            )

        gpu_before, execution = _admit_and_run(
            GPU.query_registered_desktop_gpu_shared,
            launch_owned,
        )
        execution_path = condition_folder / "execution.json"
        OWNED.write_new(execution_path, execution)
        execution_ref = {"path": str(execution_path), "sha256": _sha(execution_path)}
        phase = "component_validation"
        body_path = Path(validated["paths"]["output_path"])
        body = JSON_IO._read_json(body_path)
        reasons = PRODUCER.verify_body(body, validated, execution)
        reasons.extend(PRODUCER._shutdown_handoff_reasons(condition_folder, body, _sha(body_path)))
        reasons.extend(PRODUCER._owned_execution_reasons(execution))
        reasons.extend(PRODUCER._execution_log_reasons(condition_folder))
        if reasons:
            raise ValueError("component validation failed: " + ",".join(sorted(set(reasons))))
        if LEASE.lease_context().get("leaseId") != lease_before.get("leaseId"):
            raise ValueError("lease changed after owned child")
        gpu_after = GPU.query_registered_desktop_gpu_shared()
        owned_pids = set(execution.get("owned_process_identities", {}))
        if any(str(row.get("pid")) in owned_pids for row in gpu_after.get("processes", [])):
            raise ValueError("owned GPU process remains after post-check")
        component = POST.verify_component_receipt(
            plan, manifest_path, body_path, execution_path, repo_root=ROOT, isaaclab_root=isaaclab,
        )
        phase = "supplementary_decision"
        decision = POST.reverify_and_supplement(
            plan, _source_observation(source), Path(source["recovery_folder"]),
            source["recovery_request_id"], component, _host_gate(source),
            repo_root=ROOT, isaaclab_root=isaaclab,
        )
        result = {
            "schema": RESULT_SCHEMA, "status": "complete", "operation_id": operation_id,
            "tag": tag, "condition": "NONE", "sdk_launches": 1, "outer_sdk_cap": 1,
            "base_request_conditions": list(BODY.CONDITIONS), "executed_conditions": ["NONE"],
            "intent": intent_ref, "consumption": consumption_ref, "execution": execution_ref,
            "budget": {**budget, "remaining_after_post": budget["remaining_before_post"] - 1},
            "component_receipt": component, "supplementary_decision": decision,
            "gpu_before": gpu_before, "gpu_after": gpu_after, "lease_id": lease_before["leaseId"],
            "checkpoint_association_only": True, "producer_loads_checkpoint": False,
            "trained_policy_validated": False, "automatic_retry": False,
        }
        _write_new(out_dir / "result.json", result)
        return result
    except BaseException as error:
        _failure(out_dir, phase=phase, error=error, sdk_attempted=sdk_attempted,
                 consumption=consumption_ref, execution=execution_ref)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--source-inputs", required=True, type=Path)
    parser.add_argument("--outer-protocol", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.source_inputs, args.outer_protocol, args.authority,
                     args.isaaclab, args.tag, args.out_dir)
        print(json.dumps({"status": result["status"], "sdk_launches": result["sdk_launches"]}, sort_keys=True))
        return 0 if result["status"] == "complete" else 3
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "rejected", "error_type": type(error).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
