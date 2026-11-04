"""등록 mechanism probe의 production 진입점과 실행 가능성을 CPU에서 확인한다.

이 모듈은 GPU/SDK를 실행하지 않는다. 하나의 승인·lease 아래에서 실행할 outer
backend가 소비할 기존 launcher/child와 관측 의미를 검증한다.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA = "registered_mechanism_observer_request_v1"
PREFLIGHT_SCHEMA = "registered_mechanism_observer_preflight_v1"
PROBES = ("P_actuator_mapping", "P_optimizer_count", "P_optimizer_gradient")
ACTUATOR_CONDITIONS = ("NONE", "F2", "F3", "F4")
OPTIMIZER_ARMS = ("NONE", "R1", "INVERSE_RATIO", "RESTORED")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
LEASE_ID = re.compile(r"[a-f0-9-]{36}\Z")

ENTRYPOINTS = {
    "actuator": {
        "launcher": "bench/run_actuator_envelope_runtime.py",
        "child": "bench/actuator_envelope_runtime.py",
        "protocol": "bench/protocols/actuator_envelope_runtime_v3.json",
        "launcher_functions": ("validate_protocol", "validate_manifest", "verify_body", "execute"),
        "child_functions": ("validate_curve", "run_sdk", "child_main"),
        "run_owned": "run_runtime_readback_smoke.run_owned",
        "checkpoint_mode": "unsupported",
        "lease_mode": "gpu_admission_only_not_shared_lease_bound",
    },
    "optimizer_count": {
        "launcher": "evals/novel_optimizer_confirmation.py",
        "child": "bench/inspect_optimizer_confirmation.py",
        "worker": "bench/inspect_optimizer_observability.py",
        "protocol": "bench/protocols/novel_optimizer_confirmation_v1.json",
        "launcher_functions": ("verify_sources", "verify_runtime", "_inspect_command", "inspect_report", "execute"),
        "child_functions": ("binding", "main"),
        "worker_functions": ("verify_training_binding", "train", "main"),
        "run_owned": "run_runtime_readback_smoke.run_owned",
        "checkpoint_mode": "fresh_initialization_only",
        "lease_mode": "gpu_admission_only_not_shared_lease_bound",
    },
    "optimizer_gradient": {
        "launcher": "evals/ppo_gradient_check_v5.py",
        "child": "bench/ppo_gradient_training_v5.py",
        "inspector": "evals/ppo_gradient_invariant_v2.py",
        "contract": "evals/ppo_gradient_invariant_v1.py",
        "protocol": "bench/protocols/ppo_gradient_check_v5.json",
        "launcher_functions": ("verify", "manifest", "training_args", "execute_owned", "inspect_run", "run"),
        "child_functions": ("verify_training", "attach", "training_hooks", "train"),
        "inspector_functions": ("GradientObserver",),
        "contract_functions": ("validate_manifest", "inspect_records"),
        "run_owned": "run_optimizer_observability_batch.OWNED.run_owned",
        "checkpoint_mode": "fresh_initialization_only",
        "lease_mode": "parent_and_child_lease_context",
    },
}

ISAAC_RUNTIME_FILES = (
    "_isaac_sim/python.bat",
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "source/isaaclab/isaaclab/actuators/actuator_pd.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
)


def _canonical(value: object) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical JSON is required") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_id(core: Mapping[str, Any]) -> str:
    """request_id 필드를 제외한 canonical 요청의 digest를 만든다."""
    return _digest(dict(core))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (
        _ for _ in ()).throw(ValueError(f"non-finite JSON constant: {item}")))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")  # noqa: TRY004 - JSON 계약 오류 형식 유지
    return value


def _source_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def production_entrypoints(repo_root: Path = ROOT) -> dict[str, Any]:
    """기존 production 파일·protocol·필수 함수의 현재 SHA 결합을 반환한다."""
    root = Path(repo_root).resolve()
    result: dict[str, Any] = {}
    for name, spec in ENTRYPOINTS.items():
        code_roles = tuple(
            role for role in ("launcher", "child", "worker", "inspector", "contract") if role in spec
        )
        roles = (*code_roles, "protocol")
        files = {role: root / spec[role] for role in roles}
        missing = [f"{role}:{path}" for role, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("production entrypoint missing: " + ", ".join(missing))
        for role in code_roles:
            found = _source_functions(files[role])
            expected = set(spec[f"{role}_functions"])
            if not expected <= found:
                raise ValueError(f"{name} {role} functions differ: {sorted(expected - found)}")
        _read_object(files["protocol"])
        result[name] = {
            **{key: value for key, value in spec.items() if not key.endswith("_functions")},
            "source_sha256": {role: _sha(path) for role, path in files.items()},
            "verified_functions": {
                role: list(spec[f"{role}_functions"])
                for role in code_roles
            },
        }
    return result


def component_catalog() -> dict[str, Any]:
    """outer backend가 실행할 두 component의 관측 의미를 고정한다."""
    return {
        "actuator_mapping": {
            "probes": ["P_actuator_mapping"],
            "runtime_callable": "bench.actuator_envelope_runtime.run_sdk",
            "parent_validators": [
                "bench.run_actuator_envelope_runtime.validate_protocol",
                "bench.run_actuator_envelope_runtime.validate_manifest",
                "bench.run_actuator_envelope_runtime.verify_body",
            ],
            "semantics": "policy_free_runtime_config_readback",
            "reads": ["source_code", "runtime_config", "actuator_fault_state"],
            "checkpoint_consumed": False,
            "policy_inference_used": False,
            "runtime_updates": 0,
        },
        "optimizer_observation": {
            "probes": ["P_optimizer_count", "P_optimizer_gradient"],
            "runtime_callable": "bench.ppo_gradient_training_v5.attach",
            "receipt_inspector": "evals.ppo_gradient_invariant_v2.inspect_records",
            "semantics": "fresh_three_update_runtime_fault_observation",
            "reads": ["runtime_source", "ppo_config", "optimizer_fault_state"],
            "checkpoint_consumed": False,
            "infers_past_checkpoint_updates": False,
            "runtime_updates": 3,
        },
    }


def _checked_checkpoint(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} checkpoint fields differ")
    if not isinstance(value["path"], str) or not isinstance(value["sha256"], str):
        raise ValueError(f"{label} checkpoint values differ")  # noqa: TRY004 - JSON 계약 오류 형식 유지
    path = Path(value["path"]).resolve()
    if not path.is_file() or SHA256.fullmatch(value["sha256"]) is None:
        raise ValueError(f"{label} checkpoint is missing or SHA is invalid")
    if _sha(path) != value["sha256"]:
        raise ValueError(f"{label} checkpoint SHA mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def _checked_endpoint(value: object, label: str) -> dict[str, Any]:
    fields = {"case_id", "checkpoint", "actuator_condition", "optimizer_arm"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} endpoint fields differ")
    if not isinstance(value["case_id"], str) or not REQUEST_ID.fullmatch(value["case_id"]):
        raise ValueError(f"{label} case_id is invalid")
    if value["actuator_condition"] not in ACTUATOR_CONDITIONS:
        raise ValueError(f"{label} actuator condition is unsupported")
    if value["optimizer_arm"] not in OPTIMIZER_ARMS:
        raise ValueError(f"{label} optimizer arm is unsupported")
    return {
        "case_id": value["case_id"],
        "checkpoint": _checked_checkpoint(value["checkpoint"], label),
        "actuator_condition": value["actuator_condition"],
        "optimizer_arm": value["optimizer_arm"],
    }


def normalize_request(value: object) -> dict[str, Any]:
    fields = {"schema", "request_id", "approval", "lease", "probes", "runtime_updates", "case", "reference"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != REQUEST_SCHEMA:
        raise ValueError("observer request fields or schema differ")
    core = {key: value[key] for key in fields - {"request_id"}}
    if not isinstance(value["request_id"], str) or value["request_id"] != request_id(core):
        raise ValueError("observer request_id differs from canonical request")
    approval = value["approval"]
    if (not isinstance(approval, Mapping)
            or set(approval) != {"request_id", "effect_key", "registry_digest"}
            or any(not isinstance(approval[key], str) or not approval[key] for key in approval)):
        raise ValueError("approval binding differs")
    if SHA256.fullmatch(approval["registry_digest"]) is None:
        raise ValueError("approval registry digest is invalid")
    lease = value["lease"]
    if (not isinstance(lease, Mapping) or set(lease) != {"lease_id", "owner"}
            or not isinstance(lease["lease_id"], str) or LEASE_ID.fullmatch(lease["lease_id"]) is None
            or not isinstance(lease["owner"], str) or not lease["owner"]):
        raise ValueError("lease binding differs")
    if value["probes"] != list(PROBES) or value["runtime_updates"] != 3:
        raise ValueError("registered probes or 3-update budget differ")
    case = _checked_endpoint(value["case"], "case")
    reference = _checked_endpoint(value["reference"], "reference")
    if case["case_id"] == reference["case_id"]:
        raise ValueError("case and reference identities must differ")
    normalized = {
        "schema": REQUEST_SCHEMA,
        "approval": dict(approval),
        "lease": dict(lease),
        "probes": list(PROBES),
        "runtime_updates": 3,
        "case": case,
        "reference": reference,
    }
    normalized["request_id"] = request_id(normalized)
    if normalized["request_id"] != value["request_id"]:
        raise ValueError("normalized request binding differs")
    return normalized


def preflight(value: object, *, repo_root: Path = ROOT, isaaclab_root: Path) -> dict[str, Any]:
    """GPU 없이 outer backend가 소비할 component/source/input 결합을 판정한다."""
    request = normalize_request(value)
    entries = production_entrypoints(repo_root)
    isaaclab = Path(isaaclab_root).resolve()
    runtime = {}
    missing_runtime = []
    for relative in ISAAC_RUNTIME_FILES:
        path = isaaclab / relative
        if not path.is_file():
            missing_runtime.append(relative)
        else:
            runtime[relative] = _sha(path)

    status = "ready_component" if not missing_runtime else "incomplete_runtime"
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "standalone_executable": False,
        "outer_backend_ready": not missing_runtime,
        "request_id": request["request_id"],
        "approval": request["approval"],
        "lease": request["lease"],
        "input_binding": {
            "case": request["case"],
            "reference": request["reference"],
            "probes": request["probes"],
            "runtime_updates": request["runtime_updates"],
        },
        "production_entrypoints": entries,
        "components": component_catalog(),
        "isaac_runtime_sha256": runtime,
        "runtime_missing": missing_runtime,
        "integration_requirements": [
            "outer backend verifies the same approval/effect and lease ID before each child",
            "outer backend binds the new registered manifest and fault state to each component receipt",
            "case/reference checkpoints remain common input identity with checkpoint_consumed=false",
            "fresh 3-update optimizer evidence is not a claim about past checkpoint updates",
        ],
        "evidence_roles": {
            "cpu_source_preflight": "current source/protocol paths, functions, checkpoint bytes and SHA are checked",
            "recorded_replay": "not executed and cannot be reported as a new probe",
            "new_sdk_measurement": "not executed by this CPU-only component preflight",
        },
        "next_contract": {
            "owner": "main U3 protocol",
            "requirements": [
                "one approval/effect and lease ID verified by every child",
                "new protocol/seed/tag registered before GPU execution",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight",))
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = preflight(
            _read_object(args.request.resolve()),
            repo_root=args.repo_root.resolve(),
            isaaclab_root=args.isaaclab.resolve(),
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["status"] == "ready_component" else 3
    except (ValueError, FileNotFoundError, OSError) as error:
        print(json.dumps({"schema": PREFLIGHT_SCHEMA, "status": "rejected",
                          "reason": type(error).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
