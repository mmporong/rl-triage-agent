"""등록 manifest만 소비하는 actuator/optimizer 진단 child.

모듈 import는 표준 라이브러리만 사용한다. Isaac/torch/RSL 모듈은 검증된
``child_main`` 실행 경계 뒤에서만 불러온다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
SCHEMA = "diagnostic_observer_child_v1"
PROTOCOL_SCHEMA = "diagnostic_observer_protocol_v1"
REGISTERED_PROTOCOL = ROOT / "bench/protocols/registered_diagnostic_observer_v1.json"
COMPONENTS = ("actuator_mapping", "optimizer_update_bundle")
CONDITIONS = ("NONE", "F2", "R1", "inverse_ratio")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
LEASE_ID = re.compile(r"[a-f0-9-]{36}\Z")
ACTUATOR_PROBES = ["P_actuator_mapping"]
OPTIMIZER_PROBES = ["P_optimizer_count", "P_optimizer_gradient"]
EXPECTED_EPOCHS = 5
EXPECTED_MINI_BATCHES = 4
OPTIMIZER_UPDATES = 3

MANIFEST_FIELDS = {
    "schema", "operation", "request_id", "effect_key", "probe", "case", "source",
    "checkpoint", "reference", "protocol", "source_anchor", "source_closure",
    "runtime_pins", "seed", "device", "num_envs", "condition", "component", "updates",
    "expected_learning_epochs", "expected_mini_batches", "outer_lease",
    "consumed_approval", "output", "checkpoint_consumed",
    "infers_past_checkpoint_updates", "condition_model_visible",
}
PROTOCOL_FIELDS = {
    "schema", "protocol_id", "source_anchor", "source_closure", "runtime_pins", "task",
    "actuator_base_protocol", "allowed_operations", "limits",
}


def _canonical(value: object) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical JSON is required") from error


def _json_copy(value: object) -> Any:
    return json.loads(_canonical(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (
        _ for _ in ()).throw(ValueError(f"non-finite JSON constant: {item}")))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")  # noqa: TRY004 - JSON 계약 오류 형식 유지
    return value


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _binding(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} binding must be a non-empty object")
    return _json_copy(dict(value))


def _file_ref(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} file reference differs")
    path, digest = value.get("path"), value.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} file reference is invalid")
    resolved = Path(path).resolve()
    if not resolved.is_file() or _sha(resolved) != digest:
        raise ValueError(f"{name} file changed")
    return {"path": str(resolved), "sha256": digest}


def _checkpoint(value: object) -> dict[str, Any]:
    result = _binding(value, "checkpoint")
    if set(result) != {"path", "sha256", "role"} or not isinstance(result["role"], str):
        raise ValueError("checkpoint binding fields differ")
    checked = _file_ref({"path": result["path"], "sha256": result["sha256"]}, "checkpoint")
    return {**checked, "role": result["role"]}


def _protocol(path: Path) -> dict[str, Any]:
    value = _read(path)
    if set(value) != PROTOCOL_FIELDS or value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("registered diagnostic protocol fields differ")
    if not isinstance(value["protocol_id"], str) or not IDENTIFIER.fullmatch(value["protocol_id"]):
        raise ValueError("protocol_id is invalid")
    if not isinstance(value["task"], str) or not value["task"]:
        raise ValueError("protocol task is missing")
    limits = value["limits"]
    if limits != {"expected_learning_epochs": EXPECTED_EPOCHS,
                  "expected_mini_batches": EXPECTED_MINI_BATCHES,
                  "optimizer_updates": OPTIMIZER_UPDATES}:
        raise ValueError("protocol optimizer limits differ")
    operations = value["allowed_operations"]
    expected = {
        "actuator_mapping": {"component": "actuator_mapping", "probe": ACTUATOR_PROBES,
                             "conditions": ["NONE", "F2"], "updates": 0},
        "optimizer_update_bundle": {"component": "optimizer_update_bundle",
                                    "probe": OPTIMIZER_PROBES,
                                    "conditions": ["NONE", "R1", "inverse_ratio"],
                                    "updates": OPTIMIZER_UPDATES},
    }
    if operations != expected:
        raise ValueError("protocol allowed operations differ")
    base = value["actuator_base_protocol"]
    if (not isinstance(base, Mapping) or set(base) != {"path", "sha256"}
            or not isinstance(base.get("path"), str) or Path(base["path"]).is_absolute()
            or ".." in Path(base["path"]).parts
            or not isinstance(base.get("sha256"), str) or SHA256.fullmatch(base["sha256"]) is None):
        raise ValueError("actuator base protocol reference differs")
    return value


def build_manifest(
    *,
    protocol_path: Path,
    operation: str,
    request_id: str,
    effect_key: str,
    probe: str | list[str],
    case_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    reference_binding: Mapping[str, Any],
    source_anchor: str,
    source_closure: Mapping[str, str],
    runtime_pins: Mapping[str, str],
    seed: int,
    device: str,
    num_envs: int,
    condition: str,
    component: str,
    outer_lease: Mapping[str, Any],
    consumed_approval: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """parent가 write-once로 저장할 exact child manifest를 만든다."""
    protocol_path = Path(protocol_path).resolve()
    protocol = _protocol(protocol_path)
    operation_spec = protocol["allowed_operations"].get(operation)
    probes = [probe] if isinstance(probe, str) else list(probe)
    if (operation_spec is None or component != operation_spec["component"]
            or probes != operation_spec["probe"] or condition not in operation_spec["conditions"]):
        raise ValueError("operation, component, probe, or condition differs from protocol")
    if source_anchor != protocol["source_anchor"] or dict(source_closure) != protocol["source_closure"]:
        raise ValueError("source anchor or closure differs from protocol")
    if dict(runtime_pins) != protocol["runtime_pins"]:
        raise ValueError("runtime pins differ from protocol")
    updates = operation_spec["updates"]
    return {
        "schema": SCHEMA,
        "operation": operation,
        "request_id": request_id,
        "effect_key": effect_key,
        "probe": probes,
        "case": _binding(case_binding, "case"),
        "source": _binding(source_binding, "source"),
        "checkpoint": _binding(checkpoint_binding, "checkpoint"),
        "reference": _binding(reference_binding, "reference"),
        "protocol": {"path": str(protocol_path), "sha256": _sha(protocol_path)},
        "source_anchor": source_anchor,
        "source_closure": dict(source_closure),
        "runtime_pins": dict(runtime_pins),
        "seed": seed,
        "device": device,
        "num_envs": num_envs,
        "condition": condition,
        "component": component,
        "updates": updates,
        "expected_learning_epochs": EXPECTED_EPOCHS,
        "expected_mini_batches": EXPECTED_MINI_BATCHES,
        "outer_lease": _binding(outer_lease, "outer_lease"),
        "consumed_approval": _binding(consumed_approval, "consumed_approval"),
        "output": {"path": str(Path(output_dir).resolve())},
        "checkpoint_consumed": False,
        "infers_past_checkpoint_updates": False,
        "condition_model_visible": False,
    }


def _verify_anchor(root: Path, anchor: str) -> None:
    if SHA1.fullmatch(anchor) is None:
        raise ValueError("source anchor is invalid")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()
    ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", anchor, head], check=False)
    if ancestor.returncode != 0:
        raise ValueError("source anchor is not an ancestor of HEAD")


def _verify_anchor_blobs(root: Path, anchor: str, closure: Mapping[str, str]) -> None:
    """protocol 파일과 별개로 source closure가 anchor의 등록 blob과 같은지 확인한다."""
    for relative, digest in closure.items():
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{anchor}:{relative}"],
            check=False, capture_output=True,
        )
        if result.returncode != 0 or hashlib.sha256(result.stdout).hexdigest() != digest:
            raise ValueError(f"source closure blob differs from anchor: {relative}")


def _verify_hash_map(
    root: Path, value: object, name: str, *, normalize_lf: bool = False,
    allow_isaac_sdk_alias: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    root = root.resolve(strict=True)
    result = {}
    for relative, digest in value.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(digest, str)
                or SHA256.fullmatch(digest) is None):
            raise ValueError(f"{name} entry is invalid")
        path = (root / relative).resolve()
        allowed_root = root
        if allow_isaac_sdk_alias and Path(relative).parts[0] == "_isaac_sim":
            # Isaac Lab 설치의 고정 SDK alias만 허용하고 그 안의 추가 이탈은 거부한다.
            allowed_root = (root / "_isaac_sim").resolve(strict=True)
            if not allowed_root.is_dir():
                raise ValueError("Isaac SDK alias is not a directory")
        if not path.is_relative_to(allowed_root) or not path.is_file():
            raise ValueError(f"{name} path changed: {relative}")
        actual = _sha_lf(path) if normalize_lf else _sha(path)
        if actual != digest:
            raise ValueError(f"{name} path changed: {relative}")
        result[relative] = digest
    return result


def _verify_approval(value: object, request_id: str, effect_key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"request", "approval", "consumption"}:
        raise ValueError("consumed approval refs differ")
    result = {name: _file_ref(ref, f"consumed approval {name}") for name, ref in value.items()}
    documents = {name: _read(Path(ref["path"])) for name, ref in result.items()}
    for name, document in documents.items():
        if document.get("request_id") != request_id or document.get("effect_key") != effect_key:
            raise ValueError(f"{name} approval lineage differs")
    return result


def _verify_lease(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"ref", "lease_id", "owner_pid"}:
        raise ValueError("outer lease fields differ")
    if (not isinstance(value["lease_id"], str) or LEASE_ID.fullmatch(value["lease_id"]) is None
            or type(value["owner_pid"]) is not int or value["owner_pid"] <= 0):
        raise ValueError("outer lease identity is invalid")
    ref = _file_ref(value["ref"], "outer lease")
    document = _read(Path(ref["path"]))
    lineage = importlib.import_module("lineage_source_batch_v1")
    if (document.get("owner") != lineage.OWNER
            or document.get("leaseId") != value["lease_id"]
            or document.get("state") != "running"
            or document.get("completedUtc") is not None
            or document.get("childPid") != value["owner_pid"]
            or type(document.get("wrapperPid")) is not int):
        raise ValueError("immutable outer lease admission snapshot differs")
    return {"ref": ref, "lease_id": value["lease_id"], "owner_pid": value["owner_pid"],
            "owner": lineage.OWNER, "wrapper_pid": document["wrapperPid"]}


def verify_manifest(
    path: Path,
    *,
    repo_root: Path = ROOT,
    isaaclab_root: Path,
    expected_protocol_path: Path | None = None,
) -> dict[str, Any]:
    """manifest와 등록 protocol/source/runtime/approval/lease를 CPU에서 다시 결합한다."""
    path, root, isaaclab = Path(path).resolve(), Path(repo_root).resolve(), Path(isaaclab_root).resolve()
    manifest = _read(path)
    if set(manifest) != MANIFEST_FIELDS or manifest.get("schema") != SCHEMA:
        raise ValueError("diagnostic child manifest fields differ")
    if (not isinstance(manifest["request_id"], str) or not IDENTIFIER.fullmatch(manifest["request_id"])
            or not isinstance(manifest["effect_key"], str) or not IDENTIFIER.fullmatch(manifest["effect_key"])):
        raise ValueError("request/effect identity is invalid")
    protocol_ref = manifest["protocol"]
    if not isinstance(protocol_ref, Mapping) or set(protocol_ref) != {"path", "sha256"}:
        raise ValueError("protocol reference differs")
    protocol_path = Path(protocol_ref["path"]).resolve()
    if expected_protocol_path is not None and protocol_path != Path(expected_protocol_path).resolve():
        raise ValueError("manifest protocol is not the registered parent protocol")
    if not protocol_path.is_file() or _sha(protocol_path) != protocol_ref["sha256"]:
        raise ValueError("registered parent protocol is absent or changed")
    protocol = _protocol(protocol_path)
    operation = protocol["allowed_operations"].get(manifest["operation"])
    if (operation is None or manifest["component"] != operation["component"]
            or manifest["probe"] != operation["probe"] or manifest["condition"] not in operation["conditions"]
            or manifest["updates"] != operation["updates"]):
        raise ValueError("manifest operation binding differs")
    if manifest["component"] not in COMPONENTS or manifest["condition"] not in CONDITIONS:
        raise ValueError("component or condition is unsupported")
    if (manifest["expected_learning_epochs"] != EXPECTED_EPOCHS
            or manifest["expected_mini_batches"] != EXPECTED_MINI_BATCHES
            or manifest["checkpoint_consumed"] is not False
            or manifest["infers_past_checkpoint_updates"] is not False
            or manifest["condition_model_visible"] is not False):
        raise ValueError("fixed observation semantics differ")
    if (type(manifest["seed"]) is not int or manifest["seed"] < 0
            or type(manifest["num_envs"]) is not int or manifest["num_envs"] <= 0
            or not isinstance(manifest["device"], str) or not manifest["device"]):
        raise ValueError("runtime scalar binding differs")
    if (manifest["source_anchor"] != protocol["source_anchor"]
            or manifest["source_closure"] != protocol["source_closure"]
            or manifest["runtime_pins"] != protocol["runtime_pins"]):
        raise ValueError("manifest protocol commitments differ")
    _verify_anchor(root, manifest["source_anchor"])
    closure = _verify_hash_map(
        root, manifest["source_closure"], "source closure", normalize_lf=True,
    )
    runtime_pins = manifest["runtime_pins"]
    if (not isinstance(runtime_pins, Mapping)
            or set(runtime_pins) != {"logical_root", "sha256"}
            or runtime_pins.get("logical_root") != "isaaclab"):
        raise ValueError("runtime pins logical root differs")
    _verify_anchor_blobs(root, manifest["source_anchor"], closure)
    runtime = _verify_hash_map(
        isaaclab, runtime_pins["sha256"], "runtime pins", allow_isaac_sdk_alias=True,
    )
    checkpoint = _checkpoint(manifest["checkpoint"])
    approval = _verify_approval(manifest["consumed_approval"], manifest["request_id"], manifest["effect_key"])
    lease = _verify_lease(manifest["outer_lease"])
    for name in ("case", "source", "reference"):
        _binding(manifest[name], name)
    output_value = manifest["output"]
    if not isinstance(output_value, Mapping) or set(output_value) != {"path"}:
        raise ValueError("output binding differs")
    output = Path(output_value["path"]).resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("child output must be fresh with an existing parent")
    return {
        "manifest_path": path,
        "manifest_sha256": _sha(path),
        "manifest": deepcopy(manifest),
        "protocol_path": protocol_path,
        "protocol": protocol,
        "repo_root": root,
        "isaaclab_root": isaaclab,
        "output": output,
        "source_closure": closure,
        "runtime_pins": runtime,
        "checkpoint": checkpoint,
        "approval": approval,
        "outer_lease": lease,
    }


def _live_lease(validated: Mapping[str, Any]) -> None:
    expected = validated["outer_lease"]
    lineage = importlib.import_module("lineage_source_batch_v1")
    current = lineage.lease_context()
    if (current.get("owner") != lineage.OWNER
            or current.get("leaseId") != expected["lease_id"]
            or current.get("state") != "running"
            or current.get("completedUtc") is not None
            or current.get("childPid") != expected["owner_pid"]
            or current.get("wrapperPid") != expected["wrapper_pid"]):
        raise ValueError("live outer lease differs from the immutable admission snapshot")


def _observer_manifest(validated: Mapping[str, Any]) -> dict[str, Any]:
    manifest, protocol = validated["manifest"], validated["protocol"]
    return {
        "schema": "optimizer_observability_manifest_v1",
        "arm": manifest["condition"],
        "seed": manifest["seed"],
        "task": protocol["task"],
        "num_envs": manifest["num_envs"],
        "max_iterations": OPTIMIZER_UPDATES,
        "configured_learning_epochs": EXPECTED_EPOCHS,
        "configured_mini_batches": EXPECTED_MINI_BATCHES,
        "expected_optimizer_steps_per_update": EXPECTED_EPOCHS * EXPECTED_MINI_BATCHES,
        "expected_minibatches_per_update": EXPECTED_EPOCHS * EXPECTED_MINI_BATCHES,
        "runtime_updates_to_observe": OPTIMIZER_UPDATES,
        "source_commit_sha": manifest["source_anchor"],
        "protocol_path": str(validated["protocol_path"]),
        "protocol_sha256_lf": _sha_lf(validated["protocol_path"]),
        "source_sha256_lf": manifest["source_closure"],
        "gradient_invariant": {
            "schema": "ppo_surrogate_gradient_v1", "atol": 1.0e-6, "rtol": 1.0e-5,
            "scale": "batch_size", "clip_param": 0.2,
        },
        "inspection_primitives": ["optimizer_step_count", "minibatch_count", "ppo_surrogate_gradient"],
    }


@contextmanager
def _optimizer_hooks(validated: Mapping[str, Any], sink):
    manifest = validated["manifest"]
    observer_manifest = _observer_manifest(validated)
    gradient = importlib.import_module("ppo_gradient_invariant_v2")
    mutation = importlib.import_module("ppo_ratio_mutation_v1")
    v5_training = importlib.import_module("ppo_gradient_training_v5")
    ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
    runner_class = importlib.import_module("rsl_rl.runners").OnPolicyRunner
    original_init, original_learn = ppo_class.__init__, runner_class.learn
    state: dict[str, Any] = {
        "observer": None, "restore_fault": None, "v5_finish": None, "v5_cleanup": None,
        "v5_evidence": None, "learn_completed": False,
    }

    def init(this, *args, **kwargs):
        original_init(this, *args, **kwargs)
        if state["observer"] is not None:
            raise ValueError("one PPO instance is required")
        condition = manifest["condition"]
        if condition == "R1":
            this.num_learning_epochs = 1
            restore_fault, _ = mutation.install(this, "NONE")
            state["restore_fault"] = restore_fault
            state["observer"] = gradient.GradientObserver(this, observer_manifest, sink).install()
        else:
            fixed = {**observer_manifest, "arm": (
                "INVERSE_RATIO" if condition == "inverse_ratio" else "NONE"),
                "observer_enabled": True}
            finish, cleanup = v5_training.attach(this, fixed, sink)
            state["v5_finish"], state["v5_cleanup"] = finish, cleanup

    def learn(this, *args, **kwargs):
        iterations = kwargs.get("num_learning_iterations", args[0] if args else None)
        if state["learn_completed"] or iterations != OPTIMIZER_UPDATES:
            raise ValueError("registered three-update learn call is required")
        result = original_learn(this, *args, **kwargs)
        if state["v5_finish"] is not None:
            state["v5_evidence"] = state["v5_finish"]()
        state["learn_completed"] = True
        return result

    ppo_class.__init__, runner_class.learn = init, learn
    try:
        yield state, observer_manifest
    finally:
        ppo_class.__init__, runner_class.learn = original_init, original_learn
        if state["observer"] is not None:
            state["observer"].remove()
        if state["restore_fault"] is not None:
            state["restore_fault"]()
        if state["v5_cleanup"] is not None:
            state["v5_cleanup"]()


def _optimizer_run(validated: Mapping[str, Any]) -> dict[str, Any]:
    guard = importlib.import_module("isaac_training_guard_v1")
    gradient = importlib.import_module("ppo_gradient_invariant_v2")
    output, manifest = validated["output"], validated["manifest"]
    output.mkdir(parents=False, exist_ok=False)
    records_path = output / guard.RECORDS
    upstream = validated["isaaclab_root"] / "scripts/reinforcement_learning/rsl_rl/train.py"
    original_argv, original_path = sys.argv[:], sys.path[:]
    original_hydra = os.environ.get("HYDRA_FULL_ERROR")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        def sink(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        with _optimizer_hooks(validated, sink) as (state, observer_manifest):
            def before_close(_app):
                if not state["learn_completed"]:
                    raise ValueError("three-update learning did not complete")
                stream.flush()
                os.fsync(stream.fileno())
                rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
                inspection = gradient.inspect_records(rows, observer_manifest)
                if len(rows) != OPTIMIZER_UPDATES or inspection.get("records") != OPTIMIZER_UPDATES:
                    raise ValueError("count/gradient records are incomplete")
                evidence = {
                    "schema": "diagnostic_optimizer_observation_v1",
                    "status": "complete",
                    "manifest_sha256": validated["manifest_sha256"],
                    "updates": OPTIMIZER_UPDATES,
                    "inspection": inspection,
                    "v5_runtime_evidence": state["v5_evidence"],
                    "data_binding": {
                        "request_id": manifest["request_id"], "effect_key": manifest["effect_key"],
                        "case": manifest["case"], "source": manifest["source"],
                        "checkpoint": manifest["checkpoint"], "reference": manifest["reference"],
                        "checkpoint_consumed": False, "infers_past_checkpoint_updates": False,
                        "condition_model_visible": False,
                    },
                }
                guard.durable_new(output / guard.EVIDENCE, evidence)
                guard.durable_new(output / guard.CLOSE_READY, {
                    "schema": guard.CLOSE_SCHEMA,
                    "manifest_sha256": validated["manifest_sha256"],
                    "updates": OPTIMIZER_UPDATES,
                    "evidence_sha256_lf": _sha_lf(output / guard.EVIDENCE),
                    "records_sha256_lf": _sha_lf(records_path),
                })

            try:
                sys.path.insert(0, str(upstream.parent))
                sys.argv = [str(upstream), "--task", validated["protocol"]["task"],
                            "--num_envs", str(manifest["num_envs"]),
                            "--max_iterations", str(OPTIMIZER_UPDATES), "--seed", str(manifest["seed"]),
                            "--device", manifest["device"], "--headless",
                            "--run_name", f"diagnostic_{manifest['request_id'][:24]}"]
                guard.run_upstream(upstream, output, before_close)
            finally:
                sys.argv, sys.path[:] = original_argv, original_path
                if original_hydra is None:
                    os.environ.pop("HYDRA_FULL_ERROR", None)
                else:
                    os.environ["HYDRA_FULL_ERROR"] = original_hydra
    guard.validate_completion(output, validated["manifest_sha256"])
    return _read(output / guard.EVIDENCE)


def _actuator_run(validated: Mapping[str, Any]) -> dict[str, Any]:
    runtime = importlib.import_module("actuator_envelope_runtime")
    parent = importlib.import_module("run_actuator_envelope_runtime")
    base_ref = validated["protocol"]["actuator_base_protocol"]
    base_path = (validated["repo_root"] / base_ref["path"]).resolve()
    if not base_path.is_file() or _sha(base_path) != base_ref["sha256"]:
        raise ValueError("actuator base protocol changed")
    base_protocol = _read(base_path)
    source_receipt = parent.validate_protocol(
        base_protocol, repo_root=validated["repo_root"], isaaclab_root=validated["isaaclab_root"],
    )
    condition = validated["manifest"]["condition"]
    sdk_validated = {
        "protocol": base_protocol,
        "manifest": {"condition": condition},
        "isaaclab_root": validated["isaaclab_root"],
    }
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]  # 등록 SDK에서만 실행
    app = AppLauncher(headless=True, fast_shutdown=True).app
    lifecycle: dict[str, Any] = {}
    output = validated["output"]
    output.mkdir(parents=False, exist_ok=False)
    try:
        result, _env = runtime.run_sdk(sdk_validated, lifecycle)
        runtime._close_environment(lifecycle)
        receipt = {
            "schema": "diagnostic_actuator_mapping_v1", "status": "complete",
            "manifest_sha256": validated["manifest_sha256"],
            "source_receipt": source_receipt,
            "observation": result,
            "data_binding": {
                "request_id": validated["manifest"]["request_id"],
                "effect_key": validated["manifest"]["effect_key"],
                "case": validated["manifest"]["case"], "source": validated["manifest"]["source"],
                "checkpoint": validated["manifest"]["checkpoint"],
                "reference": validated["manifest"]["reference"],
                "checkpoint_consumed": False, "condition_model_visible": False,
            },
        }
        _write_new(output / "observation.json", receipt)
        _write_new(output / "close_ready.json", {
            "schema": "diagnostic_actuator_close_ready_v1",
            "manifest_sha256": validated["manifest_sha256"],
            "observation_sha256": _sha(output / "observation.json"),
            "environment_closed": True,
            "app_close_pending": True,
        })
        return receipt
    except BaseException as error:
        try:
            _write_new(output / "failure.json", {
                "schema": "diagnostic_observer_failure_v1", "status": "invalid",
                "type": type(error).__name__, "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                "manifest_sha256": validated["manifest_sha256"], "sealed_before_app_close": True,
            })
        except FileExistsError:
            pass
        raise
    finally:
        try:
            runtime._close_environment(lifecycle)
        finally:
            app.close()


def child_main(manifest_path: Path) -> int:
    """검증된 manifest의 한 component를 실행하고 성공 시에만 0을 반환한다."""
    manifest_path = Path(manifest_path).resolve()
    validated = verify_manifest(
        manifest_path,
        repo_root=ROOT,
        isaaclab_root=Path.cwd(),
        expected_protocol_path=REGISTERED_PROTOCOL,
    )
    output = validated["output"]
    try:
        _live_lease(validated)
        result = (_actuator_run(validated) if validated["manifest"]["component"] == "actuator_mapping"
                  else _optimizer_run(validated))
        if result.get("status") != "complete":
            raise RuntimeError("component did not produce a complete observation")
        return 0
    except BaseException as error:
        try:
            if not output.exists():
                output.mkdir(parents=False, exist_ok=False)
            _write_new(output / "failure.json", {
                "schema": "diagnostic_observer_failure_v1", "status": "invalid",
                "type": type(error).__name__, "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                "manifest_sha256": validated["manifest_sha256"],
            })
        except FileExistsError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    return child_main(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
