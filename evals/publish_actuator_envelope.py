"""검증된 actuator envelope 측정을 공개 수치 재생 묶음으로 내보낸다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import actuator_envelope_runtime as B  # noqa: E402
import run_actuator_envelope_runtime as R  # noqa: E402


RESULTS = ROOT / "evals/results"
PRIVATE_ROOT = ROOT / "bench/private/actuator_envelope_runtime"
SCHEMA = "actuator_envelope_public_v1"
REPLAY_SCOPE = "numeric_replay_only_not_execution_identity_attestation"
PUBLIC_FILES = {"result.json"}
PUBLISHER_SOURCES = {"evals/publish_actuator_envelope.py"} | R.PROJECT_SOURCE_FILES
FORBIDDEN_KEYS = {
    "pid", "process_id", "process_identity", "owned_process_identities", "processes",
    "process_list", "stdout", "stderr", "path", "isaaclab_root", "gpu_before", "gpu_after",
}
CONDITION_KEYS = {
    "condition", "development_expected_curve_matches", "development_expected_assessment",
    "curve_matches_expected", "curve_assessment", "protocol_status", "expectation_met",
    "helper_result", "state_before", "state_after", "state_restoration", "sdk_versions",
    "actuator_binding", "source_sha256", "private_input_sha256", "process_time",
    "owned_execution",
}
STATE_KEYS = {"schema", "buffers", "common_step_counter", "sim_step_counter"}
STATE_BUFFER_KEYS = {"shape", "values"}
EXECUTION_KEYS = {
    "job_assigned_before_release", "natural_exit", "timed_out", "error_type", "exit_code",
    "cleanup_verified", "wall_s", "wall_scope",
}
PARENT_CONDITION_KEYS = {
    "condition", "status", "validation_reasons", "body_sha256", "measurement_status",
    "curve_assessment", "protocol_status", "execution",
}
PUBLIC_RUNTIME_CONFIG = {"effort_limit": 23.5, "saturation_effort": 23.5, "velocity_limit": 30.0}
PARENT_WALL_SCOPE = "parent_launch_to_exit_and_owned_job_empty_verification"


def _read(path: Path) -> dict[str, Any]:
    return R.S._read_json(path)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publisher_sources() -> dict[str, str]:
    result = {}
    for relative in sorted(PUBLISHER_SOURCES):
        raw = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        result[relative] = hashlib.sha256(raw).hexdigest()
    return result


def _public_value(value: object, key: str | None = None) -> None:
    if key is not None and key.lower() in FORBIDDEN_KEYS:
        raise ValueError(f"private execution field is not publishable: {key}")
    if isinstance(value, dict):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ValueError("public object keys must be strings")
            _public_value(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _public_value(child)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
                or "bench/private" in normalized):
            raise ValueError("absolute or private path leaked into public output")


def _public_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": value["schema"],
        "buffers": {
            name: {"shape": deepcopy(value["buffers"][name]["shape"]),
                   "values": deepcopy(value["buffers"][name]["values"])}
            for name in B.STATE_BUFFER_FIELDS
        },
        "common_step_counter": value["common_step_counter"],
        "sim_step_counter": value["sim_step_counter"],
    }


def _validate_public_state(value: object) -> None:
    if (not isinstance(value, Mapping) or set(value) != STATE_KEYS
            or value.get("schema") != "actuator_envelope_state_v1"):
        raise ValueError("public actuator state schema differs")
    buffers = value.get("buffers")
    if not isinstance(buffers, Mapping) or set(buffers) != set(B.STATE_BUFFER_FIELDS):
        raise ValueError("public actuator state buffer set differs")
    for name in B.STATE_BUFFER_FIELDS:
        row = buffers[name]
        values = row.get("values") if isinstance(row, Mapping) else None
        if (not isinstance(row, Mapping) or set(row) != STATE_BUFFER_KEYS
                or row.get("shape") != [1, 12] or not isinstance(values, list) or len(values) != 12
                or any(type(item) not in (int, float) or isinstance(item, bool)
                       or not math.isfinite(float(item)) for item in values)):
            raise ValueError("public actuator state buffer is invalid")
    for key in ("common_step_counter", "sim_step_counter"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError("public actuator state counter is invalid")


def _public_restoration(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    checks = {name: before["buffers"][name] == after["buffers"][name]
              for name in B.STATE_BUFFER_FIELDS}
    checks["common_step_counter"] = before["common_step_counter"] == after["common_step_counter"]
    checks["sim_step_counter"] = before["sim_step_counter"] == after["sim_step_counter"]
    return {"checks": checks, "all_restored": all(checks.values())}


def _owned_execution(value: object) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or type(value.get("exit_code")) is not int
            or value.get("exit_code") != 0 or R._owned_execution_reasons(value)):
        raise ValueError("owned execution did not finish naturally and cleanly")
    result = {key: deepcopy(value.get(key)) for key in EXECUTION_KEYS}
    if set(result) != EXECUTION_KEYS:
        raise AssertionError("owned execution public schema differs")
    wall = result["wall_s"]
    if (type(wall) not in (int, float) or isinstance(wall, bool)
            or not math.isfinite(float(wall)) or wall < 0):
        raise ValueError("owned execution wall time is invalid")
    if result["wall_scope"] != PARENT_WALL_SCOPE:
        raise ValueError("owned execution wall scope is invalid")
    return result


def _condition_public(condition: str, body: Mapping[str, Any], execution: Mapping[str, Any],
                      manifest_path: Path, body_path: Path) -> dict[str, Any]:
    helper = body["helper_result"]
    config = body["actuator_binding"]["runtime_config"]
    curve_reasons = B.validate_curve(helper, config)
    if helper.get("status") != "valid" or curve_reasons:
        raise ValueError(f"{condition} helper measurement is invalid")
    actual_match = helper["curve_matches_expected"]
    actual_assessment = "normal" if actual_match else "curve_difference"
    expected_match = B.EXPECTED_MATCH[condition]
    protocol_status = "expected" if actual_match is expected_match else "unexpected"
    before = _public_state(body["state_before"])
    after = _public_state(body["state_after"])
    restoration = _public_restoration(before, after)
    if restoration["all_restored"] is not True:
        raise ValueError(f"{condition} actuator state was not restored")
    binding = body["actuator_binding"]
    source_receipt = body["source_receipt"]
    result = {
        "condition": condition,
        "development_expected_curve_matches": expected_match,
        "development_expected_assessment": "normal" if expected_match else "curve_difference",
        "curve_matches_expected": actual_match,
        "curve_assessment": actual_assessment,
        "protocol_status": protocol_status,
        "expectation_met": actual_match is expected_match,
        "helper_result": deepcopy(helper),
        "state_before": before,
        "state_after": after,
        "state_restoration": restoration,
        "sdk_versions": deepcopy(body["sdk_versions"]),
        "actuator_binding": {
            "class_name": binding["class_name"], "class_module": binding["class_module"],
            "joint_names": deepcopy(binding["joint_names"]), "joint_count": binding["joint_count"],
            "joint_velocity_shape": deepcopy(binding["joint_velocity_shape"]),
            "device": binding["device"], "runtime_config": deepcopy(config),
        },
        "source_sha256": {
            "project": deepcopy(source_receipt["project_source_files"]),
            "isaaclab": deepcopy(source_receipt["isaaclab_source_files"]),
            "rsl_rl": deepcopy(source_receipt["rsl_rl_source_files"]),
            "actuator_class": binding["class_source"]["sha256"],
            "clip_effort_file": binding["clip_effort_source"]["sha256"],
            "clip_effort_code": binding["clip_effort_source"]["code_sha256"],
            "asset": binding["asset_source"]["sha256"],
        },
        "private_input_sha256": {
            "manifest": _sha(manifest_path), "body": _sha(body_path),
            "stdout_log": _sha(manifest_path.parent / "stdout.log"),
            "stderr_log": _sha(manifest_path.parent / "stderr.log"),
        },
        "process_time": {key: body[key] for key in ("started_at", "finished_at", "wall_elapsed_s")},
        "owned_execution": _owned_execution(execution),
    }
    if set(result) != CONDITION_KEYS:
        raise AssertionError("public condition schema differs")
    return result


def export(tag: str, isaaclab: Path) -> dict[str, Any]:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) is None:
        raise ValueError("invalid tag")
    private = PRIVATE_ROOT / tag
    parent_path = private / "parent_receipt.json"
    request_path = private / "request.json"
    consumption_path = private / "consumed.json"
    if not request_path.is_file() or not consumption_path.is_file():
        raise FileNotFoundError("private request lineage is missing")
    parent = _read(parent_path)
    consumption = _read(consumption_path)
    current_head = R.S._git_head(ROOT)
    parent_keys = {
        "schema", "status", "tag", "project_commit_sha", "protocol_sha256", "request_id",
        "consumption", "request_consumed", "launch_count", "conditions", "gpu_before",
        "gpu_after", "cause_or_recovery_evidence",
    }
    if (set(parent) != parent_keys or parent.get("schema") != "actuator_envelope_runtime_parent_v1"
            or parent.get("status") != "complete" or parent.get("tag") != tag
            or parent.get("project_commit_sha") != current_head
            or parent.get("request_consumed") is not True
            or parent.get("launch_count") != len(B.CONDITIONS)
            or parent.get("cause_or_recovery_evidence") is not False):
        raise ValueError("private parent receipt is incomplete or bound to another run")
    rows = parent.get("conditions")
    if (not isinstance(rows, list) or len(rows) != len(B.CONDITIONS)
            or [row.get("condition") if isinstance(row, Mapping) else None for row in rows]
            != list(B.CONDITIONS)):
        raise ValueError("private parent condition set differs")
    public_conditions = {}
    protocol_sha = None
    for condition, row in zip(B.CONDITIONS, rows):
        if (not isinstance(row, Mapping) or set(row) != PARENT_CONDITION_KEYS
                or row.get("status") != "pass"
                or row.get("validation_reasons") != [] or row.get("measurement_status") != "valid"):
            raise ValueError(f"{condition} parent validation did not pass")
        manifest_path = private / condition / "manifest.json"
        body_path = private / condition / "body.json"
        if not manifest_path.is_file() or not body_path.is_file():
            raise FileNotFoundError(f"{condition} private evidence is missing")
        log_reasons = R._execution_log_reasons(manifest_path.parent)
        if log_reasons:
            raise ValueError(f"{condition} execution log validation failed")
        validated = R.validate_manifest(
            manifest_path.resolve(), repo_root=ROOT, isaaclab_root=isaaclab.resolve(),
            completed_output=True,
        )
        if validated["manifest"]["condition"] != condition:
            raise ValueError("condition manifest is cross-bound")
        if (parent["request_id"] != validated["request_lineage"]["request_id"]
                or parent["consumption"] != consumption
                or validated["paths"]["request_path"].resolve() != request_path.resolve()
                or validated["paths"]["consumption_path"].resolve() != consumption_path.resolve()):
            raise ValueError("parent request-consumption lineage differs")
        body = _read(body_path)
        execution = row.get("execution")
        reasons = R.verify_body(body, validated, execution)
        if reasons:
            raise ValueError(f"{condition} body validation failed")
        body_sha = _sha(body_path)
        if row.get("body_sha256") != body_sha:
            raise ValueError(f"{condition} body SHA differs from parent receipt")
        shutdown_reasons = R._shutdown_handoff_reasons(manifest_path.parent, body, body_sha)
        if shutdown_reasons:
            raise ValueError(f"{condition} shutdown handoff validation failed")
        if (row.get("curve_assessment") != body.get("curve_assessment")
                or row.get("protocol_status") != body.get("protocol_status")):
            raise ValueError(f"{condition} parent summary differs from body")
        condition_protocol_sha = validated["protocol_sha256"]
        if protocol_sha is None:
            protocol_sha = condition_protocol_sha
        if condition_protocol_sha != protocol_sha or parent["protocol_sha256"] != protocol_sha:
            raise ValueError("conditions do not share the parent protocol")
        public_conditions[condition] = _condition_public(
            condition, body, execution, manifest_path, body_path,
        )
    result = {
        "schema": SCHEMA, "replay_scope": REPLAY_SCOPE, "tag": tag,
        "execution_source_commit_sha": current_head, "protocol_sha256": protocol_sha,
        "parent_receipt_sha256": _sha(parent_path),
        "private_lineage_sha256": {
            "request": _sha(request_path), "consumption": _sha(consumption_path),
        },
        "publisher_sources": _publisher_sources(), "conditions": public_conditions,
    }
    _validate_result(result)
    _public_value(result)
    encoded = _json_bytes(result)
    manifest = {
        "schema": SCHEMA, "replay_scope": REPLAY_SCOPE,
        "conditions": list(B.CONDITIONS), "publisher_sources": result["publisher_sources"],
        "files": {"result.json": hashlib.sha256(encoded).hexdigest()},
    }
    _public_value(manifest)
    output = RESULTS / tag
    if output.exists():
        raise FileExistsError("public result folder already exists")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "result.json").open("xb") as stream:
        stream.write(encoded)
    with (output / "artifact_manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest))
    return {"status": "exported", "folder": f"evals/results/{tag}",
            "replay_scope": REPLAY_SCOPE, "condition_count": len(public_conditions)}


def _validate_result(result: object) -> dict[str, dict[str, Any]]:
    keys = {
        "schema", "replay_scope", "tag", "execution_source_commit_sha", "protocol_sha256",
        "parent_receipt_sha256", "private_lineage_sha256", "publisher_sources", "conditions",
    }
    if (not isinstance(result, Mapping) or set(result) != keys or result.get("schema") != SCHEMA
            or result.get("replay_scope") != REPLAY_SCOPE
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", result.get("tag", "")) is None
            or any(re.fullmatch(r"[0-9a-f]{64}", result.get(key, "")) is None
                   for key in ("protocol_sha256", "parent_receipt_sha256"))
            or re.fullmatch(r"[0-9a-f]{40}", result.get("execution_source_commit_sha", "")) is None):
        raise ValueError("public result schema differs")
    lineage = result.get("private_lineage_sha256")
    if (not isinstance(lineage, Mapping) or set(lineage) != {"request", "consumption"}
            or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in lineage.values())):
        raise ValueError("public request lineage SHA schema differs")
    conditions = result.get("conditions")
    if not isinstance(conditions, dict) or list(conditions) != list(B.CONDITIONS):
        raise ValueError("public condition schema differs")
    return conditions


def _replay_condition(condition: str, row: object) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != CONDITION_KEYS or row.get("condition") != condition:
        raise ValueError("public condition row schema differs")
    binding = row.get("actuator_binding")
    binding_keys = {
        "class_name", "class_module", "joint_names", "joint_count", "joint_velocity_shape",
        "device", "runtime_config",
    }
    if (not isinstance(binding, Mapping) or set(binding) != binding_keys
            or binding.get("class_name") != "DCMotor"
            or binding.get("class_module") != "isaaclab.actuators.actuator_pd"
            or binding.get("joint_count") != 12 or binding.get("joint_velocity_shape") != [1, 12]
            or not isinstance(binding.get("joint_names"), list)
            or any(type(name) is not str for name in binding["joint_names"])
            or set(binding["joint_names"]) != B.GO2_JOINT_NAMES
            or binding.get("device") != R.FIXED["device"]
            or not isinstance(binding.get("runtime_config"), Mapping)
            or set(binding["runtime_config"]) != {"effort_limit", "saturation_effort", "velocity_limit"}
            or any(type(value) not in (int, float) or isinstance(value, bool)
                   or not math.isfinite(float(value)) for value in binding["runtime_config"].values())
            or binding["runtime_config"] != PUBLIC_RUNTIME_CONFIG):
        raise ValueError("public actuator binding differs")
    sdk = row.get("sdk_versions")
    if (not isinstance(sdk, Mapping)
            or set(sdk) != {"isaaclab_commit", "isaacsim", "isaacsim_raw_components", "runtime_binding"}
            or sdk.get("isaaclab_commit") != B.ISAACLAB_COMMIT
            or sdk.get("isaacsim") != "4.5.0"
            or not B.validate_isaacsim_raw_components(sdk.get("isaacsim_raw_components"), "4.5.0")
            or sdk.get("runtime_binding") != R.EXPECTED_RUNTIME_BINDING):
        raise ValueError("public SDK version binding differs")
    source = row.get("source_sha256")
    source_keys = {
        "project", "isaaclab", "rsl_rl", "actuator_class", "clip_effort_file",
        "clip_effort_code", "asset",
    }
    if not isinstance(source, Mapping) or set(source) != source_keys:
        raise ValueError("public source SHA schema differs")
    expected_source_names = {
        "project": R.PROJECT_SOURCE_FILES,
        "isaaclab": R.ISAACLAB_SOURCE_FILES,
        "rsl_rl": R.RSL_SOURCE_FILES,
    }
    for key in ("project", "isaaclab", "rsl_rl"):
        mapping = source[key]
        if (not isinstance(mapping, Mapping) or set(mapping) != expected_source_names[key]
                or any(not isinstance(name, str) or PurePosixPath(name).is_absolute()
                       or PureWindowsPath(name).is_absolute()
                       or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                       for name, digest in mapping.items())):
            raise ValueError("public source SHA map differs")
    if any(re.fullmatch(r"[0-9a-f]{64}", source[key]) is None
           for key in source_keys - {"project", "isaaclab", "rsl_rl"}):
        raise ValueError("public source SHA differs")
    private_sha = row.get("private_input_sha256")
    if (not isinstance(private_sha, Mapping)
            or set(private_sha) != {"manifest", "body", "stdout_log", "stderr_log"}
            or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None
                   for digest in private_sha.values())):
        raise ValueError("public private-input SHA summary differs")
    process_time = row.get("process_time")
    if (not isinstance(process_time, Mapping)
            or set(process_time) != {"started_at", "finished_at", "wall_elapsed_s"}
            or any(not isinstance(process_time.get(key), str) or not process_time[key]
                   for key in ("started_at", "finished_at"))
            or type(process_time.get("wall_elapsed_s")) not in (int, float)
            or isinstance(process_time.get("wall_elapsed_s"), bool)
            or not math.isfinite(float(process_time["wall_elapsed_s"]))
            or process_time["wall_elapsed_s"] < 0):
        raise ValueError("public process time summary differs")
    helper = row.get("helper_result")
    reasons = B.validate_curve(helper, binding["runtime_config"])
    if not isinstance(helper, Mapping) or helper.get("status") != "valid" or reasons:
        raise ValueError("public curve does not validate")
    actual_match = helper["curve_matches_expected"]
    expected_match = B.EXPECTED_MATCH[condition]
    assessment = "normal" if actual_match else "curve_difference"
    protocol_status = "expected" if actual_match is expected_match else "unexpected"
    if (row.get("development_expected_curve_matches") is not expected_match
            or row.get("development_expected_assessment")
            != ("normal" if expected_match else "curve_difference")
            or row.get("curve_matches_expected") is not actual_match
            or row.get("curve_assessment") != assessment
            or row.get("protocol_status") != protocol_status
            or row.get("expectation_met") is not (actual_match is expected_match)):
        raise ValueError("public curve verdict does not replay")
    before, after = row.get("state_before"), row.get("state_after")
    _validate_public_state(before)
    _validate_public_state(after)
    restoration = _public_restoration(before, after)
    if row.get("state_restoration") != restoration or restoration["all_restored"] is not True:
        raise ValueError("public actuator restoration does not replay")
    execution = row.get("owned_execution")
    if (not isinstance(execution, Mapping) or set(execution) != EXECUTION_KEYS
            or type(execution.get("exit_code")) is not int or execution.get("exit_code") != 0
            or R._owned_execution_reasons(execution)
            or execution.get("wall_scope") != PARENT_WALL_SCOPE):
        raise ValueError("public owned execution summary is invalid")
    wall = execution.get("wall_s")
    if (type(wall) not in (int, float) or isinstance(wall, bool)
            or not math.isfinite(float(wall)) or wall < 0):
        raise ValueError("public owned execution time is invalid")
    return {"curve_assessment": assessment, "protocol_status": protocol_status,
            "expectation_met": actual_match is expected_match}


def replay(folder: Path) -> dict[str, Any]:
    if folder.is_symlink():
        raise ValueError("public result folder is missing or unsafe")
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError("public result folder is missing or unsafe")
    manifest_path = folder / "artifact_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("public artifact manifest path is unsafe")
    manifest = _read(manifest_path)
    expected_manifest_keys = {"schema", "replay_scope", "conditions", "publisher_sources", "files"}
    if (set(manifest) != expected_manifest_keys or manifest.get("schema") != SCHEMA
            or manifest.get("replay_scope") != REPLAY_SCOPE
            or manifest.get("conditions") != list(B.CONDITIONS)
            or set(manifest.get("files", {})) != PUBLIC_FILES):
        raise ValueError("public artifact manifest differs")
    publisher_sources = manifest.get("publisher_sources")
    if (not isinstance(publisher_sources, dict) or set(publisher_sources) != PUBLISHER_SOURCES
            or publisher_sources != _publisher_sources()):
        raise ValueError("publisher source binding differs")
    entries = {path.relative_to(folder).as_posix(): path for path in folder.rglob("*")}
    expected_entries = PUBLIC_FILES | {"artifact_manifest.json"}
    if (set(entries) != expected_entries
            or any(path.is_symlink() or not path.is_file() for path in entries.values())):
        raise ValueError("public artifact file set differs")
    result_path = folder / "result.json"
    try:
        result_path.resolve(strict=True).relative_to(folder)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("public artifact path is unsafe") from error
    if result_path.is_symlink():
        raise ValueError("public artifact path is unsafe")
    if _sha(result_path) != manifest["files"]["result.json"]:
        raise ValueError("public artifact hash differs")
    result = _read(result_path)
    _public_value(result)
    conditions = _validate_result(result)
    if result.get("publisher_sources") != publisher_sources:
        raise ValueError("result and manifest source bindings differ")
    replayed = {condition: _replay_condition(condition, conditions[condition])
                for condition in B.CONDITIONS}
    return {"status": "valid", "replay_scope": REPLAY_SCOPE, "conditions": replayed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("tag")
    export_parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    result = export(args.tag, args.isaaclab) if args.command == "export" else replay(args.folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
