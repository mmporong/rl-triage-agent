from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import registered_mechanism_observer_v1 as O


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sdk(tmp_path: Path) -> Path:
    root = tmp_path / "IsaacLab"
    for relative in O.ISAAC_RUNTIME_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative}\n", encoding="utf-8")
    return root


def request(tmp_path: Path) -> dict:
    endpoints = {}
    for name, condition, arm in (("case", "F2", "INVERSE_RATIO"), ("reference", "NONE", "NONE")):
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes((name + " checkpoint").encode())
        endpoints[name] = {
            "case_id": name,
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
            "actuator_condition": condition,
            "optimizer_arm": arm,
        }
    value = {
        "schema": O.REQUEST_SCHEMA,
        "approval": {"request_id": "approved-request", "effect_key": "effect",
                     "registry_digest": "a" * 64},
        "lease": {"lease_id": "12345678-1234-1234-1234-123456789abc", "owner": "u1-test"},
        "probes": list(O.PROBES),
        "runtime_updates": 3,
        **endpoints,
    }
    value["request_id"] = O.request_id(value)
    return value


def test_real_production_entrypoints_and_functions_are_bound() -> None:
    entries = O.production_entrypoints(ROOT)
    assert set(entries) == {"actuator", "optimizer_count", "optimizer_gradient"}
    assert entries["actuator"]["run_owned"] == "run_runtime_readback_smoke.run_owned"
    assert entries["optimizer_gradient"]["verified_functions"]["child"] == [
        "verify_training", "attach", "training_hooks", "train",
    ]
    assert entries["optimizer_count"]["verified_functions"]["worker"] == [
        "verify_training_binding", "train", "main",
    ]
    assert entries["optimizer_gradient"]["verified_functions"]["inspector"] == [
        "GradientObserver",
    ]
    assert entries["optimizer_gradient"]["verified_functions"]["contract"] == [
        "validate_manifest", "inspect_records",
    ]
    assert all(len(value) == 64 for entry in entries.values() for value in entry["source_sha256"].values())


def test_cpu_preflight_separates_source_check_from_unexecuted_measurement(tmp_path: Path) -> None:
    result = O.preflight(request(tmp_path), repo_root=ROOT, isaaclab_root=sdk(tmp_path))
    assert result["status"] == "ready_component" and result["outer_backend_ready"] is True
    assert result["standalone_executable"] is False
    assert result["components"]["actuator_mapping"]["checkpoint_consumed"] is False
    assert result["components"]["optimizer_observation"]["checkpoint_consumed"] is False
    assert result["components"]["optimizer_observation"]["infers_past_checkpoint_updates"] is False
    assert result["evidence_roles"]["recorded_replay"].startswith("not executed")
    assert result["evidence_roles"]["new_sdk_measurement"].startswith("not executed")
    assert result["input_binding"]["case"]["checkpoint"]["sha256"] == request(tmp_path)["case"]["checkpoint"]["sha256"]


@pytest.mark.parametrize("mutation", [
    lambda value: value["case"]["checkpoint"].update(sha256="b" * 64),
    lambda value: value.update(runtime_updates=2),
    lambda value: value["probes"].reverse(),
    lambda value: value["case"].update(case_id="reference"),
    lambda value: value["lease"].update(lease_id="caller-says-verified"),
])
def test_request_tampering_is_rejected_before_source_preflight(tmp_path: Path, mutation) -> None:
    value = request(tmp_path)
    mutation(value)
    with pytest.raises(ValueError):
        O.preflight(value, repo_root=ROOT, isaaclab_root=sdk(tmp_path))


def test_missing_sdk_runtime_is_reported_without_execution(tmp_path: Path) -> None:
    result = O.preflight(request(tmp_path), repo_root=ROOT, isaaclab_root=tmp_path / "missing")
    assert result["status"] == "incomplete_runtime"
    assert result["runtime_missing"] == list(O.ISAAC_RUNTIME_FILES)
    assert result["isaac_runtime_sha256"] == {}


def test_actual_cli_returns_ready_component_receipt_without_gpu(tmp_path: Path) -> None:
    value = request(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(value), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-B", str(ROOT / "bench/registered_mechanism_observer_v1.py"),
         "preflight", "--request", str(request_path), "--isaaclab", str(sdk(tmp_path)),
         "--repo-root", str(ROOT)],
        check=False, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0 and proc.stderr == ""
    receipt = json.loads(proc.stdout)
    assert receipt["schema"] == O.PREFLIGHT_SCHEMA and receipt["status"] == "ready_component"
    assert receipt["request_id"] == value["request_id"]
    assert receipt["evidence_roles"]["new_sdk_measurement"].startswith("not executed")


def test_normalization_does_not_mutate_caller_request(tmp_path: Path) -> None:
    value = request(tmp_path)
    before = deepcopy(value)
    O.normalize_request(value)
    assert value == before
