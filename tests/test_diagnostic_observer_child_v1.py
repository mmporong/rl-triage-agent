from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import diagnostic_observer_child_v1 as C
import lineage_source_batch_v1 as LEASE


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def file_ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def anchor() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def fixture(tmp_path: Path, component: str = "optimizer_update_bundle", condition: str = "R1"):
    isaaclab = tmp_path / "IsaacLab"
    runtime_paths = [
        "_isaac_sim/python.bat",
        "scripts/reinforcement_learning/rsl_rl/train.py",
        "source/isaaclab/isaaclab/actuators/actuator_pd.py",
        "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
        "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
    ]
    for relative in runtime_paths:
        path = isaaclab / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"runtime {relative}\n", encoding="utf-8")
    runtime_pins = {"logical_root": "isaaclab",
                    "sha256": {relative: sha(isaaclab / relative) for relative in runtime_paths}}
    closure_paths = [
        "bench/actuator_envelope_runtime.py",
        "bench/run_actuator_envelope_runtime.py",
        "bench/ppo_gradient_training_v5.py",
        "evals/ppo_gradient_invariant_v1.py",
        "evals/ppo_gradient_invariant_v2.py",
        "evals/lineage_source_batch_v1.py",
        "evals/lease_timestamps.py",
    ]
    closure = {relative: sha_lf(ROOT / relative) for relative in closure_paths}
    base = ROOT / "bench/protocols/actuator_envelope_runtime_v3.json"
    protocol = {
        "schema": C.PROTOCOL_SCHEMA,
        "protocol_id": "diagnostic-test",
        "source_anchor": anchor(),
        "source_closure": closure,
        "runtime_pins": runtime_pins,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "actuator_base_protocol": {
            "path": "bench/protocols/actuator_envelope_runtime_v3.json", "sha256": sha(base),
        },
        "allowed_operations": {
            "actuator_mapping": {"component": "actuator_mapping", "probe": C.ACTUATOR_PROBES,
                                 "conditions": ["NONE", "F2"], "updates": 0},
            "optimizer_update_bundle": {"component": "optimizer_update_bundle",
                                        "probe": C.OPTIMIZER_PROBES,
                                        "conditions": ["NONE", "R1", "inverse_ratio"], "updates": 3},
        },
        "limits": {"expected_learning_epochs": 5, "expected_mini_batches": 4,
                   "optimizer_updates": 3},
    }
    protocol_path = write(tmp_path / "registered_protocol.json", protocol)
    checkpoint = tmp_path / "case.pt"
    checkpoint.write_bytes(b"checkpoint identity only")
    request_id, effect_key = "request-1", "effect-1"
    refs = {}
    for name in ("request", "approval", "consumption"):
        path = write(tmp_path / f"{name}.json", {
            "schema": f"{name}_fixture_v1", "request_id": request_id, "effect_key": effect_key,
        })
        refs[name] = file_ref(path)
    lease_path = write(tmp_path / "lease.json", {
        "owner": LEASE.OWNER,
        "leaseId": "12345678-1234-1234-1234-123456789abc",
        "state": "running", "childPid": 4242, "wrapperPid": 4200,
        "completedUtc": None,
        "acquiredUtc": "2026-11-04T12:00:00.0000000Z",
        "heartbeatUtc": "2026-11-04T12:00:01.0000000Z",
    })
    probe = C.ACTUATOR_PROBES if component == "actuator_mapping" else C.OPTIMIZER_PROBES
    manifest = C.build_manifest(
        protocol_path=protocol_path,
        operation=component,
        request_id=request_id,
        effect_key=effect_key,
        probe=probe,
        case_binding={"case_id": "case", "input_sha256": "a" * 64},
        source_binding={"source_id": "source", "source_sha256": "b" * 64},
        checkpoint_binding={"path": str(checkpoint), "sha256": sha(checkpoint), "role": "identity_only"},
        reference_binding={"case_id": "reference", "input_sha256": "c" * 64},
        source_anchor=protocol["source_anchor"],
        source_closure=closure,
        runtime_pins=runtime_pins,
        seed=2201,
        device="cuda:0",
        num_envs=64,
        condition=condition,
        component=component,
        outer_lease={"ref": file_ref(lease_path), "lease_id": "12345678-1234-1234-1234-123456789abc",
                     "owner_pid": 4242},
        consumed_approval=refs,
        output_dir=tmp_path / "outputs" / component,
    )
    manifest_path = write(tmp_path / f"{component}.manifest.json", manifest)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    return manifest_path, protocol_path, isaaclab, manifest


@pytest.mark.parametrize(("component", "condition", "probes", "updates"), [
    ("actuator_mapping", "F2", C.ACTUATOR_PROBES, 0),
    ("optimizer_update_bundle", "R1", C.OPTIMIZER_PROBES, 3),
    ("optimizer_update_bundle", "inverse_ratio", C.OPTIMIZER_PROBES, 3),
])
def test_build_and_verify_manifest_binds_registered_component(
    tmp_path: Path, component: str, condition: str, probes: list[str], updates: int,
) -> None:
    manifest_path, protocol_path, isaaclab, manifest = fixture(tmp_path, component, condition)
    checked = C.verify_manifest(
        manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
        expected_protocol_path=protocol_path,
    )
    assert checked["manifest"] == manifest
    assert manifest["probe"] == probes and manifest["updates"] == updates
    assert manifest["checkpoint_consumed"] is False
    assert manifest["infers_past_checkpoint_updates"] is False
    assert manifest["condition_model_visible"] is False
    assert checked["output"].exists() is False


def test_import_boundary_does_not_load_sdk_or_torch() -> None:
    assert "torch" not in sys.modules
    assert not any(name.startswith("isaaclab") for name in sys.modules)


def _directory_alias(link: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=True, capture_output=True, encoding="utf-8", errors="replace",
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def test_sdk_installation_alias_is_allowed_only_for_runtime_pins(tmp_path: Path) -> None:
    lab, sdk = tmp_path / "lab", tmp_path / "installed-sdk"
    lab.mkdir()
    sdk.mkdir()
    code = sdk / "ppo.py"
    code.write_text("sdk source\n", encoding="utf-8")
    _directory_alias(lab / "_isaac_sim", sdk)
    pins = {"_isaac_sim/ppo.py": sha(code)}
    with pytest.raises(ValueError, match="path changed"):
        C._verify_hash_map(lab, pins, "source closure")
    assert C._verify_hash_map(lab, pins, "runtime pins", allow_isaac_sdk_alias=True) == pins
    with pytest.raises(ValueError, match="path changed"):
        C._verify_hash_map(lab, {"_isaac_sim/ppo.py": "0" * 64}, "runtime pins",
                           allow_isaac_sdk_alias=True)


def test_sdk_alias_does_not_allow_other_or_nested_directory_escapes(tmp_path: Path) -> None:
    lab, sdk, outside = (tmp_path / name for name in ("lab", "sdk", "outside"))
    for directory in (lab, sdk, outside):
        directory.mkdir()
    code = outside / "outside.py"
    code.write_text("outside source\n", encoding="utf-8")
    _directory_alias(lab / "_isaac_sim", sdk)
    _directory_alias(lab / "unregistered", outside)
    _directory_alias(sdk / "escape", outside)
    for relative in ("unregistered/outside.py", "_isaac_sim/escape/outside.py"):
        with pytest.raises(ValueError, match="path changed"):
            C._verify_hash_map(lab, {relative: sha(code)}, "runtime pins",
                               allow_isaac_sdk_alias=True)
    for relative in ("../outside/outside.py", str(code)):
        with pytest.raises(ValueError, match="entry is invalid"):
            C._verify_hash_map(lab, {relative: sha(code)}, "runtime pins",
                               allow_isaac_sdk_alias=True)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(checkpoint_consumed=True),
    lambda value: value.update(infers_past_checkpoint_updates=True),
    lambda value: value.update(condition_model_visible=True),
    lambda value: value.update(condition="F2"),
    lambda value: value["probe"].reverse(),
    lambda value: value.update(updates=1),
    lambda value: value["source_closure"].update({"bench/actuator_envelope_runtime.py": "d" * 64}),
    lambda value: value["outer_lease"].update(lease_id="caller-verified"),
])
def test_manifest_tampering_is_rejected_before_sdk_import(tmp_path: Path, mutation) -> None:
    manifest_path, protocol_path, isaaclab, manifest = fixture(tmp_path)
    mutation(manifest)
    write(manifest_path, manifest)
    with pytest.raises((ValueError, FileExistsError)):
        C.verify_manifest(
            manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
            expected_protocol_path=protocol_path,
        )
    assert "torch" not in sys.modules


def test_checkpoint_approval_and_lease_bytes_are_rechecked(tmp_path: Path) -> None:
    manifest_path, protocol_path, isaaclab, manifest = fixture(tmp_path)
    Path(manifest["checkpoint"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint file changed"):
        C.verify_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
                          expected_protocol_path=protocol_path)

    manifest_path, protocol_path, isaaclab, manifest = fixture(tmp_path / "approval")
    Path(manifest["consumed_approval"]["approval"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="approval file changed"):
        C.verify_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
                          expected_protocol_path=protocol_path)

    manifest_path, protocol_path, isaaclab, manifest = fixture(tmp_path / "lease")
    Path(manifest["outer_lease"]["ref"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outer lease file changed"):
        C.verify_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
                          expected_protocol_path=protocol_path)


def test_fresh_output_and_no_arbitrary_manifest_fields(tmp_path: Path) -> None:
    manifest_path, protocol_path, isaaclab, manifest = fixture(tmp_path)
    output = Path(manifest["output"]["path"])
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        C.verify_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
                          expected_protocol_path=protocol_path)
    output.rmdir()
    manifest["arbitrary_args"] = ["--resume", "--checkpoint", "anything"]
    write(manifest_path, manifest)
    with pytest.raises(ValueError, match="fields"):
        C.verify_manifest(manifest_path, repo_root=ROOT, isaaclab_root=isaaclab,
                          expected_protocol_path=protocol_path)


def test_absent_registered_protocol_rejects_before_child_execution(tmp_path: Path) -> None:
    manifest_path, protocol_path, _isaaclab, _manifest = fixture(tmp_path)
    protocol_path.unlink()
    with pytest.raises(ValueError, match="registered parent protocol"):
        C.child_main(manifest_path)
    assert "torch" not in sys.modules


def _live_metadata(*, heartbeat: str = "2026-11-04T12:00:30.0000000Z", state: str = "running"):
    return {
        "owner": LEASE.OWNER,
        "leaseId": "12345678-1234-1234-1234-123456789abc",
        "state": state,
        "childPid": 4242,
        "wrapperPid": 4200,
        "completedUtc": None if state == "running" else "2026-11-04T12:00:31.0000000Z",
        "acquiredUtc": "2026-11-04T12:00:00.0000000Z",
        "heartbeatUtc": heartbeat,
    }


def test_live_lease_accepts_multihop_child_and_new_heartbeat(monkeypatch) -> None:
    metadata = _live_metadata()
    ancestors = [
        {"pid": 4303, "parent": 4302, "name": "kit.exe"},
        {"pid": 4302, "parent": 4301, "name": "cmd.exe"},
        {"pid": 4301, "parent": 4242, "name": "python.exe"},
        {"pid": 4242, "parent": 4200, "name": "python.exe"},
        {"pid": 4200, "parent": 1, "name": "pwsh.exe"},
    ]
    checked = LEASE.validate_lease(
        metadata, ancestors, 4303, datetime(2026, 11, 4, 12, 0, 31, tzinfo=timezone.utc),  # noqa: UP017 - SDK Python 3.10 호환
    )
    monkeypatch.setattr(LEASE, "lease_context", lambda: checked)
    C._live_lease({"outer_lease": {
        "lease_id": metadata["leaseId"], "owner_pid": 4242,
        "owner": LEASE.OWNER, "wrapper_pid": 4200,
    }})


@pytest.mark.parametrize("kind", ["wrong_id", "released", "stale"])
def test_live_lease_rejects_wrong_id_release_and_stale_heartbeat(monkeypatch, kind: str) -> None:
    metadata = _live_metadata(
        heartbeat=("2026-11-04T11:58:00.0000000Z" if kind == "stale"
                   else "2026-11-04T12:00:30.0000000Z"),
        state="complete" if kind == "released" else "running",
    )
    if kind == "wrong_id":
        metadata["leaseId"] = "99999999-9999-9999-9999-999999999999"
    ancestors = [
        {"pid": 4303, "parent": 4242, "name": "kit.exe"},
        {"pid": 4242, "parent": 4200, "name": "python.exe"},
        {"pid": 4200, "parent": 1, "name": "powershell.exe"},
    ]
    if kind == "wrong_id":
        checked = LEASE.validate_lease(
            metadata, ancestors, 4303, datetime(2026, 11, 4, 12, 0, 31, tzinfo=timezone.utc),  # noqa: UP017 - SDK Python 3.10 호환
        )
        monkeypatch.setattr(LEASE, "lease_context", lambda: checked)
        with pytest.raises(ValueError, match="live outer lease"):
            C._live_lease({"outer_lease": {
                "lease_id": "12345678-1234-1234-1234-123456789abc", "owner_pid": 4242,
                "owner": LEASE.OWNER, "wrapper_pid": 4200,
            }})
    else:
        with pytest.raises(ValueError, match="lease|heartbeat|기간|신원"):
            LEASE.validate_lease(
                metadata, ancestors, 4303, datetime(2026, 11, 4, 12, 0, 31, tzinfo=timezone.utc),  # noqa: UP017 - SDK Python 3.10 호환
            )
