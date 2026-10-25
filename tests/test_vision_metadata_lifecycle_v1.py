from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
M = importlib.import_module("vision_metadata_lifecycle_v1")

OWNER = {
    "id": "owned-id", "name": "owned-vm", "instance_type": "l40s-48gb.1x",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object | str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _ref(repo: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(repo).as_posix(), "sha256": _sha(path)}


def _inventory(status: str) -> dict[str, Any]:
    return {"workspaces": [{**OWNER, "status": status}]}


def _transport(
    root: Path, name: str, command: list[str], before: str, after: str,
    stdout: object | str, stderr: str = "",
) -> tuple[Path, Path, Path]:
    folder = root / name
    stdout_path = _write(
        folder / "stdout.txt",
        (
            stdout.encode()
            if isinstance(stdout, str)
            else json.dumps(stdout, ensure_ascii=False).encode()
        ),
    )
    stderr_path = _write(folder / "stderr.txt", stderr.encode())
    receipt = _write(folder / "receipt.json", {
        "command": command, "request_before_utc": before,
        "response_after_utc": after, "exit_code": 0,
        "stdout_path": "/historical/absolute/stdout.txt", "stdout_sha256": _sha(stdout_path),
        "stderr_path": "/historical/absolute/stderr.txt", "stderr_sha256": _sha(stderr_path),
    })
    return receipt, stdout_path, stderr_path


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    root = repo / ".triage-runtime/vision-brev/lifecycle/run-metadata"
    root.mkdir(parents=True)
    before = _write(root / "before.json", _inventory("STOPPED"))
    start_command = ["brev", "start", OWNER["name"], "--detached"]
    start_receipt, start_stdout, start_stderr = _transport(
        root, "start", start_command,
        "2026-10-25T12:56:18.456307+00:00", "2026-10-25T12:56:27.136879+00:00",
        "started\n",
    )
    start_native = json.loads(start_receipt.read_text(encoding="utf-8"))
    start_observation = _write(root / "start-observation.json", {
        "schema": "v04_start_observation_v1", "resource_id": OWNER["id"],
        "resource_name": OWNER["name"], "before_inventory_path": "/historical/before.json",
        "before_inventory_sha256": _sha(before), **start_native,
    })
    script = _write(root / "inspect.sh", "#!/usr/bin/env bash\nexit 0\n")
    idle_command = ["brev", "exec", OWNER["name"], "@/historical/inspect.sh"]
    intent = _write(root / "inspection-intent.json", {
        "schema": "vision_atomic_exec_script_intent_v1",
        "source_path": "/historical/source/inspect.sh", "source_sha256": _sha(script),
        "snapshot_path": "/historical/inspect.sh", "command": idle_command,
    })
    idle_receipt, idle_stdout, idle_stderr = _transport(
        root, "idle", idle_command,
        "2026-10-25T13:03:22.869947+00:00", "2026-10-25T13:03:24.691552+00:00",
        OWNER["name"] + "\n",
    )
    stop_receipt, stop_stdout, stop_stderr = _transport(
        root, "stop", ["brev", "stop", OWNER["name"]],
        "2026-10-25T13:04:00.004403+00:00", "2026-10-25T13:04:04.280506+00:00",
        OWNER["name"] + "\n",
    )
    final_receipt, final_stdout, final_stderr = _transport(
        root, "final", ["brev", "ls", "--json"],
        "2026-10-25T13:04:31.734253+00:00", "2026-10-25T13:04:34.707887+00:00",
        _inventory("STOPPED"),
    )
    metadata_root = root / "metadata"
    rows = []
    for index, relative in enumerate(sorted(M.METADATA_ONLY_PATHS)):
        path = _write(metadata_root / relative, {"index": index})
        rows.append({
            "path": path.relative_to(metadata_root).as_posix(),
            "bytes": path.stat().st_size, "sha256": _sha(path),
        })
    inventory = _write(root / "metadata-inventory.json", {
        "schema": "paced_hold_metadata_inventory_v1", "files": rows,
        "remote_root": "/home/ubuntu/workspace/run", "resource_id": OWNER["id"],
    })
    evidence_paths = {
        "start_observation": start_observation, "before_inventory": before,
        "start_receipt": start_receipt, "start_stdout": start_stdout,
        "start_stderr": start_stderr, "inspection_intent": intent,
        "inspection_script": script, "idle_receipt": idle_receipt,
        "idle_stdout": idle_stdout, "idle_stderr": idle_stderr,
        "stop_receipt": stop_receipt, "stop_stdout": stop_stdout,
        "stop_stderr": stop_stderr, "final_inventory_receipt": final_receipt,
        "final_inventory_stdout": final_stdout, "final_inventory_stderr": final_stderr,
    }
    envelope = {
        "schema": "vision_metadata_lifecycle_v1", "run_id": "run-metadata",
        "owner": OWNER,
        "quote": {
            "currency": "USD", "hourly_rate_usd": "1.74",
            "rounding": "ROUND_HALF_UP", "quantum_usd": "0.000001",
        },
        "period": {
            "started_at_utc": "2026-10-25T12:56:18.456307+00:00",
            "stopped_at_utc": "2026-10-25T13:04:34.707887+00:00",
        },
        "reported": {"wall_seconds": "496.25158", "cost_usd": "0.239855"},
        "claims": {
            "resource_proof_issued": False, "lease_issued": False,
            "measurement_resumed": False,
        },
        "evidence": {name: _ref(repo, path) for name, path in evidence_paths.items()},
        "metadata_capture": {
            "inventory": _ref(repo, inventory),
            "root": metadata_root.relative_to(repo).as_posix(), "files": 16,
        },
    }
    envelope_path = _write(root / "envelope.json", envelope)
    return {
        "repo": repo, "root": root, "envelope": envelope_path,
        "script_sha": _sha(script), "paths": evidence_paths,
        "metadata_root": metadata_root,
    }


def _validate(fx: dict[str, Any]):
    return M.validate_metadata_lifecycle(
        fx["envelope"], repo=fx["repo"], expected_owner=OWNER,
        expected_hourly_rate_usd=Decimal("1.74"),
        allowed_inspection_script_sha256={fx["script_sha"]},
    )


def _envelope(fx: dict[str, Any]) -> dict[str, Any]:
    return json.loads(fx["envelope"].read_text(encoding="utf-8"))


def _save_envelope(fx: dict[str, Any], body: dict[str, Any]) -> None:
    fx["envelope"].write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")


def _replace_json_ref(
    fx: dict[str, Any], key: str, mutate,
) -> None:
    path = fx["paths"][key]
    body = json.loads(path.read_text(encoding="utf-8"))
    mutate(body)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    envelope = _envelope(fx)
    envelope["evidence"][key]["sha256"] = _sha(path)
    _save_envelope(fx, envelope)


def test_valid_metadata_lifecycle_returns_typed_cost(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    result = _validate(fx)
    assert result == M.MetadataLifecycleCost(
        run_id="run-metadata",
        started_at_utc="2026-10-25T12:56:18.456307+00:00",
        stopped_at_utc="2026-10-25T13:04:34.707887+00:00",
        wall_seconds=Decimal("496.25158"), cost_usd=Decimal("0.239855"),
    )


def test_rejects_metadata_raw_sha_tamper(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    (fx["metadata_root"] / min(M.METADATA_ONLY_PATHS)).write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata file"):
        _validate(fx)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink 권한과 무관한 POSIX 회귀")
def test_rejects_extra_symlink_directory_in_metadata_root(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    (fx["metadata_root"] / "extra-link").symlink_to(
        fx["metadata_root"] / "admission", target_is_directory=True,
    )
    with pytest.raises(ValueError, match="metadata root에 symlink"):
        _validate(fx)


def test_rejects_different_owner(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        M.validate_metadata_lifecycle(
            fx["envelope"], repo=fx["repo"],
            expected_owner={**OWNER, "id": "other"},
            expected_hourly_rate_usd=Decimal("1.74"),
            allowed_inspection_script_sha256={fx["script_sha"]},
        )


def test_rejects_wrong_stop_command(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _replace_json_ref(fx, "stop_receipt", lambda body: body.update(
        command=["brev", "stop", "other"]
    ))
    with pytest.raises(ValueError, match="stop command"):
        _validate(fx)


def test_rejects_final_running(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    path = fx["paths"]["final_inventory_stdout"]
    path.write_text(json.dumps(_inventory("RUNNING")), encoding="utf-8")
    receipt = fx["paths"]["final_inventory_receipt"]
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["stdout_sha256"] = _sha(path)
    receipt.write_text(json.dumps(body), encoding="utf-8")
    envelope = _envelope(fx)
    envelope["evidence"]["final_inventory_stdout"]["sha256"] = _sha(path)
    envelope["evidence"]["final_inventory_receipt"]["sha256"] = _sha(receipt)
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="STOPPED"):
        _validate(fx)


def test_rejects_reversed_time(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _replace_json_ref(fx, "idle_receipt", lambda body: body.update(
        request_before_utc="2026-10-25T12:55:00+00:00",
        response_after_utc="2026-10-25T12:55:01+00:00",
    ))
    with pytest.raises(ValueError, match="시각 순서"):
        _validate(fx)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_rejects_nonfinite_cost(tmp_path: Path, value: str) -> None:
    fx = _fixture(tmp_path)
    envelope = _envelope(fx)
    envelope["reported"]["cost_usd"] = value
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="유한값"):
        _validate(fx)


def test_rejects_cost_miscalculation(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    envelope = _envelope(fx)
    envelope["reported"]["cost_usd"] = "0.239856"
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="비용 계산"):
        _validate(fx)


@pytest.mark.parametrize("body", ["", "owned-vm\n{}\n", "other\n", "owned-vm\nGPU 1\n"])
def test_rejects_idle_footer_missing_or_extra(tmp_path: Path, body: str) -> None:
    fx = _fixture(tmp_path)
    path = fx["paths"]["idle_stdout"]
    path.write_text(body, encoding="utf-8")
    receipt = fx["paths"]["idle_receipt"]
    native = json.loads(receipt.read_text(encoding="utf-8"))
    native["stdout_sha256"] = _sha(path)
    receipt.write_text(json.dumps(native), encoding="utf-8")
    envelope = _envelope(fx)
    envelope["evidence"]["idle_stdout"]["sha256"] = _sha(path)
    envelope["evidence"]["idle_receipt"]["sha256"] = _sha(receipt)
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="idle 출력"):
        _validate(fx)


def test_rejects_missing_evidence(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["paths"]["stop_stderr"].unlink()
    with pytest.raises(FileNotFoundError):
        _validate(fx)


def test_rejects_unallowed_inspection_script(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(ValueError, match="허용 SHA"):
        M.validate_metadata_lifecycle(
            fx["envelope"], repo=fx["repo"], expected_owner=OWNER,
            expected_hourly_rate_usd=Decimal("1.74"),
            allowed_inspection_script_sha256={"a" * 64},
        )


def test_rejects_synthetic_resource_proof_claim(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    envelope = _envelope(fx)
    envelope["claims"]["resource_proof_issued"] = True
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="claim"):
        _validate(fx)


def test_rejects_integer_zero_claim(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    envelope = _envelope(fx)
    envelope["claims"]["resource_proof_issued"] = 0
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="claim"):
        _validate(fx)


def test_rejects_boolean_exit_code(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _replace_json_ref(fx, "stop_receipt", lambda body: body.update(exit_code=False))
    with pytest.raises(ValueError, match="exit code"):
        _validate(fx)


@pytest.mark.parametrize("field", ["source_path", "snapshot_path"])
def test_rejects_empty_inspection_paths(tmp_path: Path, field: str) -> None:
    fx = _fixture(tmp_path)
    _replace_json_ref(fx, "inspection_intent", lambda body: body.update({field: ""}))
    with pytest.raises(ValueError, match="inspection script"):
        _validate(fx)


def test_rejects_inspection_command_not_bound_to_snapshot(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _replace_json_ref(fx, "inspection_intent", lambda body: body.update(
        command=["brev", "exec", OWNER["name"], "@/historical/other.sh"]
    ))
    with pytest.raises(ValueError, match="inspection command"):
        _validate(fx)


def test_rejects_non_metadata_inventory_path(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    source = fx["metadata_root"] / "bootstrap.json"
    target = fx["metadata_root"] / "training/checkpoint.pt"
    target.parent.mkdir()
    source.rename(target)
    inventory_ref = _envelope(fx)["metadata_capture"]["inventory"]
    inventory = fx["repo"] / inventory_ref["path"]
    body = json.loads(inventory.read_text(encoding="utf-8"))
    row = next(row for row in body["files"] if row["path"] == "bootstrap.json")
    row["path"] = "training/checkpoint.pt"
    inventory.write_text(json.dumps(body), encoding="utf-8")
    envelope = _envelope(fx)
    envelope["metadata_capture"]["inventory"]["sha256"] = _sha(inventory)
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="파일 집합"):
        _validate(fx)


def test_rejects_ref_outside_repo(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    envelope = _envelope(fx)
    envelope["evidence"]["before_inventory"]["path"] = "../outside.json"
    _save_envelope(fx, envelope)
    with pytest.raises(ValueError, match="상대경로"):
        _validate(fx)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink 권한과 무관한 POSIX 회귀")
def test_rejects_symlink_evidence(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    path = fx["paths"]["stop_stderr"]
    target = path.with_name("target.txt")
    target.write_bytes(b"")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _validate(fx)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink 권한과 무관한 POSIX 회귀")
def test_rejects_symlink_parent_of_evidence(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    original = fx["root"] / "stop"
    relocated = fx["root"] / "real-stop"
    original.rename(relocated)
    original.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _validate(fx)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink 권한과 무관한 POSIX 회귀")
def test_rejects_symlink_metadata_parent(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    original = fx["metadata_root"] / "admission"
    relocated = fx["metadata_root"] / "real-admission"
    original.rename(relocated)
    original.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _validate(fx)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink 권한과 무관한 POSIX 회귀")
def test_rejects_symlink_envelope(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    original = fx["envelope"]
    target = original.with_name("real-envelope.json")
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _validate(fx)


@pytest.mark.skipif(os.name == "nt", reason="Windows hardlink 동작과 무관한 POSIX 회귀")
def test_rejects_hardlinked_evidence(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    path = fx["paths"]["stop_stderr"]
    os.link(path, path.with_name("hardlink.txt"))
    with pytest.raises(ValueError, match="regular file"):
        _validate(fx)
