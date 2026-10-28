from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench" / "probe_isaac_camera_startup_v1.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("probe_isaac_camera_startup_v1", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_module()


def raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8", newline="\n")


class FakeJob4:
    def __init__(self, events: list[str]):
        self.events = events

    def validate_protocol(self, path: Path, **_: Any) -> dict[str, Any]:
        self.events.append("protocol")
        return json.loads(path.read_text(encoding="utf-8"))

    def validate_resource_proof(
        self, path: Path, expected_sha: str, *, now: datetime, **_: Any,
    ) -> dict[str, Any]:
        self.events.append("proof")
        if raw_sha(path) != expected_sha:
            raise M.CameraStartupError("resource proof changed")
        proof = json.loads(path.read_text(encoding="utf-8"))
        if datetime.fromisoformat(proof["expires_at_utc"]) <= now:
            raise M.CameraStartupError("resource proof expired")
        return proof

    def verify_lease(self, lease: dict[str, Any], *, now: datetime, **_: Any) -> dict[str, Any]:
        self.events.append("lease")
        if datetime.fromisoformat(lease["expires_at_utc"]) <= now:
            raise M.CameraStartupError("lease expired")
        return {"reference": {"path": lease["path"], "sha256": lease["sha256"]}}


class FakeApp:
    def __init__(
        self, events: list[str], *, update_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ):
        self.events = events
        self.update_error = update_error
        self.close_error = close_error

    def update(self) -> None:
        self.events.append("update")
        if self.update_error is not None:
            raise self.update_error

    def close(self) -> None:
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def inputs(tmp_path: Path) -> SimpleNamespace:
    now = datetime(2026, 10, 29, 0, 0, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10 target
    lease_path = tmp_path / "lease.json"
    write_json(lease_path, {"active": True})
    lease = {
        "path": str(lease_path),
        "sha256": raw_sha(lease_path),
        "expires_at_utc": (now + timedelta(seconds=20)).isoformat(),
    }
    proof_path = tmp_path / "resource-proof.json"
    write_json(proof_path, {
        "expires_at_utc": (now + timedelta(seconds=20)).isoformat(),
        "lease": lease,
    })
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, {
        "_recording_target": {"resource_id": "resource-1"},
        "source_sha256_lf": {
            relative: lf_sha(ROOT / relative) for relative in M.SOURCE_FILES
        },
    })
    return SimpleNamespace(
        now=now,
        protocol=protocol_path,
        proof=proof_path,
        proof_sha=raw_sha(proof_path),
        lease=lease_path,
        output=tmp_path / "output",
    )


def fixture(events: list[str]) -> dict[str, Any]:
    return {
        "job4": FakeJob4(events),
        "record": SimpleNamespace(fixed_profile=lambda: {"launcher": dict(M.LAUNCHER)}),
    }


def test_import_is_cpu_only() -> None:
    script = (
        "import importlib.util,json,sys;"
        f"p={str(SOURCE)!r};"
        "s=importlib.util.spec_from_file_location('camera_probe_import',p);"
        "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
        "bad=sorted(n for n in sys.modules if n.split('.')[0] in "
        "{'isaaclab','isaacsim','torch','carb','omni'});print(json.dumps(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="strict",
    )
    assert json.loads(completed.stdout) == []


def test_execute_validates_before_startup_and_records_unconfirmed_result(inputs: SimpleNamespace) -> None:
    events: list[str] = []

    def app_factory(**options: Any) -> FakeApp:
        assert options == M.LAUNCHER
        events.append("startup")
        return FakeApp(events)

    result = M.execute(
        inputs.protocol, inputs.proof, inputs.proof_sha, inputs.output,
        repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture(events),
        app_factory=app_factory,
    )
    assert events == ["protocol", "proof", "lease", "startup", "update", "close"]
    assert result["status"] == "awaiting_parent_exit_receipt"
    assert result["success_eligible_without_parent"] is False
    assert json.loads((inputs.output / "preclose-result.json").read_text(encoding="utf-8"))["status"] == "exit_unconfirmed"
    assert sorted(path.stem for path in (inputs.output / "markers").glob("*.json")) == [
        "close-entered", "close-returned", "startup-entered", "startup-returned",
        "update-entered", "update-returned",
    ]


def test_tampered_proof_is_rejected_before_output_or_app(inputs: SimpleNamespace) -> None:
    events: list[str] = []
    inputs.proof.write_text("{}", encoding="utf-8")
    with pytest.raises(M.CameraStartupError, match="resource proof changed"):
        M.execute(
            inputs.protocol, inputs.proof, inputs.proof_sha, inputs.output,
            repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture(events),
            app_factory=lambda **_: pytest.fail("SDK must not start"),
        )
    assert events == ["protocol", "proof"]
    assert not inputs.output.exists()


@pytest.mark.parametrize(("field", "message"), [
    ("proof", "resource proof expired"),
    ("lease", "lease expired"),
])
def test_expired_authority_is_rejected_before_output(
    inputs: SimpleNamespace, field: str, message: str,
) -> None:
    proof = json.loads(inputs.proof.read_text(encoding="utf-8"))
    if field == "proof":
        proof["expires_at_utc"] = (inputs.now - timedelta(seconds=1)).isoformat()
    else:
        proof["lease"]["expires_at_utc"] = (inputs.now - timedelta(seconds=1)).isoformat()
    write_json(inputs.proof, proof)
    with pytest.raises(M.CameraStartupError, match=message):
        M.execute(
            inputs.protocol, inputs.proof, raw_sha(inputs.proof), inputs.output,
            repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture([]),
            app_factory=lambda **_: pytest.fail("SDK must not start"),
        )
    assert not inputs.output.exists()


def test_missing_new_source_closure_is_rejected_before_output(inputs: SimpleNamespace) -> None:
    protocol = json.loads(inputs.protocol.read_text(encoding="utf-8"))
    protocol["source_sha256_lf"].pop("tests/test_isaac_camera_startup_v1.py")
    write_json(inputs.protocol, protocol)
    with pytest.raises(M.CameraStartupError, match="source closure missing"):
        M.execute(
            inputs.protocol, inputs.proof, inputs.proof_sha, inputs.output,
            repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture([]),
            app_factory=lambda **_: pytest.fail("SDK must not start"),
        )
    assert not inputs.output.exists()


def test_close_error_is_failure_not_success(inputs: SimpleNamespace) -> None:
    events: list[str] = []
    with pytest.raises(RuntimeError, match="close failed"):
        M.execute(
            inputs.protocol, inputs.proof, inputs.proof_sha, inputs.output,
            repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture(events),
            app_factory=lambda **_: FakeApp(events, close_error=RuntimeError("close failed")),
        )
    failure = json.loads((inputs.output / "failure.json").read_text(encoding="utf-8"))
    assert failure == {
        "error_type": "RuntimeError", "reason": "close failed", "schema": M.SCHEMA + "_failure",
        "stage": "close", "status": "failed", "success_eligible": False,
    }
    assert (inputs.output / "markers" / "close-error.json").is_file()
    assert not (inputs.output / "child-result.json").exists()


def test_sdk_system_exit_is_failure_not_success(inputs: SimpleNamespace) -> None:
    with pytest.raises(M.CameraStartupError, match=r"SDK raised SystemExit\(0\)"):
        M.execute(
            inputs.protocol, inputs.proof, inputs.proof_sha, inputs.output,
            repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture([]),
            app_factory=lambda **_: (_ for _ in ()).throw(SystemExit(0)),
        )
    failure = json.loads((inputs.output / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "SystemExit"
    assert failure["exit_code"] == 0
    assert failure["stage"] == "startup"
    assert failure["success_eligible"] is False
    assert not (inputs.output / "child-result.json").exists()


def test_update_failure_keeps_first_failure_stage_after_close(inputs: SimpleNamespace) -> None:
    events: list[str] = []
    with pytest.raises(RuntimeError, match="update failed"):
        M.execute(
            inputs.protocol, inputs.proof, inputs.proof_sha, inputs.output,
            repo=ROOT, now=inputs.now, allow_fixture=True, fixture=fixture(events),
            app_factory=lambda **_: FakeApp(events, update_error=RuntimeError("update failed")),
        )
    failure = json.loads((inputs.output / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "update"
    assert failure["reason"] == "update failed"
    assert events[-1] == "close"


def test_system_exit_zero_returns_nonzero_in_subprocess(tmp_path: Path) -> None:
    output = tmp_path / "system-exit-zero"
    output.mkdir()
    script = "\n".join([
        "import importlib.util, pathlib, sys",
        f"source = pathlib.Path({str(SOURCE)!r})",
        "spec = importlib.util.spec_from_file_location('camera_system_exit', source)",
        "module = importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name] = module",
        "spec.loader.exec_module(module)",
        f"module._run_sdk(pathlib.Path({str(output)!r}), module.LAUNCHER, allow_fixture=True, app_factory=lambda **_: (_ for _ in ()).throw(SystemExit(0)))",
    ])
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True,
        encoding="utf-8", errors="strict",
    )
    assert completed.returncode != 0
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "SystemExit"
    assert failure["exit_code"] == 0
    assert not (output / "child-result.json").exists()


@pytest.mark.parametrize("exit_code", [0, 245])
def test_close_time_process_exit_cannot_create_child_success(tmp_path: Path, exit_code: int) -> None:
    output = tmp_path / "abrupt-close"
    output.mkdir()
    script = "\n".join([
        "import importlib.util, os, pathlib, sys",
        f"source = pathlib.Path({str(SOURCE)!r})",
        "spec = importlib.util.spec_from_file_location('camera_abrupt', source)",
        "module = importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name] = module",
        "spec.loader.exec_module(module)",
        "class App:",
        "    def update(self): pass",
        f"    def close(self): os._exit({exit_code})",
        f"module._run_sdk(pathlib.Path({str(output)!r}), module.LAUNCHER, allow_fixture=True, app_factory=lambda **_: App())",
    ])
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True,
        encoding="utf-8", errors="strict",
    )
    assert completed.returncode == exit_code
    assert (output / "preclose-result.json").is_file()
    assert (output / "markers" / "close-entered.json").is_file()
    assert not (output / "markers" / "close-returned.json").exists()
    assert not (output / "child-result.json").exists()


def test_actual_mode_rejects_all_fixture_injection_before_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = tmp_path / "missing-protocol.json"
    proof = tmp_path / "missing-proof.json"
    monkeypatch.setattr(M, "os", SimpleNamespace(name="posix", getpid=M.os.getpid))
    with pytest.raises(M.CameraStartupError, match="actual validation requires POSIX and no fixture"):
        M.validate_inputs(protocol, proof, "0" * 64, allow_fixture=False, fixture={})


def test_source_does_not_use_active_cuda_gpus() -> None:
    assert "active_cuda_gpus" not in SOURCE.read_text(encoding="utf-8")
