from __future__ import annotations

import copy
import importlib
import io
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_vision_prepared_capture_v1 as W  # pyright: ignore[reportMissingImports]
import vision_prepared_capture_v1 as PREPARED  # pyright: ignore[reportMissingImports]


def _validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text('{"fixture":true}\n', encoding="utf-8")
    value = {
        "job_path": job,
        "job_sha256": W._sha(job),
        "output_directory": tmp_path / "output",
        "job": {
            "mode": W.JOB.MODE,
            "python": sys.executable,
            "isaaclab": str(tmp_path),
        },
        "resource": {"resource_id": "owned-vm", "gpu_uuid": "GPU-owned"},
    }
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    return value


def _bundles() -> list[dict[str, Any]]:
    return [
        {
            "ordinal": ordinal,
            "timing": {"read_times": {"frameNumber": 61 + ordinal}},
        }
        for ordinal in range(4)
    ]


def _capture_report() -> dict[str, Any]:
    camera_a = {
        "eye_world_m": [3.0, 3.0, 2.5],
        "lookat_world_m": [0.0, 0.0, 0.85],
    }
    camera_b = {
        "eye_world_m": [-3.0, 3.0, 2.5],
        "lookat_world_m": [0.0, 0.0, 0.85],
    }
    return {
        "schema": W.SMOKE.SCHEMA,
        "status": "pass",
        "method": W.SMOKE.METHOD,
        "capture_order": list(W.SMOKE.ORDER),
        "policy_steps": 0,
        "capture_phase_physics_steps": 0,
        "environment_reset_performed": True,
        "delta_time_s": 0.0,
        "rt_subframes": 1,
        "pause_timeline": False,
        "renderer_combination_only": True,
        "universal_zero_lag_claim": False,
        "captures": [
            {"camera": copy.deepcopy(camera_a), "atomic_bundle": _bundles()[0]},
            {"camera": copy.deepcopy(camera_a), "atomic_bundle": _bundles()[1]},
            {"camera": copy.deepcopy(camera_b), "atomic_bundle": _bundles()[2]},
            {"camera": copy.deepcopy(camera_a), "atomic_bundle": _bundles()[3]},
        ],
    }


def _prepared() -> dict[str, Any]:
    callbacks = [
        {
            "raw_index": 0,
            "window": "preparing",
            "classification": "valid",
            "error_type": None,
            "descriptor": {},
            "frame_number": 60,
        }
    ]
    callbacks.extend(
        {
            "raw_index": ordinal + 1,
            "window": f"arming{ordinal}",
            "classification": "valid",
            "error_type": None,
            "descriptor": {},
            "frame_number": 61 + ordinal,
        }
        for ordinal in range(4)
    )
    return {
        "schema": PREPARED.SCHEMA,
        "approved": True,
        "state": "sealed",
        "raw_callback_count": 5,
        "callback_rows_dropped": 0,
        "preparation": {
            "logical_ordinal": 4,
            "journal_written": False,
            "marker_code": 3,
            "bootstrap_empty_observed": False,
            "valid_raw_index": 0,
            "frame_number": 60,
        },
        "arm_sources": [
            {
                "ordinal": ordinal,
                "marker_code": ordinal,
                "raw_index": ordinal + 1,
                "frame_number": 61 + ordinal,
            }
            for ordinal in range(4)
        ],
        "callbacks": callbacks,
        "bundles": _bundles(),
        "failure": None,
    }


def _write_artifacts(value: dict[str, Any]) -> dict[str, Any]:
    output = value["output_directory"]
    output.mkdir(exist_ok=True)
    report = _capture_report()
    W.JOB.write_new(output / "job.smoke-report.json", report)
    artifacts: dict[str, Any] = {
        "report": "job.smoke-report.json",
        "journals": [],
        "audits": {},
    }
    for ordinal in range(4):
        json_leaf = f"job.atomic-{ordinal:06d}.json"
        npz_leaf = f"job.atomic-{ordinal:06d}.npz"
        (value["job_path"].parent / json_leaf).write_bytes(b"json")
        (value["job_path"].parent / npz_leaf).write_bytes(b"npz")
        artifacts["journals"].append(
            {"ordinal": ordinal, "json": json_leaf, "npz": npz_leaf}
        )
    for ordinal in (1, 2, 3):
        leaf = f"job.prepared-audit-rgb-{ordinal}.rgb"
        (output / leaf).write_bytes(bytes([ordinal]))
        artifacts["audits"][f"audit_rgb_{ordinal}"] = leaf
    return artifacts


def _envelope(value: dict[str, Any], *, sealed: bool) -> dict[str, Any]:
    lifecycle: dict[str, Any] = dict.fromkeys(W.LIFECYCLE_FLAGS, True)
    lifecycle["arms_completed"] = 4
    lifecycle["envelope_sealed"] = sealed
    lifecycle["errors"] = []
    camera_rows = [row["camera"] for row in _capture_report()["captures"]]
    phase_order = [
        "profile",
        "settings",
        "extension_enable",
        "extension_app_update",
        "replicator_startup",
        "annotator_registrations",
        "usd_context",
        "environment_config",
        "environment_create",
        "environment_reset",
        "command_configure",
        "setup_physics_before",
        "camera_configure",
        "render_product_create",
        "marker_root_position",
        "marker_create",
        "renderer_readback",
        "capture_readiness",
        "marker_bind",
        "setup_physics_after",
        "writer_attach",
        "prepare_marker_3",
        "prepare_camera_A",
        "prepare_event",
        "prepare_render",
    ]
    for ordinal in range(4):
        phase_order.extend(
            (
                f"arm_{ordinal}_marker",
                f"arm_{ordinal}_camera",
                f"arm_{ordinal}_event",
                f"arm_{ordinal}_render",
            )
        )
    phase_order.extend(
        (
            "writer_detach",
            "render_product_destroy",
            "environment_close",
            "mux_resolve",
            "mux_seal",
            "seal_report",
        )
    )
    camera_profile = W.RECORD.fixed_profile()["camera"]
    renderer = {
        "attached_hydra_engine_names": ["fake-hydra"],
        "carb_settings": dict.fromkeys(W.READY.CARB_SETTING_KEYS),
        "expected_capture_profile": {
            key: camera_profile[key]
            for key in (
                "capture_primitive",
                "rt_subframes",
                "pause_timeline",
                "delta_time_s",
            )
        },
        "renderer_accumulation_reset": "omni.usd.UsdContext.reset_renderer_accumulation",
    }
    return {
        "schema": "vision_prepared_sdk_v1",
        "prepared": _prepared(),
        "capture_report": _capture_report(),
        "artifacts": _write_artifacts(value),
        "lifecycle": lifecycle,
        "renderer": renderer,
        "setup_evidence": {
            "phase_order": phase_order,
            "setup_physics_before_after": {
                "before": {"state": 0},
                "after": {"state": 0},
                "equal": True,
            },
            "extension": {
                "name": "isaacsim.core.nodes",
                "enable_result": True,
                "app_update_completed": True,
            },
            "startup": {"before": {}, "after": {}},
            "registrations": dict.fromkeys(
                ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes"),
                True,
            ),
            "camera_poses": [
                {"phase": "prepare", **copy.deepcopy(camera_rows[0])},
                *[
                    {"phase": f"arm_{index}", **copy.deepcopy(camera)}
                    for index, camera in enumerate(camera_rows)
                ],
            ],
            "raw_callback_count": 5,
            "complete_drain_proven": False,
            "all_future_callbacks_collected": False,
        },
        "renderer_smoke_only": True,
        "approved": True,
    }


def _install_sdk_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("vision_prepared_sdk_v1")
    module.SDK_REPORT_SCHEMA = "vision_prepared_sdk_v1"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vision_prepared_sdk_v1", module)


class Settings:
    value = False

    def get_as_bool(self, _key: str) -> bool:
        return self.value

    def set_bool(self, _key: str, value: bool) -> None:
        self.value = value


class App:
    def __init__(
        self, validated: Any, *, close_error: BaseException | None = None
    ) -> None:
        self.validated, self.close_error = validated, close_error

    def close(self, *, wait_for_replicator: bool) -> None:
        assert wait_for_replicator is False
        output = self.validated["output_directory"]
        assert (
            sum((output / name).is_file() for name in ("result.json", "failure.json"))
            == 1
        )
        assert (self.validated["job_path"].parent / W.SHUTDOWN.SIDECAR).is_file()
        if self.close_error is not None:
            raise self.close_error


def _app_factory(
    value: dict[str, Any], *, close_error: BaseException | None = None
) -> Any:
    return W.SHUTDOWN.AppProxy(App(value, close_error=close_error), value, Settings())


def _run_sdk(value: dict[str, Any], _app: Any, *, seal_report: Any) -> dict[str, Any]:
    final = _envelope(value, sealed=True)
    sealed = copy.deepcopy(final)
    sealed["lifecycle"]["envelope_sealed"] = False
    seal_report(sealed)
    return final


class Reader:
    def __init__(self, tmp_path: Path, *, pids: list[int] | None = None) -> None:
        self.tmp_path, self.pids, self.count = tmp_path, pids or [], 0
        self.last_receipt: dict[str, str] | None = None

    def __call__(self) -> list[int]:
        self.count += 1
        prefix = f"{self.count:06d}"
        gpu = self.tmp_path / f"{prefix}.gpu.csv"
        process = self.tmp_path / f"{prefix}.process.csv"
        gpu.write_text("GPU-owned\n", encoding="utf-8")
        process.write_text(
            "".join(f"GPU-owned, {pid}\n" for pid in self.pids), encoding="utf-8"
        )
        receipt = self.tmp_path / f"{prefix}.receipt.json"
        W.JOB.write_new(
            receipt,
            {
                "schema": "vision_remote_gpu_observation_v1",
                "ordinal": self.count,
                "resource_id": "owned-vm",
                "gpu_uuid": "GPU-owned",
                "gpu_command": W.REMOTE.GPU_COMMAND,
                "gpu_exit_code": 0,
                "gpu_stdout_file": gpu.name,
                "gpu_stdout_sha256": W._sha(gpu),
                "gpu_stderr": "",
                "process_command": W.REMOTE.PROCESS_COMMAND,
                "process_exit_code": 0,
                "process_stdout_file": process.name,
                "process_stdout_sha256": W._sha(process),
                "process_stderr": "",
            },
        )
        self.last_receipt = {"path": str(receipt), "sha256": W._sha(receipt)}
        return self.pids


class GateInput(io.BytesIO):
    def close(self) -> None:
        self.accepted = self.getvalue()
        super().close()


class Process:
    pid = 12345
    returncode: int | None = None

    def __init__(self, action: Any, *, timeout: bool = False) -> None:
        self.stdin, self.action, self.timeout = GateInput(), action, timeout

    def wait(self, *, timeout: float) -> int:
        assert 0 < timeout <= 400
        assert self.stdin.accepted == b"GO\n"
        if self.timeout:
            raise subprocess.TimeoutExpired("fixture", timeout)
        self.returncode = self.action()
        assert self.returncode is not None
        return self.returncode


def _run_parent(
    value: dict[str, Any],
    *,
    action: Any,
    reader: Reader,
    timeout: bool = False,
    identify: Any = None,
    cleanup: Any = None,
) -> tuple[dict[str, Any], Process]:
    process = Process(action, timeout=timeout)

    def popen(command: list[str], **kwargs: Any) -> Process:
        module_file = W.__file__
        assert isinstance(module_file, str)
        assert command[1:4] == ["-I", "-S", "-c"]
        assert command[4] == W.GATE.OWNED_WORKER_GATE
        assert command[5] == command[0]
        assert (
            command[6] == "-B"
            and Path(command[7]).resolve() == Path(module_file).resolve()
        )
        assert command[8:10] == ["--sdk-child", "--job"]
        assert kwargs["start_new_session"] is True and kwargs["shell"] is False
        return process

    receipt = W.run_job(
        value["job_path"],
        popen=popen,
        gpu_reader=reader,
        identify=identify or (lambda pid: {"pid": pid, "start_ticks": 1}),
        cleanup=cleanup or (lambda _identity, _process: True),
    )
    return receipt, process


def test_child_and_parent_accept_only_cleanup_complete_four_frame_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    _install_sdk_contract(monkeypatch)
    monkeypatch.setattr(W.SMOKE, "verify_report", lambda *_args: _capture_report())
    receipt, _process = _run_parent(
        value,
        reader=Reader(tmp_path),
        action=lambda: W.sdk_child(
            value["job_path"], app_factory=_app_factory, run_sdk=_run_sdk
        ),
    )
    assert receipt["completed"] is True
    assert receipt["cleanup_verified"] is True and receipt["gpu_after"] == []
    assert receipt["renderer_smoke_only"] is True
    result = W.verify_result(value)
    assert result["capture_smoke_completed"] is True
    assert result["sdk"]["prepared"]["approved"] is True


@pytest.mark.parametrize(
    "corruption", ["sealed", "journal", "capture", "result", "failure", "shutdown"]
)
def test_verifier_rejects_tampered_terminal_or_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    value = _validated(tmp_path, monkeypatch)
    _install_sdk_contract(monkeypatch)
    monkeypatch.setattr(W.SMOKE, "verify_report", lambda *_args: _capture_report())
    assert (
        W.sdk_child(value["job_path"], app_factory=_app_factory, run_sdk=_run_sdk) == 0
    )
    output = value["output_directory"]
    result_path = output / "result.json"
    result = W._read(result_path)
    if corruption == "sealed":
        report = W._read(output / W.REPORT_NAME)
        report["lifecycle"]["writer_detached"] = False
        (output / W.REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    elif corruption == "journal":
        (value["job_path"].parent / "job.atomic-000000.json").unlink()
    elif corruption == "capture":
        result["sdk"]["capture_report"]["status"] = "hold"
        result_path.write_text(json.dumps(result), encoding="utf-8")
    elif corruption == "result":
        result["job_sha256"] = "f" * 64
        result_path.write_text(json.dumps(result), encoding="utf-8")
    elif corruption == "failure":
        W.JOB.write_new(output / "failure.json", {})
    else:
        (tmp_path / W.SHUTDOWN.SIDECAR).unlink()
    with pytest.raises(ValueError):
        W.verify_result(value)


def test_exit_zero_without_terminal_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    receipt, _process = _run_parent(value, reader=Reader(tmp_path), action=lambda: 0)
    assert receipt["exit_code"] == 0 and receipt["completed"] is False
    assert receipt["cleanup_verified"] is True and receipt["gpu_after"] == []


def test_foreign_gpu_blocks_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    calls: list[str] = []
    receipt = W.run_job(
        value["job_path"],
        gpu_reader=Reader(tmp_path, pids=[999]),
        popen=lambda *_args, **_kwargs: calls.append("launch"),
    )
    assert calls == [] and receipt["completed"] is False
    assert receipt["gpu_before"] == [999]


def test_identity_failure_never_sends_go_and_reaps_waiting_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    cleaned: list[Any] = []
    monkeypatch.setattr(
        W.GATE, "_cleanup_waiting_gate", lambda process: cleaned.append(process) or True
    )

    def identify(_pid: int) -> Any:
        raise ValueError("not owned")

    receipt, process = _run_parent(
        value, reader=Reader(tmp_path), action=lambda: 0, identify=identify
    )
    assert cleaned == [process] and process.stdin.getvalue() == b""
    assert receipt["completed"] is False and receipt["cleanup_verified"] is True


def test_timeout_cleans_attested_group_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    cleaned: list[Any] = []
    receipt, _process = _run_parent(
        value,
        action=lambda: 0,
        reader=Reader(tmp_path),
        timeout=True,
        cleanup=lambda identity, _process: cleaned.append(identity) or True,
    )
    assert cleaned == [{"pid": 12345, "start_ticks": 1}]
    assert receipt["timed_out"] is True and receipt["completed"] is False


def test_job_mutation_after_child_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)

    def action() -> int:
        value["job_path"].write_text('{"changed":true}', encoding="utf-8")
        return 0

    receipt, _process = _run_parent(value, action=action, reader=Reader(tmp_path))
    assert receipt["completed"] is False and receipt["error_type"] == "ValueError"


def test_sdk_primary_failure_and_close_failure_are_both_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    _install_sdk_contract(monkeypatch)
    monkeypatch.setattr(W.SMOKE, "verify_report", lambda *_args: _capture_report())

    class PreparedFailure(RuntimeError):
        def __init__(self, envelope: dict[str, Any]) -> None:
            super().__init__("source error")
            self.envelope = envelope

    def fail(request: Any, app: Any, *, seal_report: Any) -> Any:
        envelope = _run_sdk(request, app, seal_report=seal_report)
        raise PreparedFailure(envelope)

    assert (
        W.sdk_child(
            value["job_path"],
            run_sdk=fail,
            app_factory=lambda v: _app_factory(
                v, close_error=ValueError("close error")
            ),
        )
        == 1
    )
    failure = W._read(value["output_directory"] / "failure.json")
    shutdown = W._read(value["output_directory"] / "shutdown_failure.json")
    assert failure["error_type"] == "PreparedFailure"
    assert failure["sdk"]["approved"] is True
    assert shutdown["error_type"] == "ValueError"
    assert not (value["output_directory"] / "result.json").exists()


def test_leaf_paths_cannot_escape_bound_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    _install_sdk_contract(monkeypatch)
    monkeypatch.setattr(W.SMOKE, "verify_report", lambda *_args: _capture_report())
    envelope = _envelope(value, sealed=True)
    envelope["artifacts"]["report"] = "../smoke-report.json"
    assert W._successful_sdk_envelope(value, envelope, sealed=True) is False


@pytest.mark.parametrize(
    "mutation",
    [
        "renderer_empty",
        "renderer_profile",
        "fxaa_policy_missing",
        "setup_empty",
        "physics",
        "extension",
        "registrations",
        "raw_count",
        "drain",
        "phase_order",
        "poses",
        "journal_leaf",
        "report_leaf",
        "audit_leaf",
        "preparation_none",
        "bundles_empty",
        "bundle_report_mismatch",
        "marker_99",
        "raw_frame",
        "raw_index",
        "callback_window",
    ],
)
def test_envelope_rejects_malformed_setup_renderer_or_journal_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    value = _validated(tmp_path, monkeypatch)
    _install_sdk_contract(monkeypatch)
    monkeypatch.setattr(W.SMOKE, "verify_report", lambda *_args: _capture_report())
    envelope = _envelope(value, sealed=True)
    if mutation == "renderer_empty":
        envelope["renderer"] = {}
    elif mutation == "renderer_profile":
        envelope["renderer"]["expected_capture_profile"]["rt_subframes"] = 2
    elif mutation == "fxaa_policy_missing":
        value["job"]["source"] = {"files_lf": {W.FXAA.SOURCE_FILE: "a" * 64}}
    elif mutation == "setup_empty":
        envelope["setup_evidence"] = {}
    elif mutation == "physics":
        envelope["setup_evidence"]["setup_physics_before_after"]["after"] = {"state": 1}
    elif mutation == "extension":
        envelope["setup_evidence"]["extension"]["app_update_completed"] = False
    elif mutation == "registrations":
        envelope["setup_evidence"]["registrations"]["IsaacReadTimes"] = False
    elif mutation == "raw_count":
        envelope["setup_evidence"]["raw_callback_count"] = 4
    elif mutation == "drain":
        envelope["setup_evidence"]["complete_drain_proven"] = True
    elif mutation == "phase_order":
        envelope["setup_evidence"]["phase_order"].reverse()
    elif mutation == "poses":
        envelope["setup_evidence"]["camera_poses"][3]["eye_world_m"] = [3.0, 3.0, 2.5]
    elif mutation == "journal_leaf":
        envelope["artifacts"]["journals"][2]["json"] = "job.atomic-000000.json"
    elif mutation == "report_leaf":
        envelope["artifacts"]["report"] = "other.smoke-report.json"
    elif mutation == "audit_leaf":
        envelope["artifacts"]["audits"]["audit_rgb_2"] = "audit-2.rgb"
    elif mutation == "preparation_none":
        envelope["prepared"]["preparation"] = None
    elif mutation == "bundles_empty":
        envelope["prepared"]["bundles"] = []
    elif mutation == "bundle_report_mismatch":
        envelope["prepared"]["bundles"][0]["ordinal"] = 99
    elif mutation == "marker_99":
        envelope["prepared"]["arm_sources"][1]["marker_code"] = 99
    elif mutation == "raw_frame":
        envelope["prepared"]["arm_sources"][2]["frame_number"] = 99
    elif mutation == "raw_index":
        envelope["prepared"]["arm_sources"][3]["raw_index"] = 2
    else:
        envelope["prepared"]["callbacks"][3]["window"] = "arming0"
    assert W._successful_sdk_envelope(value, envelope, sealed=True) is False


def test_actual_sdk_fixture_round_trips_through_parent_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = importlib.import_module("vision_prepared_sdk_v1")
    fixture = importlib.import_module("test_vision_prepared_sdk_v1")
    value = _validated(tmp_path, monkeypatch)
    value["output_directory"].mkdir()
    log: list[str] = []
    dependencies = fixture._dependencies(log)
    camera = W.RECORD.fixed_profile()["camera"]
    dependencies["renderer_readback"] = lambda *_args: {
        "attached_hydra_engine_names": ["fake-hydra"],
        "carb_settings": dict.fromkeys(W.READY.CARB_SETTING_KEYS),
        "expected_capture_profile": {
            key: camera[key]
            for key in (
                "capture_primitive",
                "rt_subframes",
                "pause_timeline",
                "delta_time_s",
            )
        },
        "renderer_accumulation_reset": "omni.usd.UsdContext.reset_renderer_accumulation",
    }
    monkeypatch.setattr(sdk.R, "WIDTH", fixture.WIDTH)
    monkeypatch.setattr(sdk.R, "HEIGHT", fixture.HEIGHT)
    sealed: list[dict[str, Any]] = []
    result = sdk.run_prepared_smoke(
        value,
        fixture._App(log),
        seal_report=lambda report: sealed.append(dict(report)),
        dependencies=dependencies,
    )
    assert W._successful_sdk_envelope(value, sealed[0], sealed=False) is True
    assert W._successful_sdk_envelope(value, result, sealed=True) is True


@pytest.mark.skipif(W.os.name == "posix", reason="Windows 실제 실행 guard 회귀")
def test_default_launcher_cannot_bypass_owned_linux_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="소유 Linux process group"):
        W.run_job(value["job_path"])
