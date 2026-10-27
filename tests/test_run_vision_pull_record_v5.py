from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", Path(__file__).parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import run_vision_pull_record_v5 as RUN  # pyright: ignore[reportMissingImports]
import test_run_vision_pull_record_v2 as V2_FIXTURE  # pyright: ignore[reportMissingImports]
import test_vision_annotator_pull_checkpoint_v2 as AABA_FIXTURE  # pyright: ignore[reportMissingImports]
import vision_play_worker_v1 as PLAY  # pyright: ignore[reportMissingImports]


class Settings:
    def __init__(self) -> None:
        self.value = False

    def get_as_bool(self, _name: str) -> bool:
        return self.value

    def set_bool(self, _name: str, value: bool) -> None:
        self.value = value


class RawApp:
    def __init__(self) -> None:
        self.close_calls: list[dict[str, Any]] = []

    def update(self) -> None:
        return None

    def close(self, **kwargs: Any) -> None:
        self.close_calls.append(dict(kwargs))


def _install_play_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLAY.run_real이 output 소유 지점까지 가는 데 필요한 SDK 표면만 설치한다."""
    gym = ModuleType("gymnasium")
    gym.make = lambda *_args, **_kwargs: object()  # type: ignore[attr-defined]
    tasks = ModuleType("isaaclab_tasks")
    task_utils = ModuleType("isaaclab_tasks.utils")
    env_cfg = SimpleNamespace(
        seed=None,
        actions=SimpleNamespace(joint_pos=SimpleNamespace(scale=None)),
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                heading_command=None,
                rel_standing_envs=None,
                resampling_time_range=None,
                ranges=SimpleNamespace(lin_vel_x=None, lin_vel_y=None, ang_vel_z=None),
            )
        ),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    agent_cfg = SimpleNamespace(
        device=None,
        clip_actions=1.0,
        to_dict=dict,
    )
    task_utils.parse_env_cfg = lambda *_args, **_kwargs: env_cfg  # type: ignore[attr-defined]
    task_utils.load_cfg_from_registry = lambda *_args, **_kwargs: agent_cfg  # type: ignore[attr-defined]
    tasks.utils = task_utils  # type: ignore[attr-defined]

    base = SimpleNamespace(device="cuda:0", scene={"robot": object()})

    class Wrapper:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.unwrapped = base
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class Runner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def load(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def get_inference_policy(self, **_kwargs: Any) -> object:
            return object()

    rl = ModuleType("isaaclab_rl")
    rl_rsl = ModuleType("isaaclab_rl.rsl_rl")
    rl_rsl.RslRlVecEnvWrapper = Wrapper  # type: ignore[attr-defined]
    rl.rsl_rl = rl_rsl  # type: ignore[attr-defined]
    rsl = ModuleType("rsl_rl")
    runners = ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = Runner  # type: ignore[attr-defined]
    rsl.runners = runners  # type: ignore[attr-defined]

    omni = ModuleType("omni")
    replicator = ModuleType("omni.replicator")
    rep = ModuleType("omni.replicator.core")
    usd = ModuleType("omni.usd")
    replicator.core = rep  # type: ignore[attr-defined]
    omni.replicator = replicator  # type: ignore[attr-defined]
    omni.usd = usd  # type: ignore[attr-defined]
    pxr = ModuleType("pxr")
    pxr.UsdGeom = object()  # type: ignore[attr-defined]

    for name, module in {
        "gymnasium": gym,
        "isaaclab_tasks": tasks,
        "isaaclab_tasks.utils": task_utils,
        "isaaclab_rl": rl,
        "isaaclab_rl.rsl_rl": rl_rsl,
        "rsl_rl": rsl,
        "rsl_rl.runners": runners,
        "omni": omni,
        "omni.replicator": replicator,
        "omni.replicator.core": rep,
        "omni.usd": usd,
        "pxr": pxr,
        "torch": ModuleType("torch"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _emit_aaba(adapted: dict[str, Any], app: object, *, seal_report: Any) -> dict[str, Any]:
    worker = AABA_FIXTURE._worker()
    worker["policy_probe"]["checkpoint_sha256"] = adapted["checkpoint_sha256"]
    worker["lifecycle"].update(
        {"render_product_destroyed": True, "environment_closed": True}
    )
    return RUN.AABA_CHECKPOINT.run_annotator_pull_diagnostic(
        adapted,
        app,
        run_worker=lambda *_args, **_kwargs: worker,
        seal_report=seal_report,
    )


def _validated(tmp_path: Path) -> dict[str, Any]:
    value = V2_FIXTURE._validated(tmp_path)
    value["job"]["profile"] = RECORD.fixed_profile()
    return value


def _actual_job5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """JOB5의 historical-output 분기를 유지한 최소 실제 job을 만든다."""
    repo = tmp_path / "repo"
    lab = tmp_path / "isaaclab"
    stage = tmp_path / "stage"
    for folder in (repo, lab, stage):
        folder.mkdir()
    checkpoint = lab / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    resource = tmp_path / "resource.json"
    resource.write_text("{}", encoding="utf-8")
    python = tmp_path / "python"
    python.write_bytes(b"python")
    row = {
        "logical_id": RUN.JOB.CHECKPOINT_IDS[0],
        "isaaclab_relative": "checkpoint.pt",
        "sha256": RUN.JOB._sha(checkpoint),
    }
    protocol = {
        "source_commit": "a" * 40,
        "source_sha256_lf": {"bench/source.py": "b" * 64},
        "source_protocol": {"sha256_lf": "c" * 64},
        "_source": {"protocol_id": "source"},
    }
    job = {
        "schema": RUN.JOB.JOB_SCHEMA,
        "mode": RUN.JOB.MODE,
        "job_id": f"{RUN.JOB.PROTOCOL_ID}-{row['logical_id']}",
        "source": {
            "repo_root": str(repo.resolve()),
            "commit": protocol["source_commit"],
            "files_lf": protocol["source_sha256_lf"],
        },
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256_lf": RUN.JOB._sha(protocol_path, lf=True),
        },
        "resource_proof": {
            "path": str(resource.resolve()),
            "sha256": RUN.JOB._sha(resource),
        },
        "isaaclab": str(lab.resolve()),
        "python": str(python.resolve()),
        "output_directory": str((stage / "output").resolve()),
        "profile": RECORD.fixed_profile(),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": row["sha256"],
            "logical_id": row["logical_id"],
        },
    }
    job_path = stage / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr(RUN.JOB, "validate_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(
        RUN.JOB.RECORD_JOB,
        "_checkpoint",
        lambda *_args: (row, checkpoint.resolve()),
    )
    monkeypatch.setattr(
        RUN.JOB.COMMON,
        "_resource",
        lambda *_args: {"resource_id": "resource", "gpu_uuid": "GPU-1"},
    )
    monkeypatch.setattr(RUN.JOB.COMMON, "_python_executable", lambda *_args: None)
    monkeypatch.setattr(RUN.JOB.COMMON, "_verify_runtime_sources", lambda *_args: None)
    monkeypatch.setattr(RUN.REMOTE, "verify_observation_reference", lambda *_args: None)
    monkeypatch.setattr(RUN.AABA_SHUTDOWN, "verify_process_exit", lambda *_args: {})
    validated = RUN.JOB.validate_job(job_path, require_fresh=False)
    aaba = stage / "aaba-output"
    aaba.mkdir()
    (aaba / RUN.AABA_REPORT_NAME).write_text(
        json.dumps({"diagnostic_complete": True}), encoding="utf-8"
    )
    process_exit = stage / RUN.AABA_SHUTDOWN.PROCESS_EXIT_NAME
    process_exit.write_text("{}", encoding="utf-8")
    observation = {"path": str(stage / "gpu.json"), "sha256": "d" * 64}
    RUN.JOB.write_new(
        stage / RUN.AABA_CONTINUATION_NAME,
        RUN._continuation_body(validated, {"diagnostic_complete": True}, observation),
    )
    return validated


def test_aaba_child_uses_phase_shutdown_and_exits_zero(tmp_path: Path, monkeypatch) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    raw = RawApp()
    settings = Settings()
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda _path: validated)
    monkeypatch.setattr(
        RUN.AABA_SHUTDOWN,
        "sdk_app_factory",
        lambda adapted: RUN.AABA_SHUTDOWN.AppProxy(raw, adapted, settings),
    )

    assert RUN.aaba_child(validated["job_path"], run_diagnostic=_emit_aaba) == 0

    assert raw.close_calls == [{"wait_for_replicator": False}]
    assert settings.value is True
    assert RUN._verify_aaba_terminal(validated)["diagnostic_complete"] is True
    assert (tmp_path / RUN.AABA_SHUTDOWN.POLICY_NAME).is_file()
    assert not (tmp_path / "shutdown-policy.json").exists()


def test_parent_does_not_launch_record_without_aaba_shutdown_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    validated["job"].update({"python": sys.executable, "isaaclab": str(tmp_path)})
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda *_args, **_kwargs: validated)
    launched: list[str] = []

    class Process:
        pid = 123
        stdin = io.BytesIO()

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            adapted = RUN._aaba_validated(validated)
            _emit_aaba(adapted, object(), seal_report=lambda value: RUN.JOB.write_new(
                Path(adapted["output_directory"]) / RUN.AABA_REPORT_NAME, value
            ))
            return 0

    def popen(command: list[str], **_kwargs: Any) -> Process:
        phase = "aaba" if "--aaba-child" in command else "record"
        launched.append(phase)
        return Process()

    receipt = RUN.run_job(
        validated["job_path"],
        popen=popen,
        gpu_reader=list,
        identify=lambda _pid: {"pid": 123, "start_ticks": 1},
        cleanup=lambda *_args: True,
    )

    assert launched == ["aaba"]
    assert receipt["completed"] is False
    assert receipt["error_type"] in {"FileNotFoundError", "ValueError"}
    assert not (tmp_path / "record-launch.json").exists()


def test_parent_requires_exit_cleanup_gpu_and_policy_before_record(
    tmp_path: Path, monkeypatch
) -> None:
    validated = _validated(tmp_path)
    validated["checkpoint_path"].write_bytes(b"checkpoint")
    validated["job"].update({"python": sys.executable, "isaaclab": str(tmp_path)})
    monkeypatch.setattr(RUN.JOB, "validate_job", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(
        RUN.BASE,
        "_gpu_reference",
        lambda *_args: {"path": "gpu", "sha256": "c" * 64},
    )
    monkeypatch.setattr(RUN.REMOTE, "verify_observation_reference", lambda *_args: None)
    monkeypatch.setattr(
        RUN.V1,
        "verify_terminal",
        lambda *_args: {"completed": True, "technical_recording_complete": True},
    )
    launched: list[str] = []
    cleaned: list[str] = []

    class Process:
        pid = 123

        def __init__(self, phase: str) -> None:
            self.phase = phase
            self.stdin = io.BytesIO()

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            if self.phase == "aaba":
                adapted = RUN._aaba_validated(validated)
                raw = RawApp()
                proxy = RUN.AABA_SHUTDOWN.AppProxy(raw, adapted, Settings())
                _emit_aaba(
                    adapted,
                    proxy,
                    seal_report=lambda value: RUN.JOB.write_new(
                        Path(adapted["output_directory"]) / RUN.AABA_REPORT_NAME, value
                    ),
                )
                proxy.close()
            else:
                output = Path(validated["output_directory"])
                assert not output.exists()
                output.mkdir(parents=True, exist_ok=False)
                RUN.JOB.write_new(
                    output / "result.json",
                    {"completed": True, "technical_recording_complete": True},
                )
            return 0

    def popen(command: list[str], **_kwargs: Any) -> Process:
        phase = "aaba" if "--aaba-child" in command else "record"
        launched.append(phase)
        return Process(phase)

    def cleanup(_launch: Any, process: Process) -> bool:
        cleaned.append(process.phase)
        return True

    receipt = RUN.run_job(
        validated["job_path"],
        popen=popen,
        gpu_reader=list,
        identify=lambda _pid: {"pid": 123, "start_ticks": 1},
        cleanup=cleanup,
    )

    assert launched == cleaned == ["aaba", "record"]
    assert receipt["completed"] is True
    assert receipt["aaba_receipt"]["shutdown_policy"]["file"] == RUN.AABA_SHUTDOWN.POLICY_NAME
    assert receipt["aaba_receipt"]["close_returned"]["file"] == RUN.AABA_SHUTDOWN.CLOSE_RETURNED_NAME
    assert receipt["phases"][0]["exit_code"] == 0
    assert receipt["phases"][0]["cleanup_verified"] is True
    assert receipt["phases"][0]["gpu_after"] == []


def test_runner_imports_job5_without_fallback() -> None:
    assert RUN.__file__ is not None
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    assert "import vision_pull_record_job_v5 as JOB" in source
    assert "ImportError" not in source
    assert "vision_pull_record_job_v2 as JOB" not in source
    assert "vision_pull_record_job_v1 as JOB" not in source
    assert RUN.EXECUTION_SCHEMA == "vision_pull_record_execution_v3"
    assert RUN.JOB.LIMITS["max_wall_s_per_policy"] == 600
    assert RUN.POSTFLIGHT_RESERVE_S == 200.0
    assert (
        RUN.JOB.LIMITS["max_wall_s_per_policy"] - RUN.POSTFLIGHT_RESERVE_S
        == RUN.JOB.LIMITS["max_child_s_per_policy"]
    )


def test_actual_job5_sdk_child_leaves_output_creation_to_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _actual_job5(tmp_path, monkeypatch)
    output = Path(validated["output_directory"])
    observed: list[bool] = []

    def worker(
        value: dict[str, Any], _app: object, *, seal_report: Any
    ) -> dict[str, Any]:
        assert value["output_directory"] == output
        observed.append(output.exists())
        # vision_play_worker_v1.run_real의 단일 output 소유 동작을 같은 경계에서 실행한다.
        output.mkdir(parents=True, exist_ok=False)
        report = {"technical_recording_complete": True}
        seal_report(report)
        return report

    monkeypatch.setattr(RUN.V1, "_result_body", lambda *_args: {"completed": True})
    assert RUN.sdk_child(
        validated["job_path"],
        app_factory=lambda _value: SimpleApp(),
        run_worker=worker,
    ) == 0
    assert observed == [False]
    assert (output / RUN.REPORT_NAME).is_file()
    assert (output / "result.json").is_file()


def test_sdk_child_reaches_actual_play_output_claim_without_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _actual_job5(tmp_path, monkeypatch)
    output = Path(validated["output_directory"])
    _install_play_imports(monkeypatch)

    class EncoderReached(RuntimeError):
        pass

    def encoder(path: Path) -> object:
        assert path == output / "same_play.mp4"
        assert output.is_dir()
        raise EncoderReached("PLAY encoder boundary reached")

    monkeypatch.setattr(PLAY.R, "RawRgbFfmpegEncoder", encoder)

    assert RUN.sdk_child(
        validated["job_path"],
        app_factory=lambda _value: SimpleApp(),
        run_worker=lambda value, app, **_kwargs: PLAY.run_real(value, app)[0],
    ) == 1
    primary = json.loads((output / RUN.V1.PRIMARY_ERROR_NAME).read_text(encoding="utf-8"))
    assert primary["error_type"].endswith("EncoderReached")
    assert primary["error_message"] == "PLAY encoder boundary reached"
    assert output.is_dir()


@pytest.mark.parametrize("previous_result", [False, True])
def test_actual_job5_sdk_child_rejects_preexisting_record_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous_result: bool
) -> None:
    validated = _actual_job5(tmp_path, monkeypatch)
    output = Path(validated["output_directory"])
    output.mkdir()
    if previous_result:
        (output / "result.json").write_text("{}", encoding="utf-8")
    called = False

    def worker(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"technical_recording_complete": True}

    with pytest.raises(FileExistsError, match="PLAY 실행 전에"):
        RUN.sdk_child(
            validated["job_path"],
            app_factory=lambda _value: SimpleApp(),
            run_worker=worker,
        )
    assert called is False


def test_actual_job5_rejects_symlinked_record_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _actual_job5(tmp_path, monkeypatch)
    output = Path(validated["output_directory"])
    target = tmp_path / "redirected-output"
    target.mkdir()
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError:
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: self == output or original_is_symlink(self),
        )
    with pytest.raises((ValueError, FileExistsError, FileNotFoundError)):
        RUN.sdk_child(validated["job_path"], app_factory=lambda _value: SimpleApp())


class SimpleApp:
    def close(self) -> None:
        return None


def test_policy_failure_poll_stops_hung_close_and_worker_error_keeps_priority(
    tmp_path: Path,
) -> None:
    validated = _validated(tmp_path)
    adapted = RUN._aaba_validated(validated)
    failure = {
        "schema": "vision_aaba_close_policy_failure_v1",
        "job_sha256": validated["job_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "error_type": "ValueError",
        "error_message": "terminal-mismatch",
    }
    RUN.JOB.write_new(tmp_path / RUN.AABA_SHUTDOWN.POLICY_FAILURE_NAME, failure)

    class Hung:
        def __init__(self) -> None:
            self.args = ["aaba"]

        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired(self.args, timeout)

    with pytest.raises(RUN.ShutdownPolicyFailure, match="terminal-mismatch"):
        RUN._wait_for_child(
            Hung(),
            timeout_s=2,
            phase_name="aaba",
            stage=tmp_path,
            poll_s=0.01,
            validated=validated,
        )

    output = Path(adapted["output_directory"])
    output.mkdir()
    first = output / RUN.AABA_PRIMARY_NAME
    RUN.JOB.write_new(
        first,
        {
            "schema": RUN.PRIMARY_ERROR_SCHEMA,
            "error_type": "builtins.FileNotFoundError",
            "error_message": "worker-first",
            "traceback": "trace",
        },
    )
    RUN.JOB.write_new(
        output / RUN.AABA_FAILURE_NAME,
        {
            "schema": "vision_pull_record_aaba_failure_v2",
            "completed": False,
            "first_cause": RUN._descriptor(first),
        },
    )
    with pytest.raises(RUN.ChildFailure, match="worker-first"):
        RUN._wait_for_child(
            Hung(),
            timeout_s=2,
            phase_name="aaba",
            stage=tmp_path,
            poll_s=0.01,
            validated=validated,
        )
