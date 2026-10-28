from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_agentic_play_record_v1 as RUN  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

J = RUN.JOB
PREVIOUS = RUN.PREVIOUS
SIDECAR = RUN.SIDECAR
R = SIDECAR.R
FRAMES = J.TECHNICAL_LIMITS["frames_per_policy"]


# --- 계약 교체와 자식 명령 -------------------------------------------------------------


def test_job_contract_installs_new_job_and_v3_worker_then_restores() -> None:
    before = (PREVIOUS.JOB, PREVIOUS.WORKER, PREVIOUS.V1.WORKER, PREVIOUS.REPORT_NAME)
    with RUN._job_contract():
        assert PREVIOUS.JOB is J
        assert PREVIOUS.WORKER is RUN.WORKER and PREVIOUS.V1.WORKER is RUN.WORKER
        assert PREVIOUS.POSTFLIGHT_RESERVE_S == 200.0
    assert (PREVIOUS.JOB, PREVIOUS.WORKER, PREVIOUS.V1.WORKER, PREVIOUS.REPORT_NAME) == before


def _run5() -> str:
    return str(Path(PREVIOUS.__file__).resolve())


def test_child_launcher_rewrites_only_run5_entry(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    job = tmp_path / "job.json"
    launch = RUN._child_launcher(lambda command, **_kwargs: calls.append(command), job)
    launch(["py", "-I", "-S", "-c", "gate", "py", "-B", _run5(), "--sdk-child", "--job", str(job.absolute())])
    assert calls[0][7] == str(Path(RUN.__file__).resolve())


@pytest.mark.parametrize("case", ["v3-entry", "sidecar-entry", "no-run5", "other-job", "flag", "two-run5"])
def test_child_launcher_never_runs_f1_or_foreign_entry(tmp_path: Path, case: str) -> None:
    job = str((tmp_path / "job.json").absolute())
    command = ["py", "-B", _run5(), "--sdk-child", "--job", job]
    if case == "v3-entry":
        command = ["py", "-B", str(ROOT / "bench/run_vision_transferred_pull_record_v3.py"), *command[2:]]
        command.insert(2, _run5())
    elif case == "sidecar-entry":
        command.append(str(ROOT / "bench/record_agentic_state_sidecar_v1.py"))
    elif case == "no-run5":
        command[2] = "other.py"
    elif case == "other-job":
        command[-1] = str(tmp_path / "f1" / "job.json")
    elif case == "flag":
        command[3] = "--record-child"
    else:
        command.insert(2, _run5())
    launch = RUN._child_launcher(lambda *_args, **_kwargs: pytest.fail("실행되면 안 된다"), tmp_path / "job.json")
    with pytest.raises(ValueError):
        launch(command)


def test_run_job_rejects_f1_job_and_missing_admission_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = FIX.build_world(tmp_path, monkeypatch)

    def never(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("child가 실행되면 안 된다")

    f1 = FIX.write_job(world, schema="vision_transferred_pull_job_v3")
    with pytest.raises(ValueError, match="schema"):
        RUN.run_job(f1, popen=never, gpu_reader=list)
    missing = FIX.write_job(world, stage_name="no-proof",
                            resource_proof={"path": str(tmp_path / "absent.json"), "sha256": "0" * 64})
    with pytest.raises(ValueError):
        RUN.run_job(missing, popen=never, gpu_reader=list)
    no_lease = FIX.write_proof(tmp_path, world.protocol_path, observed=world.now, lease=FIX.OMIT, name="nl.json")
    unleased = FIX.write_job(world, stage_name="no-lease",
                             resource_proof={"path": str(no_lease), "sha256": J._sha(no_lease)})
    with pytest.raises(ValueError, match="필드"):
        RUN.run_job(unleased, popen=never, gpu_reader=list)
    world.lease_stack.close()  # 임대가 끝난 뒤의 job은 child 실행 전에 거부된다
    with pytest.raises(J.LeaseError):
        RUN.run_job(FIX.write_job(world, stage_name="released"), popen=never, gpu_reader=list)
    assert not (tmp_path / F2_STAGE / "execution.json").exists()


@pytest.mark.parametrize("swap", [None, "popen", "gpu_reader", "identify", "cleanup"])
def test_live_requires_every_production_callable(swap: str | None) -> None:
    values: dict[str, Any] = {"popen": RUN.PRODUCTION["popen"], "gpu_reader": None,
                              "identify": RUN.PRODUCTION["identify"], "cleanup": RUN.PRODUCTION["cleanup"]}
    if swap == "gpu_reader":
        values["gpu_reader"] = list
    elif swap is not None:
        values[swap] = lambda *_args, **_kwargs: None
    assert RUN.live_arguments(**values) is (swap is None)


F2_STAGE = FIX.F2_ID


@pytest.mark.skipif(sys.platform != "win32", reason="Windows에서 기본 Popen 거부를 확인한다")
def test_default_popen_is_refused_off_posix(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Linux"):
        RUN.run_job(tmp_path / "job.json")


def test_child_main_rejects_non_agentic_job_and_delegates_aaba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = FIX.build_world(tmp_path, monkeypatch)
    f1 = FIX.write_job(world, schema="vision_transferred_pull_job_v3")
    with pytest.raises(ValueError, match="agentic play job"):
        RUN.child_main("--sdk-child", f1)
    seen: list[Any] = []

    def aaba(job_path: Path, **_kwargs: Any) -> int:
        seen.append((job_path, PREVIOUS.JOB is J))
        return 3

    monkeypatch.setattr(PREVIOUS, "aaba_child", aaba)
    assert RUN.main(["--aaba-child", "--job", str(f1)]) == 3
    assert seen == [(f1, True)]
    with pytest.raises(ValueError):
        RUN.child_main("--other", f1)


# --- RUN5 실제 부모·자식 흐름 (fake process·app·Play) ----------------------------------------


class FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __getitem__(self, index: int) -> FakeTensor:
        return FakeTensor(self.value[index])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> Any:
        return copy.deepcopy(self.value)

    def any(self) -> FakeTensor:
        return FakeTensor(any(bool(item) for item in self.value))

    def item(self) -> Any:
        return self.value


def _stock_clip(self: Any, effort: Any) -> Any:
    return effort


_stock_clip.__module__ = "isaaclab.actuators.actuator_pd"
_stock_clip.__qualname__ = "DCMotor._clip_effort"


class FakeDCMotor:
    _clip_effort = _stock_clip

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
                                   stiffness=25.0, damping=0.5)
        self.joint_names = ["FL_hip_joint", "FR_hip_joint"]


def _base() -> Any:
    data = SimpleNamespace(
        joint_names=["FL_hip_joint", "FR_hip_joint"],
        joint_pos=FakeTensor([[0.1, 0.2]]), joint_vel=FakeTensor([[0.0, 0.0]]),
        joint_pos_target=FakeTensor([[0.0, 0.0]]), computed_torque=FakeTensor([[1.0, 2.0]]),
        applied_torque=FakeTensor([[1.0, 2.0]]),
    )
    contact = SimpleNamespace(
        cfg=SimpleNamespace(force_threshold=1.0, track_air_time=True), body_names=["base", "FL_foot"],
        _is_outdated=FakeTensor([False]),
        data=SimpleNamespace(net_forces_w=FakeTensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]]),
                             current_contact_time=FakeTensor([[0.0, 0.1]]),
                             current_air_time=FakeTensor([[0.0, 0.0]])),
    )
    robot = SimpleNamespace(data=data, actuators={"base_legs": FakeDCMotor()})
    return SimpleNamespace(scene={"robot": robot, "contact_forces": contact},
                           sim=SimpleNamespace(current_time=0.01), common_step_counter=0)


class FakeProcess:
    def __init__(self, code: int) -> None:
        self.code = code
        self.pid = 4242
        self.args: list[str] = []
        self.stdin = io.BytesIO()

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.code


class Chain:
    """실제 RUN5 run_job·sdk_child·새 JOB·sidecar·playback wrapper를 지나는 CPU 흐름 기록."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_frame: int | None = None,
                 aaba_code: int = 0, drop_lease_after_aaba: bool = False) -> None:
        self.events: list[Any] = []
        self.commands: list[list[str]] = []
        self.cleanups = 0
        self.fail_frame = fail_frame
        self.aaba_code = aaba_code
        self.drop_lease_after_aaba = drop_lease_after_aaba
        self.monkeypatch = monkeypatch
        self.world = FIX.build_world(tmp_path, monkeypatch)
        self.patch(monkeypatch)

    def patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shutdown = PREVIOUS.AABA_SHUTDOWN
        monkeypatch.setattr(PREVIOUS, "_verify_aaba_terminal", lambda validated: {})
        monkeypatch.setattr(PREVIOUS.BASE, "_gpu_reference",
                            lambda reader, validated: {"path": "fixture", "sha256": "0" * 64})
        monkeypatch.setattr(PREVIOUS, "verify_aaba", lambda validated, observation: {"diagnostic_complete": True})
        monkeypatch.setattr(PREVIOUS, "_verify_continuation", lambda validated: {})

        def process_exit(adapted: Any, **_kwargs: Any) -> None:
            FIX._write(Path(adapted["job_path"]).parent / shutdown.PROCESS_EXIT_NAME, {"fixture": True})

        monkeypatch.setattr(shutdown, "write_process_exit", process_exit)
        original_terminal = PREVIOUS.V1.verify_terminal

        def terminal(validated: Any) -> Any:
            if (Path(validated["output_directory"]) / "result.json").exists():
                return {"completed": True, "technical_recording_complete": True}
            return original_terminal(validated)

        monkeypatch.setattr(PREVIOUS.V1, "verify_terminal", terminal)
        import vision_play_worker_v3 as V3  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(V3, "_sdk_app_factory", lambda validated: self.app(validated))
        monkeypatch.setattr(RUN.WORKER.CAPTURE, "make_capture_factory", lambda validated, app: self.native_factory)
        monkeypatch.setattr(RUN.WORKER, "run_policy_record", self.run_policy_record)
        make = RUN.PLAYBACK.make_run_real
        monkeypatch.setattr(RUN.PLAYBACK, "make_run_real",
                            lambda registration: make(registration, runner=self.play, installer=self.install))

    def app(self, validated: Any) -> Any:
        chain = self
        stage = Path(validated["job_path"]).parent
        sidecar = Path(validated["job"]["sidecar"]["directory"])

        class App:
            def close(self) -> None:
                chain.events.append(("app.close", (sidecar / SIDECAR.MANIFEST_NAME).exists(),
                                     (sidecar / SIDECAR.FAILURE_NAME).exists(),
                                     (stage / "playback-receipt.json").exists()))

        return App()

    def native_factory(self, base: Any, rep: Any, render_product: Any, profile: Any) -> Any:
        def capture(capture_base: Any, capture_rep: Any, annotators: Any, camera_profile: Any,
                    eye: Any, lookat: Any, ordinal: int) -> dict[str, Any]:
            return {"rgb": b"", "ordinal": ordinal}

        return capture, lambda: self.events.append("capture.cleanup")

    def install(self, checked: Any) -> Any:
        if checked["fault_id"] == "NONE":
            return {"fault_id": "NONE", "class_method_changed": False}, lambda: None
        old = FakeDCMotor._clip_effort

        def patched(self: Any, effort: Any) -> Any:
            return effort

        patched.__module__ = "hidden_faults"
        patched.__qualname__ = "_before_env.<locals>._clip_effort"
        FakeDCMotor._clip_effort = patched  # type: ignore[method-assign]

        def restore() -> None:
            FakeDCMotor._clip_effort = old  # type: ignore[method-assign]
            self.events.append("playback.restore")

        return {"fault_id": "F2", "class_method_changed": True}, restore

    def play(self, validated: Any, app: Any, *, camera_capture_factory: Any) -> Any:
        """vision_play_worker_v1.run_real 순서: env 생성 → factory → step → capture."""
        self.events.append(("play.clip", FakeDCMotor._clip_effort.__module__))
        output = Path(validated["output_directory"])
        output.mkdir(parents=True)
        base = _base()
        capture, cleanup = camera_capture_factory(base, "rep", "render_product", {"camera": "fixed"})
        rows = []
        try:
            for index in range(FRAMES):
                if index:
                    base.common_step_counter += 1
                    base.sim.current_time += 0.02
                if index == self.fail_frame:
                    raise RuntimeError("native play failed")
                capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, index)
                rows.append({"frame_index": index, "capture_ordinal": index,
                             "simulation_step": base.common_step_counter,
                             "state": {"simulation_time_s": base.sim.current_time}})
        finally:
            cleanup()
        R.write_new(output / "evidence" / "raw_trace.json", {
            "schema": R.TRACE_SCHEMA, "checkpoint_sha256": validated["checkpoint_sha256"],
            "video_sha256": "c" * 64, "frame_count": len(rows), "records": rows,
        })
        R.write_new(output / "native_capture_manifest.json",
                    {"checkpoint_sha256": validated["checkpoint_sha256"], "frame_count": len(rows)})
        return {}, output / "same_play.mp4"

    def run_policy_record(self, validated: Any, app: Any, *, seal_report: Any, capture_factory: Any,
                          run_real: Any) -> dict[str, Any]:
        factory = capture_factory(validated, app)
        run_real(validated, app, camera_capture_factory=factory)
        report = {"technical_recording_complete": True, "fixture_not_live_evidence": True}
        seal_report(report)
        return report

    def popen(self, command: list[str], **kwargs: Any) -> FakeProcess:
        self.commands.append(list(command))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["shell"] is False
        flag, job = command[8], command[10]
        stage = Path(job).parent
        if flag == "--aaba-child":
            FIX._write(stage / "aaba-output" / PREVIOUS.AABA_REPORT_NAME, {"fixture": True})
            FIX._write(stage / PREVIOUS.AABA_SHUTDOWN.POLICY_NAME, {"fixture": True})
            FIX._write(stage / PREVIOUS.AABA_SHUTDOWN.CLOSE_ENTERED_NAME, {"fixture": True})
            if self.drop_lease_after_aaba:
                self.world.lease_stack.close()  # AABA 뒤 임대 종료: record 직전 재검증이 막아야 한다
            return FakeProcess(self.aaba_code)
        return FakeProcess(RUN.main([flag, "--job", job]))  # 새 entry의 child를 같은 프로세스에서 실행

    def cleanup(self, launch: Any, process: Any) -> bool:
        self.cleanups += 1
        return True

    def identify(self, pid: int) -> dict[str, Any]:
        return {"pid": pid}

    def run(self, identity: str, mode: str = "injected") -> tuple[Path, dict[str, Any]]:
        job = FIX.write_job(self.world, identity)
        if mode == "injected":
            return job, RUN.run_job(job, popen=self.popen, gpu_reader=list,
                                    identify=self.identify, cleanup=self.cleanup)
        # 운영 callable 자리에 fixture를 고정 등록한 뒤, 한 인자만 바꿔 live 판정을 본다.
        popen, identify, cleanup = self.popen, self.identify, self.cleanup
        for key, value in (("popen", popen), ("identify", identify), ("cleanup", cleanup)):
            self.monkeypatch.setitem(RUN.PRODUCTION, key, value)
        self.monkeypatch.setattr(PREVIOUS.REMOTE, "StrictRemoteGpuReader", lambda capture, folder, run: list)
        self.monkeypatch.setattr(PREVIOUS.BASE, "_gpu_query_run", lambda deadline: None)
        if mode == "identify-swapped":
            identify = lambda pid: {"pid": pid}
        elif mode == "cleanup-swapped":
            cleanup = lambda launch, process: self.cleanup(launch, process)
        return job, RUN.run_job(job, popen=popen, identify=identify, cleanup=cleanup)


@pytest.mark.parametrize(("identity", "fault", "clip"), [
    (FIX.F2_ID, "F2", "hidden_faults"),
    (FIX.NONE_ID, "NONE", "isaaclab.actuators.actuator_pd"),
])
def test_real_run5_flow_binds_new_entry_job_sidecar_and_playback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str, fault: str, clip: str
) -> None:
    chain = Chain(tmp_path, monkeypatch)
    job, final = chain.run(identity)
    stage = job.parent
    entry = str(Path(RUN.__file__).resolve())
    assert [command[7] for command in chain.commands] == [entry, entry]
    assert [command[8] for command in chain.commands] == ["--aaba-child", "--sdk-child"]
    assert all(command[4] == PREVIOUS.GATE.OWNED_WORKER_GATE for command in chain.commands)
    assert ("play.clip", clip) in chain.events
    assert chain.events[-1] == ("app.close", True, False, True)  # sidecar·playback receipt가 close 전에 봉인
    assert chain.cleanups == 2
    execution = json.loads((stage / "execution.json").read_text(encoding="utf-8"))
    assert execution["completed"] is True and execution["full_recordings"] == 1
    playback = json.loads((stage / "playback-receipt.json").read_text(encoding="utf-8"))
    assert playback["status"] == "completed" and playback["installation"]["fault_id"] == fault
    assert final["native_technical_complete"] is True
    assert final["sidecar"]["status"] == "sealed" and final["sidecar"]["error_policy"] == SIDECAR.ABORT
    assert final["join"]["joined"] is True and final["join"]["frame_count"] == FRAMES
    assert final["actuator_clip_consistent_with_registration"] is True
    assert final["final_technical_complete"] is True
    assert final["live_evidence"] is False and final["acceptable_as_live_evidence"] is False
    assert final["fault_id"] == fault and final["behavior_label"] is None
    manifest = json.loads((stage.parent / f"{identity}-sidecar" / SIDECAR.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["actuators"]["base_legs"]["clip_effort_function"].startswith(clip)
    assert manifest["source"]["source_id"] == f"agentic-play-20261028a-{identity}"
    assert json.loads((stage / RUN.FINAL_NAME).read_text(encoding="utf-8")) == final
    assert FakeDCMotor._clip_effort is _stock_clip
    lease_id = json.loads(Path(chain.world.lease["path"]).read_text(encoding="utf-8"))["lease_id"]
    assert final["lease"]["active_at_finalize"] is True and final["lease"]["lease_id"] == lease_id
    assert final["lease"]["lock_proof"] == (J.POSIX_LOCK_PROOF if sys.platform != "win32"
                                            else J.FIXTURE_LOCK_PROOF)


def test_lease_loss_between_aaba_and_record_stops_record_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = Chain(tmp_path, monkeypatch, drop_lease_after_aaba=True)
    job, final = chain.run(FIX.F2_ID)
    assert [command[8] for command in chain.commands] == ["--aaba-child"]
    assert chain.cleanups == 1
    execution = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert execution["completed"] is False and execution["error_type"] == "LeaseError"
    assert not (job.parent / "output").exists()
    assert final["lease"]["active_at_finalize"] is False and final["final_technical_complete"] is False


@pytest.mark.parametrize(("mode", "live"), [
    ("production", True), ("identify-swapped", False), ("cleanup-swapped", False),
])
def test_injected_identify_or_cleanup_is_never_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, live: bool
) -> None:
    chain = Chain(tmp_path, monkeypatch)
    _job, final = chain.run(FIX.NONE_ID, mode)
    assert final["final_technical_complete"] is True and final["live_evidence"] is live
    # Windows lock probe는 fixture 증거라 운영 인자여도 live evidence로 승격하지 않는다.
    assert final["acceptable_as_live_evidence"] is (live and final["lease"]["live_lock_proof"])
    if sys.platform == "win32":
        assert final["acceptable_as_live_evidence"] is False


def test_record_child_failure_is_sealed_and_never_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = Chain(tmp_path, monkeypatch, fail_frame=10)
    job, final = chain.run(FIX.F2_ID)
    stage = job.parent
    assert chain.events[-1] == ("app.close", False, True, True)
    assert "playback.restore" in chain.events and FakeDCMotor._clip_effort is _stock_clip
    assert (stage / "output" / "failure.json").exists()
    execution = json.loads((stage / "execution.json").read_text(encoding="utf-8"))
    assert execution["completed"] is False and execution["error_message"]
    assert chain.cleanups == 2
    sidecar_failure = json.loads((stage.parent / f"{FIX.F2_ID}-sidecar" / SIDECAR.FAILURE_NAME)
                                 .read_text(encoding="utf-8"))
    assert sidecar_failure["stage"] == "native_record" and sidecar_failure["records_written"] == 10
    assert final["native_technical_complete"] is False and final["sidecar"]["status"] == "failed"
    assert final["playback_completed"] is False and final["final_technical_complete"] is False


def test_aaba_failure_stops_before_record_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = Chain(tmp_path, monkeypatch, aaba_code=1)
    job, final = chain.run(FIX.F2_ID)
    assert [command[8] for command in chain.commands] == ["--aaba-child"]
    assert chain.cleanups == 1
    execution = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert execution["completed"] is False and "AABA" in execution["error_message"]
    assert final["sidecar"]["status"] == "missing" and final["final_technical_complete"] is False


def test_final_requires_registered_clip_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = Chain(tmp_path, monkeypatch)
    monkeypatch.setattr(chain, "install", lambda checked: ({"fault_id": "F2", "class_method_changed": True},
                                                            lambda: None))
    _job, final = chain.run(FIX.F2_ID)
    assert final["native_technical_complete"] is True and final["join"]["joined"] is True
    assert final["actuator_clip_consistent_with_registration"] is False
    assert final["final_technical_complete"] is False


def test_cli_creates_job_then_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(J, "create_job", lambda *args: seen.append(args) or tmp_path / "job.json")
    monkeypatch.setattr(RUN, "run_job", lambda job: {"acceptable_as_live_evidence": job == tmp_path / "job.json"})
    args = ["--protocol", "p", "--resource-proof", "r", "--resource-sha256", "s", "--isaaclab", "l",
            "--python", "py", "--stage", "st", "--checkpoint-logical-id", FIX.F2_ID]
    assert RUN.main(args) == 0 and seen[0][-1] == FIX.F2_ID
    with pytest.raises(SystemExit):
        RUN.main(["--protocol", "p"])
