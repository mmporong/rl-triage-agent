"""준비 callback과 625-frame 정책 영상을 한 Writer attach로 기록한다."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any

import record_vision_evidence_v1 as R
import vision_atomic_capture_v1 as ATOMIC
import vision_capture_readiness_v1 as READY
import vision_fxaa_policy_v1 as FXAA
import vision_play_worker_v1 as V1
import vision_prepared_record_v1 as PREPARED
import vision_replicator_startup_v1 as STARTUP
import vision_semantic_sdk_v1 as SDK
import vision_timing_adapter_v1 as TIMING
import vision_writer_payload_v1 as PAYLOAD

SDK_REPORT_SCHEMA = "vision_prepared_record_sdk_v1"
MODE = "prepared_policy_record_only"
EVENT_NAME = "rl-triage-prepared-record-v1"
EXTENSION = SDK.EXTENSION
ANNOTATORS = SDK.ANNOTATORS


class PreparedRecordSDKFailure(RuntimeError):
    """최종 envelope와 첫 일반 오류를 함께 전달한다."""

    def __init__(self, report: dict[str, Any], primary: Exception) -> None:
        super().__init__(f"prepared record SDK failed: {_type_name(primary)}")
        self.report = report
        self.envelope = report
        self.primary = primary


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _create_writer(rep: Any, mux: PREPARED.PreparedRecordMux) -> Any:
    """decode 전 raw payload를 mux에 넘기는 단일 Replicator Writer를 만든다."""

    class PreparedRecordWriter(rep.Writer):
        data_structure = "annotator"

        def __init__(self) -> None:
            self.annotators = [
                rep.AnnotatorRegistry.get_annotator("rgb"),
                rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": False}
                ),
                rep.AnnotatorRegistry.get_annotator("CameraParams"),
                rep.AnnotatorRegistry.get_annotator("IsaacReadTimes"),
            ]
            self.callback_error: BaseException | None = None

        def write(self, data: dict[str, Any]) -> None:
            if self.callback_error is not None:
                return
            try:
                mux.observe(data)
            except BaseException as error:  # noqa: BLE001 - callback 밖에서 재발생시킨다.
                self.callback_error = error
                if isinstance(error, Exception):
                    mux.source_error(error)

        def raise_callback_error(self) -> None:
            if self.callback_error is not None:
                raise self.callback_error

    return PreparedRecordWriter()


def _enable_extension(name: str) -> bool:
    return SDK._sdk_enable_extension(name)


def _registration_probe(rep: Any) -> dict[str, bool]:
    return {
        name: rep.AnnotatorRegistry.get_annotator(name) is not None
        for name in ANNOTATORS
    }


def _load_dependencies() -> dict[str, Any]:
    """Isaac child 안에서만 SDK 모듈을 불러오는 production dependency 표면이다."""
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]

    return {
        "rep": rep,
        "settings": READY._carb_settings,
        "usd_context": READY._usd_context,
        "enable_extension": _enable_extension,
        "startup": lambda selected_rep, settings: STARTUP.configure_replicator_startup(
            settings=settings, replicator=selected_rep
        ),
        "registration_probe": lambda: _registration_probe(rep),
        "run_real": V1.run_real,
        "writer": _create_writer,
        "physics_snapshot": READY._physics_snapshot,
        "readiness": READY._wait_capture_ready,
        "renderer_readback": READY._renderer_readback,
        "fxaa_required": FXAA.required,
        "fxaa_apply": FXAA.apply,
        "fxaa_readback": FXAA.readback,
        "fxaa_verify": FXAA.verify_readback,
        "capture_render_step": V1._capture_render_step,
        "write_reset": ATOMIC._write_reset,
        "write_atomic_journal": PAYLOAD.write_atomic_journal,
        "write_timing_journal": TIMING.write_capture_journal,
    }


_DEPENDENCY_KEYS = frozenset(
    {
        "rep",
        "settings",
        "usd_context",
        "enable_extension",
        "startup",
        "registration_probe",
        "run_real",
        "writer",
        "physics_snapshot",
        "readiness",
        "renderer_readback",
        "fxaa_required",
        "fxaa_apply",
        "fxaa_readback",
        "fxaa_verify",
        "capture_render_step",
        "write_reset",
        "write_atomic_journal",
        "write_timing_journal",
    }
)


def _validate_request(
    validated: Mapping[str, Any], seal_report: Callable[[Mapping[str, Any]], None]
) -> None:
    if not isinstance(validated, Mapping):
        raise TypeError("prepared record SDK request가 mapping이 아니다")
    if not callable(seal_report):
        raise TypeError("prepared record SDK seal_report가 callable이 아니다")
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("prepared record SDK job mode가 다르다")
    job_path = validated.get("job_path")
    output = validated.get("output_directory")
    if not isinstance(job_path, Path) or not isinstance(output, Path):
        raise TypeError("prepared record SDK job/output 경로가 Path가 아니다")
    if job_path.is_symlink() or not job_path.is_file():
        raise ValueError("prepared record SDK job은 regular file이어야 한다")
    if output.is_symlink() or output.resolve() != (job_path.parent / "output").resolve():
        raise ValueError("prepared record SDK output 경계가 다르다")
    if output.exists():
        raise FileExistsError("prepared record SDK output은 실행 전에 없어야 한다")
    job_sha = validated.get("job_sha256")
    if not isinstance(job_sha, str) or len(job_sha) != 64:
        raise ValueError("prepared record SDK job SHA-256이 잘못됐다")


def _record_request(validated: Mapping[str, Any]) -> dict[str, Any]:
    """동결 V1에는 기존 record mode만 보이는 작은 in-memory copy를 넘긴다."""
    selected = dict(validated)
    selected["job"] = {**dict(validated["job"]), "mode": R.RECORD_MODE}
    return selected


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("prepared record artifact가 regular file이 아니다")
    return {
        "file": resolved.name,
        "sha256": R.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _record_artifacts(
    produced: Mapping[str, Path], video: Path, output: Path
) -> dict[str, dict[str, Any]]:
    expected = {"trace", "measurement_1", "spec_1", "measurement_2", "spec_2"}
    if set(produced) != expected:
        raise ValueError("prepared record V1 artifact key가 동결 계약과 다르다")
    trace_path = Path(produced["trace"])
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("prepared record raw trace를 읽을 수 없다") from error
    if not isinstance(trace, dict):
        raise TypeError("prepared record raw trace가 object가 아니다")
    encoder = trace.get("encoder_receipt")
    timing = trace.get("timing_receipt")
    if not isinstance(encoder, dict) or not isinstance(timing, dict):
        raise TypeError("prepared record trace에 encoder/timing receipt가 없다")
    encoder_path = output / "encoder_receipt.json"
    timing_path = output / "timing_receipt.json"
    R.write_new(encoder_path, encoder)
    try:
        R.write_new(timing_path, timing)
    except BaseException:
        encoder_path.unlink(missing_ok=True)
        raise
    selected = {
        "raw_trace": trace_path,
        "encoder_receipt": encoder_path,
        "timing_receipt": timing_path,
        "measurement_1": Path(produced["measurement_1"]),
        "spec_1": Path(produced["spec_1"]),
        "measurement_2": Path(produced["measurement_2"]),
        "spec_2": Path(produced["spec_2"]),
        "video": Path(video),
    }
    root = output.resolve()
    for path in selected.values():
        if path.resolve().parent != root and path.resolve().parent.parent != root:
            raise ValueError("prepared record artifact가 output 경계를 벗어났다")
    return {name: _artifact(path) for name, path in selected.items()}


def _errors(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"phase": row["name"], "error_type": row["error_type"]}
        for row in phases
        if row["status"] in {"failed", "cancelled"}
    ]


def _envelope(
    prepared: Mapping[str, Any] | None,
    artifacts: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    renderer: Mapping[str, Any] | None,
    setup_evidence: Mapping[str, Any],
    record_result: Mapping[str, Any] | None,
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = _errors(phases)
    required = (
        "startup_completed",
        "run_real_started",
        "run_real_returned",
        "writer_attached",
        "prepare_completed",
        "writer_detached",
        "render_product_destroyed",
        "environment_closed",
        "mux_resolved",
        "mux_sealed",
    )
    approved = (
        prepared is not None
        and prepared.get("approved") is True
        and record_result is not None
        and lifecycle.get("arms_completed") == R.FRAME_COUNT
        and all(lifecycle.get(key) is True for key in required)
        and not errors
    )
    return {
        "schema": SDK_REPORT_SCHEMA,
        "prepared_record": deepcopy(dict(prepared)) if prepared is not None else None,
        "artifacts": deepcopy(dict(artifacts)),
        "lifecycle": {**deepcopy(dict(lifecycle)), "errors": errors},
        "renderer": deepcopy(dict(renderer)) if renderer is not None else {},
        "setup_evidence": deepcopy(dict(setup_evidence)),
        "record_result": (
            deepcopy(dict(record_result)) if record_result is not None else None
        ),
        "approved": approved,
    }


class _Coordinator:
    def __init__(
        self,
        validated: Mapping[str, Any],
        app: Any,
        deps: Mapping[str, Any],
        lifecycle: dict[str, Any],
        phase: Callable[[str, Callable[[], Any]], Any],
    ) -> None:
        self.validated = validated
        self.app = app
        self.deps = deps
        self.lifecycle = lifecycle
        self.phase = phase
        self.mux: PREPARED.PreparedRecordMux | None = None
        self.renderer: dict[str, Any] | None = None
        self.writer: Any | None = None
        self.base: Any | None = None
        self.settings: Any | None = None
        self.setup_physics = {"before": None, "after": None, "equal": None}
        self.prepare_camera: dict[str, Any] | None = None
        self.use_fxaa = False
        self._capture_result: dict[str, Any] | None = None

    def factory(
        self, base: Any, rep: Any, render_product: Any, profile: Mapping[str, Any]
    ) -> tuple[V1.CameraCapture, Callable[[], None]]:
        self.base = base
        context, stage_events = self.phase("usd_context", self.deps["usd_context"])
        self.settings = self.phase("settings_factory", self.deps["settings"])
        self.setup_physics["before"] = self.phase(
            "setup_physics_before",
            lambda: deepcopy(self.deps["physics_snapshot"](base)),
        )
        self.use_fxaa = bool(self.deps["fxaa_required"](self.validated["job"]))
        policy = None
        if self.use_fxaa:
            policy = self.phase(
                "fxaa_apply", lambda: self.deps["fxaa_apply"](rep, self.settings)
            )

        def read_renderer() -> dict[str, Any]:
            observed = self.deps["renderer_readback"](context, profile, self.settings)
            if not isinstance(observed, Mapping):
                raise TypeError("prepared record renderer readback이 mapping이 아니다")
            return dict(observed)

        selected_renderer = self.phase("renderer_readback", read_renderer)
        if not isinstance(selected_renderer, dict):
            raise TypeError("prepared record renderer readback 결과가 dict가 아니다")
        self.renderer = selected_renderer
        if policy is not None:
            selected_renderer["render_policy"] = policy
        self.phase(
            "capture_readiness",
            lambda: self.deps["readiness"](
                base,
                self.app,
                context,
                stage_events,
                Path(self.validated["job_path"]).with_suffix(
                    ".prepared-record-readiness.json"
                ),
                selected_renderer,
            ),
        )
        if self.use_fxaa:
            self.phase(
                "fxaa_verify",
                lambda: self.deps["fxaa_verify"](
                    self.deps["fxaa_readback"](self.settings)
                ),
            )

        def verify_setup() -> dict[str, Any]:
            after = deepcopy(self.deps["physics_snapshot"](base))
            self.setup_physics["after"] = after
            self.setup_physics["equal"] = self.setup_physics["before"] == after
            if self.setup_physics["equal"] is not True:
                raise RuntimeError("prepared record readiness가 physics 상태를 바꿨다")
            return after

        self.phase("setup_physics_after", verify_setup)
        render_product_ref = ATOMIC._render_product_name(render_product)
        reset_paths: dict[int, Path] = {}

        def commit(
            ordinal: int, decoded: Mapping[str, Any], bundle: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            if self._capture_result is not None:
                raise RuntimeError("prepared record 이전 frame 원문이 소비되지 않았다")
            reset = reset_paths.get(ordinal)
            if reset is None:
                raise RuntimeError("prepared record renderer reset receipt가 없다")
            job_path = Path(self.validated["job_path"])
            journal, npz = self.deps["write_atomic_journal"](
                job_path, ordinal, R.RECORD_MODE, decoded, bundle
            )
            timing = self.deps["write_timing_journal"](
                job_path,
                ordinal,
                decoded["read_times_observed"],
                decoded["reference_time"],
                bundle["rgb"]["sha256"],
            )
            self._capture_result = {
                "ordinal": ordinal,
                "rgb": decoded["rgb"].tobytes(order="C"),
                "read_times": deepcopy(decoded["read_times"]),
                "reference_time": deepcopy(decoded["reference_time"]),
                "atomic_bundle": deepcopy(dict(bundle)),
                "atomic_journal": journal,
                "atomic_npz": npz,
            }
            return {
                "ordinal": ordinal,
                "atomic_json": journal.name,
                "atomic_npz": npz.name,
                "timing": timing.name,
                "renderer_reset": reset.name,
            }

        selected_mux = PREPARED.PreparedRecordMux(
            render_product_ref=render_product_ref,
            camera_path=profile["camera_prim_path"],
            resolution=[R.WIDTH, R.HEIGHT],
            expected_count=R.FRAME_COUNT,
            commit_callback=commit,
        )
        self.mux = selected_mux
        selected_writer = self.deps["writer"](rep, selected_mux)
        if selected_writer is None:
            raise RuntimeError("prepared record Writer가 없다")
        self.writer = selected_writer
        self.lifecycle["writer_attach_attempted"] = True

        def cleanup_after_factory_failure(error: BaseException) -> None:
            if isinstance(error, Exception):
                selected_mux.source_error(error)
            else:
                selected_mux.source_error(
                    RuntimeError("prepared record factory cancellation")
                )
            self.lifecycle["writer_detach_attempted"] = True
            try:
                selected_mux.begin_detach()
                try:
                    selected_writer.detach()
                    selected_writer.raise_callback_error()
                finally:
                    selected_mux.end_detach(
                        self.deps["physics_snapshot"](base),
                        float(base.sim.current_time),
                    )
            except BaseException:  # noqa: BLE001 - 원오류를 보존하며 detached=false로 거부한다.
                return
            self.lifecycle["writer_detached"] = True

        try:
            self.phase(
                "writer_attach",
                lambda: selected_writer.attach(
                    render_product, trigger=rep.trigger.on_custom_event(EVENT_NAME)
                ),
            )
            self.lifecycle["writer_attached"] = True
        except BaseException as error:
            cleanup_after_factory_failure(error)
            raise

        root = ATOMIC._root_position(base)
        eye = [root[index] + [3.0, 3.0, 2.0][index] for index in range(3)]
        lookat = [root[index] + [0.0, 0.0, 0.35][index] for index in range(3)]
        try:
            self.phase(
                "prepare_camera_A",
                lambda: base.sim.set_camera_view(
                    eye=eye,
                    target=lookat,
                    camera_prim_path=profile["camera_prim_path"],
                ),
            )
            self.prepare_camera = {"eye_world_m": eye, "lookat_world_m": lookat}
            self.phase("prepare_renderer_reset", context.reset_renderer_accumulation)
            before = self.deps["physics_snapshot"](base)
            selected_mux.begin_prepare(before, float(base.sim.current_time))
            self.phase("prepare_event", lambda: rep.utils.send_og_event(EVENT_NAME))
            self.phase(
                "prepare_render",
                lambda: self.deps["capture_render_step"](rep.orchestrator, profile),
            )
            selected_writer.raise_callback_error()
            selected_mux.end_prepare(
                self.deps["physics_snapshot"](base), float(base.sim.current_time)
            )
            self.lifecycle["prepare_completed"] = True
        except BaseException as error:
            cleanup_after_factory_failure(error)
            raise

        def capture(
            capture_base: Any,
            capture_rep: Any,
            annotators: list[Any],
            camera_profile: Mapping[str, Any],
            eye_world: list[float],
            lookat_world: list[float],
            ordinal: int,
        ) -> dict[str, Any]:
            if capture_base is not base or capture_rep is not rep or annotators:
                raise ValueError("prepared record capture identity가 다르다")
            if camera_profile != profile:
                raise ValueError("prepared record camera profile이 바뀌었다")
            selected_writer.raise_callback_error()
            before_arm = self.deps["physics_snapshot"](base)
            selected_mux.begin_arm(ordinal, before_arm, float(base.sim.current_time))

            def reset_after_camera() -> None:
                context.reset_renderer_accumulation()
                reset_paths[ordinal] = self.deps["write_reset"](
                    self.validated,
                    ordinal,
                    use_fxaa=self.use_fxaa,
                    settings=self.settings,
                )

            proxy = READY._BaseProxy(base, reset_after_camera)
            proxy.sim.set_camera_view(
                eye=eye_world,
                target=lookat_world,
                camera_prim_path=profile["camera_prim_path"],
            )
            self.phase(
                f"arm_{ordinal}_event",
                lambda: rep.utils.send_og_event(EVENT_NAME),
            )
            self.phase(
                f"arm_{ordinal}_render",
                lambda: self.deps["capture_render_step"](rep.orchestrator, profile),
            )
            selected_writer.raise_callback_error()
            selected_mux.end_arm(
                ordinal,
                self.deps["physics_snapshot"](base),
                float(base.sim.current_time),
            )
            self.lifecycle["arms_completed"] = ordinal + 1
            result = self._capture_result
            self._capture_result = None
            if result is None or result.pop("ordinal", None) != ordinal:
                raise RuntimeError("prepared record frame 원문 handoff가 없다")
            return result

        def cleanup() -> None:
            self.lifecycle["writer_detach_attempted"] = self.writer is not None
            selected_mux.begin_detach()
            try:
                selected_writer.detach()
                selected_writer.raise_callback_error()
            finally:
                selected_mux.end_detach(
                    self.deps["physics_snapshot"](base), float(base.sim.current_time)
                )
            self.lifecycle["writer_detached"] = True

        return capture, cleanup

def run_prepared_record(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """준비 frame 뒤 동결 V1의 625-frame 정책 기록을 실행하고 봉인한다."""
    _validate_request(validated, seal_report)
    deps = dict(dependencies) if dependencies is not None else _load_dependencies()
    if set(deps) != _DEPENDENCY_KEYS:
        raise ValueError("prepared record SDK dependency 표면이 다르다")
    phases: list[dict[str, Any]] = []
    phase_order: list[str] = []
    lifecycle: dict[str, Any] = {
        "startup_completed": False,
        "run_real_started": False,
        "run_real_returned": False,
        "writer_attach_attempted": False,
        "writer_attached": False,
        "prepare_completed": False,
        "arms_completed": 0,
        "writer_detach_attempted": False,
        "writer_detached": False,
        "render_product_destroyed": False,
        "environment_closed": False,
        "mux_resolved": False,
        "mux_sealed": False,
        "envelope_sealed": False,
    }
    primary: Exception | None = None
    cancellation: BaseException | None = None
    cancellation_tb: TracebackType | None = None
    prepared: dict[str, Any] | None = None
    artifacts: dict[str, Any] = {}
    record_result: dict[str, Any] | None = None
    extension = {"name": EXTENSION, "enable_result": False, "app_update_completed": False}
    startup: dict[str, Any] = {"before": None, "after": None}
    registrations = {name: False for name in ANNOTATORS}

    def phase(name: str, call: Callable[[], Any]) -> Any:
        return SDK._phase(name, phases, phase_order, call)

    coordinator = _Coordinator(validated, app, deps, lifecycle, phase)
    try:
        rep = deps["rep"]
        settings = phase("settings_startup", deps["settings"])

        def enable() -> bool:
            if deps["enable_extension"](EXTENSION) is not True:
                raise RuntimeError("prepared record extension 활성화가 실패했다")
            return True

        extension["enable_result"] = phase("extension_enable", enable)
        phase("extension_app_update", app.update)
        extension["app_update_completed"] = True
        startup = phase(
            "replicator_startup",
            lambda: SDK._startup_readback(deps["startup"](rep, settings)),
        )
        registrations = phase(
            "annotator_registrations",
            lambda: SDK._registration_readback(
                deps["registration_probe"](), registrations
            ),
        )
        lifecycle["startup_completed"] = True
        lifecycle["run_real_started"] = True
        produced, video = phase(
            "run_real",
            lambda: deps["run_real"](
                _record_request(validated),
                app,
                camera_capture_factory=coordinator.factory,
            ),
        )
        lifecycle["run_real_returned"] = True
        lifecycle["render_product_destroyed"] = True
        lifecycle["environment_closed"] = True
        artifacts = _record_artifacts(
            produced, Path(video), Path(validated["output_directory"])
        )
        video_descriptor = artifacts["video"]
        record_result = {
            "frame_count": R.FRAME_COUNT,
            "fps": R.FPS,
            "video": deepcopy(video_descriptor),
        }
    except BaseException as error:  # noqa: BLE001 - mux 봉인과 failure evidence를 이어간다.
        primary, cancellation, cancellation_tb = SDK._remember_failure(
            error, primary, cancellation, cancellation_tb
        )

    mux = coordinator.mux
    if mux is not None:
        if primary is not None:
            mux.source_error(primary)
        elif cancellation is not None:
            mux.source_error(RuntimeError("prepared record cancellation before seal"))
        try:
            phase("mux_resolve", mux.resolve)
            lifecycle["mux_resolved"] = True
            selected_prepared = phase("mux_seal", mux.seal)
            if not isinstance(selected_prepared, dict):
                raise TypeError("prepared record mux seal 결과가 dict가 아니다")
            prepared = selected_prepared
            lifecycle["mux_sealed"] = True
            if selected_prepared.get("approved") is not True and primary is None:
                record_result = None
                primary = RuntimeError("prepared record mux가 기록을 승인하지 않았다")
        except BaseException as error:  # noqa: BLE001 - terminal envelope를 계속 만든다.
            primary, cancellation, cancellation_tb = SDK._remember_failure(
                error, primary, cancellation, cancellation_tb
            )

    setup_evidence = {
        "phase_order": list(phase_order),
        "setup_physics_before_after": deepcopy(coordinator.setup_physics),
        "extension": deepcopy(extension),
        "startup": deepcopy(startup),
        "registrations": deepcopy(registrations),
        "prepare_camera": deepcopy(coordinator.prepare_camera),
        "raw_callback_count": mux.raw_callback_count if mux is not None else 0,
        "complete_drain_proven": False,
        "all_future_callbacks_collected": False,
    }
    seal_row = {"name": "seal_report", "status": "running", "error_type": None}
    phases.append(seal_row)
    phase_order.append("seal_report")
    sealed = _envelope(
        prepared,
        artifacts,
        lifecycle,
        coordinator.renderer,
        setup_evidence,
        record_result,
        phases,
    )
    try:
        seal_report(deepcopy(sealed))
    except BaseException as error:  # noqa: BLE001 - 첫 오류와 cancellation을 보존한다.
        seal_row["status"] = "cancelled" if not isinstance(error, Exception) else "failed"
        seal_row["error_type"] = _type_name(error)
        primary, cancellation, cancellation_tb = SDK._remember_failure(
            error, primary, cancellation, cancellation_tb
        )
    else:
        seal_row["status"] = "complete"
        lifecycle["envelope_sealed"] = True

    terminal = _envelope(
        prepared,
        artifacts,
        lifecycle,
        coordinator.renderer,
        setup_evidence,
        record_result,
        phases,
    )
    if cancellation is not None:
        cancellation.__dict__["prepared_record_sdk_envelope"] = terminal
        raise cancellation.with_traceback(cancellation_tb)
    if primary is not None:
        raise PreparedRecordSDKFailure(terminal, primary) from primary
    return terminal
