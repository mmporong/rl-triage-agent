"""캡처 직전 USD asset 준비와 renderer 누적 초기화를 검증한다."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_fxaa_policy_v1 as FXAA
import vision_play_worker_v1 as V1
import vision_play_worker_v2 as V2

SCHEMA = "vision_capture_readiness_v1"
SIDECAR_SUFFIX = ".capture-readiness.json"
MAX_UPDATES = 240
TIMEOUT_S = 30.0
REQUIRED_IDLE_UPDATES = 3
PLAYER_SETTING = "/app/player/playSimulations"
CARB_SETTING_KEYS = (
    "/rtx/rendermode",
    "/rtx/post/aa/op",
    "/rtx/post/dlss/execMode",
    "/rtx/post/histogram/enabled",
    "/rtx/post/aa/autoExposureMode",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/directLighting/sampledLighting/autoEnable",
    "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "/rtx/newDenoiser/enabled",
    "/omni/replicator/RTSubframes",
    "/omni/replicator/asyncRendering",
)


def _usd_context() -> tuple[Any, Any]:
    import omni.usd  # pyright: ignore[reportMissingImports]

    return omni.usd.get_context(), omni.usd.StageEventType


def _carb_settings() -> Any:
    import carb.settings  # pyright: ignore[reportMissingImports]

    return carb.settings.get_settings()


def _tensor_values(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().flatten().tolist()]


def _physics_snapshot(base: Any) -> dict[str, Any]:
    robot = base.scene["robot"]
    return {
        "simulation_time_s": float(base.sim.current_time),
        "common_step_counter": int(base.common_step_counter),
        "root_position_world_m": _tensor_values(robot.data.root_pos_w[0, :3]),
        "root_linear_velocity_body_m_s": _tensor_values(robot.data.root_lin_vel_b[0, :3]),
        "root_angular_velocity_body_rad_s": _tensor_values(robot.data.root_ang_vel_b[0, :3]),
        "projected_gravity_body": _tensor_values(robot.data.projected_gravity_b[0, :3]),
    }


def _loading_readback(context: Any) -> dict[str, Any]:
    status = context.get_stage_loading_status()
    if (not isinstance(status, tuple) or len(status) != 3 or not isinstance(status[0], str)
            or type(status[1]) is not int or type(status[2]) is not int
            or status[1] < 0 or status[2] < 0):
        raise TypeError("USD stage loading status가 (message, loaded, total) 형식이 아니다")
    streaming = context.get_stage_streaming_status()
    if type(streaming) is not bool:
        raise TypeError("USD stage streaming status가 bool이 아니다")
    return {
        "message": status[0], "files_loaded": status[1],
        "files_total": status[2], "streaming_busy": streaming,
    }


def _asset_update(base: Any, app: Any) -> dict[str, Any]:
    """Isaac Lab render와 같은 설정으로 앱만 갱신하고 이전 값을 복원한다."""
    before = base.sim.get_setting(PLAYER_SETTING)
    if type(before) is not bool:
        raise TypeError("asset update의 simulation 활성 설정이 bool이 아니다")
    try:
        base.sim.set_setting(PLAYER_SETTING, False)
        if base.sim.get_setting(PLAYER_SETTING) is not False:
            raise RuntimeError("asset update 동안 simulation 갱신을 끄지 못했다")
        app.update()
    finally:
        base.sim.set_setting(PLAYER_SETTING, before)
        if base.sim.get_setting(PLAYER_SETTING) is not before:
            raise RuntimeError("asset update 뒤 simulation 활성 설정을 복원하지 못했다")
    return {"setting": PLAYER_SETTING, "before": before,
            "during": False, "after": before}


def _wait_capture_ready(
    base: Any,
    app: Any,
    context: Any,
    stage_events: Any,
    receipt_path: Path,
    renderer: Mapping[str, Any],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """asset event와 loading·streaming idle을 함께 확인하며 physics 불변을 검사한다."""
    event_history: list[dict[str, Any]] = []
    counts = {"loading": 0, "loaded": 0, "aborted": 0, "synthetic_loading": 0}
    loading_type = int(stage_events.ASSETS_LOADING)
    loaded_type = int(stage_events.ASSETS_LOADED)
    aborted_type = int(stage_events.ASSETS_LOAD_ABORTED)

    def record(name: str) -> None:
        event_history.append({"event": name, "counts": dict(counts)})

    def on_stage_event(event: Any) -> None:
        if event.type == loading_type:
            counts["loading"] += 1
            record("assets_loading")
        elif event.type in (loaded_type, aborted_type):
            key = "loaded" if event.type == loaded_type else "aborted"
            counts[key] += 1
            if counts["loaded"] + counts["aborted"] > counts["loading"]:
                counts["loading"] += 1
                counts["synthetic_loading"] += 1
                record("synthetic_assets_loading")
            record("assets_loaded" if key == "loaded" else "assets_load_aborted")

    started = clock()
    subscription: Any | None = None
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    before_loading: dict[str, Any] | None = None
    after_loading: dict[str, Any] | None = None
    idle_updates = 0
    updates = 0
    history: list[dict[str, Any]] = []
    status = "failed"
    reason = "unknown"
    primary_error: BaseException | None = None
    try:
        subscription = context.get_stage_event_stream().create_subscription_to_pop(
            on_stage_event, name="rl-triage-capture-readiness",
        )
        before_state = _physics_snapshot(base)
        before_loading = _loading_readback(context)
        after_loading = before_loading
        while updates < MAX_UPDATES and clock() - started <= TIMEOUT_S:
            update_control = _asset_update(base, app)
            updates += 1
            if clock() - started > TIMEOUT_S:
                raise TimeoutError("USD asset loading·streaming 캡처 준비 시간이 상한을 넘었다")
            after_loading = _loading_readback(context)
            pending = counts["loading"] - counts["loaded"] - counts["aborted"]
            history.append({"update": updates, "loading": dict(after_loading),
                            "event_counts": dict(counts), "pending_asset_batches": pending,
                            "simulation_update_control": update_control})
            if counts["aborted"]:
                raise RuntimeError("USD asset load가 중단되어 캡처 준비를 확인할 수 없다")
            idle = (
                pending == 0
                and after_loading["files_loaded"] == 0
                and after_loading["files_total"] == 0
                and after_loading["streaming_busy"] is False
            )
            idle_updates = idle_updates + 1 if idle else 0
            if idle_updates >= REQUIRED_IDLE_UPDATES:
                status = "ready"
                reason = "consecutive_loading_streaming_idle"
                break
        else:
            raise TimeoutError("USD asset loading·streaming이 캡처 준비 상한 안에 idle이 되지 않았다")
    except BaseException as error:  # noqa: BLE001 - receipt 뒤 원래 예외를 다시 올린다.
        primary_error = error
        reason = (
            "poll_budget_exhausted" if isinstance(error, TimeoutError)
            else f"{type(error).__name__}: {error}"
        )
    finally:
        if subscription is not None:
            del subscription
        try:
            after_loading = _loading_readback(context)
        except BaseException as error:  # noqa: BLE001 - 먼저 발생한 예외를 보존한다.
            if primary_error is None:
                primary_error = error
                status = "failed"
                reason = f"final_readback_{type(error).__name__}: {error}"
        try:
            after_state = _physics_snapshot(base)
        except BaseException as error:  # noqa: BLE001 - 먼저 발생한 예외를 보존한다.
            if primary_error is None:
                primary_error = error
                status = "failed"
                reason = f"physics_readback_{type(error).__name__}: {error}"
        if primary_error is None and after_state != before_state:
            primary_error = RuntimeError("캡처 준비 app.update가 physics 상태 또는 simulation time을 바꿨다")
            status = "failed"
            reason = "physics_or_simulation_time_changed"
        elapsed = max(0.0, clock() - started)
        receipt = {
            "schema": SCHEMA,
            "status": status,
            "reason": reason,
            "bounds": {
                "max_updates": MAX_UPDATES,
                "polling_elapsed_budget_s": TIMEOUT_S,
                "polling_elapsed_budget_scope": "checked_between_synchronous_app_updates_not_a_hard_watchdog",
                "required_consecutive_idle_updates": REQUIRED_IDLE_UPDATES,
            },
            "updates": updates,
            "elapsed_s": elapsed,
            "event_counts": dict(counts),
            "event_history": event_history,
            "loading_history": history,
            "before_loading": before_loading,
            "final_loading": after_loading,
            "physics_before": before_state,
            "physics_after": after_state,
            "renderer": dict(renderer),
        }
        try:
            R.write_new(receipt_path, receipt)
        except BaseException:
            if primary_error is None:
                raise
    if primary_error is not None:
        raise primary_error
    return receipt


class _ResettingSim:
    def __init__(self, sim: Any, reset: Callable[[], None]) -> None:
        self._sim = sim
        self._reset = reset

    def set_camera_view(self, *args: Any, **kwargs: Any) -> Any:
        result = self._sim.set_camera_view(*args, **kwargs)
        self._reset()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sim, name)


class _BaseProxy:
    def __init__(self, base: Any, reset: Callable[[], None]) -> None:
        self.sim = _ResettingSim(base.sim, reset)


def _renderer_readback(context: Any, camera_profile: Mapping[str, Any], settings: Any) -> dict[str, Any]:
    engines = context.get_attached_hydra_engine_names()
    if not isinstance(engines, list) or any(not isinstance(value, str) for value in engines):
        raise TypeError("USD attached Hydra engine readback이 문자열 목록이 아니다")
    return {
        "attached_hydra_engine_names": list(engines),
        "carb_settings": {key: settings.get(key) for key in CARB_SETTING_KEYS},
        "expected_capture_profile": {
            "capture_primitive": camera_profile["capture_primitive"],
            "rt_subframes": camera_profile["rt_subframes"],
            "pause_timeline": camera_profile["pause_timeline"],
            "delta_time_s": camera_profile["delta_time_s"],
        },
        "renderer_accumulation_reset": "omni.usd.UsdContext.reset_renderer_accumulation",
    }


def run_capture(validated: Mapping[str, Any], app: Any) -> tuple[dict[str, Path], Path]:
    """기존 V1/V2 결과 계약을 유지한 채 준비 장벽을 캡처 callback에 주입한다."""
    context, stage_events = _usd_context()
    smoke = validated["job"]["mode"] == R.RENDER_SYNC_SMOKE_MODE
    selected = V1.run_render_sync_smoke if smoke else V1.run_real
    journal_capture = V2._journaled_camera_capture(validated, preserve_rgb=smoke)
    sidecar = Path(validated["job_path"]).with_suffix(SIDECAR_SUFFIX)
    settings = _carb_settings()
    readiness: dict[str, Any] | None = None
    reset_count = 0
    use_fxaa = FXAA.required(validated["job"])

    def capture(
        base: Any, rep: Any, annotators: list[Any], camera_profile: Mapping[str, Any],
        eye: list[float], lookat: list[float], ordinal: int,
    ) -> dict[str, Any]:
        nonlocal readiness, reset_count
        if readiness is None:
            policy = FXAA.apply(rep, settings) if use_fxaa else None
            renderer = _renderer_readback(context, camera_profile, settings)
            if policy is not None:
                renderer["render_policy"] = policy
            readiness = _wait_capture_ready(
                base, app, context, stage_events, sidecar,
                renderer,
            )
        if use_fxaa:
            FXAA.verify_readback(FXAA.readback(settings))

        def reset_after_camera() -> None:
            nonlocal reset_count
            context.reset_renderer_accumulation()
            reset_count += 1
            reset_path = Path(validated["job_path"]).with_name(
                f"{Path(validated['job_path']).stem}.capture-{ordinal:06d}.renderer-reset.json",
            )
            reset_receipt: dict[str, Any] = {
                "schema": "vision_capture_renderer_reset_v1",
                "capture_ordinal": ordinal,
                "completed": True,
                "method": "omni.usd.UsdContext.reset_renderer_accumulation",
            }
            if use_fxaa:
                observed = FXAA.readback(settings)
                FXAA.verify_readback(observed)
                reset_receipt["render_policy_readback"] = observed
            R.write_new(reset_path, reset_receipt)

        proxy = _BaseProxy(base, reset_after_camera)
        result = journal_capture(proxy, rep, annotators, camera_profile, eye, lookat, ordinal)
        if use_fxaa:
            FXAA.verify_readback(FXAA.readback(settings))
        if reset_count != ordinal + 1:
            raise RuntimeError("camera capture마다 renderer accumulation이 한 번씩 초기화되지 않았다")
        return result

    return selected(validated, app, camera_capture=capture)
