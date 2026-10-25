"""앱 생성 뒤 Replicator capture trigger 설정을 fail-closed로 고정한다."""
from __future__ import annotations

from typing import Any

ORCHESTRATOR_ENABLED = "/exts/omni.replicator.core/Orchestrator/enabled"
CAPTURE_ON_PLAY = "/omni/replicator/captureOnPlay"


def _load_settings() -> Any:
    import carb  # pyright: ignore[reportMissingImports]

    return carb.settings.get_settings()


def _load_replicator() -> Any:
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]

    return rep


def configure_replicator_startup(
    *, settings: Any | None = None, replicator: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Orchestrator trigger를 켜고 capture-on-play를 끈 뒤 readback을 반환한다."""
    selected_settings = settings if settings is not None else _load_settings()
    before = {
        "orchestrator_enabled": selected_settings.get(ORCHESTRATOR_ENABLED),
        "capture_on_play": selected_settings.get(CAPTURE_ON_PLAY),
    }

    selected_settings.set_bool(ORCHESTRATOR_ENABLED, True)
    orchestrator_enabled = selected_settings.get(ORCHESTRATOR_ENABLED)
    if orchestrator_enabled is not True:
        raise RuntimeError("Replicator Orchestrator enabled 설정 readback이 true가 아니다")

    selected_replicator = replicator if replicator is not None else _load_replicator()
    selected_replicator.orchestrator.set_capture_on_play(False)
    capture_on_play = selected_settings.get(CAPTURE_ON_PLAY)
    if capture_on_play is not False:
        raise RuntimeError("Replicator captureOnPlay 설정 readback이 false가 아니다")

    return {
        "before": before,
        "after": {
            "orchestrator_enabled": orchestrator_enabled,
            "capture_on_play": capture_on_play,
        },
    }
