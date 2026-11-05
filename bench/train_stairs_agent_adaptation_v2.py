# pyright: reportMissingImports=false
"""과제 적응 v1의 학습을 보존하고 AppLauncher 인자를 한 경로로 전달한다."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import train_stairs_agent_adaptation_v1 as V1


def launcher_arguments(args: argparse.Namespace) -> dict:
    value = vars(args).copy()
    value.update(kit_args="--/app/vulkan=false --/app/window/hideUi=true", fast_shutdown=False)
    return value


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    for key in ("spec", "authorization", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    auth = V1.PLAY.read(args.authorization)
    if (auth.get("schema") != "stairs_agent_adaptation_authorization_v1"
            or auth.get("recipe") != V1.C.RECIPE or auth.get("task") != V1.C.TASK
            or auth.get("selected_by") != "model"):
        raise ValueError("GPU 시작 전 모델 제안·호스트 승인 계약 검증 실패")
    V1.PLAY.load_base()._read_spec(args.spec)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        app = AppLauncher(launcher_arguments(args)).app
        V1.C.write_new(args.out/"app-startup.json", {"pid": os.getpid()})
        V1.run(args, app)
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/app/fastShutdown", True)
        if settings.get_as_bool("/app/fastShutdown") is not True:
            raise ValueError("학습 증거 봉인 후 종료 설정 readback 실패")
        V1.C.write_new(args.out/"app-close-entered.json", {"pid": os.getpid(), "fast_shutdown": True,
                      "wait_for_replicator": False,
                      "training_result_sha256": V1.C.sha(args.out/"training-result.json")})
        app.close(wait_for_replicator=False)
    except BaseException as error:  # noqa: BLE001 - SDK가 삼키는 예외도 실패 증거로 봉인한다.
        V1.C.write_new(args.out/"training-failure.json", {"status": "invalid", "type": type(error).__name__,
                      "traceback": traceback.format_exc()})
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
