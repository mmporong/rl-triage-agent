"""generic recovery worker를 terminal evidence AppLauncher와 결합한다."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import brev_recovery_worker_v3 as WORKER
import local_f1_restart_terminal_v2 as TERMINAL


def install(output: Path, terminal: Path, old: set[str]) -> type[Any]:
    import carb.settings
    import isaaclab.app as app_module

    original = app_module.AppLauncher

    class BoundLauncher:
        add_app_launcher_args = staticmethod(original.add_app_launcher_args)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["fast_shutdown"] = False
            self._launcher = original(*args, **kwargs)
            self.app = TERMINAL.AppProxy(self._launcher.app, carb.settings.get_settings(), output, terminal, old)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._launcher, name)

    app_module.AppLauncher = BoundLauncher
    return original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--old-checkpoint-sha256", required=True, action="append")
    args = parser.parse_args()
    original = install(args.output.resolve(), args.terminal.resolve(), set(args.old_checkpoint_sha256))
    try:
        return WORKER.main(["--job", str(args.job.resolve()), "--output", str(args.output.resolve())])
    finally:
        import isaaclab.app as app_module
        app_module.AppLauncher = original


if __name__ == "__main__":
    raise SystemExit(main())
