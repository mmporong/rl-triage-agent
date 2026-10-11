"""Check the pinned cloud simulator on an empty stage, before diagnostic evaluation.

This is environment setup, not a diagnostic probe or a trained-policy evaluation.
It does not load a checkpoint, create robots, or alter the frozen probe catalog.
"""
from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("environment smoke report already exists")

    started = time.perf_counter()
    # These SDK imports resolve in the registered remote venv; the host only compiles this helper.
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = AppLauncher(headless=True).app
    sim = None
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext  # pyright: ignore[reportMissingImports]

        sim = SimulationContext(SimulationCfg(dt=0.01))
        sim.reset()
        sim.step()
        packages = {
            name: version(name)
            for name in ("isaacsim", "isaaclab", "isaaclab_rl", "torch", "rsl-rl-lib", "gymnasium")
        }
    finally:
        try:
            if sim is not None:
                try:
                    sim.clear_all_callbacks()
                finally:
                    sim.clear_instance()
        finally:
            app.close()

    report = {
        "schema": "cloud_environment_smoke_v1",
        "completed": True,
        "scope": "empty_stage_environment_setup_not_diagnostic_evidence",
        "checkpoint_loaded": False,
        "robot_environments": 0,
        "empty_stage_physics_steps": 1,
        "wall_s": time.perf_counter() - started,
        "cpu_s": time.process_time(),
        "cpu_scope": "smoke_python_process_lifetime",
        "device_busy_s": None,
        "packages": packages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("RL_TRIAGE_CLOUD_SMOKE " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
