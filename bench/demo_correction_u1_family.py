"""U1 전체 보상 함수군 교정 manifest를 기존 v4 학습 하네스로 실행한다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import reward_family_correction as FAMILY  # noqa: E402
import train_with_recovery_mode as RUNNER  # noqa: E402
import u1_scope_extension as H  # noqa: E402


def install_fault(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fault = manifest["fault"]
    protocol = json.loads(Path(manifest["protocol_path"]).read_text(encoding="utf-8"))
    contract = protocol["correction_source_contract"]
    environment_path = Path(manifest["output_dir"]) / "family_environment.json"
    if fault == "U1":
        import upstream_isaac
        import upstream_faults
        original_readback = upstream_faults.readback

        def checked_readback(name, env):
            row = original_readback(name, env)
            row["family_environment"] = FAMILY.environment_readback(env, contract)
            row["ok"] = bool(row.get("ok")) and row["family_environment"]["ok"]
            H.write_new(environment_path, row["family_environment"])
            return row

        upstream_faults.readback = checked_readback
        sys.modules["hidden_faults"] = upstream_isaac.hidden_faults_shim("U1")
    elif fault == "NONE":
        import gymnasium
        shim = ModuleType("hidden_faults")

        def before_env(name):
            if name != "NONE":
                raise ValueError("NONE control received another fault")

        def arm(name):
            if name != "NONE":
                raise ValueError("NONE control received another fault")
            original_make = gymnasium.make
            state = {"done": False}

            def make(*args, **kwargs):
                if state["done"]:
                    return original_make(*args, **kwargs)
                state["done"] = True
                shim._before_env(name)
                env = original_make(*args, **kwargs)
                row = FAMILY.environment_readback(env, contract)
                H.write_new(environment_path, row)
                print("[reward_family] " + json.dumps(row), flush=True)
                return env

            gymnasium.make = make

        shim._before_env = before_env
        shim.arm = arm
        shim.FAULTS = {"NONE": {"category": "control"}}
        sys.modules["hidden_faults"] = shim
    else:
        raise ValueError("unregistered fault")
    sys.modules["correction_adapter"] = FAMILY
    return fault


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    install_fault(args.manifest)
    RUNNER.execute(args.manifest, H)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
