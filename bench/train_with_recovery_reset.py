"""v4 교정 재개 경계 안에서 등록된 정책/optimizer 재설정을 적용한다."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
EVALS = ROOT / "evals"
for directory in (BENCH, EVALS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import recovery_reset_interventions as I  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
import run_recovery_reset as H  # noqa: E402


def execute(manifest_path: Path) -> dict:
    manifest = H.read(manifest_path)
    intervention = H.intervention_for_arm(manifest.get("arm"))
    reset_seed = H.reset_seed(manifest.get("seed"), intervention)
    original_install = R._install_runner_hooks

    def install_with_intervention(bound_manifest, helper, runtime, **kwargs):
        originals = original_install(bound_manifest, helper, runtime, **kwargs)
        runner_class = originals[3]
        verified_load = runner_class.load

        def measured_load(self, *args, **kwargs):
            result = verified_load(self, *args, **kwargs)
            if runtime.get("intervention_applied"):
                raise RuntimeError("recovery reset intervention was applied more than once")
            import torch
            row = I.apply_intervention(self.alg.policy, self.alg.optimizer, intervention, reset_seed, torch)
            if row["policy_state_before"] != runtime["source_checkpoint_state"]["policy_state_sha256"]:
                raise ValueError("intervention did not start from the verified checkpoint policy")
            if row["optimizer_state_before"] != runtime["source_checkpoint_state"]["optimizer_state_sha256"]:
                raise ValueError("intervention did not start from the verified checkpoint optimizer")
            helper.write_new(Path(bound_manifest["output_dir"]) / "intervention.json", row)
            runtime["intervention_applied"] = True
            runtime["intervention"] = row
            return result

        runner_class.load = measured_load
        return originals

    R._install_runner_hooks = install_with_intervention
    try:
        receipt = R.execute(manifest_path, H)
    finally:
        R._install_runner_hooks = original_install
    if not H.read(Path(manifest["output_dir"]) / "intervention.json"):
        raise RuntimeError("intervention readback is missing")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
