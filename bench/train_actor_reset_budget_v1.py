"""가변 예산 actor-last-reset 또는 독립 restart 교정 학습 child."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import recovery_reset_interventions as I  # noqa: E402
import train_with_correction as T  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
import actor_reset_budget_v1 as H  # noqa: E402


def execute(manifest_path: Path):
    manifest = H.read(manifest_path)
    mode, _ = H.arm_spec(manifest.get("arm", ""))
    if mode == "restart":
        return R.execute(manifest_path, H)
    original_install = R._install_runner_hooks

    def install(bound_manifest, helper, runtime, **kwargs):
        originals = original_install(bound_manifest, helper, runtime, **kwargs)
        runner_class, verified_load = originals[3], originals[3].load

        def measured_load(self, *args, **load_kwargs):
            result = verified_load(self, *args, **load_kwargs)
            if runtime.get("budget_intervention_applied"):
                raise RuntimeError("개입이 두 번 적용됐다")
            import torch
            before_policy = T.state_digest(self.alg.policy.state_dict())["sha256"]
            before_optimizer = T.state_digest(self.alg.optimizer.state_dict())["sha256"]
            critic_before = T.state_digest(self.alg.policy.critic.state_dict())["sha256"]
            std_before = T.state_digest(self.alg.policy.std)["sha256"]
            linears = [(f"actor.{name}".rstrip("."), module) for name, module in
                       self.alg.policy.actor.named_modules() if isinstance(module, torch.nn.Linear)]
            actor_before = {name: T.state_digest(module.state_dict())["sha256"] for name, module in linears}
            base = I.apply_intervention(self.alg.policy, self.alg.optimizer, "actor_last_layer_reset",
                                        H.reset_seed(bound_manifest["seed"]), torch)
            row = {**base, "schema": "actor_reset_budget_intervention_v1", "arm": bound_manifest["arm"],
                   "critic_state_before": critic_before,
                   "critic_state_after": T.state_digest(self.alg.policy.critic.state_dict())["sha256"],
                   "std_state_before": std_before,
                   "std_state_after": T.state_digest(self.alg.policy.std)["sha256"],
                   "actor_linear_states_before": actor_before,
                   "actor_linear_states_after": {
                       name: T.state_digest(module.state_dict())["sha256"] for name, module in linears}}
            helper.validate_intervention(row, bound_manifest, runtime["source_checkpoint_state"])
            helper.write_new(Path(bound_manifest["output_dir"]) / "intervention.json", row)
            runtime["budget_intervention_applied"] = True
            return result

        runner_class.load = measured_load
        return originals

    R._install_runner_hooks = install
    try:
        receipt = R.execute(manifest_path, H)
    finally:
        R._install_runner_hooks = original_install
    if not H.read(Path(manifest["output_dir"]) / "intervention.json"):
        raise RuntimeError("개입 readback이 없다")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    execute(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
