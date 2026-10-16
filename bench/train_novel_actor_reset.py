"""새 F1 checkpoint에서 continuation/actor-last-reset을 실행하는 child."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import recovery_reset_interventions as I  # noqa: E402
import train_with_correction as T  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
import novel_actor_reset_confirmation as H  # noqa: E402


def execute(manifest_path: Path):
    manifest = H.read(manifest_path)
    arm = manifest.get("arm")
    if arm not in H.ARMS:
        raise ValueError("등록되지 않은 actor reset arm")
    original_install = R._install_runner_hooks

    def install(bound_manifest, helper, runtime, **kwargs):
        originals = original_install(bound_manifest, helper, runtime, **kwargs)
        runner_class = originals[3]
        verified_load = runner_class.load

        def measured_load(self, *args, **load_kwargs):
            result = verified_load(self, *args, **load_kwargs)
            if runtime.get("novel_intervention_applied"):
                raise RuntimeError("개입이 두 번 적용됐다")
            import torch
            before_policy = T.state_digest(self.alg.policy.state_dict())["sha256"]
            before_optimizer = T.state_digest(self.alg.optimizer.state_dict())["sha256"]
            critic_before = T.state_digest(self.alg.policy.critic.state_dict())["sha256"]
            std_before = T.state_digest(self.alg.policy.std)["sha256"]
            actor_linears = [(f"actor.{name}".rstrip("."), module) for name, module in
                             self.alg.policy.actor.named_modules() if isinstance(module, torch.nn.Linear)]
            actor_before = {name: T.state_digest(module.state_dict())["sha256"] for name, module in actor_linears}
            if arm == "actor_last_layer_reset":
                base = I.apply_intervention(self.alg.policy, self.alg.optimizer, arm,
                                            H.reset_seed(bound_manifest["seed"]), torch)
            else:
                base = {
                    "schema": "recovery_reset_intervention_v1", "intervention": "none",
                    "reset_seed": None, "target_modules": [], "target_parameter_count": 0,
                    "optimizer_param_indices": [], "optimizer_moment_indices_present_before": [],
                    "optimizer_moments_present_after": None, "parameter_identity_preserved": True,
                    "policy_state_before": before_policy, "policy_state_after": before_policy,
                    "optimizer_state_before": before_optimizer, "optimizer_state_expected_after": before_optimizer,
                    "optimizer_state_after": before_optimizer,
                    "optimizer_param_groups_before": T.state_digest(self.alg.optimizer.state_dict()["param_groups"])["sha256"],
                    "optimizer_param_groups_after": T.state_digest(self.alg.optimizer.state_dict()["param_groups"])["sha256"],
                }
            row = {**base, "schema": "novel_actor_reset_intervention_v1", "arm": arm,
                   "critic_state_before": critic_before,
                   "critic_state_after": T.state_digest(self.alg.policy.critic.state_dict())["sha256"],
                   "std_state_before": std_before,
                   "std_state_after": T.state_digest(self.alg.policy.std)["sha256"],
                   "actor_linear_states_before": actor_before,
                   "actor_linear_states_after": {name: T.state_digest(module.state_dict())["sha256"]
                                                  for name, module in actor_linears}}
            helper.validate_intervention(row, bound_manifest, runtime["source_checkpoint_state"])
            helper.write_new(Path(bound_manifest["output_dir"]) / "intervention.json", row)
            runtime["novel_intervention_applied"] = True
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
