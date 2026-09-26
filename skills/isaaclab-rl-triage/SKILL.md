---
name: isaaclab-rl-triage
description: "Triage a failed or anomalous NVIDIA Isaac Lab RSL-RL training run against a healthy reference run. Identifies which config change caused the deviation from TensorBoard telemetry, checks the rejected-intervention ledger, and registers exactly one next experiment as a preregistration JSON for human approval. Use when a locomotion or manipulation policy fails its training gate after several config changes. Never edits configs, safety gates, joint limits or reward definitions."
---

# Isaac Lab RL training triage

## When to use

- An Isaac Lab (RSL-RL PPO) training run deviates from a known healthy reference run.
- Several config changes were made at once (hydra overrides), and you need the one that explains the deviation.
- You must hand a single, testable next experiment to a human, not a patched config.

## Inputs

A workspace with:

- `reference/telemetry.json` and `reference/params/{env,agent}.yaml` from the healthy run
- `cases/<case_id>/telemetry.json` (TensorBoard scalars, see `bench/extract_tb.py`) and `cases/<case_id>/case.json` (override list)
- optional `cases/<case_id>/ledger.json` (interventions already rejected)

## Procedure

1. `python -m rl_triage.cli list-changes <case_id>` and `python -m rl_triage.cli ledger <case_id>`.
2. `python -m rl_triage.cli overview <case_id>`: read ratios for reward, episode length, termination causes,
   velocity tracking error, value/surrogate loss, entropy, action noise std, and each reward term.
3. For each change, write the mechanism it would act through and the telemetry signature it predicts:

   | Mechanism | Typical signature |
   |---|---|
   | Termination / episode length | `Train/mean_episode_length` capped at a constant, `Episode_Termination/time_out` dominant |
   | Actuator / action scale | `Episode_Termination/base_contact` spikes, very short episodes, high `action_rate_l2` |
   | Physics (dt, mass) | base contact terminations, short episodes, unstable reward curve |
   | Reward sign / weight | one `Episode_Reward/*` term dominates or flips sign, total reward negative while episodes stay long |
   | Exploration | `Policy/mean_noise_std` collapses or explodes |
   | Optimizer (gamma, value loss) | `Loss/value_function` abnormal, reward stalls with long episodes |

4. Test the predictions with `series` or `analyze` (Python on `run` and `ref` dicts). Prefer numbers over the size of a config value.
5. Never propose an intervention that is in the ledger.
6. Register one experiment: `python -m rl_triage.cli prereg <case_id> --suspected chX --ranking ... --hypothesis ... --variable ... --gate ... --signature ...`.

## Safety boundary

Run inside an NVIDIA OpenShell sandbox with `policies/triage_agent.yaml`:
telemetry and reference params are read-only, writes go only to `scratch/` and `preregistrations/`,
network is limited to `POST /v1/chat/completions` on `integrate.api.nvidia.com`.
Requests for more access go through the OpenShell Policy Advisor and are checked by `openshell-prover`
against `policies/site_boundary.yaml`; unprovable or out-of-boundary requests are rejected.

## Evaluation

`evals/` in this skill points to the fault-injection benchmark (`bench/cases.json`, 10 faults x 2 seeds on
`Isaac-Velocity-Flat-Unitree-Go2-v0`). Score = top-1 root-cause accuracy vs single-prompt baseline and random (1/3).
