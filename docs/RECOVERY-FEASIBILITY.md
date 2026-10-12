# Callback correction feasibility

This development experiment asks whether a registered environment callback correction can improve the behavior of one existing frozen Go2 policy. It uses a previously inspected T1 input. It is not an unseen performance evaluation or an agent diagnosis result.

The experiment precedes further model integration because the proposed intervention must have a measurable behavioral effect before it can support a recovery demonstration.

## Fixed comparison

- Checkpoint: training seed 2028, 4096 environments, 300 iterations; SHA-256 `ed28e874db3e984b6946aea7fe2a193f5f0661cb3799e0f6aacc04cbecc9f647`.
- Three fresh simulator processes, in order: `fault_before` (X1), `fault_sham` (X1), `callback_fixed` (X1_REF).
- Each process uses the unchanged `fixed_eval_v1`: 1040 environments, 1000 control steps, evaluation seed 2026, 26 command conditions. No extra resets or retraining.
- Callback magnitudes, policy weights, observations, command grid and outcome thresholds stay fixed. Source files and configuration hashes are verified before and after execution.
- The existing numerical recovery criteria apply: the corrected condition walks; both fault conditions fail the same nonempty set of relative behavioral criteria; the two fault conditions satisfy the existing bidirectional equivalence bands.
- The first invalid arm stops the bundle. A consumed bundle cannot be retried with another tag. Infrastructure repair requires a new committed protocol and preservation of the original failure.

## Outcome wording registered before execution

| Result | Interpretation and next action |
|---|---|
| `recovery_conditions_met` | A behavioral effect is observed for this known policy and registered intervention under these evaluation conditions. Proceed to a separately registered agent integration; this is not evidence that an agent chose or diagnosed the intervention. |
| `no_failure_observed` | The fixed comparison does not produce the required failure contrast. Do not adopt this immediate callback correction as a recovery scene or increase the disturbance/reset count to make it succeed. |
| `no_recovery` | The corrected condition does not meet the walking criterion. Do not present it as recovery. A training-time repair would be a different experiment. |
| `control_invalid` | The sham comparison or failure consistency is insufficient. Do not attribute the difference to the correction. |
| `invalid` | Execution or provenance failed. Preserve the failure; no effect is counted. |

The host chooses the registered intervention. There are zero model calls, no synthetic assessment, no agent approval consumption and no entry into the existing grounded recovery flow. The separate runner reuses only its source checks, SDK evaluation and numeric validators. Parent receipts retain process identities, cleanup evidence, configuration and report hashes. The original active diagnosis denominator remains unchanged.

## Local execution

The original local checkpoint and pinned Windows Isaac installation are required. Preparation writes a new protocol once; commit it and the source before execution. The checked-in protocol is already prepared.

```powershell
cd $HOME\rl-triage-agent
.venv\Scripts\python.exe evals/run_recovery_feasibility.py execute --tag <new_tag>
.venv\Scripts\python.exe evals/run_recovery_feasibility.py verify --tag <same_tag>
```

Private raw artifacts are under `bench/private/recovery_feasibility/<tag>/`. A public result must contain portable reports and hashes without account paths or raw private logs. This experiment does not estimate GPU savings, diagnosis superiority, statistical significance or generalization.
