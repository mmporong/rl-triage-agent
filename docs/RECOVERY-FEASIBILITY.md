# Callback correction feasibility

This development experiment asks whether a registered environment callback correction can improve the behavior of one existing frozen Go2 policy. It uses a previously inspected T1 input. It is not an unseen performance evaluation or an agent diagnosis result.

The completed experiment did not produce the preregistered failure contrast. This immediate callback correction is not a recovery demonstration. A training-time correction remains a separate, untested proposal.

## Observed result

[Public result and audit summary](../evals/results/recovery_feasibility_20261012/result.json), recorded at source commit `322f49de39bc4b3f5211cf6c5812629198d0009f` under `recovery_feasibility_v3.json`:

| Condition | Falls / 1040 | Mean survival, s | Linear velocity RMSE, m/s | Yaw rate RMSE, rad/s |
|---|---:|---:|---:|---:|
| Fault callback | 1 | 19.981731 | 0.161387145 | 0.305408041 |
| Unchanged repeat | 1 | 19.981731 | 0.161387145 | 0.305408041 |
| Corrected callback | 0 | 20.000000 | 0.160025177 | 0.303961280 |

The corrected condition's linear and yaw errors were 0.8439% and 0.4737% lower. Both fault conditions still passed every registered relative behavioral check. The unchanged repeat matched the first condition, so the control was valid, but there was no failure to count as recovered: **`no_failure_observed`**.

All three simulator executions passed the local source, configuration, process and cleanup checks. A separate code reviewer verified the 36 recorded artifact hashes and the numerical verdict. The completed triplet took 167.812 seconds of parent-process wall time. Including the two preserved infrastructure failures, there were five SDK launches and 232.718 seconds of parent-process wall time; model calls were zero. These timings include launch, exit and cleanup and are not GPU kernel time or a savings estimate.

[Attempt a](../evals/results/recovery_feasibility_20261012/attempt_a.json) failed on tuple/list readback comparison before a behavioral report completed. [Attempt b](../evals/results/recovery_feasibility_20261012/attempt_b.json) completed one numeric report but failed to serialize a slice in the SDK readback. Both remain invalid. Committed serialization fixes preceded the final attempt; disturbance, reset count, policy and behavioral criteria were unchanged.

The separate [replay of all three historical training pairs](../evals/results/t1_pair_replay_20261012/result.json) compares different policies trained under fault and reference callbacks, then evaluated in the stock environment. Linear error changes were +4.137%, −14.109% and +12.886% reduction for seeds 2026, 2027 and 2028 respectively; yaw error reductions were 5.438%, 19.582% and 2.280%. These mixed results neither establish agent-selected repair nor predict success on a new training run.

## Public numerical replay

The three unmodified fixed reports are public. This command pins the published result by its LF-normalized hash, verifies report hashes, checkpoint and fixed-evaluation bindings, reaggregates all 26 command conditions per report and recomputes the registered verdict. It also checks the cost totals against the arm and preserved failure records. It uses no model or GPU:

```powershell
cd $HOME\rl-triage-agent
python evals/replay_recovery_feasibility.py
```

Expected: `status=pass`, `experiment_status=no_failure_observed`, `validated_reports=3`. This public replay does not authenticate private execution logs. The local execution audit and its hashes are recorded separately in the result; full raw process and account-path records remain private.

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

The original local checkpoint and pinned Windows Isaac installation are required. Preparation writes a new protocol once; commit it and the source before execution. The checked-in v3 protocol is already consumed on the original host and must not be rerun with another tag.

```powershell
cd $HOME\rl-triage-agent
.venv\Scripts\python.exe evals/run_recovery_feasibility.py execute --tag <new_tag>
.venv\Scripts\python.exe evals/run_recovery_feasibility.py verify --tag <same_tag>
```

Private raw artifacts are under `bench/private/recovery_feasibility/<tag>/`. A public result must contain portable reports and hashes without account paths or raw private logs. This experiment does not estimate GPU savings, diagnosis superiority, statistical significance or generalization.

The private `verify` command checks the exact recorded source commit, original inputs and pinned SDK files. A later documentation or result commit changes HEAD, so that verification must use the recorded source checkout and original bound paths. It passed before publishing these results; use the public numerical replay on later checkouts. Do not rewrite the saved request or manifest to match a newer HEAD.
