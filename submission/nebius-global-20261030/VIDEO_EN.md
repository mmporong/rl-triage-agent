# Video handoff

**Title:** RL Triage Agent: inspect a robot learning failure, then verify the repair

**Description:**

RL Triage Agent connects recorded Nemotron diagnosis, small mechanism inspections, registered corrections, retraining, and fixed behavioral evaluation in Isaac Lab.

This 2:50 demo contains 1:48 of the working local evidence replay and 25 seconds of newly synchronized normal/F1 simulation captures. The replay reads and hashes original public result records. It is not live GPU training. The simulation clips are normal/faulty policies, not a repaired-policy comparison.

The R1 example is one known-defect development run (source seed2071, restart3071). The model selected the first reward probe; registered follow-up inspections found an optimizer update-count discrepancy. Separate v2 compound cases2120/2121 required two corrections. These results do not establish unseen-fault generalization, model superiority, or compute savings.

Compute: Brev-provisioned Nebius L40S. Model inference: NVIDIA hosted Nemotron. No Token Factory runtime call is claimed. Speech is synthesized offline in English. No music is used.

Code and test-build instructions: https://github.com/mmporong/rl-triage-agent/tree/main/submission/nebius-global-20261030

00:00 The problem
00:10 Normal simulation
00:22 F1 faulty simulation
00:35 Recorded application replay
00:53 Model hypothesis and inspections
01:11 Optimizer counts
01:29 Approval and correction
01:47 Fixed behavioral evaluation
02:05 Residual diagnosis
02:23 Platform disclosure
02:38 Reproduce on CPU

**Files:** `RL_Triage_Agent_Nebius_Demo.mp4`, burned English captions, matching `demo.en.srt`, and `narration.json`.

**Publication status:** Local candidate. The repository path above becomes valid after the project owner integrates these new files. Public YouTube upload and final Devpost submission have not been performed by the package worker.
