# RL Triage Agent

**Tagline:** Find why a robot policy stopped learning, inspect the suspected mechanism, and test a registered repair.

**Track:** Physical AI

## Inspiration

A simulated robot can survive an entire evaluation while barely following a motion command. A healthy-looking reward curve or a plausible language-model explanation is not enough to decide what to fix. We wanted a workflow that connects a diagnosis to a small executable inspection, a bounded correction, and a separate behavioral evaluation.

## What it does

RL Triage Agent investigates failed reinforcement-learning runs in NVIDIA Isaac Lab. It packages training telemetry and registered observations for a Nemotron model, validates the model's structured decision, and routes approved inspections through an execution contract. A correction must match the registered target and recovery mode before retraining can run. Fixed command-tracking evaluations decide whether behavior recovered.

The demo replays a recorded Go2 run with an incorrectly applied PPO epoch count. Nemotron first selected a reward inspection, which returned normal. The registered sequence continued until optimizer instrumentation found four steps per update where twenty were expected. After an approved epoch restoration and restart with a different training seed, the corrected policy passed the fixed behavioral evaluation. Its linear velocity error was 0.148 m/s, compared with 1.169 m/s for the faulty policy and 0.146 m/s for the normal control.

Separate compound-defect development runs show why the loop checks residual failure: seeds 2120 and 2121 required two corrections, in opposite orders, before passing evaluation. The public replay exposes the original JSON records and verifies their SHA-256 hashes. The simulation clips show a newly captured normal/F1 comparison, not the repaired R1 policy.

## How we built it

The project combines Python, Isaac Lab, Isaac Sim, RSL-RL/PyTorch training, NVIDIA hosted Nemotron inference, structured response validation, and explicit approval and resource contracts. The R1 demo used `nvidia/nemotron-3-super-120b-a12b`. A newer Nemotron 3.5 investigation path has separate development evidence; its ongoing execution is not substituted for the completed R1 run.

Isaac Lab simulation and the fresh normal/F1 render capture ran on **Brev-provisioned Nebius L40S compute**. Hosted NVIDIA model inference is a separate service. We have not made a Token Factory runtime call and do not describe issued promotional credits as redeemed or consumed. The submission includes the compute registration and capture provenance for review.

## Challenges we ran into

Training failure and recovery are different questions. Correct update counts do not guarantee a useful policy, and fixing a configuration does not guarantee that continuing an existing checkpoint will recover. We separated mechanism inspections, correction scope, restart rules, and behavioral acceptance criteria. We also found that a renderer could show stale robot geometry; the submitted normal/F1 clips come from the subsequent synchronized capture, with older affected videos excluded.

## Accomplishments

We connected recorded diagnosis, inspection, approval, retraining, and evaluation artifacts in a completed R1 development run. We added optimizer update-count and PPO gradient checks, correction-scope checks, and residual diagnosis after an insufficient first repair. The judge package runs a CPU-only evidence verifier and an English replay without credentials, a GPU, or model calls.

## What we learned

A model's first hypothesis can be wrong while a constrained investigation still finds a measurable discrepancy. The useful artifact is the chain from observation to test and from test to evaluation. A final success label is more meaningful when its checkpoint, source, approval history, and evaluation can be inspected together.

## What's next

Complete the current separately registered investigation and repair workflow, then evaluate it on a broader preregistered set of faults and robots. Measure when additional visual observations change inspection selection. Add a separately registered Token Factory provider once model availability and credentials are confirmed.

## Built with

Python; NVIDIA Isaac Lab; NVIDIA Isaac Sim; NVIDIA Nemotron; RSL-RL; PyTorch; NVIDIA NeMo Agent Toolkit; OpenAI-compatible model client; NVIDIA Brev; Nebius L40S compute; JavaScript; SHA-256.

## Scope of the results

These are known-defect development cases, including designs informed by earlier results. They do not establish unseen-fault generalization, model superiority, or compute savings. The compound runs' final checkpoints reproduce their same-seed normal controls. U1 seed2062 is an expanded-scope confirmation with no model decision loop.
