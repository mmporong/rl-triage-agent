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

The project combines Python, Isaac Lab, Isaac Sim, RSL-RL/PyTorch training, Nemotron inference, structured response validation, and explicit approval and resource contracts. The completed R1 repair shown in the demo used NVIDIA-hosted `nvidia/nemotron-3-super-120b-a12b`.

We also made runtime calls to NVIDIA Nemotron 3 Super and `nvidia/Nemotron-3.5-Super-VL` through Nebius Token Factory. The new provider contract preserves the exact model ID, request parameters, response bytes, and cited training facts. It passes model decisions through the existing investigation functions and inspection-request validator. In a separate preregistered development comparison, all eight replies produced valid host decisions: four HOLD decisions on identical-reference controls and four inspection proposals on the faulty training record. These proposals require a separately registered GPU execution phase; the completed R1 repair metrics belong to the earlier NVIDIA-hosted run.

Isaac Lab simulation and the fresh normal/F1 render capture ran on Brev-provisioned Nebius L40S compute. Its registered machine type is `gpu-l40s-d.1gpu-16vcpu-96gb`. Token Factory inference and Nebius simulation compute are separate platform uses, with separate request and artifact records.

## Nemotron 3.5 evidence

In a historical NVIDIA-hosted NIM development comparison on October 16, mean API-request waiting time was 39.66% shorter for diagnosis/correction requests (4.488 to 2.708 seconds) and 56.44% shorter for code-evidence requests (15.065 to 6.562 seconds). Each model made 33 and 24 requests, respectively: 11 and 8 existing inputs repeated three times. These are request-segment observations, not Token Factory speed measurements or reductions in GPU inspection and training time.

Nemotron 3.5 also added policy-frame observations: 18/18 replies met the output contract and 17/18 corresponded to fixed-evaluation groups (normal 7/7, failed 10/11). This is correspondence with behavioral evaluation classes, not human-labeled video or fault-diagnosis accuracy. Three separate seed-2080 observations were auxiliary; 3 Super made that demonstration's diagnosis and correction decisions.

In a separate NVIDIA-hosted NIM development run, 3.5 received six synchronized policy frames with training facts, selected the reward inspection, and cited the abnormal inspection outcome when proposing the registered reward correction. The first choice cited six telemetry facts and no frames. Retraining and 26-condition evaluation completed, but the new policy did not recover: its yaw error ratio was 1.61985 against a 1.1719 limit. This establishes a working inspection/correction connection without attributing better selection to images or claiming a new 3.5 recovery.

Quality changes were mixed. B2 citation errors fell from 6/24 to 2/24 and contract-valid correct top-1 answers rose from 15/24 to 18/24, while raw top-1 answers changed from 21/24 to 20/24 and B1 invalid replies from 4/33 to 5/33. The full preregistered quality criteria were not met. The evidence panel keeps these NIM measurements separate from the actual Token Factory replay and the earlier 3 Super recovery.

## Challenges we ran into

Training failure and recovery are different questions. Correct update counts do not guarantee a useful policy, and fixing a configuration does not guarantee that continuing an existing checkpoint will recover. We separated mechanism inspections, correction scope, restart rules, and behavioral acceptance criteria. We also found that a renderer could show stale robot geometry; the submitted normal/F1 clips come from the subsequent synchronized capture, with older affected videos excluded.

## Accomplishments

We connected recorded diagnosis, inspection, approval, retraining, and evaluation artifacts in a completed R1 development run. We added optimizer update-count and PPO gradient checks, correction-scope checks, and residual diagnosis after an insufficient first repair. The judge package runs a CPU-only evidence verifier and an English replay without credentials, a GPU, or model calls.

## What we learned

A model's first hypothesis can be wrong while a constrained investigation still finds a measurable discrepancy. The useful artifact is the chain from observation to test and from test to evaluation. A final success label is more meaningful when its checkpoint, source, approval history, and evaluation can be inspected together.

## What's next

Run the Token Factory inspection proposals under a fresh execution registration, approval, and resource lease, then measure inspection findings and any authorized repair with fixed behavioral evaluations. Extend the preregistered evaluation to new faults and robot tasks. Measure when additional training history or visual observations change inspection selection.

Both official model cards list the same nominal 120B total / 12B active class. The 3.5 C-RADIO vision encoder and connector support policy-frame inputs. Mamba/MoE and MTP are shared foundations; the cause of our hosted response-time difference was not isolated.

## Built with

Python; NVIDIA Isaac Lab; NVIDIA Isaac Sim; NVIDIA Nemotron 3 Super; NVIDIA Nemotron 3.5 Super VL; Nebius Token Factory; RSL-RL; PyTorch; NVIDIA NeMo Agent Toolkit; OpenAI-compatible model client; NVIDIA Brev; Nebius L40S compute; JavaScript; SHA-256.

## Scope of the results

These are known-defect development cases, including designs informed by earlier results. They do not establish unseen-fault generalization, model superiority, or compute savings. The compound runs' final checkpoints reproduce their same-seed normal controls. U1 seed2062 is an expanded-scope confirmation with no model decision loop.
