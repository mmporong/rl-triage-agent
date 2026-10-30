# Nemotron 3.5 evidence: source, phase and unit

This read-only view uses the accepted evidence at source commit b29a791db9f6397c4b04698b349ef68606273ae5. It adds no inference, GPU run or new performance measurement.

## API response time: NVIDIA-hosted NIM, October 16, 2026

| Development group | Requests per model | Inputs and repeats | 3 Super mean (s) | 3.5 mean (s) | Mean wait shorter |
|---|---:|---|---:|---:|---:|
| B1 diagnosis/correction | 33 | 11 existing inputs × 3 | 4.48794656 | 2.70807414 | 39.66% |
| B2 code evidence | 24 | 8 existing inputs × 3 | 15.06543961 | 6.56213270 | 56.44% |

Only the model ID changed within each pair. The measured unit is the model-request wall-clock segment. It excludes host validation, file processing, GPU inspections, learning and fixed evaluation. These are not Token Factory latency results, 57 independent new faults, or statistical evidence of general speed superiority. Both-valid pairs were a post-hoc subset: B1 26 pairs/39.91%, B2 15 pairs/51.84%, shown with the full results.

Quality was mixed: B1 invalid replies 4/33→5/33; B2 citation errors 6/24→2/24; contract-valid correct top-1 15/24→18/24; parsed top-1 ignoring the contract 21/24→20/24. The full preregistered acceptance criteria were not met. Sources: [accepted analysis](nim-evidence/latency_analysis.json), [57 paired requests](nim-evidence/latency_pairs.csv), [quality counts](nim-evidence/model_quality.csv).

## VL observations: a separate development cohort

The 18 policy observations had 18/18 valid responses and 17/18 correspondence with fixed-evaluation classes: normal 7/7, failed 10/11. This is not human-labeled video accuracy or cause-diagnosis accuracy. All B4 recordings were not independently covered by the later render audit. Three separate seed-2080 observations also corresponded, with used_for_decision=false; their diagnosis/correction model was 3 Super. Sources: [18 records](nim-evidence/vl_observations_18.json), [3 auxiliary records](nim-evidence/vl_aux_observations_3.json).

## Actual NIM3.5 inspection/correction connection

Six synchronized seed-2221 policy frames and training facts were supplied. The first P_reward choice cited six telemetry facts and zero frames. The actual inspection separated normal and abnormal; the follow-up model cited its abnormal outcome and proposed restore_body_frame_reward. A fresh restart at seed3221 trained for 300 updates and completed 26-condition evaluation. The outcome remained not_recovered because the yaw ratio 1.61985 exceeded 1.1719. Image benefit and a new 3.5 recovery are not established. Sources: [evaluation](nim-evidence/recovery_summary.json), [frame/selection scope](nim-evidence/vl_selection_and_render.json).

The completed R1 repair in the existing video used 3 Super. The actual eight-record Token Factory experiment is another provider/phase and has four HOLDs and four inspection proposals awaiting new authorization. These histories are not pooled.

## What changed in the model family

The official [3 Super card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) and [3.5 VL card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) both list a nominal 120B total / 12B active class. Mamba/MoE, MTP and NVFP4 are shared foundations. The 3.5 C-RADIO encoder and vision-language connector enable visual inputs; its VL/agentic post-training and serving work are relevant design changes. They are not isolated causes of the measured API wait difference. Server hardware, precision, queueing and enabled MTP/cache settings were not recorded. We do not explain this observation as a smaller model or a proven kernel-level speedup.

Detailed project source: [evidence report](https://github.com/mmporong/rl-triage-agent/blob/b29a791db9f6397c4b04698b349ef68606273ae5/docs/NEMOTRON35-EVIDENCE-REPORT.md), [official-source ledger](https://github.com/mmporong/rl-triage-agent/blob/b29a791db9f6397c4b04698b349ef68606273ae5/docs/NEMOTRON35-ARCHITECTURE-SOURCES.md), [source inventory](nim-evidence/sources.json). NIM route IDs and served raw aliases are recorded; served weight revisions were not attested.
