# Isaac Lab RL Training Triage Agent

| Catalog field | Value |
| --- | --- |
| Description | Investigates Isaac Lab RL training failures from telemetry, confirms the cause with short approved probes, applies a registered correction with an approved retraining run, and judges recovery with a fixed behavior evaluation. It can also assess a single run without a healthy reference. |
| Industry | 🤖 Physical AI |
| Requirements | Linux or WSL2 · Docker · OpenShell 0.1.1 · NVIDIA API key (build.nvidia.com) · Isaac Lab 2.1.1 + RTX GPU only for re-running experiments |
| OpenShell | 0.1.1 |
| Collection | Hackathon |

A robot RL run can look fine on its training curves and still fail at the task. In the reward-frame fault below, episodes last almost to the time limit and the training reward rises, but the Go2 ignores velocity commands: its tracking error equals that of standing still. Was it the reward, the actuator, exploration noise, PPO settings, physics or termination?

This agent reads the training telemetry, ranks the likely mechanisms with Nemotron, and asks for one short Isaac Lab probe at a time to confirm or rule them out. Once a mechanism is confirmed, it maps it to a registered code correction and a recovery mode (continue training or retrain), and a fixed 26-command behavior evaluation decides whether the policy recovered. Probes and retraining run only after an approval is recorded, and each approval is consumed once.

[![Failed, retrained and healthy Go2 policies in Isaac Sim](docs/media/go2-e2e-e2e_reward_s2030_20261013/comparison.gif)](docs/media/go2-e2e-e2e_reward_s2030_20261013/comparison.mp4)

*Seed 2030, reward-frame fault: the failed policy stands still, the policy retrained after the registered correction walks, and the healthy policy is shown for reference. Same Play settings for all three. Recovery is judged by the fixed evaluation, not by this video.*

[Presentation replay](https://mmporong.github.io/rl-triage-agent/replay/) · [Interactive architecture diagram](https://mmporong.github.io/rl-triage-agent/?view=single-dark) · [한국어 요약](#한국어-요약)

## What the agent does

1. **Notice.** A tracking alarm and a fixed behavior evaluation show that the policy does not follow commands.
2. **Rank.** Nemotron reads the telemetry and ranks mechanisms such as reward, actuator and exploration.
3. **Probe.** It proposes one short Isaac Lab probe (about 20 seconds). After approval the probe runs, and frozen rules rule mechanisms in or out. The loop repeats until one mechanism remains or it stops.
4. **Correct.** A confirmed mechanism maps to one registered correction and a recovery mode. For reward faults, a scope check can flag error terms that the registered correction does not cover, before any retraining.
5. **Verify.** After approval the run is retrained, and the same fixed evaluation compares it with the paired healthy reference.

Unsupported mechanisms and insufficient evidence end as HOLD without a correction request.

## Key results

| Question | Result | Evidence |
| --- | --- | --- |
| Does the full loop recover a failed run? | Three runs on new seeds recovered under preregistered criteria: a reward-frame fault (seed 2030), a motor torque-speed fault (seed 2033) and a reward-frame fault decided through the NeMo Agent Toolkit tool-calling path (seed 2034). In the motor run the first judgment ranked reward first; three approved probes rejected reward and exploration and confirmed the actuator. A fourth motor-fault run (seed 2046) added a validation-feedback retry to both model decisions and also recovered; both first answers were valid, so no retry was used. | [End-to-end demonstration](docs/E2E-REWARD-DEMO.md) |
| Does the loop carry over to a second robot? | On ANYmal-C flat, a robot with an LSTM actuator network and push and center-of-mass randomization during training, the frozen probes, fixed evaluation contract, registered corrections and model decision path were reused; only the task id, the training and evaluation entry points and the packet's task text changed, and the fixed evaluation turns the two randomization events off. Three healthy runs set the ANYmal-C judgment band, with no false alarm from the probes or the tracking alarm between them. A reward-frame fault on a new seed (2047) was then confirmed with one approved probe, and the retrained policy recovered under the preregistered criteria. | [ANYmal-C port](docs/ANYMAL-C-PORT.md) |
| Can it flag an incomplete correction, and can a wider registration recover the run? | For a reward fault ported from UniLab PR #1254, the scope check flagged an uncovered yaw-tracking term, and the original correction did not recover. After seeing that result, the designers registered a correction restoring all four affected reward functions. The failed source policy on seed 2051 recovered after a corrected restart on seed 3051. A preregistered confirmation on new source seed 2062 also recovered after a corrected restart on seed 3062. The NONE controls stayed healthy in both runs. These development confirmations did not run the model decision loop. | [Scope extension](docs/U1-SCOPE-EXTENSION.md) · [Seed 2051](evals/results/u1_scope_extension_s2051_20261015/summary.json) · [New seed 2062](evals/results/novel_u1_scope_confirmation_s2062_20261016/summary.json) |
| Fix and continue training, or retrain? | After the reward-frame correction, continued training and continued training with reset action noise did not recover on two seeds; retraining from scratch did. | [Recovery-mode comparison](docs/TRAINING-CORRECTION.md#회복-방식-비교-v4) |
| Can a partial reset recover the reward-fault policy? | Resetting the actor's last layer recovered the previously studied seed 2026 but not 2028; optimizer-state and critic resets recovered neither. In the preregistered comparison on new seeds 2063 and 2064, the actor reset recovered 2/2 while corrected continuation recovered 0/2. This meets the registered conditional-candidate rule. The global recovery policy is unchanged, and the failure mechanism remains unconfirmed. | [Reset comparison](docs/TRAINING-CORRECTION.md#새-seed-출력층-재설정-확인) · [Earlier seeds](evals/results/reset_interventions_v2_20261015/summary.json) · [New seeds](evals/results/novel_actor_reset_confirmation_20261016/summary.json) |
| How many probes does it take? | On 18 hidden-fault runs, all six probes confirm 17. Probes ordered from the agent's top-3 hypotheses confirm 14 with 2.1 probes on average and no wrong confirmation. | [Probe loop](#detailed-results) |
| What happens with faults it was not built for? | Three external-source faults ended unidentified with zero wrong confirmations. An optimizer fault ported from rsl_rl issue #38 (PPO ignoring the configured number of learning epochs) made the policy stand still on a new seed; all six probes read normal, and every probe ordering, including the model's, stopped without confirming a cause. | [External faults](docs/T1-EXTERNAL-FAULTS.md) · [R1 result](evals/results/upstream_r1_loop_20261014/result.json) |
| Can a new inspection detect the missed PPO fault? | After the original R1 result, a count inspection detected 4 optimizer steps and minibatches per update against the configured 20; all 22 non-R1 development runs matched 20. A preregistered confirmation on new seed 2060 reproduced the R1 count mismatch. NONE on seed 2060 and an additional NONE run on seed 2061 both matched 20. Each inspection used three fresh PPO updates, separate from the 300-update source training; the full model loop has not been rerun with this inspection. | [Optimizer inspection](docs/OPTIMIZER-OBSERVABILITY.md) · [Development set](evals/results/optimizer_observability_gpu_20261015/summary.json) · [New-seed confirmation](evals/results/novel_optimizer_confirmation_20261016/summary.json) |
| Are model decisions repeatable? | 55 repeated calls on 11 inputs: every valid output gave the same decision, but 6 outputs broke the output contract. In a separate run, returning the validation errors and asking once more left 1 of 55 invalid. Forcing a JSON schema made it worse (37 of 55 broke the contract). | [Repeated decisions](docs/N2-PASS-K.md) |
| Can it find the harmful change among several code changes? | Each case mixed one harmful change with seven harmless ones, and confirmation retrains were counted for bisection, one-by-one reverts, a probe-category rule and model rankings. Preregistered verdict: no comparison, because only two cases were valid; in the third, reverting the harmful change alone still left yaw tracking outside the healthy band. In the four model calls on the valid cases the harmful change was ranked first, but one output cited diff line numbers that do not exist and was rejected. | [Multi-change experiment](docs/D3-MULTI-CHANGE.md) |
| What does the new Nemotron add? | The decision model reads only text, so it could not look at policy playback. The new Nemotron endpoint (Nemotron 3.5 Super VL, released 2026-10-15) accepts images: in a preregistered comparison its observations of 18 Play clips matched the fixed-evaluation policy labels in 17 (healthy 7/7, failed 10/11). On the same inputs, made-up evidence citations fell from 6 to 2 and NAT tool-calling HTTP requests from 16 to 11; these three comparisons did not reach their preregistered resolution criteria. Video observations are auxiliary evidence; recovery is still judged by the fixed evaluation, and the existing model remains the default decision model. | [Comparison](docs/NEMOTRON-ENDPOINT-COMPARISON-V3.md) |
| Is it better than rules? | Not claimed. On the hidden-fault set the agent ranked the true mechanism first in 8 of 18 runs against 2 to 6 for the baselines, a small sample. On the first benchmark, simple rules solved all held-out cases. | [Detailed results](#detailed-results) |

[![U1 failed, corrected restart and healthy reference policies](docs/media/scope-transfer-u1-s2051-v1/comparison.gif)](docs/media/scope-transfer-u1-s2051-v1/comparison.mp4)

*U1 scope extension: failed policy (seed 2051), four-function correction followed by retraining (seed 3051), and healthy reference (seed 2051). [Video provenance](docs/media/scope-transfer-u1-s2051-v1/source_manifest.json).*

[![ANYmal-C failed, corrected restart and healthy reference policies](docs/media/scope-transfer-anymal-s2047-v1/comparison.gif)](docs/media/scope-transfer-anymal-s2047-v1/comparison.mp4)

*ANYmal-C: failed policy (seed 2047), corrected retraining (seed 3047), and healthy reference (seed 2047). All three policies in each comparison use the same Play settings. The videos preserve the linked fixed-evaluation labels and do not supply a new recovery judgment. [Video provenance](docs/media/scope-transfer-anymal-s2047-v1/source_manifest.json).*

The earlier [six-policy recovery-mode video](docs/media/go2-correction-reward_v4_s2026_20261013/comparison.mp4) compares continuation, noise reset and retraining on seed 2026. Its retrained checkpoint is byte-identical to the original healthy checkpoint, so that recovery confirms restoration of the code path. [Result and video replay](docs/replay/additional-checks.html).

**What these results do not show.** Faults, probes and corrections were written by the same team, and each result comes from two flat-terrain locomotion tasks (mostly Go2; one fault demonstration on ANYmal-C) and a few seeds. They are integration evidence, not evidence of diagnosis accuracy, generalization, an advantage of the model over rules, or GPU savings. Approvals were operator approvals delegated by the user; approver strings are recorded, not authenticated.

## At A Glance

| Question | Answer |
| --- | --- |
| Category | Community Recipe |
| Contributor or provenance | Team Mollet (Korea Agentic AI Hackathon 2026) |
| Use this when | An Isaac Lab RSL-RL training run fails, with or without a known healthy reference |
| You will get | Ranked hypotheses with telemetry evidence and one next measurement or experiment. Reference-based workflows can register an experiment and obtain a verification verdict; single-run assessments are saved under `preregistrations/assessments/<case>.json`. |
| Runs on | Linux or Windows 11 + WSL2 (Ubuntu 24.04) with Docker; Isaac Lab on a Windows or Linux RTX host for retraining |
| Requires | NVIDIA API key for `nvidia/nemotron-3-super-120b-a12b`, OpenShell 0.1.1 gateway, `uv` |
| Verified on | Windows 11 + WSL2 Ubuntu 24.04, Docker 29.8.1, OpenShell 0.1.1, NeMo Agent Toolkit 1.9.0, Isaac Sim 4.5.0 / Isaac Lab 2.1.1, RTX 3060 12 GB |
| Evidence level | live end-to-end |
| Support and maturity | Best-effort community support; hackathon prototype |
| External access, data, and actions | Sends telemetry summaries, selected configuration facts when supplied, and agent messages to `integrate.api.nvidia.com` (NVIDIA API). Tools write assessments, preregistrations and analysis scratch files in the workspace. Approved verification writes evaluation records and new runs under the Isaac Lab log directory. |
| Start here | [Quickstart](#quickstart) |
| Confirm success | [Verification](#verification) |

## How it works

[Open the interactive diagram](https://mmporong.github.io/rl-triage-agent/?view=single-dark) · [Archify JSON source](docs/diagrams/rl-triage-unified.architecture.json)

The default sandbox has no bridge submission tool or access; the host submits the approval request. The diagram also shows the offline permission checks by `evals/prove_policies.py`, separately from the agent runtime.

| NVIDIA component | Role here |
|---|---|
| Nemotron 3 Super 120B (build.nvidia.com) | Mechanism ranking, probe choice, correction proposals |
| NeMo Agent Toolkit 1.9 (`nvidia-nat`) | Agent workflows (`configs/*.yml`) and tools (`src/rl_triage/nat_functions.py`, `src/rl_triage/nat_single_run.py`); the tool-calling decision path of the seed 2034 demonstration |
| OpenShell 0.1.1 | Sandbox image, Landlock filesystem policy, L7 REST network policy, NVIDIA provider (the agent only sees a placeholder key) |
| openshell-prover 0.1.1 | SMT check of every requested permission against `policies/site_boundary.yaml` |
| NVIDIA Skills | `skills-lock.json`; own skill `skills/isaaclab-rl-triage` in NVIDIA skill format; the checklist of the Isaac Lab `isaaclab-debugging-rl-training` skill (experimental) validates model correction proposals (`evals/correction_model_call.py`) |
| Isaac Lab 2.1.1 / Isaac Sim 4.5 | Fault injection, probes, retraining, fixed evaluation and recorded clips |

After a diagnosis is confirmed, [`evals/correct.py`](evals/correct.py) turns the mechanism into at most one correction request. The [correction registry](src/rl_triage/correction_registry.py) maps the reward, actuator and termination mechanisms to one registered correction each and sets the recovery mode with a preregistered rule. An append-only [correction ledger](src/rl_triage/correction_ledger.py) binds each request to the loop state, correction and recovery mode, refuses duplicate effects, and consumes one recorded approval before a run writes its receipt. A model correction proposal is validated and attached as evidence; the request always carries the rule-based correction. The rule was visible to the model, so agreement between the two is not independent evidence.

This recipe follows the **OpenShell path** described in the NemoClaw docs ("you use OpenShell as the platform and supply your own container, policy YAML, provider setup"), because the workload is a custom analysis image rather than a NemoClaw reference harness. NemoClaw's current blueprint pins OpenShell 0.0.116, while the prover features used here are in OpenShell 0.1.x. On this Windows host the NemoClaw onboarding stopped at preflight with `host.platform.wsl_native_docker_unqualified`, so the NemoClaw path needs Docker Desktop. The skill in `skills/isaaclab-rl-triage` is in NVIDIA skill format for `nemohermes <name> skill install`, but that path is not verified here.

## Detailed results

**First benchmark.** 10 injected faults (reward, actuator, exploration, optimizer, physics, termination) on `Isaac-Velocity-Flat-Unitree-Go2-v0`, 1024 envs, 100 iterations. Each case mixes 1 harmful and 2 benign hydra overrides, shuffled with a secret seed.

| Task | Set | RL Triage Agent | Single-prompt Nemotron 3 Super (summary-only input) | Random |
|---|---|---:|---:|---:|
| A. Which change broke it? (change list + telemetry) | dev (seeds 42, 7) | 20/20 | 20/20 | 33% |
| B. Unknown cause: which mechanism? (telemetry only) | dev (seeds 42, 7) | 10/20 | 8/20 | 17% |
| B. Unknown cause | held-out (seed 123) | 6/10 | 2/10 (not significant, n=10) | 17% |
| B. Unknown cause, top-2 | held-out (seed 123) | 6/10 | 6/10 | 33% |
| A. Which change broke it? | held-out (seed 123) | 10/10 | 10/10 | 33% |

Deterministic baselines without a model solved the same held-out set: fixed rules on normalized telemetry, the nearest dev case and a config diff each scored 10/10 top-1 ([`p0a2_holdout_20261008`](evals/results/p0a2_holdout_20261008/)). The held-out seed reuses the same 10 fault templates, so this set tests recognition of known templates, and we withdraw the claim that the agent diagnoses better than a deterministic rule set on it. At 100 iterations even the healthy reference barely follows velocity commands, so later experiments use the Isaac Lab default budget (4096 envs × 300 iterations), where the reference walks (tracking error 0.15 m/s against 1.18 m/s standing still).

**Hidden-fault set.** Six faults that change code or runtime state rather than the saved config (reward frame, DC-motor curve, sampling noise, frozen critic, runtime friction, early time-out), one per mechanism, trained on seeds 2026 to 2028. All 18 fault runs fail the fixed evaluation, and their saved params differ from the reference only by run name ([`replay_p0c_20261009`](evals/results/replay_p0c_20261009/)).

| Method | Input | Top-1 | Top-2 |
|---|---|---:|---:|
| Label frequency / config diff | none / params | 3/18 | 6/18 |
| Frozen rules from the first benchmark | telemetry | 5/18 | 9/18 |
| Nearest case from the first benchmark | telemetry + its labels | 6/18 | 6/18 |
| Single prompt, full time series | same telemetry as the agent | 2/18 | 8/18 |
| RL Triage Agent | telemetry + analysis tools | 8/18 | 12/18 |

Six fault types over three seeds is a small sample and the rules came from a different training budget, so this is not a claim of superiority. The agent's analysis code ran under a Landlock sandbox that cannot read the answer key.

**Probe loop.** Instead of retraining, the loop ([`evals/loop.py`](evals/loop.py), [`p1a_loop_compare_20261009`](evals/results/p1a_loop_compare_20261009/)) proposes one measurement probe at a time: noise, critic learning, reward recomputation, torque saturation, runtime physics and episode length. On the 18 hidden-fault runs, all six probes confirm 17; probes ordered from the agent's top-3 hypotheses confirm 14 with 2.1 probes on average and no wrong confirmation. The probes and the faults were written by the same person, so this shows that the loop works, not that it generalizes.

**External faults and healthy inputs.** With a frozen probe set, three external-source faults ended as `none_supported` with zero wrong confirmations ([protocol and results](docs/T1-EXTERNAL-FAULTS.md)). A no-intervention gate returned `do_not_open_loop` for all three healthy development inputs, with no model calls or probes ([protocol and results](docs/T2-NO-INTERVENTION.md)).

**Security** (`evals/results/policy_proofs.json`, `evals/results/sandbox_kernel_tests.txt`):

| Threat | Control | Evidence |
| --- | --- | --- |
| Agent edits configs, joint limits or rewards to pass the gate | No config path in the sandbox filesystem policy; bridge has no config API (405) | kernel test PASS; prover counterexample `PUT /config/` |
| Checkpoint or robot model exfiltration | Only `POST /v1/chat/completions` on `integrate.api.nvidia.com` | kernel test PASS; prover counterexample `huggingface.co:443` |
| GPU retraining abuse | Preregistration required + human CLI approval; no HTTP approval route | `tests/test_security.py` |
| API key leakage | OpenShell provider injects the key at egress; the sandbox sees `openshell:resolve:...` | observed in sandbox run |
| Writing reference data | Landlock read-only; prover returns `unsupported` → fail-closed reject | kernel test PASS |

## Quickstart

```bash
git clone https://github.com/mmporong/rl-triage-agent.git "$HOME/rl-triage-agent"
cd "$HOME/rl-triage-agent"
uv sync
uv run pytest -q tests                                   # unit + app-level security tests
uv run python bench/build_workspace.py 42                # benchmark telemetry + public reference params -> workspace/seed42
export TRIAGE_WORKSPACE=$PWD/workspace/seed42            # tools read this workspace
export NVIDIA_API_KEY=nvapi-...                          # build.nvidia.com
uv run nat run --config_file configs/triage_workflow.yml \
  --input "Triage failed training case_id=c01. Find the root-cause change and register the next experiment."
```

Inside OpenShell (WSL2/Linux with Docker):

```bash
openshell provider profile import -f policies/providers/nvidia.yaml --global
docker build -f docker/Dockerfile.sandbox -t rl-triage-sandbox:0.1 .
bash scripts/sandbox_up.sh 42                            # provider + workspace image + sandbox with policies/triage_agent.yaml
openshell sandbox exec -n rl-triage --env HOME=/tmp -- bash -c 'cd /sandbox/app && nat run --config_file configs/triage_workflow.yml --input "Triage failed training case_id=c03."'
```

Human-approved retraining (Isaac Lab host):

```bash
uv run python -m rl_triage.eval_bridge serve             # agent side can only submit
uv run python -m rl_triage.eval_bridge list              # human
uv run python -m rl_triage.eval_bridge approve <id>      # human: retrain with one variable changed -> verdict
```

### Assess one run without a healthy reference

Prepare a separate workspace containing:

```text
workspace/single_run/
  reference/telemetry.json
  cases/<case_id>/telemetry.json
  cases/<case_id>/configuration.json  # required when context.json is present
  cases/<case_id>/context.json        # optional source hashes and selected facts
```

Set `reference/telemetry.json` to exactly `{"reference_status":"absent","summary":{},"series":{}}`. The case telemetry must contain `summary` and `series` objects. Each summary metric uses `n`, `first`, `last`, `min`, `max`, `mean_first_20pct` and `mean_last_20pct`; each series maps a metric name to its scalar samples. To provide reward, observation or sensor definitions, follow [the configuration evidence format](docs/SINGLE-RUN-CONTEXT.md).

```bash
cd "$HOME/rl-triage-agent"
export TRIAGE_WORKSPACE="$PWD/workspace/single_run"
export TRIAGE_ANALYSIS_SANDBOX=required
uv run nat run --config_file configs/triage_single_run.yml \
  --input "Assess failed run case_id=<case_id>. Task context: <task and observed failure>."
```

The assessment is saved once to `preregistrations/assessments/<case_id>.json` and marked `hypotheses_only`: one run alone cannot establish normality, recovery or a confirmed cause. This workflow makes model API calls but does not start Isaac or retrain.

## Verification

**Evidence level:** live end-to-end

```bash
uv run python evals/prove_policies.py
OPENSHELL_SANDBOX=1 python -m pytest -q tests/sandbox    # inside the sandbox
uv run python evals/run_eval.py --task blind --seed 123 --mode both --tag heldout
```

**Expected result:**

```text
triage_agent           result=within_boundary   gate=human_review PASS
bad_exfil_checkpoint   result=exceeds_boundary  gate=auto_reject  PASS
bad_patch_config       result=exceeds_boundary  gate=auto_reject  PASS
bad_write_reference    result=unsupported       gate=auto_reject  PASS
ok_request_eval        result=within_boundary   gate=human_review PASS
8 passed                                                  # tests/sandbox
```

**This verifies:** permission proofs, kernel enforcement inside a live OpenShell sandbox, agent scores on recorded Isaac Lab telemetry, and the approve → retrain → verdict loop on a real Isaac Lab run.

**This does not verify:** faults that only appear in long training (>100 iterations), rough terrain, robots other than Go2 and ANYmal-C, manipulation tasks, real robots, or multi-sandbox fleets.

### Offline replay (no API key, network or GPU)

A clean checkout can recompute the stored scores from public files only. The replay and the workspace builder use the Python standard library; they do not import the model client or NeMo Agent Toolkit.

```bash
python evals/replay.py evals/results/blind_v1 evals/results/heldout_blind evals/results/heldout_changes \
  --traces evals/results/traces             # add --tag <new-folder> to save evals/results/<new-folder>/replay.json
python bench/build_workspace.py 123 --blind # public params from bench/reference/; the blind leak check runs automatically
python evals/replay.py evals/results/p0a2_holdout_20261008
```

For each (task, seed, case, mode), the last row with `infra_error: false` counts; every retry in the results files was triggered only by an infrastructure error (overload, empty or truncated response), never by a wrong answer. Truth, top-1 and top-2 are recomputed from the public answer keys, and the replay fails if any stored value differs. `--traces` fails on answer-file or `../..` markers in saved tool calls. Expected:

```text
evals/results/blind_v1: rows=47 infra=7 mismatches=0
  blind    agent         top-1 10/20  top-2 12/20  seeds=7,42
  blind    control       top-1 8/20  top-2 10/20  seeds=7,42
evals/results/heldout_blind: rows=22 infra=2 mismatches=0
  blind    agent         top-1 6/10  top-2 6/10  seeds=123
  blind    control       top-1 2/10  top-2 6/10  seeds=123
evals/results/heldout_changes: rows=22 infra=2 mismatches=0
  changes  agent         top-1 10/10  top-2 10/10  seeds=123
  changes  control       top-1 10/10  top-2 10/10  seeds=123
evals/results/traces: trace files=68 leak hits=0
status=pass
```

The [presentation replay](docs/replay/index.html) reads only committed public results. `python evals/build_replay_page.py` checks each source file against its commit, replays the public correction bundles, binds each comparison video to the hash in its recording manifest and writes `docs/replay/data.js`; `--check` reports whether `data.js` matches the committed results. Correction and recovery experiments publish their own replay commands in the linked documents.

## Development records

Earlier development steps, each with its own protocol, results and limits:

- [Grounded single-run workflow](docs/GROUNDED-SINGLE-RUN.md): checks structured hypothesis premises against source-bound evidence and registers one read-only next check.
- [Active diagnosis driver](docs/ACTIVE-DIAGNOSIS.md) and [its integration results](docs/ACTIVE-DIAGNOSIS-V2-RESULT.md): a fresh Nemotron probe choice connected to an approval ledger and a fixed Isaac measurement; 0 of 8 model cases met the measurement-dependent judgment-change criterion.
- [Training correction by continued training](docs/TRAINING-CORRECTION.md): three development cases (reward frame, motor curve, timeout) returned `no_recovery` after 300 additional updates; also the policy-free motor-curve inspection.
- [Callback correction feasibility](docs/RECOVERY-FEASIBILITY.md): the fault conditions also passed the behavior checks, so this correction is not used as a recovery demonstration.
- [Live approval-loop records](docs/LIVE-LOOP-EVIDENCE.md): eight cases after twelve approved probes, with acceptance limits.
- [Evidence video](docs/media/evidence-v2-20261011-recovery-q4/evidence.mp4) and its [manifest](docs/media/evidence-v2-20261011-recovery-q4/manifest.json): failure observation, probe selection, measurement and HOLD.

## Limitations

- Benchmark circularity: faults, probes, corrections and the agent were built by the same team. Mitigations: benign changes mixed in with a secret seed, prompts committed before each evaluation, same-model single-prompt baseline, held-out seeds, and a small number of faults ported from external sources.
- Two flat-terrain locomotion tasks: most results use `Isaac-Velocity-Flat-Unitree-Go2-v0`; `Isaac-Velocity-Flat-Anymal-C-v0` has three healthy runs and one reward-fault demonstration.
- The main loop's registered corrections cover three mechanisms. Other faults end as HOLD or trigger a scope warning. The ported UniLab fault first triggered that warning; a four-function correction registered after seeing the result recovered follow-up source seeds 2051 and 2062 outside the model decision loop.
- Single model calls sometimes break the output contract; a validation-feedback retry reduced this on already seen inputs and is not yet part of the demonstrations.
- The skill's mechanism table (`skills/isaaclab-rl-triage/SKILL.md`) was written after looking at benchmark telemetry; scored runs do not load the skill.
- Network rules apply to child processes, so code run by `run_analysis` can still reach the allowed NVIDIA inference path.

## 한국어 요약

Isaac Lab 로봇 강화학습이 실패했을 때 원인을 찾아 고치는 에이전트입니다. 학습 기록을 읽어 원인 후보를 고르고(Nemotron), 20초 정도 걸리는 검사로 원인을 확인한 뒤, 미리 등록한 수정과 재학습을 승인받아 실행합니다. 회복 여부는 26개 속도 명령으로 정책을 돌리는 고정 평가로 판정합니다.

- Go2의 새 seed에서 학습부터 수정·재학습까지 이어 본 네 번의 실행이 모두 회복했습니다. 모터 결함에서는 첫 판단이 보상을 의심했지만 검사 세 번으로 바로잡았습니다.
- 다른 로봇(ANYmal-C 평지 보행)에서도 같은 검사·평가·교정 코드를 그대로 썼습니다. 정상 학습 세 번으로 판정 기준을 정했고, 새 seed의 보상 결함 한 건을 검사 한 번으로 확정한 뒤 재학습으로 회복했습니다.
- 다른 프레임워크에서 옮겨 온 U1 보상 버그에서는 등록 교정의 범위 부족이 표시됐고, 해당 교정 뒤 재학습도 회복하지 못했습니다. 설계자가 이 결과를 본 뒤 보상 함수 네 항을 복원하도록 교정을 넓혔고, source seed 2051에서 재학습(seed 3051)으로 회복했습니다. 사전등록한 새 source seed 2062에서도 재학습(seed 3062)이 고정 평가를 통과했습니다. 두 실행의 NONE 대조는 정상을 유지했으며 모델 판단 고리는 사용하지 않았습니다. [근거](docs/U1-SCOPE-EXTENSION.md) · [2051 비교 영상](docs/media/scope-transfer-u1-s2051-v1/comparison.mp4)
- 보상 버그 교정 뒤 회복 방식을 비교한 v4에서는 이어 학습·탐색 잡음 재설정이 두 seed에서 회복하지 못했고, 처음부터 다시 학습하면 회복했습니다. [사실] actor 마지막 층 재설정은 기존 seed 2026에서 회복하고 2028에서는 미회복이었습니다. 이후 사전등록한 새 seed 2063·2064에서는 재설정이 2/2 회복, 같은 결함 체크포인트에서 출발한 이어 학습은 0/2였습니다. 이에 따라 **등록 조건부 후보**가 되었으며 전역 회복 정책은 자동 변경하지 않았습니다. [추론] 실패 메커니즘은 아직 미확정입니다. [근거](docs/TRAINING-CORRECTION.md#새-seed-출력층-재설정-확인)
- 동결 검사 여섯 종이 놓친 R1을 본 뒤 만든 계측 검사는 PPO가 설정한 갱신 횟수를 지키는지 셉니다. 기존 개발 측정 23건에서는 R1 한 건만 이상이었고 나머지 22건은 설정과 일치했습니다. 새 seed 2060의 R1에서도 update당 20회 대신 4회가 관측됐으며, NONE 2060·2061은 모두 20회였습니다. 각각 새 PPO update 세 번을 계측한 검사 결과이며, 모델 진단 고리를 재실행한 결과는 아닙니다. [근거](docs/OPTIMIZER-OBSERVABILITY.md)
- 숨은 결함 18건에서 에이전트가 고른 순서로 평균 2.1번 검사해 14건을 확정했고, 틀린 확정은 없었습니다.
- 코드 변경 8개 중 원인 하나를 찾는 실험은 유효 사례가 둘뿐이라 사전등록대로 방법 비교를 판정하지 않았습니다. 유효 사례의 모델 호출 네 번 모두 실제 원인을 1위로 냈지만, 한 번은 없는 코드 줄을 근거로 적어 무효 처리됐습니다.
- 새 Nemotron(3.5 Super VL, 2026-10-15 출시)은 이미지를 볼 수 있어, 로봇 영상 18개를 보고 걷는지·제자리인지 판단한 결과가 고정 평가와 17개 일치했습니다. 같은 입력에서 지어낸 근거 인용은 6건에서 2건으로 줄었습니다.
- 같은 입력으로 모델을 55번 호출하면 유효한 출력의 판단은 늘 같았지만 6번은 출력 형식을 어겼습니다. 틀린 곳을 알려 주고 한 번 더 물으면 1번으로 줄었습니다.

결함·검사·수정을 같은 팀이 만들었고 평지 보행 로봇 작업 둘(대부분 Go2, ANYmal-C는 결함 시연 한 건)과 적은 seed로 얻은 결과라, 진단 정확도·일반화·규칙 대비 우위·비용 절감은 주장하지 않습니다. 자세한 수치와 근거 파일은 위 영어 본문과 연결 문서에 있습니다.
