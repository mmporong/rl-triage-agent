# Isaac Lab RL Training Triage Agent

| Catalog field | Value |
| --- | --- |
| Description | Finds which config change broke an Isaac Lab RL locomotion training run from its telemetry, registers one next experiment for human approval, and verifies it by retraining — inside an OpenShell sandbox that cannot touch configs, safety gates or reward definitions. |
| Industry | ✨ Other |
| Requirements | Linux or WSL2 · Docker · OpenShell 0.1.1 · NVIDIA API key (build.nvidia.com) · Isaac Lab 2.1.1 + RTX GPU only for re-running experiments |
| NemoClaw | N/A |
| Harness | N/A |
| OpenShell | 0.1.1 |
| Collection | Hackathon |

A robot RL engineer changes several settings, trains, and the run collapses. In our own Go2 training ([isaac-walk-rl](https://github.com/mmporong/isaac-walk-rl)), three command-shrinking interventions were rejected in a row (rev28 to rev30), while the same joint-limit symptom had already been recorded 12 days earlier (rev11 Gate01, commit `973769a`, 2026-08-28; attributed in rev31, commit `55159ac`, 2026-09-09). Was it the reward, the actuator scale, exploration noise, PPO settings, physics or termination? This agent reads the training telemetry against a healthy reference run, tests hypotheses with its own analysis code, and hands back **one preregistered experiment** instead of a patched config. A human approves it, the eval bridge retrains in Isaac Lab with only that variable changed, and the result confirms or rejects the diagnosis.

> 한국어 요약은 [아래](#한국어-요약)에 있습니다.

## Screenshot

| Healthy reference (checkpoint at 100 iterations) | Injected actuator fault (action scale 0.25 → 1.5) |
|---|---|
| ![Go2 robots walking under velocity commands](docs/media/baseline_s42_play.gif) | ![Go2 robots collapsing and rearing](docs/media/c04_s42_play.gif) |

Both clips are Isaac Sim 4.5 off-screen renders of trained checkpoints (`bench/record_play.ps1`). The agent never sees video; it diagnoses from TensorBoard scalars only.

## At A Glance

| Question | Answer |
| --- | --- |
| Category | Community Recipe |
| Contributor or provenance | Team Mollet (Korea Agentic AI Hackathon 2026) |
| Use this when | An Isaac Lab RSL-RL training run deviates from a known healthy run after config changes, or for an unknown reason |
| You will get | A ranked root cause with telemetry evidence, a preregistered single-variable experiment (`preregistrations/<case>.json`), and optionally a retraining verdict |
| Runs on | Linux or Windows 11 + WSL2 (Ubuntu 24.04) with Docker; Isaac Lab on a Windows or Linux RTX host for retraining |
| Requires | NVIDIA API key for `nvidia/nemotron-3-super-120b-a12b`, OpenShell 0.1.1 gateway, `uv` |
| Verified on | Windows 11 + WSL2 Ubuntu 24.04, Docker 29.8.1, OpenShell 0.1.1, NeMo Agent Toolkit 1.9.0, Isaac Sim 4.5.0 / Isaac Lab 2.1.1, RTX 3060 12 GB |
| Evidence level | live end-to-end |
| Support and maturity | Best-effort community support; hackathon prototype |
| External access, data, and actions | Sends telemetry summaries and agent messages to `integrate.api.nvidia.com` (NVIDIA API). Retraining writes new runs under the Isaac Lab log directory. No other writes. |
| Start here | [Quickstart](#quickstart) |
| Confirm success | [Verification](#verification) |

## How it works

![RL triage: sandboxed diagnosis, human approval, single-variable retraining, and permission checks](docs/media/rl-triage-architecture.png)

The default sandbox has no bridge submission tool or access; the host submits the approval request.

[Archify JSON source](docs/diagrams/rl-triage.architecture.json) · [Interactive diagram (download HTML)](docs/diagrams/rl-triage.html)

![Offline policy proposal checks: within boundary goes to human review; exceeds boundary or unprovable requests are rejected](docs/media/rl-policy-architecture.png)

Policy proposals are checked offline by `evals/prove_policies.py`, separately from the agent runtime.

[Policy diagram JSON source](docs/diagrams/rl-policy.architecture.json) · [Interactive policy diagram (download HTML)](docs/diagrams/rl-policy.html)

| NVIDIA component | Role here |
|---|---|
| Nemotron 3 Super 120B (build.nvidia.com) | Planning, tool calling, reasoning |
| NeMo Agent Toolkit 1.9 (`nvidia-nat`) | Agent workflow (`configs/*.yml`), 7 registered tools (`src/rl_triage/nat_functions.py`) |
| OpenShell 0.1.1 | Sandbox image, Landlock filesystem policy, L7 REST network policy, NVIDIA provider (the agent only sees a placeholder key) |
| openshell-prover 0.1.1 | SMT check of every requested permission against `policies/site_boundary.yaml` |
| NVIDIA Skills | `skills-lock.json` (nemo-rl-auto-research, nemoclaw-user-guide, nemotron-policy-generator); own skill `skills/isaaclab-rl-triage` in NVIDIA skill format |
| Isaac Lab 2.1.1 / Isaac Sim 4.5 | Fault-injection benchmark, retraining for verification, recorded clips |

This recipe follows the **OpenShell path** described in the NemoClaw docs ("you use OpenShell as the platform and supply your own container, policy YAML, provider setup"), because the workload is a custom analysis image rather than a NemoClaw reference harness. NemoClaw's current blueprint pins OpenShell 0.0.116, while the prover features used here are in OpenShell 0.1.x. We also tried the NemoClaw path with the Hermes harness on a separate WSL2 distro (NemoClaw installer, OpenShell 0.0.116, `nemohermes onboard --non-interactive`). Onboarding stopped at preflight with `host.platform.wsl_native_docker_unqualified` ("Native Docker Engine inside WSL is not the qualified Docker Desktop integration"), so on this Windows host the NemoClaw path needs Docker Desktop. The skill in `skills/isaaclab-rl-triage` is in NVIDIA skill format for `nemohermes <name> skill install`, but that path is not verified here.

## Results

Benchmark: 10 injected faults (reward, actuator, exploration, optimizer, physics, termination) on `Isaac-Velocity-Flat-Unitree-Go2-v0`, 1024 envs, 100 iterations. Each case mixes 1 harmful and 2 benign hydra overrides, shuffled with a secret seed. The benchmark separates cleanly: benign-only runs 2/2 recovered, fault runs 20/20 not recovered.

| Task | Set | RL Triage Agent | Single-prompt Nemotron 3 Super (same inputs) | Random |
|---|---|---:|---:|---:|
| A. Which change broke it? (change list + telemetry) | dev (seeds 42, 7) | 20/20 | 20/20 | 33% |
| B. Unknown cause: which mechanism? (telemetry only) | dev (seeds 42, 7) | 10/20 (5 truncated at 4096 tokens) | 8/20 | 17% |
| B. Unknown cause | **held-out (seed 123), config v1.1** | **6/10** | 2/10 | 17% |
| B. Unknown cause, top-2 | held-out (seed 123) | 6/10 | 6/10 | 33% |
| A. Which change broke it? | held-out (seed 123) | 10/10 | 10/10 | 33% |

Task A dev: the agent column is the re-run with the OpenAI-compatible client (`evals/results/v1_openai_client/`) after the first NAT `nim` client dropped tool-call responses; the baseline column is the first run (`evals/results/v1_nim_client/`). Every retry in every results file was triggered only by an infrastructure error (overload, empty or truncated response), never by a wrong answer.

On the held-out set the agent was right and the baseline wrong in 4 cases (c01, c04, c05, c07), never the reverse; with n=10 this is not statistically significant (two-sided binomial p≈0.125). Both methods never ranked the true mechanism first for **physics** or **optimizer** faults, on dev or held-out. The baseline tends to answer "reward" from the summary table; the agent pulls the termination and action curves before deciding.

Loop verification (`evals/results/bridge_smoke.json`): reverting the true culprit recovered the run (episode-length ratio 0.999, reward ratio 1.024); reverting a benign change did not (0.05). A wrong diagnosis is caught by retraining.

Security (`evals/results/policy_proofs.json`, `evals/results/sandbox_kernel_tests.txt`):

| Threat | Control | Evidence |
|---|---|---|
| Agent edits configs, joint limits or rewards to pass the gate | No config path in the sandbox filesystem policy; bridge has no config API (405) | kernel test PASS; prover counterexample `PUT /config/` |
| Checkpoint or robot model exfiltration | Only `POST /v1/chat/completions` on `integrate.api.nvidia.com` | kernel test PASS; prover counterexample `huggingface.co:443` |
| GPU retraining abuse | Preregistration required + human CLI approval; no HTTP approval route | `tests/test_security.py` |
| API key leakage | OpenShell provider injects the key at egress; the sandbox sees `openshell:resolve:...` | observed in sandbox run |
| Writing reference data | Landlock read-only; prover returns `unsupported` → fail-closed reject | kernel test PASS |

## Quickstart

```bash
git clone <this repo> && cd rl-triage-agent
uv sync
uv run pytest -q tests                                   # unit + app-level security tests
uv run python bench/build_workspace.py 42                # benchmark telemetry -> agent workspace
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
6 passed                                                  # tests/sandbox
```

**This verifies:** permission proofs, kernel enforcement inside a live OpenShell sandbox, agent accuracy on recorded Isaac Lab telemetry, and the approve → retrain → verdict loop on a real Isaac Lab run.

**This does not verify:** faults that only appear in long training (>100 iterations), rough terrain, manipulation tasks, real robots, or multi-sandbox fleets.

## Limitations

- Benchmark circularity: the fault catalog and the agent were built by the same team. Mitigations: benign changes mixed in with a secret seed, prompts committed before each evaluation (`4acb1ed`, `517f613`, `c7e8cd9`), same-model single-prompt baseline, held-out seed.
- Task A saturated: with extreme fault values both the agent and the single-prompt baseline score 20/20. We tried plausible-magnitude changes (v2, `bench/v2/labels.json`); at 100 iterations most did not break training, and a reward-ratio verdict mislabels reward-weight changes, so v2 was not used for scoring.
- The skill's mechanism table (`skills/isaaclab-rl-triage/SKILL.md`) was written after looking at benchmark telemetry; scored runs do not load the skill.
- Network rules apply to child processes, so code run by `run_analysis` can still reach the allowed NVIDIA inference path.
- The NVIDIA free endpoint was often overloaded; the harness retries only infrastructure errors and logs every attempt.

## 한국어 요약

**Isaac Lab 학습 실패 원인 추적 에이전트.** 로봇 RL 엔지니어가 설정 여러 개를 바꾸고 학습했는데 학습이 무너졌을 때, 텔레메트리를 정상 기준 실행과 비교해 원인을 찾고, 설정을 직접 고치는 대신 **다음 실험 하나를 사전등록**합니다. 사람이 승인하면 재평가 브리지가 Isaac Lab에서 그 변수 하나만 바꿔 다시 학습해 진단을 확인합니다. 에이전트는 OpenShell 샌드박스 안에서 설정·안전 게이트·보상을 건드릴 수 없고, 더 많은 권한 요청은 openshell-prover가 현장 경계와 비교해 증명되지 않으면 자동 거절합니다.
