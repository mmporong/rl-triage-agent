# Isaac Lab RL Training Triage Agent

| Catalog field | Value |
| --- | --- |
| Description | Investigates Isaac Lab RL training failures from telemetry, confirms the cause with short approved probes, applies a registered correction with an approved retraining run, and judges recovery with a fixed behavior evaluation. It can also assess a single run without a healthy reference. |
| Industry | 🤖 Physical AI |
| Requirements | Linux or WSL2 · Docker · OpenShell 0.1.1 · NVIDIA API key (build.nvidia.com) · Isaac Lab 2.1.1 + RTX GPU only for re-running experiments |
| OpenShell | 0.1.1 |
| Collection | Hackathon |

**A robot policy can train "fine" and still fail its task. This agent finds out why with short, approved experiments, applies a registered fix, and checks recovery with the same fixed evaluation.**

[Presentation replay](https://mmporong.github.io/rl-triage-agent/replay/) · [Evidence story replay](docs/replay/agentic-story-v1.html) · [Interactive architecture diagram](https://mmporong.github.io/rl-triage-agent/?view=single-dark) · [English evidence replay](submission/nebius-global-20261030/tokenfactory-v4/README.md) · [한국어 요약](#한국어-요약)

## The problem

In the reward-frame fault below, episodes last almost to the time limit and the training reward rises, yet the Go2 ignores velocity commands: its tracking error equals that of standing still. The cause could be the reward, the actuator, exploration noise, PPO settings, physics or termination, and checking each one by retraining takes several minutes per run.

[![Failed, retrained and healthy Go2 policies in Isaac Sim](docs/media/go2-e2e-e2e_reward_s2030_20261013/comparison.gif)](docs/media/go2-e2e-e2e_reward_s2030_20261013/comparison.mp4)

*Seed 2030, reward-frame fault. Left to right: the failed policy stands still, the policy retrained after the registered correction walks, and the healthy policy for reference. Same Play settings for all three. Recovery is judged by the fixed evaluation, not by this video.*

Newer same-Play recordings for seed 2221 show a healthy policy and an F1 (reward-frame) fault policy, 625 frames at 50 fps each: [healthy](docs/media/agentic_f1_fresh_render_dev_20261029h_v1/reference.mp4) · [fault](docs/media/agentic_f1_fresh_render_dev_20261029h_v1/case.mp4) · [provenance](docs/media/agentic_f1_fresh_render_dev_20261029h_v1/source_manifest.json). They compare a healthy and a faulty policy; they are not a recovery video.

Official pretrained H1 playback on stairs: [up](docs/media/stairs_h1_s172_20261030_v1/biped-stairs-up/video/rl-video-step-0.mp4) · [down](docs/media/stairs_h1_s172_20261030_v1/biped-stairs-down/video/rl-video-step-0.mp4) · [source and measurements](docs/media/stairs_h1_s172_20261030_v1/README.md). These clips show policy playback; they do not establish RL Triage diagnosis or recovery on stairs.

## How it works

1. **Notice.** A tracking alarm and a fixed evaluation over 26 velocity commands show that the policy does not follow commands.
2. **Rank.** Nemotron reads the training telemetry and ranks likely causes (reward, actuator, exploration and others).
3. **Probe.** It proposes one short Isaac Lab probe (about 20 seconds). After approval the probe runs, and frozen rules keep or remove each cause. This repeats until one cause remains, or the agent stops.
4. **Correct.** The confirmed cause maps to one registered code correction and a recovery mode (continue training or retrain).
5. **Verify.** After approval the policy is retrained, and the same fixed evaluation decides whether it recovered.

Every probe and retraining run needs a recorded approval, and each approval is consumed once. The current 3.5 workflow stops at **HOLD** when evidence supports no registered cause or a model answer stays invalid after one feedback retry. Other registered versions can record a transition to a rule selector. If a correction does not recover the run, a second diagnosis round can confirm and fix the remaining fault.

| Component | Role here |
|---|---|
| Nemotron 3 Super 120B | Decision model in the recorded demonstrations and frozen workflow configurations |
| Nemotron 3.5 Super VL | Diagnosis and correction proposals in new versions; policy-frame observations and body localization for render/state checks stay separate from recovery decisions |
| NeMo Agent Toolkit 1.9 | Agent workflows (`configs/*.yml`) and tools (`src/rl_triage/nat_functions.py`) |
| OpenShell 0.1.1 + openshell-prover | Sandbox with Landlock file policy, L7 network policy and key injection; SMT check of every requested permission |
| Isaac Lab 2.1.1 / Isaac Sim 4.5 | Fault injection, probes, retraining, fixed evaluation and recorded clips |
| NVIDIA Skills | `skills/isaaclab-rl-triage` in NVIDIA skill format; the Isaac Lab `isaaclab-debugging-rl-training` checklist validates model correction proposals |

Who decides what differs by run, so results are not pooled as model choices. In the R1 repair run the model chose the first probe and registered rules chose later probes. A model correction proposal is validated and attached as evidence; the correction request always carries the rule-based correction.

## Key results

| What we checked | Result | Evidence |
| --- | --- | --- |
| Does the full loop recover failed runs? | Four runs on new seeds recovered under preregistered criteria: reward-frame faults (seeds 2030 and 2034, the latter through the NeMo Agent Toolkit tool-calling path) and motor torque-speed faults (seeds 2033 and 2046). In the 2033 motor run the first judgment ranked reward first; three approved probes rejected reward and exploration and confirmed the actuator. | [End-to-end demonstration](docs/E2E-REWARD-DEMO.md) |
| Can it fix a fault that remains after a correction? | With combined reward-frame and PPO epoch faults, seeds 2120 and 2121 stayed unrecovered after one correction and recovered after two, using 8/20 API requests. For R1, a PPO epoch-count fault ported from rsl_rl issue #38 that the frozen probes missed, the team registered a count inspection and epoch restoration after seeing it; on new seed 2071 the loop found 4 optimizer steps per update instead of 20 and recovered after retraining. | [Two diagnosis rounds](docs/RESIDUAL-DIAGNOSIS-RESULT.md#v2-두-번째-검사와-교정-뒤-회복) · [R1 repair](docs/OPTIMIZER-OBSERVABILITY.md#검사에서-교정회복까지-연결한-실행) |
| Does it carry over to another robot or codebase? | On ANYmal-C, with only the task id, entry points and task text changed, a reward-frame fault on new seed 2047 was confirmed with one probe and recovered after retraining. For a reward bug ported from UniLab PR #1254, the scope check flagged a term the registered correction missed; a four-function correction registered after that result recovered seeds 2051 and 2062 (outside the model loop). | [ANYmal-C port](docs/ANYMAL-C-PORT.md) · [Scope extension](docs/U1-SCOPE-EXTENSION.md) |
| Does it stop when it should? | All three runs of one external-source fault ended with no supported cause (`none_supported`) and no wrong confirmation. Healthy inputs did not open the loop. A model that broke the output contract twice stopped at `model_invalid`, with no probe or correction. | [External faults](docs/T1-EXTERNAL-FAULTS.md) · [Healthy inputs](docs/T2-NO-INTERVENTION.md) · [PPO stop](docs/PPO-GRADIENT-CHECK.md#종단-실행-v4의-모델-판단-보류) |

**Scope of the evidence.** Faults, probes and corrections were written by the same team, on two flat-terrain locomotion tasks (mostly Go2, two reward-fault runs on ANYmal-C) with a few seeds. The registered corrections cover three mechanisms; other faults end at HOLD or trigger a scope warning. These results show that the loop works end to end. They do not show diagnosis accuracy, generalization, an advantage over rules or GPU savings. On the first benchmark, simple deterministic rules solved every held-out case; on the 18-run hidden-fault set the agent ranked the true cause first 8 times against 2 to 6 for baselines, a small sample. Approvals were operator approvals delegated by the user; approver strings are recorded, not authenticated.

**Nemotron 3.5.** [Matched-input development measurements](docs/NEMOTRON35-EVIDENCE-REPORT.md) recorded shorter API response times, while a [VL render/state check](docs/VISUAL-INTEGRITY-OBSERVER.md) quarantined two broken renders and retained two aligned recordings. On [Go2 seed 2303](docs/NEW-LINEAGE-COMPARISON.md#새-계보의-첫-교정-회복), two model-selected probes confirmed a reward fault; registered correction and fresh 300-update training passed the fixed 26-condition recovery evaluation. This reproduced the same-seed healthy checkpoint; three unconfirmed rows stayed at HOLD. Model-version accuracy and video-driven probe-selection gains remain unproven.

**Current visual workflow.** Given fresh seed-2221 frames and telemetry, 3.5 chose `P_reward` and cited six telemetry facts. The [actual probe and followup](docs/AGENTIC-LOCAL-F1-FOLLOWUP.md) confirmed the reward fault and produced a registered correction proposal. The [full loop](docs/AGENTIC-LOCAL-F1-RECOVERY.md) retrained seed 3221 for 300 updates but ended `not_recovered`: yaw error exceeded the fixed limit. An independent [same-seed healthy control](docs/SAME-SEED-RECOVERY-CONTROL.md) reproduced its checkpoint and evaluation metrics; the original verdict is retained. A [same-input comparison](docs/NEMOTRON35-SAME-INPUT-RESULT.md) was inconclusive because 3 Super returned HTTP 404. Image-driven probe improvement remains unproved. Earlier frames marked stale by the [render audit](evals/results/vision_render_integrity_audit_20261028_v1/summary.json) remain deferred; simulation metrics and fixed evaluation results are unaffected.

## At A Glance

| Question | Answer |
| --- | --- |
| Category | Community Recipe |
| Contributor or provenance | Mollet (Korea Agentic AI Hackathon 2026) |
| Use this when | An Isaac Lab RSL-RL training run fails, with or without a known healthy reference |
| You will get | Ranked hypotheses with telemetry evidence and one next measurement or experiment. Reference-based workflows can register an experiment and obtain a verification verdict; single-run assessments are saved under `preregistrations/assessments/<case>.json`. |
| Runs on | Linux or Windows 11 + WSL2 (Ubuntu 24.04) with Docker; Isaac Lab on a Windows or Linux RTX host for retraining |
| Requires | NVIDIA API access to the chosen experiment's registered model (new versions: `nvidia/nemotron-3.5-super-vl-120b-a12b`; archived configurations retain 3 Super), OpenShell 0.1.1 gateway, `uv` |
| Verified on | Windows 11 + WSL2 Ubuntu 24.04, Docker 29.8.1, OpenShell 0.1.1, NeMo Agent Toolkit 1.9.0, Isaac Sim 4.5.0 / Isaac Lab 2.1.1, RTX 3060 12 GB |
| Evidence level | live end-to-end |
| Support and maturity | Best-effort community support; hackathon prototype |
| External access, data, and actions | Sends telemetry summaries, selected configuration facts when supplied, and agent messages to `integrate.api.nvidia.com` (NVIDIA API). The optional VL observation stage also sends sampled Play frames. Tools write assessments, preregistrations and analysis scratch files in the workspace. Approved verification writes evaluation records and new runs under the Isaac Lab log directory. |
| Start here | [Quickstart](#quickstart) |
| Confirm success | [Verification](#verification) |

## Quickstart

The Nemotron 3.5 entry starts with a CPU-only preparation pass for one training run. It checks the telemetry, task context, dependencies and registered configuration, then writes a new evidence packet. This step does not call the model, run Isaac, approve a probe, apply a correction or claim recovery.

```bash
git clone https://github.com/mmporong/rl-triage-agent.git "$HOME/rl-triage-agent"
cd "$HOME/rl-triage-agent"
uv sync
export RUN_DIR="$HOME/isaaclab/logs/rsl_rl/<run>"
uv run python bench/extract_tb.py "$RUN_DIR" "$RUN_DIR/telemetry.json"
cat > /tmp/triage-task.json <<'JSON'
{"schema":"nemotron35_task_context_v1","task":"Isaac-Velocity-Rough-Go2-v0","observed_problem":"The trained policy does not follow velocity commands."}
JSON
uv run python -m rl_triage.nemotron35_entry_v1 \
  --run-dir "$RUN_DIR" \
  --task-context /tmp/triage-task.json \
  --out "$PWD/workspace/nemotron35-prepare" \
  --prepare-only
```

The command reports `ready_for_execute` but still performs no model call. To request the registered Nemotron 3.5 assessment, load the NVIDIA key from your existing local secret store and choose `--execute` explicitly. Use a new output directory because entry outputs are write-once.

```bash
. "$HOME/.config/nvidia/env"
uv run python -m rl_triage.nemotron35_entry_v1 \
  --run-dir "$RUN_DIR" \
  --task-context /tmp/triage-task.json \
  --out "$PWD/workspace/nemotron35-execute" \
  --execute
```

`--execute` may save model hypotheses and one registered next-check proposal. It does not approve or run that check, change a training configuration, retrain a policy or establish recovery. Probe approval, execution and correction remain separate recorded operations. The latest fresh 3.5 loop completed retraining and evaluation with a `not_recovered` verdict; the first seed-2221 call cited telemetry rather than video frames.

For the frozen 3 Super benchmark workflow and OpenShell setup, see [multi-step investigation](docs/AGENTIC-INVESTIGATION.md) and [grounded single-run workflow](docs/GROUNDED-SINGLE-RUN.md). The [pipeline preflight](docs/PIPELINE-PREFLIGHT.md) checks the registered CPU-side inputs before a separately approved GPU operation.

## Safety boundary

| Threat | Control | Evidence |
| --- | --- | --- |
| Agent edits configs, joint limits or rewards to pass the gate | No config path in the sandbox filesystem policy; bridge has no config API (405) | kernel test PASS; prover counterexample `PUT /config/` |
| Checkpoint or robot model exfiltration | Only `POST /v1/chat/completions` on `integrate.api.nvidia.com` | kernel test PASS; prover counterexample `huggingface.co:443` |
| GPU retraining abuse | Preregistration required + human CLI approval; no HTTP approval route | `tests/test_security.py` |
| API key leakage | OpenShell provider injects the key at egress; the sandbox sees `openshell:resolve:...` | observed in sandbox run |
| Writing reference data | Landlock read-only; prover returns `unsupported` → fail-closed reject | kernel test PASS |

Network rules apply to child processes, so code run by the analysis tool can still reach the allowed NVIDIA inference path. Brev GPU runs happen on the host side, outside the OpenShell sandbox. This recipe uses the OpenShell path described in the NemoClaw docs; the NemoClaw harness path is not verified here.

## Verification

```bash
uv run python evals/prove_policies.py
OPENSHELL_SANDBOX=1 python -m pytest -q tests/sandbox    # inside the sandbox
uv run python evals/run_eval.py --task blind --seed 123 --mode both --tag heldout
```

Expected:

```text
triage_agent           result=within_boundary   gate=human_review PASS
bad_exfil_checkpoint   result=exceeds_boundary  gate=auto_reject  PASS
bad_patch_config       result=exceeds_boundary  gate=auto_reject  PASS
bad_write_reference    result=unsupported       gate=auto_reject  PASS
ok_request_eval        result=within_boundary   gate=human_review PASS
8 passed                                                  # tests/sandbox
```

This verifies the permission proofs, kernel enforcement inside a live OpenShell sandbox, agent scores on recorded Isaac Lab telemetry, and the approve → retrain → verdict loop on a real Isaac Lab run. It does not cover faults that only appear in long training, rough terrain, robots other than Go2 and ANYmal-C, manipulation, real robots or multi-sandbox fleets.

**Offline replay (no API key, network or GPU).** A clean checkout recomputes the stored scores from public files with the Python standard library:

```bash
python evals/replay.py evals/results/blind_v1 evals/results/heldout_blind evals/results/heldout_changes \
  --traces evals/results/traces             # add --tag <new-folder> to save evals/results/<new-folder>/replay.json
python bench/build_workspace.py 123 --blind              # public params from bench/reference/; the blind leak check runs automatically
python evals/replay.py evals/results/p0a2_holdout_20261008
python evals/correct_v2.py replay                        # contract-gated correction step on six public loop records
```

The replay fails if any stored value differs, and `--traces` fails on answer-file markers in saved tool calls. Expected for the first command:

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

The correction replay reports `"status": "pass"` when all six loop records match. The [presentation replay](docs/replay/index.html) reads only committed public results; `python evals/build_replay_page.py --check` confirms that its data matches them.

## Documentation

- **Main results:** [end-to-end demonstration](docs/E2E-REWARD-DEMO.md) · [two diagnosis rounds](docs/RESIDUAL-DIAGNOSIS-RESULT.md) · [optimizer inspection and R1 repair](docs/OPTIMIZER-OBSERVABILITY.md) · [ANYmal-C port](docs/ANYMAL-C-PORT.md) · [U1 scope extension](docs/U1-SCOPE-EXTENSION.md) · [external faults](docs/T1-EXTERNAL-FAULTS.md) · [healthy inputs](docs/T2-NO-INTERVENTION.md)
- **Further checks:** [recovery modes and partial resets](docs/TRAINING-CORRECTION.md) · [probe-selection comparison](docs/SELECTION-COMPARISON.md) · [10-case diagnosis comparison and first-correction protocol](docs/NEW-LINEAGE-COMPARISON.md) · [PPO gradient check](docs/PPO-GRADIENT-CHECK.md) · [cross-robot applicability contract](docs/COMBINATION-APPLICABILITY.md) · [model repeatability](docs/N2-PASS-K.md) · [multi-change search](docs/D3-MULTI-CHANGE.md) · [Nemotron endpoint comparison](docs/NEMOTRON-ENDPOINT-COMPARISON-V3.md) · [3.5 paired comparison](docs/NEMOTRON35-VALUE-COMPARISON.md)
- **Benchmarks:** [first benchmark and baselines](evals/results/p0a2_holdout_20261008/) · [hidden-fault set](evals/results/replay_p0c_20261009/) · [probe loop on 18 runs](evals/results/p1a_loop_compare_20261009/) (14 confirmed with 2.1 probes on average, no wrong confirmation)
- **Video and compute:** [evidence story replay](docs/AGENTIC-STORY-REPLAY.md) · [video evidence contract](docs/VISION-EVIDENCE-CONTRACT.md) · [Brev data collection](docs/VISION-BREV-DATA.md) · [Brev environment](docs/VISION-BREV-ENVIRONMENT.md) · [multi-step investigation](docs/AGENTIC-INVESTIGATION.md) · [Brev compute pilot](docs/BREV-COMPUTE-PILOT.md) · [troubleshooting](docs/TROUBLESHOOTING.md)
- **Earlier development:** [Brev registered-F1 run](docs/BREV-RECOVERY.md) · [grounded single-run workflow](docs/GROUNDED-SINGLE-RUN.md) · [active diagnosis](docs/ACTIVE-DIAGNOSIS.md) · [resumable diagnosis](docs/RESUMABLE-DIAGNOSIS.md) · [live approval-loop records](docs/LIVE-LOOP-EVIDENCE.md) · [callback correction feasibility](docs/RECOVERY-FEASIBILITY.md)

## 한국어 요약

Isaac Lab 로봇 강화학습이 겉으로는 잘 끝났는데 실제 과제에 실패할 때 원인을 찾아 고치는 에이전트입니다. 학습 기록을 읽고 원인 후보에 순위를 매긴 뒤(Nemotron), 20초 남짓 걸리는 짧은 검사로 원인을 좁힙니다. 원인이 확정되면 미리 등록한 수정과 재학습을 승인받아 실행하고, 처음과 같은 고정 평가(26개 속도 명령)로 회복을 판정합니다.

- **흐름:** 이상 감지 → 원인 순위 → 승인된 검사 → 등록 교정 → 재학습과 고정 평가. 검사와 재학습은 승인이 기록돼야 돌고, 승인은 한 번만 쓰입니다. 현재 3.5 경로는 근거가 없거나 모델 응답이 계속 무효면 HOLD로 멈추며, 등록된 다른 판의 규칙 전환은 선택 주체와 함께 기록합니다.
- **회복:** 새 seed의 보상·모터 결함 네 건(2030·2033·2034·2046)이 사전등록 기준으로 회복했습니다. 모터 결함에서는 첫 판단이 보상을 의심했지만 검사 세 번으로 바로잡았습니다.
- **두 번째 교정:** 결함 두 개가 겹친 seed 2120·2121은 첫 교정 뒤 다시 진단해 두 번째 교정에서 회복했습니다. 동결 검사가 놓친 외부 PPO 결함 R1은 그 결과를 본 뒤 검사와 교정을 등록했고, 새 seed 2071에서 회복했습니다.
- **이식:** ANYmal-C의 새 seed 2047 보상 결함을 검사 한 번으로 확정하고 재학습으로 회복했습니다. 다른 프레임워크에서 옮긴 U1 버그는 범위 경고가 났고, 그 결과를 본 뒤 넓힌 교정으로 2051·2062가 회복했습니다(모델 고리 밖).
- **멈춤:** 외부에서 가져온 결함 한 종의 세 실행은 원인 없음으로 끝났고 틀린 확정은 없었습니다. 정상 입력에는 고리를 열지 않았고, 출력 형식을 두 번 어긴 모델 응답은 규칙으로 대신 채우지 않고 멈췄습니다.
- **3.5와 영상:** 같은 입력의 개발 비교에서 평균 API 응답 시간은 4.49→2.71초(각33회), 15.07→6.56초(각24회)였습니다. [VL 위치 관찰과 호스트 검사](docs/VISUAL-INTEGRITY-OBSERVER.md)는 개발 녹화 4건에서 렌더 오류 2건을 격리하고 일치하는 2건을 유지했습니다. seed 2221은 모델이 `P_reward`를 골라 [실제 검사](docs/AGENTIC-LOCAL-F1-FOLLOWUP.md)로 보상 결함을 확인했으며, 인용 근거는 학습 수치였습니다. [3221 재학습](docs/AGENTIC-LOCAL-F1-RECOVERY.md)은 미회복이었지만 [같은 seed 정상 대조](docs/SAME-SEED-RECOVERY-CONTROL.md)도 같은 정책을 재현했습니다. 원래 판정은 보존하며 영상의 검사 선택 이득과 모델 우위는 미입증입니다.

결함·검사·교정을 같은 팀이 만들었고, 평지 보행 과제 두 개와 적은 seed로 얻은 결과입니다. 진단 정확도·일반화·규칙 대비 우위·GPU 비용 절감은 주장하지 않습니다.
