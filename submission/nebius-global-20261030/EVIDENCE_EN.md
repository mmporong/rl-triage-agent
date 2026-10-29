# Claim-to-evidence map

All copied records are byte-identical to the repository files named in `source-map.json`. The replay is a new English presentation of recorded results. It does not rerun diagnosis, create approvals, or reestablish live process identity.

| Claim | Local evidence | Interpretation |
| --- | --- | --- |
| R1 failure/recovery: linear RMSE 1.169→0.148m/s; normal0.146 | [summary](data/r1_summary.json), [fixed evaluation](data/r1_eval.json) | One known-defect development run; source2071/restart3071; all four behavioral checks pass. |
| Model first chose P_reward; follow-up sequence was registered | [model response](data/r1_model.json), [probe events](data/r1_probes.json) | Nemotron3 Super, not a claim that the newest3.5 caller completed this run. |
| Optimizer steps4 vs20 across three updates | [probe measurement](data/r1_probes.json), [training evidence](data/r1_training.json) | Fresh instrumentation under each arm configuration; not checkpoint-loaded inspection. |
| Epoch restoration and recorded single-consumption approval chain | [probe/correction events](data/r1_probes.json) | Historical proposed→approved→consumed→receipt. Public replay verification does not authorize execution. |
| Compound v2 cases2120/2121 recover after two corrections | [residual summary](data/residual.json) | Post-result development; opposite orders; final checkpoints/metrics equal same-seed normal controls. |
| U1 expanded four-term correction2062/3062 | [U1 summary](data/u1.json) | Post-result scope confirmation; model loop disabled; not an autonomous-discovery result. |
| Newly synchronized normal/F1 footage | [capture summary](data/fresh_render.json), [video provenance](data/video_provenance.json), [normal clip](media/reference_video.mp4), [F1 clip](media/case_video.mp4) | Seed2221, 625frames×50fps,12.5seconds each; new training0, new recovery0. |
| CPU evidence replay functions without model calls | `verify.py`, `app.js`, [validation](VALIDATION.json) | New local replay implementation; original result data displayed and hashed at load. |

Broader project documentation:

- PPO calculation invariant: https://github.com/mmporong/rl-triage-agent/blob/main/docs/PPO-GRADIENT-CHECK.md
- Correction-scope gates: https://github.com/mmporong/rl-triage-agent/blob/main/docs/COMBINATION-APPLICABILITY.md
- Optimizer observation: https://github.com/mmporong/rl-triage-agent/blob/main/docs/OPTIMIZER-OBSERVABILITY.md
- Brev/Nebius capture registration: https://github.com/mmporong/rl-triage-agent/blob/d9f94442adb872e83174672cf84aeb6236906263/bench/protocols/agentic_f1_render_capture_phase_20261029h_v10.json

The full repository and the smaller portable replay serve different reproduction scopes. Simulator/checkpoint reproduction requires its registered inputs and resources; the portable package is sufficient for replaying and checking these public evidence joins.
