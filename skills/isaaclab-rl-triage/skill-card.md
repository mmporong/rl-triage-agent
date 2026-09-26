## Description: <br>
Triages a failed or anomalous Isaac Lab RSL-RL training run against a healthy reference run, finds the config change that explains the deviation, and registers one next experiment for human approval. <br>

## Owner
mmporong (community skill, not NVIDIA-verified; no `skill.oms.sig`) <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Robotics RL engineers training locomotion or manipulation policies in NVIDIA Isaac Lab who changed several settings at once and need to know which change broke training before spending another run. <br>

## Known Risks and Mitigations: <br>
Risk: The agent may try to make the gate pass by editing thresholds, joint limits or rewards (specification gaming). <br>
Mitigation: The skill never edits configs; the OpenShell sandbox policy has no write access to config paths, and permission requests are checked by openshell-prover against a site boundary. <br>
Risk: Diagnosis may be wrong. <br>
Mitigation: Output is a preregistered single-variable experiment with an acceptance gate, not an applied fix. <br>
Risk: Benchmark circularity — the fault catalog was designed by the same authors who built the agent. <br>
Mitigation: Cases are mixed with benign changes and shuffled with a secret seed; the agent prompt and tools were committed before any evaluation run; a single-prompt baseline uses the same model and inputs. <br>

## Skill Output: <br>
**Output Type(s):** [Files] <br>
**Output Format:** [JSON preregistration] <br>
