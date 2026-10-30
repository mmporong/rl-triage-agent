# RL Triage Agent: Token Factory investigation replay

Inspect an actual Nebius Token Factory decision from training evidence to an approval-required inspection proposal. This bundle contains eight recorded Nemotron 3.5 replies and a functioning English investigation screen. It makes no new model or GPU requests and needs no credential.

## Open the judge replay

Python 3.10 or later is sufficient for the recorded replay. Extract the bundle, then run:

```powershell
cd "$HOME/Downloads/RL_Triage_Nebius_Nemotron_Evidence_v4"
python -B verify.py
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/index.html`. Select a recorded candidate or identical-reference control. The four stages show supplied training facts, the actual model decision and citations, host checks, and an inspection review JSON. HOLD records cannot export an inspection. Stop the server with Ctrl+C when finished.

The browser checks the evidence bytes and their relationships before enabling controls. It rejects changed citations, mismatched provenance, repeated episode IDs, and approval claims. These hashes check published bytes; they are not a provider signature or a fresh approval.

## Inspect the Nemotron evidence panel

Select **Nemotron evidence** to open three read-only topics: historical NIM API response times, VL observations, and the actual seed-2221 inspection/correction workflow. The header switches provider labels while this panel is open. **Back to TF investigation** restores the selected Token Factory record and its approval boundary.

The 40%/56% figures describe historical NVIDIA-hosted NIM request waits, not Token Factory or total workflow speed. VL correspondence is reported against fixed-evaluation groups, not human-labeled accuracy. The newer NIM3.5 retraining result is not_recovered. `NEMOTRON_EVIDENCE_EN.md` provides the source, cohort and unit details. Accepted inputs are pinned by `NIM_SOURCE_ANCHOR.json`; changed scope or data fails closed.

## Re-run the original CPU investigation functions

The public project is https://github.com/mmporong/rl-triage-agent (Apache-2.0). A checkout matching the 13 LF source hashes in `CORE_SOURCES_LF.json` can reproduce the CPU preparation:

```powershell
cd "$HOME/Downloads/RL_Triage_Nebius_Nemotron_Evidence_v4"
python -B prepare_demo.py --repo-root "$HOME/rl-triage-agent" --output "$HOME/rl-triage-agent/.triage-runtime/tf-judge-replay"
```

This runs the original `Runtime._request`, message validation, `Investigation.propose/probe_choice`, and `C.request_spec` functions. It reconstructs model-facing messages and parameters and requires exact equality with the recorded HTTP requests. Local context paths can differ; the published view contains logical IDs and hashes. See `CORE_REPLAY_README.md` for the original replay scope.

## What the evidence shows

The separately preregistered Token Factory comparison used known training records with two repeats per input and context policy. Four identical-reference controls returned HOLD; four candidate sessions proposed inspections. The optional additional training read was not selected. This measures a working decision-and-validation path, not retrieval gain, model superiority, unseen-fault generalization, or a new GPU repair.

`RL_Triage_Agent_Nebius_TokenFactory_v3.mp4` is the 168-second English demo. The first 143 seconds reuse the accepted simulation and completed R1 replay sequence. The final 25 seconds show the v3 Token Factory application in operation. The video is reused byte for byte; the new NIM evidence panel and its additional figures are provided in the app and documents, not added to this video. The completed R1 repair used NVIDIA-hosted Nemotron 3 Super; its recovery metrics are separate from the Token Factory experiment. New Token Factory inspection proposals still need a fresh execution registration, approval and resource lease.

## Submission fields and feedback

`DEVPOST_EN.md` contains the updated project description. `PLATFORM_FEEDBACK_EN.txt` records feedback from actual Token Factory use and distinguishes it from Brev-provisioned Nebius L40S compute. Credit redemption, account balance and invoiced charges remain unverified. `VIDEO_EN.md` describes the new video; `demo-with-tokenfactory.en.srt` provides its English captions.

This bundle preserves the previously frozen submission files and original 170-second video. It is a new delivery candidate for the owner to publish and connect to the existing Devpost submission 1264681. It does not report an upload or final submission.

## License

Project code: Apache-2.0, included in `LICENSE`. Model and simulator licenses remain separate; see `THIRD_PARTY_NOTICES.md`. No model weights, credentials or checkpoint binaries are included.
