# RL Triage Agent — judge replay bundle

This package provides an English, CPU-only replay of recorded development evidence. It performs no training, provider lifecycle operation, approval, or model request. The full project is at https://github.com/mmporong/rl-triage-agent (Apache-2.0).

## Run the test build

Use Python 3.10 or later. This replay uses only the standard library. From a downloaded bundle:

```powershell
cd "$HOME/Downloads/Nebius_RL_Triage_Submission_20261030/public"
python ./verify.py
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/index.html` in a browser. The six stage buttons inspect the recorded R1 investigation and a separate compound-defect result. Add `?demo=1` for the 108-second automatic replay. JSON links expose the byte-verified inputs. Stop the local server with Ctrl+C when finished. The page uses loopback resources only and requires no login.

From the source repository instead:

```powershell
cd "$HOME/rl-triage-agent/submission/nebius-global-20261030"
python ./verify.py
python -m http.server 8765 --bind 127.0.0.1
```

Expected verifier result: `status: PASS`, 11 source files verified, and ten recorded-consistency checks true. This checks public evidence bytes and joins; it does not reauthenticate a live lease, approval ledger, or remote GPU process. The browser verifies the nine JSON files before showing the replay. The Python verifier also checks both original video files. A hash mismatch fails closed.

## What to inspect

1. **Failure:** normal and R1 survive 20 seconds with zero falls, but R1 has high tracking error.
2. **Inspect:** the recorded Nemotron 3 Super response selects `P_reward`; five inspections return normal. The subsequent sequence is registered, not five further model choices.
3. **Confirm:** a fresh three-update instrumentation run records 4 optimizer steps per update for R1 versus 20 for the normal control.
4. **Correct:** the recorded approval event chain precedes the registered epoch restoration and restart seed3071.
5. **Evaluate:** all four fixed behavioral checks pass for the corrected checkpoint. Its metrics and SHA match the summary.
6. **Residual:** separate v2 compound runs2120/2121 remain unrecovered after one correction and pass after both.

The two `media/*.mp4` files are synchronized normal/F1 seed2221 simulation captures. They are not corrected-policy footage or an additional recovery experiment. See `source-map.json` for original repository paths, byte hashes, and scope.

## Full training application

The repository root README documents the training/diagnosis setup and the public `docs/` files describe each registered experiment. Its Python package requires Python3.11–3.13 and pinned NVIDIA NeMo Agent Toolkit1.9; Isaac Lab simulation uses its separate compatible simulator environment. Do not install the package into an arbitrary Isaac Python environment. A model-enabled diagnostic requires a provider credential; full GPU reproduction additionally requires Isaac Lab, compatible NVIDIA hardware, registered checkpoint/input artifacts, and fresh approvals/resources. Those are not created by this test build. The recorded evidence and video are available for judging without them.

## Files and license

`DEVPOST_EN.md` contains the project fields; `PLATFORM_AND_FEEDBACK_EN.md` separates compute, inference, and credits; `CHANGES_DURING_PERIOD_EN.md` describes the dated implementation evidence. `VIDEO_EN.md` and `demo.en.srt` accompany `RL_Triage_Agent_Nebius_Demo.mp4`. `source-map.json` maps every copied source. `SHA256SUMS.txt` covers the final bundle except itself. Project code is Apache-2.0; the root license is copied as `LICENSE`. See `THIRD_PARTY_NOTICES.md` for simulator/model boundaries.
