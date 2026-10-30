# Token Factory recorded evidence and core replay

This local candidate contains recorded Nebius Token Factory requests and replies, an offline evidence verifier, and a CLI that runs the original RL Triage investigation modules. It requires no API key and makes no network or GPU requests.

With Python 3.10 or later, extract this bundle into `$HOME/Downloads/TokenFactory_Record_Addendum_v7`, then run:

```powershell
cd "$HOME/Downloads/TokenFactory_Record_Addendum_v7"
python -B verify.py
python -B replay_core.py --repo-root "$HOME/rl-triage-agent" --output "$HOME/rl-triage-agent/.triage-runtime/tf-recorded-replay"
```

The core replay requires a project checkout matching the 13 source hashes in `CORE_SOURCES_LF.json`. Project source and installation instructions: https://github.com/mmporong/rl-triage-agent . The standalone `verify.py` uses only the Python standard library.

The CLI reconstructs local investigation contexts from the published training summary. It checks that the resulting model-facing messages and params exactly match each recorded HTTP request, then passes the recorded response through the original `Runtime._request`, message validation, `Investigation.propose/probe_choice`, and `C.request_spec` functions. The eight records produce four HOLD decisions and four inspection requests awaiting authorization. Replaying the same output directory preserves the same records.

These contexts are new local reconstructions. They do not reauthenticate the original private capsule, approval ledger, provider, or resource. The CLI cannot approve or dispatch the exported inspection requests. This is a demonstration written after the recorded API experiment, not a new model evaluation.

The actual core experiment used NVIDIA Nemotron 3.5 Super VL through Token Factory. Four sessions received all eight training facts; four received two reward facts and could request a registered read of six more facts. None requested that read. Each policy produced two HOLD decisions for the identical-reference control and two inspection proposals for the recorded candidate. The inputs are known development records with two repeats per policy/input. They do not establish generalization, retrieval gain, or an optimal inspection policy.

The bundle also includes one earlier Nemotron 3 Super compatibility sample. The completed R1 repair in the original judge replay remains a separate NVIDIA-hosted Nemotron run. These Token Factory proposals have not produced a new GPU repair or learning-effect result. Live caller integration into the public application is still pending. The original submission files and 170-second video are unchanged.

Run the targeted replay tests against the same checkout:

```powershell
python -B test_replay_core.py --repo-root "$HOME/rl-triage-agent"
```

`PUBLIC_INDEX.json` binds the files. `TOKEN_FACTORY_UPDATE_EN.md` is a proposed factual update for the owner to integrate. Project license: Apache-2.0.
