# Configuration evidence for a single run

The single-run workflow can read selected configuration facts alongside telemetry. A reward weight or sensor definition establishes what was configured. Task success, measured contact and the cause of a failure still require observations.

Use a separate workspace with an absent reference:

```text
workspace/single_run/
  reference/telemetry.json
  cases/<case_id>/telemetry.json
  cases/<case_id>/configuration.json  # required when context.json is present
  cases/<case_id>/context.json        # optional source binding and selected facts
```

The reference is exactly `{"reference_status":"absent","summary":{},"series":{}}`. Without `context.json`, the context tool returns `status: not_provided`. Missing context leaves configuration unknown; it does not establish that a reward or sensor is missing.

## Select facts from a snapshot

Keep the JSON configuration snapshot in the same case directory. Include configuration known to belong to this run, with diagnostic labels and later attribution records excluded from agent inputs. The context tool returns only the selected values; it does not return the whole configuration file.

For example, this **synthetic** configuration describes a reward and a contact observation:

```json
{
  "rewards": {"task_success": {"weight": 10.0}},
  "observations": {
    "policy": {
      "foot_contacts": {"body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]}
    }
  }
}
```

Create a manifest after supplying the telemetry and configuration snapshot. The following example creates `context.json` once and computes hashes from the actual file bytes:

```bash
cd "$HOME/rl-triage-agent"
export TRIAGE_WORKSPACE="$PWD/workspace/single_run"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

case = Path(os.environ["TRIAGE_WORKSPACE"]) / "cases" / "example"
sha = lambda name: hashlib.sha256((case / name).read_bytes()).hexdigest()
manifest = {
    "schema": "single_run_context_v1",
    "telemetry_sha256": sha("telemetry.json"),
    "configuration_sha256": sha("configuration.json"),
    "facts": [
        {"id": "success_weight", "path": ["rewards", "task_success", "weight"]},
        {"id": "contact_bodies", "path": ["observations", "policy", "foot_contacts", "body_names"]},
    ],
}
with (case / "context.json").open("x", encoding="utf-8") as output:
    json.dump(manifest, output, indent=2)
PY
```

Replace `example` and the selected paths with those of your run. Each fact has a unique `id` and a nonempty `path` containing JSON object keys or nonnegative list indices. The first path segment must be a configuration category: `task`, `simulation`, `rewards`, `observations`, `sensors`, `scene`, `actions`, `terminations`, `commands`, `curriculum` or `events`. Selected values are JSON scalars or flat lists of scalars; numbers must be finite. Missing paths are errors. Other manifest fields and diagnostic roots such as `truth` or `answer_key` are rejected.

Hashes bind the chosen configuration bytes to the telemetry bytes supplied in the case. They establish file consistency, without attesting that an operator's snapshot came from the simulator. Retain the original configuration export when that provenance matters. Files and case directories used by this context path must stay within their case; symlink aliases are rejected.

## Inspect and cite evidence

The context tool is available without a model, API key or GPU:

```bash
cd "$HOME/rl-triage-agent"
export TRIAGE_WORKSPACE="$PWD/workspace/single_run"
uv run python - <<'PY'
import json
from rl_triage.single_run import get_context
print(json.dumps(get_context("example"), indent=2))
PY
```

The agent reads `single_run_context` and the telemetry overview before forming hypotheses. When selected configuration facts are present, `write_single_assessment` requires at least one structured claim:

```json
{"id": "success_weight", "value": 10.0}
```

At save time, the tool checks the source hashes again and compares every cited value with its selected fact. Unknown or duplicate IDs and mismatched values stop the save. An explicit zero-weight claim is rejected if the source weight is nonzero. The saved assessment retains the checked configuration premises and their source binding.

This check covers structured configuration claims. It does not verify every sentence in a hypothesis or establish a causal explanation. Assessments remain `hypotheses_only` and are written once. Analysis code continues to receive telemetry `run` and an empty `ref`; the configuration snapshot is not added to its filesystem permissions. This workflow does not start Isaac, change a setting, retrain or consume an approval.
