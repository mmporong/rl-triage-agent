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

## Prepare and inspect a run

`rl-triage prepare-run` creates this layout and computes the context hashes. It
accepts a telemetry JSON export with `summary` and `series` objects. TensorBoard
event files must first be exported with the existing `bench/extract_tb.py` tool
in the Isaac environment; the importer does not parse binary event files.

```bash
cd "$HOME/rl-triage-agent"
uv sync
uv run rl-triage prepare-run --telemetry bench/runs/c01_s42.telemetry.json \
  --workspace workspace/imported-c01 --case c01
uv run rl-triage inspect-run c01 --workspace workspace/imported-c01 \
  --report workspace/c01-input-report.json
```

This example reads a recorded public run. Both commands run locally without model
calls or a GPU. The report contains the measured summaries, the available evidence
states, and selected configuration facts. `model_evaluation: not_run` makes the
remaining model step explicit. Missing values remain unavailable; preparation
does not replace them with zeros or infer a cause from a configured reward.

When adding configuration evidence, pass both `--configuration` and `--facts`.
The facts file contains only the selection array, for example:

```json
[
  {"id": "success_weight", "path": ["rewards", "task_success", "weight"]},
  {"id": "contact_bodies", "path": ["observations", "policy", "foot_contacts", "body_names"]}
]
```

The selection refers to the synthetic configuration example below; choose paths
that exist in the snapshot from your own run.

```bash
uv run rl-triage prepare-run \
  --telemetry "$HOME/rl-triage-input/telemetry.json" \
  --configuration "$HOME/rl-triage-input/configuration.json" \
  --facts "$HOME/rl-triage-input/facts.json" \
  --workspace workspace/my-run --case my-run
```

Original telemetry bytes and prepared telemetry bytes have separate hashes in
`import.json`. Preparation exports only `summary` and `series`, with supported
summary fields, so unrelated run paths and metadata stay out of the agent input.
The configuration snapshot is copied byte for byte. Its hash and the selected
paths are checked through the existing context contract. These hashes establish
file consistency, not simulator provenance or a correct diagnosis.

Inspection verifies every prepared file against `import.json` before returning
evidence. Changed or missing inputs stop inspection. Existing workspaces and
report files are rejected instead of replaced. For a new export or assessment
attempt, choose a new workspace path.

After inspection, set `TRIAGE_WORKSPACE` to the prepared directory and choose the
existing single-run model workflow. Model calls require a configured API account;
input preparation and inspection do not create one.

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
