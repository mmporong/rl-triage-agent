"""V17 continuation summary with the controller-observed HOLD transition preserved."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from evals import summarize_balanced_recovery_continuation_v2 as V2

E = V2.E
V1 = V2.V1
ROOT = V2.ROOT
PLAN_SCHEMA = V2.PLAN_SCHEMA
PROFILE = V2.PROFILE
SEAL = V2.SEAL
POSTPROCESSING_NOTE = V2.POSTPROCESSING_NOTE
SOURCE_RELATIVE = "evals/summarize_balanced_recovery_continuation_v3.py"


def _join_order(plan: dict, prior: dict, current: dict) -> list[tuple[str, str]]:
    """Join the interrupted prefix to the resumed controller transition stream."""
    expected = E._expected_order(plan)
    prior_order = E._observed_order(prior)
    current_order = E._observed_order(current)
    if prior_order != expected[:8] or current_order != expected[7:]:
        raise E.ExportError("continuation controller order differs")
    combined = [*prior_order, *current_order[1:]]
    if combined != expected or current_order.count(E.EXPECTED_HOLD) != 1:
        raise E.ExportError("continuation HOLD/order join differs")
    return combined


def export(
    prepared: Path,
    out: Path,
    *,
    controller_sha: str,
    terminal: Path,
    terminal_sha: str,
    prior_terminal: Path,
    prior_terminal_sha: str,
) -> dict[str, Any]:
    root, target = V1._guard(prepared).resolve(strict=True), V1._guard(out).resolve()
    if target.exists():
        raise FileExistsError("continuation export must be fresh")
    if E._lf_sha(ROOT / "evals/run_registered_recovery_comparison_v1.py") != controller_sha:
        raise E.ExportError("controller source differs")
    manifest = E.controller._load(root)
    if E.controller.verify_prepared_manifest(
        root, repo_root=ROOT, registration_anchor=manifest["registration_anchor"]
    ) != manifest:
        raise E.ExportError("prepared continuation is not canonical")
    _, plan = V1._read(manifest["plan"], "registered continuation")
    status = E.controller.status(root)
    if json.loads((root / "final-controller-result.json").read_bytes()) != status:
        raise E.ExportError("final controller snapshot differs")
    V1._validate(manifest, plan, status)
    prior_timing, old_terminal = V1._prefix(
        manifest, plan, prior_terminal, prior_terminal_sha
    )
    current_timing = E.timing_evidence.read_timing(root)
    order = _join_order(plan, prior_timing, current_timing)
    current_terminal = E._terminal(
        V1._guard(terminal), terminal_sha, expected_mode="recovery_v17"
    )
    current_raw = json.loads(V1._guard(terminal).read_bytes())
    arms, totals, refs = V1._resources(status)
    prior = plan["continuation"]
    cumulative = V2._validate_accounting(plan, totals)
    staging = target.parent / f".{target.name}.base-{uuid.uuid4().hex}"
    try:
        base = E.base.summarize(root, staging)
        base_private = json.loads(
            (staging / "private/source-refs.json").read_text(encoding="utf-8")
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    resume_gap = (
        V1._utc(current_raw["supervisor_call_started_utc"])
        - V1._utc(old_terminal["finished_utc"])
    ).total_seconds()
    if resume_gap < 0:
        raise E.ExportError("parent phases overlap")
    public = {
        "schema": "balanced_recovery_resource_continuation_summary_v3",
        "status": "complete",
        "postprocessing_correction": (
            "cpu_launch_limits_are_caps_and_controller_hold_transition_is_observed"
        ),
        "measurement_source_closure_changed": False,
        "measurement_inputs_or_criteria_changed": False,
        "development_remeasurement": True,
        "new_heldout_claim": False,
        "planned_arms": 16,
        "attempted_arms": 15,
        "diagnostic_HOLD": 1,
        "imported_completed_effects": 7,
        "imported_completed_training": 1,
        "fresh_effects": 7,
        "api_calls": 0,
        "actual_order": [list(pair) for pair in order],
        "actual_order_matches_registered": True,
        "order_scope": "actual_controller_transition_order_including_original_diagnostic_HOLD",
        "observed_prior_transition_order": [
            list(pair) for pair in E._observed_order(prior_timing)
        ],
        "observed_new_transition_order": [
            list(pair) for pair in E._observed_order(current_timing)
        ],
        "interrupted_boundary_transition_collapsed_once": True,
        "original_diagnostic_HOLD_has_controller_transition": True,
        "original_diagnostic_HOLD_has_no_GPU_transition": True,
        "arms": arms,
        "resources": totals,
        "base_summary": base,
        "outcomes": {
            name: sum(arm["recovery_outcome"] == name for arm in base["arms"])
            for name in E.base.OUTCOMES
        },
        "sdk_accounting": {
            "prior_recovery_study": prior["prior_whole_study_sdk_launches"],
            "new": totals["new_sdk_launches"],
            "cumulative": cumulative,
            "cap": prior["cumulative_gpu_sdk_launch_cap"],
            "historical_not_added_twice": True,
            "HOLD_controller_transition_added_as_SDK": False,
        },
        "cpu_launch_accounting": {
            "checkpoint_observed": totals["cpu_checkpoint_launches"],
            "checkpoint_cap": plan["limits"]["cpu_checkpoint_launch_cap"],
            "telemetry_observed": totals["cpu_telemetry_launches"],
            "telemetry_cap": plan["limits"]["cpu_telemetry_launch_cap"],
            "caps_are_not_quotas": True,
        },
        "outer_supervisor": current_terminal,
        "elapsed": {
            "prior_supervisor_wall_s": old_terminal["supervisor_call_wall_s"],
            "new_supervisor_wall_s": current_raw["supervisor_call_wall_s"],
            "resume_gap_wall_s": resume_gap,
            "observed_calendar_span_s": (
                V1._utc(current_raw["finished_utc"])
                - V1._utc(old_terminal["supervisor_call_started_utc"])
            ).total_seconds(),
            "gap_includes_preparation_not_added_again": True,
            "continuous_per_arm_E2E_s": None,
            "complete_campaign_wall_s": None,
        },
        "sources": [
            {"logical_id": "prior_terminal", "sha256": prior_terminal_sha},
            {"logical_id": "terminal", "sha256": terminal_sha},
            {"logical_id": "registered_plan", "sha256": manifest["plan"]["sha256"]},
            {"logical_id": "prior_timing_seal", "sha256": E.io.file_sha(ROOT / SEAL)},
            {
                "logical_id": "postprocessing_source",
                "sha256": E._lf_sha(ROOT / SOURCE_RELATIVE),
            },
            {
                "logical_id": "postprocessing_correction_note",
                "sha256": E._lf_sha(ROOT / POSTPROCESSING_NOTE),
            },
        ],
    }
    E._assert_public(public)
    target.mkdir(parents=True)
    (target / "public").mkdir()
    (target / "public/summary.json").write_bytes(
        (
            json.dumps(public, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    )
    (target / "private").mkdir()
    timing_refs = [
        {"logical_id": f"prior-timing:{Path(ref['path']).name}", **ref}
        for ref in prior_timing["private_refs"]
    ]
    (target / "private/source-refs.json").write_bytes(
        (
            json.dumps(
                {"refs": [*base_private["refs"], *refs, *timing_refs]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode()
    )
    return public


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("prepared", "out", "terminal", "prior-terminal"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("controller-sha", "terminal-sha", "prior-terminal-sha"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    prepared, out = args.pop("prepared"), args.pop("out")
    result = export(prepared, out, **args)
    print(json.dumps({
        "status": result["status"], "outcomes": result["outcomes"],
        "SDK": result["resources"]["new_sdk_launches"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
