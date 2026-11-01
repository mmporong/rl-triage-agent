# pyright: reportMissingImports=false
"""새 10-case×5-policy 결과를 source 독립 case 단위로 집계한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import lineage_five_policy_driver_v1 as DRIVER
import lineage_five_policy_nim_transport_v1 as NIM
import run_lineage_five_policy_v1 as RUNNER

SCHEMA = "lineage_five_policy_summary_v1"
BATCH_SCHEMA = "lineage_five_policy_batch_result_v1"
DESIGN_SCHEMA = "agentic_five_type_evaluation_design_v1"
SOURCE_FILES = (
    "evals/lineage_five_policy_summary_v1.py",
    "tests/test_lineage_five_policy_summary_v1.py",
    "evals/lineage_five_policy_driver_v1.py",
    "bench/run_lineage_five_policy_v1.py",
)
TECHNICAL_HOLDS = frozenset(
    {
        "same_play_frames_unavailable",
        "visual_quality_proof_missing",
        "gpu_resource_busy",
        "probe_outcome_unknown",
        "transport_pending_no_resend",
        "model_infra_unknown",
        "model_provider_unknown",
    }
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자: {token}")
        ),
    )
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON object여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve()
    _require(
        target.is_file() and not target.is_symlink(), f"원문 파일이 없다: {target.name}"
    )
    return {"path": str(target), "sha256": _sha(target)}


def _bound_file(value: object, label: str) -> Path:
    _require(
        isinstance(value, Mapping) and set(value) == {"path", "sha256"}, f"{label} ref"
    )
    ref = cast(Mapping[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve()
    _require(
        path.is_file() and not path.is_symlink() and _sha(path) == ref.get("sha256"),
        f"{label} SHA",
    )
    return path


def _write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")


def source_files() -> tuple[str, ...]:
    return tuple(
        sorted(set(SOURCE_FILES) | set(DRIVER.SOURCE_FILES) | set(RUNNER.SOURCE_FILES))
    )


def _study(path: Path) -> dict[str, Any]:
    value = _read(path)
    policies = [
        *value.get("primary_policies", []),
        *value.get("auxiliary_policies", []),
    ]
    rules = value.get("outcome_rules", {})
    uncertainty = value.get("uncertainty", {})
    _require(
        value.get("schema") == DESIGN_SCHEMA
        and policies == list(DRIVER.ARMS)
        and value.get("primary_contrast") == ["logs_only", "fixed_eligible"]
        and value.get("secondary_contrast") == ["logs_plus_frames", "logs_only"]
        and value.get("known_fault_types") is True
        and value.get("new_seed_is_not_new_fault_family") is True
        and rules.get("minimum_qualified_scope_for_positive_summary")
        == {"cases": 6, "types": 3}
        and rules.get("probe_count_to_total_time_savings_conversion") is False
        and rules.get("small_sample_general_superiority_claim") is False
        and uncertainty.get("repeat_calls_or_frames_are_independent_cases") is False
        and uncertainty.get("unit") == "independent_policy_lineage",
        "study v2 결과 판정 계약이 다르다",
    )
    case_plan = value.get("case_plan")
    _require(
        isinstance(case_plan, list)
        and len(case_plan) == 10
        and [row.get("case_id") for row in case_plan]
        == [f"case-{i:02d}" for i in range(1, 11)],
        "study case10 순서가 다르다",
    )
    case_plan_value = cast(list[dict[str, Any]], case_plan)
    _require(
        [row.get("arm") for row in case_plan_value]
        == ["NONE", "NONE", "F1", "F1", "F2", "F2", "COMPOUND", "COMPOUND", "F5", "F5"],
        "study case10 fault arm이 다르다",
    )
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0,
        f"{label}은 유한한 음이 아닌 수여야 한다",
    )
    return float(cast(int | float, value))


def _call_rows(
    result_path: Path, row: Mapping[str, Any], *, allow_fixture: bool
) -> list[dict[str, Any]]:
    expected = row.get("model_calls")
    _require(type(expected) is int and expected >= 0, "model_calls 형식")
    expected_count = cast(int, expected)
    calls: list[dict[str, Any]] = []
    for path in sorted(result_path.parent.glob("step[0-9][0-9]/model-result.json")):
        response = _read(path)
        ref = response.get("call_ref")
        if ref is None:
            _require(allow_fixture, "actual model result에 verified call ref가 없다")
            continue
        _require(
            isinstance(ref, Mapping)
            and set(ref) == {"path", "sha256"}
            and isinstance(ref.get("path"), str),
            "provider call ref 형식",
        )
        call_ref = cast(Mapping[str, Any], ref)
        call_path = Path(cast(str, call_ref["path"]))
        verified = NIM.verify_call(call_path)
        _require(
            verified.get("call_ref") == call_ref,
            "provider call ref가 verified receipt와 다르다",
        )
        calls.append(verified)
    if not allow_fixture:
        _require(
            len(calls) == expected_count,
            "actual model_calls와 verified provider calls 수가 다르다",
        )
    else:
        _require(
            len(calls) <= expected_count,
            "fixture verified calls가 model_calls보다 많다",
        )
    return calls


def _row_metrics(
    result_path: Path, row: dict[str, Any], *, allow_fixture: bool
) -> dict[str, Any]:
    calls = _call_rows(result_path, row, allow_fixture=allow_fixture)
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_unknown = 0
    provider_wall = 0.0
    call_ids: list[str] = []
    for call in calls:
        ref = call["call_ref"]
        call_ids.append(str(Path(ref["path"]).resolve()))
        provider_wall += _finite_nonnegative(call.get("wall_s"), "provider wall_s")
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            usage_unknown += 1
            continue
        aliases = {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
        }
        for target, names in aliases.items():
            value = next((usage[name] for name in names if name in usage), None)
            _require(type(value) is int and value >= 0, f"provider {target}")
            token_totals[target] += cast(int, value)
    steps = row.get("steps")
    _require(isinstance(steps, list) and len(steps) <= 3, "paired probe steps")
    probe_steps = cast(list[object], steps)
    owned = 0.0
    owned_unknown = 0
    for step_value in probe_steps:
        step = cast(Mapping[str, Any], step_value)
        _require(isinstance(step, Mapping), "probe step shape")
        value = step.get("owned_child_wall_s")
        if value is None:
            owned_unknown += 1
        else:
            owned += _finite_nonnegative(value, "owned child wall_s")
    parent_wall = _finite_nonnegative(row.get("parent_wall_s"), "parent wall_s")
    scoring = row.get("scoring")
    qualification = row.get("qualification")
    _require(
        isinstance(scoring, Mapping) and isinstance(qualification, Mapping),
        "row scoring/qualification",
    )
    scoring_value = cast(Mapping[str, Any], scoring)
    qualification_value = cast(Mapping[str, Any], qualification)
    semantic = row.get("semantic_invalid")
    infra = row.get("infra_unknown")
    provider_unknown = row.get("provider_unknown")
    _require(
        all(
            type(value) is int and value >= 0
            for value in (semantic, infra, provider_unknown)
        ),
        "invalid/unknown counters",
    )
    semantic_count = cast(int, semantic)
    infra_count = cast(int, infra)
    provider_unknown_count = cast(int, provider_unknown)
    hold_reason = row.get("hold_reason")
    frame_citations = row.get("frame_fact_citations")
    source_digest = row.get("source_digest")
    source_index = row.get("source_index")
    checkpoint_bindings = row.get("checkpoint_bindings")
    normal_proposal = row.get("native_normal_intervention_proposal")
    _require(
        type(frame_citations) is int and frame_citations >= 0,
        "frame_fact_citations 형식",
    )
    _require(
        isinstance(source_digest, str)
        and len(source_digest) == 64
        and all(character in "0123456789abcdef" for character in source_digest),
        "source_digest 형식",
    )
    _require(isinstance(source_index, Mapping), "source_index 형식")
    _require(isinstance(checkpoint_bindings, Mapping), "checkpoint_bindings 형식")
    _require(
        normal_proposal is None or type(normal_proposal) is bool,
        "normal proposal 형식",
    )
    return {
        "case_id": row["case_id"],
        "policy": row["arm"],
        "qualification": deepcopy(qualification_value),
        "scoring": deepcopy(scoring_value),
        "correct": scoring_value.get("correct_any_family") is True,
        "wrong": scoring_value.get("wrong_confirmation") is True,
        "compound_partial": scoring_value.get("compound_partial") is True,
        "compound_full": scoring_value.get("full_cause_confirmation") is True
        and qualification_value.get("arm") == "COMPOUND",
        "normal_shadow_proposal": normal_proposal is True,
        "normal_shadow_observed": type(normal_proposal) is bool,
        "host_gate": qualification_value.get("arm") == "NONE"
        and hold_reason in {"normal_host_gate", "normal_shadow_only_host_gate"}
        and not probe_steps,
        "contract_invalid_events": semantic_count,
        "infra_unknown_events": infra_count + provider_unknown_count,
        "provider_unknown_events": provider_unknown_count,
        "source_exclusion": (
            hold_reason[len("source_") :]
            if isinstance(hold_reason, str) and hold_reason.startswith("source_")
            else None
        ),
        "hold": row.get("end_state") == "HOLD",
        "hold_reason": hold_reason,
        "paired_probes": len(probe_steps),
        "model_calls": row["model_calls"],
        "provider_tokens": token_totals,
        "provider_usage_unknown_calls": usage_unknown,
        "provider_wall_s": provider_wall,
        "owned_child_wall_s": owned,
        "owned_child_wall_unknown_probes": owned_unknown,
        "parent_wall_s": parent_wall,
        "frame_fact_citations": cast(int, frame_citations),
        "source_digest": source_digest,
        "source_index": deepcopy(source_index),
        "checkpoint_bindings": deepcopy(checkpoint_bindings),
        "confirmed_subtype": row.get("confirmed_subtype"),
        "steps": deepcopy(probe_steps),
        "call_ids": call_ids,
        "technical_missing": bool(
            semantic_count
            or infra_count
            or provider_unknown_count
            or hold_reason in TECHNICAL_HOLDS
        ),
    }


def _grid(
    batch_path: Path, *, allow_fixture: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch = _read(batch_path)
    rows = batch.get("rows")
    _require(
        batch.get("schema") == BATCH_SCHEMA
        and batch.get("status") == "complete"
        and batch.get("planned_cases") == 10
        and batch.get("planned_rows") == 50
        and isinstance(rows, list)
        and len(rows) == 50,
        "five-policy batch complete50 계약이 아니다",
    )
    expected = {
        (f"case-{case:02d}", policy) for case in range(1, 11) for policy in DRIVER.ARMS
    }
    observed: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    result_paths: set[Path] = set()
    scopes: set[str] = set()
    for outer in cast(list[Any], rows):
        _require(isinstance(outer, Mapping), "batch row shape")
        case_id, policy = outer.get("case_id"), outer.get("arm")
        key = (case_id, policy)
        _require(
            key in expected and key not in observed, "case×policy 누락·중복·미등록"
        )
        observed.add(cast(tuple[str, str], key))
        result_path = _bound_file(outer.get("result"), f"{case_id}/{policy} result")
        _require(
            result_path not in result_paths, "여러 row가 같은 result 원문을 재사용한다"
        )
        result_paths.add(result_path)
        detail = _read(result_path)
        _require(
            detail.get("schema") == DRIVER.RESULT_SCHEMA
            and detail.get("case_id") == case_id
            and detail.get("arm") == policy
            and detail.get("end_state") == outer.get("end_state")
            and detail.get("scoring") == outer.get("scoring"),
            "batch row/result 원문 결합이 다르다",
        )
        scope = detail.get("scope")
        _require(scope in {"actual_backend", "cpu_fixture"}, "result scope")
        scopes.add(cast(str, scope))
        results.append(_row_metrics(result_path, detail, allow_fixture=allow_fixture))
    _require(observed == expected, "case10×policy5 grid가 완결되지 않았다")
    expected_scope = "cpu_fixture" if allow_fixture else "actual_backend"
    _require(
        scopes == {expected_scope},
        "cpu_fixture는 allow_fixture에서만 허용되며 actual로 승격할 수 없다",
    )
    call_ids = [identifier for row in results for identifier in row["call_ids"]]
    _require(
        len(call_ids) == len(set(call_ids)),
        "provider call receipt가 여러 row에 중복 집계됐다",
    )
    return batch, results


def _bind_actual_sources(
    batch: Mapping[str, Any], rows: list[dict[str, Any]], *, allow_fixture: bool
) -> None:
    if allow_fixture:
        return
    protocol_path = _bound_file(batch.get("execution_protocol"), "execution protocol")
    config_path = _bound_file(batch.get("execution_config"), "execution config")
    anchor = batch.get("execution_anchor")
    _require(
        isinstance(anchor, str)
        and len(anchor) == 40
        and all(character in "0123456789abcdef" for character in anchor),
        "execution anchor 형식",
    )
    _protocol, _config, sources = RUNNER.preflight(protocol_path, anchor, config_path)
    _require(
        set(sources) == {f"case-{case:02d}" for case in range(1, 11)}, "actual source10"
    )
    for row in rows:
        source = sources[row["case_id"]]["source"]
        _require(
            row["source_digest"] == DRIVER.BASE.digest(source)
            and row["source_index"] == source["index"]
            and row["checkpoint_bindings"] == source["packet"]["input_bindings"],
            "result source digest/index/CP binding이 preflight source와 다르다",
        )


def _source_independence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest_by_case: dict[str, set[str]] = {}
    index_by_case: dict[str, set[str]] = {}
    bindings_by_case: dict[str, set[str]] = {}
    qualification_by_case: dict[str, set[str]] = {}
    for row in rows:
        case_id = row["case_id"]
        digest_by_case.setdefault(case_id, set()).add(row["source_digest"])
        index_by_case.setdefault(case_id, set()).add(
            json.dumps(row["source_index"], sort_keys=True)
        )
        bindings_by_case.setdefault(case_id, set()).add(
            json.dumps(row["checkpoint_bindings"], sort_keys=True)
        )
        qualification_by_case.setdefault(row["case_id"], set()).add(
            json.dumps(row["qualification"], sort_keys=True)
        )
    _require(
        len(digest_by_case) == 10
        and all(len(digests) == 1 for digests in digest_by_case.values()),
        "같은 case의 arm별 source_digest가 다르다",
    )
    _require(
        all(len(values) == 1 for values in index_by_case.values())
        and all(len(values) == 1 for values in bindings_by_case.values())
        and all(len(values) == 1 for values in qualification_by_case.values()),
        "같은 case의 arm별 source index/CP/qualification이 다르다",
    )
    case_cp_groups: dict[str, list[str]] = {}
    reference_cp_groups: dict[str, list[str]] = {}
    for case_id, encoded_values in bindings_by_case.items():
        bindings = json.loads(next(iter(encoded_values)))
        _require(set(bindings) == {"case", "reference"}, "checkpoint role binding")
        for role in ("case", "reference"):
            binding = bindings.get(role)
            _require(
                isinstance(binding, Mapping)
                and isinstance(binding.get("checkpoint_sha256"), str)
                and len(binding["checkpoint_sha256"]) == 64,
                f"{role} checkpoint binding",
            )
        case_cp = bindings["case"]["checkpoint_sha256"]
        reference_cp = bindings["reference"]["checkpoint_sha256"]
        _require(
            case_cp != reference_cp, "case가 자기 checkpoint를 reference로 사용한다"
        )
        case_cp_groups.setdefault(case_cp, []).append(case_id)
        reference_cp_groups.setdefault(reference_cp, []).append(case_id)
    duplicate_case_groups = sorted(
        (sorted(case_ids) for case_ids in case_cp_groups.values() if len(case_ids) > 1),
        key=lambda case_ids: case_ids[0],
    )
    reference_dependency_groups = sorted(
        (
            sorted(case_ids)
            for case_ids in reference_cp_groups.values()
            if len(case_ids) > 1
        ),
        key=lambda case_ids: case_ids[0],
    )
    return {
        "verified": not duplicate_case_groups,
        "duplicate_case_groups": duplicate_case_groups,
        "unique_case_checkpoints": len(case_cp_groups),
        "reference_checkpoint_dependency_groups": reference_dependency_groups,
    }


def _policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    selected = [row for row in rows if row["policy"] == policy]
    _require(len(selected) == 10, f"{policy} case denominator")
    exclusions: dict[str, int] = {}
    for row in selected:
        reason = row["source_exclusion"]
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
    tokens = {
        name: sum(row["provider_tokens"][name] for row in selected)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    qualified = [
        row
        for row in selected
        if row["qualification"].get("arm") != "NONE"
        and row["qualification"].get("diagnosis_eligible") is True
    ]
    return {
        "planned_cases": 10,
        "independent_source_lineages": 10,
        "normal_cases": sum(
            row["qualification"].get("arm") == "NONE" for row in selected
        ),
        "qualified_fault_cases": len(qualified),
        "correct_any_family": sum(row["correct"] for row in qualified),
        "compound_partial": sum(row["compound_partial"] for row in qualified),
        "compound_full": sum(row["compound_full"] for row in qualified),
        "wrong_confirmation": sum(row["wrong"] for row in qualified),
        "normal_model_shadow_proposal": sum(
            row["normal_shadow_proposal"] for row in selected
        ),
        "normal_model_observed": sum(
            row["normal_shadow_observed"]
            for row in selected
            if row["qualification"].get("arm") == "NONE"
        ),
        "normal_model_unobserved": sum(
            not row["normal_shadow_observed"]
            for row in selected
            if row["qualification"].get("arm") == "NONE"
        ),
        "normal_host_gate": sum(row["host_gate"] for row in selected),
        "contract_invalid_rows": sum(
            row["contract_invalid_events"] > 0 for row in selected
        ),
        "contract_invalid_events": sum(
            row["contract_invalid_events"] for row in selected
        ),
        "infra_unknown_rows": sum(row["infra_unknown_events"] > 0 for row in selected),
        "infra_unknown_events": sum(row["infra_unknown_events"] for row in selected),
        "provider_unknown_events": sum(
            row["provider_unknown_events"] for row in selected
        ),
        "source_exclusion": exclusions,
        "hold_rows": sum(row["hold"] for row in selected),
        "paired_probes": sum(row["paired_probes"] for row in selected),
        "model_calls": sum(row["model_calls"] for row in selected),
        "provider_tokens": tokens,
        "provider_usage_unknown_calls": sum(
            row["provider_usage_unknown_calls"] for row in selected
        ),
        "provider_wall_s": sum(row["provider_wall_s"] for row in selected),
        "owned_child_wall_s": sum(row["owned_child_wall_s"] for row in selected),
        "owned_child_wall_unknown_probes": sum(
            row["owned_child_wall_unknown_probes"] for row in selected
        ),
        "parent_wall_s": sum(row["parent_wall_s"] for row in selected),
        "frame_fact_citations": sum(
            cast(int, row["frame_fact_citations"]) for row in selected
        ),
    }


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _paired(left: str, right: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["case_id"], row["policy"]): row for row in rows}
    pairs = []
    for case in range(1, 11):
        case_id = f"case-{case:02d}"
        a, b = by_key[(case_id, left)], by_key[(case_id, right)]
        eligible = (
            a["qualification"].get("arm") != "NONE"
            and a["qualification"].get("diagnosis_eligible") is True
        )
        _require(
            a["qualification"] == b["qualification"], "policy별 qualification이 다르다"
        )
        valid = eligible and not a["technical_missing"] and not b["technical_missing"]
        pairs.append(
            {
                "case_id": case_id,
                "eligible": eligible,
                "technical_valid": valid,
                "left_correct": a["correct"],
                "right_correct": b["correct"],
                "left_wrong": a["wrong"],
                "right_wrong": b["wrong"],
                "left_paired_probes": a["paired_probes"],
                "right_paired_probes": b["paired_probes"],
                "left_owned_child_wall_s": a["owned_child_wall_s"],
                "right_owned_child_wall_s": b["owned_child_wall_s"],
            }
        )
    valid = [row for row in pairs if row["technical_valid"]]
    left_wins = sum(row["left_correct"] and not row["right_correct"] for row in valid)
    right_wins = sum(row["right_correct"] and not row["left_correct"] for row in valid)
    discordant = left_wins + right_wins
    if discordant:
        tail = sum(
            math.comb(discordant, k) for k in range(min(left_wins, right_wins) + 1)
        )
        exact_p = min(1.0, 2 * tail / (2**discordant))
    else:
        exact_p = 1.0
    common_correct = [
        row for row in valid if row["left_correct"] and row["right_correct"]
    ]
    return {
        "left": left,
        "right": right,
        "case_pairs": pairs,
        "qualified_cases": sum(row["eligible"] for row in pairs),
        "technical_valid_pairs": len(valid),
        "technical_missing_pairs": sum(
            row["eligible"] and not row["technical_valid"] for row in pairs
        ),
        "left_correct": sum(row["left_correct"] for row in valid),
        "right_correct": sum(row["right_correct"] for row in valid),
        "left_wrong": sum(row["left_wrong"] for row in valid),
        "right_wrong": sum(row["right_wrong"] for row in valid),
        "left_better": left_wins,
        "right_better": right_wins,
        "discordant": discordant,
        "exact_two_sided_sign_p": exact_p,
        "left_win_fraction_wilson95": _wilson(left_wins, discordant),
        "small_sample": len(valid) < 30,
        "general_superiority_claim": False,
        "common_correct_cases": len(common_correct),
        "common_correct_left_probe_sum": sum(
            row["left_paired_probes"] for row in common_correct
        ),
        "common_correct_right_probe_sum": sum(
            row["right_paired_probes"] for row in common_correct
        ),
        "common_correct_left_owned_wall_s": sum(
            row["left_owned_child_wall_s"] for row in common_correct
        ),
        "common_correct_right_owned_wall_s": sum(
            row["right_owned_child_wall_s"] for row in common_correct
        ),
    }


def _primary(
    pair: dict[str, Any],
    metrics: Mapping[str, Any],
    minimum: Mapping[str, int],
    qualified_types: int,
) -> str:
    left, right = metrics[pair["left"]], metrics[pair["right"]]
    enough = (
        pair["qualified_cases"] >= minimum["cases"]
        and qualified_types >= minimum["types"]
    )
    complete = pair["technical_valid_pairs"] == pair["qualified_cases"]
    if not enough or not complete:
        return "inconclusive"
    normal_denominator_equal = (
        left["normal_model_observed"] == right["normal_model_observed"]
    )
    normal_ok = normal_denominator_equal and (
        left["normal_model_shadow_proposal"] <= right["normal_model_shadow_proposal"]
    )
    wrong_ok = pair["left_wrong"] <= pair["right_wrong"]
    if pair["left_correct"] > pair["right_correct"] and wrong_ok and normal_ok:
        return "positive_in_registered_scope"
    if pair["left_correct"] < pair["right_correct"] or not wrong_ok:
        return "regression"
    if not normal_denominator_equal:
        return "inconclusive"
    if not normal_ok:
        return "regression"
    lower_probe = (
        pair["left_correct"] == pair["right_correct"]
        and wrong_ok
        and normal_ok
        and pair["common_correct_cases"] >= 2
        and pair["common_correct_left_probe_sum"]
        < pair["common_correct_right_probe_sum"]
    )
    return "same_accuracy_lower_probe_count" if lower_probe else "no_observed_gain"


def _visual_confirmation(row: Mapping[str, Any]) -> bool:
    if (
        not row["correct"]
        or not isinstance(row["frame_fact_citations"], int)
        or row["frame_fact_citations"] <= 0
    ):
        return False
    confirmed = row["confirmed_subtype"]
    registry = DRIVER.C.RC.registry()
    confirming = (
        DRIVER.C.S.M._confirming(registry, confirmed)
        if isinstance(confirmed, str)
        else None
    )
    return confirming is not None and any(
        step.get("probe") == confirming
        and step.get("outcome") == "abnormal"
        and any(
            str(fact).startswith("vision:frame:") for fact in step.get("fact_ids", [])
        )
        for step in row["steps"]
    )


def _secondary(
    pair: dict[str, Any], metrics: Mapping[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left, right = metrics[pair["left"]], metrics[pair["right"]]
    valid_visual = sorted(
        row["case_id"]
        for row in rows
        if row["policy"] == pair["left"] and _visual_confirmation(row)
    )
    complete = pair["technical_valid_pairs"] == pair["qualified_cases"]
    wrong_ok = pair["left_wrong"] <= pair["right_wrong"]
    normal_denominator_equal = (
        left["normal_model_observed"] == right["normal_model_observed"]
    )
    normal_ok = normal_denominator_equal and (
        left["normal_model_shadow_proposal"] <= right["normal_model_shadow_proposal"]
    )
    accuracy_gain = pair["left_correct"] > pair["right_correct"]
    cost_gain = (
        pair["left_correct"] == pair["right_correct"]
        and pair["common_correct_cases"] >= 2
        and pair["common_correct_left_probe_sum"]
        < pair["common_correct_right_probe_sum"]
    )
    positive = (
        complete
        and wrong_ok
        and normal_ok
        and bool(valid_visual)
        and (accuracy_gain or cost_gain)
    )
    return {
        "outcome": (
            "inconclusive"
            if not complete or not normal_denominator_equal
            else "visual_gain"
            if positive
            else "no_observed_gain"
        ),
        "valid_frame_citation_and_confirming_probe_cases": valid_visual,
        "frame_citation_count_alone_is_effect": False,
        "accuracy_gain": accuracy_gain,
        "conditional_probe_cost_gain": cost_gain,
        "normal_model_observed_denominator_equal": normal_denominator_equal,
        "video_advantage_confirmed": positive,
    }


def _input_unavailable_summary(
    batch_path: Path, study_path: Path, out: Path, study: Mapping[str, Any]
) -> dict[str, Any]:
    batch = _read(batch_path)
    rows = batch.get("rows")
    unavailable_count = batch.get("input_unavailable_rows")
    _require(
        batch.get("schema") == BATCH_SCHEMA
        and batch.get("status") == "complete"
        and batch.get("comparison_status") == "inconclusive_input_unavailable"
        and batch.get("planned_cases") == 10
        and batch.get("planned_rows") == 50
        and isinstance(rows, list)
        and len(rows) == 50
        and type(unavailable_count) is int
        and unavailable_count > 0,
        "input-unavailable batch complete50 계약이 아니다",
    )
    protocol_path = _bound_file(batch.get("execution_protocol"), "execution protocol")
    config_path = _bound_file(batch.get("execution_config"), "execution config")
    anchor = batch.get("execution_anchor")
    _require(
        isinstance(anchor, str)
        and len(anchor) == 40
        and all(character in "0123456789abcdef" for character in anchor),
        "execution anchor 형식",
    )
    protocol, _config, sources = RUNNER.preflight(protocol_path, anchor, config_path)
    expected = {
        (f"case-{case:02d}", policy) for case in range(1, 11) for policy in DRIVER.ARMS
    }
    observed: set[tuple[str, str]] = set()
    paths: set[Path] = set()
    missing_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    call_ids: list[str] = []
    study_arms = {row["case_id"]: row["arm"] for row in study["case_plan"]}
    for outer_value in cast(list[Any], rows):
        _require(isinstance(outer_value, Mapping), "batch row shape")
        outer = cast(Mapping[str, Any], outer_value)
        case_id, policy = outer.get("case_id"), outer.get("arm")
        key = (case_id, policy)
        _require(
            key in expected and key not in observed, "case×policy 누락·중복·미등록"
        )
        observed.add(cast(tuple[str, str], key))
        result_path = _bound_file(outer.get("result"), f"{case_id}/{policy} result")
        _require(result_path not in paths, "여러 row가 같은 result 원문을 재사용한다")
        paths.add(result_path)
        detail = _read(result_path)
        _require(
            detail.get("schema") == DRIVER.RESULT_SCHEMA
            and detail.get("scope") == "actual_backend"
            and detail.get("case_id") == case_id
            and detail.get("arm") == policy
            and detail.get("end_state") == outer.get("end_state")
            and detail.get("scoring") == outer.get("scoring")
            and detail.get("source_digest") == protocol["source_digests"][case_id],
            "input-unavailable batch row/result/source 결합이 다르다",
        )
        item = sources[case_id]
        row_ref = {
            "case_id": case_id,
            "policy": policy,
            "result": _ref(result_path),
        }
        if detail.get("input_unavailable") == 1:
            guard = detail.get("input_guard")
            _require(
                detail.get("end_state") == "HOLD"
                and detail.get("hold_reason") == "registered_input_unavailable"
                and detail.get("qualification") is None
                and detail.get("scoring") is None
                and detail.get("checkpoint_bindings") is None
                and detail.get("steps") == []
                and detail.get("model_calls") == 0
                and detail.get("native_normal_intervention_proposal") is None
                and detail.get("confirmed_subtype") is None
                and detail.get("final_view_digest") is None
                and detail.get("remaining_candidates") is None
                and isinstance(guard, Mapping)
                and guard.get("schema") == DRIVER.GUARD.SCHEMA
                and guard.get("decision") == "HOLD"
                and guard.get("stage") == "input"
                and guard.get("reason") == "registered_input_unavailable"
                and guard.get("approval_granted") is False
                and guard.get("dispatch_granted") is False
                and guard.get("automatic_retry") is False
                and item.get("input_hold", {}).get("reason")
                == "registered_input_unavailable",
                "registered missing-input HOLD 계약이 다르다",
            )
            missing_rows.append(row_ref)
            continue
        _require(detail.get("input_unavailable") == 0, "input availability count")
        checked = _row_metrics(result_path, detail, allow_fixture=False)
        source = item.get("source")
        _require(
            isinstance(source, Mapping)
            and checked["source_digest"] == DRIVER.BASE.digest(source)
            and checked["source_index"] == source.get("index")
            and checked["checkpoint_bindings"]
            == source.get("packet", {}).get("input_bindings")
            and checked["qualification"].get("arm") == study_arms[case_id],
            "observed row source binding이 다르다",
        )
        call_ids.extend(checked["call_ids"])
        observed_rows.append(row_ref)
    _require(observed == expected, "case10×policy5 grid가 완결되지 않았다")
    _require(
        len(missing_rows) == unavailable_count
        and len(missing_rows) + len(observed_rows) == 50,
        "input_unavailable_rows count가 원문과 다르다",
    )
    _require(len(call_ids) == len(set(call_ids)), "provider call receipt 중복")
    batch_parent_wall = _finite_nonnegative(
        batch.get("parent_wall_s"), "batch parent wall_s"
    )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "actual_backend_inconclusive_input_unavailable",
        "batch": _ref(batch_path),
        "study": _ref(study_path),
        "planned_cases": 10,
        "planned_policies": 5,
        "planned_rows": 50,
        "denominator": {
            "unit": "independent_source_lineage",
            "planned_cases": 10,
            "valid_result_cases": None,
            "normal_cases": None,
            "independent_cases": None,
            "qualified_fault_cases": None,
            "qualified_fault_types": None,
            "reason": "registered_input_unavailable",
        },
        "policies": {policy: None for policy in DRIVER.ARMS},
        "primary": {
            "outcome": "inconclusive",
            "reason": "registered_input_unavailable",
            "preregistered_contrast": ["logs_only", "fixed_eligible"],
            "case_pairs": None,
            "accuracy": None,
            "inspection_cost": None,
        },
        "secondary": {
            "outcome": "inconclusive",
            "reason": "registered_input_unavailable",
            "preregistered_contrast": ["logs_plus_frames", "logs_only"],
            "case_pairs": None,
            "accuracy": None,
            "inspection_cost": None,
            "video_advantage_confirmed": False,
        },
        "input_unavailable": {
            "rows": len(missing_rows),
            "missing_rows": missing_rows,
            "observed_rows": observed_rows,
            "accuracy_observed": False,
            "cost_observed": False,
            "qualification_observed_for_missing": False,
        },
        "claims": {
            "general_superiority": False,
            "model_version_gain": False,
            "three_vs_three_point_five_version_gain": False,
            "probe_time_converted_to_gpu_savings": False,
            "video_advantage_confirmed": False,
            "fixture_promoted_to_actual": False,
        },
        "batch_parent_wall_s": batch_parent_wall,
        "source_closure_count": len(source_files()),
        "source_files_lf": {
            name: _sha(ROOT / name, lf=True) for name in source_files()
        },
    }
    _write_new(out, result)
    return deepcopy(result)


def summarize(
    batch_result: Path, study_path: Path, out: Path, *, allow_fixture: bool = False
) -> dict[str, Any]:
    """50개 원문을 재검증해 case-level 짝비교를 write-once로 봉인한다."""
    batch_path, study_path, out = map(Path, (batch_result, study_path, out))
    _require(not out.exists(), "summary out은 fresh 파일이어야 한다")
    study = _study(study_path)
    initial_batch = _read(batch_path)
    if initial_batch.get("comparison_status") == "inconclusive_input_unavailable":
        _require(
            not allow_fixture, "input-unavailable 종단은 actual batch에만 허용한다"
        )
        return _input_unavailable_summary(batch_path, study_path, out, study)
    batch, rows = _grid(batch_path, allow_fixture=allow_fixture)
    _bind_actual_sources(batch, rows, allow_fixture=allow_fixture)
    independence = _source_independence(rows)
    study_arms = {row["case_id"]: row["arm"] for row in study["case_plan"]}
    _require(
        all(
            row["qualification"].get("arm") == study_arms[row["case_id"]]
            for row in rows
        ),
        "result qualification이 study case arm과 다르다",
    )
    batch_parent_wall = _finite_nonnegative(
        batch.get("parent_wall_s"), "batch parent wall_s"
    )
    metrics = {policy: _policy(rows, policy) for policy in DRIVER.ARMS}
    primary = _paired("logs_only", "fixed_eligible", rows)
    secondary = _paired("logs_plus_frames", "logs_only", rows)
    eligible_rows = [
        row
        for row in rows
        if row["policy"] == "logs_only"
        and row["qualification"].get("arm") != "NONE"
        and row["qualification"].get("diagnosis_eligible") is True
    ]
    qualified_types = len({row["qualification"]["arm"] for row in eligible_rows})
    minimum = study["outcome_rules"]["minimum_qualified_scope_for_positive_summary"]
    primary_outcome = _primary(primary, metrics, minimum, qualified_types)
    visual = _secondary(secondary, metrics, rows)
    if not independence["verified"]:
        primary_outcome = "inconclusive"
        visual["outcome"] = "inconclusive"
        visual["video_advantage_confirmed"] = False
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "cpu_fixture_development_measurement"
        if allow_fixture
        else "actual_backend",
        "batch": _ref(batch_path),
        "study": _ref(study_path),
        "planned_cases": 10,
        "planned_policies": 5,
        "planned_rows": 50,
        "denominator": {
            "unit": "independent_source_lineage",
            "planned_cases": 10,
            "valid_result_cases": 10,
            "normal_cases": sum(
                row["qualification"].get("arm") == "NONE"
                for row in rows
                if row["policy"] == "logs_only"
            ),
            "independent_cases": 10 if independence["verified"] else None,
            "source_cp_independence_verified": independence["verified"],
            "duplicate_source_case_groups": independence["duplicate_case_groups"],
            "unique_case_checkpoints": independence["unique_case_checkpoints"],
            "reference_checkpoint_dependency_groups": independence[
                "reference_checkpoint_dependency_groups"
            ],
            "known_fault_types": True,
            "new_seeds_are_new_fault_types": False,
            "repeated_frames_or_calls_increase_denominator": False,
            "qualified_fault_cases": len(eligible_rows),
            "qualified_fault_types": qualified_types,
        },
        "policies": metrics,
        "primary": {
            **primary,
            "outcome": primary_outcome,
            "preregistered_contrast": ["logs_only", "fixed_eligible"],
        },
        "secondary": {
            **secondary,
            **visual,
            "preregistered_contrast": ["logs_plus_frames", "logs_only"],
        },
        "claims": {
            "general_superiority": False,
            "model_version_gain": False,
            "three_vs_three_point_five_version_gain": False,
            "probe_time_converted_to_gpu_savings": False,
            "video_advantage_confirmed": visual["video_advantage_confirmed"],
            "fixture_promoted_to_actual": False,
        },
        "batch_parent_wall_s": batch_parent_wall,
        "source_closure_count": len(source_files()),
        "source_files_lf": {
            name: _sha(ROOT / name, lf=True) for name in source_files()
        },
    }
    _write_new(out, result)
    return deepcopy(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-result", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args(argv)
    summarize(args.batch_result, args.study, args.out, allow_fixture=args.allow_fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
