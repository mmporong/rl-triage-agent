# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bench import run_lineage_five_policy_v1 as RUN
from evals import lineage_five_policy_summary_v1 as S

ARMS = ("NONE", "NONE", "F1", "F1", "F2", "F2", "COMPOUND", "COMPOUND", "F5", "F5")
STUDY = (
    S.ROOT / "bench/protocols/agentic_five_type_evaluation_s2550_s2559_20261101_v2.json"
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def make_batch(
    tmp_path: Path, *, fixed_all: bool = False, visual_gain: bool = False
) -> tuple[Path, dict[tuple[str, str], Path]]:
    rows = []
    paths: dict[tuple[str, str], Path] = {}
    index_path = tmp_path / "fixture-index.json"
    write(index_path, {"fixture": True})
    index_ref = S._ref(index_path)
    for index, fault in enumerate(ARMS, start=1):
        case_id = f"case-{index:02d}"
        checkpoint_bindings = {
            "case": {
                "checkpoint_sha256": hashlib.sha256(
                    f"case-{index}".encode()
                ).hexdigest(),
                "telemetry_sha256": hashlib.sha256(
                    f"case-t-{index}".encode()
                ).hexdigest(),
            },
            "reference": {
                "checkpoint_sha256": hashlib.sha256(
                    f"ref-{index}".encode()
                ).hexdigest(),
                "telemetry_sha256": hashlib.sha256(
                    f"ref-t-{index}".encode()
                ).hexdigest(),
            },
        }
        for policy in S.DRIVER.ARMS:
            eligible = fault not in {"NONE", "F5"}
            label = "eligible" if fault != "F5" else "source_failure_absent"
            qualification = {
                "arm": fault,
                "label": label,
                "diagnosis_eligible": eligible,
            }
            correct = eligible and (
                policy != "fixed_eligible" or fixed_all or index <= 7
            )
            if visual_gain and policy == "logs_only" and index == 8:
                correct = False
            if visual_gain and policy == "logs_plus_frames" and index == 8:
                correct = True
            probes = (
                2
                if fixed_all and policy == "fixed_eligible" and correct
                else (1 if correct else 0)
            )
            confirmed = (
                (
                    "reward"
                    if fault in {"F1", "COMPOUND"}
                    else "actuator"
                    if fault == "F2"
                    else None
                )
                if correct
                else None
            )
            steps = [
                {
                    "probe": "P_reward" if confirmed == "reward" else "P_torque",
                    "outcome": "abnormal",
                    "fact_ids": ["telemetry:x"],
                    "owned_child_wall_s": 1.0,
                }
                for _ in range(probes)
            ]
            citations = 0
            if visual_gain and policy == "logs_plus_frames" and index == 8:
                steps[-1]["fact_ids"] = ["vision:frame:f1"]
                citations = 1
            normal = fault == "NONE"
            exclusion = fault == "F5"
            hold_reason = (
                "normal_shadow_only_host_gate"
                if normal and policy in S.DRIVER.MODEL_ARMS
                else "normal_host_gate"
                if normal
                else "source_source_failure_absent"
                if exclusion
                else None
                if correct
                else "model_hold"
            )
            scoring = {
                "correct_any_family": correct,
                "wrong_confirmation": False,
                "compound_partial": fault == "COMPOUND" and correct,
                "full_cause_confirmation": correct and fault != "COMPOUND",
            }
            detail = {
                "schema": S.DRIVER.RESULT_SCHEMA,
                "arm": policy,
                "case_id": case_id,
                "source_digest": hashlib.sha256(case_id.encode()).hexdigest(),
                "source_index": index_ref,
                "checkpoint_bindings": checkpoint_bindings,
                "scope": "cpu_fixture",
                "qualification": qualification,
                "scoring": scoring,
                "steps": steps,
                "model_calls": 0,
                "native_normal_intervention_proposal": False,
                "hold_reason": hold_reason,
                "semantic_invalid": 0,
                "infra_unknown": 0,
                "provider_unknown": 0,
                "frame_fact_citations": citations,
                "confirmed_subtype": confirmed,
                "end_state": "confirmed" if correct else "HOLD",
                "parent_wall_s": float(probes) + 0.5,
            }
            path = tmp_path / "rows" / case_id / policy / "result.json"
            write(path, detail)
            paths[(case_id, policy)] = path
            rows.append(
                {
                    "case_id": case_id,
                    "arm": policy,
                    "result": S._ref(path),
                    "end_state": detail["end_state"],
                    "scoring": scoring,
                }
            )
    batch = {
        "schema": S.BATCH_SCHEMA,
        "status": "complete",
        "planned_cases": 10,
        "planned_rows": 50,
        "rows": rows,
        "parent_wall_s": 50.0,
    }
    batch_path = tmp_path / "batch-result.json"
    write(batch_path, batch)
    return batch_path, paths


def refresh(batch_path: Path, paths: dict[tuple[str, str], Path]) -> None:
    batch = S._read(batch_path)
    for row in batch["rows"]:
        path = paths[(row["case_id"], row["arm"])]
        detail = S._read(path)
        row["result"] = S._ref(path)
        row["end_state"] = detail["end_state"]
        row["scoring"] = detail["scoring"]
    batch_path.unlink()
    write(batch_path, batch)


def test_fixture_requires_explicit_flag_and_preserves_case_denominator(
    tmp_path: Path,
) -> None:
    batch, _ = make_batch(tmp_path)
    with pytest.raises(ValueError, match="allow_fixture"):
        S.summarize(batch, STUDY, tmp_path / "default.json")
    result = S.summarize(batch, STUDY, tmp_path / "summary.json", allow_fixture=True)
    assert result["scope"] == "cpu_fixture_development_measurement"
    assert result["denominator"] == {
        "unit": "independent_source_lineage",
        "planned_cases": 10,
        "valid_result_cases": 10,
        "normal_cases": 2,
        "independent_cases": 10,
        "source_cp_independence_verified": True,
        "duplicate_source_case_groups": [],
        "unique_case_checkpoints": 10,
        "reference_checkpoint_dependency_groups": [],
        "known_fault_types": True,
        "new_seeds_are_new_fault_types": False,
        "repeated_frames_or_calls_increase_denominator": False,
        "qualified_fault_cases": 6,
        "qualified_fault_types": 3,
    }
    logs = result["policies"]["logs_only"]
    assert logs["correct_any_family"] == 6
    assert logs["compound_partial"] == 2 and logs["compound_full"] == 0
    assert logs["source_exclusion"] == {"source_failure_absent": 2}
    assert logs["paired_probes"] == 6 and logs["model_calls"] == 0
    assert logs["provider_tokens"]["total_tokens"] == 0
    assert result["primary"]["outcome"] == "positive_in_registered_scope"
    assert result["secondary"]["outcome"] == "no_observed_gain"
    assert result["claims"]["general_superiority"] is False
    assert result["claims"]["probe_time_converted_to_gpu_savings"] is False


def test_missing_or_duplicate_case_policy_is_rejected(tmp_path: Path) -> None:
    batch, _ = make_batch(tmp_path)
    value = S._read(batch)
    value["rows"][-1] = value["rows"][0]
    batch.unlink()
    write(batch, value)
    with pytest.raises(ValueError, match="누락·중복"):
        S.summarize(batch, STUDY, tmp_path / "summary.json", allow_fixture=True)


def test_same_accuracy_lower_probe_requires_two_common_correct_cases(
    tmp_path: Path,
) -> None:
    batch, _ = make_batch(tmp_path, fixed_all=True)
    result = S.summarize(batch, STUDY, tmp_path / "summary.json", allow_fixture=True)
    primary = result["primary"]
    assert primary["left_correct"] == primary["right_correct"] == 6
    assert primary["common_correct_cases"] == 6
    assert primary["common_correct_left_probe_sum"] == 6
    assert primary["common_correct_right_probe_sum"] == 12
    assert primary["outcome"] == "same_accuracy_lower_probe_count"


def test_visual_gain_requires_actual_frame_fact_and_confirming_probe(
    tmp_path: Path,
) -> None:
    batch, paths = make_batch(tmp_path, visual_gain=True)
    positive = S.summarize(batch, STUDY, tmp_path / "positive.json", allow_fixture=True)
    assert positive["secondary"]["outcome"] == "visual_gain"
    assert positive["secondary"]["valid_frame_citation_and_confirming_probe_cases"] == [
        "case-08"
    ]
    assert positive["claims"]["video_advantage_confirmed"] is True

    frame_path = paths[("case-08", "logs_plus_frames")]
    detail = S._read(frame_path)
    detail["steps"][0]["fact_ids"] = ["telemetry:x"]
    detail["frame_fact_citations"] = 0
    frame_path.unlink()
    write(frame_path, detail)
    refresh(batch, paths)
    negative = S.summarize(batch, STUDY, tmp_path / "negative.json", allow_fixture=True)
    assert negative["secondary"]["accuracy_gain"] is True
    assert negative["secondary"]["outcome"] == "no_observed_gain"
    assert negative["secondary"]["frame_citation_count_alone_is_effect"] is False


def test_technical_unknown_makes_preregistered_contrast_inconclusive(
    tmp_path: Path,
) -> None:
    batch, paths = make_batch(tmp_path)
    path = paths[("case-03", "logs_only")]
    detail = S._read(path)
    detail["infra_unknown"] = 1
    detail["hold_reason"] = "model_infra_unknown"
    path.unlink()
    write(path, detail)
    refresh(batch, paths)
    result = S.summarize(batch, STUDY, tmp_path / "summary.json", allow_fixture=True)
    assert result["policies"]["logs_only"]["infra_unknown_rows"] == 1
    assert result["primary"]["technical_missing_pairs"] == 1
    assert result["primary"]["outcome"] == "inconclusive"


def test_source_digest_must_match_across_arms_and_case_cp_must_be_independent(
    tmp_path: Path,
) -> None:
    batch, paths = make_batch(tmp_path)
    first_bindings = S._read(paths[("case-01", "logs_only")])["checkpoint_bindings"]
    for policy in S.DRIVER.ARMS:
        path = paths[("case-02", policy)]
        detail = S._read(path)
        detail["checkpoint_bindings"]["case"]["checkpoint_sha256"] = first_bindings[
            "case"
        ]["checkpoint_sha256"]
        path.unlink()
        write(path, detail)
    refresh(batch, paths)
    result = S.summarize(
        batch, STUDY, tmp_path / "duplicate-source.json", allow_fixture=True
    )
    assert result["denominator"]["source_cp_independence_verified"] is False
    assert result["denominator"]["duplicate_source_case_groups"] == [
        ["case-01", "case-02"]
    ]
    assert result["primary"]["outcome"] == "inconclusive"
    assert result["secondary"]["outcome"] == "inconclusive"

    batch2, paths2 = make_batch(tmp_path / "arm-mismatch")
    path = paths2[("case-01", "logs_only")]
    detail = S._read(path)
    detail["source_digest"] = "f" * 64
    path.unlink()
    write(path, detail)
    refresh(batch2, paths2)
    with pytest.raises(ValueError, match="arm별 source_digest"):
        S.summarize(batch2, STUDY, tmp_path / "mismatch.json", allow_fixture=True)

    batch3, paths3 = make_batch(tmp_path / "self-reference")
    for policy in S.DRIVER.ARMS:
        path = paths3[("case-01", policy)]
        detail = S._read(path)
        detail["checkpoint_bindings"]["reference"]["checkpoint_sha256"] = detail[
            "checkpoint_bindings"
        ]["case"]["checkpoint_sha256"]
        path.unlink()
        write(path, detail)
    refresh(batch3, paths3)
    with pytest.raises(ValueError, match="자기 checkpoint"):
        S.summarize(batch3, STUDY, tmp_path / "self-reference.json", allow_fixture=True)


def test_normal_proposal_observation_denominator_mismatch_is_inconclusive(
    tmp_path: Path,
) -> None:
    batch, paths = make_batch(tmp_path)
    path = paths[("case-01", "logs_only")]
    detail = S._read(path)
    detail["native_normal_intervention_proposal"] = None
    path.unlink()
    write(path, detail)
    refresh(batch, paths)
    result = S.summarize(
        batch, STUDY, tmp_path / "normal-missing.json", allow_fixture=True
    )
    logs = result["policies"]["logs_only"]
    fixed = result["policies"]["fixed_eligible"]
    assert (logs["normal_model_observed"], logs["normal_model_unobserved"]) == (1, 1)
    assert (fixed["normal_model_observed"], fixed["normal_model_unobserved"]) == (2, 0)
    assert result["primary"]["outcome"] == "inconclusive"


def test_actual_scope_reconsumes_runner_preflight_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch, paths = make_batch(tmp_path)
    sources: dict[str, dict[str, object]] = {}
    for case_number in range(1, 11):
        case_id = f"case-{case_number:02d}"
        sample = S._read(paths[(case_id, "logs_only")])
        source = {
            "case_id": case_id,
            "index": sample["source_index"],
            "packet": {"input_bindings": sample["checkpoint_bindings"]},
        }
        sources[case_id] = {"source": source}
        digest = S.DRIVER.BASE.digest(source)
        for policy in S.DRIVER.ARMS:
            path = paths[(case_id, policy)]
            detail = S._read(path)
            detail["scope"] = "actual_backend"
            detail["source_digest"] = digest
            path.unlink()
            write(path, detail)
    refresh(batch, paths)
    protocol = tmp_path / "execution-protocol.json"
    config = tmp_path / "execution-config.json"
    write(protocol, {"registered": True})
    write(config, {"private": True})
    batch_value = S._read(batch)
    batch_value.update(
        execution_protocol=S._ref(protocol),
        execution_anchor="a" * 40,
        execution_config=S._ref(config),
    )
    batch.unlink()
    write(batch, batch_value)
    seen: list[tuple[Path, str, Path]] = []

    def preflight(protocol_path: Path, anchor: str, config_path: Path):
        seen.append((protocol_path, anchor, config_path))
        return {}, {}, sources

    monkeypatch.setattr(S.RUNNER, "preflight", preflight)
    result = S.summarize(batch, STUDY, tmp_path / "actual-summary.json")
    assert result["scope"] == "actual_backend"
    assert seen == [(protocol.resolve(), "a" * 40, config.resolve())]


def test_runner_written_missing_input_batch_reaches_write_once_inconclusive_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "execution-protocol.json"
    config_path = tmp_path / "execution-config.json"
    index_path = tmp_path / "source-index.json"
    write(protocol_path, {"registered": True})
    write(config_path, {"private": True})
    write(index_path, {"registered": True})
    cases = [f"case-{number:02d}" for number in range(1, 11)]
    protocol = {
        "execution_config_sha256": S._sha(config_path),
        "sources_lf": {},
        "wall_cap_s": 144000,
        "paired_cap": 150,
        "case_order": cases,
        "arm_order": {
            case: list(RUN.DRIVER.ARMS[index % 5 :] + RUN.DRIVER.ARMS[: index % 5])
            for index, case in enumerate(cases)
        },
        "source_digests": {
            case: hashlib.sha256(case.encode()).hexdigest() for case in cases
        },
        "tool_protocols": {
            "paired_probe": {"path": "unused"},
            "view": {"path": "unused"},
        },
    }
    config = {"lab": str(tmp_path), "source_index": S._ref(index_path)}
    gate = {
        "schema": RUN.DRIVER.GUARD.SCHEMA,
        "decision": "HOLD",
        "stage": "input",
        "reason": "registered_input_unavailable",
        "approval_granted": False,
        "dispatch_granted": False,
        "automatic_retry": False,
        "evidence": {
            "fault": {
                "schema": RUN.DRIVER.GUARD.FAULT_SCHEMA,
                "stage": "input",
                "disposition": "HOLD",
                "reason": "registered_input_unavailable",
                "error_type": "FileNotFoundError",
            }
        },
    }
    sources = {
        case: {
            "source": None,
            "inputs": {},
            "input_hold": gate,
            "registered_input_roots": (),
        }
        for case in cases
    }

    def preflight(*_args):
        return protocol, config, sources

    monkeypatch.setattr(RUN, "preflight", preflight)
    monkeypatch.setattr(S.RUNNER, "preflight", preflight)
    monkeypatch.setattr(
        RUN.LEASE,
        "lease_context",
        lambda: {"owner": RUN.LEASE.OWNER, "leaseId": "fixture-missing-input"},
    )
    monkeypatch.setattr(RUN.NIM, "initialize", lambda *_args, **kwargs: kwargs)
    monkeypatch.setattr(
        RUN.NIM,
        "status",
        lambda *_: {"fixture": True, "reservations_charged": 0},
    )
    batch = RUN.execute(
        protocol_path, "a" * 40, config_path, tmp_path / "runner-output"
    )
    assert batch["input_unavailable_rows"] == 50
    assert batch["comparison_status"] == "inconclusive_input_unavailable"

    out = tmp_path / "summary.json"
    result = S.summarize(tmp_path / "runner-output/result.json", STUDY, out)
    assert result["scope"] == "actual_backend_inconclusive_input_unavailable"
    assert result["input_unavailable"]["rows"] == 50
    assert result["input_unavailable"]["observed_rows"] == []
    assert result["denominator"]["qualified_fault_cases"] is None
    assert result["primary"]["accuracy"] is None
    assert result["primary"]["inspection_cost"] is None
    assert result["primary"]["outcome"] == "inconclusive"
    assert result["secondary"]["outcome"] == "inconclusive"
    assert all(value is None for value in result["policies"].values())
    assert len(result["input_unavailable"]["missing_rows"]) == 50
    with pytest.raises(ValueError, match="fresh"):
        S.summarize(tmp_path / "runner-output/result.json", STUDY, out)

    batch_path = tmp_path / "runner-output/result.json"
    batch_value = S._read(batch_path)
    first = batch_value["rows"][0]
    detail_path = Path(first["result"]["path"])
    original_detail = S._read(detail_path)

    changed = json.loads(json.dumps(original_detail))
    changed["source_digest"] = "f" * 64
    detail_path.unlink()
    write(detail_path, changed)
    first["result"] = S._ref(detail_path)
    batch_path.unlink()
    write(batch_path, batch_value)
    with pytest.raises(ValueError, match="row/result/source"):
        S.summarize(batch_path, STUDY, tmp_path / "source-mismatch.json")

    detail_path.unlink()
    write(detail_path, original_detail)
    first["result"] = S._ref(detail_path)
    second = batch_value["rows"][1]
    second_arm = second["arm"]
    second["arm"] = first["arm"]
    batch_path.unlink()
    write(batch_path, batch_value)
    with pytest.raises(ValueError, match="누락·중복"):
        S.summarize(batch_path, STUDY, tmp_path / "duplicate-case.json")

    second["arm"] = second_arm
    malformed = json.loads(json.dumps(original_detail))
    malformed["input_guard"]["approval_granted"] = True
    detail_path.unlink()
    write(detail_path, malformed)
    first["result"] = S._ref(detail_path)
    batch_path.unlink()
    write(batch_path, batch_value)
    with pytest.raises(ValueError, match="HOLD 계약"):
        S.summarize(batch_path, STUDY, tmp_path / "malformed-hold.json")


def test_provider_tokens_and_wall_use_verified_call_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch, paths = make_batch(tmp_path)
    path = paths[("case-03", "logs_only")]
    detail = S._read(path)
    detail["model_calls"] = 1
    path.unlink()
    write(path, detail)
    call = tmp_path / "nim/calls/0001"
    ref = {"path": str(call.resolve()), "sha256": "a" * 64}
    write(path.parent / "step01/model-result.json", {"call_ref": ref})
    monkeypatch.setattr(
        S.NIM,
        "verify_call",
        lambda _path: {
            "call_ref": ref,
            "wall_s": 0.25,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    )
    refresh(batch, paths)
    result = S.summarize(batch, STUDY, tmp_path / "summary.json", allow_fixture=True)
    logs = result["policies"]["logs_only"]
    assert logs["model_calls"] == 1
    assert logs["provider_tokens"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert logs["provider_wall_s"] == 0.25
    assert logs["owned_child_wall_s"] == 6.0
    assert logs["parent_wall_s"] == 11.0


def test_summary_output_is_write_once(tmp_path: Path) -> None:
    batch, _ = make_batch(tmp_path)
    out = tmp_path / "summary.json"
    S.summarize(batch, STUDY, out, allow_fixture=True)
    with pytest.raises(ValueError, match="fresh"):
        S.summarize(batch, STUDY, out, allow_fixture=True)
