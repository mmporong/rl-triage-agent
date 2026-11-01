# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import export_five_policy_recovery_v1 as E


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return E.FIRST.ref(path)


def make_fixture(tmp_path: Path, *, not_recovered: bool = False,
                 with_second: bool = False) -> tuple[Path, Path | None, dict]:
    protocol = save(tmp_path / "first-protocol.json", {})
    design = save(tmp_path / "conditional-design.json", {"pipeline_denominator": 4})
    config = save(tmp_path / "config.json", {})
    batch = save(tmp_path / "batch.json", {})
    seeds = {f"{case}:{policy}": 7000 + index
             for index, (case, policy) in enumerate(E.TARGETS)}
    candidates = {}
    records = []
    for index, (case, policy) in enumerate(E.TARGETS):
        status = "validated_candidate" if policy == "fixed_eligible" else "HOLD"
        candidate = {
            "schema": E.FIRST.SCHEMA, "case_id": case, "policy": policy,
            "source_arm": "F1" if case == "case-03" else "COMPOUND",
            "source_seed": 2552 if case == "case-03" else 2556,
            "source_checkpoint_sha256": ("a" if case == "case-03" else "b") * 64,
            "status": status, "correction": "restore_body_frame_reward"
            if status != "HOLD" else None, "hold_reason": "model_hold"
            if status == "HOLD" else None,
        }
        candidate_path = save(tmp_path / f"candidate-{index}.json", candidate)
        candidates[str(candidate_path.resolve())] = candidate
        if status == "HOLD":
            records.append({"case_id": case, "policy": policy, "candidate": ref(candidate_path),
                            "registered_seed": seeds[f"{case}:{policy}"],
                            "outcome": "HOLD", "result": None, "epoch2_input": None})
            continue
        outcome = "not_recovered" if not_recovered and index == 0 else "recovered"
        evidence = save(tmp_path / f"evidence-{index}.json", {
            "fresh_initialization": True, "checkpoint_loaded": False,
            "resume_requested": False,
            "source_checkpoint_sha256": candidate["source_checkpoint_sha256"],
        })
        approval = save(tmp_path / f"approval-{index}.json", {
            "schema": E.FIRST.APPROVAL_SCHEMA, "decision": "approve",
            "approver": "codex-main-user-delegated", "candidate": ref(candidate_path),
            "protocol": ref(protocol), "anchor": "a" * 40,
            "registered_seed": seeds[f"{case}:{policy}"], "lease_id": "lease-first",
        })
        receipt = {
            "schema": E.FIRST.RECEIPT_SCHEMA, "status": outcome,
            "candidate": ref(candidate_path), "approval": ref(approval),
            "protocol": ref(protocol), "registered_seed": seeds[f"{case}:{policy}"],
            "lease_id": "lease-first", "training_mode": "fresh_restart",
            "source_checkpoint_loaded": False,
            "source_checkpoint_sha256": candidate["source_checkpoint_sha256"],
            "training": {"evidence": ref(evidence),
                         "checkpoint": {"sha256": str(index + 1) * 64}},
            "fixed26": {"conditions": 26, "execution": {}},
        }
        receipt_path = save(tmp_path / f"receipt-{index}.json", receipt)
        result = save(tmp_path / f"result-{index}.json", {
            "schema": E.FIRST.RESULT_SCHEMA, "status": "complete",
            "receipt": ref(receipt_path), "outcome": outcome,
        })
        epoch_ref = None
        if outcome == "not_recovered":
            epoch = save(tmp_path / "epoch2.json", {
                "schema": E.FIRST.EPOCH2_SCHEMA, "prior_receipt": ref(receipt_path),
                "new_protocol_required": True, "execution_allowed": False,
            })
            epoch_ref = ref(epoch)
        records.append({"case_id": case, "policy": policy, "candidate": ref(candidate_path),
                        "registered_seed": seeds[f"{case}:{policy}"], "outcome": outcome,
                        "result": ref(result), "epoch2_input": epoch_ref})
    plan = save(tmp_path / "first-plan.json", {
        "schema": E.PARENT.FIRST_PLAN_SCHEMA, "automatic_epoch2": False,
        "recovery_protocol": ref(protocol), "config": ref(config),
        "batch_result": ref(batch), "anchor": "a" * 40,
    })
    first = save(tmp_path / "first-result.json", {
        "schema": E.PARENT.FIRST_RESULT_SCHEMA, "status": "complete",
        "automatic_epoch2": False, "plan": ref(plan), "records": records,
        "parent_wall_s": 12.5,
    })
    second_path = None
    second_protocol = None
    second_seeds = {key: seed + 1000 for key, seed in seeds.items()}
    sources = {}
    second_candidates = {}
    if with_second:
        key = "case-03:fixed_eligible"
        source = {"case_id": "case-03", "policy": "fixed_eligible", "fixture": True}
        source_path = save(tmp_path / "second-source.json", source)
        sources[str(source_path.resolve())] = source
        second_protocol = save(tmp_path / "second-protocol.json", {})
        diagnosis = save(tmp_path / "diagnosis.json", {"steps": [], "model_calls": 0})
        second_candidate = {
            "case_id": "case-03", "policy": "fixed_eligible", "status": "HOLD",
            "source_checkpoint_sha256": "1" * 64, "second_correction": None,
            "hold_reason": "model_hold",
        }
        second_candidate_path = save(tmp_path / "second-candidate.json", second_candidate)
        second_candidates[str(second_candidate_path.resolve())] = second_candidate
        second_records = []
        for case, policy in E.TARGETS:
            if (case, policy) == ("case-03", "fixed_eligible"):
                second_records.append({"case_id": case, "policy": policy,
                    "status": "registered", "source": ref(source_path),
                    "registered_seed": second_seeds[key], "diagnosis": ref(diagnosis),
                    "candidate": ref(second_candidate_path), "outcome": "HOLD", "result": None})
            else:
                second_records.append({"case_id": case, "policy": policy,
                    "status": "not_in_second_subset", "source": None,
                    "outcome": "not_in_second_subset"})
        second_plan = save(tmp_path / "second-plan.json", {
            "schema": E.PARENT.SECOND_PLAN_SCHEMA, "first_result": ref(first),
            "loop_protocol": ref(second_protocol), "anchor": "a" * 40,
        })
        second_path = save(tmp_path / "second-result.json", {
            "schema": E.PARENT.SECOND_RESULT_SCHEMA, "status": "complete",
            "epoch3_allowed": False, "plan": ref(second_plan), "records": second_records,
            "parent_wall_s": 3.5,
        })
    verifiers = {
        "first_candidate": lambda path: candidates[str(path.resolve())],
        "first_protocol": lambda *_args: {"recovery_seeds": seeds},
        "second_candidate": lambda path: second_candidates[str(path.resolve())],
        "second_protocol": lambda *_args: {
            "second_seeds": second_seeds,
            "source_digests": {"case-03:fixed_eligible": E.SECOND.digest(
                sources[str((tmp_path / "second-source.json").resolve())])},
        },
        "second_source": lambda value: value,
        "verify_call": lambda _path: {},
        "conditional_design": lambda _protocol, second_seeds=None: (
            design, {"pipeline_denominator": 4, "second_seeds": second_seeds}),
    }
    return first, second_path, verifiers


def test_first_terminal_results_export_all_four_pipelines(tmp_path: Path) -> None:
    first, _second, verifiers = make_fixture(tmp_path)
    out = tmp_path / "public"
    E.export(first, out, lab=tmp_path, allow_fixture=True, verifiers=verifiers)
    summary = E._read(out / "summary.json")
    cases = E._read(out / "cases.json")
    assert summary["pipeline_denominator"] == 4
    assert summary["independent_source_lineages"] == 2
    assert summary["final_outcomes"] == {"HOLD": 2, "not_recovered": 0, "recovered": 2}
    assert len(cases["rows"]) == 4
    assert all(row["source_checkpoint_loaded"] is False
               for row in (item["first_training"] for item in cases["rows"]
                            if item["first_training"] is not None))
    text = "\n".join(path.read_text(encoding="utf-8") for path in out.glob("*.json"))
    assert str(tmp_path) not in text and "bench/private" not in text


def test_not_recovered_requires_complete_second_result(tmp_path: Path) -> None:
    first, _second, verifiers = make_fixture(tmp_path, not_recovered=True)
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="second 결과"):
        E.export(first, out, lab=tmp_path, allow_fixture=True, verifiers=verifiers)
    assert not out.exists()


def test_second_hold_is_preserved_without_success_credit(tmp_path: Path) -> None:
    first, second, verifiers = make_fixture(
        tmp_path, not_recovered=True, with_second=True)
    out = tmp_path / "public"
    E.export(first, out, lab=tmp_path, second_result=second,
             allow_fixture=True, verifiers=verifiers)
    summary = E._read(out / "summary.json")
    cases = E._read(out / "cases.json")["rows"]
    target = next(row for row in cases
                  if row["case_id"] == "case-03" and row["policy"] == "fixed_eligible")
    assert target["first_outcome"] == "not_recovered"
    assert target["second_outcome"] == target["final_outcome"] == "HOLD"
    assert target["second_diagnosis_cost"]["model_api_calls"] == 0
    assert target["original_source_checkpoint_sha256"] == "a" * 64
    assert target["first_new_checkpoint_sha256"] == "1" * 64
    assert summary["second_required_pipelines"] == 1
    assert summary["final_outcomes"] == {"HOLD": 3, "not_recovered": 0, "recovered": 1}


def test_source_checkpoint_mismatch_and_overwrite_are_rejected(tmp_path: Path) -> None:
    first, _second, verifiers = make_fixture(tmp_path)
    candidates = {}
    for path in tmp_path.glob("candidate-*.json"):
        candidates[str(path.resolve())] = E._read(path)
    candidates[str((tmp_path / "candidate-1.json").resolve())]["source_checkpoint_sha256"] = "f" * 64
    changed = dict(verifiers)
    changed["first_candidate"] = lambda path: candidates[str(path.resolve())]
    with pytest.raises(ValueError, match="같은 case"):
        E.export(first, tmp_path / "bad", lab=tmp_path,
                 allow_fixture=True, verifiers=changed)

    out = tmp_path / "public"
    E.export(first, out, lab=tmp_path, allow_fixture=True, verifiers=verifiers)
    with pytest.raises(ValueError, match="fresh"):
        E.export(first, out, lab=tmp_path, allow_fixture=True, verifiers=verifiers)


def test_model_hold_counts_attempt_without_completed_probe_step(tmp_path: Path) -> None:
    diagnosis = save(tmp_path / "diagnosis.json", {"steps": [], "model_calls": 1})
    call_dir = tmp_path / "call-001"
    call_dir.mkdir()
    call_ref = {"path": str(call_dir.resolve()), "sha256": "c" * 64}
    save(tmp_path / "step01/model-result.json", {"call_ref": call_ref, "status": "HOLD"})

    cost = E._model_cost(diagnosis, call_verifier=lambda path: {
        "call_ref": call_ref, "wall_s": 0.25,
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    })

    assert cost["model_attempts"] == 1
    assert cost["model_api_calls"] == 1
    assert cost["paired_probe_sdk_executions"] == 0
    assert cost["provider_tokens"]["total_tokens"] == 14


def test_model_transport_hold_preserves_attempt_with_missing_receipt(tmp_path: Path) -> None:
    diagnosis = save(tmp_path / "diagnosis.json", {"steps": [], "model_calls": 1})
    save(tmp_path / "step01/model-result.json", {
        "schema": "local_pair_input_runtime_fault_v1",
        "decision": "HOLD", "reason": "typed_transport_error"})

    cost = E._model_cost(
        diagnosis, call_verifier=lambda _path: pytest.fail("receipt verifier called"))

    assert cost["model_attempts"] == 1
    assert cost["model_api_calls"] == 0
    assert cost["provider_receipt_missing_attempts"] == 1
    assert cost["provider_tokens"] is None
    assert cost["provider_wall_s"] is None


@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_second_records_must_be_exact_unique_four(tmp_path: Path, mode: str) -> None:
    first, second, verifiers = make_fixture(
        tmp_path, not_recovered=True, with_second=True)
    assert second is not None
    value = E._read(second)
    if mode == "duplicate":
        value["records"][3] = dict(value["records"][0])
    else:
        value["records"].pop()
    save(second, value)
    # second plan은 first 결과 ref만 결합하므로 second 원문 갱신 뒤에도 유효하다.
    with pytest.raises(ValueError, match="second exact4"):
        E.export(first, tmp_path / f"bad-{mode}", lab=tmp_path,
                 second_result=second, allow_fixture=True, verifiers=verifiers)


def test_production_rejects_injected_verifiers_before_writing(tmp_path: Path) -> None:
    first, _second, verifiers = make_fixture(tmp_path)
    out = E.ROOT / "evals/results/ri391-production-injection-fixture"
    assert not out.exists()
    with pytest.raises(ValueError, match="verifier injection"):
        E.export(first, out, lab=tmp_path, verifiers=verifiers)
    assert not out.exists()
