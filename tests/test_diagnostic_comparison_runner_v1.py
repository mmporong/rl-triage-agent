# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_triage import investigate, mechanism_confirmation
from rl_triage.investigation_session import reference, write_new

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import model_decision_journal_v1 as MODEL_JOURNAL
import run_diagnostic_comparison_v1 as RUNNER


def _registry() -> dict:
    return {
        "mechanisms": ["m1", "m2"],
        "outcomes": ["abnormal", "normal", "unknown"],
        "probes": {
            "p1": {
                "measures": "first",
                "cost_gpu_s": 0,
                "args": {},
                "expect": {"m1": "abnormal", "m2": "normal"},
            },
            "p2": {
                "measures": "second",
                "cost_gpu_s": 0,
                "args": {},
                "expect": {"m1": "normal", "m2": "abnormal"},
            },
        },
        "fixed_order": ["p1", "p2"],
    }


def _fixture_source(path: Path, case_id: str) -> None:
    write_new(
        path,
        {
            "schema": "fixture_source_v1",
            "case_id": case_id,
            "checkpoint_sha256": "a" * 64,
            "facts": {"visible_metric": 1},
            "binding": {
                "reference_sha256": "b" * 64,
                "fixture_outcomes": {"p1": "abnormal", "p2": "normal"},
            },
        },
    )


def _setup(tmp_path: Path, *, probe_scope: list[str] | None = None) -> tuple[dict, dict, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    code = repo / "runner-code.py"
    code.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    protocol = repo / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    plan_path = repo / "plan.json"
    output = tmp_path / "run"
    case_id = "case-01"
    plan = {
        "schema": RUNNER.SCHEMA,
        "source_anchor": "1" * 40,
        "source_closure": {"runner-code.py": RUNNER._lf_sha(code)},
        "methods": list(RUNNER.METHODS),
        "cases": [{"case_id": case_id, "rotation": list(RUNNER.METHODS)}],
        "limits": {"max_checks": 3, "api_cap": 40, "watchdog_s": 660},
        "inputs": [{"role": "protocol", "path": "protocol.json", "sha256": RUNNER._lf_sha(protocol)}],
        "backend": RUNNER.ACTUAL_BACKEND,
        "authorization": {"actor": "fixture_actor", "probe_scope": probe_scope or ["p1", "p2"]},
    }
    plan_path.write_bytes(RUNNER.canonical(plan) + b"\n")
    plan_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    shared_api = str((output / "shared-api").resolve())
    arms = []
    for method in RUNNER.METHODS:
        source = output / "bound" / case_id / method / "source.json"
        source.parent.mkdir(parents=True)
        _fixture_source(source, case_id)
        session = output / "bound" / case_id / method / "session"
        arms.append(
            {
                "method": method,
                "source_path": str(source.resolve()),
                "session_path": str(session.resolve()),
                "prepare_payload": {
                    "source_path": str(source.resolve()),
                    "registry": _registry(),
                    "model_binding": (
                        {
                            "source_anchor": "1" * 40,
                            "protocol_sha256": "2" * 64,
                            "params": {"temperature": 0},
                            "api_cap": 40,
                            "api_root": shared_api,
                        }
                        if method in {"one_shot_llm", "iterative_agent"}
                        else None
                    ),
                    "method": method,
                    "max_checks": 3,
                    "gpu_budget_per_check_s": 0,
                    "api_root": shared_api if method in {"one_shot_llm", "iterative_agent"} else None,
                },
            }
        )
    binding = {
        "schema": RUNNER.JOB_SCHEMA,
        "plan": plan_ref,
        "api_root": shared_api,
        "cases": [{"case_id": case_id, "arms": arms}],
    }
    return plan_ref, binding, repo, output


def test_cpu_fixture_runs_all_four_arms_in_preregistered_rotation(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    prepared: list[str] = []

    def invoke(command, session, payload, **kwargs):
        if command == "prepare":
            prepared.append(payload["method"])
        return investigate.run(command, session, payload, **kwargs)

    report = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=invoke,
    )

    assert report["status"] == "completed"
    assert prepared == list(RUNNER.METHODS)
    assert [row["method"] for row in report["arms"]] == list(RUNNER.METHODS)
    assert len(report["arms"]) == 4
    assert all("timing" in row for row in report["arms"])
    assert all(row["status"] not in {"not_started", "integrity_failure"} for row in report["arms"])
    approvals = sorted(output.glob("case-01/*/runner_approvals/*.json"))
    assert approvals
    assert all(json.loads(path.read_text(encoding="utf-8"))["approved"] is True for path in approvals)
    fixture_cost = report["arms"][0]["resource_cost"]
    assert len(fixture_cost["events"]) == 1
    assert fixture_cost["events"][0]["costs"]["gpu_wall_s"] == 0.0
    assert fixture_cost["events"][0]["costs"]["model_calls"] == 0
    assert fixture_cost["model_api_summary"]["reserved_attempts"] == 0
    request_cost = fixture_cost["request_costs"][0]
    assert request_cost["actual_owned"]["owned_sdk_execution_count"] == 0
    assert request_cost["effective_cost_source"] == "host_receipt"
    assert request_cost["double_count_prevented"] is True
    assert fixture_cost["actual_owned_summary"]["owned_sdk_execution_count"] == 0


def test_scope_rejection_writes_no_backend_approval_or_dispatch(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path, probe_scope=["p2"])
    commands: list[str] = []

    def invoke(command, session, payload, **kwargs):
        commands.append(command)
        return investigate.run(command, session, payload, **kwargs)

    report = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=invoke,
    )

    assert report["arms"][0]["status"] == "rejected_probe_scope"
    assert "approve" not in commands
    assert "dispatch" not in commands
    assert not list(output.glob("case-01/*/runner_approvals/*.json"))


def test_transport_failure_is_costed_and_never_redispatched_on_new_process_resume(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    dispatch_calls = 0

    def uncertain_transport(command, session, payload, **kwargs):
        nonlocal dispatch_calls
        if command == "dispatch" and dispatch_calls == 0:
            dispatch_calls += 1
            raise ConnectionError("transport outcome unknown")
        if command == "dispatch":
            dispatch_calls += 1
        return investigate.run(command, session, payload, **kwargs)

    first = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=uncertain_transport,
    )
    assert first["arms"][0]["status"] == "dispatch_transport_unknown"
    timing_path = output / "case-01/00-fixed_sequence/controller-timing.jsonl"
    events = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines()]
    assert any(row["step"] == "dispatch" and row["status"] == "error" for row in events)
    assert first["arms"][0]["controller_timing"]["sha256"] == RUNNER.file_sha(timing_path)

    second = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=uncertain_transport,
    )
    assert second["arms"][0]["status"] == "hold_transport_unknown"
    assert second["arms"][0]["reexecute_allowed"] is False
    assert dispatch_calls == 1 + sum(
        row["method"] != "fixed_sequence" for row in first["arms"] if row["status"] not in {"model_hold"}
    )


def test_plan_commit_ancestry_and_blobs_are_checked_for_actual_preflight(tmp_path: Path) -> None:
    repo = tmp_path / "scratch"
    repo.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()

    git("init")
    git("config", "user.email", "runner@example.invalid")
    git("config", "user.name", "Runner Test")
    code = repo / "runner-code.py"
    code.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    git("add", "runner-code.py")
    git("commit", "-m", "code")
    code_anchor = git("rev-parse", "HEAD")

    protocol = repo / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8", newline="\n")
    plan = {
        "schema": RUNNER.SCHEMA,
        "source_anchor": code_anchor,
        "source_closure": {"runner-code.py": RUNNER._lf_sha(code)},
        "methods": list(RUNNER.METHODS),
        "cases": [{"case_id": "case-01", "rotation": list(RUNNER.METHODS)}],
        "limits": {"max_checks": 3, "api_cap": 40, "watchdog_s": 660},
        "inputs": [{"role": "protocol", "path": "protocol.json", "sha256": RUNNER._lf_sha(protocol)}],
        "backend": RUNNER.ACTUAL_BACKEND,
        "authorization": {"actor": "registered-host", "probe_scope": ["p1"]},
    }
    plan_path = repo / "plan.json"
    plan_path.write_bytes(RUNNER.canonical(plan) + b"\n")
    plan_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    git("add", "plan.json", "protocol.json")
    git("commit", "-m", "registration")
    registration_anchor = git("rev-parse", "HEAD")

    loaded = RUNNER.load_plan(
        plan_ref,
        repo_root=repo,
        registration_anchor=registration_anchor,
        require_committed=True,
    )
    assert loaded["source_anchor"] == code_anchor

    protocol.write_text('{"changed":true}\n', encoding="utf-8", newline="\n")
    with pytest.raises(RUNNER.ComparisonContractError, match="plan input changed"):
        RUNNER.load_plan(
            plan_ref,
            repo_root=repo,
            registration_anchor=registration_anchor,
            require_committed=True,
        )
    protocol.write_text("{}\n", encoding="utf-8", newline="\n")
    code.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    with pytest.raises(RUNNER.ComparisonContractError, match="source closure current"):
        RUNNER.load_plan(
            plan_ref,
            repo_root=repo,
            registration_anchor=registration_anchor,
            require_committed=True,
        )


def test_ambiguous_approval_transport_is_not_retransmitted(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    approvals = 0

    def uncertain_approval(command, session, payload, **kwargs):
        nonlocal approvals
        if command == "approve" and approvals == 0:
            approvals += 1
            raise ConnectionError("approval response lost")
        if command == "approve":
            approvals += 1
        return investigate.run(command, session, payload, **kwargs)

    first = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=uncertain_approval,
    )
    assert first["arms"][0]["status"] == "approval_transport_failure"

    second = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=uncertain_approval,
    )
    assert second["arms"][0]["status"] == "hold_approval_unknown"
    assert second["arms"][0]["reexecute_allowed"] is False
    assert approvals == 1 + sum(
        row["method"] != "fixed_sequence" for row in first["arms"] if row["status"] != "model_hold"
    )


def test_invalid_rotation_or_stage_alias_stops_before_backend(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    plan_path = repo / plan_ref["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["cases"][0]["rotation"] = [RUNNER.METHODS[0]] * 4
    plan_path.write_bytes(RUNNER.canonical(plan) + b"\n")
    bad_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    binding["plan"] = bad_ref
    with pytest.raises(RUNNER.ComparisonContractError, match="permutation"):
        RUNNER.run_comparison(
            bad_ref, binding, repo_root=repo, output_root=output, fixture=True
        )

    plan_ref, binding, repo, output = _setup(tmp_path / "second")
    binding["cases"][0]["arms"][1]["session_path"] = binding["cases"][0]["arms"][0]["session_path"]
    with pytest.raises(RUNNER.ComparisonContractError, match="stage layout|isolated"):
        RUNNER.run_comparison(
            plan_ref, binding, repo_root=repo, output_root=output, fixture=True
        )


def test_plan_rejects_self_reference_absolute_input_and_short_watchdog(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    plan_path = repo / plan_ref["path"]
    original = json.loads(plan_path.read_text(encoding="utf-8"))

    self_bound = json.loads(json.dumps(original))
    self_bound["source_closure"]["protocol.json"] = RUNNER._lf_sha(repo / "protocol.json")
    plan_path.write_bytes(RUNNER.canonical(self_bound) + b"\n")
    self_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    binding["plan"] = self_ref
    with pytest.raises(RUNNER.ComparisonContractError, match="self-bind"):
        RUNNER.run_comparison(
            self_ref, binding, repo_root=repo, output_root=output, fixture=True
        )

    absolute = json.loads(json.dumps(original))
    absolute["inputs"][0]["path"] = str((repo / "protocol.json").resolve())
    plan_path.write_bytes(RUNNER.canonical(absolute) + b"\n")
    absolute_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    binding["plan"] = absolute_ref
    with pytest.raises(RUNNER.ComparisonContractError, match="relative path|escapes repo"):
        RUNNER.run_comparison(
            absolute_ref, binding, repo_root=repo, output_root=output, fixture=True
        )

    short = json.loads(json.dumps(original))
    short["limits"]["watchdog_s"] = RUNNER.MIN_MODEL_WATCHDOG_S - 1
    plan_path.write_bytes(RUNNER.canonical(short) + b"\n")
    short_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    binding["plan"] = short_ref
    with pytest.raises(RUNNER.ComparisonContractError, match="watchdog"):
        RUNNER.run_comparison(
            short_ref, binding, repo_root=repo, output_root=output, fixture=True
        )


def test_binding_rejects_different_api_root_and_case_protocol(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    binding["cases"][0]["arms"][2]["prepare_payload"]["api_root"] = str(
        (output / "other-api").resolve()
    )
    with pytest.raises(RUNNER.ComparisonContractError, match="API root"):
        RUNNER.run_comparison(
            plan_ref, binding, repo_root=repo, output_root=output, fixture=True
        )

    plan_ref, binding, repo, output = _setup(tmp_path / "protocol")
    other = repo / "other-protocol.json"
    other.write_text("{}\n", encoding="utf-8")
    binding["cases"][0]["arms"][0]["prepare_payload"]["protocol"] = reference(other)
    with pytest.raises(RUNNER.ComparisonContractError, match="case registration"):
        RUNNER.run_comparison(
            plan_ref, binding, repo_root=repo, output_root=output, fixture=True
        )


def test_controller_deadline_is_consumed_before_state_transition(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)

    class Clock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            self.value += 400.0
            return self.value

    clock = Clock()
    report = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        clock=clock,
    )
    assert report["arms"][0]["status"] == "hold_watchdog"


def test_actual_child_watchdog_cannot_exceed_plan(tmp_path: Path) -> None:
    config = tmp_path / "execution.json"
    config.write_text('{"watchdog_s":700}\n', encoding="utf-8")
    protocol = {
        "components": {
            "p1": {
                "execution_class": "actual",
                "artifacts": {"execution": {"artifact_id": "execution-1"}},
            }
        }
    }
    payload = {"artifact_resolver": {"execution-1": reference(config)}}
    with pytest.raises(RUNNER.ComparisonContractError, match="exceeds plan"):
        RUNNER._validate_child_watchdogs(protocol, payload, 660)

    config.write_text('{"watchdog_s":600}\n', encoding="utf-8")
    payload["artifact_resolver"]["execution-1"] = reference(config)
    RUNNER._validate_child_watchdogs(protocol, payload, 660)


def test_each_case_requires_a_registered_protocol_role(tmp_path: Path) -> None:
    plan_ref, _binding, repo, _output = _setup(tmp_path)
    plan = json.loads((repo / plan_ref["path"]).read_text(encoding="utf-8"))
    plan["cases"].append({"case_id": "case-02", "rotation": list(reversed(RUNNER.METHODS))})
    plan["inputs"][0]["role"] = "protocol:case-01"
    with pytest.raises(RUNNER.ComparisonContractError, match="case-02"):
        RUNNER._validate_plan(plan)

    second_protocol = repo / "protocol-02.json"
    second_protocol.write_text("{}\n", encoding="utf-8")
    plan["inputs"].append(
        {"role": "protocol:case-02", "path": "protocol-02.json",
         "sha256": RUNNER._lf_sha(second_protocol)}
    )
    RUNNER._validate_plan(plan)
    assert RUNNER._protocol_input(plan, "case-02")["path"] == "protocol-02.json"


def test_sanitized_cpu_fixture_matches_actual_failure_cost_shape(tmp_path: Path) -> None:
    """Public regression uses sanitized local files; private evidence is validated separately."""
    session = _synthetic_actual_session(
        tmp_path, execution=True, settled=False, wall_s=61.125, failure_artifacts=2
    )
    raw = RUNNER._raw_refs(session, 1, case_id="case-03", method="fixed_sequence")
    cost = raw["resource_cost"]
    assert len(cost["events"]) == 1
    event = cost["events"][0]
    assert event["status"] == "failed"
    assert event["costs"]["gpu_wall_s"] == 61.125
    assert event["costs"]["execution_wall_s"] == 61.125
    request = cost["request_costs"][0]
    assert request["request_id"] == "request-1"
    assert request["effect_key"] == "effect-1"
    assert request["actual_owned"] == {
        "stage_present": True,
        "owned_sdk_execution_count": 1,
        "owned_wall_s": 61.125,
        "natural_exit_count": 1,
        "non_natural_exit_count": 0,
        "failure_count": 1,
        "incomplete_count": 1,
        "failure_artifact_count": 2,
        "missing_reasons": [],
    }
    refs = request["raw_refs"]["stage"]
    assert set(refs) == {"request", "manifest", "execution", "failure", "training_failure", "records"}
    assert refs["execution"]["sha256"] == RUNNER.file_sha(
        session / "actual_stages/request-1/execution.json"
    )
    assert request["logical_settlement"]["ledger_settlement_gpu_s"] is None
    assert cost["actual_owned_summary"] == {
        "owned_sdk_execution_count": 1,
        "whole_owned_wall_s": 61.125,
        "failure_count": 1,
        "incomplete_count": 1,
        "known_owned_sdk_execution_count": 1,
        "known_owned_wall_s": 61.125,
        "missing_request_count": 0,
    }
    assert "condition" not in json.dumps(request, sort_keys=True)


def _synthetic_actual_session(
    tmp_path: Path,
    *,
    execution: bool,
    settled: bool,
    wall_s: float = 12.5,
    failure_artifacts: int = 1,
) -> Path:
    session = tmp_path / "session"
    stage = session / "actual_stages/request-1"
    (stage / "output").mkdir(parents=True)
    source_sha = "a" * 64
    write_new(
        session / "manifest.json",
        {"source": {"path": "private-source", "sha256": source_sha}, "execution_class": "actual"},
    )
    events: list[dict[str, object]] = [
        {"event": "proposed", "request_id": "request-1", "effect_key": "effect-1"},
        {"event": "consumed", "request_id": "request-1", "effect_key": "effect-1"},
    ]
    if settled:
        events.append(
            {
                "event": "receipt",
                "request_id": "request-1",
                "effect_key": "effect-1",
                "outcome": "unknown",
                "gpu_s": 0.0,
            }
        )
    (session / "execution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    write_new(
        stage / "request.json",
        {
            "request_id": "request-1",
            "effect_key": "effect-1",
            "request": {"request_id": "request-1", "effect_key": "effect-1"},
        },
    )
    write_new(
        stage / "manifest.json",
        {
            "request_id": "request-1",
            "effect_key": "effect-1",
            "source": {"source_sha256": source_sha},
        },
    )
    if execution:
        write_new(stage / "execution.json", {"wall_s": wall_s, "natural_exit": True})
        if failure_artifacts >= 1:
            write_new(stage / "output/failure.json", {"status": "invalid"})
        if failure_artifacts >= 2:
            write_new(stage / "output/training_failure.json", {"status": "invalid"})
        (stage / "output/records.jsonl").write_text('{"sanitized":true}\n', encoding="utf-8")
    return session


def test_missing_execution_cost_is_null_with_reason_not_zero(tmp_path: Path) -> None:
    session = _synthetic_actual_session(tmp_path, execution=False, settled=False)
    raw = RUNNER._raw_refs(session, 1, case_id="case-01", method="fixed_sequence")
    event = raw["resource_cost"]["events"][0]
    request = raw["resource_cost"]["request_costs"][0]
    assert event["costs"]["gpu_wall_s"] is None
    assert event["costs"]["execution_wall_s"] is None
    assert request["actual_owned"]["owned_sdk_execution_count"] is None
    assert request["actual_owned"]["owned_wall_s"] is None
    assert request["actual_owned"]["missing_reasons"] == ["stage_execution_missing"]
    assert raw["resource_cost"]["actual_owned_summary"]["whole_owned_wall_s"] is None
    assert raw["resource_cost"]["actual_owned_summary"]["known_owned_wall_s"] == 0


def test_resume_settlement_zero_does_not_erase_or_double_actual_spend(tmp_path: Path) -> None:
    session = _synthetic_actual_session(tmp_path, execution=True, settled=True)
    first = RUNNER._raw_refs(session, 1, case_id="case-01", method="fixed_sequence")
    second = RUNNER._raw_refs(session, 1, case_id="case-01", method="fixed_sequence")
    for raw in (first, second):
        cost = raw["resource_cost"]
        assert len(cost["events"]) == 1
        assert cost["events"][0]["costs"]["gpu_wall_s"] == 12.5
        request = cost["request_costs"][0]
        assert request["logical_settlement"]["ledger_settlement_gpu_s"] == 0.0
        assert request["actual_owned"]["owned_wall_s"] == 12.5
        assert request["double_count_prevented"] is True


def _cpu_model_journal_session(
    tmp_path: Path, *, pending: bool = False, failed: bool = False
) -> Path:
    label = "pending" if pending else "failed" if failed else "completed"
    session = tmp_path / f"{label}-session"
    session.mkdir()
    api_root = tmp_path / f"{label}-api"
    params = {"temperature": 0}
    source_anchor = "1" * 40
    protocol_sha = "2" * 64
    checkpoint_sha = "a" * 64
    write_new(
        session / "manifest.json",
        {
            "source": {"path": "sanitized-source", "sha256": "b" * 64},
            "source_verification": {"case_id": "case-model", "checkpoint_sha256": checkpoint_sha},
            "method": "one_shot_llm",
            "model_binding": {
                "source_anchor": source_anchor,
                "protocol_sha256": protocol_sha,
                "params": params,
                "api_cap": 40,
                "api_root": str(api_root.resolve()),
            },
            "execution_class": "fixture",
        },
    )
    messages = [{"role": "user", "content": "sanitized CPU fixture"}]
    decision = MODEL_JOURNAL.DecisionJournal(
        session / "model/0000",
        api_root,
        {
            "run_id": "case-model:one_shot_llm:0",
            "source_anchor": source_anchor,
            "protocol_sha256": protocol_sha,
            "checkpoint_sha256": checkpoint_sha,
            "snapshot_sha256": "c" * 64,
            "initial_messages_sha256": MODEL_JOURNAL.digest(messages),
        },
        params,
        api_cap=40,
    )

    def engine(complete):
        complete(messages, params)
        return {"selection": {"action": "hold"}}

    def complete(_messages, _params):
        if pending:
            raise KeyboardInterrupt
        if failed:
            raise ConnectionError("sanitized fixture transport failure")
        return {
            "content": '{"action":"hold"}',
            "status": "ok",
            "http_status": 200,
            "wall_s": 1.25,
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
                "cached_tokens": 2,
            },
        }

    if pending:
        with pytest.raises(KeyboardInterrupt):
            decision.advance(engine, complete, lambda: datetime.now(UTC))
    else:
        decision.advance(engine, complete, lambda: datetime.now(UTC))
    return session


def test_cpu_fixture_model_journal_cost_uses_audited_transport_once(tmp_path: Path) -> None:
    session = _cpu_model_journal_session(tmp_path)

    first = RUNNER._raw_refs(session, 3, case_id="case-model", method="one_shot_llm")
    second = RUNNER._raw_refs(session, 3, case_id="case-model", method="one_shot_llm")

    for raw in (first, second):
        cost = raw["resource_cost"]
        assert len(cost["events"]) == 1
        event = cost["events"][0]
        assert event["status"] == "completed"
        assert event["costs"] == {
            "model_calls": 1,
            "input_tokens": 11,
            "output_tokens": 3,
            "gpu_wall_s": 0.0,
            "cpu_s": None,
            "human_review_s": None,
            "execution_wall_s": 1.25,
            "simulator_steps": 0,
        }
        assert event["model_api"]["usage"]["cached_tokens"] == 2
        assert event["model_api"]["hosted_provider_gpu_s"] is None
        assert event["model_api"]["actual_billing_usd"] is None
        assert event["model_api"]["reproducible_estimate_usd"] is None
        assert cost["model_api_summary"] == {
            "reserved_attempts": 1,
            "completed_attempts": 1,
            "normal_results": 1,
            "infra_error_results": 0,
            "failed_results": 0,
            "unconfirmed_reservations": 0,
            "model_calls": 1,
            "known_model_calls": 1,
            "confirmed_http_responses": 1,
            "confirmed_no_http_responses": 0,
            "unknown_http_response_count": 0,
            "input_tokens": 11,
            "output_tokens": 3,
            "hosted_provider_gpu_s": None,
            "actual_billing_usd": None,
            "reproducible_estimate_usd": None,
        }
    assert first["resource_cost"]["events"][0]["attempt_id"] == second["resource_cost"]["events"][0]["attempt_id"]


def test_cpu_fixture_pending_model_reservation_counts_attempt_without_fake_usage(tmp_path: Path) -> None:
    session = _cpu_model_journal_session(tmp_path, pending=True)

    raw = RUNNER._raw_refs(session, 1, case_id="case-model", method="one_shot_llm")

    event = raw["resource_cost"]["events"][0]
    assert event["status"] == "unknown"
    assert event["costs"]["model_calls"] is None
    assert event["costs"]["input_tokens"] is None
    assert event["costs"]["output_tokens"] is None
    assert event["costs"]["execution_wall_s"] is None
    assert event["model_api"]["raw_refs"]["response"] is None
    summary = raw["resource_cost"]["model_api_summary"]
    assert summary["reserved_attempts"] == 1
    assert summary["completed_attempts"] == 0
    assert summary["unconfirmed_reservations"] == 1
    assert summary["model_calls"] is None
    assert summary["known_model_calls"] == 0


def test_cpu_fixture_failed_model_transport_counts_attempt_and_audits_receipt(tmp_path: Path) -> None:
    session = _cpu_model_journal_session(tmp_path, failed=True)

    raw = RUNNER._raw_refs(session, 1, case_id="case-model", method="one_shot_llm")

    event = raw["resource_cost"]["events"][0]
    assert event["status"] == "failed"
    assert event["costs"]["model_calls"] == 1
    assert event["costs"]["input_tokens"] is None
    assert event["costs"]["execution_wall_s"] >= 0
    assert event["model_api"]["raw_refs"]["response"] is not None
    assert event["model_api"]["raw_refs"]["receipt"] is not None


def test_runtime_confirmation_cpu_fixture_all_normal_ends_unidentifiable_without_fourth_cost(
    tmp_path: Path,
) -> None:
    runtime = mechanism_confirmation.mechanism_registry(profile="runtime_confirmation_v1")
    probes = list(runtime["fixed_order"])
    plan_ref, binding, repo, output = _setup(tmp_path, probe_scope=probes)
    plan_path = repo / plan_ref["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["authorization"]["probe_scope"] = probes
    plan_path.write_bytes(RUNNER.canonical(plan) + b"\n")
    plan_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    binding["plan"] = plan_ref
    for arm in binding["cases"][0]["arms"]:
        arm["prepare_payload"]["registry"] = runtime
        source_path = Path(arm["source_path"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["binding"]["fixture_outcomes"] = dict.fromkeys(probes, "normal")
        source_path.write_bytes(RUNNER.canonical(source) + b"\n")

    fixed_session = Path(binding["cases"][0]["arms"][0]["session_path"])
    dispatches = 0

    def invoke(command, session, payload, **kwargs):
        nonlocal dispatches
        if command == "dispatch" and session == fixed_session:
            dispatches += 1
        return investigate.run(command, session, payload, **kwargs)

    report = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        fixture=True,
        invoke=invoke,
    )

    fixed = report["arms"][0]
    assert fixed["status"] == "unidentifiable"
    assert dispatches == 3
    assert len(fixed["resource_cost"]["request_costs"]) == 3
    assert fixed["resource_cost"]["model_api_summary"]["reserved_attempts"] == 0
    assert sum(event["costs"]["gpu_wall_s"] for event in fixed["resource_cost"]["events"]) == 0


def test_actual_source_integrity_failure_stops_batch_and_sampler_in_finally(tmp_path: Path) -> None:
    plan_ref, binding, repo, output = _setup(tmp_path)
    protocol_path = repo / "protocol.json"
    protocol_path.write_bytes(
        RUNNER.canonical(
            {
                "model": {"api_cap": 40, "source_anchor": "1" * 40},
                "actors": ["fixture_actor"],
                "components": {"p1": {}, "p2": {}},
            }
        )
        + b"\n"
    )
    plan_path = repo / plan_ref["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["inputs"] = [{"role": "protocol", "path": "protocol.json",
                       "sha256": RUNNER._lf_sha(protocol_path)}]
    plan_path.write_bytes(RUNNER.canonical(plan) + b"\n")
    plan_ref = {"path": "plan.json", "sha256": RUNNER._lf_sha(plan_path)}
    binding["plan"] = plan_ref
    protocol_ref = reference(protocol_path)
    for arm in binding["cases"][0]["arms"]:
        arm["prepare_payload"] = {
            "source_path": arm["source_path"],
            "case_id": "case-01",
            "execution_class": "actual",
            "method": arm["method"],
            "protocol": protocol_ref,
            "artifact_resolver": {},
            "api_root": binding["api_root"] if arm["method"] in {
                "one_shot_llm", "iterative_agent"
            } else None,
            "max_checks": 3,
            "gpu_budget_per_check_s": 0,
        }
    blobs = {
        plan_path.relative_to(repo).as_posix(): plan_path.read_bytes(),
        protocol_path.relative_to(repo).as_posix(): protocol_path.read_bytes(),
        "runner-code.py": (repo / "runner-code.py").read_bytes(),
    }

    def git_call(command, **_kwargs):
        if command[1] == "show":
            return SimpleNamespace(returncode=0, stdout=blobs[command[2].split(":", 1)[1]], stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def invoke(command, _session, _payload, **_kwargs):
        if command == "prepare":
            return {"status": "open"}
        raise ValueError("verified source changed")

    class Sampler:
        def __init__(self):
            self.started = 0
            self.stopped = 0

        def start(self):
            self.started += 1
            return {"scope": "device_wide", "receipt": "start"}

        def stop(self):
            self.stopped += 1
            return {"scope": "device_wide", "receipt": "stop"}

    sampler = Sampler()
    report = RUNNER.run_comparison(
        plan_ref,
        binding,
        repo_root=repo,
        output_root=output,
        registration_anchor="2" * 40,
        fixture=False,
        invoke=invoke,
        git_call=git_call,
        gpu_sampler=sampler,
    )

    assert report["status"] == "stopped_integrity_failure"
    assert len(report["arms"]) == 1
    assert report["arms"][0]["status"] == "integrity_failure"
    assert sampler.started == sampler.stopped == 1
