"""기존 Runtime의 CPU 경계를 사용하는 input/busy guard 회귀."""
from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src"), str(ROOT / "tests")]
import test_agentic_investigation_runtime_v1 as F
from test_agentic_investigation_evidence_v1 import case  # noqa: F401

from evals import agentic_runtime_input_guard_v1 as G

R, J = G.BASE, G.J


def setup(root: Path, snapshot: dict, *, evidence: dict | None = None, decisions: int = 4) -> tuple:
    api = root / "api"
    F.save(api / "manifest.json", {"schema": J.API_SCHEMA, "api_cap": 12})
    reg = R.registration(snapshot, goal="CPU guard fixture", api=R.capture_api(api, baseline=0),
                         max_decisions=decisions, evidence=evidence, allow_fixture=True)
    reg_ref = F.save(root / "registration.json", reg)
    names = [
        "evals/agentic_investigation_runtime_v1.py",
        "tests/test_agentic_investigation_runtime_v1.py",
        "evals/agentic_runtime_input_guard_v1.py",
        "tests/test_agentic_runtime_input_guard_v1.py",
    ]
    sources = {name: (ROOT / name).read_bytes() for name in names}
    protocol = {"schema": R.SCHEMA + "_protocol", "registration_sha256": reg_ref["sha256"],
                "api_cap": 12, "sources_lf": {name: G.source_hash(ROOT / name) for name in names}}
    files = {**sources, "bench/protocols/guard.json": J._bytes(protocol)}
    repo = root / "source-objects"
    anchor = F.commit_objects(repo, files)
    return (root / "run", reg_ref, R.ref(repo / "bench/protocols/guard.json"), anchor), repo


def runtime(args: tuple, repo: Path) -> G.Runtime:
    return G.Runtime(*args, repo=repo)


def test_requires_guard_and_test_in_registered_sources(tmp_path: Path) -> None:
    args, repo = F.setup(tmp_path / "old", F.probe_source(tmp_path))
    with pytest.raises(ValueError, match="guard/test sources"):
        G.Runtime(*args, repo=repo)
    assert not args[0].exists()


def test_missing_registered_artifact_writes_unknown_hold_once(case: dict, tmp_path: Path) -> None:  # noqa: F811
    snapshot, evidence = F.adapter_for_case(case, tmp_path)
    args, repo = setup(tmp_path / "guard", snapshot, evidence=evidence)
    missing = Path(case["roots"]["fixture"]) / "case-telemetry.json"
    raw = missing.read_bytes()
    missing.unlink()
    calls: list[int] = []

    def model(*_args):
        calls.append(1)
        raise AssertionError("unavailable input cannot call model")

    first = runtime(args, repo).advance(complete=model)
    assert first["status"] == "input_unavailable_hold" and first["outcome"] == "unknown"
    assert first["automatic_retry"] is False and calls == []
    receipt = J._read(Path(first["receipt"]["path"]))
    assert receipt["artifact"] == {"root": "fixture", "relative_path": "case-telemetry.json"}
    missing.write_bytes(raw)
    assert runtime(args, repo).advance(complete=model) == first
    assert calls == [] and len(list((tmp_path / "guard/api/reservations").glob("*.json"))) == 0


def test_integrity_value_error_is_not_converted_to_hold(tmp_path: Path) -> None:
    args, repo = setup(tmp_path / "guard", F.probe_source(tmp_path))
    registration = Path(args[1]["path"])
    raw = J._read(registration)
    raw["goal"] = "tampered"
    registration.write_bytes(J._bytes(raw))
    with pytest.raises(ValueError):
        runtime(args, repo).advance(complete=F.no_model)
    assert not (args[0] / "input-unavailable.json").exists()


def test_source_mutation_wins_over_missing_registered_input(case: dict, tmp_path: Path) -> None:  # noqa: F811
    snapshot, evidence = F.adapter_for_case(case, tmp_path)
    args, repo = setup(tmp_path / "guard", snapshot, evidence=evidence)
    instance = runtime(args, repo)
    Path(case["roots"]["fixture"], "case-telemetry.json").unlink()
    source = repo / "evals/agentic_runtime_input_guard_v1.py"
    source.write_bytes(source.read_bytes() + b"\n# mutated\n")
    with pytest.raises(ValueError):
        instance.advance(complete=F.no_model)
    assert not (args[0] / "input-unavailable.json").exists()


def test_explicit_gpu_busy_holds_before_approve_or_dispatch(tmp_path: Path) -> None:
    args, repo = setup(tmp_path / "guard", F.probe_source(tmp_path))
    backend = F.Backend(tmp_path, outcomes=[])
    observed: list[str] = []

    def busy(operation: str, _intent: dict) -> dict:
        observed.append(operation)
        return {"status": "busy", "observation_id": "fixture-busy-1"}

    answer = runtime(args, repo).advance(complete=F.Model([("run_probe", "P_episode")]),
                                         host=G.guarded_host(backend.callbacks(), busy))
    assert answer["status"] == "host_hold" and len(observed) == 1
    assert backend.approvals == backend.dispatches == []
    event = J._read(args[0] / "steps/0001/host-events/0001.json")
    assert event["phase"] == "approve" and event["response"]["busy_observation"]["status"] == "busy"


def test_available_preflight_keeps_original_approval_and_dispatch(tmp_path: Path) -> None:
    args, repo = setup(tmp_path / "guard", F.probe_source(tmp_path))
    backend = F.Backend(tmp_path, outcomes=["normal"])
    host = G.guarded_host(backend.callbacks(),
                          lambda *_: {"status": "available", "observation_id": "fixture-free-1"})
    answer = runtime(args, repo).advance(complete=F.Model([("run_probe", "P_episode")]), host=host)
    assert answer["events"][-1]["host_judgment"]["outcome"] == "normal"
    assert len(backend.approvals) == len(backend.dispatches) == 1


def test_dispatch_file_not_found_is_not_input_hold(case: dict, tmp_path: Path) -> None:  # noqa: F811
    snapshot, evidence = F.adapter_for_case(case, tmp_path)
    args, repo = setup(tmp_path / "guard", snapshot, evidence=evidence)
    backend = F.Backend(tmp_path, outcomes=[])
    missing = Path(case["roots"]["fixture"]) / "dispatch-artifact.json"

    def dispatch(operation: str, intent: dict, authorization: dict) -> dict:
        backend.dispatches.append(operation)
        raise FileNotFoundError(2, "fixture dispatch source missing", str(missing))

    callbacks = R.HostCallbacks(backend.approve, dispatch, backend.lookup)
    with pytest.raises(FileNotFoundError, match="dispatch source missing"):
        runtime(args, repo).advance(complete=F.Model([("run_probe", "P_episode")]), host=callbacks)
    assert not (args[0] / "input-unavailable.json").exists()
    event = J._read(args[0] / "steps/0001/host-events/0002.json")
    assert event["phase"] == "dispatch" and event["interrupted"] == "FileNotFoundError"
    assert len(backend.approvals) == len(backend.dispatches) == 1


def test_cpu_evidence_read_does_not_call_busy_reader(case: dict, tmp_path: Path) -> None:  # noqa: F811
    snapshot, evidence = F.adapter_for_case(case, tmp_path)
    args, repo = setup(tmp_path / "guard", snapshot, evidence=evidence)
    backend = F.Backend(tmp_path, outcomes=[])

    def denied(*_args):
        raise AssertionError("CPU evidence read cannot query GPU busy")

    answer = runtime(args, repo).advance(
        complete=F.Model([("read_evidence", "evidence:training-01")]),
        host=G.guarded_host(backend.callbacks(), denied),
    )
    assert answer["decisions_left"] == 3 and backend.approvals == backend.dispatches == []


@pytest.mark.parametrize(
    ("provider_result", "expected_status", "expected_calls"),
    [
        ({"infra_error": "TimeoutError"}, "paused_infra", 1),
        ({"provider_model": R.CALL.O.MODEL, "finish_reason": "stop", "content": ""}, "hold", 2),
        ({"error_type": "UncertainProviderState"}, "hold", 1),
    ],
)
def test_provider_timeout_empty_and_uncertain_keep_original_status(
    tmp_path: Path, provider_result: dict, expected_status: str, expected_calls: int,
) -> None:
    args, repo = setup(tmp_path / "guard", F.probe_source(tmp_path))
    calls: list[int] = []

    def provider(*_args):
        calls.append(1)
        return deepcopy(provider_result)

    now = datetime(2026, 11, 1, tzinfo=timezone.utc)  # noqa: UP017
    answer = runtime(args, repo).advance(complete=provider, now_utc=lambda: now)
    assert answer["status"] == expected_status and len(calls) == expected_calls
    assert not (args[0] / "input-unavailable.json").exists()


def test_pending_journal_restart_does_not_resend(tmp_path: Path) -> None:
    args, repo = setup(tmp_path / "guard", F.probe_source(tmp_path))
    calls: list[int] = []

    def crash(*_args):
        calls.append(1)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runtime(args, repo).advance(complete=crash)
    answer = runtime(args, repo).advance(complete=F.no_model)
    assert answer["status"] == "blocked_pending" and calls == [1]


def test_host_pending_restart_uses_lookup_without_new_approval_or_dispatch(tmp_path: Path) -> None:
    args, repo = setup(tmp_path / "guard", F.probe_source(tmp_path))
    backend = F.Backend(tmp_path, outcomes=["normal"])
    backend.pending = True
    busy_calls: list[int] = []

    def available(*_args):
        busy_calls.append(1)
        return {"status": "available", "observation_id": "fixture-free-2"}

    host = G.guarded_host(backend.callbacks(), available)
    assert runtime(args, repo).advance(complete=F.Model([("run_probe", "P_episode")]), host=host)["status"] == "blocked_host"
    assert runtime(args, repo).advance(complete=F.no_model, host=host)["status"] == "blocked_host"
    assert len(backend.approvals) == len(backend.dispatches) == len(backend.lookups) == 1
    assert busy_calls == [1]
