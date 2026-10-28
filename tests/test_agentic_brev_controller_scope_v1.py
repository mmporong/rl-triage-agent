"""POSIX 검증기와 기존 승인·seal 연결을 합성 CPU 원문으로 검증한다."""
from __future__ import annotations

import hashlib
import sys
import threading
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / folder) for folder in ("evals", "tests", "src")]
import agentic_brev_controller_scope_v1 as S  # pyright: ignore[reportMissingImports]
import agentic_investigation_loop_v1 as L  # pyright: ignore[reportMissingImports]
from test_agentic_brev_probe_evidence_v1 import build, fixture
from test_agentic_investigation_runtime_v1 import commit_objects
from test_selection_comparison_controller_v1 import FACTS, VIEW_SHA

C, E = S.C, S.E
DEPENDENCIES = ("evals/brev_recovery_backend_v4.py", "evals/lease_timestamps.py",
    "evals/model_decision_journal_v1.py", "evals/vision_remote_gpu_v1.py",
    "evals/optimizer_observability.py", "bench/residual_training_v1.py")


def source_registration(root, *, omitted=None):
    sources = {name: (ROOT / name).read_bytes() for name in S.SOURCE_FILES}
    protocol = {"schema": "agentic_investigation_runtime_v1_protocol",
        "controller_extensions": [E.GPU_CONTRACT], "sources_lf": {
            name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
            for name, raw in sources.items() if name != omitted}}
    relative = "bench/protocols/scope.json"
    anchor = commit_objects(root, {**sources, relative: E.J._bytes(protocol)})
    return {"protocol": {"path": str((root / relative).resolve()),
                         "sha256": C.digest_file(root / relative)}, "anchor": anchor,
            "repo": str(root.resolve())}


def binding(root, *, actual=False):
    request, roots = fixture(root, kind=E.ACTUAL if actual else E.FIXTURE)
    evidence = build(request, roots, allow_fixture=not actual)
    path = root / "completion.json"
    E.write_new(path, evidence)
    ref = {"path": str(path.resolve()), "sha256": C.digest_file(path)}
    expected = {key: evidence[key] for key in S.EXPECTED_KEYS - {"operation_id"}}
    expected["operation_id"] = request["operation_id"]
    kwargs = {"expected": expected, "roots": roots,
              "request_sha256": E.digest(request), "roots_sha256": E.digest(roots)}
    plan = {"stage": str(root / "stage"), "gpu_contract": E.GPU_CONTRACT,
            **{k: evidence[k] for k in ("checkpoint_sha256", "seed", "run_name")}}
    return evidence, ref, kwargs, plan


def test_scope_delegates_legacy_and_restores_on_error(tmp_path):
    original = {k: getattr(C, k) for k in S._HOOKS}
    with pytest.raises(RuntimeError, match="test interruption"), S.BrevProbeScope(allow_fixture=True):
        assert E.GPU_CONTRACT in C.GPU_CONTRACTS
        assert C._probe_measurement({"timeout_ratio": 1}, C.FIXTURE_EVIDENCE) == {"timeout_ratio": 1}
        raise RuntimeError("test interruption")
    assert all(getattr(C, k) is v for k, v in original.items())


@pytest.mark.parametrize("actual", [True, False])
def test_registered_measurement_keeps_fixture_and_actual_separate(tmp_path, actual):
    evidence, ref, kwargs, plan = binding(tmp_path, actual=actual)
    registration = source_registration(tmp_path / "repo") if actual else None
    with S.BrevProbeScope(allow_fixture=not actual, source_registration=registration) as scope:
        scope.register(ref, **kwargs)
        kind = C.ACTUAL_EVIDENCE if actual else C.FIXTURE_EVIDENCE
        assert C._probe_measurement(evidence, kind) == evidence["measurement"]
        C._validate_evidence_plan(evidence, plan)
        other = C.FIXTURE_EVIDENCE if actual else C.ACTUAL_EVIDENCE
        with pytest.raises(ValueError, match="cannot cross"):
            C._probe_measurement(evidence, other)


def test_fixture_is_rejected_by_default(tmp_path):
    _, ref, kwargs, _ = binding(tmp_path)
    with pytest.raises(ValueError):
        S.BrevProbeScope().register(ref, **kwargs)


def test_fixture_switch_cannot_accept_actual_without_registered_sources(tmp_path):
    _, ref, kwargs, _ = binding(tmp_path, actual=True)
    with pytest.raises(ValueError, match="actual completion requires source registration"):
        S.BrevProbeScope(allow_fixture=True).register(ref, **kwargs)


@pytest.mark.parametrize("name", DEPENDENCIES)
def test_each_verifier_dependency_is_required(tmp_path, name):
    registration = source_registration(tmp_path / "repo", omitted=name)
    with pytest.raises(ValueError, match="source closure"):
        S.BrevProbeScope(source_registration=registration)
    assert E.GPU_CONTRACT not in C.GPU_CONTRACTS


@pytest.mark.parametrize("name", DEPENDENCIES)
def test_each_dependency_change_after_commit_is_rejected(tmp_path, name):
    registration = source_registration(tmp_path / "repo")
    S.validate_source_registration(registration)
    source = tmp_path / "repo" / name
    source.write_bytes(source.read_bytes() + b"\n# modified after registration\n")
    with pytest.raises(ValueError, match="source/blob"):
        S.BrevProbeScope(source_registration=registration)


def test_source_change_inside_scope_is_rejected_before_authorization(tmp_path):
    registration = source_registration(tmp_path / "repo")
    with S.BrevProbeScope(source_registration=registration):
        source = tmp_path / "repo" / DEPENDENCIES[0]
        source.write_bytes(source.read_bytes() + b"\n# modified during execution\n")
        with pytest.raises(ValueError, match="source/blob"):
            C._execution_plan({"gpu_contract": E.GPU_CONTRACT}, {})


def test_unregistered_and_wrong_intent_are_rejected(tmp_path):
    evidence, ref, kwargs, _ = binding(tmp_path)
    with S.BrevProbeScope(allow_fixture=True) as scope:
        with pytest.raises(ValueError, match="unregistered"):
            C._probe_measurement(evidence, C.FIXTURE_EVIDENCE)
        wrong = deepcopy(kwargs)
        wrong["expected"]["probe"] = "P_noise"
        with pytest.raises(ValueError, match="host intent"):
            scope.register(ref, **wrong)
        scope.register(ref, **kwargs)
        with pytest.raises(ValueError, match="duplicate"):
            scope.register(ref, **kwargs)


@pytest.mark.parametrize("key,value", [("stage", "/wrong"), ("seed", 3),
    ("run_name", "wrong"), ("checkpoint_sha256", "f" * 64), ("gpu_contract", "residual_v1")])
def test_wrong_execution_plan_is_rejected(tmp_path, key, value):
    evidence, ref, kwargs, plan = binding(tmp_path)
    with S.BrevProbeScope(allow_fixture=True) as scope:
        scope.register(ref, **kwargs)
        with pytest.raises(ValueError, match="plan binding"):
            C._validate_evidence_plan(evidence, {**plan, key: value})


def test_raw_mutation_and_foreign_thread_are_rejected(tmp_path):
    evidence, ref, kwargs, _ = binding(tmp_path)
    with S.BrevProbeScope(allow_fixture=True) as scope:
        scope.register(ref, **kwargs)
        errors = []

        def foreign():
            try:
                C._probe_measurement(evidence, C.FIXTURE_EVIDENCE)
            except ValueError as error:
                errors.append(str(error))
            try:
                C._execution_plan({"gpu_contract": E.GPU_CONTRACT}, {})
            except ValueError as error:
                errors.append(str(error))

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and errors == ["controller scope owner thread"] * 2
        with pytest.raises(ValueError, match="nested"), S.BrevProbeScope(allow_fixture=True):
            pytest.fail("nested context entered")
        (tmp_path / "stage/job.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            C._probe_measurement(evidence, C.FIXTURE_EVIDENCE)


def probe_snapshot(tmp_path, cp):
    inputs = {"case": {"checkpoint_sha256": cp, "telemetry_sha256": "2" * 64},
              "reference": {"checkpoint_sha256": cp, "telemetry_sha256": "4" * 64}}
    packet = C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                             facts=FACTS, input_bindings=inputs, execution_budget_s=1200)
    arms = tmp_path / "arms"
    ledger = arms / "model_adaptive/epoch1/probes.ledger.jsonl"
    return arms, ledger, C.V.build_snapshot(packet, "posix_cpu", "model_adaptive", VIEW_SHA, ledger, None)


def test_fixture_scope_cannot_consume_actual_authorization_without_sources(tmp_path):
    evidence, ref, _, plan = binding(tmp_path / "source")
    arms, ledger, snapshot = probe_snapshot(tmp_path, evidence["checkpoint_sha256"])
    loop = L.Investigation(snapshot, goal="CPU")
    content = E.J._bytes({"action": "run_probe", "target": "P_episode", "fact_ids": [next(iter(FACTS))]}).decode()
    assert loop.propose(content)["status"] == "valid"
    choice = loop.probe_choice()
    plan.update(args=C.request_spec(snapshot, choice)["args"], reference={
        "evidence_path": ref["path"], "evidence_sha256": ref["sha256"], **plan})
    with S.BrevProbeScope(allow_fixture=True), pytest.raises(ValueError, match="authorization requires source"):
        C.authorize_probe(arms, "model_adaptive", 1, snapshot, choice, "cpu_explicit",
                          evidence_kind=C.ACTUAL_EVIDENCE, execution_plan=plan)
    assert not ledger.exists() and not list(arms.rglob("authorization_*.json"))


def test_existing_controller_authorizes_seals_and_loop_consumes_fixture(tmp_path):
    evidence, ref, kwargs, plan = binding(tmp_path / "source")
    arms, _, snapshot = probe_snapshot(tmp_path, evidence["checkpoint_sha256"])
    fixture_loop = L.Investigation(snapshot, goal="CPU", allow_fixture=True)
    production_loop = L.Investigation(snapshot, goal="CPU")
    content = E.J._bytes({"action": "run_probe", "target": "P_episode", "fact_ids": [next(iter(FACTS))]}).decode()
    assert fixture_loop.propose(content)["status"] == "valid"
    assert production_loop.propose(content)["status"] == "valid"
    choice = fixture_loop.probe_choice()
    plan.update(args=C.request_spec(snapshot, choice)["args"], reference={
        "evidence_path": ref["path"], "evidence_sha256": ref["sha256"], **plan})
    with S.BrevProbeScope(allow_fixture=True, source_registration=source_registration(tmp_path / "repo")) as scope:
        scope.register(ref, **kwargs)
        auth = C.authorize_probe(arms, "model_adaptive", 1, snapshot, choice, "cpu_explicit",
                                 evidence_kind=C.FIXTURE_EVIDENCE, execution_plan=plan)
        sealed = C.seal_probe(auth, snapshot, Path(ref["path"]), Path(ref["path"]), 10, 0)
        next_snapshot = C.V.build_snapshot(**snapshot["inputs"])
        with pytest.raises(ValueError, match="fixture"):
            production_loop.complete_probe(sealed, next_snapshot)
        fixture_loop.complete_probe(sealed, next_snapshot)
        assert fixture_loop.snapshot["view"]["observations"][-1] == {"probe": "P_episode", "outcome": "normal"}
        assert C.verify_probe(Path(sealed["path"]), snapshot)["evidence_kind"] == C.FIXTURE_EVIDENCE
