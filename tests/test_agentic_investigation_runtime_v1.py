"""CPU callback fixture와 불변 원문으로 실행·재시작 경계를 검증한다. 외부 실행은 없다."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zlib
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src"), str(ROOT / "tests")]
import agentic_investigation_runtime_v1 as R
import test_selection_comparison_controller_v1 as F
from test_agentic_investigation_evidence_v1 import case  # noqa: F401 - pytest fixture

C, CALL, J = R.C, R.CALL, R.J
POLICY = "model_adaptive"


def save(path: Path, value: dict) -> dict:
    J._write_new(path, value)
    return R.ref(path)


def commit_objects(repo: Path, files: dict, parent: str | None = None) -> str:
    """개인 fixture에 object만 쓴다. Git 프로세스를 실행하지 않는다."""
    metadata = repo / ".git"
    (metadata / "objects").mkdir(parents=True, exist_ok=True)

    def object_(kind: str, body: bytes) -> str:
        raw = f"{kind} {len(body)}".encode() + b"\0" + body
        oid = hashlib.sha1(raw).hexdigest()
        path = metadata / "objects" / oid[:2] / oid[2:]
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(zlib.compress(raw))
        return oid

    tree: dict = {}
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        cursor = tree
        for part in Path(name).parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[Path(name).name] = object_("blob", content)

    def tree_(value: dict) -> str:
        rows = []
        for name, child in sorted(value.items()):
            oid = tree_(child) if isinstance(child, dict) else child
            rows.append((b"40000 " if isinstance(child, dict) else b"100644 ") + name.encode()
                        + b"\0" + bytes.fromhex(oid))
        return object_("tree", b"".join(rows))

    body = f"tree {tree_(tree)}\n" + (f"parent {parent}\n" if parent else "")
    oid = object_("commit", (body + "author CPU <cpu@example.invalid> 1 +0000\n"
                  "committer CPU <cpu@example.invalid> 1 +0000\n\nCPU fixture\n").encode())
    (metadata / "HEAD").write_text(oid, encoding="ascii")
    return oid


def setup(root: Path, snapshot: dict, *, evidence: dict | None = None, decisions: int = 10,
          fixture: bool = True, prefix: list | None = None, api: Path | None = None,
          repo: Path | None = None, parent: str | None = None, extra_files: dict | None = None,
          omitted_source: str | None = None) -> tuple:
    api = api or root / "api"
    if not api.exists():
        save(api / "manifest.json", {"schema": J.API_SCHEMA, "api_cap": 12})
    baseline = len(list((api / "reservations").glob("*.json")))
    reg = R.registration(snapshot, goal="CPU 조사", api=R.capture_api(api, baseline=baseline),
        max_decisions=decisions, evidence=evidence, prefix=prefix, allow_fixture=fixture)
    reg_ref = save(root / "registration.json", reg)
    sources = {name: (ROOT / name).read_bytes() for name in (
        "evals/agentic_investigation_runtime_v1.py", "tests/test_agentic_investigation_runtime_v1.py")}
    protocol = {"schema": R.SCHEMA + "_protocol", "registration_sha256": reg_ref["sha256"],
        "api_cap": 12, "sources_lf": {name: R.source_hash(ROOT / name)
                                     for name in sources if name != omitted_source}}
    files = {**(extra_files or {}), **sources, "bench/protocols/execution.json": J._bytes(protocol)}
    repo = repo or root / "source-objects"
    anchor = commit_objects(repo, files, parent)
    protocol_ref = R.ref(repo / "bench/protocols/execution.json")
    args = (root / "run", reg_ref, protocol_ref, anchor)
    return args, repo


class Model:
    def __init__(self, choices: list[tuple]):
        self.choices, self.calls = iter(choices), []

    def __call__(self, messages: list, params: dict) -> dict:
        assert params == CALL.params_for_messages(messages)
        self.calls.append(deepcopy(messages))
        action, target = next(self.choices)
        content = messages[1]["content"]
        view = json.loads(content[0]["text"] if isinstance(content, list) else content)
        facts = [key for key in view["evidence_facts"] if key.startswith("observation:")]
        body = {"action": action, "target": target, "fact_ids": facts or ["config:action_scale"]}
        return {"provider_model": CALL.O.MODEL, "finish_reason": "stop", "wall_s": 0.01,
                "content": json.dumps(body)}


def no_model(*_args):
    pytest.fail("unexpected model callback")


class Backend:
    def __init__(self, root: Path, *, outcomes: list[str], current: dict | None = None,
                 reference: dict | None = None):
        self.root, self.outcomes = root, iter(outcomes)
        self.current, self.reference = current, reference
        self.approvals, self.dispatches, self.lookups = [], [], []
        self.jobs: dict = {}
        self.crash_dispatch = False
        self.pending = False
        self.missing_approval = False
        self.corrupt_result: Callable[[dict], None] | None = None

    def callbacks(self) -> R.HostCallbacks:
        return R.HostCallbacks(self.approve, self.dispatch, self.lookup)

    def approve(self, operation: str, intent: dict) -> dict:
        self.approvals.append(operation)
        if self.missing_approval:
            return {"status": "hold", "reason": "no_approval"}
        snap = intent["snapshot"]
        epoch = snap["inputs"]["packet"]["epoch"]
        arm = Path(snap["inputs"]["ledger_path"]).parent.parent.name
        if intent["action"] == "run_probe":
            probe = intent["target"]
            stage = self.root / "raw" / operation
            stage.mkdir(parents=True)
            reference = stage / "reference.json"
            save(reference, F.measurement(probe, "normal"))
            spec = intent["request_spec"]
            plan = {"stage": str(stage.resolve()), "run_name": "cpu_probe_" + operation[:12],
                "checkpoint_sha256": spec["checkpoint_sha256"], "seed": 0, "args": spec["args"],
                "gpu_contract": "residual_v1", "reference": {"evidence_path": str(reference.resolve()),
                    "evidence_sha256": J.file_sha(reference), "checkpoint_sha256": spec["reference_checkpoint_sha256"],
                    "stage": str(stage.resolve()), "run_name": "cpu_reference", "seed": 0, "gpu_contract": "residual_v1"}}
            auth = C.authorize_probe(self.root, arm, epoch, snap, intent["choice"], "cpu-operator",
                                     evidence_kind=C.FIXTURE_EVIDENCE, execution_plan=plan)
        else:
            ledger = self.root / arm / "corrections.ledger.jsonl"
            seals = [job["result"]["seal"] for job in self.jobs.values()
                     if job["status"] == "completed" and job["action"] == "run_probe"
                     and C.read(Path(job["result"]["seal"]["path"]))["epoch"] == epoch]
            seals = [{**s, "snapshot": C.read(Path(s["path"]))["snapshot"]} for s in seals]
            auth = {"authorization": C.authorize_repair(ledger, arm, snap,
                snap["inputs"]["packet"]["applied_corrections"], seals, "cpu-operator", 3600),
                "ledger_path": str(ledger.resolve())}
        response = {"status": "approved", "authorization": auth, "action": intent["action"]}
        self.jobs[operation] = response
        return response

    def dispatch(self, operation: str, intent: dict, authorization: dict) -> dict:
        self.dispatches.append(operation)
        if self.pending:
            self.jobs[operation] = {"status": "pending", "authorization": authorization, "action": intent["action"]}
            return {"status": "pending"}
        result = self.finish(operation, intent, authorization)
        if self.crash_dispatch:
            raise KeyboardInterrupt
        return result

    def finish(self, operation: str, intent: dict, authorization: dict) -> dict:
        before = intent["snapshot"]
        if intent["action"] == "run_probe":
            auth = C.read(Path(authorization["path"]))
            plan = auth["execution_plan"]
            outcome = next(self.outcomes)
            if outcome == "unknown":
                seal = C.fail_probe(authorization, before, gpu_s=1.0, exit_code=1, reason="cpu-failure")
            else:
                case_path = Path(plan["stage"]) / "case.json"
                save(case_path, F.measurement(intent["target"], outcome))
                seal = C.seal_probe(authorization, before, case_path, Path(plan["reference"]["evidence_path"]), 1.0)
            seal = {"path": seal["path"], "sha256": seal["sha256"]}
            after = C.V.build_snapshot(**before["inputs"])
            result = {"seal": seal, "next_snapshot": after}
        else:
            assert self.current is not None and self.reference is not None
            packet = before["inputs"]["packet"]
            epoch, applied = packet["epoch"], packet["applied_corrections"]
            arm = self.current["arm"]
            after_applied = [*applied, intent["target"]]
            output = F.artifact(self.root, arm, POLICY, epoch, after_applied,
                f"corrected-{epoch}".encode(), self.current["checkpoint"]["sha256"])
            report = F.report(self.root, "s2120_first.json" if epoch == 1 else "s2120_both.json", f"out{epoch}", output)
            reference_report = self.root / "reports/reference.json"
            if not reference_report.exists():
                reference_report = F.report(self.root, "s2120_none.json", "reference", self.reference)
            seal_path = self.root / arm / f"epoch{epoch}/repair.json"
            parent = self.root / arm / "epoch1/repair.json" if epoch == 2 else None
            C.seal_repair(seal_path, ledger_path=Path(authorization["ledger_path"]),
                authorization=authorization["authorization"], arm=arm, policy=POLICY, epoch=epoch,
                applied_before=applied, input_artifact=self.current, output_artifact=output,
                output_report_path=report, reference_artifact=self.reference, reference_report_path=reference_report,
                bands=F.BANDS, gpu_s=30.0, parent_path=parent)
            after = None
            if epoch == 1:
                prior = {"verdict": "not_recovered", "correction_id": intent["target"],
                    "result_sha256": C.digest_file(seal_path), "checkpoint_sha256": output["checkpoint"]["sha256"]}
                after = F.snapshot(self.root, arm, POLICY, epoch=2, applied=after_applied, prior=prior)
            self.current = output
            result = {"seal": R.ref(seal_path), "ledger_path": authorization["ledger_path"],
                      "bands": F.BANDS, "next_snapshot": after, "evidence": None}
        envelope = {"status": "completed", "authorization": authorization, "result": result,
                    "action": intent["action"]}
        if self.corrupt_result:
            self.corrupt_result(envelope)
        self.jobs[operation] = envelope
        return envelope

    def lookup(self, operation: str, _intent: dict) -> dict:
        self.lookups.append(operation)
        return self.jobs.get(operation, {"status": "not_found"})


def advance(args, repo, model=None, backend=None):
    # 매번 새 Runtime과 새 Investigation을 만들어 메모리 재사용 없이 재개한다.
    return R.Runtime(*args, repo=repo).advance(complete=model, host=backend.callbacks() if backend else None)


def probe_source(root: Path) -> dict:
    return F.snapshot(root, "case", POLICY)


@pytest.mark.parametrize("missing", ["evals/agentic_investigation_runtime_v1.py",
                                     "tests/test_agentic_investigation_runtime_v1.py"])
def test_runtime_requires_both_registered_sources_before_creating_run(tmp_path, missing):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path), omitted_source=missing)
    assert (repo / missing).is_file()  # blob이 있어도 공개 sources_lf 누락은 허용하지 않는다.
    with pytest.raises(ValueError, match="runtime public registration binding"):
        R.Runtime(*args, repo=repo)
    assert not args[0].exists()


def test_retrieval_frames_normal_then_abnormal_proposal(case, tmp_path):  # noqa: F811 - pytest fixture
    source = case["snapshot"]
    # backend 원장 경로가 snapshot과 일치하도록 case의 조사 원장을 명시적으로 배치한다.
    inputs = deepcopy(source["inputs"])
    inputs["ledger_path"] = tmp_path / "case/epoch1/probes.ledger.jsonl"
    source = C.V.build_snapshot(**inputs)
    bridge = R.B.register(case["index"], case["roots"], source,
        expected_index_sha256=R.B.E.index_sha256(case["index"]),
        source_lineage_id=case["registration"]["session"]["source_lineage_id"],
        aliases=case["registration"]["aliases"])
    spec = R.evidence_spec(R.B.EvidenceAdapter(case["index"], case["roots"], bridge,
        expected_registration_sha256=R.B.E.index_sha256(bridge)))
    args, repo = setup(tmp_path / "runtime", source, evidence=spec)
    model = Model([("read_evidence", "evidence:training-01"), ("inspect_frames", "evidence:clip-01"),
                   ("run_probe", "P_episode"), ("run_probe", "P_reward"),
                   ("propose_correction", C.RC.CORRECTIONS["reward"])])
    backend = Backend(tmp_path, outcomes=["normal", "abnormal"])
    assert advance(args, repo, model, backend)["decisions_left"] == 9
    assert advance(args, repo, model, backend)["decisions_left"] == 8
    assert advance(args, repo, model, backend)["events"][-1]["host_judgment"]["outcome"] == "normal"
    assert advance(args, repo, model, backend)["events"][-1]["host_judgment"]["outcome"] == "abnormal"
    # 교정 제안은 명시적 host 없이는 승인하지 않는다.
    proposal = advance(args, repo, model)
    assert proposal["status"] == "blocked_host" and len(model.calls) == 5
    assert len(backend.dispatches) == 2 and proposal["events"][-1]["receipt_fact_ids_cited"]
    followup = J._read(args[0] / "steps/0005/request.json")["context"]["model_view"]
    assert followup["observations"][-1] == {"probe": "P_reward", "outcome": "abnormal"}
    assert sum(b["type"] == "image_url" for b in model.calls[2][1]["content"]) == 2
    assert not proposal["causal_effect_measured"]


def repair_setup(tmp_path, monkeypatch):
    current = F.artifact(tmp_path, "case", POLICY, 0, [], b"fault-source")
    reference = F.artifact(tmp_path, "NONE", "shared_reference", 0, [], b"healthy-reference")
    bindings = deepcopy(F.BINDINGS)
    bindings["case"]["checkpoint_sha256"] = current["checkpoint"]["sha256"]
    bindings["reference"]["checkpoint_sha256"] = reference["checkpoint"]["sha256"]
    monkeypatch.setattr(F, "BINDINGS", bindings)
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["abnormal", "abnormal"], current=current, reference=reference)
    return args, repo, backend


def test_two_distinct_repairs_require_nonrecovery_then_clear_old_evidence(tmp_path, monkeypatch):
    args, repo, backend = repair_setup(tmp_path, monkeypatch)
    model = Model([("run_probe", "P_reward"), ("propose_correction", C.RC.CORRECTIONS["reward"]),
                   ("run_probe", C.RC.NEW_PROBE), ("propose_correction", C.RC.CORRECTIONS[C.RC.COUNT])])
    results = [advance(args, repo, model, backend) for _ in range(4)]
    assert results[-1]["status"] == "recovered"
    assert results[1]["events"][-1]["host_judgment"]["outcome"] == "not_recovered"
    assert len(model.calls) == len(backend.dispatches) == 4
    assert results[-1]["events"][-1]["latest_probe_followup"]["cited"] is True
    view = json.loads(model.calls[2][1]["content"])
    assert view["available_actions"]["inspect_frames"] == view["available_actions"]["read_evidence"] == []
    assert advance(args, repo, no_model, backend) == results[-1]


def test_duplicate_second_correction_holds_without_another_dispatch(tmp_path, monkeypatch):
    args, repo, backend = repair_setup(tmp_path, monkeypatch)
    model = Model([("run_probe", "P_reward"), ("propose_correction", C.RC.CORRECTIONS["reward"]),
                   ("run_probe", C.RC.NEW_PROBE), ("propose_correction", C.RC.CORRECTIONS["reward"])])
    for _ in range(4):
        answer = advance(args, repo, model, backend)
    assert answer["status"] == "hold" and answer["events"][-1]["model_status"] == "invalid"
    assert len(backend.dispatches) == 3


def test_recovery_thresholds_cannot_be_changed_by_backend(tmp_path, monkeypatch):
    args, repo, backend = repair_setup(tmp_path, monkeypatch)
    model = Model([("run_probe", "P_reward"), ("propose_correction", C.RC.CORRECTIONS["reward"])])
    advance(args, repo, model, backend)

    def corrupt(envelope):
        envelope["result"]["bands"] = {**F.BANDS, "unregistered": 1}

    backend.corrupt_result = corrupt
    with pytest.raises(ValueError, match="registered recovery bands"):
        advance(args, repo, model, backend)


@pytest.mark.parametrize("choice", [("hold", None), ("run_probe", "P_unregistered")])
def test_model_hold_or_invalid_has_no_backend(tmp_path, choice):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=[])
    assert advance(args, repo, Model([choice]), backend)["status"] == "hold"
    assert not backend.approvals and not backend.dispatches


def test_unknown_receipt_stops_next_model(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["unknown"])
    model = Model([("run_probe", "P_episode")])
    assert advance(args, repo, model, backend)["status"] == "hold"
    assert advance(args, repo, no_model, backend)["status"] == "hold"
    assert len(model.calls) == 1


def test_dispatch_interruption_is_settled_by_lookup_after_new_process(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["normal"])
    backend.crash_dispatch = True
    with pytest.raises(KeyboardInterrupt):
        advance(args, repo, Model([("run_probe", "P_episode")]), backend)
    operation = backend.dispatches[0]
    mailbox = tmp_path / "backend-completed.json"
    save(mailbox, backend.jobs[operation])
    script = tmp_path / "resume_cpu.py"
    script.write_text('''import json, sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
import agentic_investigation_runtime_v1 as R
config=R.J._read(Path(sys.argv[3]))
def denied(*args):
    raise AssertionError("resume cannot approve/dispatch")
def lookup(operation, intent):
    R.J._write_new(Path(config["trace"]), {"operation_id":operation})
    return R.J._read(Path(config["mailbox"]))
r=R.Runtime(Path(config["folder"]), config["registration"], config["protocol"],
            config["anchor"],repo=Path(config["repo"]))
print(json.dumps(r.advance(host=R.HostCallbacks(denied,denied,lookup))))
''', encoding="utf-8")
    config = tmp_path / "resume-config.json"
    save(config, {"folder": str(args[0]), "registration": args[1], "protocol": args[2],
        "anchor": args[3], "repo": str(repo), "mailbox": str(mailbox), "trace": str(tmp_path / "lookup-trace.json")})
    child = subprocess.run([sys.executable, "-B", str(script), str(ROOT / "evals"), str(ROOT / "src"), str(config)],
                           capture_output=True, text=True, encoding="utf-8", check=False)
    (tmp_path / "resume-child.stdout.txt").write_text(child.stdout, encoding="utf-8")
    (tmp_path / "resume-child.stderr.txt").write_text(child.stderr, encoding="utf-8")
    assert child.returncode == 0, (f"resume child exit={child.returncode}\n"
                                   f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}")
    result = json.loads(child.stdout)
    assert result["events"][-1]["host_judgment"]["outcome"] == "normal"
    assert len(backend.approvals) == len(backend.dispatches) == 1
    assert J._read(tmp_path / "lookup-trace.json")["operation_id"] == operation


def test_interrupted_approval_uses_lookup_before_first_dispatch(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["normal"])
    original = backend.approve

    def interrupt(operation, intent):
        original(operation, intent)
        raise KeyboardInterrupt

    callbacks = R.HostCallbacks(interrupt, backend.dispatch, backend.lookup)
    with pytest.raises(KeyboardInterrupt):
        R.Runtime(*args, repo=repo).advance(complete=Model([("run_probe", "P_episode")]), host=callbacks)
    result = advance(args, repo, no_model, backend)
    assert result["events"][-1]["host_judgment"]["outcome"] == "normal"
    assert len(backend.approvals) == len(backend.dispatches) == len(backend.lookups) == 1


def test_pending_never_calls_model_or_resubmits_and_lookup_can_settle(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["normal"])
    backend.pending = True
    assert advance(args, repo, Model([("run_probe", "P_episode")]), backend)["status"] == "blocked_host"
    assert advance(args, repo, no_model, backend)["status"] == "blocked_host"
    operation = backend.dispatches[0]
    folder = args[0] / "steps/0001"
    backend.finish(operation, J._read(folder / "intent.json"), J._read(folder / "approval.json"))
    assert advance(args, repo, no_model, backend)["events"][-1]["host_judgment"]["outcome"] == "normal"
    assert len(backend.dispatches) == 1


def test_missing_approval_holds_before_dispatch(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=[])
    backend.missing_approval = True
    assert advance(args, repo, Model([("run_probe", "P_episode")]), backend)["status"] == "host_hold"
    assert not backend.dispatches


def test_host_can_be_injected_after_explicit_block_without_recalling_model(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    assert advance(args, repo, Model([("run_probe", "P_episode")]))["status"] == "blocked_host"
    backend = Backend(tmp_path, outcomes=["normal"])
    result = advance(args, repo, no_model, backend)
    assert result["events"][-1]["host_judgment"]["outcome"] == "normal"
    assert len(backend.approvals) == len(backend.dispatches) == 1


@pytest.mark.parametrize("bad", ["missing_receipt", "wrong_lineage", "wrong_cp", "raw_outcome", "mutated_seal"])
def test_tool_tampering_or_missing_receipt_fails_closed(tmp_path, bad):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["normal"])

    def corrupt(envelope):
        result = envelope["result"]
        if bad == "missing_receipt":
            Path(result["seal"]["path"]).unlink()
        elif bad == "wrong_lineage":
            result["next_snapshot"]["inputs"]["run_id"] = "other_case"
        elif bad == "wrong_cp":
            result["next_snapshot"]["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "9" * 64
        elif bad == "raw_outcome":
            result.clear()
            result["outcome"] = "normal"
        else:
            path = Path(result["seal"]["path"])
            path.write_bytes(path.read_bytes() + b" ")

    backend.corrupt_result = corrupt
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        advance(args, repo, Model([("run_probe", "P_episode")]), backend)
    assert len(backend.dispatches) == 1


def test_default_actual_guard_rejects_fixture_and_zero_baseline(tmp_path):
    snapshot = probe_source(tmp_path)
    args, repo = setup(tmp_path / "runtime", snapshot, fixture=False)
    with pytest.raises(ValueError, match="baseline"):
        R.Runtime(*args, repo=repo)
    api = tmp_path / "seeded-api"
    for index in range(7):
        context = R.A.prepare(snapshot, goal="CPU API prefix fixture")
        capsule = CALL.freeze(context, tmp_path / f"seed-{index}", source_anchor="a" * 40, protocol_sha256="b" * 64)
        CALL.advance(capsule, api, Model([("hold", None)]))
    args, repo = setup(tmp_path / "guard", snapshot, fixture=False, api=api)
    runtime = R.Runtime(*args, repo=repo)
    backend = Backend(tmp_path, outcomes=["normal"])
    with pytest.raises(ValueError, match="actual evidence"):
        runtime.advance(complete=Model([("run_probe", "P_episode")]), host=backend.callbacks())
    assert not backend.dispatches


def prefix_setup(tmp_path):
    snapshot = probe_source(tmp_path)
    context = R.A.prepare(snapshot, goal="CPU 조사", decisions_left=4)
    repo = tmp_path / "source-objects"
    old_protocol = {"schema": "cpu_prefix_fixture", "sources_lf":
        {"evals/agentic_investigation_runtime_v1.py": R.source_hash(ROOT / "evals/agentic_investigation_runtime_v1.py")}}
    files = {"evals/agentic_investigation_runtime_v1.py": (ROOT / "evals/agentic_investigation_runtime_v1.py").read_bytes(),
             "bench/protocols/original.json": J._bytes(old_protocol)}
    old_anchor = commit_objects(repo, files)
    old_ref = R.ref(repo / "bench/protocols/original.json")
    capsule = CALL.freeze(context, tmp_path / "original-call", source_anchor=old_anchor,
                          protocol_sha256=old_ref["sha256"])
    api = tmp_path / "api"
    CALL.advance(capsule, api, Model([("run_probe", "P_episode")]))
    args, repo = setup(tmp_path / "runtime", snapshot, decisions=4, api=api,
        prefix=[{"capsule": capsule, "protocol": old_ref}], repo=repo, parent=old_anchor, extra_files=files)
    return args, repo, capsule


def test_completed_prefix_replays_without_api_then_dispatches_same_probe(tmp_path):
    args, repo, _capsule = prefix_setup(tmp_path)
    backend = Backend(tmp_path, outcomes=["normal"])
    answer = advance(args, repo, None, backend)
    assert answer["events"][0]["imported_prefix"] is True
    assert answer["decisions_left"] == 3 and len(backend.dispatches) == 1
    assert len(list((tmp_path / "api/reservations").glob("*.json"))) == 1


@pytest.mark.parametrize("bad", ["budget", "policy", "checkpoint", "capsule", "terminal"])
def test_prefix_mutation_budget_policy_and_checkpoint_are_rejected(tmp_path, bad):
    args, repo, capsule = prefix_setup(tmp_path)
    runtime = R.Runtime(*args, repo=repo)
    if bad == "budget":
        runtime.reg["max_decisions"] = 5
    elif bad in {"policy", "checkpoint"}:
        initial = deepcopy(runtime.initial["snapshot"])
        inputs = initial["inputs"]
        if bad == "policy":
            inputs["policy"] = "fixed_eligible"
        else:
            inputs["packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "9" * 64
        changed = C.V.build_snapshot(**inputs)
        runtime.initial = {"snapshot": changed, "live_snapshot": changed}
    else:
        path = Path(capsule["path"]) if bad == "capsule" else Path(capsule["path"]).parent / "journal/terminal.json"
        raw = J._read(path)
        raw["schema"] = "changed"
        path.write_bytes(J._bytes(raw))
    with pytest.raises(ValueError):
        runtime.advance(complete=no_model)


def test_api_prefix_cannot_move_or_change_cap(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    runtime = R.Runtime(*args, repo=repo)
    runtime.reg["api"]["api_root"] = str(tmp_path / "reset-api")
    with pytest.raises(ValueError, match="binding|reset|move"):
        runtime.advance(complete=no_model)


def test_shared_api12_exhaustion_is_terminal_and_replays_without_callback(tmp_path):
    snapshot = probe_source(tmp_path)
    api = tmp_path / "api"
    for index in range(12):
        context = R.A.prepare(snapshot, goal="CPU API cap fixture")
        capsule = CALL.freeze(context, tmp_path / f"seed-{index}", source_anchor="a" * 40, protocol_sha256="b" * 64)
        CALL.advance(capsule, api, Model([("hold", None)]))
    args, repo = setup(tmp_path / "runtime", snapshot, api=api)
    result = advance(args, repo, no_model)
    assert result["status"] == "hold"
    assert result["events"][-1]["model_status"] == "unavailable"
    assert advance(args, repo, no_model) == result
    assert len(list((api / "reservations").glob("*.json"))) == 12


def test_new_checkpoint_rejects_old_evidence_registration(case, tmp_path, monkeypatch):  # noqa: F811
    old_bridge = adapter_for_case(case, tmp_path)
    snapshot, spec = old_bridge
    bindings = snapshot["inputs"]["packet"]["input_bindings"]
    monkeypatch.setattr(F, "BINDINGS", bindings)
    files = case["entry"]["training"]
    current = F.artifact(tmp_path, "case", POLICY, 0, [],
                         R.B.E._ref(files["case"]["checkpoint"], case["roots"]).read_bytes())
    reference = F.artifact(tmp_path, "NONE", "shared_reference", 0, [],
                           R.B.E._ref(files["reference"]["checkpoint"], case["roots"]).read_bytes())
    args, repo = setup(tmp_path / "runtime", snapshot, evidence=spec)
    backend = Backend(tmp_path, outcomes=["abnormal"], current=current, reference=reference)
    model = Model([("run_probe", "P_reward"), ("propose_correction", C.RC.CORRECTIONS["reward"])])
    advance(args, repo, model, backend)

    def stale(envelope):
        envelope["result"]["evidence"] = spec

    backend.corrupt_result = stale
    with pytest.raises(ValueError, match="stale session"):
        advance(args, repo, model, backend)
    assert len(backend.dispatches) == 2


def adapter_for_case(case_data, root):
    inputs = deepcopy(case_data["snapshot"]["inputs"])
    inputs["run_id"] = "controller_case"
    inputs["ledger_path"] = root / "case/epoch1/probes.ledger.jsonl"
    snapshot = C.V.build_snapshot(**inputs)
    reg = R.B.register(case_data["index"], case_data["roots"], snapshot,
        expected_index_sha256=R.B.E.index_sha256(case_data["index"]),
        source_lineage_id=case_data["registration"]["session"]["source_lineage_id"],
        aliases=case_data["registration"]["aliases"])
    bridge = R.B.EvidenceAdapter(case_data["index"], case_data["roots"], reg,
                                 expected_registration_sha256=R.B.E.index_sha256(reg))
    return snapshot, R.evidence_spec(bridge)


def test_recorded_request_mutation_rejected_on_restart(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    advance(args, repo, Model([("hold", None)]))
    path = args[0] / "steps/0001/request.json"
    raw = J._read(path)
    raw["messages"][0]["content"] = "changed"
    path.write_bytes(J._bytes(raw))
    with pytest.raises(ValueError, match="immutable"):
        advance(args, repo, no_model)


def test_host_event_mutation_is_detected_on_restart(tmp_path):
    args, repo = setup(tmp_path / "runtime", probe_source(tmp_path))
    backend = Backend(tmp_path, outcomes=["normal"])
    advance(args, repo, Model([("run_probe", "P_episode")]), backend)
    path = args[0] / "steps/0001/host-events/0002.json"
    raw = J._read(path)
    raw["response"]["status"] = "unknown"
    path.write_bytes(J._bytes(raw))
    with pytest.raises(ValueError, match="event/result"):
        advance(args, repo, no_model, backend)
