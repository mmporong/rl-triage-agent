import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import vision_corrected_native_handoff_v2 as H

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
ANCHOR = "a" * 40


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_text(str(value), encoding="utf-8")
    return path


class Evidence:
    def __init__(self, repo):
        self.repo = repo

    def file(self, commit, relative):
        assert commit == ANCHOR and relative in {H.BRIDGE_RELATIVE, H.RUNNER_RELATIVE}
        return "blob", (self.repo / relative).read_bytes()


class FakeRuntime:
    calls: ClassVar[list] = []

    def __init__(self, repo, plan, context, scope, require_active):
        self.path = repo / ".triage-runtime/vision-selection-e2e/demo"
        self.path.mkdir(parents=True, exist_ok=True)
        self.correction = repo / "correction.json"
        self.approval = repo / "approval.json"
        FakeRuntime.calls.append((plan, scope, require_active))

    def recording_bindings(self):
        FakeRuntime.calls.append("recording_bindings")
        return {"corrected-s2220": {"correction_result": self.correction,
                                     "approval_receipt": self.approval}}


class FakeRunner:
    def __init__(self, protocol, binding):
        self.protocol = protocol
        self.binding = binding
        self.calls = []

    def validate_protocol(self, path):
        self.calls.append("validate_protocol")
        return self.protocol

    def _verified_binding(self, logical, correction, approval):
        self.calls.append(("binding", logical, correction, approval))
        return self.binding

    def build_resource_proof(self, protocol, **kwargs):
        self.calls.append(("build_resource_proof", protocol, kwargs))
        return {"schema": "fresh-proof", "observed_at_utc": kwargs["observed_at_utc"]}

    def validate_resource_proof(self, *args, **kwargs):
        self.calls.append(("validate_resource_proof", args, kwargs))
        return {"status": "granted"}


class FakeService:
    calls: ClassVar[list] = []
    fail = False

    def __init__(self, protocol):
        self.protocol = protocol

    def record(self, **kwargs):
        FakeService.calls.append(kwargs)
        if FakeService.fail:
            raise RuntimeError("record failed")
        return {"status": "complete", "job": str(kwargs["stage"] / "job.json"),
                "native_receipt": {"path": str(kwargs["receipt_output"] / "receipt.json"),
                                   "sha256": "b" * 64, "bytes": 1}}


def fixture(tmp_path, monkeypatch):
    repo = tmp_path.resolve()
    for relative, source in ((H.BRIDGE_RELATIVE, Path(H.__file__)),
                             (H.RUNNER_RELATIVE, Path(H.RUNNER2.__file__))):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    source_map = {relative: H.E.ADAPTER.file_sha(repo / relative, lf=True)
                  for relative in (H.BRIDGE_RELATIVE, H.RUNNER_RELATIVE)}
    correction = _write(repo / "correction.json", {"result": "recovered"})
    approval = _write(repo / "approval.json", {"approved": True})
    binding = {
        "logical_id": "corrected-s2220",
        "checkpoint": {"path": str(repo / "checkpoint.pt"), "sha256": "c" * 64},
        "correction_result": H._ref(correction),
        "approval_receipt": H._ref(approval),
    }
    protocol = {
        "source_sha256_lf": source_map,
        "source_commit": ANCHOR,
        "checkpoints": [binding],
    }
    paths = {}
    for name in ("plan", "scope", "execution", "lease"):
        paths[name] = _write(repo / f"{name}.json", {"name": name})
    paths["protocol"] = _write(repo / "bound.json", protocol)
    template = _write(repo / "template.json", {"template": True})
    registration = _write(repo / "registration-ref.json", {"path": "bench/protocols/seal.registration.json",
                                                               "sha256": "d" * 64})
    context = {"protocol": {"execution_bindings": {"corrected_render_plan": {
        "path": template.relative_to(repo).as_posix(), "sha256": H.E.ADAPTER.file_sha(template)}}}}
    runner = FakeRunner(protocol, binding)
    monkeypatch.setattr(H, "_context", lambda *args: context)
    monkeypatch.setattr(H.E, "native_components", lambda: (runner, SimpleNamespace(
        NativeCorrectedRecordService=FakeService)))
    monkeypatch.setattr(H.E, "verify_native_registration", lambda *args: {"verified": True})
    monkeypatch.setattr(H.E.ADAPTER, "verify_record_service", lambda *args: None)
    old_root = repo / ".triage-runtime/vision-selection-e2e/demo"
    old_stage = old_root / "native-stage/source-01"
    old_stage.mkdir(parents=True)
    old_intent = _write(old_root / "intents/native-source-01.json", {"old": "intent"})
    old_failure = _write(old_root / "failures/native-source-01.json", {"old": "failure"})
    ticket = repo / "ticket.json"
    value = H.prepare(repo, paths["plan"], paths["scope"], paths["execution"],
                      H.E.ADAPTER.file_sha(paths["execution"]), paths["lease"],
                      H.E.ADAPTER.file_sha(paths["lease"]), paths["protocol"], registration,
                      ANCHOR, "source-01", ticket, now=NOW, runtime_factory=FakeRuntime,
                      evidence_factory=Evidence)
    return SimpleNamespace(repo=repo, paths=paths, protocol=protocol, runner=runner,
                           binding=binding, ticket=ticket, value=value,
                           old_stage=old_stage, old_intent=old_intent, old_failure=old_failure)


def test_prepare_runs_static_runtime_bindings_registration_and_writes_120s_ticket(tmp_path, monkeypatch):
    FakeRuntime.calls = []
    fx = fixture(tmp_path, monkeypatch)
    assert FakeRuntime.calls[-1] == "recording_bindings"
    assert fx.value["ttl_s"] == 120
    assert datetime.fromisoformat(fx.value["expires_at_utc"]) - datetime.fromisoformat(
        fx.value["created_at_utc"]) == timedelta(seconds=120)
    assert fx.value["bridge_source"] == {"source_commit": ANCHOR,
        "files_sha256_lf": fx.protocol["source_sha256_lf"]}
    assert Path(fx.value["expected"]["cache"]).parts[-2:] == ("native-record", "source-01.json")
    assert Path(fx.value["expected"]["stage"]).parts[-2:] == ("native-stage-v2", "source-01")
    assert Path(fx.value["expected"]["receipt_output"]).parts[-2:] == ("native-receipt-v2", "source-01")
    assert fx.old_stage.is_dir() and H._read(fx.old_intent) == {"old": "intent"}
    assert H._read(fx.old_failure) == {"old": "failure"}
    assert ("binding", "corrected-s2220", fx.repo / "correction.json",
            fx.repo / "approval.json") in fx.runner.calls


def test_record_builds_fresh_proof_before_intent_and_writes_cache_only_after_recheck(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    FakeService.calls, FakeService.fail = [], False
    provider = _write(fx.repo / "provider.json", {})
    gpu = _write(fx.repo / "gpu.csv", "GPU-test\n")
    processes = _write(fx.repo / "process.csv", "")
    checked = []
    monkeypatch.setattr(H.E, "recheck_native_result", lambda *args: checked.append(args) or {"ok": True})
    result = H.record(fx.repo, fx.ticket, H.E.ADAPTER.file_sha(fx.ticket), provider, gpu, processes, NOW.isoformat(),
                      (NOW + timedelta(seconds=25)).isoformat(), fx.repo / "proof.json",
                      fx.repo, Path(sys.executable), now=NOW + timedelta(seconds=1),
                      evidence_factory=Evidence,
                      components=lambda: (fx.runner, SimpleNamespace(NativeCorrectedRecordService=FakeService)))
    assert result["status"] == "complete" and len(checked) == 1 and len(FakeService.calls) == 1
    assert FakeService.calls[0]["run"] is H.RUNNER2.run_job
    intent = Path(fx.value["expected"]["intent"])
    cache = Path(fx.value["expected"]["cache"])
    assert intent.is_file() and cache.is_file()
    assert H._read(cache) == result
    assert fx.runner.calls[-1][0] == "build_resource_proof"


@pytest.mark.parametrize("mutation", ["expired", "input", "closure", "approval"])
def test_record_rejects_stale_or_changed_ticket_inputs_before_service(tmp_path, monkeypatch, mutation):
    fx = fixture(tmp_path, monkeypatch)
    FakeService.calls, FakeService.fail = [], False
    provider = _write(fx.repo / "provider.json", {})
    gpu = _write(fx.repo / "gpu.csv", "GPU-test\n")
    processes = _write(fx.repo / "process.csv", "")
    current = NOW + timedelta(seconds=1)
    if mutation == "expired":
        current = NOW + timedelta(seconds=121)
    elif mutation == "input":
        fx.paths["plan"].write_text("changed", encoding="utf-8")
    elif mutation == "closure":
        fx.protocol["source_sha256_lf"][H.BRIDGE_RELATIVE] = "0" * 64
        fx.paths["protocol"].write_text(json.dumps(fx.protocol), encoding="utf-8")
        ticket = H._read(fx.ticket)
        ticket["inputs"]["bound_protocol"]["sha256"] = H.E.ADAPTER.file_sha(fx.paths["protocol"])
        fx.ticket.write_text(json.dumps(ticket), encoding="utf-8")
    else:
        (fx.repo / "approval.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        H.record(fx.repo, fx.ticket, H.E.ADAPTER.file_sha(fx.ticket), provider, gpu, processes, NOW.isoformat(),
                 (NOW + timedelta(seconds=25)).isoformat(), fx.repo / "proof.json",
                 fx.repo, Path(sys.executable), now=current, evidence_factory=Evidence,
                 components=lambda: (fx.runner, SimpleNamespace(NativeCorrectedRecordService=FakeService)))
    assert FakeService.calls == []


def test_failed_record_preserves_intent_and_blocks_automatic_rerun(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    FakeService.calls, FakeService.fail = [], True
    provider = _write(fx.repo / "provider.json", {})
    gpu = _write(fx.repo / "gpu.csv", "GPU-test\n")
    processes = _write(fx.repo / "process.csv", "")
    kwargs = {"repo": fx.repo, "ticket_path": fx.ticket,
              "ticket_sha256": H.E.ADAPTER.file_sha(fx.ticket),
              "provider_inventory": provider, "gpu_csv": gpu,
              "process_csv": processes, "observed_at_utc": NOW.isoformat(),
              "expires_at_utc": (NOW + timedelta(seconds=25)).isoformat(),
              "resource_proof_output": fx.repo / "proof.json", "isaaclab": fx.repo,
              "python": Path(sys.executable), "now": NOW + timedelta(seconds=1),
              "evidence_factory": Evidence,
              "components": lambda: (fx.runner, SimpleNamespace(NativeCorrectedRecordService=FakeService))}
    with pytest.raises(RuntimeError, match="record failed"):
        H.record(**kwargs)
    assert Path(fx.value["expected"]["intent"]).is_file()
    assert Path(fx.value["expected"]["failure"]).is_file()
    assert not Path(fx.value["expected"]["cache"]).exists()
    with pytest.raises((ValueError, FileExistsError)):
        H.record(**{**kwargs, "resource_proof_output": fx.repo / "proof-2.json"})
    assert len(FakeService.calls) == 1


def test_ticket_byte_tamper_is_rejected_before_proof_intent_cache_or_service(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    original_sha = H.E.ADAPTER.file_sha(fx.ticket)
    ticket = H._read(fx.ticket)
    ticket["expires_at_utc"] = (NOW + timedelta(seconds=119)).isoformat()
    fx.ticket.write_text(json.dumps(ticket), encoding="utf-8")
    provider = _write(fx.repo / "provider.json", {})
    gpu = _write(fx.repo / "gpu.csv", "GPU-test\n")
    processes = _write(fx.repo / "process.csv", "")
    FakeService.calls, FakeService.fail = [], False
    before = list(fx.runner.calls)
    with pytest.raises(ValueError, match="prepare 영수증"):
        H.record(fx.repo, fx.ticket, original_sha, provider, gpu, processes, NOW.isoformat(),
                 (NOW + timedelta(seconds=25)).isoformat(), fx.repo / "proof.json",
                 fx.repo, Path(sys.executable), now=NOW + timedelta(seconds=1),
                 evidence_factory=Evidence,
                 components=lambda: (fx.runner, SimpleNamespace(NativeCorrectedRecordService=FakeService)))
    assert fx.runner.calls == before and FakeService.calls == []
    assert not (fx.repo / "proof.json").exists()
    assert not Path(fx.value["expected"]["intent"]).exists()
    assert not Path(fx.value["expected"]["cache"]).exists()


def test_prepare_is_write_once(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        H.prepare(fx.repo, fx.paths["plan"], fx.paths["scope"], fx.paths["execution"],
                  H.E.ADAPTER.file_sha(fx.paths["execution"]), fx.paths["lease"],
                  H.E.ADAPTER.file_sha(fx.paths["lease"]), fx.paths["protocol"],
                  fx.repo / "registration-ref.json", ANCHOR, "source-01", fx.ticket,
                  now=NOW, runtime_factory=FakeRuntime, evidence_factory=Evidence)
