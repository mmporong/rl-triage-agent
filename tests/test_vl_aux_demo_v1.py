from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evals import vl_aux_demo_v1 as D


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64


def test_configured_sets_expected_context_and_restores_every_identity():
    names = ("VARIANTS", "PRIVATE", "PROTOCOL_DIR", "CODE", "fixed", "PREDICTIONS",
             "dynamic_fields", "correction_commands")
    before = {name: getattr(D.E, name) for name in names}
    original_p = D.DC.P
    with D.configured():
        assert D.E.VARIANTS == {"f1": {"fault": "F1", "case_id": "h02", "mechanism": "reward", "seed": 2080}}
        assert D.E.PRIVATE == D.PRIVATE and D.E.PROTOCOL_DIR == D.PROTOCOL_DIR
        assert isinstance(D.DC.P, D.OwnedProxy) and D.DC.P.original is original_p
    assert all(getattr(D.E, name) is value for name, value in before.items())
    assert D.DC.P is original_p and D._ACTIVE is False


def test_configured_restores_after_exception_and_rejects_nesting():
    original_fixed, original_p = D.E.fixed, D.DC.P
    with pytest.raises(RuntimeError, match="boom"):
        with D.configured():
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with D.configured():
                    pass
            raise RuntimeError("boom")
    assert D.E.fixed is original_fixed and D.DC.P is original_p and D._ACTIVE is False


def test_owned_proxy_only_replaces_training_child_entrypoint(tmp_path):
    calls = []

    class Original:
        marker = "preserved"

        def run_owned(self, folder, argv, cwd, timeout):
            calls.append(list(argv))
            return {"argv": list(argv)}

    proxy = D.OwnedProxy(Original())
    python = tmp_path / "python.bat"
    training = [str(python), str(D.ROOT / "bench/demo_correction_v4.py"), "--manifest", "m.json"]
    fixed_eval = [str(python), str(D.ROOT / "evals/fixed_eval.py"), "--protocol", "p.json"]
    assert proxy.run_owned(tmp_path, training, tmp_path, 1)["argv"][1] == str(
        D.ROOT / "bench/demo_correction_vl_v1.py")
    assert proxy.run_owned(tmp_path, fixed_eval, tmp_path, 1)["argv"] == fixed_eval
    assert proxy.marker == "preserved"


def _observe_inputs(tmp_path: Path):
    report = tmp_path / "fixed.json"
    report.write_text('{"checkpoint":{"sha256":"' + SHA_A + '"}}', encoding="utf-8")
    artifacts = {
        "checkpoint": {"path": str(tmp_path / "model.pt"), "sha256": SHA_A},
        "environment": {"path": str(tmp_path / "params/env.yaml"), "sha256": "c" * 64},
        "agent": {"path": str(tmp_path / "params/agent.yaml"), "sha256": "d" * 64},
        "ignored": {"path": "private", "sha256": "e" * 64},
    }
    protocol = {"auxiliary": {"recording": {"schema": "recording"}}}
    return artifacts, report, protocol


def test_observe_sends_only_two_sha_bindings_and_api_failure_is_nonfatal(tmp_path, monkeypatch):
    artifacts, report, protocol = _observe_inputs(tmp_path)
    video, manifest = tmp_path / "clip.mp4", tmp_path / "manifest.json"
    video.write_bytes(b"video")
    manifest.write_text("{}", encoding="utf-8")
    seen = {}

    def record_clip(selected, fixed_report, output, isaaclab, recording):
        seen["selected"] = selected
        seen["recording"] = recording
        return {"video_path": str(video), "manifest_path": str(manifest)}

    def prepare(source, output, binding):
        seen["binding"] = binding
        packet = tmp_path / "packet.json"
        packet.write_text("{}", encoding="utf-8")
        return packet

    monkeypatch.setattr(D.MEDIA, "record_clip", record_clip)
    monkeypatch.setattr(D.OBS, "prepare", prepare)
    monkeypatch.setattr(D.OBS, "invoke", lambda *_: (_ for _ in ()).throw(ConnectionError("private detail")))
    result = D.observe("opaque1", artifacts, report, tmp_path / "run", tmp_path, protocol,
                       arm="FAULT", label="unhealthy")
    assert result["status"] == "observer_error" and result["error_type"] == "ConnectionError"
    assert seen["binding"] == {"checkpoint_sha256": SHA_A, "fixed_eval_sha256": D.E.sha(report)}
    assert set(seen["selected"]) == {"checkpoint", "environment", "agent"}
    assert seen["recording"]["fixed_evaluation_label"] == "unhealthy"
    assert "binding" not in json.dumps(seen.get("recording"))
    assert json.loads((tmp_path / "run/auxiliary/opaque1/result.json").read_text())["status"] == "observer_error"


def test_observe_recording_failure_stops_and_records_error(tmp_path, monkeypatch):
    artifacts, report, protocol = _observe_inputs(tmp_path)
    invoked = []
    monkeypatch.setattr(D.MEDIA, "record_clip", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gpu cleanup")))
    monkeypatch.setattr(D.OBS, "invoke", lambda *_: invoked.append(True))
    with pytest.raises(RuntimeError, match="gpu cleanup"):
        D.observe("opaque2", artifacts, report, tmp_path / "run", tmp_path, protocol,
                  arm="NONE", label="healthy")
    saved = json.loads((tmp_path / "run/auxiliary/opaque2/result.json").read_text())
    assert saved["status"] == "recording_error" and saved["error_type"] == "RuntimeError"
    assert invoked == []


def _pipeline_run(tmp_path: Path, monkeypatch, observation_status: str, arms=("restart",)):
    events = []
    folder = tmp_path / observation_status
    folder.mkdir()

    def event(name, result=None):
        def called(*args, **kwargs):
            events.append((name, args, kwargs))
            return result
        return called

    reports = {arm: {"path": f"reports/{arm}.json"} for arm in ("NONE", "FAULT")}
    evaluated = {"reports": reports, "gate": {"none_walks": True,
                                                "fault_verdict": {"label": "unhealthy"}}}
    fix = {"arms": list(arms), "treatment": arms[-1], "verdict": {"label": "healthy"},
           "reference": {"against_original_none": {"label": "healthy"}}}
    fake_e = SimpleNamespace(
        begin=event("begin"), folder_for=lambda _tag: folder, train=event("train"), alarm=event("alarm"),
        evaluate=event("evaluate", evaluated), _check_artifacts=lambda tag, arm: {"arm": arm},
        read=lambda _path: {"protocol": True}, reference=event("reference"), packet=event("packet"),
        loop_start=event("loop_start", {"status": "started"}),
        ctx_for=lambda _tag: SimpleNamespace(cases={"FAULT": "fault-case"}), LOOP_BUDGET=2,
        loop_close=event("loop_close"), specificity=event("specificity"),
        correction_evidence=event("correction_evidence"),
        correction_propose=event("correction_propose", {"request_id": "correction-1"}),
        approve_correction=event("approve_correction"),
        correction=event("correction", {"result": fix}),
        summarize=event("summarize", {"outcome": "recovered", "demo_success": True}),
    )
    loads = iter([
        (None, {"status": "open", "pending": "probe-1"}, SimpleNamespace(approve=event("probe_approve"))),
        (None, {"status": "closed", "pending": None}, SimpleNamespace(approve=event("unused"))),
    ])
    fake_loop = SimpleNamespace(
        _load=lambda _case: next(loads), run=event("probe_run"),
        isaac_execute=lambda *_args, **_kwargs: None, isaac_preflight=lambda *_args, **_kwargs: None,
    )

    def fake_observe(clip_id, artifacts, report, run_folder, isaaclab, protocol, *, arm, label):
        events.append(("observe", (clip_id, arm, label), {}))
        return {"status": observation_status, "secret_marker": observation_status}

    def fake_decision(kind, packet, output):
        events.append(("decision_call", (kind, packet.name, output.name), {}))
        return {"requests": 1}

    monkeypatch.setattr(D, "E", fake_e)
    monkeypatch.setattr(D, "LOOP", fake_loop)
    monkeypatch.setattr(D, "observe", fake_observe)
    monkeypatch.setattr(D, "decision_call", fake_decision)
    monkeypatch.setattr(D, "_corrected_artifacts", lambda _folder, arm: {"arm": arm})
    result = D.pipeline("tag", tmp_path, "operator")
    return result, events


@pytest.mark.parametrize("arms", [("restart_corrected",), ("clean_resume_reference", "resume_corrected")])
def test_pipeline_keeps_auxiliary_results_out_of_base_decision_and_runs_expected_order(tmp_path, monkeypatch, arms):
    valid_result, valid = _pipeline_run(tmp_path, monkeypatch, "valid", arms=arms)
    invalid_result, invalid = _pipeline_run(tmp_path, monkeypatch, "invalid", arms=arms)
    assert valid_result == invalid_result == {"outcome": "recovered", "demo_success": True}
    valid_base = [(name, args, kwargs) for name, args, kwargs in valid if name != "observe"]
    invalid_base = [(name, args, kwargs) for name, args, kwargs in invalid if name != "observe"]
    def normalized(events):
        def convert(value):
            if isinstance(value, Path):
                return value.name
            if callable(value):
                return "<callable>"
            return str(value)
        return [(name, json.loads(json.dumps([args, kwargs], default=convert)))
                for name, args, kwargs in events]
    assert normalized(valid_base) == normalized(invalid_base)
    names = [name for name, _, _ in valid]
    assert names[:7] == ["begin", "train", "train", "alarm", "evaluate", "observe", "observe"]
    assert names.index("decision_call") > names.index("observe")
    assert names.index("probe_run") > names.index("loop_start")
    assert names.index("correction") > names.index("correction_propose")
    assert names[-len(arms):] == ["observe"] * len(arms)
    assert sum(name == "observe" for name in names) == 2 + len(arms)
    assert all("secret_marker" not in json.dumps([args, kwargs], default=str)
               for name, args, kwargs in valid_base)


def test_configured_source_binding_covers_new_runtime_dependency_closure():
    required = {
        "evals/vl_aux_demo_v1.py", "evals/vl_aux_observer_v1.py", "evals/vl_aux_worker_v1.py",
        "evals/feedback_model_call.py",
        "evals/nemotron_bottleneck_vision_v2.py", "evals/nemotron_bottleneck_ledger_v2.py",
        "evals/nemotron_endpoint_ledger_v3.py", "bench/demo_correction_vl_v1.py",
        "bench/record_vl_aux_play_v1.py", "bench/record_evidence_media.py",
        "bench/run_vl_aux_demo_v1.ps1", "tests/test_vl_aux_demo_v1.py",
        "tests/test_vl_aux_observer_v1.py",
    }
    with D.configured():
        assert required <= D.E.CODE
        assert D.B4_RESULT in D.E.CODE and D.B4_PROTOCOL in D.E.CODE


def test_correction_child_help_is_cpu_only_and_importable():
    completed = subprocess.run([sys.executable, str(ROOT / "bench/demo_correction_vl_v1.py"), "--help"],
                               cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    assert completed.returncode == 0
    assert "--manifest" in completed.stdout


def test_nested_auxiliary_output_blocks_a_new_start(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "nested")
    with D.configured():
        (D.PRIVATE / D.TAG / "auxiliary_api/reservations").mkdir(parents=True)
        with pytest.raises(FileExistsError, match="sidecar"):
            D.output_preflight(D.TAG)
