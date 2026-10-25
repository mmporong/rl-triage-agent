from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import test_vision_play_worker_v3 as PRIOR

W = importlib.import_module("vision_play_worker_v4")
R = W.R


def test_atomic_supervisor_owns_its_child_entry_and_waits_before_sealing(
    tmp_path, monkeypatch
):
    evidence = importlib.import_module("vision_execution_evidence_v2")
    job, validated = PRIOR._validated(tmp_path)
    monkeypatch.setattr(R, "validate_job", lambda _path: validated)
    observed = {}

    def popen(argv, **kwargs):
        observed.update(argv=argv, kwargs=kwargs)
        process = PRIOR.Process(lambda: PRIOR._write_success(validated))
        observed["process"] = process
        return process

    def seal(job_path, result_path, *, sdk_child_exit_code):
        assert observed["process"].waited and sdk_child_exit_code == 0
        assert (
            job_path == job
            and result_path == validated["output_directory"] / "result.json"
        )
        observed["sealed"] = True
        return job.parent / "job.execution-evidence.json"

    monkeypatch.setattr(evidence, "write_manifest", seal)
    assert W.supervise(job, popen=popen) == 0
    assert observed["sealed"]
    assert W.__file__ is not None
    assert observed["argv"] == [
        sys.executable,
        "-B",
        str(Path(W.__file__).resolve()),
        "--sdk-child",
        "--job",
        str(job.resolve()),
    ]
    assert observed["kwargs"] == {"shell": False, "start_new_session": False}


@pytest.mark.parametrize(
    "case", ["missing", "nonzero", "job_tamper", "atomic_evidence_missing"]
)
def test_atomic_supervisor_preserves_failure_even_if_sdk_exits_zero(
    tmp_path, monkeypatch, case
):
    job, validated = PRIOR._validated(tmp_path)
    monkeypatch.setattr(R, "validate_job", lambda _path: validated)

    def action():
        if case == "atomic_evidence_missing":
            PRIOR._write_success(validated)
        elif case == "job_tamper":
            job.write_text('{"changed":true}', encoding="utf-8")

    assert (
        W.supervise(
            job,
            popen=lambda *_a, **_k: PRIOR.Process(
                action, 7 if case == "nonzero" else 0
            ),
        )
        == 1
    )
    assert not (job.parent / "job.execution-evidence.json").exists()
    failure = R.read_object(validated["output_directory"] / "supervisor_failure.json")
    assert failure["child_exit_code"] == (7 if case == "nonzero" else 0)


@pytest.mark.parametrize("mode", [R.RECORD_MODE, R.RENDER_SYNC_SMOKE_MODE])
def test_atomic_runner_routes_factory_and_smoke_collector(tmp_path, monkeypatch, mode):
    capture = importlib.import_module("vision_atomic_capture_v1")
    smoke = importlib.import_module("vision_atomic_smoke_v1")
    worker = importlib.import_module("vision_play_worker_v1")
    job, validated = PRIOR._validated(tmp_path)
    validated["job"]["mode"] = mode
    factory, app = object(), object()
    monkeypatch.setattr(
        capture,
        "make_capture_factory",
        lambda v, a: factory if (v is validated and a is app) else None,
    )
    observed = {}
    marker = ({"artifact": tmp_path / "artifact"}, tmp_path / "primary")

    def record(v, a, **kwargs):
        assert mode == R.RECORD_MODE and v is validated and a is app
        assert kwargs == {"camera_capture_factory": factory}
        return marker

    def smoke_run(v, a, **kwargs):
        assert mode == R.RENDER_SYNC_SMOKE_MODE and v is validated and a is app
        assert kwargs["camera_capture_factory"] is factory
        token = object()
        kwargs["smoke_collector"](token)
        assert observed == {"capture": token, "job": job}
        return marker

    monkeypatch.setattr(worker, "run_real", record)
    monkeypatch.setattr(worker, "run_render_sync_smoke", smoke_run)
    monkeypatch.setattr(
        smoke, "collect", lambda c, j: observed.update(capture=c, job=j)
    )
    assert W.run_capture(validated, app) == marker
