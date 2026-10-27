from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_vision_transferred_reuse_v1 as R

PROTOCOL = ROOT / "bench/protocols/vision_transferred_pull_record_20261027c_v1.json"


def _admission() -> dict:
    target = dict(R.TARGET)
    return {
        "schema": "vision_transferred_reuse_admission_v1",
        "inventory_before_start": {"target": target, "status": "STOPPED", "observed_at_utc": "2026-10-27T12:00:00+00:00"},
        "fresh_quote": {"provider": "nebius", "instance_type": R.TARGET["instance_type"], "usd_per_h": "2.09616", "observed_at_utc": "2026-10-27T12:00:10+00:00"},
        "start": {"target": target, "started_at_utc": "2026-10-27T12:00:20+00:00", "receipt_sha256": "a" * 64},
        "ready_observations": [
            {"target": target, "status": "RUNNING", "build_status": "COMPLETED", "shell_status": "READY", "health": "HEALTHY", "observed_at_utc": "2026-10-27T12:01:00+00:00"},
            {"target": target, "status": "RUNNING", "build_status": "COMPLETED", "shell_status": "READY", "health": "HEALTHY", "observed_at_utc": "2026-10-27T12:01:10+00:00"},
        ],
        "environment": {"status": "match", "reuse_allowed": True, "interpreter": R.REMOTE_PYTHON, "isaaclab": R.REMOTE_LAB, "versions": dict(R.EXPECTED_ENVIRONMENT)},
        "gpu": {"uuid": "GPU-fixture", "idle": True, "compute_processes": []},
        "owned_processes": [],
        "budget": {"prior_upper_estimate_usd": "24.454911", "followup_max_usd": "5.000000", "program_cap_usd": "35.000000"},
        "pending_intent": None,
    }


def _protocol() -> dict:
    return R.validate_protocol(PROTOCOL, repo=ROOT, now=datetime(2026, 10, 27, 12, tzinfo=timezone.utc))


def test_protocol_and_backup_are_seed2221_bound(tmp_path: Path) -> None:
    protocol = _protocol()
    for row in protocol["checkpoints"]:
        path = tmp_path / row["isaaclab_relative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # This test exercises verification mechanics without claiming the fixture is the real checkpoint.
        path.write_bytes(row["logical_id"].encode())
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert [row["logical_id"] for row in R.verify_checkpoint_backup(protocol, tmp_path)] == list(R.CHECKPOINT_IDS)


def test_admission_accepts_new_start_two_ready_and_idle() -> None:
    result = R.validate_admission(_admission(), _protocol())
    assert result["ready_at_utc"] == "2026-10-27T12:01:10+00:00"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda x: x["start"].update(started_at_utc="2026-10-27T08:19:49+00:00"), "새 Start"),
        (lambda x: x["fresh_quote"].update(usd_per_h="2.20"), "fresh quote"),
        (lambda x: x["ready_observations"].pop(), "연속 두 건"),
        (lambda x: x["environment"]["versions"].update(torch="wrong"), "환경 버전"),
        (lambda x: x["gpu"].update(idle=False), "idle"),
        (lambda x: x["owned_processes"].append({"pid": 7}), "소유 프로세스"),
        (lambda x: x["budget"].update(program_cap_usd="29.00"), "budget"),
        (lambda x: x.update(pending_intent={"phase": "record_none"}), "자동 반복"),
    ],
)
def test_admission_fail_closed(mutation, message: str) -> None:
    admission = _admission()
    mutation(admission)
    with pytest.raises(ValueError, match=message):
        R.validate_admission(admission, _protocol())


def test_plan_stops_after_retrieval_and_has_no_post_stop_exec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(_admission()), encoding="utf-8")
    monkeypatch.setattr(R, "verify_checkpoint_backup", lambda *_: [{"logical_id": value} for value in R.CHECKPOINT_IDS])
    plan = R.build_plan(PROTOCOL, admission_path, tmp_path, repo=ROOT)
    assert [row["phase"] for row in plan["steps"]] == ["record_none", "record_f1", "retrieve", "stop", "observe_stopped"]
    assert plan["steps"][1]["only_after"] == "record_none_completed"
    assert plan["no_exec_after_stopped"] is True
    assert plan["unresolved_intent_policy"] == "hold_without_automatic_replay"
    assert all("run_vision_transferred_reuse_v1.py record" in " ".join(row["command"])
               for row in plan["steps"][:2])


def test_terminal_requires_retrieval_before_stop_and_rejects_exec_after_stopped() -> None:
    terminal = {
        "schema": "vision_transferred_reuse_terminal_v1",
        "target": dict(R.TARGET),
        "retrieval": {"manifest_sha256": "b" * 64, "completed_at_utc": "2026-10-27T12:20:00+00:00"},
        "stop": {"command": ["brev", "stop", R.TARGET["resource_name"]],
                 "receipt_sha256": "c" * 64, "requested_at_utc": "2026-10-27T12:20:10+00:00"},
        "stopped": {"target": dict(R.TARGET), "status": "STOPPED",
                    "observed_at_utc": "2026-10-27T12:21:00+00:00"},
        "actions": [{"kind": "remote_exec", "at_utc": "2026-10-27T12:19:00+00:00"}],
    }
    assert R.validate_terminal(terminal)["retrieval_manifest_sha256"] == "b" * 64
    invalid = copy.deepcopy(terminal)
    invalid["actions"].append({"kind": "remote_exec", "at_utc": "2026-10-27T12:21:00+00:00"})
    with pytest.raises(ValueError, match="STOPPED 뒤"):
        R.validate_terminal(invalid)


def test_record_cli_keeps_seed2221_context_through_v3_main_and_job_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    lab = tmp_path / "lab"
    lab.mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    proof = tmp_path / "resource.json"
    proof.write_text("{}", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("python", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    job = {
        "schema": R.JOB3.JOB_SCHEMA,
        "mode": R.JOB3.MODE,
        "job_id": f"{R.PROTOCOL_ID}-{R.CHECKPOINT_IDS[0]}",
        "source": {"repo_root": str(ROOT), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(PROTOCOL), "sha256_lf": R.JOB1._sha(PROTOCOL, lf=True)},
        "resource_proof": {"path": str(proof), "sha256": R.JOB1._sha(proof)},
        "isaaclab": str(lab),
        "python": str(python),
        "output_directory": str(stage / "output"),
        "profile": R.JOB3.RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint), "sha256": R.CHECKPOINTS[R.CHECKPOINT_IDS[0]]["sha256"],
                       "logical_id": R.CHECKPOINT_IDS[0]},
    }
    job_path = stage / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr(R.JOB1.RECORD_JOB, "_checkpoint", lambda *_args: (
        {"logical_id": R.CHECKPOINT_IDS[0], "sha256": R.CHECKPOINTS[R.CHECKPOINT_IDS[0]]["sha256"]},
        checkpoint,
    ))
    monkeypatch.setattr(R.JOB1, "validate_resource_proof", lambda *_args, **_kwargs: {"resource_id": R.TARGET["resource_id"]})
    monkeypatch.setattr(R.JOB1.COMMON, "_python_executable", lambda *_args: None)
    monkeypatch.setattr(R.JOB1.COMMON, "_verify_runtime_sources", lambda *_args: None)

    observed = {}

    def gpu_boundary(path: Path, **_kwargs):
        checked = R.JOB3.validate_job(path, now=datetime(2026, 10, 27, 12, tzinfo=timezone.utc),
                                     require_fresh=False)
        observed["logical_id"] = checked["job"]["checkpoint"]["logical_id"]
        return {"completed": True}

    monkeypatch.setattr(R.RUN3, "run_job", gpu_boundary)
    assert R.main(["record", "--job", str(job_path)]) == 0
    assert observed == {"logical_id": "source-s2221-none"}
    assert R.JOB3.CHECKPOINT_IDS == ("source-s2220-none", "source-s2220-f1")

    script = f"""
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'bench')!r})
import run_vision_transferred_reuse_v1 as R
checkpoint = Path({str(checkpoint)!r})
R.JOB1.RECORD_JOB._checkpoint = lambda *_args: ({{'logical_id': 'source-s2221-none', 'sha256': {R.CHECKPOINTS[R.CHECKPOINT_IDS[0]]['sha256']!r}}}, checkpoint)
R.JOB1.validate_resource_proof = lambda *_args, **_kwargs: {{'resource_id': R.TARGET['resource_id']}}
R.JOB1.COMMON._python_executable = lambda *_args: None
R.JOB1.COMMON._verify_runtime_sources = lambda *_args: None
def boundary(path, **_kwargs):
    checked = R.JOB3.validate_job(path, now=datetime(2026, 10, 27, 12, tzinfo=timezone.utc), require_fresh=False)
    print(checked['job']['checkpoint']['logical_id'])
    return {{'completed': True}}
R.RUN3.run_job = boundary
raise SystemExit(R.main(['record', '--job', {str(job_path)!r}]))
"""
    subprocess_result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )
    assert subprocess_result.returncode == 0, subprocess_result.stderr
    assert subprocess_result.stdout.strip() == "source-s2221-none"


def test_record_child_launcher_reenters_reuse_cli_and_subprocess_accepts_seed2221() -> None:
    commands = []

    def capture(command, **_kwargs):
        commands.append(command)
        return object()

    old = str(Path(R.RUN3.PREVIOUS.__file__).resolve())
    launcher = R._reuse_child_launcher(capture)
    launcher([sys.executable, old, "--aaba-child", "--job", "fixture.json"])
    expected = [sys.executable, str(Path(R.__file__).resolve()), "record", "--aaba-child", "--job", "fixture.json"]
    assert commands == [expected]
    result = subprocess.run(
        [sys.executable, str(Path(R.__file__).resolve()), "record", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0
    assert "source-s2221-none" in result.stdout
    assert "source-s2221-f1" in result.stdout
    assert "source-s2220-none" not in result.stdout
