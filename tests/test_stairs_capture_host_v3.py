from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

# pyright: reportMissingImports=false

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "bench/stairs_capture_host_v3.py"


def load():
    spec = importlib.util.spec_from_file_location("stairs_capture_host_tested", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load()


def registration():
    common = {name: "1" * 64 for name in (
        "source/isaaclab/isaaclab/terrains/config/rough.py",
        "source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py",
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
    )}
    return {
        "runner_contract": {
            "isaaclab_commit": "2" * 40,
            "common_files_lf": common,
            "role_files_lf": {
                "biped": {"source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py": "3" * 64},
                "quadruped": {"source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py": "4" * 64},
            },
            "runner_sha256_lf": "5" * 64,
        },
        "sources": {"runner": {"path": "bench/run_stairs_playback_v2.py", "sha256_lf": "5" * 64}},
        "smoke": {"tag": "stairs-s172-host173-biped-up-v1", "mode": "first_full_clip",
                  "clip_id": "biped-stairs-up"},
        "clip_acceptance": {"manifest": "manifest.json", "trace": "state-trace.jsonl",
                            "video_glob": "video/*.mp4",
                            "app_startup_marker": "shutdown/01-app-startup.json",
                            "result_ready_marker": "shutdown/02-result-ready.json",
                            "app_close_entered_marker": "shutdown/03-app-close-entered.json",
                            "app_close_marker": "shutdown/04-app-close-returned.json",
                            "trace_rows": 626, "video_frames": 625, "fps": 50},
    }


class FakeDriver:
    def __init__(self, root: Path, fail_label: str | None = None):
        self.metadata = {"remote": "/remote/run", "repo": "/remote/repo", "python": "/remote/python"}
        self.root = root
        self.fail_label = fail_label
        self.budgets = []

    def budget(self, seconds):
        self.budgets.append(seconds)


class FakeOwner:
    def __init__(self, root: Path, fail_label: str | None = None):
        self.driver = FakeDriver(root, fail_label)
        self.lifecycle = {"resource_name": H.RESOURCE["resource_name"]}
        self.host_module = object()
        self.events = []
        self.archive = b"registered archive bytes"

    def start(self):
        self.events.append(("start",))

    def boot(self):
        self.events.append(("boot",))

    def copy(self, label, source, target, timeout=180):
        self.events.append(("copy", label, str(source), str(target), timeout))
        if label == "stairs-collect":
            Path(target).write_bytes(self.archive)
        return {"label": label}

    def script(self, label, body, timeout):
        self.events.append(("script", label, timeout, body))
        if label == self.driver.fail_label:
            raise RuntimeError("registered clip failure")
        if label == "stairs-pack":
            stdout = self.driver.root / "pack.stdout"
            packet = {"path": "/remote/run/stairs-host173.tar.gz",
                      "sha256": hashlib.sha256(self.archive).hexdigest(), "files": 7}
            stdout.write_text("STAIRS_ARCHIVE=" + json.dumps(packet) + "\n", encoding="utf-8")
            return {"stdout_path": str(stdout)}
        return {"label": label}


def checkpoint_files(root: Path):
    result = {}
    for role in H.CHECKPOINTS:
        path = root / H.CHECKPOINTS[role][0]
        path.write_bytes(role.encode())
        result[role] = path
    return result


def collector(archive, destination, backup):
    assert Path(archive).read_bytes() == b"registered archive bytes"
    return {"all_match": True, "local": str(destination), "backup": str(backup), "files_verified": 7}


def test_build_specs_keeps_role_specific_source_closure():
    specs = H.build_specs(registration(), {"biped": "/h1.pt", "quadruped": "/go2.pt"})
    assert [row["clip_id"] for row in specs] == [row[0] for row in H.CLIPS]
    assert len(specs[0]["source"]["isaaclab_files_lf"]) == 4
    assert "config/h1/rough_env_cfg.py" in next(name for name in specs[0]["source"]["isaaclab_files_lf"] if "/h1/" in name)
    assert any("/go2/" in name for name in specs[2]["source"]["isaaclab_files_lf"])


def test_registered_sequence_starts_once_then_smoke_four_clips_pack_and_collect(tmp_path):
    owner = FakeOwner(tmp_path)
    out, backup = tmp_path / "out", tmp_path / "backup"
    out.mkdir()
    result = H.run_sequence(owner, registration(), checkpoint_files(tmp_path), out, backup,
                            collector=collector)
    scripts = [event[1] for event in owner.events if event[0] == "script"]
    assert scripts == ["stairs-prepare", *["stairs-" + row[0] for row in H.CLIPS], "stairs-pack"]
    assert sum(event[0] == "start" for event in owner.events) == 1
    assert result["completed_clips"] == [row[0] for row in H.CLIPS]
    assert result["collection"]["all_match"] is True


def test_clip_failure_preserves_partial_and_does_not_start_remaining(tmp_path):
    failed = "stairs-" + H.CLIPS[1][0]
    owner = FakeOwner(tmp_path, failed)
    out, backup = tmp_path / "out", tmp_path / "backup"
    out.mkdir()
    with pytest.raises(RuntimeError, match="registered clip failure"):
        H.run_sequence(owner, registration(), checkpoint_files(tmp_path), out, backup,
                       collector=collector)
    scripts = [event[1] for event in owner.events if event[0] == "script"]
    assert scripts == ["stairs-prepare", "stairs-" + H.CLIPS[0][0], failed, "stairs-pack"]
    failure = json.loads((out / "primary-failure.json").read_text(encoding="utf-8"))
    assert failure["completed_clips"] == [H.CLIPS[0][0]]
    assert failure["automatic_retry"] is False
    assert (out / "collection.json").is_file()


def test_clip_child_rejects_nonzero_and_checks_bound_evidence():
    body = H._clip_body({"repo": "/repo", "python": "/python", "runner": "bench/runner.py",
                         "spec": "/spec", "out": "/out", "acceptance": registration()["clip_acceptance"]})
    assert "if r.returncode!=0" in body
    assert "manifest source/checkpoint binding" in body
    assert "app startup marker missing" in body
    assert "result ready marker missing" in body
    assert "app close entered marker missing" in body
    assert "app close marker missing" in body
    assert "ffprobe" in body


def test_validate_plan_refuses_other_resource_or_deadline_protocol(tmp_path):
    plan = {
        "resource_id": H.RESOURCE["resource_id"], "environment_protocol_anchor": H.ENV_ANCHOR,
        "environment_phase": {"path": "/repo/" + H.ENV_REF["path"], "sha256": H.ENV_REF["sha256"]},
        "lifecycle_protocol": {"path": "/repo/" + H.LIFECYCLE_REF["path"], "sha256": H.LIFECYCLE_REF["sha256"]},
        "driver": {"path": "/driver", "sha256": "a" * 64},
    }
    reg = {"sources": {"driver": {"sha256": "a" * 64}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert H.validate_plan(path, reg)["resource_id"] == H.RESOURCE["resource_id"]
    plan["resource_id"] = "other"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="자원이 다르다"):
        H.validate_plan(path, reg)


def test_draft_registration_is_never_executable(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps({"schema": H.SCHEMA, "status": "draft_not_registered_not_executable"}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="실행 가능"):
        H.validate_registration(path, tmp_path, require_committed=False)


def test_uncommitted_registration_bytes_are_refused_before_execution(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    protocol = repo / "bench/protocols/stairs.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(H, "_git", lambda *_args: b"different")
    with pytest.raises(ValueError, match="커밋된 preregistration"):
        H.validate_registration(protocol, repo)


def test_registered_source_anchor_and_current_runner_bytes_are_consumed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "bench").mkdir(parents=True)
    host = repo / "bench/stairs_capture_host_v3.py"
    runner = repo / "bench/run_stairs_playback_v2.py"
    host.write_bytes(SOURCE.read_bytes())
    runner.write_text("RUNNER = 'v2'\n", encoding="utf-8")
    value = registration()
    value.update({
        "schema": H.SCHEMA,
        "resource": H.RESOURCE,
        "environment": {"anchor": H.ENV_ANCHOR, **H.ENV_REF},
        "lifecycle": H.LIFECYCLE_REF,
        "lease_basis": H.LEASE_REF,
        "budget": {
            "budget_origin": "2026-10-28T21:05:36.779993+00:00",
            "compute_cap_usd": "85", "program_cap_usd": "250", "lease_batch_budget_usd": "55",
            "deadline_utc": H.DEADLINE, "stop_cutoff_utc": H.STOP_CUTOFF, "reset": False,
        },
        "clips": [
            {"clip_id": clip, "robot_role": role, "direction": direction,
             "training_task": training, "play_task": play}
            for clip, role, direction, training, play in H.CLIPS
        ],
        "smoke": {"tag": "stairs-s172-host184-biped-up-v3", "mode": "first_full_clip",
                  "clip_id": "biped-stairs-up"},
        "execution": {"new_vm": 0, "install": 0, "new_training": 0, "model_api": 0,
                      "maximum_stop_requests": 1, "automatic_retry": False},
    })
    value["sources"].update({
        "host": {"path": "bench/stairs_capture_host_v3.py", "sha256_lf": "0" * 64},
        "lifecycle_adapter": {"sha256": "a" * 64},
        "driver": {"sha256": "b" * 64},
    })
    value["status"] = "registered_executable"
    value["source_anchor"] = "a" * 40
    value["sources"]["host"]["sha256_lf"] = H.sha(host, lf=True)
    value["sources"]["runner"]["sha256_lf"] = H.sha(runner, lf=True)
    value["runner_contract"]["runner_sha256_lf"] = H.sha(runner, lf=True)
    registered = tmp_path / "registered.json"
    registered.write_text(json.dumps(value), encoding="utf-8")

    def fake_git(_repo, *args):
        if args[0] == "show":
            relative = args[1].split(":", 1)[1]
            return (repo / relative).read_bytes()
        return b""

    monkeypatch.setattr(H, "_git", fake_git)
    assert H.validate_registration(registered, repo, host_source=host,
                                   require_committed=False)["source_anchor"] == "a" * 40
    runner.write_text("RUNNER = 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="현재 바이트"):
        H.validate_registration(registered, repo, host_source=host, require_committed=False)
