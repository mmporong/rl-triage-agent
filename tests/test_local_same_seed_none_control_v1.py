from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

CANDIDATE = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(CANDIDATE))

import local_same_seed_none_control_v1 as CONTROL

TEMPLATE = CANDIDATE / "protocols/local_same_seed_none_control_s3221_v1.template.json"


def protocol() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def report(
    err_xy: float, err_yaw: float, *, survival: float = 20.0, fall: float = 0.0
) -> dict:
    return {
        "metrics": {
            "lin_vel_rmse_mps": err_xy,
            "yaw_rate_rmse_radps": err_yaw,
            "mean_survival_s": survival,
            "fall_rate": fall,
        },
        "standstill": {"lin_vel_rmse_mps": 1.0},
        "horizon_steps": 1000,
        "step_dt": 0.02,
    }


def dump(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_inputs(tmp_path: Path, value: dict) -> Path:
    root = tmp_path / "ri213"
    checkpoint = root / "corrected.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    value["old_checkpoint_sha256"][2] = checkpoint_sha
    bodies = {
        "result": {"outcome": "not_recovered", "ratios": value["original_result"]},
        "none_report": report(0.1, 0.1),
        "corrected_report": report(0.1, 0.2),
        "corrected_worker": {
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "readback": {
                "remaining_faults": [],
                "reward": {
                    "before_code_sha256": "a" * 64,
                    "after_code_sha256": "a" * 64,
                },
            },
        },
        "train_job": {
            "seed": 3221,
            "iterations": 300,
            "arm": "CORRECTED",
            "runtime_source_sha256_lf": value["runtime"]["source_sha256_lf"],
            "python_package_source_sha256_lf": value["python_package"][
                "source_sha256_lf"
            ],
        },
    }
    for name, body in bodies.items():
        row = value["ri213_inputs"][name]
        row["relative"] = f"{name}.json"
        row["sha256"] = dump(root / row["relative"], body)
    reports = {
        "NONE": {"sha256": value["ri213_inputs"]["none_report"]["sha256"]},
        "CORRECTED": {"sha256": value["ri213_inputs"]["corrected_report"]["sha256"]},
    }
    row = value["ri213_inputs"]["evaluate_worker"]
    row["relative"] = "evaluate_worker.json"
    row["sha256"] = dump(
        root / row["relative"], {"completed": True, "reports": reports}
    )
    return root


def test_preregistered_interpretation_branches_and_invalid_reference() -> None:
    original, corrected = report(0.1, 0.1), report(0.1, 0.2)
    assert (
        CONTROL.classify(original, corrected, copy.deepcopy(corrected))[
            "interpretation"
        ]
        == "seed_or_command_coverage_consistent"
    )
    assert (
        CONTROL.classify(original, corrected, copy.deepcopy(original))["interpretation"]
        == "correction_path_influence_possible"
    )
    assert (
        CONTROL.classify(original, original, copy.deepcopy(original))["interpretation"]
        == "inconclusive"
    )
    assert (
        CONTROL.classify(original, corrected, report(0.0, 0.0, survival=0.0))[
            "interpretation"
        ]
        == "inconclusive"
    )


def test_build_jobs_uses_none_and_real_source_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = protocol()
    value.update(
        {
            "status": "registered",
            "registration_anchor": subprocess.run(
                ["git", "-C", str(CONTROL.ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip(),
            "source_sha256_lf": {},
        }
    )
    input_root = synthetic_inputs(tmp_path, value)
    repo_sources = {name: "1" * 64 for name in CONTROL.TRAINING.MINIMUM_BOUND_SOURCES}
    repo_sources["bench/local_same_seed_none_control_v1.py"] = "1" * 64
    captured: dict = {}
    monkeypatch.setattr(CONTROL, "PROTOCOL", TEMPLATE)
    monkeypatch.setattr(
        CONTROL, "_registered", lambda _commit: (copy.deepcopy(value), repo_sources)
    )
    monkeypatch.setattr(
        CONTROL.BASE,
        "runtime_snapshot",
        lambda *_args: value["runtime"]["source_sha256_lf"],
    )
    monkeypatch.setattr(
        CONTROL.BASE,
        "package_snapshot",
        lambda *_args: value["python_package"]["source_sha256_lf"],
    )
    monkeypatch.setattr(
        CONTROL.TRAINING,
        "verify_source_binding",
        lambda manifest, runtime_root=None: captured.update(
            {"manifest": manifest, "root": runtime_root}
        ),
    )
    train, evaluate, inputs = CONTROL.build_jobs(
        input_root,
        Path("relative-ri217-stage"),
        Path.home() / "IsaacLab",
        value["registration_anchor"],
        Path.home() / "IsaacLab/_isaac_sim/python.bat",
    )
    assert Path(train["stage"]).is_absolute()
    assert train["arm"] == "NONE" and train["source_checkpoint_sha256"] is None
    assert train["seed"] == 3221 and train["iterations"] == 300
    assert set(evaluate["artifacts"]) == {"HEALTHY_NONE_S3221"}
    assert set(inputs) == set(value["ri213_inputs"])
    assert (
        captured["manifest"]["runtime_source_sha256_lf"]
        == value["runtime"]["source_sha256_lf"]
    )
    assert captured["root"] == Path.home() / "IsaacLab"


def test_protocol_keeps_three_old_checkpoints_and_one_attempt() -> None:
    value = protocol()
    assert len(value["old_checkpoint_sha256"]) == 3
    assert len(set(value["old_checkpoint_sha256"])) == 3
    assert value["automatic_retry"] == 0
    assert value["training"] == {
        "seed": 3221,
        "num_envs": 4096,
        "iterations": 300,
        "num_steps_per_env": 24,
        "action_scale": 0.25,
    }


def test_finalizer_accepts_documented_relative_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import finalize_local_same_seed_none_control_v1 as finalize

    template = tmp_path / "template.json"
    dump(template, protocol())
    target = tmp_path / "registered.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finalize, "PROTOCOL", target)
    monkeypatch.setattr(finalize, "ROOT", tmp_path)
    monkeypatch.setattr(finalize, "SOURCE_FILES", ())
    monkeypatch.setattr(finalize.BASE, "git_blob_map", lambda *_args: {})
    monkeypatch.setattr(sys, "argv", ["finalize", "--source-commit", "a" * 40,
                                    "--template", str(template), "--out", "registered.json"])
    assert finalize.main() == 0
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "registered"


def test_none_terminal_rejects_corrected_and_resumed_manifests(tmp_path: Path) -> None:
    terminal = CONTROL.TERMINAL
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"synthetic-new-training")
    manifest = tmp_path / "manifest.json"
    body = {"arm": "NONE", "seed": 3221, "max_iterations": 300,
            "source_checkpoint_sha256": None, "original_faults": [], "applied_corrections": []}
    manifest_sha = dump(manifest, body)
    output = tmp_path / "worker.json"
    worker = {"completed": True, "arm": "NONE", "seed": 3221, "iterations": 300,
              "checkpoint": {"path": str(checkpoint), "sha256": CONTROL.BASE.sha(checkpoint)},
              "manifest_path": str(manifest), "manifest_sha256": manifest_sha,
              "readback": {"remaining_faults": []}}
    dump(output, worker)
    assert terminal.training_evidence(output, set())["arm"] == "NONE"
    worker["arm"] = "CORRECTED"
    dump(output, worker)
    with pytest.raises(terminal.TerminalEvidenceError):
        terminal.training_evidence(output, set())
    worker["arm"] = "NONE"
    body["source_checkpoint_sha256"] = "b" * 64
    worker["manifest_sha256"] = dump(manifest, body)
    dump(output, worker)
    with pytest.raises(terminal.TerminalEvidenceError):
        terminal.training_evidence(output, set())


def test_none_terminal_actual_proxy_and_nonzero_exit(tmp_path: Path) -> None:
    terminal = CONTROL.TERMINAL
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"synthetic-new-training")
    manifest = tmp_path / "manifest.json"
    manifest_sha = dump(manifest, {"arm": "NONE", "seed": 3221, "max_iterations": 300,
                                  "source_checkpoint_sha256": None,
                                  "original_faults": [], "applied_corrections": []})
    output = tmp_path / "worker.json"
    dump(output, {"completed": True, "arm": "NONE", "seed": 3221, "iterations": 300,
                  "checkpoint": {"path": str(checkpoint), "sha256": CONTROL.BASE.sha(checkpoint)},
                  "manifest_path": str(manifest), "manifest_sha256": manifest_sha,
                  "readback": {"remaining_faults": []}})

    class Settings:
        value = False

        def get_as_bool(self, _name: str) -> bool:
            return self.value

        def set_bool(self, _name: str, value: bool) -> None:
            self.value = value

    class App:
        def close(self, *, wait_for_replicator: bool) -> None:
            assert wait_for_replicator is False

    folder = tmp_path / "terminal"
    terminal.AppProxy(App(), Settings(), output, folder, set()).close()
    assert terminal.verify(folder, output, 0, set())["arm"] == "NONE"
    with pytest.raises(terminal.TerminalEvidenceError):
        terminal.verify(folder, output, 1, set())
