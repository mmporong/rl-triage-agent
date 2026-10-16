"""R1 execution v2의 prepared-owned adapter를 GPU와 Isaac 실행 없이 검증한다."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import r1_repair_execution_v2 as E  # noqa: E402


def _binding(iterations=300, source_checkpoint=None):
    return {
        "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/r1_repair_demo_v1.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {
            "bench/r1_repair_training.py": "c" * 64,
            "bench/optimizer_faults.py": "c" * 64,
            "evals/optimizer_observability.py": "c" * 64,
        },
        "runtime_source_sha256_lf": {"scripts/reinforcement_learning/rsl_rl/train.py": "d" * 64},
        "source_checkpoint_sha256": source_checkpoint,
        "task": E.TASK, "num_envs": 4096, "max_iterations": iterations,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
    }


def _receipt():
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True,
            "owned_process_identities": {}, "wall_s": 0.01}


def test_preflight_checks_resolved_read_only_paths(tmp_path, monkeypatch):
    isaac = tmp_path / "IsaacLab"
    files = {
        "ppo": "runtime/rsl_rl/ppo.py",
        "train": "scripts/reinforcement_learning/rsl_rl/train.py",
    }
    monkeypatch.setattr(E.R1, "SOURCE_FILES", files)
    for relative in ["_isaac_sim/python.bat", *files.values()]:
        path = isaac / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (isaac / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)

    result = E.preflight(isaac)

    assert result["isaaclab_root"] == str(isaac.resolve())
    assert result["runner_files"] == {name: str((isaac / relative).resolve())
                                      for name, relative in sorted(files.items())}
    assert Path(result["private_root"]).is_absolute()
    assert result["processes_started"] == result["gpu_queries"] == 0


@pytest.mark.parametrize("failure", ["cwd", "executable"])
def test_prepared_owned_rejects_invalid_launch_before_folder_creation(tmp_path, failure):
    folder = tmp_path / "owned"
    cwd = tmp_path / "missing" if failure == "cwd" else tmp_path
    executable = tmp_path / "missing.exe" if failure == "executable" else Path(sys.executable)
    with pytest.raises(FileNotFoundError):
        E.run_prepared_owned(folder, [str(executable)], cwd, 1, lambda *_: _receipt())
    assert not folder.exists()


def test_prepared_owned_rejects_duplicate_folder(tmp_path):
    calls = []

    def owned(folder, args, cwd, watchdog):
        assert Path(folder).is_dir()
        calls.append(Path(folder))
        return _receipt()

    folder = tmp_path / "owned"
    E.run_prepared_owned(folder, [sys.executable], tmp_path, 1, owned)
    with pytest.raises(FileExistsError):
        E.run_prepared_owned(folder, [sys.executable], tmp_path, 1, owned)
    assert calls == [folder]


def test_prepared_owned_does_not_hide_missing_parent_stage(tmp_path):
    called = False

    def owned(*_args):
        nonlocal called
        called = True
        return _receipt()

    with pytest.raises(FileNotFoundError):
        E.run_prepared_owned(tmp_path / "missing_stage/owned", [sys.executable], tmp_path, 1, owned)
    assert called is False and not (tmp_path / "missing_stage").exists()


def test_four_apis_prepare_all_v1_owned_call_sites_including_inspection(tmp_path, monkeypatch):
    seen = []

    def injected(folder, args, cwd, watchdog):
        folder = Path(folder)
        assert folder.is_dir()
        seen.append(folder.relative_to(tmp_path).as_posix())
        return _receipt()

    argv, cwd = [sys.executable, "-c", "pass"], tmp_path

    def fake_train(stage, isaaclab, arm, seed, run_name, binding, owned):
        Path(stage).mkdir()
        for name in ("owned", "checkpoint_owned", "extract"):
            owned(Path(stage) / name, argv, cwd, 1)
        return {"api": "train"}

    def fake_inspect(stage, isaaclab, arm, seed, run_name, binding, owned):
        Path(stage).mkdir()
        for name in ("owned", "checkpoint_owned"):
            owned(Path(stage) / name, argv, cwd, 1)
        return {"api": "inspect"}

    def fake_fixed(stage, isaaclab, artifacts, run_name, binding, owned):
        Path(stage).mkdir()
        owned(Path(stage) / "owned", argv, cwd, 1)
        return {"api": "fixed"}

    def fake_probe(stage, isaaclab, artifacts, run_name, binding, owned, *, probes=None):
        Path(stage).mkdir()
        owned(Path(stage) / "owned", argv, cwd, 1)
        return {"api": "probe", "probes": probes}

    monkeypatch.setattr(E.V1, "train", fake_train)
    monkeypatch.setattr(E.V1, "inspect", fake_inspect)
    monkeypatch.setattr(E.V1, "fixed_evaluate", fake_fixed)
    monkeypatch.setattr(E.V1, "legacy_probe", fake_probe)

    assert E.train(tmp_path / "train", cwd, "NONE", 1, "train", {}, injected)["api"] == "train"
    assert E.inspect(tmp_path / "inspect", cwd, "R1", 1, "inspect", {}, injected)["api"] == "inspect"
    assert E.fixed_evaluate(tmp_path / "fixed", cwd, {}, "fixed", {}, injected)["api"] == "fixed"
    assert E.legacy_probe(tmp_path / "probe", cwd, {}, "probe", {}, injected, probes=["P_value"])["api"] == "probe"
    assert seen == [
        "train/owned", "train/checkpoint_owned", "train/extract",
        "inspect/owned", "inspect/checkpoint_owned", "fixed/owned", "probe/owned",
    ]


def test_actual_v1_training_and_inspection_call_sites_receive_prepared_folders(tmp_path, monkeypatch):
    isaac = tmp_path / "IsaacLab"
    launcher = isaac / "_isaac_sim/python.bat"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("", encoding="utf-8")
    monkeypatch.setattr(E.V1, "_query_gpu", lambda: {"busy": False})
    monkeypatch.setattr(E.V1, "_verify_binding", lambda *_: None)
    monkeypatch.setattr(E.V1, "_inspection", lambda *_: {
        "status": "normal", "optimizer_steps": [20, 20, 20], "minibatches": [20, 20, 20],
        "records": [{"update_index": index} for index in range(3)],
        "report": {"status": "normal", "records": 3},
    })
    seen = []

    def owned(folder, argv, cwd, watchdog):
        folder = Path(folder)
        assert folder.is_dir()
        seen.append(folder.relative_to(tmp_path).as_posix())
        if "extract_tb.py" in " ".join(argv):
            Path(argv[-1]).write_text(json.dumps({"summary": {}}), encoding="utf-8")
        elif "--inspect-checkpoint" in argv:
            final = Path(argv[argv.index("--inspect-checkpoint") + 1])
            output = Path(argv[argv.index("--output") + 1])
            iteration = int(argv[argv.index("--expected-final-iteration") + 1])
            output.write_text(json.dumps({"device": "cpu", "first": {"iteration": 0},
                                          "final": {"iteration": iteration, "sha256": E.sha(final)}}),
                              encoding="utf-8")
        else:
            manifest = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            Path(manifest["records_path"]).write_text("{}\n{}\n{}\n", encoding="utf-8")
            arm = manifest["arm"]
            fault_epochs = 5 if arm == "NONE" else 1
            final_epochs = 1 if arm == "R1" else 5
            restoration = {"registry_value": 5, "runtime_value": 5} if arm == "CORRECTED" else None
            Path(manifest["readback_path"]).write_text(json.dumps({
                "arm": arm, "before": {"learning_epochs": 5, "mini_batches": 4},
                "after_fault": {"learning_epochs": fault_epochs, "mini_batches": 4},
                "restoration": restoration,
                "after": {"learning_epochs": final_epochs, "mini_batches": 4},
            }), encoding="utf-8")
            run = isaac / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-16_12-00-00_{manifest['run_name']}"
            (run / "params").mkdir(parents=True)
            (run / "params/agent.yaml").write_text(
                f"seed: {manifest['seed']}\nmax_iterations: {manifest['max_iterations']}\n"
                "algorithm:\n  num_learning_epochs: 5\n  num_mini_batches: 4\n",
                encoding="utf-8",
            )
            (run / "model_0.pt").write_bytes(b"first")
            (run / f"model_{manifest['max_iterations'] - 1}.pt").write_bytes(b"final")
        return _receipt()

    trained = E.train(tmp_path / "training", isaac, "CORRECTED", 3071, "corrected_s3071",
                      _binding(source_checkpoint="e" * 64), owned)
    inspected = E.inspect(tmp_path / "inspection", isaac, "R1", 2071, "inspect_r1_s2071",
                          _binding(3, "f" * 64), owned)

    assert trained["arm"] == "CORRECTED" and inspected["arm"] == "R1"
    assert seen == [
        "training/owned", "training/checkpoint_owned", "training/extract",
        "inspection/owned", "inspection/checkpoint_owned",
    ]


def test_actual_v1_fixed_eval_and_probe_receive_prepared_folders(tmp_path, monkeypatch):
    isaac = tmp_path / "IsaacLab"
    launcher = isaac / "_isaac_sim/python.bat"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"policy")
    digest = hashlib.sha256(b"policy").hexdigest()
    artifact = {"arm": "NONE", "seed": 2071,
                "checkpoint": {"path": str(checkpoint), "sha256": digest}}
    monkeypatch.setattr(E.V1, "_query_gpu", lambda: {"busy": False})
    monkeypatch.setattr(E.V1, "_verify_runtime", lambda *_: None)
    seen = []

    def owned(folder, argv, cwd, watchdog):
        folder = Path(folder)
        assert folder.is_dir()
        seen.append(folder.relative_to(tmp_path).as_posix())
        jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
        for job in jobs["jobs"]:
            Path(job["output"]).write_text(json.dumps({
                "name": job["name"], "task": E.TASK, "seed": 2071,
                "checkpoint": {"sha256": digest},
                "protocol_sha256_lf": E.sha(E.FIXED_EVAL_PROTOCOL, True),
                "probes": {probe: {"measurement": {}, "exit": 0} for probe in job.get("probes", [])},
            }), encoding="utf-8")
        return _receipt()

    fixed = E.fixed_evaluate(tmp_path / "fixed", isaac, {"NONE": artifact}, "fixed_s2071", _binding(), owned)
    probed = E.legacy_probe(tmp_path / "probe", isaac, artifact, "probe_s2071", _binding(), owned)

    assert fixed["reports"]["NONE"]["checkpoint"]["sha256"] == digest
    assert set(probed["report"]["probes"]) == set(E.V1.PROBES)
    assert seen == ["fixed/owned", "probe/owned"]


def test_actual_harmless_cpu_owned_process_succeeds_and_cleans_up(tmp_path):
    receipt = E.run_prepared_owned(
        tmp_path / "owned", [sys.executable, "-c", "print('v2-cpu-ok')"], tmp_path, 10,
    )
    assert {key: receipt[key] for key in (
        "job_assigned_before_release", "natural_exit", "timed_out", "error_type",
        "exit_code", "cleanup_verified",
    )} == {
        "job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
        "error_type": None, "exit_code": 0, "cleanup_verified": True,
    }
    assert (tmp_path / "owned/launch.json").is_file()
    assert (tmp_path / "owned/stdout.log").read_text(encoding="utf-8").strip() == "v2-cpu-ok"
    assert (tmp_path / "owned/stderr.log").read_bytes() == b""
