"""부모 쪽 공용 U1 API(가짜 소유 Job, GPU 없음): 학습·probe·교정 학습의 인자, 산출물, 거부 경로, loop.run 호환."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import loop  # noqa: E402
import upstream_runs as UR  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

READBACK = {"fault": "U1", "ok": True, "swapped": {}, "bound_to_reward_terms": {}}
AUX = {"schema": "upstream_aux_fingerprints_v1", "used_for_verdict": False, "differs_from_clean": ["lin_vel_z_l2"]}
LOG_DIR = "2026-10-14_00-00-00_"


class FakeOwned:
    """run_owned 대역: argv를 기록하고 Isaac 쪽 진입점이 남길 산출물을 만든다."""

    def __init__(self, isaaclab: Path, *, readback=READBACK, aux=AUX, exit_code=0, measurement=None, timeout=False,
                 restored=True):
        self.isaaclab, self.calls = isaaclab, []
        self.readback, self.aux, self.exit_code, self.timeout, self.restored = readback, aux, exit_code, timeout, restored
        self.measurement = measurement or {"track_lin_vel_xy_exp_rel_error": 0.4, "track_ang_vel_z_exp_rel_error": 0.9}

    def __call__(self, folder, argv, cwd, watchdog_s):
        self.calls.append((Path(folder).name, list(argv), Path(cwd), watchdog_s))
        stdout = []
        if Path(argv[1]).name == "extract_tb.py":
            Path(argv[3]).write_text(json.dumps({"series": {}}), encoding="utf-8")
        else:
            mode = argv[2]
            rest = argv[argv.index("--") + 1:]
            if mode in ("train", "correct"):
                run_name = rest[rest.index("--run_name") + 1]
                run_dir = self.isaaclab / "logs/rsl_rl/unitree_go2_flat" / f"{LOG_DIR}{run_name}"
                run_dir.mkdir(parents=True)
                (run_dir / "model_299.pt").write_bytes(b"ckpt-" + run_name.encode())
            if mode == "correct":
                readback = Path(argv[argv.index("--readback") + 1])
                readback.write_text(json.dumps({"correction_id": argv[argv.index("--correction") + 1],
                                                "registered_target_state": "restored_to_clean" if self.restored else "not_restored"}),
                                    encoding="utf-8")
                readback.with_name(readback.stem + ".aux.json").write_text(json.dumps(self.aux), encoding="utf-8")
            if mode == "probes":
                jobs_path = Path(rest[rest.index("--jobs") + 1])
                jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
                job = jobs["jobs"][0]
                checkpoint_sha = hashlib.sha256(Path(job["checkpoint"]).read_bytes()).hexdigest()
                Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
                Path(job["output"]).write_text(json.dumps({
                    "name": job["name"], "checkpoint": {"file": "model_299.pt", "sha256": checkpoint_sha},
                    "probes": {job["probes"][0]: {"measurement": self.measurement, "exit": 0}}}), encoding="utf-8")
                Path(rest[rest.index("--metrics") + 1]).write_text(json.dumps({
                    "schema": "probe_process_metrics_v1", "completed": True, "phase": "before_app_close", "cpu_s": 3.0,
                    "cpu_scope": "isaac", "simulator_steps_scope": "steps",
                    "jobs_sha256": hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
                    "probes_sha256_lf": UR.sha(UR.PROBES, True),
                    "measurements": [{"probe": job["probes"][0], "checkpoint_sha256": checkpoint_sha, "num_envs": 256,
                                      "successful_step_calls": 200, "simulator_steps": 51200, "completed": True}]}),
                    encoding="utf-8")
            if self.readback is not None:
                stdout.append("[upstream_fault] " + json.dumps(self.readback))
            if self.aux is not None:
                stdout.append("[upstream_aux] " + json.dumps(self.aux))
        (Path(folder) / "stdout.log").write_text("\n".join(stdout) + "\n", encoding="utf-8")
        return {"job_assigned_before_release": True, "natural_exit": not self.timeout, "timed_out": self.timeout,
                "error_type": None, "exit_code": 124 if self.timeout else self.exit_code, "cleanup_verified": True,
                "owned_process_identities": {"4242": 1}, "wall_s": 12.5}


@pytest.fixture
def lab(tmp_path):
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    return isaaclab


def _train(lab, tmp_path, owned, run_name="u1_s2040"):
    return UR.train(lab, fault="U1", seed=2040, run_name=run_name, telemetry_path=tmp_path / "runs" / f"{run_name}.telemetry.json",
                    meta_path=tmp_path / "private" / f"{run_name}.meta.json", work_dir=tmp_path / "work" / run_name,
                    owned=owned)


def test_train_uses_the_run_case_arguments_and_meta_keys(lab, tmp_path):
    owned = FakeOwned(lab)
    out = _train(lab, tmp_path, owned)
    (name, argv, cwd, watchdog), (extract_name, extract_argv, _, _) = owned.calls
    assert (name, cwd, watchdog) == ("train", lab, 1800) and Path(argv[0]).name == "python.bat"
    assert argv[1:5] == [str(UR.ENTRY), "train", "--fault", "U1"]
    assert argv[argv.index("--") + 1:] == ["--task", UR.TASK, "--num_envs", "4096", "--max_iterations", "300",
                                            "--seed", "2040", "--headless", "--run_name", "u1_s2040"]
    assert extract_name == "extract" and Path(extract_argv[1]).name == "extract_tb.py"
    meta = json.loads(Path(out["meta_path"]).read_text(encoding="utf-8"))
    assert {k: meta[k] for k in ("case_id", "seed", "num_envs", "max_iterations", "task", "fault", "train_exit_code")} == {
        "case_id": "u1", "seed": 2040, "num_envs": 4096, "max_iterations": 300, "task": UR.TASK, "fault": "U1",
        "train_exit_code": 0}
    assert meta["upstream_readback"]["ok"] is True and meta["upstream_aux"]["used_for_verdict"] is False
    assert Path(meta["run_dir"]).name == f"{LOG_DIR}u1_s2040"
    with pytest.raises(FileExistsError):
        _train(lab, tmp_path, FakeOwned(lab))


@pytest.mark.parametrize("case", ["no_readback", "readback_not_ok", "no_aux", "exit_1", "timeout", "bad_name"])
def test_train_refuses_unconfirmed_or_failed_runs(lab, tmp_path, case):
    owned = {"no_readback": FakeOwned(lab, readback=None), "readback_not_ok": FakeOwned(lab, readback={**READBACK, "ok": False}),
             "no_aux": FakeOwned(lab, aux=None), "exit_1": FakeOwned(lab, exit_code=1),
             "timeout": FakeOwned(lab, timeout=True), "bad_name": FakeOwned(lab)}[case]
    with pytest.raises((ValueError, RuntimeError)):
        _train(lab, tmp_path, owned, "u1 s2040" if case == "bad_name" else "u1_s2040")
    assert not (tmp_path / "private/u1_s2040.meta.json").exists()


def test_probe_writes_the_run_probes_artifacts_with_the_fault_armed(lab, tmp_path):
    _train(lab, tmp_path, FakeOwned(lab))
    owned = FakeOwned(lab)
    out = UR.probe(lab, fault="U1", seed=2040, run_name="u1_s2040", probe="P_reward", tag="t_loop",
                   work_dir=tmp_path / "work/probe", owned=owned, results=tmp_path / "results")
    (_, argv, _, watchdog), = owned.calls
    assert argv[1:5] == [str(UR.ENTRY), "probes", "--fault", "U1"] and watchdog == 600
    jobs = json.loads((tmp_path / "work/probe/jobs.json").read_text(encoding="utf-8"))
    assert (jobs["fault"], jobs["upstream_fault"], jobs["seed"]) == ("NONE", "U1", 2040)  # hidden_faults는 아무것도 하지 않는다
    assert out["exit_code"] == 0 and out["measurement"]["track_ang_vel_z_exp_rel_error"] == 0.9
    paths = UR.probe_paths("t_loop", "u1_s2040", "P_reward", tmp_path / "results")
    meta = json.loads(paths["execution"].read_text(encoding="utf-8"))
    report_sha = UR.sha(paths["report"], True)
    assert (meta["schema"], meta["metered"], meta["report_sha256_lf"], meta["upstream_readback_ok"]) == (
        "probe_execution_v1", True, report_sha, True)
    resources = json.loads(paths["resources"].read_text(encoding="utf-8"))
    assert resources["report_sha256_lf"] == report_sha and resources["simulator_steps"] == 51200
    with pytest.raises(FileExistsError):
        UR.probe(lab, fault="U1", seed=2040, run_name="u1_s2040", probe="P_reward", tag="t_loop",
                 work_dir=tmp_path / "work/probe2", owned=FakeOwned(lab), results=tmp_path / "results")


@pytest.mark.parametrize("owned_kwargs", [{"readback": None}, {"readback": {**READBACK, "ok": False}}, {"aux": None},
                                          {"timeout": True}])
def test_probe_without_a_confirmed_fault_gives_no_measurement(lab, tmp_path, owned_kwargs):
    _train(lab, tmp_path, FakeOwned(lab))
    out = UR.probe(lab, fault="U1", seed=2040, run_name="u1_s2040", probe="P_reward", tag="t_loop",
                   work_dir=tmp_path / "work/probe", owned=FakeOwned(lab, **owned_kwargs), results=tmp_path / "results")
    assert out["measurement"] is None and out["exit_code"] not in (0, None)


def test_loop_executor_plugs_into_loop_run(lab, tmp_path, monkeypatch):
    """승인 → 소비 → U1 probe → classify → receipt가 기존 loop.run으로 이어진다."""
    _train(lab, tmp_path, FakeOwned(lab))
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    reference = {"P_reward": {"track_lin_vel_xy_exp_rel_error": 1.7e-8, "track_ang_vel_z_exp_rel_error": 1.7e-8}}
    message = loop.start("u1_s2040", ["reward", "actuator", "exploration"], 3, 4, "c" * 64, reference)
    rid = next(iter(loop._load("u1_s2040")[2].requests()))
    assert "P_noise" in message or "P_reward" in message
    _, _, led = loop._load("u1_s2040")
    led.approve(rid, "operator")
    execute, preflight = UR.loop_executor(lab, fault="U1", seed=2040, tag="t_loop", work_root=tmp_path / "work/loop",
                                          owned=FakeOwned(lab, measurement={**reference["P_reward"], "noise_ratio": 1.0}),
                                          results=tmp_path / "results")
    loop.run("u1_s2040", rid, execute, preflight=preflight)
    receipt = next(e for e in loop._load("u1_s2040")[2].events() if e["event"] == "receipt")
    assert receipt["exit_code"] == 0 and receipt["outcome"] in L.OUTCOMES


def test_correction_train_runs_capture_fault_apply_and_checks_the_registered_target(lab, tmp_path):
    owned = FakeOwned(lab)
    out = UR.correction_train(lab, fault="U1", correction_id="restore_body_frame_reward", seed=3040, run_name="u1_fix_s3040",
                              work_dir=tmp_path / "work/fix", contract={"targets": {}}, owned=owned)
    (_, argv, _, _), = owned.calls
    assert argv[1:7] == [str(UR.ENTRY), "correct", "--fault", "U1", "--correction", "restore_body_frame_reward"]
    assert argv[argv.index("--") + 1:][-2:] == ["--run_name", "u1_fix_s3040"] and "3040" in argv
    assert json.loads((tmp_path / "work/fix/contract.json").read_text(encoding="utf-8")) == {"targets": {}}
    assert out["correction"]["registered_target_state"] == "restored_to_clean" and out["aux"]["used_for_verdict"] is False
    assert out["checkpoint"]["sha256"] == UR.sha(Path(out["checkpoint"]["path"]))
    with pytest.raises(ValueError, match="등록 표적"):
        UR.correction_train(lab, fault="U1", correction_id="restore_body_frame_reward", seed=3041, run_name="u1_fix_s3041",
                            work_dir=tmp_path / "work/fix2", contract={}, owned=FakeOwned(lab, restored=False))
