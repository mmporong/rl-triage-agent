"""관측 wrapper는 probe 반환값·예외·step 호출을 보존하고 실제 전이만 센다."""
import hashlib
import importlib.util
import json
from types import SimpleNamespace

import pytest

from test_offline_replay import ROOT


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_observer_preserves_result_arguments_and_restores_step(tmp_path):
    meter = module("meter_observer", "evals/metered_probes.py")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls, records, expected = [], [], {"original": "measurement"}
    original_step = lambda action: calls.append(action) or ("observation", action)
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=7), step=original_step)

    def measure(probe, got_env, runner, ckpt):
        assert (probe, got_env, runner, ckpt) == ("P_noise", env, "runner", checkpoint)
        assert env.step("a") == ("observation", "a")
        assert env.step("b") == ("observation", "b")
        return expected

    assert meter.observe_measure(measure, records)("P_noise", env, "runner", checkpoint) is expected
    assert env.step is original_step and calls == ["a", "b"]
    assert records[0]["successful_step_calls"] == 2 and records[0]["simulator_steps"] == 14
    assert records[0]["completed"] is True


def test_failed_step_is_not_counted_and_original_exception_survives(tmp_path):
    meter = module("meter_failed", "evals/metered_probes.py")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fail(_):
        raise RuntimeError("original failure")

    env, records = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=3), step=fail), []
    measured = meter.observe_measure(lambda p, e, r, c: e.step("a"), records)
    with pytest.raises(RuntimeError, match="original failure"):
        measured("P_reward", env, None, checkpoint)
    assert env.step is fail and records[0]["simulator_steps"] == 0 and not records[0]["completed"]


def test_wrapper_writes_real_cpu_and_zero_steps_for_read_only_probe(tmp_path, monkeypatch):
    meter = module("meter_entry", "evals/metered_probes.py")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    jobs = tmp_path / "jobs.json"
    jobs.write_text('{"jobs":[]}', encoding="utf-8")
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=64), step=lambda _: pytest.fail("no rollout"))
    monkeypatch.setattr(meter.probes, "measure", lambda *_: {"critic_change": 0.5})
    monkeypatch.setattr(meter.probes, "main", lambda: meter.probes.measure("P_value", env, None, checkpoint) and 0)
    out = tmp_path / "metrics.json"
    assert meter.main(["--jobs", str(jobs), "--headless", "--metrics", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["completed"] and report["cpu_s"] > 0
    assert report["measurements"][0]["simulator_steps"] == 0
    assert report["jobs_sha256"] == hashlib.sha256(jobs.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="덮어쓰지"):
        meter.main(["--jobs", str(jobs), "--metrics", str(out)])


def test_agreement_reports_numeric_and_classification_differences():
    live = module("live_agreement", "evals/live_loop_evidence.py")
    prereg = {"relative_tolerance": 1e-3, "absolute_tolerance": 1e-8,
              "scalar_fields": {"P_noise": ["noise_ratio"], "P_reward": ["track_lin_vel_xy_exp_rel_error", "track_ang_vel_z_exp_rel_error"]}}
    good = live.measurement_agreement("P_noise", {"noise_ratio": 1.0005}, {"noise_ratio": 1.0}, None, prereg)
    assert good["scalar_agreement"] and good["live_outcome"] == good["prior_outcome"] == "normal"
    shifted = live.measurement_agreement("P_noise", {"noise_ratio": 1.01}, {"noise_ratio": 1.0}, None, prereg)
    assert not shifted["scalar_agreement"] and shifted["live_outcome"] == shifted["prior_outcome"]
    changed = live.measurement_agreement("P_noise", {"noise_ratio": 0.1}, {"noise_ratio": 1.0}, None, prereg)
    assert not changed["scalar_agreement"] and changed["live_outcome"] != changed["prior_outcome"]


def test_metered_driver_is_single_case_single_probe(tmp_path, monkeypatch):
    driver = module("metered_driver", "evals/run_probes.py")
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.subprocess, "run", lambda *_a, **_kw: pytest.fail("must fail before GPU"))
    for argv in (["--runs", "baseline_p0c_s2027"],
                 ["--runs", "baseline_p0c_s2027", "h01_s2027", "--probes", "P_physics", "--per-probe-output"]):
        with pytest.raises(SystemExit) as exc:
            driver.main(["--tag", "test", "--metered", *argv])
        assert exc.value.code == 2


@pytest.mark.parametrize("tampered", [False, True])
def test_metered_driver_binds_costs_to_jobs_and_output(tmp_path, monkeypatch, tampered):
    driver = module("metered_driver_artifacts", "evals/run_probes.py")
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(driver, "KEY_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(driver, "T1_KEY_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(driver, "find_run", lambda *_: tmp_path)
    monkeypatch.setattr(driver, "last_checkpoint", lambda *_: tmp_path / "model_299.pt")
    source = tmp_path / "evals/probes.py"
    source.parent.mkdir()
    source.write_bytes((ROOT / "evals/probes.py").read_bytes())
    calls = []

    def fake_isaac(cmd, **_):
        assert cmd[1].endswith("metered_probes.py")
        calls.append(cmd)
        jobs_path = driver.Path(cmd[cmd.index("--jobs") + 1])
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        job = jobs["jobs"][0]
        out = driver.Path(job["output"])
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps({"name": job["name"], "checkpoint": {"sha256": "a" * 64},
                                   "probes": {"P_physics": {"measurement": {"mismatch_count": 0}, "exit": 0}}}), encoding="utf-8")
        metric = {"jobs_sha256": "b" * 64 if tampered else hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
                  "probes_sha256_lf": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                  "completed": True, "cpu_s": 3.25, "simulator_steps_scope": "successful_measure_env_step_calls_times_actual_num_envs",
                  "measurements": [{"probe": "P_physics", "checkpoint_sha256": "a" * 64, "num_envs": 64,
                                    "successful_step_calls": 0, "simulator_steps": 0, "completed": True}]}
        driver.Path(cmd[cmd.index("--metrics") + 1]).write_text(json.dumps(metric), encoding="utf-8")
        return SimpleNamespace(returncode=1)  # Windows launcher 종료 코드와 유효한 Isaac 출력은 분리한다.

    monkeypatch.setattr(driver.subprocess, "run", fake_isaac)
    argv = ["--tag", "meter", "--runs", "p0ccal_NONE_s7", "--probes", "P_physics", "--per-probe-output", "--metered"]
    if tampered:
        with pytest.raises(SystemExit, match="sidecar"):
            driver.main(argv)
        assert not (tmp_path / "evals/results/meter/resources").exists()
    else:
        assert driver.main(argv) == 0
        resource = json.loads((tmp_path / "evals/results/meter/resources/p0ccal_NONE_s7__P_physics.json").read_text(encoding="utf-8"))
        assert resource["cpu_s"] >= 3.25 and resource["simulator_steps"] == 0
        assert resource["launcher_exit_code"] == 1 and resource["checkpoint_sha256"] == "a" * 64
        assert len(list((tmp_path / "evals/results/meter/probes").glob("*.json"))) == 1
    assert len(calls) == 1


def test_persisted_interruption_recovery_and_full_audit_without_gpu(tmp_path, monkeypatch):
    """실제 Ledger·register/finalize·worker를 연결하고 Isaac 호출만 가짜로 바꾼다."""
    from test_behavior_oracle import GATES, report

    live = module("live_persisted", "evals/live_loop_evidence.py")
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live.loop, "ROOT", tmp_path)
    monkeypatch.setattr(live.loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(live, "verify_prereg", lambda *_: None)
    monkeypatch.setattr(live, "gpu_snapshot", lambda: {"utilization_pct": 0, "ollama_models": 0})
    protocol = tmp_path / "bench/protocols/fixed_eval_v1.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((ROOT / "bench/protocols/fixed_eval_v1.json").read_bytes())
    case, probe, tag = "h04_s2027", "P_torque", "offline_test"
    ref = {"P_reward": {"measurement": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0}},
           "P_torque": {"measurement": {"low_speed_saturation": 0.002}}}
    prior = {"probes": {probe: {"measurement": {"low_speed_saturation": 0.2}}}}
    objects = {"prior_probes": prior, "probe_reference": {"probes": ref}, "before": report(case, "b" * 64, lin=0.8),
               "reference": report(), "telemetry": {"series": []}}
    sources = {}
    for name, obj in objects.items():
        path = tmp_path / f"{name}.json"
        live.A._write_new(path, obj)
        sources[name] = {"path": str(path), "sha256_lf": live.sha(path)}
    prereg = {"cases": [{"case": case, "sources": sources, "ranking": ["reward", "actuator"], "ranking_kind": "cached_agent",
                         "checkpoint_sha256": "b" * 64, "seed": 2027}], "top_k": 3, "budget_probes": 4,
              "interrupt_case": case, "absolute_gates": GATES, "limits": {"gpu_wall_s": 480}, "end_to_end_wall_limit_s": 900,
              "diagnosis_scope": "cached", "scalar_fields": {probe: ["low_speed_saturation"]},
              "relative_tolerance": 1e-3, "absolute_tolerance": 1e-8}
    path = tmp_path / "prereg.json"
    live.A._write_new(path, prereg)
    calls = []

    def fake_isaac(c, p, result_tag, **_):
        calls.append((c, p))
        out = live.loop._probe_path(c, p, result_tag)
        live.A._write_new(out, {"name": c, "seed": 2027, "checkpoint": {"sha256": "b" * 64},
                               "probes": {p: {"measurement": {"low_speed_saturation": 0.2}, "exit": 0}}})
        live.A._write_new(live.loop._execution_path(c, p, result_tag),
                         {"case": c, "probe": p, "exit_code": 0, "checkpoint_sha256": "b" * 64,
                          "report_sha256_lf": live.sha(out), "gpu_wall_s": 0.001,
                          "cpu_s": 0.002, "parent_cpu_s": 0.001, "simulator_steps": 14})
        return {"low_speed_saturation": 0.2}, 0.001, 0

    def fake_worker(cmd, **_):
        assert "execute" in cmd and "--interrupt-after-output" in cmd
        rid = cmd[cmd.index("--request-id") + 1]
        try:
            return SimpleNamespace(returncode=live.worker(case, rid, tag, True))
        except SystemExit as exc:
            return SimpleNamespace(returncode=exc.code)

    monkeypatch.setattr(live.loop, "isaac_execute", fake_isaac)
    monkeypatch.setattr(live.subprocess, "run", fake_worker)
    result = live.run_case(path, tag, case)
    assert calls == [(case, probe)] and result["loop_status"] == "confirmed"
    assert result["classification_agreement"] and result["scalar_agreement"]
    assert result["recovery"]["extra_gpu_wall_s"] == 0 and result["recovery"]["artifact_unchanged"]
    assert result["recovery"]["execution_records_before"] == result["recovery"]["execution_records_after"] == 1
    assert result["recovery"]["orphans_after"] == []
    assert all(v is not None for v in result["costs"].values())
    assert result["costs"]["simulator_steps"] == 14 and result["costs"]["model_calls"] == 0
    assert result["end_to_end_wall"]["coverage_complete"]
    audit = live.read(tmp_path / f"evals/results/{tag}/cases/{case}/audit.json")
    assert audit["behavior"]["outcome"] == "abstained" and not audit["repair_execution_verified"]
    _, _, ledger = live.loop._load(case)
    assert sum(e["event"] == "rejected" for e in ledger.events()) == 1
    assert sum(e["event"] == "consumed" for e in ledger.events()) == 1
    assert sum(e["event"] == "receipt" for e in ledger.events()) == 1
