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
                  "cpu_scope": "isaac_python_lifetime_through_env_close_before_app_close", "phase": "before_app_close",
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
        assert not resource["app_close_cpu_included"]
        assert len(list((tmp_path / "evals/results/meter/probes").glob("*.json"))) == 1
    assert len(calls) == 1


def test_metrics_survive_process_shutdown_that_never_returns(tmp_path):
    import subprocess
    import sys

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    jobs = tmp_path / "jobs.json"
    jobs.write_text('{"jobs":[]}', encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    script = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import metered_probes as meter
env=SimpleNamespace(unwrapped=SimpleNamespace(num_envs=7), step=lambda _: 'observation')
meter.probes.measure=lambda p,e,r,c: e.step('a') or {'ok': True}
meter.probes.run=lambda args: meter.probes.measure('P_noise',env,None,Path(sys.argv[2]))
def shutdown_main():
    meter.probes.run(None)
    os._exit(0)
meter.probes.main=shutdown_main
meter.main(['--jobs',sys.argv[3],'--metrics',sys.argv[4]])
"""
    result = subprocess.run([sys.executable, "-c", script, str(ROOT / "evals"), str(checkpoint), str(jobs), str(metrics)])
    assert result.returncode == 0
    report = json.loads(metrics.read_text(encoding="utf-8"))
    assert report["completed"] and report["phase"] == "before_app_close"
    assert report["cpu_s"] > 0 and report["measurements"][0]["simulator_steps"] == 7
    assert not metrics.with_suffix(".post_close.json").exists()


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


@pytest.mark.parametrize("tamper", [None, "artifact", "ledger"])
def test_amendment_pins_artifacts_and_allows_only_ledger_append(tmp_path, monkeypatch, tamper):
    live = module("live_amendment", "evals/live_loop_evidence.py")
    monkeypatch.setattr(live, "ROOT", tmp_path)
    previous, artifact, ledger = tmp_path / "v1.json", tmp_path / "report.json", tmp_path / "ledger.jsonl"
    previous.write_text('{}\n', encoding="utf-8")
    artifact.write_text('{"original":true}\n', encoding="utf-8")
    ledger.write_bytes(b'{"event":"first"}\r\n')
    prefix = ledger.read_bytes().replace(b"\r\n", b"\n")
    prereg = {"amendment": {"previous_protocol": "v1.json", "previous_sha256_lf": live.sha(previous),
                           "prior_artifacts": {"report": {"path": "report.json", "sha256_lf": live.sha(artifact)}},
                           "ledger_prefix": {"path": "ledger.jsonl", "bytes_lf": len(prefix), "line_count": 1,
                                             "sha256_lf": hashlib.sha256(prefix).hexdigest()}}}
    live.verify_amendment(prereg)
    ledger.write_bytes(ledger.read_bytes() + b'{"event":"later"}\n')
    live.verify_amendment(prereg)
    if tamper == "artifact":
        artifact.write_text('{"original":false}\n', encoding="utf-8")
    if tamper == "ledger":
        ledger.write_bytes(ledger.read_bytes().replace(b"first", b"other"))
    if tamper:
        with pytest.raises(ValueError, match="증거|prefix"):
            live.verify_amendment(prereg)


def test_resume_preserves_first_failure_nulls_and_never_repeats_probe(tmp_path, monkeypatch):
    from test_behavior_oracle import GATES, report

    resume = module("resume_evidence", "evals/resume_live_loop_evidence.py")
    live = resume.live
    monkeypatch.setattr(resume, "ROOT", tmp_path)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live.loop, "ROOT", tmp_path)
    monkeypatch.setattr(live.loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(live, "verify_prereg", lambda path, prereg: live.verify_amendment(prereg))
    monkeypatch.setattr(live, "gpu_snapshot", lambda: {"gpu": "fake"})
    case, tag = "h01_s2027", "offline_resume"
    folder = tmp_path / f"evals/results/{tag}/cases/{case}"
    reference = {"P_noise": {"measurement": {"noise_ratio": 1.0}}, "P_reward": {"measurement": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0}}}
    objects = {"prior_probes": {"probes": reference}, "probe_reference": {"probes": reference},
               "reference": report(), "before": report(case, "b" * 64, lin=0.8)}
    sources = {}
    for name, obj in objects.items():
        path = tmp_path / f"{name}.json"
        live.A._write_new(path, obj)
        sources[name] = {"path": str(path), "sha256_lf": live.sha(path)}
    for name, obj in (("gates", GATES), ("limits", {"gpu_wall_s": 480}), ("information", {"ranking": ["exploration", "optimizer", "reward"]})):
        live.A._write_new(folder / f"{name}.json", obj)
    live.loop.start(case, ["exploration", "optimizer", "reward"], 3, 4, "b" * 64,
                    {p: r["measurement"] for p, r in reference.items()})
    contract = live.A.register(SimpleNamespace(loop_dir=live.loop._dir(case), reference=tmp_path / "reference.json", normals=[],
                                               gates=folder / "gates.json", limits=folder / "limits.json", information=folder / "information.json",
                                               method="cached", wall_budget=900, protocol=ROOT / "bench/protocols/fixed_eval_v1.json"))
    live.A._write_new(folder / "contract.json", contract)
    _, state, ledger = live.loop._load(case)
    live.loop.reject(case, state["pending"], live.ACTOR, "first proposal")
    _, state, ledger = live.loop._load(case)
    rid = state["pending"]
    ledger.approve(rid, live.ACTOR)
    live.loop.run(case, rid, lambda *_: (None, 18.4, 1))
    first = live.loop._probe_path(case, "P_noise", tag)
    live.A._write_new(first, {"name": case, "seed": 2027, "checkpoint": {"sha256": "b" * 64}, "probes": reference})
    live.A._write_new(live.loop._execution_path(case, "P_noise", tag), {"exit_code": 1, "gpu_wall_s": 18.4})
    snapshot = folder / "state_snapshot.json"
    snapshot.write_bytes((live.loop._dir(case) / "state.json").read_bytes())
    previous = tmp_path / "bench/protocols/live_loop_v1.json"
    live.A._write_new(previous, {})
    data = ledger.path.read_bytes().replace(b"\r\n", b"\n")
    prereg = {"cases": [{"case": case, "sources": sources, "checkpoint_sha256": "b" * 64, "seed": 2027}],
              "diagnosis_scope": "cached", "scalar_fields": {"P_noise": ["noise_ratio"], "P_reward": ["track_lin_vel_xy_exp_rel_error", "track_ang_vel_z_exp_rel_error"]},
              "relative_tolerance": 1e-3, "absolute_tolerance": 1e-8,
              "amendment": {"case": case, "tag": tag, "previous_protocol": str(previous), "previous_sha256_lf": live.sha(previous),
                            "prior_artifacts": {"state_snapshot": {"path": str(snapshot), "sha256_lf": live.sha(snapshot)}, "report": {"path": str(first), "sha256_lf": live.sha(first)}},
                            "ledger_prefix": {"path": str(ledger.path), "bytes_lf": len(data), "line_count": len(data.splitlines()), "sha256_lf": hashlib.sha256(data).hexdigest()}}}
    prereg_path = tmp_path / "prereg.json"
    live.A._write_new(prereg_path, prereg)
    calls = []

    def fake_isaac(c, p, result_tag, **_):
        assert p == "P_reward"
        calls.append(p)
        out = live.loop._probe_path(c, p, result_tag)
        live.A._write_new(out, {"name": c, "seed": 2027, "checkpoint": {"sha256": "b" * 64}, "probes": {p: reference[p]}})
        live.A._write_new(live.loop._execution_path(c, p, result_tag), {"exit_code": 0, "gpu_wall_s": 1.0, "cpu_s": 2.0, "parent_cpu_s": 0.1, "simulator_steps": 14})
        return reference[p]["measurement"], 1.0, 0

    def fake_worker(cmd, **_):
        return SimpleNamespace(returncode=live.worker(case, cmd[cmd.index("--request-id") + 1], tag, False))

    monkeypatch.setattr(live.loop, "isaac_execute", fake_isaac)
    monkeypatch.setattr(live.subprocess, "run", fake_worker)
    original = first.read_bytes()
    result = resume.resume_case(prereg_path, tag, case)
    assert calls == ["P_reward"] and first.read_bytes() == original
    assert result["observed"] == {"P_noise": "unknown", "P_reward": "normal"}
    assert result["raw_classification_agreement"] and not result["classification_agreement"]
    assert result["costs"]["gpu_wall_s"] == 19.4
    assert result["costs"]["cpu_s"] is None and result["costs"]["simulator_steps"] is None
    assert result["costs"]["execution_wall_s"] is None and not result["end_to_end_wall"]["coverage_complete"]
    with pytest.raises(ValueError, match="덮어쓰지"):
        resume.resume_case(prereg_path, tag, case)
    assert calls == ["P_reward"]
