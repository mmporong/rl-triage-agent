"""T1 짝 연결·blind 경계·고정 평가 제외 규칙을 GPU·모델 없이 확인한다."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def key():
    return {"secret_seed": 987654321, "holdout_seeds": [2026, 2027, 2028],
            "cases": {"e01": {"reference": "ref_e01", "note": "private cause one"},
                      "e02": {"reference": "baseline_p0c", "note": "private cause two"}}}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def setup_inputs(root):
    w = load("bench/build_workspace_t1.py", "workspace_t1_test")
    w.ROOT = root
    logs = root / "logs"
    for cell in w.cells(key()):
        for name in (cell["run"], cell["reference"]):
            value = 20 if name.startswith("baseline") else 10
            write(w.telemetry(name), {"run_dir_name": f"now_{name}", "summary": {"value": value}})
        params = logs / f"now_{cell['reference']}" / "params"
        params.mkdir(parents=True, exist_ok=True)
        for name in ("env.yaml", "agent.yaml"):
            (params / name).write_bytes(f"seed: {cell['seed']}\r\n".encode())
    return w, logs


def test_workspaces_use_per_case_reference_and_preserve_bytes(tmp_path):
    w, logs = setup_inputs(tmp_path)
    w.register(key(), logs)
    w.build(key())
    manifest = w.load_manifest(key())
    assert len(manifest["cells"]) == 6
    for cell in manifest["cells"]:
        ws = w.validate_cell(cell, key(), workspace=True)
        ref = json.loads((ws / "reference/telemetry.json").read_text())
        assert ref["run_dir_name"] == f"now_{cell['reference']}"
        assert ref["summary"]["value"] == (10 if cell["case_id"] == "e01" else 20)
        assert b"\r\n" in (ws / "reference/params/env.yaml").read_bytes()
        assert not any("answer" in p.name for p in ws.rglob("*"))
    with pytest.raises(ValueError, match="이미"):
        w.register(key(), logs)
    with pytest.raises(ValueError, match="이미"):
        w.build(key())


def test_changed_reference_or_workspace_input_is_rejected(tmp_path):
    w, logs = setup_inputs(tmp_path)
    w.register(key(), logs)
    w.build(key())
    cell = w.load_manifest(key())["cells"][0]
    ws = w.validate_cell(cell, key(), workspace=True)
    (ws / "reference/telemetry.json").write_text("{}")
    with pytest.raises(ValueError, match="등록 사본"):
        w.validate_cell(cell, key(), workspace=True)
    (tmp_path / cell["reference_telemetry"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        w.validate_cell(cell, key())


def test_leaked_answer_or_absolute_path_rejected(tmp_path):
    w, logs = setup_inputs(tmp_path)
    w.register(key(), logs)
    w.build(key())
    cell = w.load_manifest(key())["cells"][0]
    ws = tmp_path / cell["workspace"]
    write(ws / "answer_key.json", {"secret_seed": 987654321})
    with pytest.raises(ValueError, match="유출 검사"):
        w.validate_cell(cell, key(), workspace=True)
    path = tmp_path / "bad.json"
    path.write_text('{"source":"/home/someone/private"}')
    with pytest.raises(ValueError, match="경계 위반"):
        w.check_text(path, key())


def fixed(name, *, lin=0.2, fall=0):
    return {"name": name, "protocol": "fixed_eval_v1", "protocol_sha256_lf": "hash", "eval_seed": 2026,
            "num_envs": 1040, "horizon_steps": 1000, "step_dt": 0.02, "action_scale": 0.25,
            "standstill": {"lin_vel_rmse_mps": 1.0},
            "metrics": {"lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": 0.3,
                        "fall_rate": fall, "mean_survival_s": 20.0}}


def test_summary_excludes_bad_references_and_separates_failure_absent(tmp_path):
    s = load("evals/t1_holdout_summary.py", "t1_summary_test")
    k = key(); k["cases"].pop("e02")
    for seed in k["holdout_seeds"]:
        ref, run = f"ref_e01_s{seed}", f"e01_s{seed}"
        write(tmp_path / "runs" / f"{ref}.json", fixed(ref, fall=0.06 if seed == 2028 else 0))
        write(tmp_path / "runs" / f"{run}.json", fixed(run, lin=0.24 if seed == 2026 else 0.2))
    result = s.summarize(tmp_path, k, "hash")
    assert (result["planned"], result["valid"], result["unhealthy"], result["failure_absent"]) == (3, 2, 1, 1)
    assert result["runs"][0]["failed"] == ["err_xy_ratio"]
    assert result["runs"][2]["included"] is False and result["runs"][2]["exclusion_reason"]
    (tmp_path / "runs/e01_s2028.json").unlink()
    with pytest.raises(ValueError, match="누락"):
        s.summarize(tmp_path, k, "hash")


def test_summary_rejects_mixed_fixed_protocols(tmp_path):
    s = load("evals/t1_holdout_summary.py", "t1_summary_protocol_test")
    k = key(); k["cases"].pop("e02")
    for seed in k["holdout_seeds"]:
        for prefix in ("e01", "ref_e01"):
            name = f"{prefix}_s{seed}"
            r = fixed(name)
            if prefix == "e01" and seed == 2028:
                r["action_scale"] = 1.0
            write(tmp_path / "runs" / f"{name}.json", r)
    with pytest.raises(ValueError, match="프로토콜"):
        s.summarize(tmp_path, k, "hash")


def test_rankings_require_all_six_unique_mechanisms():
    r = load("evals/t1_rankings.py", "t1_rankings_test")
    assert r.valid_ranking(list(r.R.MECHANISMS))
    assert not r.valid_ranking(["physics"] * 6)
    assert not r.valid_ranking(["physics", "reward"])
    assert not r.valid_ranking(None)
    assert not r.valid_ranking([{}] * 6)


def ranking_fixture(tmp_path, monkeypatch):
    w, logs = setup_inputs(tmp_path)
    w.register(key(), logs)
    w.build(key())
    write(tmp_path / "bench/private/answer_key_t1.json", key())
    r = load("evals/t1_rankings.py", "t1_ranking_driver_test")
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "W", w)
    monkeypatch.setattr(r, "code_version", lambda *a: {"git_head": "committed", "git_dirty": False})
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "required")
    monkeypatch.setattr(r.E, "MODEL", r.MODEL)
    return r


def test_agent_runner_preserves_all_cells_and_keeps_raw_output_private(tmp_path, monkeypatch):
    r = ranking_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setenv("NVIDIA_API_KEY", "example-key-do-not-use")
    monkeypatch.setattr(r.E, "analysis_canary", lambda *a: {"denied": True, "sandbox": "landlock"})

    def agent(ws, cid):
        calls.append((ws.name, cid))
        ranking = list(r.R.MECHANISMS)
        return {"ranking": ranking, "suspected": ranking[0], "analysis_calls": 1, "analysis_sandboxed": 1,
                "exit_code": 0, "elapsed_s": 1.0, "stdout_tail": "example-key-do-not-use",
                "stderr_tail": "private raw diagnostics"}

    monkeypatch.setattr(r.E, "run_agent_blind", agent)
    assert r.main(["--mode", "agent", "--tag", "test_agent"]) == 0
    assert len(calls) == 6 and len(set(calls)) == 6
    folder = tmp_path / "evals/results/test_agent"
    rows = [json.loads(line) for p in folder.glob("seed*.jsonl") for line in p.read_text().splitlines()]
    assert len(rows) == 6 and all(row["valid_ranking"] for row in rows)
    assert json.loads((folder / "rankings.json").read_text())["completed_cells"] == 6
    assert all("stdout_tail" not in row and "stderr_tail" not in row for row in rows)
    private = tmp_path / "bench/private/test_agent/e01_s2026_attempt1.json"
    assert "example-key-do-not-use" not in private.read_text() and "<redacted>" in private.read_text()
    with pytest.raises(SystemExit, match="이미"):
        r.main(["--mode", "agent", "--tag", "test_agent"])


def test_failed_canary_prevents_model_call(tmp_path, monkeypatch):
    r = ranking_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(r.E, "analysis_canary", lambda *a: {"denied": False, "sandbox": "unavailable"})
    monkeypatch.setattr(r.E, "run_agent_blind", lambda *a: pytest.fail("canary 실패 뒤 모델 호출"))
    with pytest.raises(SystemExit, match="canary 실패"):
        r.main(["--mode", "agent", "--tag", "test_blocked"])
    assert not list((tmp_path / "evals/results/test_blocked").glob("seed*.jsonl"))


def test_uncommitted_inputs_prevent_canary_and_model(tmp_path, monkeypatch):
    r = ranking_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(r, "code_version", lambda *a: {"git_head": "committed", "git_dirty": True})
    monkeypatch.setattr(r.E, "analysis_canary", lambda *a: pytest.fail("미커밋 입력으로 canary 실행"))
    monkeypatch.setattr(r.E, "run_agent_blind", lambda *a: pytest.fail("미커밋 입력으로 모델 호출"))
    with pytest.raises(SystemExit, match="커밋"):
        r.main(["--mode", "agent", "--tag", "test_uncommitted"])
    assert not (tmp_path / "evals/results/test_uncommitted").exists()
