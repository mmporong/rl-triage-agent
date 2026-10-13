"""D3 실험 실행기: 가짜 Isaac Lab·GPU·소유 Job으로 사례 하나의 전 과정과 관문(전제·원장·캐시·재시도·상호작용)을 고정한다."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import d3_agent as AG  # noqa: E402
import d3_changes as D  # noqa: E402
import d3_methods as MT  # noqa: E402
import run_d3_experiment as X  # noqa: E402

ISAACLAB = Path.home() / "IsaacLab"
P0C_RUNS = ROOT / "evals/results/p0c_fixed_eval_20261009/runs"
TELEMETRY = ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json"
needs_inputs = pytest.mark.skipif(not (ISAACLAB / D.REWARDS[1]).is_file() or not (P0C_RUNS / "h02_s2026.json").is_file(),
                                  reason="로컬 Isaac Lab v2.1.1 소스나 P0-C 고정 평가가 없다")
SMOKE_TAG = "d3_harmless_summary_20261014"


def checkpoint_bytes(seed: int, spec: str) -> bytes:
    return f"ckpt {seed} {spec}".encode()


def readback_for(ids) -> dict:
    code = {cid: {t["qualname"]: {"swapped": True, "bound": True} for t in D.code_targets(cid)}
            for cid in ids if D.CHANGES[cid]["kind"] == "code"}
    config = {cid: {key: True if key.startswith("env.") else "params" for key in D.CHANGES[cid]["overrides"]}
              for cid in ids if D.CHANGES[cid]["kind"] == "config"}
    return {"changes": list(ids), "code": code, "config": config, "ok": True}


def measurement(spec: str) -> dict:
    harmful = set(spec.split(",")) & set(D.HARMFUL)
    reward = bool(harmful & {"U1", "F1"})
    return {"noise_ratio": 1.0, "critic_change": 0.5, "track_lin_vel_xy_exp_rel_error": 0.5 if reward else 0.0,
            "track_ang_vel_z_exp_rel_error": 0.0, "low_speed_saturation": 0.2 if "F2" in harmful else 0.005,
            "mismatch_count": 0, "timeout_ratio": 1.0}


@pytest.fixture(scope="module")
def protocol():
    return json.loads(json.dumps({**X.FIXED, **X.dynamic_fields(ISAACLAB)}))


@pytest.fixture()
def lab(tmp_path, monkeypatch, protocol):
    import run_runtime_readback_smoke as P
    H = X.H
    fake_lab = tmp_path / "IsaacLab"
    for relative in X.isaaclab_files():  # diff 렌더링은 실제 v2.1.1 소스 사본을 읽는다
        (fake_lab / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ISAACLAB / relative, fake_lab / relative)
    for name, value in (("PRIVATE", tmp_path / "private"), ("RESULTS", tmp_path / "results/d3_experiment_v1"),
                        ("RUNS_DIR", tmp_path / "runs"), ("SUMMARIES", tmp_path / "results"),
                        ("PROTOCOL", tmp_path / "d3_experiment_v1.json")):
        monkeypatch.setattr(X, name, value)
    X.PROTOCOL.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(H, "PRIVATE", tmp_path / "smoke_private")
    monkeypatch.setattr(H, "RESULTS", tmp_path / "smoke_results")
    state = {"calls": [], "gpu_busy": False, "timeout": set(), "flip": set(), "verify_calls": 0, "nondeterministic": False,
             "skip_param": None}

    def smoke(outcome="pass", checkpoint=None):
        H.write_new(H.PRIVATE / "consumed" / f"{H.sha(H.PROTOCOL, True)[:16]}_summary.json", {"tag": SMOKE_TAG})
        H.write_new(H.RESULTS / SMOKE_TAG / "summary.json", {"outcome": outcome, "changes": list(D.HARMLESS)})
        if checkpoint is not None:
            tag = "d3_harmless_smoke_20261014_s2026"
            H.write_new(H._consumed(2026, "final"), {"tag": tag, "outcome": "harmless_pass"})
            H.write_new(H.RESULTS / tag / "result.json", {"tag": tag, "checkpoint_sha256": checkpoint,
                                                          "changes": list(D.HARMLESS)})

    def verify(_isaaclab, verify_committed=None):
        state["verify_calls"] += 1
        return json.loads(X.PROTOCOL.read_text(encoding="utf-8")), "0" * 40
    monkeypatch.setattr(X, "verify", verify)
    monkeypatch.setattr(X, "bound_sources_same", lambda _start, _now: True)
    monkeypatch.setattr(X, "_admission", lambda: {"busy": state["gpu_busy"], "processes": [], "busy_reasons": ["x"]})
    monkeypatch.setattr(X, "_process_created", lambda _pid: None)
    monkeypatch.setattr(X.time, "sleep", lambda _s: None)

    def spec_of_dir(run_dir: Path) -> tuple[int, str]:
        seed, spec = (run_dir / X.FINAL_CHECKPOINT).read_bytes().decode().split("|")[0].split(" ", 2)[1:]
        return int(seed), spec

    def load_params(run_dir):
        run_dir = Path(run_dir)
        _, spec = spec_of_dir(run_dir)
        ids = () if spec == "NONE" else tuple(spec.split(","))
        params = {"env.yaml:sim.dt": 0.005, "agent.yaml:run_name": run_dir.name.split("_", 2)[2]}
        for cid in D.CHANGES:
            for key, (old, new) in D.expected_params([cid]).items():
                params[key] = new if cid in ids and key != state["skip_param"] else old
        return params
    monkeypatch.setattr(X, "load_params", load_params)

    def run_owned(folder, args, cwd, watchdog_s):
        name = Path(args[1]).name
        state["calls"].append(name)
        stdout = ""
        if name == "train_with_d3_changes.py":
            spec = os.environ[D.ENV_VAR]
            ids = () if spec == "NONE" else tuple(spec.split(","))
            run_name, seed = args[args.index("--run_name") + 1], int(args[args.index("--seed") + 1])
            run_dir = fake_lab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-14_00-00-00_{run_name}"
            run_dir.mkdir(parents=True)
            noise = f"|{run_name}".encode() if state["nondeterministic"] else b""  # 비결정적이면 실행마다 체크포인트가 다르다
            (run_dir / X.FINAL_CHECKPOINT).write_bytes(checkpoint_bytes(seed, spec) + noise)
            stdout = D.READBACK_PREFIX + json.dumps(readback_for(ids)) + "\n"
        elif name == "extract_tb.py":
            shutil.copy(TELEMETRY, args[3])
        elif name == "fixed_eval.py":
            job = X.read(Path(args[args.index("--jobs") + 1]))["jobs"][0]
            seed, spec = spec_of_dir(Path(job["checkpoint"]).parent)
            bad = bool(set(spec.split(",")) & set(D.HARMFUL)) != (job["name"] in state["flip"])
            report = X.read(P0C_RUNS / ("h02_s2026.json" if bad else f"baseline_p0c_s{seed}.json"))
            report.update(name=job["name"], checkpoint={"file": X.FINAL_CHECKPOINT, "sha256": X.sha(Path(job["checkpoint"]))})
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["output"]).write_text(json.dumps(report), encoding="utf-8")
        elif name == "d3_isaac.py":
            spec = os.environ[D.ENV_VAR]
            ids = () if spec == "NONE" else tuple(spec.split(","))
            jobs = X.read(Path(args[args.index("--jobs") + 1]))
            job = jobs["jobs"][0]
            assert jobs["fault"] == "NONE" and "--headless" in args
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["output"]).write_text(json.dumps({
                "name": job["name"], "checkpoint": {"file": X.FINAL_CHECKPOINT, "sha256": X.sha(Path(job["checkpoint"]))},
                "probes": {p: {"measurement": measurement(spec), "exit": 0} for p in job["probes"]}}), encoding="utf-8")
            stdout = D.READBACK_PREFIX + json.dumps(readback_for(ids)) + "\n"
        (folder / "stdout.log").write_text(stdout, encoding="utf-8")
        timed_out = any(t in str(folder) for t in state["timeout"])
        return {"job_assigned_before_release": True, "natural_exit": not timed_out, "timed_out": timed_out,
                "error_type": None, "exit_code": 124 if timed_out else 0, "cleanup_verified": True,
                "owned_process_identities": {"4242": 1}, "wall_s": 1.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake_lab, state=state, smoke=smoke, tmp=tmp_path)


def _record(case: str, method: str, *, top: str | None = None, n: int = 1, **extra) -> Path:
    packet_path = X.ROOT / X.step(case, "packet")["packets"][method]["path"] if not str(
        X.step(case, "packet")["packets"][method]["path"]).startswith(str(X.PRIVATE)) else None
    packet_path = Path(X.folder(case) / "model" / f"{method}.packet.json")
    packet = json.loads(packet_path.read_bytes())
    spec = X.case_spec(case)
    top = top or AG.labels(8)[spec["harmful_index"]]
    others = [label for label in AG.labels(8) if label != top]
    diff = next(c["diff"] for c in packet["changes"] if c["label"] == top)
    content = {"ranking": [top, *others], "uncertain": False, "reasons": [
        {"change": top, "facts": [next(iter(packet["facts"]))], "probes": ["P_reward"] if method == "B4" else [],
         "diff_lines": [min(AG.new_lines(diff))]}]}
    record = {"schema": "d3_model_call_v1", "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
              "request": {"model": X.MODEL, "temperature": 0}, "exit_code": 0, "total_tokens": 100,
              "content": json.dumps(content), **extra}
    path = X.folder(case) / "model" / f"{method}.call{n}.json"
    X.write_new(path, record)
    return path


def _drive(case: str, lab, approver: str = "operator") -> list[dict]:
    """plan → 승인 → 재학습을 방법이 모두 끝날 때까지 돌린다."""
    retrains = []
    for _ in range(12):
        planned = X.plan(case, lab.path)
        if planned.get("status") == "interaction_observed":
            break
        if planned["proposed"]:
            X.approve(case, [p["request_id"] for p in planned["proposed"]], approver)
        pending = [rid for rid, r in X.ledger(case).requests().items() if r["state"] == "approved"]
        if not pending:
            break
        for rid in pending:
            if X.terminal(case):  # 상호작용이 관측되면 남은 승인 요청은 실행하지 않는다
                return retrains
            retrains.append(X.retrain(case, rid, lab.path))
    return retrains


def _through_ranks(case: str, lab) -> None:
    X.begin(case, lab.path)
    X.setup(case, "none", lab.path)
    X.setup(case, "failure", lab.path)
    X.reference(case, lab.path)
    X.loop(case, lab.path)
    X.packet(case, lab.path)
    X.rank(case, "B4", _record(case, "B4"), lab.path)
    X.rank(case, "B5", _record(case, "B5"), lab.path)


@needs_inputs
def test_a_development_case_runs_end_to_end_with_a_shared_cache(lab):
    lab.smoke(checkpoint=hashlib.sha256(checkpoint_bytes(2026, ",".join(D.HARMLESS))).hexdigest())
    _through_ranks("D0", lab)
    loop = X.step("D0", "loop")
    assert (loop["status"], loop["conclusion"], loop["probes"]) == ("confirmed", "reward", ["P_episode", "P_noise", "P_reward"])
    retrains = _drive("D0", lab)
    assert [r["answer"] for r in retrains] == ["healthy", "healthy", "healthy"]  # 합집합 0xF0·0xFE·0xFC만 학습
    assert lab.state["calls"].count("train_with_d3_changes.py") == 2 + 3
    determinism = X.determinism("D0", lab.path)
    assert (determinism["same"], determinism["source"]) == (True, "smoke")
    result = X.close("D0")
    counts = {m: (row["status"], row["confirmed"], row["cost"]["retrains"]) for m, row in result["methods"].items()}
    assert counts == {"B1": ("confirmed", "U1", 3), "B2": ("confirmed", "U1", 1), "B3": ("confirmed", "U1", 1),
                      "B4": ("confirmed", "U1", 1), "B5": ("confirmed", "U1", 1)}
    assert result["methods"]["B4"]["probe_parent_wall_s"] > 0 and result["methods"]["B5"]["probe_parent_wall_s"] == 0
    assert result["valid"] and result["methods"]["B4"]["model"]["total_tokens"] == 100
    text = json.dumps(result)
    assert "LIMMM" not in text and "C:" not in text
    with pytest.raises(FileExistsError):
        X.close("D0")
    assert X.status("D0")["methods"]["B1"] == {"status": "confirmed", "queried": 3, "next": None}


@needs_inputs
@pytest.mark.parametrize("deterministic", [True, False])
def test_determinism_reruns_when_the_smoke_checkpoint_differs_and_gates_the_main_cases(lab, deterministic):
    lab.smoke(checkpoint="0" * 64)  # smoke 체크포인트가 D0와 다르다
    _through_ranks("D0", lab)
    _drive("D0", lab)
    assert X.determinism("D0", lab.path)["status"] == "rerun_needed"
    if not deterministic:
        lab.state["nondeterministic"] = True
    record = X.determinism("D0", lab.path, rerun=True)
    assert (record["same"], record["source"], record["smoke"]["same"]) == (deterministic, "rerun", False)
    assert record["rerun"]["run_name"].endswith("_rep") and record["rerun"]["answer"] == "trained"
    if deterministic:
        assert X.begin("C2", lab.path)["determinism"] == {"same": True, "source": "rerun"}
    else:
        with pytest.raises(ValueError, match="D0 결정성"):
            X.begin("C2", lab.path)
    with pytest.raises(FileExistsError):
        X.determinism("D0", lab.path, rerun=True)


@needs_inputs
def test_a_retrain_whose_overrides_did_not_reach_the_params_is_an_injection_unknown(lab):
    lab.smoke()
    _through_ranks("D0", lab)
    lab.state["skip_param"] = "agent.yaml:algorithm.entropy_coef"
    rid = X.plan("D0", lab.path)["proposed"][0]["request_id"]
    X.approve("D0", [rid], "operator")
    out = X.retrain("D0", rid, lab.path)
    assert (out["answer"], out["cause"], out["final"]) == ("unknown", "injection", True)
    assert "fixed_eval.py" not in lab.state["calls"][-2:]  # 학습 측 검사 실패면 평가하지 않는다
    with pytest.raises(ValueError, match="끝나지 않은"):
        X.close("D0")


@needs_inputs
def test_begin_needs_a_passed_smoke_and_main_cases_need_determinism_and_unused_seeds(lab):
    with pytest.raises(ValueError, match="smoke"):
        X.begin("D0", lab.path)
    lab.smoke(outcome="fail")
    with pytest.raises(ValueError, match="pass"):
        X.begin("D0", lab.path)
    shutil.rmtree(X.H.PRIVATE)
    shutil.rmtree(X.H.RESULTS)
    lab.smoke()
    with pytest.raises(ValueError, match="D0 결정성"):
        X.begin("C1", lab.path)
    X.write_new(X.step_path("D0", "determinism"), {"same": True, "source": "rerun"})
    X.RUNS_DIR.mkdir(parents=True)
    (X.RUNS_DIR / "other_s2035.telemetry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="2035"):
        X.begin("C1", lab.path)
    assert X.begin("C2", lab.path)["seed_evidence"]["hits"] == []
    with pytest.raises(FileExistsError):
        X.begin("C2", lab.path)


@needs_inputs
def test_a_failure_run_that_is_healthy_makes_the_case_invalid(lab):
    lab.smoke()
    X.begin("D0", lab.path)
    X.setup("D0", "none", lab.path)
    lab.state["flip"].add(X.run_name("D0", 0xFF))
    X.setup("D0", "failure", lab.path)
    assert X.terminal("D0")["outcome"] == "case_invalid"
    with pytest.raises(ValueError, match="끝났다"):
        X.plan("D0", lab.path)
    result = X.close("D0")
    assert (result["valid"], result["invalid_reason"]) == (False, "case_invalid")


@needs_inputs
def test_retrains_need_an_approval_consumed_once_and_the_cache_is_never_trained_twice(lab):
    lab.smoke()
    _through_ranks("D0", lab)
    planned = X.plan("D0", lab.path)
    rid = planned["proposed"][0]["request_id"]
    assert X.plan("D0", lab.path)["proposed"] == []  # 대기 중인 요청은 다시 제안하지 않는다
    with pytest.raises(ValueError, match="승인된"):
        X.retrain("D0", rid, lab.path)
    with pytest.raises(ValueError, match="승인자"):
        X.approve("D0", [rid], " ")
    X.approve("D0", [rid], "operator")
    lab.state["gpu_busy"] = True
    with pytest.raises(X.GPUBusy):
        X.retrain("D0", rid, lab.path)
    assert X.ledger("D0").requests()[rid]["state"] == "approved"  # GPU 사용 중이면 승인을 소비하지 않는다
    lab.state["gpu_busy"] = False
    assert X.retrain("D0", rid, lab.path)["final"] is True
    with pytest.raises(ValueError, match="승인된"):
        X.retrain("D0", rid, lab.path)
    mask = X.ledger("D0").requests()[rid]["mask"]
    book = X.ledger("D0")
    again = book.propose(mask, X.ids_of(X.case_spec("D0")["order"], mask), ["test"])
    book.approve(again["request_id"], "operator")
    with pytest.raises(ValueError, match="캐시"):
        X.retrain("D0", again["request_id"], lab.path)


@needs_inputs
def test_an_infrastructure_failure_is_retried_once_then_leaves_the_method_unresolved(lab):
    lab.smoke()
    _through_ranks("D0", lab)
    target = X.run_name("D0", MT.single_revert(8, 0))
    lab.state["timeout"] = {target}
    _drive("D0", lab)
    retried = [r for r in X.ledger("D0").requests().values() if r.get("repeat_of")]
    assert len(retried) == 1 and X.entry("D0", MT.single_revert(8, 0))["last"]["attempt"] == 2
    assert X.entry("D0", MT.single_revert(8, 0))["answer"] == "unknown"  # 두 번째도 인프라 실패(이름에 _a2가 붙어도 timeout)
    states = X.method_states("D0")
    assert states["B2"]["status"] == "unresolved" and states["B2"]["reason"] == "unanswered"


@needs_inputs
def test_a_harmless_only_subset_that_is_unhealthy_stops_the_case_as_interaction_observed(lab):
    lab.smoke()
    _through_ranks("D0", lab)
    lab.state["flip"].add(X.run_name("D0", 0b11110000))  # 해로운 변경(위치 1)이 빠졌는데 unhealthy
    _drive("D0", lab)
    assert X.terminal("D0")["outcome"] == "interaction_observed"
    assert X.terminal("D0")["detail"]["masks"] == [0b11110000]
    assert X.close("D0")["invalid_reason"] == "interaction_observed"


@needs_inputs
def test_model_calls_retry_only_infrastructure_errors_and_an_invalid_ranking_leaves_b4_unresolved(lab):
    lab.smoke()
    X.begin("D0", lab.path)
    for arm in ("none", "failure"):
        X.setup("D0", arm, lab.path)
    X.reference("D0", lab.path)
    X.loop("D0", lab.path)
    out = X.packet("D0", lab.path)
    assert "d3_model_call.py" in out["wsl_commands"]["B4"] and X.MODEL in out["wsl_commands"]["B4"]
    assert X.rank("D0", "B4", _record("D0", "B4", infra_error="APITimeoutError", exit_code=2), lab.path)["status"] == "infra_error"
    assert X.rank("D0", "B4", _record("D0", "B4", n=2), lab.path)["status"] == "valid"
    assert X.rank("D0", "B5", _record("D0", "B5", n=1, content="nope"), lab.path)["status"] == "model_invalid"
    states = X.method_states("D0")
    assert states["B5"]["reason"] == "model_invalid" and states["B4"]["status"] == "pending"
    with pytest.raises(FileExistsError):
        X.rank("D0", "B5", _record("D0", "B5", n=2), lab.path)


@needs_inputs
def test_a_packet_with_clues_is_not_written(lab, monkeypatch):
    lab.smoke()
    X.begin("D0", lab.path)
    for arm in ("none", "failure"):
        X.setup("D0", arm, lab.path)
    X.reference("D0", lab.path)
    X.loop("D0", lab.path)
    monkeypatch.setattr(X.AG, "clues", lambda packet: ["messages:id:U1"])
    with pytest.raises(ValueError, match="단서"):
        X.packet("D0", lab.path)
    assert not (X.folder("D0") / "model").exists()


@needs_inputs
def test_an_orphan_request_is_closed_as_an_infrastructure_unknown_and_retried(lab, monkeypatch):
    lab.smoke()
    _through_ranks("D0", lab)
    rid = X.plan("D0", lab.path)["proposed"][0]["request_id"]
    X.approve("D0", [rid], "operator")

    def crash(*_args, **_kwargs):
        raise KeyboardInterrupt  # 학습 도중 실행기가 죽음
    monkeypatch.setattr(X, "_run_subset", crash)
    with pytest.raises(KeyboardInterrupt):
        X.retrain("D0", rid, lab.path)
    assert X.ledger("D0").orphans() == [rid]
    with pytest.raises(ValueError, match="결과 없이"):
        X.recover("D0", "000000000000")
    receipt = X.recover("D0", rid)
    assert (receipt["answer"], receipt["cause"], receipt["reason"]) == ("unknown", "infrastructure", "orphan_without_attempt")
    assert any(r.get("repeat_of") == rid for r in X.ledger("D0").requests().values()) is False
    monkeypatch.undo()


def test_the_retrain_ledger_reuses_the_correction_ledger_event_order(tmp_path):
    book = X.RetrainLedger(tmp_path / "ledger.jsonl", "a" * 64, "D0")
    row = book.propose(3, ("H1", "H2"), ["B1"])
    with pytest.raises(X.LedgerError, match="진행 중"):
        book.propose(3, ("H1", "H2"), ["B2"])
    with pytest.raises(X.LedgerError):
        book.consume(row["request_id"], 3)  # 승인 전
    book.approve(row["request_id"], "operator")
    with pytest.raises(X.LedgerError, match="부분집합"):
        book.consume(row["request_id"], 4)
    book.consume(row["request_id"], 3)
    with pytest.raises(X.LedgerError):
        book.receipt(row["request_id"], "recovered", None, None, 1.0)
    assert book.receipt(row["request_id"], "healthy", None, "b" * 64, 10_000.0)["answer"] == "unknown"  # 예산 초과
    other = X.RetrainLedger(tmp_path / "ledger.jsonl", "c" * 64, "D0")  # 다른 프로토콜의 승인은 묶이지 않는다
    second = book.propose(3, ("H1", "H2"), ["B1"], repeat_of=row["request_id"])
    with pytest.raises(X.LedgerError, match="등록 표"):
        other.approve(second["request_id"], "operator")
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "ledger.jsonl").write_text("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n", encoding="utf-8")
    with pytest.raises(X.LedgerError):
        book.requests()


def test_summarize_needs_every_main_case_and_writes_one_summary(tmp_path, monkeypatch):
    for name, value in (("PRIVATE", tmp_path / "private"), ("RESULTS", tmp_path / "results/d3_experiment_v1"),
                        ("SUMMARIES", tmp_path / "results"), ("PROTOCOL", tmp_path / "p.json")):
        monkeypatch.setattr(X, name, value)
    X.PROTOCOL.write_text("{}", encoding="utf-8")

    def write_case(case: str, counts: dict[str, int], valid: bool = True):
        spec = X.case_spec(case)
        harmful = spec["harmful"]
        methods = {m: {"status": "confirmed", "reason": None, "confirmed": harmful,
                       "queries": [{"reverted": [], "answer": "healthy"}] * counts[m]} for m in MT.METHODS}
        X.write_new(X.RESULTS / case / "case_result.json", {"aggregate": spec["aggregate"], "valid": valid,
                                                           "invalid_reason": None if valid else "case_invalid",
                                                           "methods": methods})
    write_case("C1", {"B1": 4, "B2": 6, "B3": 4, "B4": 1, "B5": 2})
    with pytest.raises(ValueError, match="C2"):
        X.summarize("d3_experiment_summary_20261020")
    write_case("C2", {"B1": 3, "B2": 5, "B3": 4, "B4": 1, "B5": 1})
    write_case("C3", {"B1": 3, "B2": 7, "B3": 2, "B4": 1, "B5": 3})
    with pytest.raises(ValueError, match="tag"):
        X.summarize("summary")
    assert X.summarize("d3_experiment_summary_20261020")["outcome"] == "agent_fewer_retrains"
    summary = X.read(tmp_path / "results/d3_experiment_summary_20261020/summary.json")
    assert summary["sums"]["B4"] == 3 and summary["development_cases"] == []
    with pytest.raises(FileExistsError):
        X.summarize("d3_experiment_summary_20261021")


@needs_inputs
def test_the_protocol_fixes_cases_positions_predictions_and_rules(protocol):
    assert {c: (v["seed"], v["harmful"], v["position"]) for c, v in protocol["cases"].items()} == {
        "D0": (2026, "U1", 1), "C1": (2035, "U1", 2), "C2": (2036, "F1", 7), "C3": (2037, "F2", 2)}
    assert protocol["predictions"]["computed"]["C3"] == {"B1": 4, "B2": 2, "B3": 1}
    assert protocol["version_history"]["v1"]["status"] == "시작하지 않음" and X.PROTOCOL.name == "d3_experiment_v2.json"
    assert all("S4" in c["order"] and "H2" not in c["order"] for c in protocol["cases"].values())
    assert protocol["cap"] == 8 and protocol["model"]["id"] == X.MODEL
    assert set(protocol["rendered_diffs"]) == set(D.CHANGES)
    assert all(c["order"][c["position"] - 1] == c["harmful"] and len(c["order"]) == 8 for c in protocol["cases"].values())
    from build_replay_page import claim_violations
    assert not [p for text in protocol["outcomes"].values() for p in claim_violations(text)]
    assert "C:" not in json.dumps(protocol) and "LIMMM" not in json.dumps(protocol)
    X.check_protocol(protocol, {k: protocol[k] for k in X.DYNAMIC})
    with pytest.raises(ValueError, match="고정 계약"):
        X.check_protocol({**protocol, "cap": 9}, {k: protocol[k] for k in X.DYNAMIC})


@needs_inputs
def test_a_written_protocol_file_is_current(protocol):
    if not X.PROTOCOL.is_file():
        pytest.skip("프로토콜을 아직 prepare하지 않았다")
    X.check_protocol(X.read(X.PROTOCOL), {k: protocol[k] for k in X.DYNAMIC})


def test_main_reports_a_busy_gpu_and_requires_the_case(monkeypatch, capsys):
    def busy(*_args):
        raise X.GPUBusy("x")
    monkeypatch.setattr(X, "setup", busy)
    assert X.main(["setup", "--case", "D0", "--arm", "none"]) == 3
    assert "gpu_busy_no_launch" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        X.main(["plan"])
    assert X.main(["plan", "--case", "Z9"]) == 2


def test_run_names_and_subsets_are_readable_and_never_carry_change_ids():
    order = X.case_spec("C1")["order"]
    assert X.ids_of(order, 0) == () and X.spec_of(()) == "NONE"
    assert X.ids_of(order, MT.single_revert(8, X.case_spec("C1")["harmful_index"])) == tuple(D.HARMLESS)
    assert X.run_name("C1", 0xDF) == "d3v2_c1_s2035_mdf" and X.run_name("C1", 0xDF, 2) == "d3v2_c1_s2035_mdf_a2"
    assert not re.search(r"\b(H[1-7]|U1|F1|F2)\b", X.run_name("C1", 0xDF, rep=True))
    assert subprocess.list2cmdline(X.train_command(ISAACLAB, 2035, "n")).count("U1") == 0
