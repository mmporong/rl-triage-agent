"""X-R1 고리 단계: 사전등록 예측과 동결 고리의 일치, 방식별 결론·전체 결과, 조건 분기와 실행 흐름(GPU·모델 없이)."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

from rl_triage import probe_loop as L  # noqa: E402
import run_upstream_r1_loop as R  # noqa: E402
import run_upstream_r1_smoke as S  # noqa: E402

U = S.U
PREDICTED = {probe: spec["outcome"] for probe, spec in R.FIXED["prediction"]["probes"].items()}
NAMES = S.RUN_NAMES[1]


def _rankings():
    for top in itertools.permutations(L.MECHANISMS, R.LOOP_TOP_K):
        yield [*top, *[m for m in L.MECHANISMS if m not in top]]


# --- 예측과 동결 고리 ---
def test_the_predicted_table_stops_every_fixed_strategy():
    assert set(PREDICTED) == set(L.PROBES) and set(PREDICTED.values()) == {"normal"}
    rows = R.strategy_rows(PREDICTED, {"status": "model_invalid"})
    assert {row["strategy"]: row["status"] for row in rows if row["strategy"] != "discriminate"} == {
        "fixed": "none_supported", "exhaustive": "none_supported", **{f"random{s}": "none_supported" for s in R.RANDOM_SEEDS}}
    assert R.overall(rows) == "stopped" and not any(row["correct"] or row["wrong"] for row in rows)


def test_any_model_ranking_and_first_probe_ends_none_supported_on_the_predicted_table():
    for ranking in _rankings():
        for model_probe in (None, *L.PROBES):
            run = R.discriminate(PREDICTED, ranking, model_probe)
            assert run["status"] == "none_supported" and run["probes_used"] <= R.LOOP_BUDGET, (ranking, model_probe)


def test_discriminate_without_a_usable_model_probe_equals_the_frozen_simulator():
    tables = [PREDICTED, {**PREDICTED, "P_torque": "abnormal"}, {**PREDICTED, "P_episode": "abnormal"},
              {**PREDICTED, "P_value": "abnormal"}, {**PREDICTED, "P_reward": "unknown"}]
    for table, ranking in itertools.product(tables, _rankings()):
        ours = R.discriminate(table, ranking, None)
        frozen = L.simulate("discriminate", table, ranking=ranking, top_k=R.LOOP_TOP_K, budget=R.LOOP_BUDGET)
        assert (ours["status"], ours["conclusion"], [s["probe"] for s in ours["steps"]]) == (
            frozen["status"], frozen["conclusion"], [s["probe"] for s in frozen["steps"]])


@pytest.mark.parametrize(("probe", "conclusion", "outcome"), [("P_torque", "actuator", "wrong_confirmation"),
                                                               ("P_episode", "termination", "wrong_confirmation"),
                                                               ("P_value", "optimizer", "correct_confirmation")])
def test_the_registered_scenarios(probe, conclusion, outcome):
    table = {**PREDICTED, probe: "abnormal"}
    ranking = [conclusion, *[m for m in L.MECHANISMS if m != conclusion]]
    rows = R.strategy_rows(table, {"status": "valid", "decision": {"action": "continue", "ranking": ranking, "probe": None}})
    assert {(row["status"], row["conclusion"]) for row in rows} == {("confirmed", conclusion)}
    assert R.overall(rows) == outcome
    gate = R.specificity(rows, table)
    assert (set(gate) == {row["strategy"] for row in rows} and all(g["passed"] for g in gate.values())) if conclusion == "actuator" \
        else gate == {}
    assert probe in R.FIXED["prediction"]["scenarios"]


def test_model_states_leave_the_discriminate_cell_without_probes_and_out_of_the_prediction():
    for model, status in (({"status": "model_invalid"}, "model_invalid"), ({"status": "model_failed"}, "model_failed"),
                          ({"status": "valid", "decision": {"action": "hold", "ranking": list(L.MECHANISMS), "probe": None}},
                           "model_hold")):
        rows = R.strategy_rows(PREDICTED, model)
        cell = rows[-1]
        assert (cell["strategy"], cell["status"], cell["probes_used"]) == ("discriminate", status, 0)
        check = R.prediction_check(PREDICTED, rows, R.overall(rows))
        assert "discriminate" not in check["strategies"] and all(check["strategies"].values()) and check["outcome"]


def test_unknown_measurements_and_failed_probes_are_unknown_not_normal():
    reports = {arm: {"probes": _measurements(arm)} for arm in S.ARMS}
    assert R.classify_table(reports) == PREDICTED
    reports["fault"]["probes"]["P_value"]["exit"] = 1
    reports["reference"]["probes"]["P_torque"] = {"exit": 0, "measurement": None}
    table = R.classify_table(reports)
    assert table["P_value"] == "unknown" and table["P_torque"] == "unknown"


def test_the_probe_driver_never_updates_the_policy():
    """R1은 update만 바꾸므로 probe 프로세스에 결함을 걸지 않는다는 전제를 동결 probes.py 본문으로 확인한다."""
    text = (ROOT / "evals/probes.py").read_text(encoding="utf-8")
    assert ".update(" not in text and "load_optimizer=False" in text and "evals/probes.py" in R.CODE


def test_the_protocol_records_the_condition_branch_model_rule_and_predictions():
    fixed = R.FIXED
    assert "condition_not_met" in fixed["condition"]["failure_absent"] and fixed["prediction"]["outcome"] == "stopped"
    assert "재호출하지 않는다" in fixed["model"]["invalid"] and fixed["model"]["id"] == R.MODEL
    assert set(fixed["outcomes"]) == set(fixed["wording"]) == set(R.OUTCOMES)
    assert fixed["prediction"]["strategies"]["fixed"] == fixed["prediction"]["strategies"]["exhaustive"] == "none_supported"
    assert "직접" not in json.dumps(fixed, ensure_ascii=False)
    dynamic = {"source_files": {"a": "1"}, "isaaclab_sources": {}, "smoke_protocol_sha256_lf": "s"}
    protocol = {**copy.deepcopy(fixed), **dynamic}
    R.check_protocol(protocol, dynamic)
    with pytest.raises(ValueError, match="고정 계약"):
        R.check_protocol({**protocol, "no_tuning_rule": "x"}, dynamic)
    with pytest.raises(ValueError, match="smoke"):
        R.check_protocol(protocol, {**dynamic, "smoke_protocol_sha256_lf": "t"})


# --- 실행 흐름(가짜 smoke 결과·Isaac Lab·소유 Job·모델 검토) ---
def _measurements(arm: str, torque: float | None = None, exit_codes: dict | None = None) -> dict:
    reference = arm == "reference"
    values = {"P_noise": {"noise_ratio": 1.0}, "P_value": {"critic_change": 1.45 if reference else 0.6},
              "P_reward": {"track_lin_vel_xy_exp_rel_error": 1e-8, "track_ang_vel_z_exp_rel_error": 1e-8},
              "P_torque": {"low_speed_saturation": 0.0005 if reference else (torque if torque is not None else 0.002)},
              "P_physics": {"mismatch_count": 0}, "P_episode": {"timeout_ratio": 1.0}}
    return {p: {"measurement": v, "exit": (exit_codes or {}).get(p, 0), "elapsed_s": 10.0} for p, v in values.items()}


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    import e2e_reward_demo_v2 as V2
    import run_active_diagnosis as RA
    import run_runtime_readback_smoke as P
    for module, name, value in ((R, "ROOT", tmp_path), (R, "PROTOCOL", tmp_path / "bench/protocols/loop.json"),
                                (R, "PRIVATE", tmp_path / "bench/private/loop"), (R, "RESULTS", tmp_path / "evals/results"),
                                (S, "PROTOCOL", tmp_path / "bench/protocols/smoke.json"),
                                (S, "PRIVATE", tmp_path / "bench/private/smoke"), (S, "RESULTS", tmp_path / "evals/results"),
                                (S, "RUNS_DIR", tmp_path / "bench/runs_upstream"), (RA, "ROOT", tmp_path)):
        monkeypatch.setattr(module, name, value)
    for path, text in ((R.PROTOCOL, '{"loop": 1}'), (S.PROTOCOL, '{"smoke": 1}')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    fake_lab = tmp_path / "IsaacLab"
    runs, telemetry = {}, {}
    for arm, name in NAMES.items():
        run_dir = fake_lab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-15_00-00-00_{name}"
        run_dir.mkdir(parents=True)
        (run_dir / "model_0.pt").write_bytes(b"zero")
        (run_dir / S.FINAL_CHECKPOINT).write_bytes(f"final {name}".encode())
        path = S.telemetry_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": {"Loss/learning_rate": {"mean_last_20pct": 1e-3 if arm == "reference" else 3e-3}},
                                    "series": {}}), encoding="utf-8")
        runs[arm] = {"checkpoint_sha256": U.sha(run_dir / S.FINAL_CHECKPOINT), "telemetry_sha256": U.sha(path)}
    protocol = {"smoke_protocol_sha256_lf": U.sha(S.PROTOCOL, True)}
    state = {"verify_calls": 0, "gpu_busy": False, "gpu_processes": [], "runs": [], "torque": None, "exit": [{}],
             "process_exit": [0], "review": [{"status": "valid", "decision": {
                 "action": "continue", "ranking": ["reward", "actuator", "exploration", "optimizer", "physics", "termination"],
                 "probe": "P_reward"}}]}

    def verify(_isaaclab):
        state["verify_calls"] += 1
        return protocol, "0" * 40
    monkeypatch.setattr(R, "verify", verify)
    monkeypatch.setattr(U, "_admission", lambda: {"busy": state["gpu_busy"], "processes": state["gpu_processes"],
                                                  "busy_reasons": ["other"] if state["gpu_busy"] else []})
    monkeypatch.setattr(U, "_process_created", lambda _pid: 1)
    original_review = V2.review_diagnosis_record
    monkeypatch.setattr(V2, "review_diagnosis_record", lambda record, packet: (
        original_review(record, packet) if state["review"] is None else state["review"].pop(0)))

    def run_owned(folder, args, cwd, watchdog_s):
        assert Path(args[1]).name == "probes.py" and watchdog_s == R.WATCHDOG_S
        jobs = U.read(Path(args[args.index("--jobs") + 1]))
        assert jobs["fault"] == "NONE" and jobs["seed"] == 2042 and len(jobs["jobs"]) == 1  # 실행마다 프로세스 하나
        job = jobs["jobs"][0]
        arm = next(a for a, name in NAMES.items() if name == job["name"])
        assert Path(folder).name == f"run_{arm}" and (arm == "reference" or (Path(folder).parent / "run_reference").is_dir())
        state["runs"].append(arm)
        exits = (state["exit"].pop(0) if state["exit"] else {}) if arm == "fault" else None
        assert job["probes"] == list(R.PROBE_ORDER)
        out = Path(job["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"name": job["name"], "checkpoint": {"file": "model_299.pt",
                                                                         "sha256": U.sha(Path(job["checkpoint"]))},
                                   "seed": 2042, "probes": _measurements(arm, state["torque"], exits)}), encoding="utf-8")
        code = state["process_exit"].pop(0) if state["process_exit"] else 0
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False, "error_type": None,
                "exit_code": code, "cleanup_verified": True, "owned_process_identities": {"4242": 1}, "wall_s": 250.0}
    monkeypatch.setattr(P, "run_owned", run_owned)
    return SimpleNamespace(path=fake_lab, state=state, runs=runs, tmp=tmp_path)


def _smoke(lab, outcome: str, tag: str = "upstream_r1_smoke_20261015") -> None:
    sha = U.sha(S.PROTOCOL, True)
    U.write_new(S.RESULTS / tag / "result.json", {"protocol_sha256_lf": sha, "outcome": outcome, "run_names": NAMES,
                                                  "runs": lab.runs})
    U.write_new(S.PRIVATE / "consumed" / f"{sha[:16]}_final.json", {"tag": tag, "outcome": outcome})


TAG = "upstream_r1_loop_20261015"


def _record(lab, number: int = 1) -> Path:
    path = R.folder(TAG) / "model" / f"diagnosis.call{number}.json"
    path.write_text(json.dumps({"schema": "active_model_call_v1", "total_tokens": 1000, "execution_wall_s": 3.0}),
                    encoding="utf-8")
    return path


def test_the_gate_waits_for_a_final_smoke_result_and_writes_nothing(lab):
    with pytest.raises(ValueError, match="smoke 최종 결과"):
        R.gate(TAG, lab.path)
    assert not R.PRIVATE.exists() and not (R.RESULTS / TAG).exists()


def test_failure_absent_ends_with_condition_not_met_once(lab):
    _smoke(lab, "failure_absent")
    out = R.gate(TAG, lab.path)
    result = U.read(R.RESULTS / TAG / "result.json")
    assert out["status"] == result["outcome"] == "condition_not_met" and result["probes_measured"] == result["model_calls"] == 0
    assert result["wording"] == R.FIXED["wording"]["condition_not_met"]
    with pytest.raises(FileExistsError):
        R.gate("upstream_r1_loop_20261015b", lab.path)  # 프로토콜마다 한 번
    with pytest.raises(FileExistsError):
        R.packet(TAG, lab.path)


def test_a_smoke_result_from_another_protocol_is_refused(lab):
    _smoke(lab, "failure_present")
    result = U.read(S.RESULTS / "upstream_r1_smoke_20261015/result.json")
    (S.RESULTS / "upstream_r1_smoke_20261015/result.json").write_text(json.dumps({**result, "protocol_sha256_lf": "0" * 64}),
                                                                       encoding="utf-8")
    with pytest.raises(ValueError, match="맞지 않는다"):
        R.gate(TAG, lab.path)


def _through_model(lab) -> None:
    _smoke(lab, "failure_present")
    assert R.gate(TAG, lab.path)["status"] == "gate_passed"
    out = R.packet(TAG, lab.path)
    packet = U.read(R.ROOT / out["packet"])
    assert packet["schema"] == "active_decision_packet_v2" and "telemetry:case:Loss/learning_rate" in packet["facts"]
    assert ". \"$HOME/.config/nvidia/env\"" in out["model_command"] and "/mnt/" in out["model_command"]
    assert "upstream_r1" not in json.dumps(packet)
    R.rank(TAG, _record(lab), lab.path)


def test_the_full_path_stops_as_predicted(lab):
    _through_model(lab)
    assert R.probes(TAG, lab.path)["status"] == "complete"
    out = R.summarize(TAG, lab.path)
    result = U.read(R.RESULTS / TAG / "result.json")
    assert out["outcome"] == result["outcome"] == "stopped" and result["counts"] == {"correct": 0, "wrong": 0, "stopped": 8}
    assert all(result["prediction_match"]["probes"].values()) and all(result["prediction_match"]["strategies"].values())
    cell = result["strategies"][-1]
    assert cell["strategy"] == "discriminate" and cell["model_probe_used"] and cell["steps"][0]["probe"] == "P_reward"
    assert result["model_tokens"] == 1000 and result["model_calls"] == 1 and result["specificity_gate"] == {}
    text = json.dumps(result)
    assert "C:" not in text and "LIMMM" not in text
    with pytest.raises(FileExistsError):
        R.summarize(TAG, lab.path)


def test_torque_saturation_gives_a_wrong_confirmation_that_passes_the_specificity_gate(lab):
    _through_model(lab)
    lab.state["torque"] = 0.05
    R.probes(TAG, lab.path)
    result = U.read(R.RESULTS / R.summarize(TAG, lab.path)["result"].split("evals/results/")[1])
    assert result["outcome"] == "wrong_confirmation" and result["table"]["P_torque"] == "abnormal"
    assert all(g["passed"] for g in result["specificity_gate"].values()) and "discriminate" in result["specificity_gate"]


def test_model_invalid_is_recorded_without_a_second_call(lab):
    lab.state["review"] = [{"status": "invalid", "reason": "ranking은 순열이어야 한다"}]
    _through_model(lab)
    with pytest.raises(FileExistsError):
        R.rank(TAG, _record(lab, 2), lab.path)
    R.probes(TAG, lab.path)
    R.summarize(TAG, lab.path)
    result = U.read(R.RESULTS / TAG / "result.json")
    assert result["model"]["status"] == "model_invalid" and result["strategies"][-1]["status"] == "model_invalid"
    assert result["outcome"] == "stopped" and "discriminate" not in result["prediction_match"]["strategies"]


def test_an_infrastructure_error_allows_one_more_call_in_order(lab):
    lab.state["review"] = [{"status": "infra_error", "reason": "empty_or_truncated_response"},
                           lab.state["review"][0]]
    _smoke(lab, "failure_present")
    R.gate(TAG, lab.path)
    R.packet(TAG, lab.path)
    assert R.rank(TAG, _record(lab), lab.path)["status"] == "retry_allowed"
    with pytest.raises(ValueError, match="call2"):
        R.rank(TAG, _record(lab, 3), lab.path)
    assert R.rank(TAG, _record(lab, 2), lab.path)["status"] == "valid"
    assert R.step(TAG, "model")["attempts"] == 2


def _real_record(lab, content: str, number: int = 1, **extra) -> Path:
    """active_model_call_v1 형식의 기록. packet·프롬프트 SHA를 실제 계약 함수로 계산한다."""
    import e2e_reward_demo_v2 as V2
    from rl_triage import active_diagnosis as D
    packet_path = R.ROOT / R.step(TAG, "packet")["path"]
    data = packet_path.read_bytes()
    record = {"schema": "active_model_call_v1", "packet_sha256": hashlib.sha256(data).hexdigest(),
              "messages_sha256": L.digest(D.decision_messages(json.loads(data))),
              "request": {"model": V2.MODEL, "temperature": 0}, "exit_code": 0, "content": content,
              "total_tokens": 900, "execution_wall_s": 2.0, **extra}
    path = R.folder(TAG) / "model" / f"diagnosis.call{number}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _decision(ranking: list[str], probe: str) -> str:
    return json.dumps({"ranking": ranking, "probe": probe, "action": "continue", "reason": "추종 학습이 늦다",
                       "claims": [{"id": "telemetry:case:Loss/learning_rate", "value": 3e-3}],
                       "expected_outcomes": {m: L.PROBES[probe]["expect"][m] for m in ranking[:R.LOOP_TOP_K]}})


def test_the_real_contract_review_accepts_a_valid_record_and_keeps_the_model_probe(lab):
    import e2e_reward_demo_v2 as V2
    assert V2.MODEL == R.MODEL and V2.LOOP_TOP_K == R.LOOP_TOP_K and V2.LOOP_BUDGET == R.LOOP_BUDGET
    lab.state["review"] = None
    _smoke(lab, "failure_present")
    R.gate(TAG, lab.path)
    R.packet(TAG, lab.path)
    ranking = ["optimizer", "reward", "actuator", "exploration", "physics", "termination"]
    assert R.rank(TAG, _real_record(lab, _decision(ranking, "P_value")), lab.path)["status"] == "valid"
    R.probes(TAG, lab.path)
    R.summarize(TAG, lab.path)
    cell = U.read(R.RESULTS / TAG / "result.json")["strategies"][-1]
    assert cell["status"] == "none_supported" and cell["steps"][0] == {
        "probe": "P_value", "outcome": "normal", "dropped": ["optimizer"], "left": ["reward", "actuator"]}


@pytest.mark.parametrize(("content", "extra", "status"), [
    ("not json", {}, "model_invalid"),
    (_decision(["reward", "optimizer", "actuator", "exploration", "physics", "termination"], "P_value")
     .replace('"value": 0.003', '"value": 0.5'), {}, "model_invalid"),  # 사실 값과 다른 claim
    ("", {"infra_error": "empty_or_truncated_response"}, "retry_allowed"),
])
def test_the_real_contract_review_classifies_invalid_and_infrastructure_records(lab, content, extra, status):
    lab.state["review"] = None
    _smoke(lab, "failure_present")
    R.gate(TAG, lab.path)
    R.packet(TAG, lab.path)
    assert R.rank(TAG, _real_record(lab, content, **extra), lab.path)["status"] == status


def test_a_failed_probe_is_measured_once_more_then_invalid(lab):
    _through_model(lab)
    lab.state["exit"] = [{"P_value": 1}, {"P_value": 1}]
    assert R.probes(TAG, lab.path)["status"] == "retry_allowed"
    assert R.probes(TAG, lab.path)["status"] == "final_incomplete"
    with pytest.raises(FileExistsError):
        R.probes(TAG, lab.path)
    R.summarize(TAG, lab.path)
    result = U.read(R.RESULTS / TAG / "result.json")
    assert result["outcome"] == "invalid" and result["failed_checks"] == ["probes_complete"] and result["strategies"] is None


def test_a_crashed_probe_process_is_retried_and_can_complete(lab):
    _through_model(lab)
    lab.state["process_exit"] = [1]  # 정상 쪽 프로세스가 죽으면 R1 쪽은 띄우지 않는다
    assert R.probes(TAG, lab.path)["status"] == "retry_allowed" and lab.state["runs"] == ["reference"]
    assert R.probes(TAG, lab.path)["status"] == "complete" and lab.state["runs"] == ["reference", "reference", "fault"]
    assert set(R.step(TAG, "probes")["process_wall_s"]) == {"reference", "fault"}


def test_probes_refuse_a_changed_checkpoint_or_a_busy_gpu_before_launch(lab):
    _through_model(lab)
    lab.state["gpu_busy"] = True
    assert R.probes(TAG, lab.path)["status"] == "gpu_busy_no_launch" and lab.state["runs"] == []
    lab.state["gpu_busy"] = False
    checkpoint = next((lab.path / "logs/rsl_rl/unitree_go2_flat").glob(f"*_{NAMES['fault']}")) / S.FINAL_CHECKPOINT
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="체크포인트"):
        R.probes(TAG, lab.path)
    assert lab.state["runs"] == []


def test_steps_run_in_order(lab):
    _smoke(lab, "failure_present")
    R.gate(TAG, lab.path)
    with pytest.raises(ValueError, match="packet"):
        R.rank(TAG, R.folder(TAG) / "model" / "diagnosis.call1.json", lab.path)
    with pytest.raises(ValueError, match="probes"):
        R.summarize(TAG, lab.path)
