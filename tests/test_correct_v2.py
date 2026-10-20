"""적용성 계약을 거치는 기본 교정 경로의 CPU 수용 검사. 공개 loop 기록은 임시 폴더 사본만 쓴다."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import correct_v2 as V  # noqa: E402

F1_LOOP, U1_LOOP = "h02_s2080", "u1_s2040"


def loop_copy(tmp_path, case, *, with_correction_ledger=False):
    target = tmp_path / case
    target.mkdir()
    names = V.LOOP_FILES + (("correction_ledger.jsonl",) if with_correction_ledger else ())
    for name in names:
        shutil.copyfile(V.LOOPS / case / name, target / name)
    return tmp_path


def rebind(manifest, root):
    """바꾼 manifest 사실로 관측 원본을 다시 써서 원본 SHA 결합만으로는 막히지 않게 한다."""
    record = root / "evidence.json"
    record.write_text(json.dumps({"schema": "combination_observation_v1",
        "combination": {key: manifest[key] for key in ("robot", "task", "fault", "correction_id")},
        "stages": manifest["stages"]}), encoding="utf-8")
    manifest["evidence"] = [{"path": "evidence.json", "sha256_lf": V.K.sha(record)}]


def recovered(calls):
    def execute(request):
        calls.append(request)
        return {"exit_code": 0, "gpu_s": 1.0, "verdict": "recovered", "result_sha": "c" * 64, "tag": request["tag"]}
    return execute


def test_public_loop_replay_reaches_each_exit_without_touching_tracked_records():
    tracked = sorted(p for case in V.REPLAYS for p in (V.LOOPS / case).iterdir())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    out = V.replay()
    assert out["status"] == "pass"
    ready = out["cases"][F1_LOOP]
    assert ready["contract_status"] == "ready" and ready["trace"] == list(V.K.ORDER)
    assert (ready["correction_id"], ready["recovery_mode"], ready["selection_actor"]) == (
        "restore_body_frame_reward", "restart", "model")
    assert out["cases"][U1_LOOP]["trace"] == list(V.K.ORDER[:4])
    assert out["cases"]["e01_s2027"]["contract_status"] is None
    assert out["cases"]["h05_s2027"]["reason"] == "unregistered_mechanism"
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(p for case in V.REPLAYS
            for p in (V.LOOPS / case).iterdir())} == before


def test_ready_request_needs_approval_and_runs_once_with_rechecked_contract(tmp_path):
    loops = loop_copy(tmp_path, F1_LOOP)
    manifest = V._fixture_manifest("go2_f1_ready")
    out = V.propose(F1_LOOP, manifest, "model", loops=loops)
    assert out["decision"] == "proposed" and out["evidence"]["restoration_targets"] == ["track_lin_vel_xy_exp"]
    rid, calls = out["request_id"], []
    with pytest.raises(V.LedgerError):
        V.run(F1_LOOP, rid, recovered(calls), manifest, "model", loops=loops)
    V._ledger(F1_LOOP, loops).approve(rid, "operator")
    receipt = V.run(F1_LOOP, rid, recovered(calls), manifest, "model", loops=loops)
    assert receipt["verdict"] == "recovered" and len(calls) == 1
    assert calls[0]["restoration_targets"] == ["track_lin_vel_xy_exp"] and calls[0]["recovery_mode"] == "restart"
    with pytest.raises(V.LedgerError):
        V.run(F1_LOOP, rid, recovered(calls), manifest, "model", loops=loops)
    with pytest.raises(V.LedgerError):
        V.propose(F1_LOOP, manifest, "model", loops=loops)
    assert len(calls) == 1
    assert [e["event"] for e in V._ledger(F1_LOOP, loops).events()] == ["proposed", "approved", "consumed", "receipt"]


@pytest.mark.parametrize("mutation", [
    lambda m: m["stages"]["source"]["fault"]["checkpoint"].update(sha256="0" * 64),
    lambda m: m["stages"]["scope"]["measurement"].update(track_lin_vel_xy_exp_rel_error=0.0),
    lambda m: m["stages"]["scope"].update(probe="P_torque"),
    lambda m: m.update(V._fixture_manifest("go2_u1_legacy_scope_insufficient")),
    lambda m: m.pop("stages"),
])
def test_manifest_from_another_policy_or_observation_is_refused_before_ledger_write(tmp_path, mutation):
    loops = loop_copy(tmp_path, F1_LOOP)
    manifest = V._fixture_manifest("go2_f1_ready")
    mutation(manifest)
    with pytest.raises(ValueError):
        V.propose(F1_LOOP, manifest, "model", loops=loops)
    assert not (loops / F1_LOOP / "correction_ledger.jsonl").exists()


def test_removing_the_loop_binding_breaks_that_acceptance(tmp_path, monkeypatch):
    loops = loop_copy(tmp_path, F1_LOOP)
    other = V._fixture_manifest("go2_u1_legacy_scope_insufficient")
    monkeypatch.setattr(V, "_bind", lambda *_: None)
    out = V.propose(F1_LOOP, other, "model", loops=loops)
    # 결합 검사가 없으면 다른 정책의 근거가 이 loop의 원장에 HOLD로 기록된다.
    assert out["decision"] == V.HOLD and (loops / F1_LOOP / "correction_ledger.jsonl").exists()


@pytest.mark.parametrize("manifest_name, actor", [(None, "model"), ("go2_f1_ready", None),
                                                  ("go2_f1_ready", "operator")])
def test_confirmed_loop_needs_manifest_and_declared_actor(tmp_path, manifest_name, actor):
    loops = loop_copy(tmp_path, F1_LOOP)
    manifest = None if manifest_name is None else V._fixture_manifest(manifest_name)
    with pytest.raises(ValueError):
        V.propose(F1_LOOP, manifest, actor, loops=loops)
    assert not (loops / F1_LOOP / "correction_ledger.jsonl").exists()


def test_existing_ledger_with_another_registry_is_not_reused(tmp_path):
    loops = loop_copy(tmp_path, F1_LOOP, with_correction_ledger=True)
    before = (loops / F1_LOOP / "correction_ledger.jsonl").read_bytes()
    with pytest.raises(V.LedgerError):
        V.propose(F1_LOOP, V._fixture_manifest("go2_f1_ready"), "model", loops=loops)
    with pytest.raises(V.LedgerError):
        V.status(F1_LOOP, loops)
    assert (loops / F1_LOOP / "correction_ledger.jsonl").read_bytes() == before


def test_loop_state_or_actor_change_after_approval_blocks_run(tmp_path):
    loops = loop_copy(tmp_path, F1_LOOP)
    manifest = V._fixture_manifest("go2_f1_ready")
    rid = V.propose(F1_LOOP, manifest, "model", loops=loops)["request_id"]
    V._ledger(F1_LOOP, loops).approve(rid, "operator")
    calls = []
    with pytest.raises(V.LedgerError):
        V.run(F1_LOOP, rid, recovered(calls), manifest, "registered_rule", loops=loops)
    path = loops / F1_LOOP / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**state, "budget": state["budget"] + 1}), encoding="utf-8")
    with pytest.raises(V.LedgerError):
        V.run(F1_LOOP, rid, recovered(calls), manifest, "model", loops=loops)
    assert calls == [] and V._ledger(F1_LOOP, loops).requests()[rid]["state"] == "approved"


GATE_MUTATIONS = {
    "applicability": (lambda s: s["applicability"]["targets"]["track_lin_vel_xy_exp"].update(effect_reachable=False),
                      "not_applicable"),
    "injection": (lambda s: s["injection"].update(after=deepcopy(s["injection"]["before"])), "invalid"),
    "noop": (lambda s: s["noop"]["after"].update(track_lin_vel_xy_exp="c" * 64), "invalid"),
    "recovery": (lambda s: s["recovery"].update(mode="resume"), "invalid"),
}


def assert_held_at(out, gate, status):
    assert out["decision"] == V.HOLD and out["reason"] == status and out["contract"]["reason"] == gate


@pytest.mark.parametrize("gate", sorted(GATE_MUTATIONS))
def test_each_gate_stops_the_base_runner_and_disabling_it_breaks_acceptance(tmp_path, monkeypatch, gate):
    mutate, status = GATE_MUTATIONS[gate]
    loops = loop_copy(tmp_path, F1_LOOP)
    manifest = V._fixture_manifest("go2_f1_ready")
    mutate(manifest["stages"])
    rebind(manifest, tmp_path)
    assert_held_at(V.propose(F1_LOOP, manifest, "model", loops=loops, root=tmp_path), gate, status)
    assert V._ledger(F1_LOOP, loops).requests() == {}
    monkeypatch.setitem(V.K.GATES, gate, lambda m, row: (None, {"mutation": "gate_disabled"}))
    disabled = V.propose(F1_LOOP, manifest, "model", loops=loops, root=tmp_path)
    with pytest.raises(AssertionError):
        assert_held_at(disabled, gate, status)
    assert disabled["decision"] == "proposed"


def test_unregistered_fault_label_holds_as_invalid_contract(tmp_path):
    loops = loop_copy(tmp_path, F1_LOOP)
    manifest = V._fixture_manifest("go2_f1_ready")
    manifest["fault"] = "F9"
    rebind(manifest, tmp_path)
    out = V.propose(F1_LOOP, manifest, "model", loops=loops, root=tmp_path)
    assert out["decision"] == V.HOLD and out["reason"] == "invalid"
    assert V._ledger(F1_LOOP, loops).requests() == {}


def test_cli_replay_exit_code_and_fixture_wrapper(tmp_path, capsys):
    assert V.main(["replay"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    wrapped = V._read_manifest(V.FIXTURES / "go2_f1_ready.json")
    assert wrapped == V._fixture_manifest("go2_f1_ready")
