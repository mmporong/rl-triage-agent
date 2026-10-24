"""실제 상태 전이·승인 원장과 가짜 GPU backend로 재개의 중복 실행을 확인한다."""
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_resumable_demo_v1 as D  # pyright: ignore[reportMissingImports]
from test_ppo_ratio_repair_demo_v3 import Backend as PriorBackend  # pyright: ignore[reportMissingImports]


class Backend(PriorBackend):
    def train(self, stage, isaaclab, arm, seed, run_name, binding):
        value = super().train(stage, isaaclab, arm, seed, run_name, binding)
        checkpoint = Path(value["checkpoint"]["path"])
        checkpoint.write_bytes((value["arm"] + " CPU fixture").encode())
        value["checkpoint"]["sha256"] = D.E.sha(checkpoint)
        return value

    def fixed_evaluate(self, stage, isaaclab, artifacts, run_name, binding):
        value = super().fixed_evaluate(stage, isaaclab, artifacts, run_name, binding)
        for arm, report in value["reports"].items():
            report["checkpoint"]["sha256"] = artifacts[arm]["checkpoint"]["sha256"]
            Path(value["report_paths"][arm]).write_text(json.dumps(report), encoding="utf-8")
        return value

    def inspect(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("inspect", arm, run_name))
        stage.mkdir(parents=True)
        return {"measurement": {"count_status": "normal", "gradient_status": (
            self.fault_gradient if arm == D.FAULT else self.control_gradient),
            "first_mismatch_update": 0 if arm == D.FAULT else None,
            "optimizer_steps": [20] * 3, "minibatches": [20] * 3}, "execution": {"wall_s": 45.0}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    original_sha = D.E.sha
    monkeypatch.setattr(D.E, "sha", lambda p, lf=False: "b" * 64 if p == D.PROTOCOL else original_sha(p, lf))
    return tmp_path


def caller(clock, replies, http, first_probe="P_reward"):
    def run(folder, packet, binding):
        journal = D.J.DecisionJournal(folder / "decision", folder / "api", binding,
                                     D.SELECT.CALL.F.request_params(D.SELECT.MODEL))

        def complete(sent, params):
            http.append(sent)
            kind = replies.pop(0)
            if kind == "500":
                return {"error_type": "InternalServerError", "infra_error": "InternalServerError",
                        "status_code": 500, "wall_s": 0.2}
            if kind == "invalid":
                content = "{}"
            else:
                content = json.dumps({"action": "continue", "probe": first_probe,
                                      "fact_ids": [next(iter(packet["facts"]))]})
            return {"content": content, "provider_model": D.SELECT.MODEL, "finish_reason": "stop",
                    "input_tokens": 20, "output_tokens": 10, "total_tokens": 30, "wall_s": 0.5}

        return journal.advance(lambda callback: D.SELECT.engine(D.SELECT.build_snapshot(packet), callback),
                               complete, lambda: clock[0])
    return run


def execute(backend, callback):
    return D.execute("ppo_resumable_s2200_20261025", Path("unused"), "operator", True,
        backend=backend, caller=callback, seed_guard=lambda *a, **k: {"status": "fresh"})


def test_pause_new_controller_resume_reuses_training_evaluation_and_api(setup):
    backend, http = Backend(), []
    clock = [datetime(2026, 10, 25, tzinfo=UTC)]
    callback = caller(clock, ["500", "valid"], http)
    first = execute(backend, callback)
    assert first["status"] == "paused_infra" and len(http) == 1
    before = list(backend.calls)
    assert execute(backend, callback)["status"] == "paused_infra"
    assert backend.calls == before and len(http) == 1
    clock[0] += timedelta(seconds=31)
    final = execute(backend, callback)
    assert final["outcome"] == "recovered" and final["demo_success"]
    assert len(http) == 2
    assert len([row for row in backend.calls if row[0] == "train"]) == 3
    assert len([row for row in backend.calls if row[0] == "evaluate"]) == 2
    assert final["probe_steps"][0]["selected_by"] == "model"
    assert all(row["selected_by"] == "registered_rule" for row in final["probe_steps"][1:])
    calls = list(backend.calls)
    assert execute(backend, callback) == final
    assert backend.calls == calls and len(http) == 2
    folder = D.PRIVATE / final["run_id"]
    ledger = D.OLD.CorrectionLedger(folder / "correction.ledger.jsonl", D.C.correction_registry())
    assert [row["event"] for row in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]


def test_two_invalid_responses_stop_before_gpu_probe_or_correction(setup):
    backend, http = Backend(), []
    result = execute(backend, caller([datetime.now(UTC)], ["invalid", "invalid"], http))
    assert result["outcome"] == "HOLD" and len(http) == 2
    assert not result["probe_steps"]
    assert not any(row[:2] == ("train", "CORRECTED") for row in backend.calls)


def test_changed_completed_checkpoint_blocks_before_retry(setup):
    backend, http = Backend(), []
    clock = [datetime.now(UTC)]
    callback = caller(clock, ["500", "valid"], http)
    assert execute(backend, callback)["status"] == "paused_infra"
    checkpoint = next(D.PRIVATE.rglob("model_299.pt"))
    checkpoint.write_bytes(b"changed")
    clock[0] += timedelta(seconds=31)
    with pytest.raises(ValueError, match="SHA"):
        execute(backend, callback)
    assert len(http) == 1


def test_partial_training_is_never_automatically_repeated(setup):
    class Broken(Backend):
        def train(self, stage, isaaclab, arm, seed, run_name, binding):
            self.calls.append(("crash",))
            raise RuntimeError("child status unknown")

    backend = Broken()
    with pytest.raises(RuntimeError, match="unknown"):
        execute(backend, None)
    before = list(backend.calls)
    with pytest.raises(RuntimeError, match="blocked_stage_unknown"):
        execute(backend, None)
    assert backend.calls == before


def test_old_result_folder_without_new_manifest_is_not_adopted(setup):
    folder = D.PRIVATE / "ppo_resumable_s2200_20261025"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text('{"outcome":"HOLD"}', encoding="utf-8")
    with pytest.raises(ValueError, match="인수"):
        execute(Backend(), None)


def test_concurrent_controller_is_rejected_before_work(setup):
    folder = D.PRIVATE / "ppo_resumable_s2200_20261025"
    folder.mkdir(parents=True)
    (folder / ".controller_active").touch()
    backend = Backend()
    with pytest.raises(D.J.JournalError):
        execute(backend, None)
    assert backend.calls == []


def test_replaced_controller_lock_is_preserved_for_inspection(tmp_path):
    path = tmp_path / ".controller_active"
    with pytest.raises(D.J.JournalIntegrityError), D.exclusive(tmp_path, {"run": "fixture"}):
        replacement = D.read(path)
        replacement["nonce"] = "changed"
        path.write_text(json.dumps(replacement), encoding="utf-8")
    assert path.is_file()


@pytest.mark.parametrize("which", ["model", "ledger"])
def test_deleting_evidence_does_not_create_a_fresh_journal_or_approval(setup, monkeypatch, which):
    backend, http = Backend(), []
    callback = caller([datetime.now(UTC)], ["valid"], http)
    advance = D.C.advance

    def interrupt(*args):
        raise KeyboardInterrupt("between stages")

    monkeypatch.setattr(D.C, "advance", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(backend, callback)
    monkeypatch.setattr(D.C, "advance", advance)
    folder = D.PRIVATE / "ppo_resumable_s2200_20261025"
    if which == "model":
        (folder / "model/decision/manifest.json").unlink()
    else:
        (folder / "probes.ledger.jsonl").unlink()
    before = list(backend.calls)
    with pytest.raises(ValueError, match="SHA|원장"):
        execute(backend, callback)
    assert backend.calls == before and len(http) == 1


def test_terminal_approval_evidence_cannot_be_removed(setup):
    backend, http = Backend(), []
    callback = caller([datetime.now(UTC)], ["valid"], http)
    assert execute(backend, callback)["demo_success"]
    next(D.PRIVATE.rglob("correction.ledger.jsonl")).unlink()
    with pytest.raises(ValueError, match="SHA|원장"):
        execute(backend, callback)
    assert len(http) == 1


def test_restart_between_completed_probe_and_next_stage_does_not_consume_twice(setup, monkeypatch):
    backend, http = Backend(), []
    callback = caller([datetime.now(UTC)], ["valid"], http)
    advance = D.C.advance

    def interrupt(*args):
        raise KeyboardInterrupt("controller ended between stages")

    monkeypatch.setattr(D.C, "advance", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(backend, callback)
    calls = list(backend.calls)
    monkeypatch.setattr(D.C, "advance", advance)
    final = execute(backend, callback)
    assert final["demo_success"] and len(http) == 1
    for row in calls:
        if row[0] in ("train", "evaluate", "legacy_probe"):
            assert backend.calls.count(row) == 1
    folder = D.PRIVATE / final["run_id"]
    ledger = D.C.M.RegistryLedger(folder / "probes.ledger.jsonl", D.C.registry())
    assert len(ledger.requests()) == len(final["probe_steps"])


def test_bad_correction_closes_approval_with_unknown_receipt(setup):
    backend, http = Backend(corrected_gradient="abnormal"), []
    final = execute(backend, caller([datetime.now(UTC)], ["valid"], http))
    assert final["outcome"] == "correction_unknown"
    folder = D.PRIVATE / final["run_id"]
    ledger = D.OLD.CorrectionLedger(folder / "correction.ledger.jsonl", D.C.correction_registry())
    assert ledger.events()[-1]["event"] == "receipt"
    assert not any(row[:2] == ("evaluate", ("CORRECTED",)) for row in backend.calls)


def test_overbudget_receipt_cannot_be_reported_as_recovered(setup):
    class Overbudget(Backend):
        def train(self, stage, isaaclab, arm, seed, run_name, binding):
            value = super().train(stage, isaaclab, arm, seed, run_name, binding)
            if arm == "CORRECTED":
                value["execution"]["wall_s"] = D.OLD.CORRECTION_BUDGET_S + 1
            return value

    http = []
    final = execute(Overbudget(), caller([datetime.now(UTC)], ["valid"], http))
    assert final["outcome"] == "correction_unknown" and not final["demo_success"]
    assert final["conditions"]["corrected_fixed_eval_healthy"]
    assert not final["conditions"]["approved_correction_once"]


@pytest.mark.parametrize("control,fault", [("abnormal", "abnormal"), ("normal", "normal")])
def test_nonseparating_observation_never_repairs(setup, control, fault):
    backend, http = Backend(control_gradient=control, fault_gradient=fault), []
    result = execute(backend, caller([datetime.now(UTC)], ["valid"], http))
    assert result["outcome"] == "HOLD"
    assert not any(row[:2] == ("train", "CORRECTED") for row in backend.calls)
