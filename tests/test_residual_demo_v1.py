"""GPU 없이 두 번째 진단 고리의 상태 전이와 산출물 결합을 재생한다."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import residual_demo_v1 as D  # noqa: E402
import residual_faults_v1 as FAULTS  # noqa: E402


def finished(seconds=1.0):
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True, "wall_s": seconds}


NORMAL_MEASUREMENTS = {
    "P_value": {"critic_change": 1.0}, "P_noise": {"noise_ratio": 1.0},
    "P_episode": {"timeout_ratio": 1.0}, "P_physics": {"mismatch_count": 0},
    "P_torque": {"low_speed_saturation": 0.0},
    "P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0},
}


class Backend:
    """실파일 SHA를 만들고 검증하되 학습·평가의 행동값만 저장 공개 결과에서 재생한다."""

    def __init__(self, *, source_healthy=False, first_healthy=False, unknown_probe=False):
        self.source_healthy = source_healthy
        self.first_healthy = first_healthy
        self.unknown_probe = unknown_probe
        self.calls = []
        self.artifacts = {}

    def preflight(self, isaaclab):
        self.calls.append(("preflight",))
        return {"processes_started": 0, "gpu_queries": 0, "cpu_fixture": True}

    def train(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("train", arm, tuple(binding["applied_corrections"])))
        stage.mkdir(parents=True, exist_ok=False)
        checkpoint = stage / "model_299.pt"
        checkpoint.write_bytes(f"{seed}:{arm}:{binding['applied_corrections']}".encode())
        telemetry = stage / "telemetry.json"
        summary = {"Metrics/base_velocity/error_vel_xy": {"mean_last_20pct": 1.2 if arm != "NONE" else .15}}
        D.write_new(telemetry, {"summary": summary, "series": {}})
        manifest, readback, records = stage / "manifest.json", stage / "readback.json", stage / "records.jsonl"
        D.write_new(manifest, {"binding": binding, "arm": arm, "seed": seed, "run_name": run_name})
        D.write_new(readback, {"original_faults": binding["original_faults"],
                               "applied_corrections": binding["applied_corrections"]})
        records.write_text("{\"update_index\":0}\n{\"update_index\":1}\n{\"update_index\":2}\n", encoding="utf-8")
        run_dir = stage / "run"
        run_dir.mkdir()
        original = list(binding["original_faults"])
        applied = list(binding["applied_corrections"])
        remaining = list(FAULTS.remaining_faults(original, applied))
        status = "abnormal" if "R1" in remaining else "normal"
        artifact = {"arm": arm, "seed": seed, "run_name": run_name,
            "original_faults": original, "applied_corrections": applied, "remaining_faults": remaining,
            "checkpoint_loaded": False, "source_checkpoint_sha256": binding["source_checkpoint_sha256"],
            "checkpoint": {"path": str(checkpoint), "sha256": D.sha(checkpoint)},
            "telemetry": {"path": str(telemetry), "sha256": D.sha(telemetry), "summary": summary},
            "manifest_path": str(manifest), "readback_path": str(readback), "records_path": str(records),
            "readback": D.read(readback), "run_dir": str(run_dir),
            "inspection": {"status": status, "optimizer_steps": [4 if status == "abnormal" else 20] * 3,
                           "minibatches": [4 if status == "abnormal" else 20] * 3},
            "report": {"status": status}, "configuration": {"seed": seed, "max_iterations": 300,
                "num_learning_epochs": 5, "num_mini_batches": 4},
            "checkpoint_evidence": {"final": {"sha256": D.sha(checkpoint)}},
            "execution": finished(10), "checkpoint_execution": finished(), "extract_execution": finished()}
        self.artifacts[arm] = artifact
        return artifact

    def validate_artifact(self, artifact, binding, arm, seed):
        assert artifact["arm"] == arm and artifact["seed"] == seed and artifact["checkpoint_loaded"] is False
        assert D.read(Path(artifact["manifest_path"])) == {
            "binding": binding, "arm": arm, "seed": seed, "run_name": artifact["run_name"]}
        for key in ("manifest_path", "readback_path", "records_path"):
            assert Path(artifact[key]).is_file()
        for key in ("checkpoint", "telemetry"):
            assert D.sha(Path(artifact[key]["path"])) == artifact[key]["sha256"]
        assert artifact["remaining_faults"] == list(FAULTS.remaining_faults(
            artifact["original_faults"], artifact["applied_corrections"]))
        assert all(D.E.V1._finished(artifact[name]) for name in
                   ("execution", "checkpoint_execution", "extract_execution"))
        return {"valid": True}

    def _healthy(self, arm):
        return (arm in {"NONE", "BOTH"} or arm == "COMPOUND" and self.source_healthy
                or arm == "FIRST" and (self.first_healthy or not self.artifacts[arm]["remaining_faults"]))

    def fixed_evaluate(self, stage, isaaclab, artifacts, run_name, binding):
        self.calls.append(("evaluate", tuple(artifacts)))
        stage.mkdir(parents=True, exist_ok=False)
        reports, paths = {}, {}
        for arm, artifact in artifacts.items():
            source = "upstream_r1_none_s2042" if self._healthy(arm) else "upstream_r1_s2042"
            report = deepcopy(D.read(ROOT / f"evals/results/upstream_r1_smoke_20261014/runs/{source}.json"))
            report["name"] = f"{run_name}_{arm.lower()}"
            report["checkpoint"] = {"file": "model_299.pt", "sha256": artifact["checkpoint"]["sha256"]}
            path = stage / f"{arm}.json"
            D.write_new(path, report)
            reports[arm], paths[arm] = report, str(path)
        return {"reports": reports, "report_paths": paths, "execution": finished(2)}

    def legacy_probe(self, stage, isaaclab, artifacts, run_name, binding, probes=None):
        arm, artifact = next(iter(artifacts.items()))
        selected = list(probes or D.C.L.PROBES)
        self.calls.append(("legacy_probe", arm, tuple(selected)))
        stage.mkdir(parents=True, exist_ok=False)
        rows = {}
        for name in selected:
            measurement = deepcopy(NORMAL_MEASUREMENTS[name])
            exit_code = 0
            if name == "P_reward" and "F1" in artifact["remaining_faults"]:
                measurement["track_lin_vel_xy_exp_rel_error"] = .4
            if self.unknown_probe and arm != "NONE":
                measurement, exit_code = None, 1
            rows[name] = {"measurement": measurement, "exit": exit_code}
        return {"report": {"probes": rows}, "execution": finished()}

    def inspect(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("inspect", arm, tuple(binding["applied_corrections"])))
        stage.mkdir(parents=True, exist_ok=False)
        remaining = FAULTS.remaining_faults(binding["original_faults"], binding["applied_corrections"])
        status = "abnormal" if "R1" in remaining else "normal"
        if self.unknown_probe and arm != "NONE":
            status = "unknown"
        steps = [4 if status == "abnormal" else 20] * 3
        return {"measurement": {"status": status, "optimizer_steps": steps, "minibatches": steps},
                "execution": finished()}


def model_caller(order="reward_first", *, invalid=False, hold=False):
    calls = []

    def complete(messages, params):
        calls.append(messages)
        if invalid:
            content = "{}"
        else:
            bodies = []
            for row in messages:
                try:
                    bodies.append(json.loads(row["content"]))
                except (json.JSONDecodeError, TypeError):
                    pass
            body = next((row for row in bodies if "epoch" in row), None)
            if body is not None:
                epoch = body["epoch"]
                first = ("reward" if order == "reward_first" else "optimizer") if epoch == 1 else (
                    "optimizer" if order == "reward_first" else "reward")
                ranking = [first, *[name for name in D.C.L.MECHANISMS if name != first]]
                probe = "P_reward" if first == "reward" else D.C.NEW_PROBE
                fact = next(iter(body["evidence_facts"]))
                decision = {"ranking": ranking, "probe": None if hold else probe,
                    "action": "hold" if hold else "continue", "reason": "등록 근거 확인",
                    "claims": [{"id": fact, "value": body["evidence_facts"][fact]}],
                    "expected_outcomes": {} if hold else {
                        name: body["probe_catalog"][probe]["expect"][name] for name in ranking[:3]}}
                content = json.dumps(decision)
            else:
                body = next(row for row in bodies if "confirmed_mechanism" in row)
                proposal = {"correction_id": body["facts"]["registered_correction"],
                    "recovery_mode": "restart", "hold_reason": None,
                    "cited_facts": ["confirmed_subtype"],
                    "checklist": {key: "rule_mode" for key in body["skill_checklist"]}}
                content = json.dumps(proposal)
        return {"response_id": f"r{len(calls)}", "provider_model": D.C.MODEL, "finish_reason": "stop",
                "content": content, "reasoning_content": None, "input_tokens": 20,
                "output_tokens": 10, "total_tokens": 30, "wall_s": .1}

    setattr(complete, "calls", calls)
    return complete


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    original_sha = D.sha
    monkeypatch.setattr(D, "sha", lambda path, lf=False: "b" * 64 if Path(path) == D.PROTOCOL else original_sha(path, lf))
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return protocol


def execute(protocol, backend, caller):
    return D.execute("residual_v1_20261017", Path("unused"), "operator", True,
                     backend=backend, caller=caller,
                     seed_guard=lambda *args, **kwargs: {"status": "fresh", "seeds": list(D.SEEDS)})


@pytest.mark.parametrize("order", ["reward_first", "optimizer_first"])
def test_two_correction_orders_recover_with_second_epoch_receipt(setup, order):
    backend, caller = Backend(), model_caller(order)
    result = execute(setup, backend, caller)
    assert result["status"] == "complete" and result["demo_success"] is True
    assert result["api"]["reserved"] == 8 and result["api"]["pending"] == 0
    for case in result["cases"]:
        assert case["outcome"] == "recovered" and case["second_epoch_requests"] == 1
        expected = ["reward", D.C.COUNT] if order == "reward_first" else [D.C.COUNT, "reward"]
        assert [row["confirmed_component"] for row in case["epochs"]] == expected
        assert case["epochs"][0]["verified_receipt"]["verdict"] == "not_recovered"
        assert case["epochs"][1]["verified_receipt"]["verdict"] == "recovered"


def test_source_failure_absent_uses_no_model_request(setup):
    caller = model_caller()
    result = execute(setup, Backend(source_healthy=True), caller)
    assert [row["outcome"] for row in result["cases"]] == ["source_failure_absent"] * 2
    assert result["api"]["reserved"] == 0 and getattr(caller, "calls") == []


@pytest.mark.parametrize(("faults", "order", "expected_correction"), [
    (("F1",), "reward_first", D.C.REWARD), (("R1",), "optimizer_first", D.C.EPOCHS),
])
def test_single_fault_controller_replay_has_no_second_request(setup, faults, order, expected_correction):
    """단일 mask를 controller에 넣고 교정으로 잔여 결함이 사라지는 CPU fixture를 재생한다."""
    backend = Backend()
    folder = D.PRIVATE / "single" / "s2110"
    result = D.run_case(folder, "residual_v1_20261017", 2110, Path("unused"), setup, "c" * 40,
                        "operator", folder.parent / "api", backend=backend, caller=model_caller(order),
                        original_faults=faults)
    assert result["outcome"] == "recovered_after_one" and result["second_epoch_requests"] == 0
    assert D.MC.audit_budget(folder.parent / "api")["reserved"] == 2
    first_masks = [row for row in backend.calls if row[0] == "train" and row[1] == "FIRST"]
    assert first_masks and all(row[2] == (expected_correction,) for row in first_masks)
    assert backend.artifacts["FIRST"]["original_faults"] == list(faults)
    assert backend.artifacts["FIRST"]["remaining_faults"] == []
    assert not (folder / "epoch2").exists()


def test_none_mask_controller_replay_has_no_model_or_correction_request(setup):
    backend, caller = Backend(), model_caller()
    folder = D.PRIVATE / "none" / "s2110"
    result = D.run_case(folder, "residual_v1_20261017", 2110, Path("unused"), setup, "c" * 40,
                        "operator", folder.parent / "api", backend=backend, caller=caller, original_faults=())
    assert result["outcome"] == "source_failure_absent" and result["epochs"] == []
    assert getattr(caller, "calls") == [] and not (folder / "epoch1").exists()
    assert list(backend.artifacts) == ["NONE"]


def test_unknown_probe_and_model_hold_both_stop_without_correction(setup):
    unknown = execute(setup, Backend(unknown_probe=True), model_caller())
    assert all(row["outcome"] == "HOLD" and row["reason"] == "unconfirmed_or_unregistered_component"
               for row in unknown["cases"])
    # 새 tag root로 분리한다.
    D.PRIVATE = D.PRIVATE.parent / "hold-runs"
    held_backend = Backend()
    held = execute(setup, held_backend, model_caller(hold=True))
    assert all(row["outcome"] == "HOLD" and row["reason"] == "diagnosis_hold" for row in held["cases"])
    assert not any(call[0] == "train" and call[1] in {"FIRST", "BOTH"} for call in held_backend.calls)


def test_invalid_model_response_is_hold_and_never_falls_back_to_rule(setup):
    backend = Backend()
    result = execute(setup, backend, model_caller(invalid=True))
    assert all(row["outcome"] == "HOLD" and row["reason"] == "diagnosis_invalid" for row in result["cases"])
    assert result["api"]["reserved"] == 4
    assert not any(call[0] == "train" and call[1] in {"FIRST", "BOTH"} for call in backend.calls)


def test_global_budget_20_stops_demo_before_model_callback_21(setup):
    class CapBackend(Backend):
        filled = False

        def train(self, stage, *args, **kwargs):
            artifact = super().train(stage, *args, **kwargs)
            if artifact["arm"] == "COMPOUND" and not self.filled:
                api_root = Path(stage).parents[2] / "api"
                for index in range(20):
                    complete = D.MC.MeteredComplete(api_root, "diagnosis", "a" * 64,
                        f"prefill/{index}", lambda messages, params: {
                            "response_id": "prefill", "provider_model": D.C.MODEL,
                            "finish_reason": "stop", "content": "{}", "reasoning_content": None,
                            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "wall_s": .01})
                    complete([{"role": "user", "content": str(index)}], {"model": D.C.MODEL})
                self.filled = True
            return artifact

    backend, callback_calls = CapBackend(), []
    result = execute(setup, backend, lambda *args: callback_calls.append(1))
    assert result["cases"][0]["outcome"] == "HOLD"
    assert result["cases"][0]["reason"] == "api_budget_or_pending" and callback_calls == []
    assert result["api"]["reserved"] == 20 and result["api"]["remaining"] == 0


@pytest.mark.parametrize("target", ["checkpoint", "evaluation"])
def test_tampered_first_receipt_files_block_second_epoch(setup, monkeypatch, target):
    original = D.verify_repair
    seen = 0

    def verify_then_tamper(path, *args, **kwargs):
        nonlocal seen
        seen += 1
        value = original(path, *args, **kwargs)
        if seen == 1:
            bundle = D.read(path)
            changed = (Path(bundle["artifact"]["checkpoint"]["path"]) if target == "checkpoint" else
                       Path(bundle["evaluation"]["report_paths"]["FIRST"]))
            changed.write_bytes(changed.read_bytes() + b" ")
        return value

    monkeypatch.setattr(D, "verify_repair", verify_then_tamper)
    with pytest.raises(ValueError, match="SHA"):
        execute(setup, Backend(), model_caller())
    saved = D.read(D.PRIVATE / "residual_v1_20261017/result.json")
    assert saved["status"] == "invalid" and saved["failure_type"] == "ValueError"


def test_forged_correction_ledger_receipt_is_rejected(setup, monkeypatch):
    original = D.CorrectionLedger.receipt
    forged = False

    def forge(self, *args, **kwargs):
        nonlocal forged
        row = original(self, *args, **kwargs)
        if not forged:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            event = json.loads(lines[-1])
            event["result_sha"] = "0" * 64
            lines[-1] = json.dumps(event, ensure_ascii=False, sort_keys=True)
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            forged = True
        return row

    monkeypatch.setattr(D.CorrectionLedger, "receipt", forge)
    with pytest.raises(ValueError, match="receipt"):
        execute(setup, Backend(), model_caller())
    assert D.read(D.PRIVATE / "residual_v1_20261017/result.json")["failure_type"] == "ValueError"


def test_over_budget_receipt_becomes_unknown_and_never_opens_epoch_two(setup):
    class Expired(Backend):
        def train(self, stage, *args, **kwargs):
            artifact = super().train(stage, *args, **kwargs)
            if artifact["arm"] == "FIRST":
                artifact["execution"]["wall_s"] = D.CORRECTION_BUDGET_S + 1
            return artifact

    result = execute(setup, Expired(), model_caller())
    assert all(row["outcome"] == "HOLD" and row["reason"] == "correction_receipt_unknown"
               and row["second_epoch_requests"] == 0 for row in result["cases"])


def test_missing_sealed_receipt_is_invalid_and_cannot_open_epoch_two(setup, monkeypatch):
    original = D.seal_repair

    def remove_after_seal(path, **kwargs):
        value = original(path, **kwargs)
        Path(path).unlink()
        return value

    monkeypatch.setattr(D, "seal_repair", remove_after_seal)
    with pytest.raises(FileNotFoundError):
        execute(setup, Backend(), model_caller())
    saved = D.read(D.PRIVATE / "residual_v1_20261017/result.json")
    assert saved["status"] == "invalid" and saved["failure_type"] == "FileNotFoundError"


def test_existing_tag_is_never_reused(setup):
    backend = Backend(source_healthy=True)
    execute(setup, backend, model_caller())
    before = list(backend.calls)
    with pytest.raises(FileExistsError, match="기존 tag"):
        execute(setup, backend, model_caller())
    assert backend.calls == before


def test_correction_approval_binds_accepted_model_evidence(setup, monkeypatch):
    original = D.CorrectionLedger.propose

    def wrong_model(self, *args, **kwargs):
        kwargs["evidence"] = deepcopy(kwargs["evidence"])
        kwargs["evidence"]["accepted_model"]["model_review_digest"] = "0" * 64
        return original(self, *args, **kwargs)

    monkeypatch.setattr(D.CorrectionLedger, "propose", wrong_model)
    with pytest.raises(ValueError, match="원장 결합"):
        execute(setup, Backend(), model_caller())
    folder = D.PRIVATE / "residual_v1_20261017" / "s2110"
    assert not (folder / "epoch2").exists()


def test_model_response_receipt_is_revalidated_before_second_epoch(setup, monkeypatch):
    original = D.verify_repair
    seen = 0

    def corrupt_api_after_validation(path, *args, **kwargs):
        nonlocal seen
        value = original(path, *args, **kwargs)
        seen += 1
        if seen == 1:
            root = Path(D.read(path)["api_root"])
            response_path = root / "responses/0002.json"
            response = D.read(response_path)
            response["result"]["content"] = "{}"
            response_path.write_text(json.dumps(response), encoding="utf-8")
        return value

    monkeypatch.setattr(D, "verify_repair", corrupt_api_after_validation)
    with pytest.raises(ValueError):
        execute(setup, Backend(), model_caller())
    assert not (D.PRIVATE / "residual_v1_20261017/s2110/epoch2").exists()
