"""모터 결함(F2) 회복 방식 비교의 CPU 회귀: 고정 계약, 판정·budget curve, 규칙 갱신 결정, 실행 커밋 고정, 공개 재생."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    sys.path.insert(0, str(directory))

import build_replay_page as PAGE  # noqa: E402
import run_training_correction_motor as W  # noqa: E402
import test_publish_training_correction_v4 as PT  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

SEED = 2026
COMMIT = "a" * 40


def _strings(node, key=None):
    if key == "forbidden_claims":
        return
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for child_key, child in node.items():
            yield from _strings(child, child_key)
    elif isinstance(node, list):
        for child in node:
            yield from _strings(child)


def _originals(fault_lin=1.0):
    return {"clean_reference": PT._report(f"baseline_p0c_s{SEED}", lin=0.2, checkpoint="model_299.pt"),
            "fault": PT._report(f"h06_s{SEED}", lin=fault_lin, checkpoint="model_299.pt")}


def _failed(originals):
    behavior = {key: C._behavior(value) for key, value in originals.items()}
    return RC.verdict(RC.compare(behavior["fault"], behavior["clean_reference"]), PT.BANDS)["failed"]


def _initial(arm):
    initial, _receipt = PT._arm_rows(Path("/tmp/w1"), arm)
    return initial


def _inputs(corrected=0.21, reference=0.2, budget=(0.6, 0.4, 0.3), budget_reference=0.2):
    reports = {"clean_resume_reference": PT._report("clean_resume_reference", lin=reference, checkpoint="model_598.pt"),
               "resume_corrected": PT._report("resume_corrected", lin=corrected, checkpoint="model_598.pt")}
    curve = {}
    for k, lin in zip(W.BUDGET_CHECKPOINTS, budget):
        curve[W.budget_name("resume_corrected", k)] = PT._report(W.budget_name("resume_corrected", k), lin=lin,
                                                                 checkpoint=f"model_{k}.pt")
        curve[W.budget_name("clean_resume_reference", k)] = PT._report(
            W.budget_name("clean_resume_reference", k), lin=budget_reference, checkpoint=f"model_{k}.pt")
    initials = {arm: _initial(arm) for arm in W.ARMS}
    return reports, initials, curve


def _summarize(reports, initials, curve=None, failed_arm=None, budget_failed=False, fault_lin=1.0):
    originals = _originals(fault_lin)
    return W.summarize(SEED, reports, initials, originals, PT.BANDS, _failed(originals), failed_arm, curve, budget_failed)


# --- 고정 계약 ---
def test_contract_registers_two_resume_arms_budget_curve_and_no_restart():
    assert W.ARMS == ("clean_resume_reference", "resume_corrected") and W.FIXED["arms"] == list(W.ARMS)
    assert all(spec["recovery_mode"] == "resume" for spec in W.ARM_SPECS.values())
    assert "restart_corrected" in W.FIXED["excluded_arms"] and "restart" not in json.dumps(W.ARM_SPECS)
    assert (W.FIXED["fault"], W.FIXED["fault_mechanism"], W.FIXED["correction"]) == ("F2", "actuator", "restore_motor_torque_speed")
    assert W.INPUT_RUNS == {"2026": {"NONE": "baseline_p0c_s2026", "F2": "h06_s2026"},
                            "2028": {"NONE": "baseline_p0c_s2028", "F2": "h06_s2028"}}
    assert W.FIXED["budget_curve"]["checkpoints"] == ["model_350.pt", "model_450.pt", "model_550.pt"]
    assert W.FIXED["budget_curve"]["used_for_labels"] is False and W.FIXED["budget_curve"]["training"] is False
    estimate = W.GPU_ESTIMATE
    per_seed = [2 * estimate["arm_total_s"][i] + estimate["budget_curve_eval_per_seed_s"][i] for i in (0, 1)]
    assert [round(v, 1) for v in per_seed] == estimate["per_seed_s"]
    assert [round(2 * v, 1) for v in per_seed] == estimate["two_seeds_s"]
    assert estimate["two_seeds_s"][1] < estimate["limit_two_seeds_s"]
    assert json.loads(json.dumps(W.FIXED)) == W.FIXED


def test_public_wording_has_no_forbidden_claim_or_private_path():
    texts = list(_strings(W.FIXED))
    assert texts and not [p for text in texts for p in PAGE.claim_violations(text) + PAGE.path_violations(text)]
    assert "직접" not in json.dumps(W.FIXED, ensure_ascii=False) + Path(W.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize(("mechanism", "ratio", "expected"), [
    ("actuator", 0.864, "resume"), ("actuator", 0.95, "resume"), ("reward", 0.95, "restart"), ("reward", 0.85, "resume")])
def test_current_rule_recommendation(mechanism, ratio, expected):
    assert W.rule_recommendation(mechanism, ratio) == expected


# --- 판정과 budget curve ---
def test_recovered_resume_keeps_budget_curve_as_secondary_record():
    reports, initials, curve = _inputs(corrected=0.21, budget=(0.6, 0.3, 0.22))
    result = _summarize(reports, initials, curve)
    assert (result["status"], result["labels"], result["prediction_match"]) == (
        "complete", {"resume_corrected": "recovered"}, {"resume_corrected": False})
    points = result["budget_curve"]["points"]
    assert [p["iteration"] for p in points] == [299, 350, 450, 550, 598]
    assert [p["source"] for p in points] == ["original_fixed_eval", *["budget_checkpoint"] * 3, "arm_fixed_eval"]
    assert points[-1]["err_xy_ratio"] == pytest.approx(0.21 / 0.2) and result["budget_curve"]["used_for_labels"] is False
    assert result["budget_curve"]["first_healthy_iteration"] == 550 and result["budget_curve"]["budget_limited_candidate"] is False
    assert result["recovery_mode_rule"]["outcome"] == "recommended_recovered"
    assert result["recovery_mode_rule"]["recommended"] == "resume"


def test_not_recovered_resume_records_budget_limited_candidate_and_secondary_predictions():
    reports, initials, curve = _inputs(corrected=0.3, budget=(0.7, 0.45, 0.33))
    result = _summarize(reports, initials, curve)
    assert result["labels"] == {"resume_corrected": "not_recovered"} and result["prediction_match"] == {"resume_corrected": True}
    assert result["secondary_prediction_results"] == {"final_error_below_half_of_original_fault": True,
                                                      "final_error_below_first_budget_point": True}
    assert result["budget_curve"]["budget_limited_candidate"] is True
    assert result["recovery_mode_rule"]["outcome"] == "recommended_not_recovered"
    assert result["wording"] == W.FIXED["result_wording"]["not_recovered"]
    reports, initials, curve = _inputs(corrected=0.34, budget=(0.25, 0.28, 0.33))
    plateau = _summarize(reports, initials, curve)
    assert plateau["labels"] == {"resume_corrected": "not_recovered"}
    assert plateau["budget_curve"]["budget_limited_candidate"] is False
    assert plateau["secondary_prediction_results"]["final_error_below_first_budget_point"] is False


def test_invalid_reference_failed_arm_and_missing_budget_curve():
    reports, initials, curve = _inputs(reference=1.0)
    invalid = _summarize(reports, initials, curve)
    assert (invalid["status"], invalid["label_reasons"]) == ("invalid", {"resume_corrected": "resume_reference_invalid"})
    reports, initials, _ = _inputs()
    failed = _summarize({"clean_resume_reference": reports["clean_resume_reference"]},
                        {"clean_resume_reference": initials["clean_resume_reference"]}, failed_arm="resume_corrected")
    assert (failed["status"], failed["label_reasons"]["resume_corrected"]) == ("invalid", "execution_failed")
    assert failed["budget_curve"]["status"] == "not_executed"
    partial = _summarize(reports, initials, None, budget_failed=True)
    assert (partial["status"], partial["budget_curve"]["status"]) == ("partial", "failed")
    assert partial["secondary_prediction_results"]["final_error_below_first_budget_point"] is None


def test_budget_report_errors_invalidate_only_the_budget_curve():
    reports, initials, curve = _inputs(corrected=0.3)
    curve.pop(W.budget_name("resume_corrected", 450))
    result = _summarize(reports, initials, curve)
    assert (result["status"], result["labels"]) == ("partial", {"resume_corrected": "not_recovered"})
    assert (result["budget_curve"]["status"], result["budget_curve"]["points"]) == ("invalid", [])
    reports, initials, curve = _inputs(corrected=0.3)
    curve[W.budget_name("clean_resume_reference", 350)]["checkpoint"]["file"] = "model_598.pt"
    assert _summarize(reports, initials, curve)["budget_curve"]["status"] == "invalid"
    reports, initials, curve = _inputs(corrected=0.3)
    curve["resume_corrected_model_500"] = copy.deepcopy(curve[W.budget_name("resume_corrected", 450)])
    assert _summarize(reports, initials, curve)["budget_curve"]["status"] == "invalid"


def test_budget_marks_require_the_saved_iteration_to_match_the_file_name(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for k in (*W.BUDGET_CHECKPOINTS, 598):
        (run / f"model_{k}.pt").write_bytes(str(k).encode())
    receipt = {"checkpoint": {"path": str(run / "model_598.pt")}}
    monkeypatch.setattr(W, "checkpoint_iteration", lambda path: int(Path(path).stem.split("_")[1]))
    marks = W.budget_marks(receipt)
    assert marks["450"] == {"file": "model_450.pt", "sha256": W.sha(run / "model_450.pt"), "iteration": 450}
    monkeypatch.setattr(W, "checkpoint_iteration", lambda path: 299)
    with pytest.raises(ValueError, match="iteration differs"):
        W.budget_marks(receipt)
    reports, initials, curve = _inputs()
    initials["clean_resume_reference"]["resume_checkpoint"] = initials["resume_corrected"]["resume_checkpoint"]
    with pytest.raises(ValueError, match="clean reference resumed the failed checkpoint"):
        _summarize(reports, initials, curve)


# --- 규칙 갱신 결정 ---
def _row(seed, label, protocol="p" * 64):
    return {"seed": seed, "labels": {"resume_corrected": label}, "protocol_sha256_lf": protocol,
            "budget_curve": {"budget_limited_candidate": label == "not_recovered"}}


@pytest.mark.parametrize(("labels", "decision"), [
    (("recovered", "recovered"), "keep"), (("recovered", "not_recovered"), "change"),
    (("not_recovered", "not_recovered"), "change"), (("recovered", "invalid"), "undetermined")])
def test_rule_update_follows_the_preregistered_criteria(labels, decision):
    result = W.aggregate([_row(2026, labels[0]), _row(2028, labels[1])])
    update = result["rule_update"]
    assert update["decision"] == decision and update["wording"] == W.FIXED["result_wording"]["rule_update"][decision]
    assert update["next_rule"] == {"keep": W.RULE_CURRENT, "change": W.RULE_NEXT_IF_CHANGED, "undetermined": None}[decision]


def test_rule_update_needs_both_seeds_from_one_protocol():
    assert W.aggregate([_row(2026, "recovered")])["rule_update"]["decision"] == "undetermined"
    with pytest.raises(ValueError, match="different protocols"):
        W.aggregate([_row(2026, "recovered"), _row(2028, "recovered", "q" * 64)])
    with pytest.raises(ValueError, match="distinct registered"):
        W.aggregate([_row(2026, "recovered"), _row(2026, "recovered")])


# --- 실행 커밋 고정 ---
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True).stdout.strip()


def test_frozen_commit_requires_ancestor_and_unchanged_bound_blobs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "bound.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "other.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "bound.py", "other.md")
    _git(repo, "commit", "-q", "-m", "experiment")
    experiment = _git(repo, "rev-parse", "HEAD")
    (repo / "other.md").write_text("y\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "unrelated")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(W, "ROOT", repo)
    assert W.frozen_commit(experiment, head, {"bound.py"}) == experiment
    assert W.frozen_commit(head, head, {"bound.py"}) == head
    with pytest.raises(ValueError, match="differs"):
        W.frozen_commit(experiment, head, {"other.md"})
    (repo / "bound.py").write_text("a = 2\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "bound change")
    with pytest.raises(ValueError, match="differs"):
        W.frozen_commit(experiment, _git(repo, "rev-parse", "HEAD"), {"bound.py"})
    with pytest.raises(ValueError, match="not an ancestor"):
        W.frozen_commit(head, experiment, {"bound.py"})
    with pytest.raises(ValueError, match="malformed"):
        W.frozen_commit("HEAD", head, {"bound.py"})


# --- 학습 자식 manifest와 중간 체크포인트 ---
def test_manifest_matches_the_shared_training_entry_schema(tmp_path, monkeypatch):
    run = tmp_path / "IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-10-09_03-38-49_h06_s2026"
    (run / "params").mkdir(parents=True)
    paths = {"checkpoint": run / "model_299.pt", "environment": run / "params/env.yaml", "agent": run / "params/agent.yaml"}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"resume_inputs": {"seeds": {"2026": {"runs": {
        "F2": {"hashes": {"checkpoint": "c" * 64}}, "NONE": {"hashes": {"checkpoint": "d" * 64}}}}}}}), encoding="utf-8")
    monkeypatch.setattr(W, "PROTOCOL", protocol)
    monkeypatch.setattr(W, "source_paths", lambda root, seed, fault: paths)
    request = {"seed": 2026, "source_commit_sha": COMMIT, "protocol_sha256_lf": "p" * 64,
               "isaaclab_root": str(tmp_path / "IsaacLab")}
    manifest = W.manifest_for(tmp_path / "motor_tag", request, "resume_corrected")
    assert set(manifest) == W.R.MANIFEST_FIELDS and W.R.validate_manifest_mode(manifest) == "resume"
    assert (manifest["fault"], manifest["correction_id"], manifest["final_iteration"]) == ("F2", W.CORRECTION, 598)
    assert manifest["resume_checkpoint_sha256"] == "c" * 64 and manifest["run_name"] == "motor_tag_resume_corrected"
    assert manifest["load_run_regex"] == "^2026\\-10\\-09_03\\-38\\-49_h06_s2026$" and manifest["init_noise_std"] is None
    clean = W.manifest_for(tmp_path / "motor_tag", request, "clean_resume_reference")
    assert (clean["fault"], clean["correction_id"], clean["resume_checkpoint_sha256"]) == ("NONE", None, "d" * 64)


def test_budget_checkpoints_come_from_the_verified_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for k in (*W.BUDGET_CHECKPOINTS, 598):
        (run / f"model_{k}.pt").write_bytes(str(k).encode())
    receipt = {"checkpoint": {"path": str(run / "model_598.pt")}}
    assert {k: p.name for k, p in W.budget_checkpoints(receipt).items()} == {350: "model_350.pt", 450: "model_450.pt",
                                                                            550: "model_550.pt"}
    (run / "model_450.pt").unlink()
    with pytest.raises(ValueError, match="intermediate checkpoint is missing"):
        W.budget_checkpoints(receipt)


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """GPU 없이 execute 흐름을 돌린다. 학습·평가 프로세스와 실행 검증만 대역으로 바꾼다."""
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    originals = _originals()
    original_paths = {key: _json(tmp_path / f"original_{key}.json", value) for key, value in originals.items()}
    reference = PT._report("clean_resume_reference", lin=0.2, checkpoint="model_598.pt")
    protocol = {"fixed_eval_sha256_lf": W.sha(W.FIXED_EVAL, True),
                "resume_inputs": {"bands": PT.BANDS, "runtime_versions": reference["versions"], "summary_sha256_lf": "2" * 64,
                                  "seeds": {str(SEED): {"runs": {
                                      "NONE": {"run": f"baseline_p0c_s{SEED}", "hashes": {"checkpoint": "3" * 64}},
                                      "F2": {"run": f"h06_s{SEED}", "hashes": {"checkpoint": "4" * 64},
                                             "original_failed": _failed(originals)}}}}}}
    receipts, calls, plan = {}, [], {"fail_train": None, "fail_budget": False}
    lins = {"clean_resume_reference": 0.2, "resume_corrected": 0.3}
    for arm in W.ARMS:
        _initial_row, receipt = PT._arm_rows(tmp_path, arm)
        run = Path(receipt["checkpoint"]["path"]).parent
        run.mkdir(parents=True, exist_ok=True)
        for k in (*W.BUDGET_CHECKPOINTS, 598):
            (run / f"model_{k}.pt").write_bytes(f"{arm}-{k}".encode())
        receipt["checkpoint"]["sha256"] = W.sha(run / "model_598.pt")
        receipts[arm] = receipt

    def run_owned(folder, argv, cwd, watchdog):
        calls.append((Path(folder).name, Path(argv[1]).name, watchdog))
        if "train-child" in argv:
            arm = Path(folder).parent.name
            initial, _ = PT._arm_rows(tmp_path, arm)
            _json(Path(folder) / "initial.json", initial)
            _json(Path(folder) / "receipt.json", receipts[arm])
            return {"complete": plan["fail_train"] != arm, "wall_s": 500.0, "wall_scope": W.Q.PARENT_WALL_SCOPE,
                    "owned_process_identities": {"11": 1}}
        jobs = json.loads((Path(folder) / "jobs.json").read_text(encoding="utf-8"))["jobs"]
        complete = not (plan["fail_budget"] and Path(folder).name == "budget")
        for job in jobs if complete else []:
            arm = job["name"].split("_model_")[0]
            lin = lins[arm] + (0.2 if "_model_" in job["name"] and arm == "resume_corrected" else 0.0)
            report = PT._report(job["name"], lin=lin, checkpoint=Path(job["checkpoint"]).name)
            report["checkpoint"]["sha256"] = W.sha(Path(job["checkpoint"]))
            _json(Path(job["output"]), report)
        return {"complete": complete, "wall_s": 55.0, "wall_scope": W.Q.PARENT_WALL_SCOPE,
                "owned_process_identities": {"12": 2}}

    def verify_training(folder, arm, execution):
        if not execution["complete"]:
            raise ValueError("training execution failed")
        return receipts[arm]

    monkeypatch.setattr(W, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(W, "PROTOCOL", _json(tmp_path / "protocol.json", {"note": "fixture protocol"}))
    monkeypatch.setattr(W, "verify_protocol", lambda root, experiment=None: (protocol, COMMIT))
    monkeypatch.setattr(W, "manifest_for", lambda folder, request, arm: {"arm": arm, "seed": request["seed"]})
    monkeypatch.setattr(W, "verify_training", verify_training)
    monkeypatch.setattr(W, "execution_valid", lambda row: row.get("complete") is True)
    monkeypatch.setattr(W, "source_paths", lambda root, seed, fault: {
        "fixed_report": original_paths["clean_reference" if fault == "NONE" else "fault"]})
    monkeypatch.setattr(W.M, "query_gpu", lambda profile=None: {"busy": False, "processes": []})
    monkeypatch.setattr(W.P, "run_owned", run_owned)
    monkeypatch.setattr(W, "checkpoint_iteration", _iteration_from_name)
    monkeypatch.setattr(W, "RESULTS", tmp_path / "results")
    return {"isaaclab": isaaclab, "calls": calls, "plan": plan, "receipts": receipts}


def test_execute_runs_two_arms_then_one_budget_evaluation(fake_runtime):
    result = W.execute("motor_w1_s2026", SEED, fake_runtime["isaaclab"])
    assert [(folder, script) for folder, script, _ in fake_runtime["calls"]] == [
        ("train", "run_training_correction_motor.py"), ("eval", "fixed_eval.py"),
        ("train", "run_training_correction_motor.py"), ("eval", "fixed_eval.py"), ("budget", "fixed_eval.py")]
    assert fake_runtime["calls"][-1][2] == W.FIXED["budget_eval_watchdog_s"]
    assert (result["status"], result["labels"]) == ("complete", {"resume_corrected": "not_recovered"})
    assert [p["iteration"] for p in result["budget_curve"]["points"]] == [299, 350, 450, 550, 598]
    folder = W.PRIVATE / "motor_w1_s2026"
    assert json.loads((folder / "budget/parent.json").read_text(encoding="utf-8"))["eval_parent_wall_s"] == 55.0
    assert result["budget_eval_parent_wall_s"] == 55.0 and result["source_commit_sha"] == COMMIT
    with pytest.raises(FileExistsError):
        W.execute("motor_w1_s2026", SEED, fake_runtime["isaaclab"])


def test_execute_stops_on_the_first_failed_arm_without_budget_curve(fake_runtime):
    fake_runtime["plan"]["fail_train"] = "resume_corrected"
    result = W.execute("motor_w1_fail", SEED, fake_runtime["isaaclab"])
    folder = W.PRIVATE / "motor_w1_fail"
    assert (result["status"], result["failed_arm"]) == ("invalid", "resume_corrected")
    assert json.loads((folder / "failure.json").read_text(encoding="utf-8"))["arm"] == "resume_corrected"
    assert not (folder / "budget").exists() and result["budget_curve"]["status"] == "not_executed"


def test_execute_keeps_the_label_when_only_the_budget_curve_fails(fake_runtime):
    fake_runtime["plan"]["fail_budget"] = True
    result = W.execute("motor_w1_budget", SEED, fake_runtime["isaaclab"])
    assert (result["status"], result["labels"]["resume_corrected"]) == ("partial", "not_recovered")
    assert (W.PRIVATE / "motor_w1_budget/budget_failure.json").is_file() and result["budget_curve"]["status"] == "failed"


def test_execute_records_intermediate_checkpoints_and_detects_later_changes(fake_runtime, monkeypatch):
    original_run = W.P.run_owned
    run = Path(fake_runtime["receipts"]["clean_resume_reference"]["checkpoint"]["path"]).parent

    def tamper_during_second_arm(folder, argv, cwd, watchdog):
        if "train-child" in argv and Path(folder).parent.name == "resume_corrected":
            (run / "model_450.pt").write_bytes(b"replaced while the second arm trained")
        return original_run(folder, argv, cwd, watchdog)
    monkeypatch.setattr(W.P, "run_owned", tamper_during_second_arm)
    result = W.execute("motor_w1_marks", SEED, fake_runtime["isaaclab"])
    folder = W.PRIVATE / "motor_w1_marks"
    marks = json.loads((folder / "clean_resume_reference/budget_checkpoints.json").read_text(encoding="utf-8"))
    assert set(marks) == {"350", "450", "550"} and marks["450"]["iteration"] == 450
    failure = json.loads((folder / "budget_failure.json").read_text(encoding="utf-8"))
    assert "changed after its arm was trained" in failure["detail"]
    assert (result["status"], result["labels"]["resume_corrected"]) == ("partial", "not_recovered")


def test_executed_seed_exports_and_replays(fake_runtime):
    W.execute("motor_w1_flow", SEED, fake_runtime["isaaclab"])
    exported = W.export("motor_w1_flow", fake_runtime["isaaclab"])
    replayed = W.replay(W.RESULTS / "motor_w1_flow")
    assert exported["labels"] == replayed["labels"] == {"resume_corrected": "not_recovered"}
    assert replayed["budget_curve"]["status"] == "complete" and replayed["parent_wall_s"]["budget_eval_parent_wall_s"] == 55.0


def test_intermediate_record_failure_keeps_arm_labels_and_fails_only_the_budget_curve(fake_runtime, monkeypatch):
    monkeypatch.setattr(W, "checkpoint_iteration",
                        lambda path: 0 if "clean_resume_reference" in str(path) else _iteration_from_name(path))
    result = W.execute("motor_w1_nomarks", SEED, fake_runtime["isaaclab"])
    folder = W.PRIVATE / "motor_w1_nomarks"
    assert result["completed_arms"] == list(W.ARMS) and result["labels"] == {"resume_corrected": "not_recovered"}
    assert (result["status"], result["budget_curve"]["status"]) == ("partial", "failed")
    assert (folder / "clean_resume_reference/budget_checkpoints_failure.json").is_file()
    assert json.loads((folder / "clean_resume_reference/parent.json").read_text(encoding="utf-8"))["budget_checkpoints_sha256"] is None
    W.export("motor_w1_nomarks", fake_runtime["isaaclab"])
    replayed = W.replay(W.RESULTS / "motor_w1_nomarks")
    evidence = json.loads((W.RESULTS / "motor_w1_nomarks/training_evidence.json").read_text(encoding="utf-8"))
    assert replayed["status"] == "partial" and evidence["arms"]["clean_resume_reference"]["budget_checkpoints"] is None


def test_train_child_requires_a_manifest():
    with pytest.raises(SystemExit):
        W.main(["train-child"])


# --- 공개 묶음 ---
def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _iteration_from_name(path: Path) -> int:
    return int(Path(path).stem.split("_")[1])


def _make_private_run(tmp_path, monkeypatch, tag, *, reference=0.2, with_budget=True):
    """execute가 남기는 비공개 산출물을 형식대로 만든다. 실행 검증 함수와 체크포인트 읽기만 대역으로 바꾼다."""
    private, results = tmp_path / "private", tmp_path / "results"
    folder = private / tag
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "PROTOCOL", _json(tmp_path / "protocol.json", {"note": "fixture protocol"}))
    monkeypatch.setattr(W, "checkpoint_iteration", _iteration_from_name)
    reports, initials, _ = _inputs(corrected=0.3, reference=reference)
    originals = _originals()
    receipts, costs, marks, curve = {}, {}, {}, {}
    for index, arm in enumerate(W.ARMS):
        initial, receipt = PT._arm_rows(tmp_path, arm)
        run = Path(receipt["checkpoint"]["path"]).parent
        run.mkdir(parents=True, exist_ok=True)
        for k in W.BUDGET_CHECKPOINTS:
            (run / f"model_{k}.pt").write_bytes(f"{arm}-{k}".encode())
        initials[arm], receipts[arm] = initial, receipt
        marks[arm] = W.budget_marks(receipt)
        for k, lin in zip(W.BUDGET_CHECKPOINTS, (0.7, 0.45, 0.33) if arm == "resume_corrected" else (0.2, 0.2, 0.2)):
            report = PT._report(W.budget_name(arm, k), lin=lin, checkpoint=f"model_{k}.pt")
            report["checkpoint"]["sha256"] = marks[arm][str(k)]["sha256"]
            curve[W.budget_name(arm, k)] = report
        training, evaluation = folder / arm / "train", folder / arm / "eval"
        _json(training / "initial.json", initial)
        _json(training / "receipt.json", receipt)
        _json(training / "execution.json", {"complete": True, "wall_s": 500.0 + index, "wall_scope": W.Q.PARENT_WALL_SCOPE})
        _json(evaluation / "execution.json", {"complete": True, "wall_s": 55.0, "wall_scope": W.Q.PARENT_WALL_SCOPE})
        _json(evaluation / "report.json", reports[arm])
        _json(folder / arm / "budget_checkpoints.json", marks[arm])
        costs[arm] = {"train_parent_wall_s": 500.0 + index, "eval_parent_wall_s": 55.0}
        _json(folder / arm / "parent.json", {
            "status": "pass", "training_receipt_sha256": W.sha(training / "receipt.json"),
            "fixed_report_sha256": W.sha(evaluation / "report.json"),
            "budget_checkpoints_sha256": W.sha(folder / arm / "budget_checkpoints.json"), **costs[arm]})
    if with_budget:
        budget = folder / "budget"
        checkpoints = {W.budget_name(arm, k): {"path": str(path), "sha256": marks[arm][str(k)]["sha256"]}
                       for arm in W.ARMS for k, path in W.budget_checkpoints(receipts[arm]).items()}
        for name, report in curve.items():
            _json(budget / f"{name}.json", report)
        _json(budget / "checkpoints.json", checkpoints)
        _json(budget / "execution.json", {"complete": True, "wall_s": 240.0, "wall_scope": W.Q.PARENT_WALL_SCOPE})
        _json(budget / "parent.json", {"status": "pass", "checkpoints_sha256": W.sha(budget / "checkpoints.json"),
                                       "reports_sha256": {name: W.sha(budget / f"{name}.json") for name in curve},
                                       "eval_parent_wall_s": 240.0})
    else:
        _json(folder / "budget_failure.json", {"stage": "budget_curve", "error_type": "RuntimeError", "detail": "x"})
    original_paths = {key: _json(tmp_path / f"original_{key}.json", value) for key, value in originals.items()}
    source_runs = {"NONE": {"run": f"baseline_p0c_s{SEED}", "hashes": {"checkpoint": "3" * 64}},
                   "F2": {"run": f"h06_s{SEED}", "hashes": {"checkpoint": "4" * 64}, "original_failed": _failed(originals)}}
    protocol = {"fixed_eval_sha256_lf": W.sha(W.FIXED_EVAL, True),
                "resume_inputs": {"bands": PT.BANDS, "runtime_versions": reports["clean_resume_reference"]["versions"],
                                  "summary_sha256_lf": "2" * 64, "seeds": {str(SEED): {"runs": source_runs}}}}
    _json(folder / "request.json", {"tag": tag, "seed": SEED, "source_commit_sha": COMMIT,
                                    "protocol_sha256_lf": W.sha(W.PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve())})
    result = W.summarize(SEED, reports, initials, originals, PT.BANDS, _failed(originals), None,
                         curve if with_budget else None, not with_budget, protocol["fixed_eval_sha256_lf"])
    result.update(tag=tag, costs=costs, budget_eval_parent_wall_s=240.0 if with_budget else None, model_calls=0,
                  source_commit_sha=COMMIT, protocol_sha256_lf=W.sha(W.PROTOCOL, True))
    _json(folder / "result.json", result)
    monkeypatch.setattr(W, "PRIVATE", private)
    monkeypatch.setattr(W, "RESULTS", results)
    monkeypatch.setattr(W, "verify_protocol", lambda root, experiment=None: (protocol, COMMIT))
    monkeypatch.setattr(W, "verify_training", lambda folder_, arm, execution: receipts[arm])
    monkeypatch.setattr(W, "execution_valid", lambda row: row.get("complete") is True)
    monkeypatch.setattr(W, "source_paths", lambda root, seed, fault: {
        "fixed_report": original_paths["clean_reference" if fault == "NONE" else "fault"]})
    return {"tag": tag, "isaaclab": isaaclab, "results": results, "result": result, "tmp": tmp_path,
            "folder": folder, "receipts": receipts}


@pytest.fixture
def private_run(tmp_path, monkeypatch):
    return _make_private_run(tmp_path, monkeypatch, "motor_w1_s2026")


def _rehash(folder: Path, *names: str) -> None:
    manifest_path = folder / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        manifest["files"][name] = W.sha(folder / name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_export_and_replay_round_trip_without_private_paths(private_run):
    exported = W.export(private_run["tag"], private_run["isaaclab"])
    assert (exported["status"], exported["labels"]) == ("complete", {"resume_corrected": "not_recovered"})
    folder = private_run["results"] / private_run["tag"]
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert names == W.public_files(list(W.ARMS), True) | {"artifact_manifest.json"}
    rendered = "\n".join(p.read_text(encoding="utf-8") for p in folder.rglob("*.json"))
    assert str(private_run["tmp"]) not in rendered and "bench/private" not in rendered
    replayed = W.replay(folder)
    assert replayed["labels"] == {"resume_corrected": "not_recovered"}
    assert replayed["budget_curve"]["budget_limited_candidate"] is True
    assert replayed["parent_wall_s"]["budget_eval_parent_wall_s"] == 240.0
    assert replayed["parent_wall_s"]["arms"]["resume_corrected"]["total_s"] == 556.0
    evidence = json.loads((folder / "training_evidence.json").read_text(encoding="utf-8"))
    assert set(evidence["arms"]["resume_corrected"]["budget_checkpoints"]) == {"350", "450", "550"}
    assert W.aggregate_public([folder])["rule_update"]["decision"] == "undetermined"
    with pytest.raises(FileExistsError):
        W.export(private_run["tag"], private_run["isaaclab"])


def test_replay_rejects_tampered_or_rehashed_public_files(private_run, monkeypatch):
    W.export(private_run["tag"], private_run["isaaclab"])
    folder = private_run["results"] / private_run["tag"]
    name = f"budget_reports/{W.budget_name('resume_corrected', 450)}.json"
    report = json.loads((folder / name).read_text(encoding="utf-8"))
    report["metrics"]["lin_vel_rmse_mps"] = 0.25
    (folder / name).write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        W.replay(folder)
    _rehash(folder, name)
    # 재집계가 맞지 않는 budget 보고서는 budget curve만 무효로 만들고, 그 결과가 공개 수치와 달라 거부된다.
    with pytest.raises(ValueError, match="does not replay"):
        W.replay(folder)
    # 재집계와 맞는 보고서로 바꾸고 결과 라벨도 고친 뒤 해시를 맞추면 공개 수치 재계산이 막는다.
    replacement = PT._report(W.budget_name("resume_corrected", 450), lin=0.45, checkpoint="model_450.pt")
    replacement["checkpoint"]["sha256"] = report["checkpoint"]["sha256"]
    (folder / name).write_text(json.dumps(replacement), encoding="utf-8")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    result["numeric"]["labels"] = {"resume_corrected": "recovered"}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    _rehash(folder, name, "result.json")
    with pytest.raises(ValueError, match="does not replay"):
        W.replay(folder)
    monkeypatch.setattr(W, "_publisher_sources", lambda: {"x": "0" * 64})
    with pytest.raises(ValueError, match="publisher source"):
        W.replay(folder)


@pytest.mark.parametrize(("target", "mutate", "message"), [
    ("training_evidence.json", lambda doc: doc["budget_curve"]["checkpoints"]["resume_corrected_model_450"].update(sha256="0" * 64),
     "not bound to its intermediate checkpoint"),
    ("training_evidence.json", lambda doc: doc["arms"]["resume_corrected"]["parent_wall_s"].update(total_s=1.0),
     "total differs"),
    ("training_evidence.json", lambda doc: doc["arms"]["resume_corrected"]["checkpoint"].update(sha256="0" * 64),
     "checkpoint evidence differs"),
    ("result.json", lambda doc: doc.update(source_commit_scope="other"), "bindings differ"),
])
def test_replay_checks_public_evidence_against_reports(private_run, target, mutate, message):
    W.export(private_run["tag"], private_run["isaaclab"])
    folder = private_run["results"] / private_run["tag"]
    document = json.loads((folder / target).read_text(encoding="utf-8"))
    mutate(document)
    (folder / target).write_text(json.dumps(document), encoding="utf-8")
    _rehash(folder, target)
    with pytest.raises(ValueError, match=message):
        W.replay(folder)


def test_export_rebinds_intermediate_checkpoints_to_the_verified_run(private_run):
    run = Path(private_run["receipts"]["clean_resume_reference"]["checkpoint"]["path"]).parent
    (run / "model_350.pt").write_bytes(b"changed after training")
    with pytest.raises(ValueError, match="changed after its arm was trained"):
        W.export(private_run["tag"], private_run["isaaclab"])


def test_invalid_seed_is_publishable_and_aggregates_as_undetermined(tmp_path, monkeypatch):
    run = _make_private_run(tmp_path, monkeypatch, "motor_w1_invalid", reference=1.0)
    assert run["result"]["status"] == "invalid"
    W.export(run["tag"], run["isaaclab"])
    folder = run["results"] / run["tag"]
    replayed = W.replay(folder)
    assert (replayed["status"], replayed["labels"]) == ("invalid", {"resume_corrected": "invalid"})
    assert W.aggregate_public([folder])["rule_update"]["decision"] == "undetermined"


def test_partial_seed_without_budget_curve_publishes_labels_only(tmp_path, monkeypatch):
    run = _make_private_run(tmp_path, monkeypatch, "motor_w1_partial", with_budget=False)
    W.export(run["tag"], run["isaaclab"])
    folder = run["results"] / run["tag"]
    assert not (folder / "budget_reports").exists()
    replayed = W.replay(folder)
    assert (replayed["status"], replayed["budget_curve"]["status"]) == ("partial", "failed")


def test_export_refuses_a_private_result_that_does_not_replay(private_run):
    result = copy.deepcopy(private_run["result"])
    result["labels"] = {"resume_corrected": "recovered"}
    (private_run["folder"] / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from replayed evidence"):
        W.export(private_run["tag"], private_run["isaaclab"])
