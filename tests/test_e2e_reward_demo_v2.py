"""E4 종단 시연 v2: v1과 달라진 부분(seed 2030·3030, 실험 커밋 고정 검증, 단일 실험 커밋 조건, restart 결합)만 시험한다.

v1에서 그대로 복제한 단계 로직은 tests/test_e2e_reward_demo.py가 v1 모듈로 시험한다. GPU·모델 없음.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import demo_restart_correction as DR1  # noqa: E402
import demo_restart_correction_v2 as DR  # noqa: E402
import e2e_reward_demo as V1  # noqa: E402
import e2e_reward_demo_v2 as E  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402

EXPERIMENT, HEAD = "a" * 40, "b" * 40


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """v2 시연 출력·고리·결과·프로토콜 경로를 임시 폴더로 돌린다."""
    import correct as CR
    import loop
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(DR, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "results")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(E, "PROTOCOL", protocol)
    monkeypatch.setattr(DR, "PROTOCOL", protocol)
    monkeypatch.setattr(loop, "LOOPS", tmp_path / "loops")
    monkeypatch.setattr(CR, "LOOPS", tmp_path / "loops")
    return E


def _begin(demo, tag, commit=EXPERIMENT):
    demo._write_step(tag, "begin", demo._guard(tag, "begin"), commit, {"seed_unused": {"hits": []}})


# --- 상수·결합 집합: v1은 그대로 ---
def test_v2_constants_and_binding_leave_v1_unchanged():
    assert (E.SEED, E.RESTART_SEED, DR.SEED, DR.RESTART_SEED, E.FIXED["restart_seed"]) == (2030, 3030, 2030, 3030, 3030)
    assert E.CASES == {"NONE": "baseline_p0c_s2030", "F1": "h02_s2030"} and DR.F1_RUN == "h02_s2030"
    assert E.PROTOCOL.name == DR.PROTOCOL.name == "e2e_reward_demo_v2.json"
    assert (V1.SEED, V1.PROTOCOL.name, DR1.SEED, DR1.PROTOCOL.name) == (2029, "e2e_reward_demo_v1.json",
                                                                         2029, "e2e_reward_demo_v1.json")
    assert E.CODE ^ V1.CODE == {"evals/e2e_reward_demo.py", "bench/demo_restart_correction.py",
                                "evals/e2e_reward_demo_v2.py", "bench/demo_restart_correction_v2.py"}
    assert all((ROOT / name).is_file() for name in E.CODE)
    assert set(E.FIXED["success_conditions"]) == set(E.SUCCESS_CONDITIONS) == set(V1.SUCCESS_CONDITIONS)
    assert set(E.OUTCOMES) == set(V1.OUTCOMES) and E.TERMINAL_OUTCOMES == V1.TERMINAL_OUTCOMES
    dynamic = {k: f"v_{k}" for k in E.DYNAMIC}
    E.check_protocol({**E.FIXED, **dynamic}, dynamic)
    with pytest.raises(ValueError):  # v1 프로토콜 내용으로는 v2를 실행할 수 없다
        E.check_protocol({**V1.FIXED, **dynamic}, dynamic)


def _functions(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize("v1,v2,changed", [
    ("evals/e2e_reward_demo.py", "evals/e2e_reward_demo_v2.py",
     {"verify_protocol", "train", "alarm", "evaluate", "reference", "packet", "loop_start", "loop_close",
      "correction", "correction_approval", "conditions", "summarize", "main"}),
    ("bench/demo_restart_correction.py", "bench/demo_restart_correction_v2.py", {"_demo", "validate_manifest", "_run"}),
])
def test_v2_copies_v1_except_registered_changes(v1, v2, changed):
    """복제한 함수가 v1과 갈라지면 실패한다. 의도한 변경은 changed에 등록한다(v1 로직 테스트가 v2에도 유효한 근거)."""
    old, new = _functions(v1), _functions(v2)
    assert set(old) <= set(new)
    differing = {name for name in old if ast.dump(old[name]) != ast.dump(new[name])}
    assert differing == changed


def test_every_step_after_begin_verifies_against_the_experiment_commit():
    calls = {}
    for name, node in _functions("evals/e2e_reward_demo_v2.py").items():
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "verify_protocol":
                calls.setdefault(name, []).append(len(call.args))
    assert calls.pop("begin") == [1]
    assert calls and all(counts == [2] for counts in calls.values())
    restart = [len(call.args) for call in ast.walk(ast.parse((ROOT / "bench/demo_restart_correction_v2.py")
                                                              .read_text(encoding="utf-8")))
               if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "verify_protocol"]
    assert restart == [2, 2, 2]


# --- 실험 커밋 고정 검증(임시 git 저장소) ---
def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, files, message):
    for name, data in files.items():
        (repo / name).write_bytes(data)
    _git(repo, "add", "--", *files)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_frozen_commit_accepts_unbound_commits_and_rejects_bound_changes(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    monkeypatch.setattr(E, "ROOT", repo)
    bound = {"bound.py"}
    experiment = _commit(repo, {"bound.py": b"x = 1\n", "notes.md": b"a\n"}, "experiment")
    assert E.frozen_commit(experiment, experiment, bound) == experiment
    unrelated = _commit(repo, {"notes.md": b"b\n"}, "unrelated")  # 실행 중 결합 밖 파일 커밋(v1 실패 상황)
    assert E.frozen_commit(experiment, unrelated, bound) == experiment
    crlf = _commit(repo, {"bound.py": b"x = 1\r\n"}, "line endings only")  # LF 정규화 SHA는 같다
    assert E.frozen_commit(experiment, crlf, bound) == experiment
    changed = _commit(repo, {"bound.py": b"x = 2\n"}, "bound change")
    with pytest.raises(ValueError, match="differs"):
        E.frozen_commit(experiment, changed, bound)
    with pytest.raises(ValueError, match="missing"):  # 실험 커밋에 없던 결합 파일
        E.frozen_commit(experiment, _commit(repo, {"later.py": b"y\n"}, "new file"), bound | {"later.py"})
    _git(repo, "checkout", "-q", "-b", "side", experiment)
    side = _commit(repo, {"notes.md": b"side\n"}, "side")
    with pytest.raises(ValueError, match="ancestor"):  # 실험 커밋이 HEAD의 조상이 아니다
        E.frozen_commit(side, changed, bound)
    for malformed in ("HEAD", experiment[:12], None):
        with pytest.raises(ValueError, match="malformed"):
            E.frozen_commit(malformed, changed, bound)


def test_verify_protocol_returns_head_at_begin_and_experiment_commit_after(monkeypatch):
    calls = []
    monkeypatch.setattr(E, "read", lambda path: {"protocol": True})
    monkeypatch.setattr(E, "check_protocol", lambda protocol, dynamic: None)
    monkeypatch.setattr(E, "dynamic_fields", lambda isaaclab: {})
    monkeypatch.setattr(E, "S", SimpleNamespace(_git_head=lambda p: E.FIXED["isaaclab_commit"],
                                                _git_tracked_clean=lambda p: True))
    monkeypatch.setattr(E, "P", SimpleNamespace(verify_committed=lambda repo, paths: HEAD))
    monkeypatch.setattr(E, "frozen_commit", lambda experiment, head, bound: calls.append((experiment, head, bound))
                        or experiment)
    assert E.verify_protocol(Path("isaaclab")) == ({"protocol": True}, HEAD) and not calls
    assert E.verify_protocol(Path("isaaclab"), EXPERIMENT)[1] == EXPERIMENT
    experiment, head, bound = calls[0]
    assert (experiment, head) == (EXPERIMENT, HEAD)
    assert {"evals/e2e_reward_demo_v2.py", "bench/demo_restart_correction_v2.py",
            "bench/protocols/e2e_reward_demo_v2.json"} <= bound


# --- 단계 기록은 실험 커밋 하나 ---
def _owned(code=0):
    def run(folder, argv, cwd, watchdog_s):
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False, "error_type": None,
                "exit_code": code, "cleanup_verified": True, "owned_process_identities": {}, "wall_s": 12.3,
                "argv": argv}
    return run


def test_steps_after_begin_verify_and_record_the_experiment_commit(demo, monkeypatch, tmp_path):
    seen = []

    def verify(isaaclab, experiment=None):
        seen.append(experiment)
        return {"gpu_admission_profile": "x"}, experiment or HEAD

    monkeypatch.setattr(demo, "verify_protocol", verify)
    monkeypatch.setattr(demo, "_admit_gpu", lambda protocol: {"busy": False})
    monkeypatch.setattr(demo, "training_artifacts", lambda arm, isaaclab: {"run": arm})
    monkeypatch.setattr(demo, "_rel", lambda p: Path(p).name)
    _begin(demo, "t1")
    out = demo.train("t1", "NONE", tmp_path / "IsaacLab", owned=_owned())
    argv = out["execution"]["argv"]
    assert seen == [EXPERIMENT] and out["source_commit_sha"] == EXPERIMENT
    assert argv[argv.index("-Seed") + 1] == "2030" and argv[argv.index("-CaseId") + 1] == "baseline_p0c"
    assert demo.experiment_commit("t1") == EXPERIMENT


def _steps(**overrides):
    call = {"response_id": "r", "execution_wall_s": 2.0, "input_tokens": 100, "output_tokens": 50}
    base = {"protocol_sha256_lf": "p", "source_commit_sha": EXPERIMENT}
    steps = {
        "begin": {**base, "seed_unused": {"hits": []}},
        **{name: {**base, "execution": {"parent_wall_s": 10.0}} for name in ("train_NONE", "train_F1", "reference")},
        "evaluate": {**base, "execution": {"parent_wall_s": 80.0}, "gate": {"outcome": "failure_present"}},
        "loop_start": {**base, "call": call, "decision": {"action": "continue"}, "started": {"model_probe_used": True}},
        "loop_close": {**base, "proceed_to_correction": True,
                       "loop": {"requests": [{"state": "done", "approver": "op", "artifact": {"probe": "x"},
                                              "outcome": "abnormal", "gpu_s": 18.0}]}},
        "correction": {**base, "call": call, "model_proposal": {"accepted": True},
                       "receipt": {"event": "receipt", "verdict": "recovered"},
                       "result": {"costs": {"train_parent_wall_s": 500.0, "eval_parent_wall_s": 50.0}}},
    }
    steps.update(overrides)
    return steps


def test_protocol_committed_requires_one_experiment_commit_other_conditions_match_v1():
    steps = _steps()
    assert all(E.conditions(steps, "recovered").values())
    moved = _steps(alarm={"protocol_sha256_lf": "p", "source_commit_sha": HEAD})  # v1 seed 2029처럼 커밋이 바뀐 기록
    assert not E.conditions(moved, "recovered")["protocol_committed"]
    assert V1.conditions(moved, "recovered")["protocol_committed"]
    v1_failure = _steps(correction={**steps["correction"], "receipt": {"event": "receipt", "verdict": "unknown"},
                                    "result": None})  # v1: 학습 뒤 검증 거부로 restart 결과(비용)가 없었다
    for case, outcome in ((steps, "recovered"), (moved, "recovered"), (v1_failure, "correction_unknown"),
                          ({k: v for k, v in steps.items() if k != "correction"}, "in_progress")):
        v2, v1 = E.conditions(case, outcome), V1.conditions(case, outcome)
        assert {k: v for k, v in v2.items() if k != "protocol_committed"} == {
            k: v for k, v in v1.items() if k != "protocol_committed"}
    assert not E.conditions(v1_failure, "correction_unknown")["costs_recorded"]
    assert E.conditions(v1_failure, "correction_unknown")["correction_approved_consumed_once"]


def test_summarize_uses_v2_schema(demo):
    _begin(demo, "t2")
    demo._terminal("t2", "baseline_invalid", {"gate": {"outcome": "baseline_invalid"}})
    out = demo.summarize("t2")
    assert out["schema"] == "e2e_reward_demo_result_v2" and out["seed"] == 2030
    assert out["source_commit_sha"] == [EXPERIMENT] and out["approver_identity_attested"] is False


def test_seed_unused_checks_2030_and_3030(demo, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    for folder in (demo.RUNS_DIR, "bench/private", "evals/loops"):
        (repo / folder).mkdir(parents=True)
    monkeypatch.setattr(demo, "ROOT", repo)
    isaaclab = tmp_path / "IsaacLab"
    run = isaaclab / "logs/rsl_rl/unitree_go2_flat/2026-10-14_00-00-00_other_run"
    (run / "params").mkdir(parents=True)
    (run / "params/agent.yaml").write_text("seed: 2029\n", encoding="utf-8")  # v1 seed는 v2를 막지 않는다
    (repo / demo.RUNS_DIR / "h02_s2029.telemetry.json").write_text("{}", encoding="utf-8")
    evidence = demo.seed_unused_evidence(isaaclab, "t8")
    assert not evidence["hits"] and evidence["checked_seeds"] == [2030, 3030]
    (run / "params/agent.yaml").write_text("seed: 3030\n", encoding="utf-8")
    (repo / "evals/loops/h02_s2030").mkdir()
    hits = demo.seed_unused_evidence(isaaclab, "t8")["hits"]
    assert any("seed 3030" in hit for hit in hits) and any(hit.endswith("h02_s2030") for hit in hits)


# --- restart 교정: 실험 커밋 결합 ---
def _restart_request(demo, tmp_path, monkeypatch, *, commit=EXPERIMENT):
    tag = "d"
    _begin(demo, tag)
    monkeypatch.setattr(DR, "source_run", lambda demo_tag: {"environment": {"path": "env.yaml"},
                                                            "agent": {"path": "agent.yaml"}})
    folder = DR.fix_folder(tag)
    (folder / "train").mkdir(parents=True)
    request = {"demo_tag": tag, "tag": DR.fix_tag(tag), "failure_seed": DR.SEED, "restart_seed": DR.RESTART_SEED,
               "case": DR.F1_RUN, "arm": DR.ARM, "correction_request_id": "r", "source_commit_sha": commit,
               "protocol_sha256_lf": DR.sha(DR.PROTOCOL, True), "isaaclab_root": str(tmp_path)}
    DR.write_new(folder / "request.json", request)
    DR.write_new(folder / "approval.json", {"request_sha256": DR.sha(folder / "request.json")})
    DR.write_new(folder / "consumed.json", {"approval_sha256": DR.sha(folder / "approval.json"), "arm": DR.ARM,
                                            "train_limit": 1, "eval_limit": 1})
    DR.write_new(folder / "train/manifest.json", DR.manifest_for(folder, request))
    return folder / "train/manifest.json"


def test_restart_manifest_is_bound_to_the_experiment_commit(demo, monkeypatch, tmp_path):
    seen = []

    def verify(isaaclab, experiment=None):  # 실행 중 HEAD가 전진해도 결합 파일이 같으면 실험 커밋을 돌려준다
        seen.append(experiment)
        return {}, experiment

    monkeypatch.setattr(demo, "verify_protocol", verify)
    manifest_path = _restart_request(demo, tmp_path, monkeypatch)
    manifest, _ = DR.validate_manifest(manifest_path)
    assert seen == [EXPERIMENT] and manifest["source_commit_sha"] == EXPERIMENT
    assert set(manifest) == R.MANIFEST_FIELDS and R.validate_manifest_mode(manifest) == "restart"
    assert manifest["seed"] == 3030 and manifest["run_name"] == "d_fix_restart_corrected"

    def bound_changed(isaaclab, experiment=None):
        raise ValueError("bound source differs between the experiment commit and HEAD: ['evals/probes.py']")

    monkeypatch.setattr(demo, "verify_protocol", bound_changed)
    with pytest.raises(ValueError, match="differs"):
        DR.validate_manifest(manifest_path)


def test_restart_request_recorded_at_head_instead_of_experiment_commit_is_rejected(demo, monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "verify_protocol", lambda isaaclab, experiment=None: ({}, experiment))
    manifest_path = _restart_request(demo, tmp_path, monkeypatch, commit=HEAD)
    with pytest.raises(ValueError, match="request source commitment changed"):
        DR.validate_manifest(manifest_path)


def test_restart_training_configuration_uses_v2_seeds(monkeypatch, tmp_path):
    src = tmp_path / "src"
    (src / "params").mkdir(parents=True)
    env = "seed: 2030\nscene:\n  num_envs: 4096\n"
    agent = "seed: 2030\nnum_steps_per_env: 24\nmax_iterations: 300\nrun_name: h02_s2030\n"
    (src / "params/env.yaml").write_text(env, encoding="utf-8")
    (src / "params/agent.yaml").write_text(agent, encoding="utf-8")
    artifacts = {k: {"path": str(src / "params" / f), "sha256": DR.sha(src / "params" / f)}
                 for k, f in (("environment", "env.yaml"), ("agent", "agent.yaml"))}
    monkeypatch.setattr(DR, "source_run", lambda demo_tag: artifacts)
    manifest = {"tag": DR.fix_tag("d"), "run_name": "d_fix_restart_corrected", "seed": 3030,
                "source_env_config_path": artifacts["environment"]["path"],
                "source_agent_config_path": artifacts["agent"]["path"]}
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    (run / "params/env.yaml").write_text(env.replace("seed: 2030", "seed: 3030"), encoding="utf-8")
    (run / "params/agent.yaml").write_text(
        agent.replace("h02_s2030", "d_fix_restart_corrected").replace("seed: 2030", "seed: 3030"), encoding="utf-8")
    DR.validate_training_configuration(run, manifest)
    (run / "params/env.yaml").write_text(env.replace("seed: 2030", "seed: 3029"), encoding="utf-8")
    with pytest.raises(ValueError):  # v1 재학습 seed는 v2 등록 seed가 아니다
        DR.validate_training_configuration(run, manifest)


def test_restart_executor_and_correction_use_v2_paths(demo, monkeypatch, tmp_path):
    def boom(request, demo_tag, isaaclab):
        raise ValueError("request source commitment changed")

    monkeypatch.setattr(DR, "_run", boom)
    out = DR.make_executor("d", tmp_path)({"request_id": "r"})
    assert out == {"exit_code": 1, "gpu_s": None, "verdict": "unknown", "result_sha": None, "tag": "d_fix"}
    assert json.loads((DR.fix_folder("d") / "failure.json").read_text(encoding="utf-8"))["error_type"] == "ValueError"
    import correct as CR
    from rl_triage.correction_ledger import CorrectionLedger
    monkeypatch.setattr(demo, "_guard", lambda tag, name: None)
    monkeypatch.setattr(demo, "experiment_commit", lambda tag: EXPERIMENT)
    monkeypatch.setattr(demo, "verify_protocol", lambda isaaclab, experiment=None: (
        {"gpu_admission_profile": "x"}, experiment))
    record = tmp_path / "correction.call1.json"
    record.write_text(json.dumps({"schema": "correction_model_call_v1"}), encoding="utf-8")
    led = CorrectionLedger(tmp_path / "loops" / "h02_s2030" / "correction_ledger.jsonl")
    evidence = {"model_proposal": {"record_sha256": DR.sha(record, True), "accepted": True}, "rule_mode": "restart"}
    rid = led.propose("s", "reward", "r", CR.R.REGISTRY["reward"]["correction_id"], "restart",
                      {"case": "case_01", "arm": "corrected"}, 1000, evidence)["request_id"]
    led.approve(rid, "claude_operator_user_authorized")
    with pytest.raises(ValueError, match="예산"):  # 승인 소비 전 거부(v2 원장 경로 h02_s2030)
        demo.correction("t9", tmp_path, rid, record)
    assert led.requests()[rid]["state"] == "approved"
