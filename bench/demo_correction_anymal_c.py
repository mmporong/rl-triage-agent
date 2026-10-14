"""ANYmal-C F1 종단 시연(evals/e2e_demo_anymal_c.py) 교정 실행기: 사전등록 회복 방식(restart·resume)대로 교정 학습·고정
평가·판정을 한 번 한다.

시연 v4 교정 실행기(bench/demo_correction_v4.py)와 arm·manifest 형식·학습 검증 항목·판정이 같다. v4 실행기는 공개 기록
replay의 소스 결합 때문에 바꾸지 않고, 순수 함수(arm 정의·판정·기준 선택·RNG 기록 검사)를 그대로 불러 쓴다. 다른 점:
- 시연 모듈 e2e_demo_anymal_c(부모·자식 모두), task·로그 폴더 anymal_c_flat, action scale은 시연 프로토콜 값(0.5).
- 고정 평가: evals/fixed_eval_task.py + bench/protocols/fixed_eval_v1_anymal_c.json, 평가 환경 readback이 단계 0과 같아야 한다.
- 고정 평가 보고서는 ANYmal-C 평가 프로토콜 이름·task로 검증한다(시연 실행기 validate_report).
v4 하위 진입점(train_with_recovery_mode.execute)에 이 모듈을 H로 넘긴다. make_executor는 evals/correct.py run의 실행 함수
계약을 따르고 실패는 unknown이다.

  (Isaac Python 하위 프로세스) python bench/demo_correction_anymal_c.py --manifest <manifest.json>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import demo_correction_v4 as DC4  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import run_training_correction as V3  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402

read = V3.read
write_new = V3.write_new
sha = V3.sha
load_config = V3.load_config
validate_readback = V3.validate_readback
execution_valid = V3.execution_valid

ARMS, RESUME_ITERATION, FINAL_ITERATION = DC4.ARMS, DC4.RESUME_ITERATION, DC4.FINAL_ITERATION
RESUME_AGENT_CHANGES, FIX_SUFFIX = DC4.RESUME_AGENT_CHANGES, DC4.FIX_SUFFIX
RESULT_SCHEMA = "e2e_correction_result_anymal_c_v1"
arm_spec, judge, reference_for = DC4.arm_spec, DC4.judge, DC4.reference_for


def _demo():
    import e2e_demo_anymal_c as E  # 부모·자식 모두 호출 시점에만 불러온다(순환 import 방지)
    return E


def fix_tag(demo_tag: str) -> str:
    _demo().folder_for(demo_tag)  # 시연 tag 형식 검사
    return demo_tag + FIX_SUFFIX


def fix_folder(demo_tag: str) -> Path:
    return _demo().folder_for(demo_tag) / "fix"


def _demo_tag_of(tag: str) -> str:
    if not tag.endswith(FIX_SUFFIX):
        raise ValueError("fix tag must end with _fix")
    demo_tag = tag[: -len(FIX_SUFFIX)]
    fix_tag(demo_tag)
    return demo_tag


def manifest_for(folder: Path, request: dict, arm: str) -> dict:
    E = _demo()
    ctx = E.ctx_of(request["variant"])
    spec = arm_spec(ctx, arm)
    if arm not in ARMS[request["recovery_mode"]]:
        raise ValueError("arm is outside the approved recovery mode")
    artifacts = E.step_record(request["demo_tag"], f"train_{spec['source']}")["artifacts"]
    resumed = spec["mode"] == "resume"
    checkpoint = Path(artifacts["checkpoint"]["path"])
    return {
        "schema": "training_correction_manifest_v2", "tag": request["tag"], "seed": spec["seed"], "arm": arm,
        "recovery_mode": spec["mode"], "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(ctx.protocol.resolve()), "fault": spec["fault"], "correction_id": spec["correction"],
        "num_envs": E.G.NUM_ENVS, "max_iterations": E.G.MAX_ITERATIONS, "task": E.G.TASK, "run_name": f"{request['tag']}_{arm}",
        "output_dir": str((folder / arm / "train").resolve()), "final_iteration": FINAL_ITERATION[spec["mode"]],
        "resume_checkpoint_path": str(checkpoint) if resumed else None,
        "resume_checkpoint_sha256": artifacts["checkpoint"]["sha256"] if resumed else None,
        "resume_iteration": RESUME_ITERATION if resumed else None,
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$" if resumed else None,
        "checkpoint_regex": "^model_299\\.pt$" if resumed else None,
        "source_env_config_path": artifacts["environment"]["path"],
        "source_agent_config_path": artifacts["agent"]["path"],
        "config_source_fault": "NONE" if spec["source"] == "NONE" else ctx.fault, "init_noise_std": None,
    }


# --- train_with_recovery_mode.execute가 쓰는 H 인터페이스(v4와 같다) ---
def validate_manifest(path: Path) -> tuple[dict, dict]:
    E = _demo()
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS:
        raise ValueError("manifest fields differ from the fixed schema")
    demo_tag = _demo_tag_of(manifest["tag"])
    folder = fix_folder(demo_tag)
    if Path(path).is_symlink() or Path(path).resolve() != (folder / manifest["arm"] / "train/manifest.json").resolve():
        raise ValueError("manifest is outside its registered arm")
    request = read(folder / "request.json")
    experiment = E.experiment_commit(demo_tag)
    protocol, commit = E.verify_protocol(request["variant"], Path(request["isaaclab_root"]), experiment)
    if (request["source_commit_sha"] != experiment or commit != experiment
            or request["protocol_sha256_lf"] != sha(E.ctx_of(request["variant"]).protocol, True)):
        raise ValueError("request source commitment changed")
    if manifest != manifest_for(folder, request, manifest["arm"]):
        raise ValueError("manifest differs from the approved request")
    if read(folder / "approval.json").get("request_sha256") != sha(folder / "request.json"):
        raise ValueError("approval record is not bound to the request")
    if read(folder / "consumed.json") != {"approval_sha256": sha(folder / "approval.json"),
                                          "arms": list(ARMS[request["recovery_mode"]]), "train_limit": 1, "eval_limit": 1}:
        raise ValueError("arm consumption record differs")
    return manifest, protocol


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    """학습 시작 직전 설정은 출처 학습 설정과 회복 방식별 허용 인자만 다를 수 있다(v4와 같다)."""
    E = _demo()
    demo_tag = _demo_tag_of(manifest["tag"])
    ctx = E.ctx_for(demo_tag)
    spec = arm_spec(ctx, manifest["arm"])
    artifacts = E.step_record(demo_tag, f"train_{spec['source']}")["artifacts"]
    source_env, source_agent = Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"])
    for path, key in ((source_env, "environment"), (source_agent, "agent")):
        if str(path) != artifacts[key]["path"] or sha(path) != artifacts[key]["sha256"]:
            raise ValueError("source configuration changed before learning")
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env = load_config(Path(log_dir) / "params/env.yaml")
    current_agent = load_config(Path(log_dir) / "params/agent.yaml")
    if spec["mode"] == "restart":
        if {k: v for k, v in original_env.items() if k != "seed"} != {k: v for k, v in current_env.items() if k != "seed"}:
            raise ValueError("environment configuration changed outside the registered code correction and seed")
        allowed, expected = {"run_name", "seed"}, {"run_name": manifest["run_name"], "seed": spec["seed"]}
        if current_env.get("seed") != spec["seed"]:
            raise ValueError("restart training seed differs from the registered restart seed")
    else:
        if current_env != original_env:
            raise ValueError("environment configuration changed outside the registered code correction")
        allowed = RESUME_AGENT_CHANGES
        expected = {"run_name": manifest["run_name"], "resume": True, "load_run": manifest["load_run_regex"],
                    "load_checkpoint": manifest["checkpoint_regex"], "max_iterations": manifest["max_iterations"]}
    if ({k: v for k, v in original_agent.items() if k not in allowed}
            != {k: v for k, v in current_agent.items() if k not in allowed}):
        raise ValueError("agent configuration changed outside the recovery mode arguments")
    if any(type(current_agent.get(k)) is not type(v) or current_agent[k] != v for k, v in expected.items()):
        raise ValueError("agent configuration differs from the recovery mode manifest")
    if (original_agent.get("seed") != ctx.seed or original_agent.get("num_steps_per_env") != 24
            or original_agent.get("max_iterations") != 300 or original_env.get("seed") != ctx.seed
            or original_env.get("scene", {}).get("num_envs") != 4096):
        raise ValueError("source training seed, environment count, or step count differs")


# --- 부모: 학습 검증·고정 평가·판정 ---
def verify_training(folder: Path, arm: str, execution: dict, protocol: dict, versions: dict) -> dict:
    """v4 verify_training과 같은 검사. action scale은 시연 프로토콜 값, 로그 폴더는 이 task의 experiment 폴더다."""
    E = _demo()
    training = folder / arm / "train"
    if not execution_valid(execution):
        raise ValueError(f"training execution failed (exit={execution.get('exit_code')}); inspect its stderr.log")
    manifest, _ = validate_manifest(training / "manifest.json")
    mode = R.validate_manifest_mode(manifest)
    resumed = mode == "resume"
    receipt, initial = read(training / "receipt.json"), read(training / "initial.json")
    if (receipt.get("completed") is not True or receipt.get("schema") != "training_correction_receipt_v2"
            or initial.get("schema") != "training_correction_initial_v2" or initial.get("recovery_mode") != mode):
        raise ValueError("training receipt or initial state schema differs")
    for key in ("tag", "seed", "arm", "recovery_mode", "fault", "correction_id", "source_commit_sha",
                "protocol_sha256_lf", "task", "run_name", "final_iteration"):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f"training receipt binding differs: {key}")
    for file, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"),
                        ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / file):
            raise ValueError("training raw artifact changed")
    identity = receipt["process_identity"]
    if execution["owned_process_identities"].get(str(identity["process_id"])) != identity["process_creation_time_filetime"]:
        raise ValueError("training receipt came from an unowned process")
    steps = 300 * 24
    if (receipt.get("actual_control_steps") != steps or receipt.get("environment_make_calls") != 1
            or receipt.get("aggregate_env_steps") != steps * 4096
            or receipt.get("budget") != {"seed": manifest["seed"], "num_envs": 4096, "max_iterations": 300,
                                         "num_steps_per_env": 24, "expected_control_steps": steps}):
        raise ValueError("training budget or environment count differs")
    final = FINAL_ITERATION[mode]
    source_state = receipt.get("source_checkpoint_state")
    if resumed:
        expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": manifest["resume_checkpoint_sha256"]}
        if (receipt.get("resume_checkpoint") != expected_resume
                or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}
                or receipt.get("checkpoint_load_calls") != 1):
            raise ValueError("training did not resume the bound policy and optimizer")
        if any(row.get(key) != RESUME_ITERATION for row in (receipt, initial)
               for key in ("resume_iteration", "start_iteration")):
            raise ValueError("resume iteration differs")
        if (not isinstance(source_state, dict) or source_state != initial.get("source_checkpoint_state")
                or source_state.get("iteration") != RESUME_ITERATION
                or initial.get("std_reset") is not None or receipt.get("std_reset_applied") is not False
                or source_state.get("policy_state_sha256") != initial["policy_state"]["sha256"]
                or source_state.get("optimizer_state_sha256") != initial["optimizer_state"]["sha256"]):
            raise ValueError("initial policy/optimizer differs from the source checkpoint")
        proofs = ((source_state, RESUME_ITERATION), (receipt.get("final_checkpoint_state") or {}, final))
    else:
        if (receipt.get("resume_checkpoint") is not None or initial.get("resume_checkpoint") is not None
                or receipt.get("resume_iteration") is not None or initial.get("resume_iteration") is not None
                or source_state is not None or initial.get("source_checkpoint_state") is not None
                or receipt.get("checkpoint_load_calls") != 0 or initial.get("std_reset") is not None
                or receipt.get("std_reset_applied") is not False
                or receipt.get("start_iteration") != 0 or initial.get("start_iteration") != 0):
            raise ValueError("restart arm loaded or reset a checkpoint")
        proofs = ((receipt.get("final_checkpoint_state") or {}, final),)
    for state, iteration in proofs:
        if state.get("iteration") != iteration or any(not DC4._sha_text(state.get(k))
                                                      for k in ("policy_state_sha256", "optimizer_state_sha256")):
            raise ValueError("checkpoint state proof is incomplete")
    resume_scope = {"policy_optimizer_restored": resumed, "full_environment_state_restored": False,
                    "rng_state_restored_from_checkpoint": False, "fresh_initialization": not resumed,
                    "action_std_reset": False}
    if (receipt.get("resume_scope") != resume_scope or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False
            or receipt.get("rng_state_before_learn") != initial.get("rng_state_before_learn")
            or not DC4._rng_ok(initial.get("rng_state_before_learn"))):
        raise ValueError("resume scope, episode initialization, or RNG observation differs")
    scale = protocol["action_scale"]
    observed = {"environment_seed": manifest["seed"], "environment_num_envs": 4096, "action_scale": scale,
                "agent_seed": manifest["seed"], "agent_max_iterations": 300, "runner_num_steps_per_env": 24}
    if (receipt.get("observed") != observed or initial.get("observed") != observed
            or receipt.get("actual_action_scale") != scale):
        raise ValueError("actual training configuration differs")
    runtime = protocol["correction_source_contract"]["runtime"]
    actual = receipt["runtime_versions"]
    if (actual != initial.get("runtime_versions") or actual.get("python") != ".".join(map(str, runtime["version"]))
            or actual.get("python_cache_tag") != runtime["cache_tag"]
            or actual.get("rsl_rl_distribution") != protocol["rsl_rl_version"]
            or actual.get("torch_distribution") != versions["torch"]
            or actual.get("isaaclab_distribution") != versions["isaaclab"]
            or actual.get("isaaclab_source_commit") != protocol["isaaclab_commit"]):
        raise ValueError("actual training runtime differs")
    correction = read(training / "correction.json")
    validate_readback(correction, manifest["fault"], manifest["correction_id"])
    for key, target in protocol["correction_source_contract"]["targets"].items():
        if correction["before_fault"][key] != target["clean_fingerprint"]:
            raise ValueError("clean live target differs from committed SDK source")
    log_root = E.G.log_root(Path(manifest["isaaclab_root"])).resolve()
    checkpoint = Path(receipt["checkpoint"]["path"])
    if (checkpoint.resolve().parent.parent != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"])
            or checkpoint.name != f"model_{final}.pt" or Path(initial["log_dir"]).resolve() != checkpoint.parent.resolve()):
        raise ValueError("training checkpoint is outside its exact run")
    if set(receipt["configs"]) != {"environment", "agent"}:
        raise ValueError("training configuration artifacts are incomplete")
    for key, filename in (("environment", "env.yaml"), ("agent", "agent.yaml")):
        if Path(receipt["configs"][key]["path"]).resolve() != (checkpoint.parent / "params" / filename).resolve():
            raise ValueError("training config path is outside its exact run")
    for artifact in [receipt["checkpoint"], *receipt["configs"].values()]:
        if sha(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("checkpoint or configuration hash differs")
    validate_training_configuration(checkpoint.parent, manifest)
    return receipt


def _run(request: dict, demo_tag: str, isaaclab: Path) -> dict:
    E = _demo()
    ctx = E.ctx_for(demo_tag)
    tag = fix_tag(demo_tag)
    mode = request.get("recovery_mode")
    if (request.get("case") != ctx.cases["FAULT"] or request.get("correction_id") != ctx.correction_id
            or mode not in ARMS or request.get("tag") != tag):
        raise ValueError("correction request is outside the registered demo arms")
    arms = ARMS[mode]
    protocol, commit = E.verify_protocol(ctx.variant, isaaclab, E.experiment_commit(demo_tag))
    evaluation_step = E.step_record(demo_tag, "evaluate")
    none_report = read(ROOT / evaluation_step["reports"]["NONE"]["path"])
    folder = fix_folder(demo_tag)
    if folder.exists():
        raise FileExistsError("fix arms have already been consumed")
    for path in (ROOT, isaaclab, folder):
        if any(char in str(path) for char in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    log_root = E.G.log_root(isaaclab)
    if log_root.is_dir() and any(path.name.endswith(f"_{tag}_{arm}") for path in log_root.iterdir() for arm in arms):
        raise FileExistsError("an SDK run already exists for a correction arm")
    if M.query_gpu(protocol["gpu_admission_profile"])["busy"]:
        raise RuntimeError("GPU busy before correction training")
    approval_event = E.E4.correction_approval(ctx, request["request_id"])
    record = {"demo_tag": demo_tag, "tag": tag, "variant": ctx.variant, "failure_seed": ctx.seed,
              "restart_seed": ctx.restart_seed, "case": ctx.cases["FAULT"], "recovery_mode": mode, "arms": list(arms),
              "correction_request_id": request["request_id"], "source_commit_sha": commit,
              "protocol_sha256_lf": sha(ctx.protocol, True), "isaaclab_root": str(isaaclab.resolve())}
    write_new(folder / "request.json", record)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"),
                                         "correction_ledger_approval": approval_event, "identity_attested": False})
    write_new(folder / "consumed.json", {"approval_sha256": sha(folder / "approval.json"), "arms": list(arms),
                                         "train_limit": 1, "eval_limit": 1})
    receipts, per_arm, owned = {}, {}, set()
    for arm in arms:
        training = folder / arm / "train"
        write_new(training / "manifest.json", manifest_for(folder, record, arm))
        argv = [E.G.python_bat(isaaclab), str(ROOT / "bench/demo_correction_anymal_c.py"),
                "--manifest", str(training / "manifest.json")]
        execution = P.run_owned(training, argv, isaaclab, protocol["watchdogs_s"]["correction_train"])
        write_new(training / "execution.json", execution)
        per_arm[arm] = {"train_parent_wall_s": execution["wall_s"]}
        owned |= set(execution["owned_process_identities"])
        receipts[arm] = verify_training(folder, arm, execution, protocol, none_report["versions"])
    evaluation = folder / "eval"
    evaluation.mkdir()
    write_new(evaluation / "jobs.json", {"action_scale": protocol["action_scale"], "jobs": [
        {"name": arm, "checkpoint": receipts[arm]["checkpoint"]["path"], "output": str(evaluation / f"{arm}.json")}
        for arm in arms]})
    gpu = M.query_gpu(protocol["gpu_admission_profile"])
    write_new(evaluation / "gpu_before.json", gpu)
    if gpu["busy"]:
        raise RuntimeError("GPU busy before fixed evaluation")
    argv = [E.G.python_bat(isaaclab), str(ROOT / "evals/fixed_eval_task.py"), "--protocol", str(E.EVAL_PROTOCOL),
            "--jobs", str(evaluation / "jobs.json"), "--headless"]
    eval_execution = P.run_owned(evaluation, argv, isaaclab, protocol["watchdogs_s"]["correction_eval"])
    write_new(evaluation / "execution.json", eval_execution)
    owned |= set(eval_execution["owned_process_identities"])
    if not execution_valid(eval_execution):
        raise ValueError("fixed evaluation did not complete and clean up")
    log = evaluation / "stdout.log"
    eval_env = E.FE.parse(log.read_text(encoding="utf-8", errors="replace") if log.is_file() else "")
    if not E.G.eval_env_ok(eval_env):
        raise ValueError("evaluation environment readback differs from the registered events")
    reports = {}
    for arm in arms:
        reports[arm] = E.validate_report(read(evaluation / f"{arm}.json"), {
            "name": arm, "checkpoint_file": f"model_{FINAL_ITERATION[mode]}.pt",
            "checkpoint_sha256": receipts[arm]["checkpoint"]["sha256"], "action_scale": protocol["action_scale"],
            "protocol_sha256_lf": protocol["eval_protocol_sha256_lf"]})
        if reports[arm]["versions"] != none_report["versions"] or reports[arm]["versions"]["rsl-rl-lib"] != protocol["rsl_rl_version"]:
            raise ValueError("fixed evaluator runtime differs from the demo NONE evaluation")
    reference = reference_for(mode, reports, none_report, protocol["bands"])
    treatment = arms[-1]
    judged = (judge(reports[treatment], reference["report"], protocol["bands"]) if reference["valid"] else
              {"verdict": None, "label": "unknown", "standstill_ratio": None, "reference_invalid": True})
    # 사후 검증: 실행 중 소스·체크포인트가 바뀌지 않았고 소유 프로세스가 GPU에 남지 않았다.
    current, current_commit = E.verify_protocol(ctx.variant, isaaclab, commit)
    if current != protocol or current_commit != commit:
        raise ValueError("source changed during the correction arms")
    if any(sha(Path(receipts[arm]["checkpoint"]["path"])) != receipts[arm]["checkpoint"]["sha256"] for arm in arms):
        raise ValueError("checkpoint changed during evaluation")
    after = M.query_gpu(protocol["gpu_admission_profile"])
    write_new(folder / "gpu_after.json", after)
    if any(str(row["pid"]) in owned for row in after["processes"]):
        raise ValueError("an owned GPU process remains")
    result = {"schema": RESULT_SCHEMA, "tag": tag, "variant": ctx.variant, "case": ctx.cases["FAULT"],
              "recovery_mode": mode, "arms": list(arms), "treatment": treatment,
              "training_seed": arm_spec(ctx, treatment)["seed"], "reference_seed": ctx.seed,
              "correction_request_id": request["request_id"], **judged,
              "reference": {k: v for k, v in reference.items() if k != "report"},
              "eval_env": eval_env,
              "costs": {"train_parent_wall_s": sum(v["train_parent_wall_s"] for v in per_arm.values()),
                        "eval_parent_wall_s": eval_execution["wall_s"], "per_arm": per_arm},
              "training_receipt_sha256": {arm: sha(folder / arm / "train/receipt.json") for arm in arms},
              "fixed_report_sha256": {arm: sha(evaluation / f"{arm}.json") for arm in arms},
              "reference_report": evaluation_step["reports"]["NONE"]["path"], "source_commit_sha": commit}
    write_new(folder / "result.json", result)
    return result


def make_executor(demo_tag: str, isaaclab: Path):
    """evals/correct.py run에 넘길 실행 함수. 실패는 기록하고 unknown(exit 1)으로 돌려줘 고아 요청을 남기지 않는다."""
    def _execute(request: dict) -> dict:
        tag = fix_tag(demo_tag)
        try:
            result = _run(request, demo_tag, isaaclab)
        except Exception as error:  # noqa: BLE001 - 원장 receipt를 unknown으로 닫기 위해 모든 실패를 기록한다
            folder = fix_folder(demo_tag)
            folder.mkdir(parents=True, exist_ok=True)
            failure = folder / "failure.json"
            if not failure.exists():
                write_new(failure, {"error_type": type(error).__name__, "detail": str(error)[:500]})
            return {"exit_code": 1, "gpu_s": None, "verdict": "unknown", "result_sha": None, "tag": tag}
        gpu_s = sum(v for v in result["costs"].values() if isinstance(v, (int, float)))
        return {"exit_code": 0, "gpu_s": round(gpu_s, 1), "verdict": result["label"],
                "result_sha": sha(fix_folder(demo_tag) / "result.json"), "tag": tag}
    return _execute


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    R.execute(args.manifest, sys.modules[__name__])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
