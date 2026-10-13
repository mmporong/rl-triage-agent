"""E2E 보상 결함 시연 v2 전용 restart 교정 단일 arm: F1 주입 → 등록 교정 복원 → 처음부터 300회 → 고정 평가 → 판정.

재학습 seed는 실패 seed(2030)와 다른 3030이다. 같은 seed 처음부터 학습은 정상 학습을 바이트 단위로 재현하므로
(v4 seed 2026·2028 관찰) 회복이 자명해진다. 판정은 같은 seed(2030) NONE 고정 평가 대비 p0c 밴드로 하며
체크포인트 동일성을 기대하지 않는다.

v4 하위 진입점(train_with_recovery_mode.execute)에 이 모듈을 H로 넘겨 재사용한다. v3·v4 실행 파일과 v1
(bench/demo_restart_correction.py)은 공개 결과의 소스 결합을 지키려고 바꾸지 않는다. v1과 달리 요청·manifest·
실행 전후 검사를 시연 begin의 실험 커밋 기준(e2e_reward_demo_v2.SOURCE_BINDING)으로 한다. 부모 쪽 함수
make_executor는 evals/correct.py run의 실행 함수 계약(execute(request) -> {exit_code, gpu_s, verdict, result_sha,
tag})을 따르고, 실패는 예외 대신 unknown으로 돌려준다.

  (Isaac Python 하위 프로세스) python bench/demo_restart_correction_v2.py --manifest <manifest.json>
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

import run_training_correction as V3  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

read = V3.read
write_new = V3.write_new
sha = V3.sha
load_config = V3.load_config
validate_readback = V3.validate_readback
execution_valid = V3.execution_valid

PROTOCOL = ROOT / "bench/protocols/e2e_reward_demo_v2.json"
PRIVATE = ROOT / "bench/private/e2e_reward_demo"
ARM = "restart_corrected"
FAULT = "F1"
CORRECTION = "restore_body_frame_reward"
SEED = 2030  # 실패 실행(NONE 기준·F1)의 학습 seed
RESTART_SEED = SEED + 1000  # 교정 재학습 seed. 같은 seed 재학습은 정상 학습을 바이트 단위로 재현하므로 다른 값을 쓴다
F1_RUN = f"h02_s{SEED}"
NUM_ENVS, MAX_ITERATIONS, NUM_STEPS_PER_ENV, ACTION_SCALE = 4096, 300, 24, 0.25
FINAL_ITERATION = MAX_ITERATIONS - 1
TRAIN_WATCHDOG_S, EVAL_WATCHDOG_S = 1800, 360
FIX_SUFFIX = "_fix"
TAG_RE = r"[a-z0-9][a-z0-9_-]{0,35}"


def _demo():
    import e2e_reward_demo_v2 as E  # 부모·자식 모두 호출 시점에만 불러온다(순환 import 방지)
    return E


def fix_tag(demo_tag: str) -> str:
    if not isinstance(demo_tag, str) or not re.fullmatch(TAG_RE, demo_tag):
        raise ValueError("invalid demo tag")
    return demo_tag + FIX_SUFFIX


def fix_folder(demo_tag: str) -> Path:
    folder = PRIVATE / fix_tag(demo_tag)[: -len(FIX_SUFFIX)] / "fix"
    if folder.is_symlink() or folder.resolve().parent.parent != PRIVATE.resolve():
        raise ValueError("fix output escapes its root")
    return folder


def _demo_tag_of(tag: str) -> str:
    if not tag.endswith(FIX_SUFFIX):
        raise ValueError("fix tag must end with _fix")
    demo_tag = tag[: -len(FIX_SUFFIX)]
    fix_tag(demo_tag)
    return demo_tag


def source_run(demo_tag: str) -> dict:
    """시연 F1 학습 단계가 결합한 F1 실행의 설정·체크포인트(경로·SHA)."""
    return _demo().step_record(demo_tag, "train_F1")["artifacts"]


def manifest_for(folder: Path, request: dict) -> dict:
    artifacts = source_run(request["demo_tag"])
    return {
        "schema": "training_correction_manifest_v2", "tag": request["tag"], "seed": RESTART_SEED, "arm": ARM,
        "recovery_mode": "restart", "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": FAULT, "correction_id": CORRECTION,
        "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS, "task": C.TASK,
        "run_name": f"{request['tag']}_{ARM}", "output_dir": str((folder / "train").resolve()),
        "final_iteration": FINAL_ITERATION, **{key: None for key in R.RESUME_FIELDS},
        "source_env_config_path": artifacts["environment"]["path"],
        "source_agent_config_path": artifacts["agent"]["path"],
        "config_source_fault": FAULT, "init_noise_std": None,
    }


# --- train_with_recovery_mode.execute가 쓰는 H 인터페이스 ---
def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS:
        raise ValueError("manifest fields differ from the fixed schema")
    demo_tag = _demo_tag_of(manifest["tag"])
    folder = fix_folder(demo_tag)
    if Path(path).is_symlink() or Path(path).resolve() != (folder / "train/manifest.json").resolve():
        raise ValueError("manifest is outside its registered arm")
    request = read(folder / "request.json")
    E = _demo()
    experiment = E.experiment_commit(demo_tag)
    protocol, commit = E.verify_protocol(Path(request["isaaclab_root"]), experiment)
    if (request["source_commit_sha"] != experiment or commit != experiment
            or request["protocol_sha256_lf"] != sha(PROTOCOL, True)):
        raise ValueError("request source commitment changed")
    if manifest != manifest_for(folder, request):
        raise ValueError("manifest differs from the approved request")
    approval = read(folder / "approval.json")
    if approval.get("request_sha256") != sha(folder / "request.json"):
        raise ValueError("approval record is not bound to the request")
    if read(folder / "consumed.json") != {"approval_sha256": sha(folder / "approval.json"), "arm": ARM,
                                          "train_limit": 1, "eval_limit": 1}:
        raise ValueError("arm consumption record differs")
    return manifest, protocol


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    """restart 학습 설정은 시연 F1 실행 설정과 seed(등록 재학습 seed)·run_name만 다를 수 있다."""
    artifacts = source_run(_demo_tag_of(manifest["tag"]))
    source_env, source_agent = Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"])
    for path, key in ((source_env, "environment"), (source_agent, "agent")):
        if str(path) != artifacts[key]["path"] or sha(path) != artifacts[key]["sha256"]:
            raise ValueError("source configuration changed before learning")
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env = load_config(Path(log_dir) / "params/env.yaml")
    current_agent = load_config(Path(log_dir) / "params/agent.yaml")
    if {k: v for k, v in original_env.items() if k != "seed"} != {k: v for k, v in current_env.items() if k != "seed"}:
        raise ValueError("environment configuration changed outside the registered code correction and seed")
    if ({k: v for k, v in original_agent.items() if k not in ("run_name", "seed")}
            != {k: v for k, v in current_agent.items() if k not in ("run_name", "seed")}):
        raise ValueError("agent configuration changed outside run_name and seed")
    if current_agent.get("run_name") != manifest["run_name"]:
        raise ValueError("agent run_name differs from the manifest")
    if current_env.get("seed") != RESTART_SEED or current_agent.get("seed") != RESTART_SEED or manifest["seed"] != RESTART_SEED:
        raise ValueError("restart training seed differs from the registered restart seed")
    if (original_agent.get("seed") != SEED or original_agent.get("num_steps_per_env") != NUM_STEPS_PER_ENV
            or original_agent.get("max_iterations") != MAX_ITERATIONS or original_env.get("seed") != SEED
            or original_env.get("scene", {}).get("num_envs") != NUM_ENVS):
        raise ValueError("source training seed, environment count, or step count differs")


# --- 부모: 학습 검증·고정 평가·판정 ---
def _sha_text(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _rng_ok(rng) -> bool:
    return (isinstance(rng, dict) and set(rng) == {"cpu_sha256", "cuda_all_sha256", "cuda_device_count"}
            and _sha_text(rng["cpu_sha256"]) and isinstance(rng["cuda_all_sha256"], list)
            and bool(rng["cuda_all_sha256"]) and rng["cuda_device_count"] == len(rng["cuda_all_sha256"])
            and all(_sha_text(value) for value in rng["cuda_all_sha256"]))


def verify_training(folder: Path, execution: dict, protocol: dict, versions: dict) -> dict:
    """v4 restart 분기와 같은 검사를 이 시연의 상수로 한다. versions는 시연 고정 평가 report의 실행 환경."""
    training = folder / "train"
    if not execution_valid(execution):
        raise ValueError(f"training execution failed (exit={execution.get('exit_code')}); inspect its stderr.log")
    manifest, _ = validate_manifest(training / "manifest.json")
    receipt, initial = read(training / "receipt.json"), read(training / "initial.json")
    if (receipt.get("completed") is not True or receipt.get("schema") != "training_correction_receipt_v2"
            or initial.get("schema") != "training_correction_initial_v2" or initial.get("recovery_mode") != "restart"):
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
    steps = MAX_ITERATIONS * NUM_STEPS_PER_ENV
    if (receipt.get("actual_control_steps") != steps or receipt.get("environment_make_calls") != 1
            or receipt.get("aggregate_env_steps") != steps * NUM_ENVS
            or receipt.get("budget") != {"seed": RESTART_SEED, "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS,
                                         "num_steps_per_env": NUM_STEPS_PER_ENV, "expected_control_steps": steps}):
        raise ValueError("training budget or environment count differs")
    if (receipt.get("resume_checkpoint") is not None or initial.get("resume_checkpoint") is not None
            or receipt.get("resume_iteration") is not None or initial.get("resume_iteration") is not None
            or receipt.get("source_checkpoint_state") is not None or initial.get("source_checkpoint_state") is not None
            or receipt.get("checkpoint_load_calls") != 0 or initial.get("std_reset") is not None
            or receipt.get("std_reset_applied") is not False
            or receipt.get("start_iteration") != 0 or initial.get("start_iteration") != 0):
        raise ValueError("restart arm loaded or reset a checkpoint")
    final_state = receipt.get("final_checkpoint_state") or {}
    if final_state.get("iteration") != FINAL_ITERATION or any(
            not _sha_text(final_state.get(k)) for k in ("policy_state_sha256", "optimizer_state_sha256")):
        raise ValueError("checkpoint state proof is incomplete")
    if (receipt.get("resume_scope") != {"policy_optimizer_restored": False, "full_environment_state_restored": False,
                                        "rng_state_restored_from_checkpoint": False, "fresh_initialization": True,
                                        "action_std_reset": False}
            or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False
            or receipt.get("rng_state_before_learn") != initial.get("rng_state_before_learn")
            or not _rng_ok(initial.get("rng_state_before_learn"))):
        raise ValueError("restart scope, episode initialization, or RNG observation differs")
    observed = {"environment_seed": RESTART_SEED, "environment_num_envs": NUM_ENVS, "action_scale": ACTION_SCALE,
                "agent_seed": RESTART_SEED, "agent_max_iterations": MAX_ITERATIONS,
                "runner_num_steps_per_env": NUM_STEPS_PER_ENV}
    if (receipt.get("observed") != observed or initial.get("observed") != observed
            or receipt.get("actual_action_scale") != ACTION_SCALE):
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
    validate_readback(correction, FAULT, CORRECTION)
    for key, target in protocol["correction_source_contract"]["targets"].items():
        if correction["before_fault"][key] != target["clean_fingerprint"]:
            raise ValueError("clean live target differs from committed SDK source")
    log_root = (Path(manifest["isaaclab_root"]) / "logs/rsl_rl/unitree_go2_flat").resolve()
    checkpoint = Path(receipt["checkpoint"]["path"])
    if (checkpoint.resolve().parent.parent != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"])
            or checkpoint.name != f"model_{FINAL_ITERATION}.pt"
            or Path(initial["log_dir"]).resolve() != checkpoint.parent.resolve()):
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


def judge(report: dict, none_report: dict, bands: dict) -> dict:
    """같은 seed NONE 고정 평가 대비 p0c 밴드 판정. healthy=recovered, unhealthy=not_recovered, 그 밖은 unknown."""
    verdict = RC.verdict(RC.compare(C._behavior(report), C._behavior(none_report)), bands)
    label = {"healthy": "recovered", "unhealthy": "not_recovered"}.get(verdict["label"], "unknown")
    return {"verdict": verdict, "label": label,
            "standstill_ratio": report["metrics"]["lin_vel_rmse_mps"] / report["standstill"]["lin_vel_rmse_mps"]}


def _run(request: dict, demo_tag: str, isaaclab: Path) -> dict:
    E = _demo()
    tag = fix_tag(demo_tag)
    if (request.get("case") != F1_RUN or request.get("correction_id") != CORRECTION
            or request.get("recovery_mode") != "restart" or request.get("tag") != tag):
        raise ValueError("correction request is outside the registered demo arm")
    protocol, commit = E.verify_protocol(isaaclab, E.experiment_commit(demo_tag))
    evaluation_step = E.step_record(demo_tag, "evaluate")
    none_report = read(ROOT / evaluation_step["reports"]["NONE"]["path"])
    folder = fix_folder(demo_tag)
    if folder.exists():
        raise FileExistsError("fix arm has already been consumed")
    for path in (ROOT, isaaclab, folder):
        if any(char in str(path) for char in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    if any(path.name.endswith(f"_{tag}_{ARM}") for path in log_root.iterdir()):
        raise FileExistsError("an SDK run already exists for this arm")
    if M.query_gpu(protocol["gpu_admission_profile"])["busy"]:
        raise RuntimeError("GPU busy before restart training")
    approval_event = E.correction_approval(request["request_id"])
    record = {"demo_tag": demo_tag, "tag": tag, "failure_seed": SEED, "restart_seed": RESTART_SEED, "case": F1_RUN, "arm": ARM,
              "correction_request_id": request["request_id"], "source_commit_sha": commit,
              "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve())}
    write_new(folder / "request.json", record)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"),
                                         "correction_ledger_approval": approval_event, "identity_attested": False})
    write_new(folder / "consumed.json", {"approval_sha256": sha(folder / "approval.json"), "arm": ARM,
                                         "train_limit": 1, "eval_limit": 1})
    training, evaluation = folder / "train", folder / "eval"
    write_new(training / "manifest.json", manifest_for(folder, record))
    argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "bench/demo_restart_correction_v2.py"),
            "--manifest", str(training / "manifest.json")]
    execution = P.run_owned(training, argv, isaaclab, TRAIN_WATCHDOG_S)
    write_new(training / "execution.json", execution)
    costs = {"train_parent_wall_s": execution["wall_s"]}
    receipt = verify_training(folder, execution, protocol, none_report["versions"])
    evaluation.mkdir()
    write_new(evaluation / "jobs.json", {"action_scale": ACTION_SCALE, "jobs": [
        {"name": ARM, "checkpoint": receipt["checkpoint"]["path"], "output": str(evaluation / "report.json")}]})
    gpu = M.query_gpu(protocol["gpu_admission_profile"])
    write_new(evaluation / "gpu_before.json", gpu)
    if gpu["busy"]:
        raise RuntimeError("GPU busy before fixed evaluation")
    argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"), "--protocol",
            str(V3.FIXED_EVAL), "--jobs", str(evaluation / "jobs.json"), "--headless"]
    eval_execution = P.run_owned(evaluation, argv, isaaclab, EVAL_WATCHDOG_S)
    write_new(evaluation / "execution.json", eval_execution)
    costs["eval_parent_wall_s"] = eval_execution["wall_s"]
    if not execution_valid(eval_execution):
        raise ValueError("fixed evaluation did not complete and clean up")
    report = C.validate_fixed_report(read(evaluation / "report.json"), {
        "name": ARM, "checkpoint_file": f"model_{FINAL_ITERATION}.pt",
        "checkpoint_sha256": receipt["checkpoint"]["sha256"], "action_scale": ACTION_SCALE,
        "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
    if report["versions"] != none_report["versions"] or report["versions"]["rsl-rl-lib"] != protocol["rsl_rl_version"]:
        raise ValueError("fixed evaluator runtime differs from the demo NONE evaluation")
    # 사후 검증: 실행 중 소스·체크포인트가 바뀌지 않았고 소유 프로세스가 GPU에 남지 않았다(v4 execute와 같은 순서).
    current, current_commit = E.verify_protocol(isaaclab, commit)
    if current != protocol or current_commit != commit:
        raise ValueError("source changed during the correction arm")
    if sha(Path(receipt["checkpoint"]["path"])) != receipt["checkpoint"]["sha256"]:
        raise ValueError("checkpoint changed during evaluation")
    after = M.query_gpu(protocol["gpu_admission_profile"])
    write_new(folder / "gpu_after.json", after)
    owned = set(execution["owned_process_identities"]) | set(eval_execution["owned_process_identities"])
    if any(str(row["pid"]) in owned for row in after["processes"]):
        raise ValueError("an owned GPU process remains")
    result = {"schema": "e2e_restart_correction_result_v2", "tag": tag, "case": F1_RUN, "arm": ARM,
              "restart_seed": RESTART_SEED, "reference_seed": SEED,
              "correction_request_id": request["request_id"], **judge(report, none_report, protocol["bands"]),
              "costs": costs, "training_receipt_sha256": sha(training / "receipt.json"),
              "fixed_report_sha256": sha(evaluation / "report.json"),
              "reference": evaluation_step["reports"]["NONE"]["path"], "source_commit_sha": commit}
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
