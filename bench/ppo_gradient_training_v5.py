"""묶음 계측을 사용하고 앱 종료 전에 완료 증거를 기록하는 v5 학습 진입점."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v5 as Q
import ppo_gradient_invariant_v2 as G
import ppo_ratio_mutation_v1 as MUT
import isaac_training_guard_v1 as GUARD

def verify_training(fixed: dict, output: Path, arguments: list[str], isaaclab: Path) -> dict:
    protocol, anchor = Q.verify(isaaclab, fixed["source_commit_sha"])
    matching = [spec for spec in [*Q.specs(), Q.dry_spec()] if spec["id"] == fixed.get("run_id")]
    if len(matching) != 1 or fixed != Q.manifest(matching[0], protocol, anchor):
        raise ValueError("학습 manifest가 등록 arm과 다르다")
    spec = matching[0]
    tag = output.parent.name
    category = "dry_runs" if spec == Q.dry_spec() else "runs"
    if not Q.TAG.fullmatch(tag) or output.resolve() != (Q.PRIVATE / category / tag / spec["id"]).resolve():
        raise ValueError("학습 출력 폴더가 등록 arm 경로와 다르다")
    expected = Q.training_args(isaaclab, output, spec, tag)
    if arguments != expected[expected.index("--") + 1:]:
        raise ValueError("학습 인자가 등록 task·seed·예산·run_name과 다르다")
    expected_launch = {"args": expected, "cwd": str(isaaclab.resolve())}
    if (Q.read(output / "launch_intent.json") != expected_launch
            or Q.read(output / "launch.json") != expected_launch):
        raise ValueError("controller 의도·run_owned launch·학습 인자가 다르다")
    for name in ("records.jsonl", "training_evidence.json", "training_failure.json", "close_ready.json"):
        if (output / name).exists():
            raise FileExistsError("학습 증거 파일을 덮어쓰지 않는다")
    plan = Q.read(output.parent / "plan.json")
    lease = Q.lease_context(plan["lease"]["childPid"])
    if lease["leaseId"] != plan["lease"]["leaseId"]:
        raise ValueError("학습 진입 전에 GPU lease가 바뀌었다")
    return spec


def attach(ppo, fixed: dict, sink):
    """모든 arm에 같은 시간 계측을 붙이고 ON에만 gradient 관찰을 붙인다."""
    import torch
    updates = fixed["max_iterations"]
    if type(updates) is not int or updates not in (1, 3):
        raise ValueError("등록 update 예산은 dry-run 1 또는 본배치 3이다")
    configuration = {"epochs": ppo.num_learning_epochs, "minibatches": ppo.num_mini_batches,
                     "clip_param": ppo.clip_param}
    if configuration != {"epochs": 5, "minibatches": 4, "clip_param": 0.2}:
        raise ValueError("실제 PPO 구성이 등록값과 다르다")
    if str(torch.__version__) != "2.7.0+cu128":
        raise ValueError("사전등록 torch 버전과 다르다")
    restore_fault, readback = MUT.install(ppo, fixed["arm"])
    observer = None
    try:
        if fixed["observer_enabled"]:
            observer = G.GradientObserver(ppo, fixed, sink).install()
    except BaseException:
        restore_fault()
        raise
    underlying = ppo.update
    wall, losses = [], []
    cuda = torch.device(ppo.device).type == "cuda"
    cleaned = False
    final_state = None

    def synchronize():
        if cuda:
            torch.cuda.synchronize(ppo.device)

    def timed(this):
        nonlocal final_state
        if len(wall) >= updates:
            raise ValueError("등록한 update를 초과했다")
        synchronize()
        started = time.perf_counter()
        result = underlying()
        synchronize()
        wall.append(time.perf_counter() - started)
        losses.append(result)
        if len(wall) == updates:
            # SimulationApp 종료 전에 CUDA RNG와 학습 상태를 확정한다.
            final_state = MUT.learning_state(ppo, cuda=cuda)
        return result

    ppo.update = MethodType(timed, ppo)

    def finish():
        try:
            if len(wall) != updates or final_state is None:
                raise ValueError("등록한 update가 완료되지 않았다")
            return {"state": final_state, "losses": losses, "updates": len(wall), "update_wall_s": wall,
                    "readback": readback, "runtime_configuration": configuration,
                    "torch_version": torch.__version__, "device_type": torch.device(ppo.device).type}
        finally:
            cleanup()

    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        ppo.update = underlying
        if observer is not None:
            observer.remove()
        restore_fault()
        cleaned = True

    return finish, cleanup


@contextmanager
def training_hooks(ppo_class, runner_class, fixed: dict, output: Path, sink):
    """실제 learn 반환 시점에서 완료 증거를 저장하며 모든 hook을 복원한다."""
    original_init, original_learn = ppo_class.__init__, runner_class.learn
    state: dict[str, Any] = {"attached": [], "learn_called": False, "learning_completed": False}

    def init(this, *args, **kwargs):
        try:
            original_init(this, *args, **kwargs)
            if state["attached"]:
                raise ValueError("PPO 인스턴스는 arm당 하나여야 한다")
            finish, cleanup = attach(this, fixed, sink)
            state["attached"].append((this, finish, cleanup))
        except BaseException as exc:
            GUARD.record_failure(output, exc, "ppo_init")
            raise

    def learn(this, *args, **kwargs):
        try:
            iterations = kwargs.get("num_learning_iterations", args[0] if args else None)
            random_ep = kwargs.get("init_at_random_ep_len", args[1] if len(args) > 1 else None)
            if state["learn_called"] or iterations != fixed["max_iterations"] or random_ep is not True:
                raise ValueError("learn 호출·예산이 등록값과 다르다")
            state["learn_called"] = True
            result = original_learn(this, *args, **kwargs)
            if len(state["attached"]) != 1 or this.alg is not state["attached"][0][0]:
                raise ValueError("runner와 계측 PPO 결합이 다르다")
            ppo, finish, _ = state["attached"][0]
            evidence = finish()
            clean = ("update" not in ppo.__dict__
                     and "get_actions_log_prob" not in ppo.policy.__dict__
                     and "mini_batch_generator" not in ppo.storage.__dict__)
            if not clean:
                raise ValueError("임시 PPO 메서드 복원 실패")
            evidence.update(schema="ppo_gradient_training_evidence_v1", method_cleanup=clean,
                manifest_sha256=Q.O.manifest_fingerprint(fixed), arm=fixed["arm"],
                observed=fixed["observer_enabled"], learning_completed=True,
                recorded_before_app_close=True)
            GUARD.durable_new(output / "training_evidence.json", evidence)
            state["learning_completed"] = True
            return result
        except BaseException as exc:
            GUARD.record_failure(output, exc, "runner_learn")
            raise

    ppo_class.__init__, runner_class.learn = init, learn
    try:
        yield state
    finally:
        ppo_class.__init__, runner_class.learn = original_init, original_learn
        for _, _, cleanup in state["attached"]:
            cleanup()


def before_close(output: Path, fixed: dict, state: dict, stream) -> None:
    if not state["learning_completed"] or (output / "training_failure.json").exists():
        raise ValueError("정상 학습 완료 증거 없이 앱을 닫을 수 없다")
    stream.flush()
    os.fsync(stream.fileno())
    GUARD.durable_new(output / "close_ready.json", {
        "schema": "ppo_gradient_close_ready_v4", "manifest_sha256": Q.O.manifest_fingerprint(fixed),
        "updates": fixed["max_iterations"], "evidence_sha256_lf": Q.sha(output / "training_evidence.json"),
        "records_sha256_lf": Q.sha(output / "records.jsonl"),
    })
    GUARD.validate_completion(output, Q.O.manifest_fingerprint(fixed))


def train(manifest_path: Path, output: Path, arguments: list[str]):
    fixed = Q.read(manifest_path)
    isaaclab = Path.cwd()
    verify_training(fixed, output, arguments, isaaclab)
    upstream = isaaclab / "scripts/reinforcement_learning/rsl_rl/train.py"
    import gymnasium
    original_make = gymnasium.make
    original_argv, original_path = sys.argv[:], sys.path[:]
    original_hydra = os.environ.get("HYDRA_FULL_ERROR")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    hooks = None
    state = None
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        def sink(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        def make(*args, **kwargs):
            nonlocal hooks, state
            try:
                if hooks is not None:
                    raise ValueError("환경은 arm당 하나여야 한다")
                ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
                runner_class = importlib.import_module("rsl_rl.runners").OnPolicyRunner
                hooks = training_hooks(ppo_class, runner_class, fixed, output, sink)
                state = hooks.__enter__()
                return original_make(*args, **kwargs)
            except BaseException as exc:
                GUARD.record_failure(output, exc, "environment_make")
                raise

        def closing(app):
            if state is None:
                raise ValueError("학습 hook 설치 증거가 없다")
            before_close(output, fixed, state, stream)

        try:
            gymnasium.make = make
            sys.path.insert(0, str(upstream.parent))
            sys.argv = [str(upstream), *arguments]
            GUARD.run_upstream(upstream, output, closing)
        finally:
            gymnasium.make = original_make
            if hooks is not None:
                hooks.__exit__(None, None, None)
            sys.argv, sys.path[:] = original_argv, original_path
            if original_hydra is None:
                os.environ.pop("HYDRA_FULL_ERROR", None)
            else:
                os.environ["HYDRA_FULL_ERROR"] = original_hydra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("upstream", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
    try:
        train(args.manifest, args.output, arguments)
    except BaseException as exc:
        try:
            GUARD.record_failure(args.output, exc, "training_entrypoint")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                # Kit의 종료 hook이 실패를 exit 0으로 바꾸기 전에 실패 코드를 확정한다.
                os._exit(1)


if __name__ == "__main__":
    main()
