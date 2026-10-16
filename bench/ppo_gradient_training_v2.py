"""등록된 세 PPO update를 실행하며 gradient 관찰과 OFF/ON 비교 증거를 남긴다."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v2 as Q
from ppo_gradient_training_v1 import attach


def verify_training(fixed: dict, output: Path, arguments: list[str], isaaclab: Path) -> dict:
    protocol, anchor = Q.verify(isaaclab, fixed["source_commit_sha"])
    matching = [spec for spec in Q.specs() if spec["id"] == fixed.get("run_id")]
    if len(matching) != 1 or fixed != Q.manifest(matching[0], protocol, anchor):
        raise ValueError("학습 manifest가 등록 arm과 다르다")
    spec = matching[0]
    tag = output.parent.name
    if not Q.TAG.fullmatch(tag) or output.resolve() != (Q.PRIVATE / "runs" / tag / spec["id"]).resolve():
        raise ValueError("학습 출력 폴더가 등록 arm 경로와 다르다")
    expected = Q.training_args(isaaclab, output, spec, tag)
    if arguments != expected[expected.index("--") + 1:]:
        raise ValueError("학습 인자가 등록 task·seed·예산·run_name과 다르다")
    expected_launch = {"args": expected, "cwd": str(isaaclab.resolve())}
    if (Q.read(output / "launch_intent.json") != expected_launch
            or Q.read(output / "launch.json") != expected_launch):
        raise ValueError("controller 의도·run_owned launch·학습 인자가 다르다")
    for name in ("records.jsonl", "training_evidence.json"):
        if (output / name).exists():
            raise FileExistsError("학습 증거 파일을 덮어쓰지 않는다")
    plan = Q.read(output.parent / "plan.json")
    lease = Q.lease_context(plan["lease"]["childPid"])
    if lease["leaseId"] != plan["lease"]["leaseId"]:
        raise ValueError("학습 진입 전에 GPU lease가 바뀌었다")
    return spec


def train(manifest_path: Path, output: Path, arguments: list[str]):
    fixed = Q.read(manifest_path)
    isaaclab = Path.cwd()
    verify_training(fixed, output, arguments, isaaclab)
    upstream = isaaclab / "scripts/reinforcement_learning/rsl_rl/train.py"
    if not upstream.is_file():
        raise FileNotFoundError(upstream)
    import gymnasium
    original_make = gymnasium.make
    original_argv, original_path = sys.argv[:], sys.path[:]
    state: dict = {"attached": [], "patched": False}
    evidence = None
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        def sink(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        def make(*args, **kwargs):
            if not state["patched"]:
                ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
                original_init = ppo_class.__init__

                def init(this, *positional, **keyword):
                    original_init(this, *positional, **keyword)
                    if state["attached"]:
                        raise ValueError("한 arm에서 PPO 인스턴스를 둘 이상 만들 수 없다")
                    finish, cleanup = attach(this, fixed, sink)
                    state["attached"].append((this, finish, cleanup))

                state.update(patched=True, ppo_class=ppo_class, original_init=original_init)
                ppo_class.__init__ = init
            return original_make(*args, **kwargs)

        try:
            gymnasium.make = make
            sys.path.insert(0, str(upstream.parent))
            sys.argv = [str(upstream), *arguments]
            runpy.run_path(str(upstream), run_name="__main__")
            if len(state["attached"]) != 1:
                raise ValueError("PPO 생성·계측 설치 증거가 없다")
            ppo, finish, _ = state["attached"][0]
            evidence = finish()
            clean = ("update" not in ppo.__dict__
                     and "get_actions_log_prob" not in ppo.policy.__dict__
                     and "mini_batch_generator" not in ppo.storage.__dict__)
            if not clean:
                raise ValueError("임시 PPO 메서드 복원 실패")
            evidence.update(schema="ppo_gradient_training_evidence_v1", method_cleanup=clean,
                manifest_sha256=Q.O.manifest_fingerprint(fixed), arm=fixed["arm"],
                observed=fixed["observer_enabled"])
        finally:
            gymnasium.make = original_make
            for _, _, cleanup in state["attached"]:
                cleanup()
            if state["patched"]:
                state["ppo_class"].__init__ = state["original_init"]
            sys.argv = original_argv
            sys.path[:] = original_path
    if evidence is None:
        raise ValueError("완료 학습 증거가 없다")
    Q.write_new(output / "training_evidence.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("upstream", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
    train(args.manifest, args.output, arguments)


if __name__ == "__main__":
    main()
