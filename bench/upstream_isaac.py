"""Isaac 쪽 공용 U1 진입점: 설계자 밖 결함(bench/upstream_faults.py)을 학습·probe·교정 학습 프로세스에 건다.

Isaac Sim 번들 파이썬으로 Isaac Lab 폴더에서 실행한다. 부모 쪽 호출은 bench/upstream_runs.py가 맡는다.
  python.bat bench/upstream_isaac.py train --fault U1 -- <train.py 인자>
  python.bat bench/upstream_isaac.py probes --fault U1 -- --jobs <jobs.json> --metrics <metrics.json> --headless
  python.bat bench/upstream_isaac.py correct --fault U1 --correction restore_body_frame_reward --contract <계약.json> \
      --readback <correction.json> -- <train.py 인자>

세 모드 모두 gymnasium.make 직전에 결함을 넣고 직후 두 줄을 남긴다.
- [upstream_fault]: upstream_faults.readback(본문 교체 유지·보상 항 결합). 교정 학습에서는 등록 표적이 원본으로 돌아가
  그 함수만 swapped가 False다.
- [upstream_aux]: 보조 지문. 보상 모듈 함수 지문을 주입 전과 환경 생성 뒤로 비교해 clean/differs_from_clean을 남긴다.
  판정에 쓰지 않는다. 결함을 라이브러리 함수 본문 교체로 이식했기 때문에 보이는 비교(B-code와 같은 종류)다.

correct 모드는 등록 교정을 capture(원본 확인) → 결함 주입 → apply(등록 표적 하나 복원) 순서로 걸고 교정 readback을 파일로
남긴다(receipt·runner hook 없는 단순 경로). 교정 시연은 bench/demo_correction_u1.py가 hidden_faults_shim으로
train_with_recovery_mode.execute를 그대로 부른다. upstream_faults.py·동결 probe·교정 학습 진입점은 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import FunctionType, ModuleType

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import correction_adapter as A  # noqa: E402
import upstream_faults as U  # noqa: E402

AUX_NOTE = ("보조 기록이며 판정에 쓰지 않는다. 등록 교정 readback은 등록 표적 3개만 대조하므로 그 밖 함수 상태는 이 기록에 있다. "
            "결함을 라이브러리 함수 본문 교체로 이식해서 보이는 비교다")


def _known(fault: str) -> None:
    if fault not in U.FAULTS:
        raise SystemExit(f"모르는 upstream 결함 {fault!r}: {sorted(U.FAULTS)}")


def snapshot() -> dict[str, str]:
    """보상 모듈에 정의된 함수의 실행 의미 지문(correction_adapter.fingerprint)."""
    module = importlib.import_module(U.REWARDS_MODULE)
    return {name: A.fingerprint(value) for name, value in sorted(vars(module).items())
            if isinstance(value, FunctionType) and value.__module__ == module.__name__}


def aux_readback(fault: str, before: dict[str, str], env) -> dict:
    """주입 전 지문 대비 지금 지문과 활성 보상 항 결합. 판정에 쓰지 않는다."""
    after = snapshot()
    differs = sorted(name for name, fingerprint in before.items() if after.get(name) != fingerprint)
    manager = env.unwrapped.reward_manager
    bound = {}
    for term in manager.active_terms:
        func = manager.get_term_cfg(term).func
        if getattr(func, "__module__", None) == U.REWARDS_MODULE:
            bound[term] = func.__name__
    return {"schema": "upstream_aux_fingerprints_v1", "fault": fault, "module": U.REWARDS_MODULE, "used_for_verdict": False,
            "note": AUX_NOTE, "differs_from_clean": differs, "clean_count": len(before) - len(differs),
            "active_reward_terms": bound, "differing_active_terms": sorted(t for t, f in bound.items() if f in differs)}


def arm(fault: str, inject=None, on_env=None) -> None:
    """gymnasium.make를 감싸 첫 환경 생성 직전에 inject(기본은 결함 주입)를 하고 직후 확인 줄 두 개를 남긴다."""
    _known(fault)
    import gymnasium

    original_make = gymnasium.make
    state = {"armed": False}

    def make(*args, **kwargs):
        if state["armed"]:
            return original_make(*args, **kwargs)
        state["armed"] = True
        before = snapshot()
        (inject or U.before_env)(fault)
        env = original_make(*args, **kwargs)
        print("[upstream_fault] " + json.dumps(U.readback(fault, env)), flush=True)
        aux = aux_readback(fault, before, env)
        print("[upstream_aux] " + json.dumps(aux), flush=True)
        if on_env is not None:
            on_env(env, aux)
        return env

    gymnasium.make = make


def hidden_faults_shim(fault: str) -> ModuleType:
    """train_with_recovery_mode.execute가 쓰는 hidden_faults의 _before_env·arm 계약을 U1로 채운 모듈.

    execute는 _before_env를 capture → 원래 _before_env → apply 순서의 함수로 바꾼 뒤 arm을 부른다. arm은 첫 환경 생성 때
    그 시점의 _before_env를 부르므로 교정 순서가 그대로 지켜진다. 주입 전 지문은 원래 _before_env 안에서 남긴다.
    """
    _known(fault)
    shim = ModuleType("hidden_faults")
    state: dict = {}

    def before_env(name: str) -> None:
        if name != fault:
            raise RuntimeError(f"shim은 {fault}만 넣는다: {name!r}")
        state["before"] = snapshot()
        U.before_env(name)

    def shim_arm(name: str) -> None:
        if name != fault:
            raise SystemExit(f"shim은 {fault}만 넣는다: {name!r}")
        import gymnasium

        original_make = gymnasium.make
        armed = {"done": False}

        def make(*args, **kwargs):
            if armed["done"]:
                return original_make(*args, **kwargs)
            armed["done"] = True
            shim._before_env(name)  # execute가 바꾼 함수(capture → before_env → apply)를 호출 시점에 찾는다
            env = original_make(*args, **kwargs)
            print("[upstream_fault] " + json.dumps(U.readback(name, env)), flush=True)
            print("[upstream_aux] " + json.dumps(aux_readback(name, state["before"], env)), flush=True)
            return env

        gymnasium.make = make

    shim._before_env = before_env
    shim.arm = shim_arm
    shim.FAULTS = dict(U.FAULTS)
    return shim


def correction_inject(fault: str, correction_id: str, isaaclab_root: Path, contract: dict, readback_path: Path):
    """단순 교정 경로의 inject: capture(원본 확인) → 결함 주입 → 등록 표적 하나 복원 → 교정 readback 파일."""
    if correction_id not in A.TARGETS:
        raise SystemExit(f"모르는 등록 교정 {correction_id!r}")

    def inject(name: str) -> None:
        snap = A.capture(isaaclab_root, contract)
        U.before_env(name)
        readback = A.apply(snap, correction_id)
        target = A.TARGETS[correction_id]
        restored = readback["after_correction"][correction_id] == readback["before_fault"][correction_id]
        body = {"schema": "upstream_correction_readback_v1", "fault": name, "correction_id": correction_id,
                "registered_target": f"{target['module']}.{target['qualname']}",
                "registered_target_state": "restored_to_clean" if restored else "not_restored",
                "scope_note": "등록 교정 readback은 등록 표적 3개만 대조한다. 그 밖 함수 상태는 보조 지문 기록([upstream_aux])에 있다",
                "registered_readback": readback}
        readback_path.parent.mkdir(parents=True, exist_ok=True)
        with readback_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(body, ensure_ascii=False, indent=1) + "\n")

    return inject


def _run_train(rest: list[str]) -> None:
    upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        raise SystemExit(f"Isaac Lab 폴더에서 실행해야 한다: {upstream}")
    sys.path.insert(0, str(upstream.parent))
    sys.argv = [str(upstream), *rest]
    runpy.run_path(str(upstream), run_name="__main__")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    head, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("train", "probes", "correct"))
    parser.add_argument("--fault", required=True)
    parser.add_argument("--correction")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--readback", type=Path)
    args = parser.parse_args(head)
    if args.mode == "probes":
        arm(args.fault)
        import metered_probes
        return metered_probes.main(rest)
    if args.mode == "correct":
        if not (args.correction and args.contract and args.readback):
            parser.error("correct에는 --correction·--contract·--readback이 필요하다")
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        aux_path = args.readback.with_name(args.readback.stem + ".aux.json")

        def write_aux(env, aux) -> None:
            with aux_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(aux, ensure_ascii=False, indent=1) + "\n")

        arm(args.fault, correction_inject(args.fault, args.correction, Path(os.getcwd()), contract, args.readback),
            write_aux)
    else:
        arm(args.fault)
    _run_train(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
