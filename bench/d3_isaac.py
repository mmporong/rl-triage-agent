"""Isaac 쪽 D3 probe 진입점: probe 프로세스에 변경 부분집합(TRIAGE_D3_CHANGES)을 건 뒤 동결 evals/probes.py를 그대로 돌린다.

Isaac Lab 폴더에서 Isaac Sim 번들 파이썬으로 실행한다(evals/run_d3_experiment.py가 호출).
  set TRIAGE_D3_CHANGES=H1,...,U1 && _isaac_sim\\python.bat <repo>\\bench\\d3_isaac.py -- --jobs <jobs.json> --headless

- 학습은 bench/train_with_d3_changes.py가 설정 변경을 hydra override로 넣는다. probes.py는 hydra 없이 기본 env 설정으로
  환경을 만들므로, 여기서는 첫 gymnasium.make 직전에 코드 교체(d3_changes.before_env)와 함께 env 설정 override를
  그 cfg 객체에 같은 값으로 넣는다. agent 설정(H7 등)은 probe 측정에 쓰이지 않는다.
- 확인 줄 [d3_changes] 하나를 남긴다(d3_changes.readback: 교체 유지·결합·env 설정 값).
- U1은 d3_changes 카탈로그가 bench/upstream_faults.SWAPS를 그대로 쓰므로 무해 변경과 같은 프로세스에서 결합된다.
  sub_2의 bench/upstream_isaac.py(결함 하나 단위)는 쓰지도 고치지도 않는다.
- jobs의 fault는 NONE으로 둔다. 동결 probes.py는 hidden_faults._before_env("NONE")·_after_env("NONE")를 부르고
  둘 다 아무것도 하지 않는다.
"""
from __future__ import annotations

from functools import reduce
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import d3_changes as D  # noqa: E402

PROBES_SCRIPT = ROOT / "evals" / "probes.py"


def apply_env_overrides(cfg, ids) -> None:
    """env.* override를 cfg 객체에 같은 값으로 넣는다(dict는 키, 설정 객체는 속성)."""
    for cid in ids:
        for key, (_old, new) in D.CHANGES[cid].get("overrides", {}).items():
            if not key.startswith("env."):
                continue
            *parents, leaf = key[len("env."):].split(".")
            parent = reduce(lambda obj, part: obj[part] if isinstance(obj, dict) else getattr(obj, part), parents, cfg)
            if isinstance(parent, dict):
                parent[leaf] = new
            else:
                setattr(parent, leaf, new)


def arm_for_probes(spec: str) -> tuple[str, ...]:
    """gymnasium.make를 감싸 첫 환경 생성 직전에 코드 교체·env 설정 적용을 하고 직후 확인 줄을 하나 남긴다."""
    try:
        ids = D.parse_subset(spec)
    except ValueError as error:
        raise SystemExit(f"{D.ENV_VAR}: {error}") from error
    import gymnasium

    original_make = gymnasium.make
    state = {"armed": False}

    def make(*args, **kwargs):
        if state["armed"]:
            return original_make(*args, **kwargs)
        state["armed"] = True
        if "cfg" not in kwargs:
            raise SystemExit("probe 환경 생성에 cfg 인자가 없다")
        D.before_env(ids)
        apply_env_overrides(kwargs["cfg"], ids)
        env = original_make(*args, **kwargs)
        try:
            line = D.readback(ids, env)
        except Exception as error:  # 확인 실패는 ok=false로 남기고 부모 쪽 판정에 맡긴다
            line = {"changes": list(ids), "ok": False, "error": type(error).__name__}
            print(f"[d3_changes_error] {error}", file=sys.stderr, flush=True)
        print(D.READBACK_PREFIX + json.dumps(line), flush=True)
        return env

    gymnasium.make = make
    return ids


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = argv[argv.index("--") + 1:] if "--" in argv else argv
    arm_for_probes(os.environ.get(D.ENV_VAR, ""))
    sys.argv = [str(PROBES_SCRIPT), *rest]
    runpy.run_path(str(PROBES_SCRIPT), run_name="__main__")  # probes.py는 끝에서 SystemExit(main())을 낸다
    return 0


if __name__ == "__main__":
    sys.exit(main())
