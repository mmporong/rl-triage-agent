"""task별 고정 평가 구동기: 동결 evals/fixed_eval.py를 그대로 부르고, 프로토콜의 env_events_disabled 이벤트만 평가 환경에서 끈다.

fixed_eval.py는 바꾸지 않는다. evaluate()가 쓰는 isaaclab_tasks.utils.parse_env_cfg의 결과에서 그 이벤트 항을 None으로 두고,
원래 켜져 있었는지와 끈 뒤 상태를 [eval_env] 줄 하나로 남긴다. 격자·지평·명령 덮어쓰기·계산·출력은 fixed_eval.py 그대로다.
학습 설정은 바꾸지 않는다(평가 환경만).

  <IsaacLab>\\_isaac_sim\\python.bat evals/fixed_eval_task.py --protocol bench/protocols/fixed_eval_v1_anymal_c.json --jobs <jobs.json> --headless
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_eval  # noqa: E402

PREFIX = "[eval_env] "


def disabled_events(protocol_path) -> list[str]:
    return list(json.loads(Path(protocol_path).read_text(encoding="utf-8")).get("env_events_disabled") or [])


def disable(cfg, names: list[str], task) -> dict:
    """환경 설정의 이벤트 항을 끄고 확인 줄에 쓸 값을 돌려준다."""
    before = {name: getattr(cfg.events, name, None) is not None for name in names}
    for name in names:
        setattr(cfg.events, name, None)
    return {"task": task, "disabled": names, "was_enabled": before,
            "now_enabled": {name: getattr(cfg.events, name, None) is not None for name in names}}


def wrap(original_evaluate, utils_module=None):
    """evaluate(args)를 감싸 그 호출 동안만 parse_env_cfg 결과에서 이벤트를 끈다. 앱이 뜬 뒤에 isaaclab_tasks를 import한다."""
    def evaluate(args):
        utils = utils_module
        if utils is None:
            import isaaclab_tasks.utils as utils
        names, original_parse = disabled_events(args.protocol), utils.parse_env_cfg
        calls = {"n": 0}

        def parse_env_cfg(*parse_args, **kwargs):
            cfg = original_parse(*parse_args, **kwargs)
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("parse_env_cfg가 평가 한 번에 두 번 불렸다")
            print(PREFIX + json.dumps(disable(cfg, names, parse_args[0] if parse_args else kwargs.get("task_name"))), flush=True)
            return cfg

        utils.parse_env_cfg = parse_env_cfg
        try:
            return original_evaluate(args)
        finally:
            utils.parse_env_cfg = original_parse
    return evaluate


def parse(log_text: str) -> list:
    out = []
    for line in log_text.splitlines():
        if line.startswith(PREFIX):
            try:
                out.append(json.loads(line[len(PREFIX):]))
            except json.JSONDecodeError:
                out.append(None)
    return out


if __name__ == "__main__":
    fixed_eval.evaluate = wrap(fixed_eval.evaluate)
    raise SystemExit(fixed_eval.main())
