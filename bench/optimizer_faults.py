"""설계자 밖 외부 결함(옵티마이저): 다른 구현에서 원인이 밝혀진 결함의 의미를 RSL-RL 2.3.3 PPO에 옮긴다.

R1 — rsl_rl #38(2024-08-30, 보고자 확인). 지금은 지워진 algorithms 브랜치(끝 커밋 96393c41)의 LegacyRunner는 설정
매핑 표에서 num_learning_epochs를 버리고(("num_learning_epochs", None)) num_mini_batches만 batch_count로 넘겼다.
그 PPO.update는 롤아웃 배치를 batch_count개 미니배치로 한 번만 돌았다. 보고자는 Isaac Lab flat ANYmal 300회에서
속도 명령을 전혀 따르지 못했고, epoch 5를 다시 넣어 해결했다고 적었다. 메인테이너 답변은 없다.

여기서는 RSL-RL 2.3.3 PPO 인스턴스가 만들어진 직후 num_learning_epochs만 1로 바꾼다. 설정(agent.yaml)에는 원래 값(Go2 5)이
그대로 남는다. 2.3.3 update는 이 값을 미니배치 생성기의 epoch 수와 손실 평균의 분모에 함께 쓰므로, 롤아웃을 미니배치
4개로 한 번 돌고 손실은 실제 갱신 4회로 평균한다(원 브랜치도 실제 갱신 수로 평균했다). 적응 학습률은 두 구현 모두
미니배치마다 같은 규칙(KL > 2·desired면 ÷1.5, < desired/2면 ×1.5, 1e-5~1e-2)이라 손대지 않는다.
결함 정의는 원전에서만 정하며 결과를 보고 고치지 않는다(bench/protocols/upstream_r1_smoke_v1.json).

bench/train_with_optimizer_fault.py가 TRIAGE_OPTIMIZER_FAULT로 arm()한다. NONE이면 아무것도 바꾸지 않고 같은 확인 줄만
남긴다(같은 진입점의 정상 기준 학습). Isaac Sim 번들 파이썬에서만 쓰인다. bench/upstream_faults.py(U1)와 별개 모듈이다.
"""
from __future__ import annotations

FAULTS = {
    "R1": {
        "category": "optimizer",
        "note": "PPO가 설정의 num_learning_epochs를 쓰지 않아 롤아웃 배치를 한 번만 갱신에 쓴다(설정 파일에는 원래 값이 남음)",
        "sources": ["https://github.com/leggedrobotics/rsl_rl/issues/38",
                    "https://github.com/leggedrobotics/rsl_rl/blob/96393c41c55c0a905eff035875b97f3b837225fe/"
                    "rsl_rl/runners/legacy_runner.py#L38"],
    },
}
EFFECTIVE_EPOCHS = {"R1": 1}
ARMABLE = ("NONE", *FAULTS)
PPO_MODULE = "rsl_rl.algorithms.ppo"
PREFIX = "[optimizer_fault] "


def apply(fault: str, ppo) -> dict:
    """만들어진 PPO 인스턴스에 결함을 건다. NONE은 바꾸지 않는다. 확인 줄에 쓸 값을 돌려준다."""
    configured = ppo.num_learning_epochs
    if fault != "NONE":
        ppo.num_learning_epochs = EFFECTIVE_EPOCHS[fault]
    return {"event": "construct", "fault": fault, "class": f"{type(ppo).__module__}.{type(ppo).__qualname__}",
            "configured_epochs": configured, "effective_epochs": ppo.num_learning_epochs,
            "num_mini_batches": ppo.num_mini_batches, "schedule": ppo.schedule, "desired_kl": ppo.desired_kl}


def count_first_update(ppo, update) -> tuple[object, int, int]:
    """update 한 번을 돌리며 생성기가 낸 미니배치 수와 optimizer.step 호출 수를 따로 센다."""
    storage = ppo.storage
    name = "recurrent_mini_batch_generator" if ppo.policy.is_recurrent else "mini_batch_generator"
    generator, step = getattr(storage, name), ppo.optimizer.step
    counts = {"batches": 0, "steps": 0}

    def counting_generator(*args, **kwargs):
        for item in generator(*args, **kwargs):
            counts["batches"] += 1
            yield item

    def counting_step(*args, **kwargs):
        counts["steps"] += 1
        return step(*args, **kwargs)

    setattr(storage, name, counting_generator)  # 인스턴스 속성이 클래스 메서드를 잠깐 가린다
    ppo.optimizer.step = counting_step
    try:
        out = update(ppo)
    finally:
        delattr(storage, name)
        del ppo.optimizer.step
    return out, counts["batches"], counts["steps"]


def patch_ppo(fault: str, ppo_class) -> None:
    """PPO 생성 직후 결함을 걸고, 생성 때와 첫 update 뒤에 확인 줄을 하나씩 남긴다."""
    import json

    original_init, original_update = ppo_class.__init__, ppo_class.update
    state = {"first_update_done": False}

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        print(PREFIX + json.dumps(apply(fault, self)), flush=True)

    def update(self):
        if state["first_update_done"]:
            return original_update(self)
        state["first_update_done"] = True
        out, batches, steps = count_first_update(self, original_update)
        print(PREFIX + json.dumps({"event": "first_update", "fault": fault, "minibatches": batches, "optimizer_steps": steps,
                                   "expected": self.num_learning_epochs * self.num_mini_batches}), flush=True)
        return out

    ppo_class.__init__, ppo_class.update = __init__, update


def arm(fault: str) -> None:
    """gymnasium.make 첫 호출(train.py가 Isaac Sim을 띄운 뒤, runner 생성 전)에 PPO 클래스를 패치한다.

    torch·rsl_rl을 앱 실행 전에 import하지 않으려고 U1·P0-C 주입과 같은 시점을 쓴다.
    """
    if fault not in ARMABLE:
        raise SystemExit(f"모르는 TRIAGE_OPTIMIZER_FAULT {fault!r}: {list(ARMABLE)}")
    import importlib

    import gymnasium

    original_make = gymnasium.make
    state = {"armed": False}

    def make(*args, **kwargs):
        if not state["armed"]:
            state["armed"] = True
            patch_ppo(fault, importlib.import_module(PPO_MODULE).PPO)
        return original_make(*args, **kwargs)

    gymnasium.make = make


def parse(log_text: str) -> list[dict]:
    """학습 stdout의 확인 줄들. 형식이 깨진 줄은 None으로 둔다."""
    import json

    out = []
    for line in log_text.splitlines():
        if line.startswith(PREFIX):
            try:
                out.append(json.loads(line[len(PREFIX):]))
            except json.JSONDecodeError:
                out.append(None)
    return out


def readback_ok(lines: list[dict], fault: str, configured_epochs: int, num_mini_batches: int) -> bool:
    """확인 줄이 생성 하나·첫 update 하나이고, 설정값은 그대로이며 실제 epoch·미니배치·optimizer 단계가 결함 정의와 같다."""
    if len(lines) != 2 or any(not isinstance(line, dict) for line in lines):
        return False
    construct, first = lines
    effective = configured_epochs if fault == "NONE" else EFFECTIVE_EPOCHS.get(fault)
    expected = None if effective is None else effective * num_mini_batches
    return (construct.get("event") == "construct" and construct.get("fault") == fault
            and construct.get("class") == f"{PPO_MODULE}.PPO" and construct.get("configured_epochs") == configured_epochs
            and construct.get("effective_epochs") == effective and construct.get("num_mini_batches") == num_mini_batches
            and first.get("event") == "first_update" and first.get("fault") == fault
            and first.get("minibatches") == expected and first.get("optimizer_steps") == expected
            and first.get("expected") == expected)
