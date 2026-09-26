"""결함 주입 벤치마크 카탈로그와 케이스 생성기.

각 케이스 = 해로운 변경 1개 + 무해한 변경 2개를 섞은 설정 변경 목록.
에이전트는 변경 목록(순서 섞임)과 텔레메트리만 받고, 어떤 변경이 학습 실패의 원인인지 찾아야 한다.
정답표는 bench/private/answer_key.json 에만 쓰며 평가가 끝난 뒤 공개한다.

무해 변경은 benign_all 실행(무해 변경 전부 적용)으로 기준선과 차이가 작음을 따로 확인한다.
"""
import json
import random
import sys
from pathlib import Path

# 원인 범주: reward / optimizer / exploration / actuator / physics / termination / command
HARMFUL = {
    "H01": {"category": "reward", "override": "env.rewards.track_lin_vel_xy_exp.weight=-1.5",
            "note": "속도 추종 보상 부호 반전"},
    "H02": {"category": "actuator", "override": "env.actions.joint_pos.scale=1.5",
            "note": "관절 위치 액션 스케일 6배"},
    "H03": {"category": "exploration", "override": "agent.policy.init_noise_std=0.01",
            "note": "초기 탐색 노이즈 거의 0"},
    "H04": {"category": "optimizer", "override": "agent.algorithm.gamma=0.5",
            "note": "할인율 과소(근시안)"},
    "H05": {"category": "reward", "override": "env.rewards.dof_torques_l2.weight=-0.2",
            "note": "토크 패널티 1000배"},
    "H06": {"category": "termination", "override": "env.episode_length_s=1.0",
            "note": "에피소드 길이 20초→1초"},
    "H07": {"category": "physics", "override": "env.sim.dt=0.02",
            "note": "물리 스텝 4배(제어 12.5Hz)"},
    "H08": {"category": "reward", "override": "env.rewards.feet_air_time.weight=5.0",
            "note": "발 체공 보상 20배"},
    "H09": {"category": "optimizer", "override": "agent.algorithm.value_loss_coef=0.0",
            "note": "가치함수 학습 끔"},
    "H10": {"category": "physics", "override": "env.events.add_base_mass.params.mass_distribution_params=[25.0,30.0]",
            "note": "몸통 질량 +25~30kg"},
}

BENIGN = {
    "B01": "env.rewards.dof_acc_l2.weight=-3.0e-07",
    "B02": "agent.algorithm.learning_rate=0.0012",
    "B03": "agent.save_interval=25",
    "B04": "env.rewards.action_rate_l2.weight=-0.012",
    "B05": "env.commands.base_velocity.resampling_time_range=[8.0,8.0]",
    "B06": "env.rewards.flat_orientation_l2.weight=-3.0",
}


def make_cases(secret_seed: int):
    rng = random.Random(secret_seed)
    harmful_ids = list(HARMFUL)
    rng.shuffle(harmful_ids)
    cases, key = [], {}
    for i, hid in enumerate(harmful_ids, start=1):
        cid = f"c{i:02d}"
        benign_ids = rng.sample(list(BENIGN), 2)
        changes = [("H", hid, HARMFUL[hid]["override"])] + [("B", b, BENIGN[b]) for b in benign_ids]
        rng.shuffle(changes)
        cases.append({"case_id": cid, "overrides": [c[2] for c in changes]})
        key[cid] = {"harmful_id": hid, "harmful_override": HARMFUL[hid]["override"],
                    "category": HARMFUL[hid]["category"], "note": HARMFUL[hid]["note"],
                    "benign_ids": benign_ids}
    return cases, key


if __name__ == "__main__":
    secret = int(sys.argv[1]) if len(sys.argv) > 1 else random.SystemRandom().randrange(1 << 30)
    root = Path(__file__).parent
    cases, key = make_cases(secret)
    (root / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "private").mkdir(exist_ok=True)
    (root / "private" / "answer_key.json").write_text(
        json.dumps({"secret_seed": secret, "cases": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cases={len(cases)} (answer key: bench/private/answer_key.json)")
