"""벤치마크 v2: 그럴듯한 크기의 변경을 하나씩 단독으로 학습해 해로움을 측정으로 라벨링한다.

v1은 결함 값이 극단적이라 단일 프롬프트 대조군이 20/20을 맞혔다(설정값만 보고도 답이 보임).
v2는 (1) 변경 후보를 모두 2~10배 안쪽의 그럴듯한 크기로 두고 (2) 각 변경을 단독 적용한 학습을
seed 2개로 돌려 회복 판정(eval_bridge.RECOVERY_BAND)으로 harmful/benign을 정한다.
케이스는 측정된 harmful 1개 + 측정된 benign 2개를 섞되, benign은 상대 변화폭이 큰 것을 우선해
"값이 크게 바뀐 쪽이 범인"이라는 추측이 통하지 않게 만든다.

사용:
  python bench/catalog_v2.py singles        # 단독 실행 목록(JSON) 출력
  python bench/catalog_v2.py label          # 단독 실행 결과로 라벨링 → bench/v2/labels.json
  python bench/catalog_v2.py cases <seed>   # 라벨로 케이스 생성 → bench/v2/cases.json, 정답표는 private
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "bench" / "v2"

# (id, override, 기본값 대비 변화 설명). 해로움 여부는 적지 않는다 — 측정으로 정한다.
CANDIDATES = [
    ("S01", "env.actions.joint_pos.scale=0.5", "액션 스케일 0.25→0.5 (2x)"),
    ("S02", "env.actions.joint_pos.scale=0.15", "액션 스케일 0.25→0.15 (0.6x)"),
    ("S03", "agent.algorithm.gamma=0.95", "할인율 0.99→0.95"),
    ("S04", "agent.algorithm.learning_rate=0.003", "학습률 1e-3→3e-3 (3x)"),
    ("S05", "agent.algorithm.learning_rate=0.0003", "학습률 1e-3→3e-4 (0.3x)"),
    ("S06", "agent.policy.init_noise_std=0.3", "초기 탐색 노이즈 1.0→0.3"),
    ("S07", "env.rewards.dof_torques_l2.weight=-0.002", "토크 패널티 10x"),
    ("S08", "env.rewards.dof_acc_l2.weight=-2.5e-06", "관절 가속 패널티 10x"),
    ("S09", "env.rewards.action_rate_l2.weight=-0.05", "액션 변화율 패널티 5x"),
    ("S10", "env.rewards.feet_air_time.weight=1.0", "발 체공 보상 4x"),
    ("S11", "env.rewards.track_lin_vel_xy_exp.weight=0.5", "선속도 추종 보상 1.5→0.5"),
    ("S12", "env.rewards.flat_orientation_l2.weight=-10.0", "자세 패널티 4x"),
    ("S13", "env.episode_length_s=5.0", "에피소드 길이 20s→5s"),
    ("S14", "env.decimation=8", "제어 주기 50Hz→25Hz (decimation 4→8)"),
    ("S15", "env.events.add_base_mass.params.mass_distribution_params=[4.0,8.0]", "몸통 추가 질량 -1~3→4~8kg"),
    ("S16", "agent.num_steps_per_env=12", "롤아웃 길이 24→12"),
    ("S17", "agent.algorithm.entropy_coef=0.001", "엔트로피 계수 0.01→0.001"),
    ("S18", "env.commands.base_velocity.resampling_time_range=[3.0,3.0]", "명령 재샘플 10s→3s"),
    ("S19", "agent.save_interval=5", "체크포인트 저장 50→5"),
    ("S20", "agent.algorithm.num_learning_epochs=3", "학습 에폭 5→3"),
]


def singles():
    return [{"case_id": f"v2_{cid}", "overrides": [ov], "note": note} for cid, ov, note in CANDIDATES]


def label(seeds=(42, 7)):
    sys.path.insert(0, str(ROOT / "src"))
    from rl_triage.eval_bridge import recovery_verdict
    runs = ROOT / "bench" / "runs"
    out = {}
    for cid, ov, note in CANDIDATES:
        verdicts = []
        for s in seeds:
            p, ref = runs / f"v2_{cid}_s{s}.telemetry.json", runs / f"baseline_s{s}.telemetry.json"
            if not p.exists():
                verdicts.append(None)
                continue
            v = recovery_verdict(json.loads(p.read_text(encoding="utf-8")), json.loads(ref.read_text(encoding="utf-8")))
            verdicts.append(v["recovered"])
        if None in verdicts:
            lab = "missing"
        elif all(verdicts):
            lab = "benign"
        elif not any(verdicts):
            lab = "harmful"
        else:
            lab = "ambiguous"  # seed에 따라 갈리면 케이스에 쓰지 않는다
        out[cid] = {"override": ov, "note": note, "recovered_by_seed": dict(zip(map(str, seeds), verdicts)), "label": lab}
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def make_cases(secret_seed: int):
    labels = json.loads((V2 / "labels.json").read_text(encoding="utf-8"))
    harmful = [k for k, v in labels.items() if v["label"] == "harmful"]
    benign = [k for k, v in labels.items() if v["label"] == "benign"]
    rng = random.Random(secret_seed)
    rng.shuffle(harmful)
    cases, key = [], {}
    for i, h in enumerate(harmful, start=1):
        cid = f"d{i:02d}"
        bs = rng.sample(benign, 2)
        changes = [("H", h)] + [("B", b) for b in bs]
        rng.shuffle(changes)
        cases.append({"case_id": cid, "overrides": [labels[c[1]]["override"] for c in changes]})
        key[cid] = {"harmful_id": h, "harmful_override": labels[h]["override"], "note": labels[h]["note"], "benign_ids": bs}
    (V2 / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    priv = ROOT / "bench" / "private"
    priv.mkdir(exist_ok=True)
    (priv / "answer_key_v2.json").write_text(json.dumps({"secret_seed": secret_seed, "cases": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    return cases


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "singles":
        print(json.dumps(singles(), ensure_ascii=False))
    elif cmd == "label":
        for k, v in label().items():
            print(k, v["label"], v["recovered_by_seed"], v["note"])
    elif cmd == "cases":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.SystemRandom().randrange(1 << 30)
        print(f"cases={len(make_cases(seed))}")
