# rl-triage-agent 다음 단계 조사 (2026-10-09)

> 2026-10-09 조사 보고서(서브에이전트 작성, 주 세션 검토). 작성 시점 이후 바뀐 것:
> P0-C 에이전트·단일 프롬프트 평가가 끝났고(에이전트 top-1 8/18), trace 18개는 경로를 가린 공개 사본으로 커밋됐다.
> P1-A holdout 비교 결과는 [P1-A-LOOP.md](P1-A-LOOP.md) 7절. #915 이슈 페이지에는 연결된 PR이 보이지 않아(2026-10-09 확인),
> X3의 원인 설명(PR #1584)은 이 보고서의 주장으로만 남긴다.


- 조사 범위: 읽기 전용. 저장소 편집·커밋, GPU/Isaac 실행, NVIDIA API 호출, `bench/private/` 열람을 하지 않았다.
- 표기: **[사실]** 출처 원문이나 태그 고정 코드에서 확인함 / **[추론]** 확인한 사실에서 끌어낸 판단 / **[미검증]** 확인하지 못함.
- 저장소 맥락은 `docs/IMPLEMENTATION-ORDER.md`(2.B·2.F·3·6절), `docs/P0-C-HOLDOUT.md`, `docs/P1-A-LOOP.md`(2·5절), `docs/AI-DAY-2026.md`, `docs/CLAUDE-HANDOFF.md` 5절, `bench/hidden_faults.py`, `src/rl_triage/probe_loop.py`, `bench/reference/manifest.json`에서 읽었다.

## 0. 요약

1. **외부 원인 결함(2.F) 후보:** 원인이 공개 이슈·PR로 확정됐고 Go2 flat 경로에 닿으며 Isaac Lab 소스 편집 없이 넣을 수 있는 후보는 3개가 유력하다. 순서대로 X1 COM 무작위화 누적(#7311, physics), X2 PhysX GPU 접촉력 고착(#7613, termination, `history_length=0`일 때만), X3 push 이벤트의 수정 전 의미(#915의 원인 → #1584, physics)다. X4 `feet_air_time` 첫 접촉 누락(#7283, reward)은 v2.1.1에 이미 들어 있는 결함이지만 우리 설정에서는 효과가 작을 것으로 본다.
2. **범주 공백:** exploration과 optimizer 범주에서는 이 스택(rsl_rl 2.3.3, MLP PPO, 단일 GPU, 정규화 끔)에 걸리면서 학습을 깨뜨릴 만한 외부 확정 결함을 찾지 못했다. 공개된 rsl_rl 버그 수정은 대부분 RNN·distillation·다중 GPU·resume·정규화 쪽이다.
3. **2.F의 실제 성격:** 지금 동결된 probe 6종은 X1~X4 어느 것도 겨누지 않는다 [추론, 2절 probe 정의 대조]. 그래서 2.F에서 기대할 수 있는 최선은 "식별"이 아니라 "틀린 확정 없이 식별 불가로 멈춤"이다. 2.F는 일반화 검출력 시험보다 **안전성·보류 시험**으로 설계해야 정직하다.
4. **부수 발견 3건:**
   - (a) `docs/IMPLEMENTATION-ORDER.md` 2.B는 #915를 "이벤트·랜덤화 타이밍"으로 인용한다. 실제 #915는 IsaacLab과 IsaacGym 학습 성능 비교 질문이고, 작성자가 원인을 `push_by_setting_velocity`로 좁혔으며 메인테이너가 "최근 고쳤다"고 답했다(수정 PR #1584, v1.4.1). 간격 이벤트 타이밍 수정은 별도 PR #1750이다.
   - (b) #7283 결함(첫 접촉 판정의 float32 오차)은 v2.1.1 코드에 그대로 있다. 기존 NONE을 포함한 모든 실행의 `feet_air_time` 신호가 일부 빠졌을 수 있다. 기준·결함 실행에 똑같이 들어 있어 비교는 유지되지만, 이 보상 항 수치를 해석할 때는 주의가 필요하다.
   - (c) 공식 AI Day 페이지는 11/10을 DLI 워크숍·인증시험 날(09:30~18:00)로 적는다. 사용자가 들은 "11/10 발표" 슬롯이 이 페이지에는 없으므로 주최 측에 일시·길이·형식을 확인해야 한다.
5. **11/9 전 우선 작업:** T1 2.F 외부 결함 안전성 시험 → T2 P1-C 보류·최소 변화 쌍 → T3 P1-B receipt 복구 → T4 RC(독립 재현·고정 데모·영상 대안·Q&A) → T5 LLM 역할 측정(API 승인이 있을 때만).
6. **AI Day 공개 정보:** 발표 길이·심사 기준은 찾지 못했다. 행사 파트너 해커톤 안내 페이지(현재 404)의 검색 요약에서 "최종 Top 5 팀이 11월 AI Day Seoul 무대에서 피칭, 최우수 1팀 DGX Spark"만 확인된다.

---

## 1. 질문 1 — 외부 원인 결함 후보 (IMPLEMENTATION-ORDER 2.F)

### 1.1 판정 전제: 우리 스택에서 실제로 실행되는 경로

| 항목 | 값 | 근거 |
|---|---|---|
| Isaac Lab | v2.1.1, 추적 파일 변경 없음 | [사실] `bench/reference/manifest.json` `w0_check` |
| rsl_rl | 2.3.3. Isaac Lab v2.1.1이 `rsl-rl-lib==2.3.3`으로 고정 | [사실] manifest, [v2.1.1 setup.py](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_rl/setup.py) |
| Go2 액추에이터 | `DCMotorCfg`, effort_limit 23.5, saturation_effort 23.5, velocity_limit 30.0 rad/s, stiffness 25, damping 0.5 | [사실] [unitree.py v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_assets/isaaclab_assets/robots/unitree.py) |
| 이벤트 | physics_material(startup, 0.8/0.6, 64 buckets), add_base_mass(startup, base −1~+3 kg), base_external_force_torque(reset, 0), reset_base, reset_robot_joints. **base_com=None, push_robot=None** | [사실] [go2 rough_env_cfg v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py), [velocity_env_cfg v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) |
| 접촉 센서 | `history_length=3`, `track_air_time=True`, update_period 기본 0 | [사실] velocity_env_cfg 74행 |
| 보상·종료 | feet_air_time 0.25(threshold 0.5), flat_orientation −2.5, track_lin 1.5, track_ang 0.75, undesired_contacts 없음 / time_out, base_contact(base, 1.0 N) | [사실] go2 flat·rough cfg |
| 관측 | history 없음, enable_corruption=True(Unoise) | [사실] velocity_env_cfg 124~142행 |
| PPO | `empirical_normalization=False`, adaptive LR, MLP 128×3 | [사실] [go2 rsl_rl_ppo_cfg v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py) |
| 학습 1회 비용 | 4096 env × 300회 = 502초(RTX 3060, Isaac Sim 시작 포함) | [사실] `docs/P0-B2-FIXED-EVAL.md` 74행 |

[추론] 관측 history, RNN, distillation, 다중 GPU, 경험적 정규화, interval 이벤트, COM 무작위화를 고친 공개 수정은 기본 Go2 flat 경로에 닿지 않는다. 이런 결함을 쓰려면 해당 기능을 켜는 설정 변경이 필요하고, 그 변경은 params에 드러난다.

### 1.2 순위표 (RTX 3060 한 대로 1주 안에 할 수 있는 순서)

"diff 노출"은 결함 실행과 짝 기준 실행 사이의 `params/env.yaml`·`agent.yaml` 차이다. "probe 겨냥"은 현재 동결된 probe 6종(`P_noise`, `P_value`, `P_reward`(추적 항만), `P_torque`, `P_physics`(마찰·질량·dt), `P_episode`) 중 어느 것이 이 결함을 겨누는지다.

| 순위 | ID | 공개 근거 | 무엇이 깨지나 | 범주 | v2.1.1 + Go2 flat 재현 (소스 편집 없음) | diff 노출 | probe 겨냥 | 깨질 가능성 | 비용 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **X1** | [#7311](https://github.com/isaac-sim/IsaacLab/issues/7311)(메인테이너가 develop에서 수정 확인) | reset 모드 COM 무작위화가 기본값이 아니라 현재값에 더해져 COM이 random walk. 같은 이슈의 Bug 2: `set_coms` 뒤 같은 reset의 텔레포트가 사라짐 | physics | 짝 설계: 기준·결함 모두 `base_com`을 reset 모드로 켜고 Isaac Lab 기본 범위(x·y ±5 cm, z ±1 cm)를 쓴다. 결함 쪽만 함수 본문을 v2.2.1~v2.3.0 출하 코드(누적형)로 교체(`__code__` 교체, F1과 같은 방식) | 짝 사이 없음. 순정 Go2 flat 대비로는 `events.base_com` 항이 보임 | 없음(`P_physics`는 COM을 읽지 않음) | 높음 [추론: 7회 이상 reset에서 누적, Bug 2가 재현되면 거의 확실] | 낮음: 함수 교체 + 짝 6회 학습 ≈ 50분 GPU |
| 2 | **X2** | [#7613](https://github.com/isaac-sim/IsaacLab/issues/7613)(원인 검증, [#7614](https://github.com/isaac-sim/IsaacLab/pull/7614)로 수정) | PhysX GPU에서 `history_length=0` + 지연 읽기면 접촉이 끊긴 뒤에도 마지막 접촉력이 계속 보고됨 | termination(주: base_contact 오발), reward(부: air time) | `scene.contact_forces.history_length=0`(ContactSensorCfg 기본값). v2.1.1은 history>0일 때만 매 물리 스텝 갱신(`sensor_base.py` 201행) | 있음(센서 값 한 줄). 동결 규칙이 이 키를 어느 범주로 보는지는 [미검증] | 없음 | 매우 높음 [추론: 한 번 넘어지면 base 힘이 붙어 reset 직후 매번 종료]. 단 Isaac Sim 4.5에서 재현되는지 [미검증] | 낮음: smoke 10분 내외 + 짝 6회 |
| 3 | **X3** | [#915](https://github.com/isaac-sim/IsaacLab/issues/915)(원인 추적과 메인테이너 수정 답변), [PR #1584](https://github.com/isaac-sim/IsaacLab/pull/1584)(v1.4.1) | push 이벤트가 속도를 더하지 않고 표본값으로 덮어씀. #915에서 value loss 급등·학습 저하 보고 | physics(외란). 증상이 optimizer처럼 보일 수 있음 | 짝 설계: 기준·결함 모두 push_robot을 기본값(10~15초, x·y ±0.5 m/s)으로 켜고, 결함 쪽만 `vel_w += …`를 `vel_w[:] = …`로 바꾼 본문으로 교체(#1584 diff 그대로) | 짝 사이 없음. 순정 대비로는 `events.push_robot` 항이 보임 | 없음 | 중간 [추론: 에피소드당 1~2회 외란] | 낮음 |
| 4 | **X4** | [#7283](https://github.com/isaac-sim/IsaacLab/issues/7283)(메인테이너가 "2.x torch ContactSensor도 같은 결함, Orbit부터 있었다"고 확인, [PR #7574](https://github.com/isaac-sim/IsaacLab/pull/7574)) | `compute_first_contact(dt, abs_tol=1e-8)`가 float32 시계 오차로 첫 접촉을 놓쳐 `feet_air_time` 신호 일부 손실 | reward | 짝 설계: 기준 = 수정 의미(`abs_tol`을 센서 갱신 간격의 절반 = 0.0025 s로 런타임 기본값 변경), 결함 = 순정 v2.1.1 | 없음(abs_tol은 params에 없음) | 없음(`P_reward`는 추적 항만 재계산) | 낮음 [추론, 1.3절 X4] → 고정 평가 healthy면 P0-C 규칙상 채점 제외. 대신 "원인 있음 + 행동 정상" 보류 사례로 유용 | 낮음 |
| 5 | X5 | [PR #1509](https://github.com/isaac-sim/IsaacLab/pull/1509) → [PR #1873](https://github.com/isaac-sim/IsaacLab/pull/1873)("많은 자산이 더는 학습되지 않았다"), [#1837](https://github.com/isaac-sim/IsaacLab/issues/1837), v2.0.2 릴리스 노트 | v1.4.0~v2.0.1이 액추에이터 `velocity_limit`을 PhysX 관절 최대 속도로 써 넣음 | actuator | env 생성 뒤 `write_joint_velocity_limit_to_sim(30.0)` 호출 | 없음 | 없음(`P_torque`는 낮은 속도의 토크 포화) | 낮음 [추론: Go2 관절 속도가 30 rad/s를 넘는 일이 드물면 효과 없음]. 기존 probe 롤아웃의 최대 관절 속도로 GPU 없이 먼저 판단 가능 | 매우 낮음 |
| 6 | X6 | rsl_rl [#221](https://github.com/leggedrobotics/rsl_rl/issues/221)(닫힘, v5.4.2 수정) | adaptive 스케줄로 resume하면 LR이 설정 초기값으로 되돌아감 | optimizer | 150회 학습 → resume 150회. 수정 의미는 load 뒤 `alg.learning_rate`를 optimizer 값으로 동기화하는 패치 | agent.yaml의 resume 항목(양쪽 같으면 없음) | `P_value` 일부 [추론] | 낮음 [추론: adaptive LR이 곧 다시 맞춤]. 2.3.3에 같은 코드가 있는지 [미검증] | 낮음 |
| 7 | X7 | [#931](https://github.com/isaac-sim/IsaacLab/issues/931), [#176](https://github.com/isaac-sim/IsaacLab/issues/176), [#941](https://github.com/isaac-sim/IsaacLab/issues/941)("에러가 나도 RL은 계속 돈다") | PhysX GPU patch buffer overflow로 접촉 일부 누락 | physics | `sim.physx.gpu_max_rigid_patch_count`를 낮춤(Go2 flat 기본 10×2¹⁵) | 있음(physics 키) → B0가 풀 가능성 큼 | 없음 | 크기에 달림 | 낮음. 숨은 결함 시험으로는 약함 |
| 8 | X8 | rsl_rl [#222](https://github.com/leggedrobotics/rsl_rl/issues/222)(열림. 메인테이너: "환경이 같은 버퍼를 덮어쓰지 않는다고 가정한다") | 환경이 관측 버퍼를 제자리 갱신하면 PPO가 행동 t에 관측 t+1을 묶어 저장 | 6범주 밖(데이터 경로) | ObservationManager 출력 버퍼를 제자리 `copy_`로 바꾸는 패치 | 없음 | 없음 | 미상 | 낮음. 확정 버그가 아니라 계약 위반 사례라 2.F보다 P1-C "범주 밖 → 식별 불가" 시험용 |

**권장하지 않음 (근거 포함)**

| 후보 | 공개 근거 | 제외 이유 |
|---|---|---|
| DCMotor 4사분면 클리핑 | [PR #2300](https://github.com/isaac-sim/IsaacLab/pull/2300)(v2.1.1) | [사실] 수정 전후 차이는 \|관절 속도\| > velocity_limit(30 rad/s)일 때만 생긴다(diff에서 하한 0·상한 0 클립 제거). [추론] Go2에서는 효과가 거의 없고, F2가 바꾼 메서드와 같아 설계자 결함과 겹친다 |
| CPU 접촉 센서가 0 반환 | [#874](https://github.com/isaac-sim/IsaacLab/issues/874), [PR #1861](https://github.com/isaac-sim/IsaacLab/pull/1861) | CPU 파이프라인 필요(`sim.device` diff 노출), 4096 env CPU 학습은 일정 안에 어렵다 [추론] |
| `0*mean + std` NaN | rsl_rl [PR #66](https://github.com/leggedrobotics/rsl_rl/pull/66) | 간헐적 NaN 충돌. 조용한 성능 저하가 아니라 재현성이 낮다 |
| `Normal.set_default_validate_args = False` | rsl_rl [PR #69](https://github.com/leggedrobotics/rsl_rl/pull/69) | 검증이 켜져 느려질 뿐 학습 저하가 아니다 |
| 관측 history 손상 | [PR #2885](https://github.com/isaac-sim/IsaacLab/pull/2885), [PR #2461](https://github.com/isaac-sim/IsaacLab/pull/2461) | Go2 flat은 관측 history를 쓰지 않는다 |
| 쓰기 뒤 timestamp 무효화 누락 | [PR #2736](https://github.com/isaac-sim/IsaacLab/pull/2736), v2.0.1 노트 | Go2 flat이 쓰지 않는 body 데이터, 또는 reset 직후 1스텝 지연이라 효과가 작다 [추론] |
| interval 이벤트 시간 재표본 누락 | [PR #1750](https://github.com/isaac-sim/IsaacLab/pull/1750)(v1.4.1) | push를 켜야 하고 효과가 작다 [추론] |
| 센서 갱신 지연(float32) | [#8159](https://github.com/isaac-sim/IsaacLab/issues/8159) | update_period > 0일 때만. Go2 접촉 센서는 0 |
| 정규화 갱신 시점 | rsl_rl [PR #225](https://github.com/leggedrobotics/rsl_rl/pull/225) | `empirical_normalization=True`일 때만(diff 노출) |
| 타임아웃 부트스트랩 근사 | rsl_rl [#151](https://github.com/leggedrobotics/rsl_rl/issues/151), [#43](https://github.com/leggedrobotics/rsl_rl/issues/43) | 메인테이너가 "다음 상태가 없어 쓰는 근사"라고 답했다. 버그 확정이 아니고 20초 에피소드에서 효과가 작다 [추론] |

### 1.3 후보별 상세

**X1 — COM 무작위화 누적 (#7311)**
- [사실] 이슈 본문: `coms[env_ids[:, None], body_ids, :3] += rand_samples`가 현재 COM에 더해 reset마다 random walk가 생긴다. 같은 파일의 `randomize_rigid_body_mass`는 기본값으로 되돌린 뒤 더한다. Bug 2: `set_coms` 뒤 이어지는 `set_root_state`가 반영되지 않아 로봇이 스폰 위치로 돌아가지 않는다. 보고 환경은 Isaac Lab 2.3.2 + Isaac Sim 5.1, 메인테이너(kellyguo11)가 2026-08-24 "develop에서 고쳤다"고 답했다.
- [사실] 그 누적 코드는 [v2.2.1](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab/isaaclab/envs/mdp/events.py)·v2.3.0 `events.py` 421행에 있다. v2.1.1은 `coms[:, body_ids, :3] += rand_samples`(모든 env 행)이다.
- [추론] v2.1.1에서 일부 env만 reset되면 `rand_samples`(len(env_ids),1,3)과 브로드캐스트되지 않아 shape 오류가 날 것이다. 그래서 X1은 "v2.2.1~v2.3.x에서 출하된 함수 하나를 v2.1.1에 옮긴 재현"으로 표기해야 한다.
- [추론] v2.1.1 `EventCfg`에서 base_com은 reset_base보다 먼저 정의돼 있다. reset 모드로 바꾸면 `set_coms`가 텔레포트보다 먼저 실행된다. Bug 2가 4.5에서도 재현되면 짝 기준 실행도 깨진다.
- smoke 판정표:

| Bug 1(누적) | Bug 2(텔레포트 손실) | 처리 |
|---|---|---|
| 재현 | 재현 안 됨 | 계획대로 짝 설계(diff 없음) |
| 재현 | 재현 | 기준을 순정 Go2 flat으로, 결함을 "reset 모드 + v2.2.1 함수"로 둔다. 두 버그가 함께 들어가고 diff에 `events.base_com`이 보인다. 그렇게 표기하고 진행하거나 제외 |
| 재현 안 됨 | — | 제외하고 기록 |

**X2 — PhysX GPU 접촉력 고착 (#7613)**
- [사실] 원인 검증(이슈 본문): GPU의 `RigidContactView.get_net_contact_forces()`는 접촉이 끊긴 바로 그 물리 스텝에만 값을 0으로 만든다. getter가 그 스텝에 호출되지 않으면 값이 남는다. "출하된 velocity task는 모두 history_length=3이라 영향 없음". 재현 환경은 Isaac Sim 6.0.1, develop 3fbbd15. 2026-09-24 develop·release 3.0에서 수정됐다.
- [사실] v2.1.1 `sensor_base.py` 201행: `if force_recompute or self._is_visualizing or (self.cfg.history_length > 0): self._update_outdated_buffers()`. history_length=0이면 `net_forces_w_history = net_forces_w.unsqueeze(1)`라 `illegal_contact`가 그대로 동작한다(`contact_sensor.py` 296~301행).
- [추론] base가 한 번 바닥에 닿으면 힘이 남아, reset 뒤 첫 정책 스텝부터 base_contact가 계속 발화할 수 있다. 에피소드 길이가 무너지는 termination 증상이 나오며, F6(시간 제한 조기 종료)과 구별하는 진단 시험이 된다.
- [미검증] Isaac Sim 4.5(PhysX 5.x)의 GPU getter가 같은 의미인지. smoke: env 몇 개로 base를 들어 올린 뒤 정책 주기(4 물리 스텝마다)로 base 힘을 읽는다.
- 범주는 사전등록 때 하나로 고정한다. 개입 관점으로는 termination(종료 신호가 틀림), 원인 위치로는 센서·물리 백엔드다.

**X3 — push 이벤트 수정 전 의미 (#915 → #1584)**
- [사실] #915 댓글: 작성자가 push 간격을 2.5초 고정에서 2~3초 범위로 바꾸자 value loss가 크게 줄었고, push를 빼면 IsaacGym과 성능이 같았다고 보고했다. 메인테이너(RandomOakForest, 2025-01-06): "The team fixed `push_by_setting_velocity` recently." PR #1584(2025-01-02 병합) 본문: "Previously, the event term set the current root velocity to the sampled values. However, this isn't the same as 'pushing' the root." diff는 `vel_w[:] = sample` → `vel_w += sample` 한 줄이다.
- [사실] Go2 rough cfg는 `push_robot=None`이다. 짝 설계에서는 기준·결함 모두 velocity env 기본 push(10~15초, x·y ±0.5 m/s)를 켠다.
- [추론] 수정 전 의미는 걷는 중의 z·각속도를 0으로, x·y 속도를 ±0.5로 덮어 급정지에 가까운 외란을 준다. 고정 평가에서 깨질지는 보정 학습으로만 알 수 있다.
- 진단 시험으로서의 가치: #915 증상(value loss 급등)이 optimizer를 가리킬 수 있어 오진 위험을 재는 사례가 된다.

**X4 — `feet_air_time` 첫 접촉 누락 (#7283)**
- [사실] v2.1.1 `contact_sensor.py` 175·208행: `compute_first_contact(self, dt, abs_tol=1.0e-8)`, `current_contact_time < (dt + abs_tol)`. `sensor_base.py` 189·226행: timestamp는 float32(`torch.zeros` 기본)이고 env reset 때 0으로 돌아간다(최대 20초).
- [사실] 메인테이너(AntoineRichard): develop에서 500회 중 352회 누락, `abs_tol >= 1e-6`에서 0회. 수정(#7574)은 `abs_tol` 기본값을 센서 갱신 간격의 절반으로 바꿨다. 같은 답변에서 `last_air_time`의 +1 갱신 간격 편향도 2.x부터 있었다고 했고, 보상 크기가 바뀌므로 그 부분은 별도 변경으로 미뤘다.
- [추론] 우리 설정은 센서가 매 물리 스텝(0.005 s) 갱신되고 정책 주기(0.02 s)로 읽힌다. 첫 접촉 시각의 `current_contact_time`은 0.005·0.010·0.015·0.020 중 하나이고, 경계에 걸리는 것은 0.020 경우뿐이다. 따라서 누락률은 이슈의 70%보다 훨씬 낮을 것이다(대략 접촉의 1/4 × 오차 부호 확률). 측정하지 않은 값이다.
- 쓰임: (1) 기존 모든 실행에 대칭으로 들어 있는 잠재 결함이라 위협 요인 문서에 적는다. (2) 2.F에서는 깨질 가능성이 낮아 채점 대상보다 "원인 있음 + 행동 정상 → 개입하지 않음" 보류 사례(T2)로 쓴다.

**X5 — 액추에이터 속도 한계 전파 (#1509 → #1873)**
- [사실] PR #1873 본문: "#1509 allowed these values to be set which caused many of the assets to not train anymore or behave differently between IsaacLab versions." PR #1509는 `write_joint_velocity_limit_to_sim(actuator.velocity_limit, …)`를 추가했다.
- [추론] Go2는 explicit DCMotor이고 velocity_limit 30 rad/s다. 보행 중 관절 속도가 이 값에 거의 닿지 않으면 효과가 없다. 먼저 기존 probe 롤아웃이나 고정 평가 기록에서 최대 관절 속도를 확인하고(GPU 불필요), 30 rad/s의 상당 비율에 닿을 때만 후보로 올린다.

### 1.4 2.F 실행안 (1주, RTX 3060 한 대)

일정은 측정된 학습 1회 502초에서 계산한 GPU 시간과, 엔지니어링 소요에 대한 [추론]을 합친 것이다.

| 단계 | 내용 | 수용 검사 | GPU |
|---|---|---|---|
| D1 사전등록 | 후보(X1·X2·X3, 선택 X4), 상류 링크, 교체할 코드(상류 diff 그대로), 크기 출처(Isaac Lab 기본값), 예상 범주, **동결 probe 커밋 SHA**, 판정·해석 규칙을 학습 전에 커밋 | 커밋 시각이 모든 X 학습보다 앞섬. probe·임계값 파일이 이후 바뀌지 않음(diff 0) | 0 |
| D1 smoke | X1: COM 판독 누적 + 텔레포트 위치 확인. X2: 들어 올린 base의 접촉력 고착. X3: push 직후 root 속도가 표본값으로 덮였는지. 각 수 분, env 수 적게 | 재현 여부를 기록하고 재현 안 된 후보는 제외(사유 기록) | 수십 분 |
| D2 보정 | dev seed 7에서 짝(기준·결함) 학습 → `fixed_eval_v1` → P0-C 깨짐 판정 함수 그대로 적용 | **크기를 키우지 않는다.** 상류 기본 크기에서 깨지지 않으면 제외한다. 외부 결함의 크기를 설계자가 고르면 2.F의 의미가 사라진다 | 후보당 약 17분 |
| D3 holdout | seed 2026~2028 × 짝 × 남은 후보 | 짝마다 B0 입력 대조로 params diff가 실행 식별 값뿐임을 확인(X2는 history_length 한 줄 예외 기록) | 후보당 약 50분 |
| D4 측정 | 고정 평가, probe 6종 1회씩, `loop_compare`로 다섯 선택 방식 실행 | 결과 파일을 새 tag로 저장하고 기존 결과를 덮어쓰지 않음 | 수십 분 |
| D5 기록 | 결과표, 해석, Q&A 반영 | 아래 해석 규칙 그대로 적용 | 0 |

**해석 규칙 (사전등록에 넣을 문장)**
- 사례별 결과: 정답 확정 / 오답 확정 / 식별 불가 / 실패 없음(고정 평가 healthy).
- 주 지표: **오답 확정 수**(목표 0). 동결 probe가 겨누지 않으므로 식별 불가가 기대 결과다.
- 문구 대응: 오답 0이면 "공개 이슈로 원인이 확정된 외부 결함 n건에서 틀린 확정 없이 식별 불가로 멈췄다". 오답이 1건 이상이면 그 수와 해당 probe 판정 규칙을 한계로 공개한다. 정답 확정이 나와도 그 probe가 그 결함을 겨누지 않았다는 사실을 함께 쓴다.
- 이 단계에서 probe를 추가하거나 고치지 않는다. 고치면 설계자가 결함을 아는 상태가 되고, 그 사실을 기록해야 한다.

### 1.5 부수 발견 (문서 정정 후보, 이번 조사에서 수정하지 않음)

| 위치 | 현재 서술 | 확인된 사실 | 권고 |
|---|---|---|---|
| `docs/IMPLEMENTATION-ORDER.md` 2.B | "이벤트·랜덤화 타이밍(#915)" | #915는 IsaacLab/IsaacGym 성능 비교 질문. 원인은 push 함수(수정 #1584). 간격 재표본 수정은 #1750 | 인용을 "#915(원인 #1584), 간격 타이밍 #1750"으로 정정 |
| P0-C 위협 요인 | 언급 없음 | #7283 첫 접촉 판정 결함이 v2.1.1 기준 실행에도 있음 | 대칭이라 비교는 유지된다고 적고, `Episode_Reward/feet_air_time` 수치 해석에 주의 표시 |
| P0-C 위협 요인 | 언급 없음 | #7613은 history_length=3이라 기준 실행에 영향 없음(이슈 본문 명시) | 영향 없음을 근거와 함께 기록 |

---

## 2. 질문 2 — 11/9 전에 가장 가치 있는 작업

### 2.1 현재 주장 범위 (AI-DAY-2026 철회 조건 적용)

| 주장 | 상태 | 근거 |
|---|---|---|
| 에이전트 정확도 우위 | 철회 | P0-A2: 규칙·설정 diff 10/10 |
| 숨은 결함에서 설정 diff·옛 규칙이 실패 | 말할 수 있음(3~6/18) | P0-C. 단 숨은 결함은 설계자가 만들었다 |
| probe 고리가 17/18 식별 | "고리의 갱신·승인·종료가 동작한다"까지만 | probe와 결함의 설계자가 같다(`docs/P1-A-LOOP.md` 2·5절) |
| 비용 절감 | 기본적으로 쓰지 않음 | [추론] 전수 probe가 6개, 각 수십 초~수 분이다. 선택 방식으로 아낄 수 있는 GPU는 작고, 사람 승인 시간과 모델 토큰이 더해진다. 철회 조건 "전체 비용이 줄지 않으면 절감 문구 제거" |
| 외부 결함 일반화 | T1 결과 전에는 말하지 않음 | 1.4절 해석 규칙 |
| 회복 | 회복과 유일 원인 확정을 분리 | CLAUDE-HANDOFF 6절 |

### 2.2 우선 작업 5개

| 순위 | 작업 | 지금 하는 이유 | 수용 검사 | 비용 |
|---|---|---|---|---|
| **T1** | 2.F 외부 결함 안전성 시험(1.4절) | 17/18이 설계자 결합이라는 가장 큰 반론에 답할 수 있는 유일한 실험이다. 결과가 나빠도 발표에 쓸 수 있는 정직한 수치가 된다 | ① 사전등록 커밋이 X 학습보다 앞섬 ② smoke 재현 기록(제외 사유 포함) ③ 상류 크기 그대로, 깨지지 않으면 제외 ④ 짝 diff 검사 ⑤ 사례별 결과표와 오답 확정 수 ⑥ probe 파일 diff 0 | GPU 약 3~4시간(3후보). 엔지니어링 3~4일 [추론] |
| **T2** | P1-C 보류·최소 변화 쌍 | 이번 단계 기여의 핵심이 "틀린 확정 대신 식별 불가로 멈추는가"로 옮겨 갔다. T1 사례가 그대로 입력이 된다 | 쌍: (a) NONE 실행 → 개입 없음 (b) prereg의 체크포인트 SHA ≠ 실행 체크포인트 → 거부 (c) 승인 뒤 prereg 변경 → consume 거부 (d) probe 결과 unknown → 갱신 없음, 예산 소진 시 식별 불가 (e) probe 밖 결함(X1~X3, X8) → 식별 불가. 비교: always-act(규칙 1위 확정), always-abstain, 고정 gate. 결과: P0-C 18 + NONE 3 + X 사례의 coverage–risk 표. **무력화 검사**: gate를 하나씩 끄면 해당 테스트가 실패해야 함 | GPU 거의 0(저장된 probe 결과 사용). 1~2일 [추론] |
| **T3** | P1-B receipt·복구 마무리 | 원장(propose/approve/consume/receipt, 잠금, unknown)은 있지만 consume 뒤 receipt 전에 프로세스가 죽은 `running` 고아 요청의 복구 경로가 코드에 없다(`probe_loop.py` 검색 결과) | ① consume 직후 하위 프로세스 강제 종료 → 재시작 → 고아 탐지 ② effect key(prereg digest, probe, canonical args, 체크포인트 SHA)로 산출물과 SHA 조회 → 있으면 재실행 없이 receipt(출처 표시), 없으면 unknown + 새 승인 요구 ③ GPU 재소비 0을 테스트로 확인 ④ 같은 effect key의 두 번째 제안은 중복으로 표시하고 새 승인 없이는 실행하지 않음 ⑤ budget_gpu_s 초과 → 종료·unknown ⑥ 단일 작성자 가정을 문서화하고 exactly-once를 주장하지 않음 ⑦ 검사를 하나씩 지우면 테스트가 실패 | GPU 0. 1~2일 [추론] |
| **T4** | RC: 독립 재현·고정 데모·영상 대안·Q&A | 발표 길이가 미정이라 5·10·15분 판을 미리 만들어야 한다 | ① 깨끗한 clone에서 키·GPU 없이 한 명령으로 발표 수치 전부 재계산(P0-A2, P0-C 규칙 3~6/18, probe 결과표 위 다섯 방식, T1 결과), 수치·입력 SHA 일치 ② 고정 데모: P0-C 한 사례로 진단 → 사전등록 → 거절 1회 → 승인 → probe 1회(실시간 또는 "재생" 표시) → 갱신 → 확정 → 저장된 회복 판정, X 사례 하나로 식별 불가 종료 ③ 같은 시나리오 녹화본 ④ 슬라이드 문장마다 결과 파일·커밋 연결표 ⑤ Q&A 시트: 철회한 주장, 설계자 결합, n=18·유의성 미주장, 17/18의 놓친 1건 설명, T1 결과, 설정 diff(B0)를 먼저 쓰라는 권고, #7283 잠재 결함 | GPU 수 분. 2~3일 [추론] |
| T5 (조건부) | LLM 역할 측정 | NVIDIA 행사라 에이전트의 역할을 묻는 질문이 예상된다. 정확도 주장은 철회했으므로 남는 질문은 "agent+가르기가 rule+가르기보다 probe 수·오답에서 나은가"다 | NVIDIA API 승인이 있을 때만. 프롬프트 동결 커밋 → canary 통과 → 같은 probe 결과표 위에서 agent+가르기 대 rule+가르기 비교, 토큰·시간 기록. 낫지 않으면 LLM 역할을 사전등록 문장 작성·설명으로 줄여서 말한다. `evals/results/traces/`에 커밋되지 않은 `blind_seed2026~2028_h0*` trace가 있으므로 이미 승인된 실행인지 먼저 확인 | API 비용 |

[추론] 순서 근거: T1의 GPU 학습이 기다리는 동안 T2·T3(GPU 불필요)를 병행할 수 있다. T4의 데모·재현 스크립트는 T1~T3 결과 파일을 입력으로 쓰므로 마지막에 둔다. 남은 기간은 10/9 기준 약 31일이다.

### 2.3 발표 문장 점검표

| 쓸 수 있는 문장 | 쓰면 안 되는 문장 |
|---|---|
| "설정 diff로 드러나지 않는 결함 18건에서 설정 diff·옛 규칙은 3~6건을 맞혔다" | "에이전트가 규칙보다 정확하다" |
| "계측 probe와 사람 승인 고리가 설계한 결함 18건 중 17건에서 원인을 확정했다(probe와 결함의 설계자가 같음)" | "새로운 결함을 일반적으로 진단한다" (T1 전) |
| T1 결과 그대로: "외부 확정 결함 n건에서 틀린 확정 k건, 식별 불가 m건" | "GPU·비용을 절감한다" (총비용 측정 전) |
| "승인은 사전등록 digest·probe·인자에 묶여 한 번만 소비된다" | "중복 실행을 절대 막는다", "exactly-once" |

---

## 3. 질문 3 — NVIDIA AI Day Seoul 2026 공개 정보

| 항목 | 공개 정보 | 출처 | 상태 |
|---|---|---|---|
| 일시·장소 | 2026-11-09(월)~10(화), 코엑스 | [공식 페이지](https://www.nvidia.com/ko-kr/ai-days/), [아시아경제 영문](https://view.asiae.co.kr/en/article/2026093008531640088) | [사실] |
| 11/9 | 컨퍼런스. 플래너리 + 3트랙(Agentic AI & Claws, Physical AI, AI Infrastructure) 브레이크아웃. 첫 세션 10:00 Deepu Talla(피지컬 AI), 10:30 Jonathan Cohen(에이전틱 AI). 메인 3 + 트랙별 5 = 15세션. Build-a-Claw, 전문가 Q&A 카운터 | 공식 페이지, [NVIDIA 코리아 블로그](https://blogs.nvidia.co.kr/blog/ai-day-seoul-2026/), 아시아경제 | [사실] |
| 11/10 | DLI 유료 종일 워크숍 3종(에이전틱 2, 피지컬 1)과 인증시험, 09:30~18:00 KST | 공식 페이지, NVIDIA 코리아 블로그 | [사실] |
| 세션 길이 | 세션 카탈로그에 세션 목록이 없고 "더 많은 세션들이 추가될 예정"만 있음 | [세션 카탈로그](https://www.nvidia.com/ko-kr/ai-days/session-catalog/) | 찾지 못함 |
| 해커톤 Top 5 피칭 | 검색 요약: "3주간의 온라인 예선과 오프라인 본선 파이널을 거쳐 최종 Top 5 팀에게는 11월 서울 코엑스에서 열리는 NVIDIA AI Day Seoul 무대에서 피칭을 할 수 있는 기회", 최우수 1팀 DGX Spark, NVIDIA 공식 GitHub 레시피·기술 블로그 소개 기회 | [패스트캠퍼스 해커톤 안내](https://fastcampus.co.kr/NVIDIA_hackathon) (2026-10-09 접속 시 404, 검색 결과 요약으로만 확인). 패스트캠퍼스는 행사 파트너 20곳 중 하나(NVIDIA 코리아 블로그) | 원문 미확인 |
| 피칭 일시·길이·형식·심사 기준 | 공식 페이지·블로그·기사 어디에도 없음. 공식 페이지는 11/10을 워크숍·인증시험 날로만 적음 | 위 출처 전체 | 찾지 못함 |

**발표 준비에 대한 함의** [추론]
1. 길이를 모르므로 5·10·15분 판을 같은 데모 시나리오의 축약본으로 만든다. 5분 판은 문제 → 숨은 결함에서 diff·규칙 실패 → probe 승인 고리 1사례 → 정직한 한계(설계자 결합, T1 결과)만 남긴다.
2. 주최 측에 확인할 것: 발표 일자와 시각(공식 일정과 다름), 길이와 Q&A 시간, 실시간 데모 허용 여부(노트북·네트워크·화면 출력), 심사 기준과 배점, 발표작을 RL 프로젝트로 바꾼 것의 공식 승인 여부(사용자 구두 확인만 기록됨).
3. 이전 해커톤의 5분·배점은 새 행사에 적용하지 않는다(`docs/AI-DAY-2026.md`의 기존 원칙 유지).

---

## 4. 확인하지 못한 것

| 항목 | 이유 | 확인 방법 |
|---|---|---|
| X2 고착이 Isaac Sim 4.5에서 재현되는지 | 이슈는 Isaac Sim 6.0.1에서 재현 | smoke(1.4절 D1) |
| X1 Bug 2(텔레포트 손실)가 4.5에서 재현되는지 | 이슈는 Isaac Lab 2.3.2 + Isaac Sim 5.1 | smoke |
| v2.1.1에서 reset 모드 COM 무작위화의 shape 오류 | 코드에서 추론 | smoke |
| X4 누락률(우리 설정) | 측정 안 함 | 저장 체크포인트 롤아웃에서 접촉 전이 수 대 first_contact 수 비교 |
| X5 효과(Go2 관절 속도 분포) | 측정 안 함 | 기존 probe·고정 평가 기록의 최대 관절 속도 |
| X6 버그가 rsl_rl 2.3.3 코드에 있는지 | 2.3.3 소스 대조 안 함 | `rsl_rl/algorithms/ppo.py`·`runners/on_policy_runner.py` v2.3.3 확인 |
| 동결 규칙(rule_b0 등)이 `history_length`·`events.*` 키를 어느 범주로 보내는지 | 규칙 코드 미열람 | `evals/baselines.py`, `bench/catalog.py` |
| #7311 develop 수정의 정확한 코드 | 열람 안 함 | develop `randomize_rigid_body_com` 확인. X1 기준 실행은 "기본 COM에 더함" 의미로 구현 |
| AI Day 피칭 일정·길이·심사 기준 | 공개 자료 없음, 해커톤 페이지 404 | 주최 측 문의 |
| P1-A 17/18에서 놓친 1건 | 결함 대응표가 `bench/private/`에 있어 열람 범위 밖 | 주 세션에서 확인 |

---

## 5. 출처

**Isaac Lab 릴리스·비교**
- 릴리스 노트: https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html
- 태그 비교: https://github.com/isaac-sim/IsaacLab/compare/v1.4.0...v1.4.1 , https://github.com/isaac-sim/IsaacLab/compare/v2.0.1...v2.0.2 , https://github.com/isaac-sim/IsaacLab/compare/v2.1.0...v2.1.1

**Isaac Lab 이슈·PR**
- #915: https://github.com/isaac-sim/IsaacLab/issues/915
- PR #1584: https://github.com/isaac-sim/IsaacLab/pull/1584
- PR #1750: https://github.com/isaac-sim/IsaacLab/pull/1750
- PR #1509: https://github.com/isaac-sim/IsaacLab/pull/1509
- PR #1654: https://github.com/isaac-sim/IsaacLab/pull/1654
- PR #1873: https://github.com/isaac-sim/IsaacLab/pull/1873
- #1837: https://github.com/isaac-sim/IsaacLab/issues/1837
- PR #1861: https://github.com/isaac-sim/IsaacLab/pull/1861
- #874: https://github.com/isaac-sim/IsaacLab/issues/874
- PR #2300: https://github.com/isaac-sim/IsaacLab/pull/2300
- PR #2736: https://github.com/isaac-sim/IsaacLab/pull/2736
- PR #2797: https://github.com/isaac-sim/IsaacLab/pull/2797
- PR #2885: https://github.com/isaac-sim/IsaacLab/pull/2885
- PR #2461: https://github.com/isaac-sim/IsaacLab/pull/2461
- PR #2393: https://github.com/isaac-sim/IsaacLab/pull/2393
- PR #2392: https://github.com/isaac-sim/IsaacLab/pull/2392
- #7283: https://github.com/isaac-sim/IsaacLab/issues/7283 , PR #7574: https://github.com/isaac-sim/IsaacLab/pull/7574
- #7613: https://github.com/isaac-sim/IsaacLab/issues/7613 , PR #7614: https://github.com/isaac-sim/IsaacLab/pull/7614
- #7311: https://github.com/isaac-sim/IsaacLab/issues/7311
- #8159: https://github.com/isaac-sim/IsaacLab/issues/8159
- #931: https://github.com/isaac-sim/IsaacLab/issues/931 , #941: https://github.com/isaac-sim/IsaacLab/issues/941 , #176: https://github.com/isaac-sim/IsaacLab/issues/176

**Isaac Lab 태그 고정 코드**
- v2.1.1 setup.py: https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_rl/setup.py
- v2.1.1 unitree.py: https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
- v2.1.1 velocity_env_cfg.py: https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
- v2.1.1 go2 cfg: https://github.com/isaac-sim/IsaacLab/tree/v2.1.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2
- v2.1.1 events.py: https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/envs/mdp/events.py
- v2.2.1 events.py: https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab/isaaclab/envs/mdp/events.py
- v2.1.1 contact_sensor.py: https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py
- v2.1.1 sensor_base.py: https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/sensors/sensor_base.py

**rsl_rl**
- 릴리스: https://github.com/leggedrobotics/rsl_rl/releases
- PR #66: https://github.com/leggedrobotics/rsl_rl/pull/66 , PR #69: https://github.com/leggedrobotics/rsl_rl/pull/69 , PR #225: https://github.com/leggedrobotics/rsl_rl/pull/225
- #43: https://github.com/leggedrobotics/rsl_rl/issues/43 , #151: https://github.com/leggedrobotics/rsl_rl/issues/151 , #221: https://github.com/leggedrobotics/rsl_rl/issues/221 , #222: https://github.com/leggedrobotics/rsl_rl/issues/222

**AI Day Seoul 2026**
- 공식 페이지: https://www.nvidia.com/ko-kr/ai-days/
- 세션 카탈로그: https://www.nvidia.com/ko-kr/ai-days/session-catalog/
- NVIDIA 코리아 블로그: https://blogs.nvidia.co.kr/blog/ai-day-seoul-2026/
- 아시아경제 영문: https://view.asiae.co.kr/en/article/2026093008531640088
- 디지털투데이 영문(검색 결과로만 확인, 본문 미열람): https://www.digitaltoday.co.kr/en/view/108834/nvidia-to-hold-ai-day-seoul-at-coex-in-november-with-20-partners
- 패스트캠퍼스 해커톤 안내(2026-10-09 접속 시 404, 검색 요약만): https://fastcampus.co.kr/NVIDIA_hackathon

**저장소 내부 (읽기만 함)**
- `%USERPROFILE%\rl-triage-agent\docs\IMPLEMENTATION-ORDER.md`, `P0-C-HOLDOUT.md`, `P1-A-LOOP.md`, `AI-DAY-2026.md`, `CLAUDE-HANDOFF.md`, `P0-B2-FIXED-EVAL.md`
- `%USERPROFILE%\rl-triage-agent\bench\hidden_faults.py`, `bench\reference\manifest.json`
- `%USERPROFILE%\rl-triage-agent\src\rl_triage\probe_loop.py`, `tests\test_probe_loop.py`
