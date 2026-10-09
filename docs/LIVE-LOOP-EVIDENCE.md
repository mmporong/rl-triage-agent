# Isaac 승인 고리 실행 사전등록

기존 순위를 재생한 뒤 승인 원장부터 실제 Isaac probe, 갱신, 종료, 비용 감사까지 기록한다. 사전등록은 `bench/protocols/live_loop_v1.json`이다. **이 문서·코드·입력 해시를 커밋한 뒤 측정한다.** 기존 결과와 probe 수식·canonical args·임계값·결함 크기는 바꾸지 않는다.

## 사례와 절차

사례는 seed 2027의 `h01`~`h06`, `baseline_p0c`, `e01` 총 8건이다. 전체 결함 6건을 포함하므로 과거 전수가 확정하지 못한 `h04_s2027`도 포함된다. 사례 교체·성공 사례만 선별·인프라 실패 뒤 무조건 재시도는 하지 않는다. 종료 상태·오차·예산 초과를 모두 남긴다.

- 결함 사례: 이미 저장된 `mode=agent` 순위 상위 3개. NONE: 기존 범주 나열 순서 상위 3개. 새 모델 호출은 없다.
- 최대 probe 4개. 모든 사례의 첫 제안을 거절한 뒤 다음 제안부터 승인·소비·실행한다. 거절은 관측이나 probe 예산으로 세지 않는다.
- 승인 기록의 담당자는 `codex_operator_user_authorized_test`다. 사용자가 위임한 로컬 실행 시험이며 실제 사람의 능동 검토 시간은 0이다. 승인 신원 인증이나 사람 사용성 평가를 주장하지 않는다.
- `h04_s2027`의 첫 실행은 원시 출력·시간 sidecar를 쓴 뒤 receipt 전에 별도 worker 프로세스를 종료 코드 75로 끝낸다. 저장된 고아를 `recover`로 한 번 닫고 계속한다. probe 재실행은 없다.
- 모든 사례에서 소비 전에 감사 계약을 등록하고 종료 뒤 `finalize --abstain`에 해당하는 감사를 수행한다. 수리·재학습·새 행동 평가는 하지 않는다.

## 재현 판정

이전 일괄 probe와 같은 checkpoint SHA256·seed를 요구한다. 실시간 `classify` 결과와 기존 결과를 같은 기준 실행 측정값으로 비교하며 **판정 일치율 100%**를 목표로 한다. 수치 일치는 상대 오차 `1e-3`, 절대 오차 `1e-8`로 별도 판정한다. 수치 오차가 커도 임계값을 바꾸지 않는다.

수치 목표는 동결 `classify`가 읽는 필드다: `noise_ratio`, `critic_change`, 보상 재계산 오차 두 항, `low_speed_saturation`, `mismatch_count`, `timeout_ratio`. 그 밖의 원시 측정값도 양쪽을 저장하되 허용 오차 통과의 분모에는 넣지 않는다. 이전 일괄 실행은 모든 probe가 환경 256개를 공유했고 단일 실행은 기존 canonical args에 따라 64개 또는 256개를 쓴다. 환경 수를 비교 결과에 맞춰 늘리지 않는다. 따라서 질량 최솟값·최댓값 같은 부가 통계가 같다고 가정하지 않는다.

기술적 수용 조건은 승인→소비→receipt 기록 4건 이상, 감사 입력 오류 0건, 같은 checkpoint·seed, 판정 100% 일치, 중단 복구의 추가 실행 0건, 비용 8필드 누락 없음, NONE 틀린 확정 0건이다. 모든 8건을 보고하고 수치 허용치·판정·예산 조건은 각각 통과 여부를 표시한다. 원인 확정률·일반화·모델 우위·GPU 절감의 평가가 아니다.

## 비용과 근거

`loop.py run --metered`는 새 관측 wrapper를 통해 동결된 `probes.py`를 호출한다. wrapper는 원래 `measure` 반환값·예외·RNG·reset 순서를 보존하고, `measure` 안의 성공한 `env.step` 호출만 센다. 스텝은 실제 호출 횟수 × 실제 환경 수이며, 환경 생성·reset·물리 내부 substep은 이 필드에 포함하지 않는다. 읽기 전용 probe의 성공한 전이는 0이다. 해당 과정의 CPU·GPU 벽시계 비용은 포함된다.

- 원시 probe: `evals/results/<tag>/probes/<case>__<probe>.json`.
- Isaac·driver 자원 기록: 같은 tag의 `resources/`. wrapper 원본 계측·작업 목록·로그는 비공개 실행 폴더에 보존한다.
- 프로세스 실행 시간·CPU·스텝·출력 해시: 같은 tag의 `execution/`. 복구는 이 시간으로 원래 실행을 한 번만 계산한다. sidecar 없는 과거 파일은 기동 시간을 알 수 없어 비용을 null로 남긴다.
- worker 프로세스 CPU: 같은 tag의 `workers/`. 사례별 입력·계약·추가 비용·timeline·audit·결과는 `cases/<case>/`.

비용 벡터는 모델 호출·입출력 토큰·GPU 벽시계 초·CPU 초·사람 검토 초·개별 실행 벽시계 초·제어 전이 수다. 모델·사람 검토가 없는 이번 범위에서는 해당 값이 0이다. CPU는 운영자 Python, worker Python, 실행 driver Python, Isaac Python 프로세스의 계측 구간 합이다. batch launcher 셸과 외부 서비스 CPU를 계측한 값은 아니다. GPU 벽시계는 기동·load·reset·측정·close를 포함하는 실행 구간이며 장치 busy 시간이 아니다. 실행별 벽시계 합과 사례 전체 경과 시간을 따로 기록한다. offline finalize·파일 저장 이후 비용은 실행 범위 밖이다.

공학 한도는 사례당 GPU 벽시계 480초, CPU·개별 실행 합·전체 경과 각각 900초, 전이 1,228,800회다. 실행 전 예상 성능이나 소요 시간으로 읽지 않는다. 기존 probe별 승인 한도도 유지하고 초과 관측은 unknown으로 남긴다. 정상 반복 자료는 추가하지 않으며 기존 상대 여유값과 정상 변동 0을 사용한다. 절대 추종 상한은 고정 명령 격자의 제자리 오차 절반, 생존 하한 0.9, 낙상 상한 0.05로 사전등록한다.

감사기의 `approval_identity_attested`, `repair_execution_verified`, `probe_execution_checkpoint_verified`는 기존 의미대로 false다. 실행 harness의 checkpoint·출력 해시 대조는 별도 자료 연결이며 호스트 서명이 아니다. 수리를 하지 않았으므로 행동 회복을 주장하지 않는다.

## 실행

PowerShell에서 저장소로 이동하고, 커밋된 사전등록을 사용한다. GPU·다른 Isaac·Ollama 확인은 실행기가 수행하며 사용 중이면 측정을 시작하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
.venv/Scripts/python.exe evals/live_loop_evidence.py run `
  --prereg bench/protocols/live_loop_v1.json --tag live_loop_20261009 --case h01_s2027
```

이미 있는 고리·결과는 덮어쓰지 않는다. 관측 불일치·예산 초과는 원시 기록과 함께 보고한다.
