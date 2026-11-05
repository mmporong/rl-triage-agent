# Go2 계단 하행: 에이전트가 제안한 정책 적응

Nemotron 3.5가 안정성 검사를 선택하고 그 결과를 받은 뒤, 등록된 300회 추가 학습을 제안했다. 호스트가 이를 실행한 후 **같은 0.25m/s 명령과 같은 계단에서 30초 평가 기준을 통과했다.** 평가 seed는 200, 추가 학습 seed는 5710이다.

[15초 전후 비교 MP4](media/go2-stairs-agent-adaptation-s200-20261106-v3/comparison-15s-1080p.mp4) · [컬러 애니메이션](media/go2-stairs-agent-adaptation-s200-20261106-v3/comparison-15s-preview.webp) · [학습 전 1440p·30초](media/go2-stairs-agent-adaptation-s200-20261106-v3/before-native-1440p-30s.mp4) · [학습 후 1440p·30초](media/go2-stairs-agent-adaptation-s200-20261106-v3/after-native-1440p-30s.mp4) · [결과 JSON](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/summary.json).

## 에이전트가 한 일과 호스트가 한 일

1. 호스트가 실제 재생 trace로 `P_stairs_task_integrity`를 실행했다. 1501개 상태에서 명령 `[0.25, 0, 0]`이 유지되고 리셋은 없었다.
2. 모델이 `P_stairs_stability`를 선택했다. 호스트의 검사 결과는 `abnormal`: 12.26초부터 뒤집힌 자세가 나타났고 30초까지 888개 표본이 기록됐다.
3. 모델이 두 검사 결과와 자세 수치를 인용해 `fine_tune_stairs_300`을 제안했다. 실제 [검사 선택 응답](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/model/decision-01.json)과 [추가 학습 제안](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/model/decision-02.json)을 보존했다.
4. 등록된 호스트 승인이 제안과 검사 영수증을 결합했다. 공식 체크포인트의 정책 가중치를 불러오고 optimizer 상태는 새로 시작해 1024개 환경에서 300회, 실제 optimizer step 6000회를 수행했다. 초기 학습률은 `1e-4`이고 기존 adaptive schedule을 유지했다.
5. 호스트가 최종 체크포인트 하나를 같은 과제에서 재생했다. 모델 응답을 회복 판정으로 쓰지 않고, 실행 전에 고정한 진행·하강·자세·추종 기준으로 판정했다.

모델은 등록된 검사와 적응 후보 중 다음 행동을 골랐다. 실행 인자와 학습 예산은 호스트가 고정했고, 코드 결함의 원인은 미확정으로 남겼다. [진단 기록](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/diagnosis.json), [승인 결합](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/adaptation-authorization.json), [학습 영수증](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/training.json)을 구분했다.

## 같은 명령에서의 전후 평가

계단 높이 0.05m·폭 0.30m·하행·명령 0.25m/s·평가 seed 200·30초·50Hz를 유지했다. 평가 설정은 체크포인트만 다르고, trace 첫 행의 기록된 몸체 상태도 같았다. 관절 초기 상태 전체를 별도로 측정했다는 주장은 하지 않는다. 관찰 중 자동 리셋을 끈 설정도 양쪽에서 같다. 학습 중에는 접지·시간 종료를 유지했다.

| 30초 관측 | 원 정책 | 에이전트 제안 후 추가 학습 정책 |
|---|---:|---:|
| 실제 명령 | 0.25m/s 유지 | 0.25m/s 유지 |
| 전진 거리 | 2.761855m | 7.217531m |
| 높이 변화 | −0.600006m | −0.352950m |
| 최소 body-up z | −0.988856 | 0.799609 |
| 뒤집힌 자세 표본 | 888 | 0 |
| 몸체 좌표 xy 속도 추종 RMSE | 0.232850m/s | 0.064029m/s |
| 리셋 | 0 | 0 |
| 등록 과제 판정 | 미통과 | 통과 |

통과 조건은 명령 보존, 리셋·뒤집힘 0, 전진 3.75m 이상, 하강 0.30m 이상, 최소 body-up z 0.5 이상, xy 추종 RMSE 0.25m/s 이하다. 이 판정은 등록된 계단 과제에 적용하며 기존 fixed26 회복 표와 분리해 집계한다.

## 모델 입력과 실행 기록

요청 모델은 `nvidia/nemotron-3.5-super-vl-120b-a12b`, 반환 모델 ID는 등록한 별칭 `private/nvidia/nemotron-3.5-super-120b-a12b`였다. 실제 실패 영상의 13초 JPEG와 해당 단계의 사실·검사 결과를 입력했다. [첫 요청](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/model/decision-01.request.json)과 [검사 후 요청](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/model/decision-02.request.json)을 공개했다. 최종 제안의 인용은 모두 검사 수치이며, 영상 추가 효과를 비교한 실행은 아니다.

이전 인용 계약은 실제로 존재하는 `source.observed_symptom`을 두 번 거절했다. 상위 fact ID만 받던 검증기와 모델의 세부 경로 인용이 맞지 않았다. 그 두 응답은 무효로 보존하고, [실제 경로를 열거하는 새 계약](../evals/stairs_agent_repair_v2.py)을 등록한 뒤 두 번의 유효 응답을 받았다. 누적 호출은 4회, 입력 5988·출력 588토큰이며 학습 후 추가 호출은 0회다. 이는 모델 성능 비교가 아닌 계약 개발 기록이다.

학습 자식의 wall time은 610.031초로 앱 시작·정책 로딩·종료 정리를 포함한다. 최종 정책 SHA는 `dd32d72400b8b21e734bfc17dec18a1488ea00237f0475c7fe764b609c97455f`다. 앞선 실행 접점 실패의 소비와 재사용한 원본도 [결과 JSON](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/summary.json)에 포함했다. GPU kernel 시간이나 비용 절감 수치로 해석하지 않는다.

설계자가 원 실패와 명령 속도 대조를 본 뒤 만든 단일 개발 계약이다. 별도 결함 주입 없이 공식 정책의 이 조건 실패를 다뤘다. 과제 적응 성공과 코드 결함 원인 확정은 구분하며, 다른 seed·조건의 성능과 규칙 대비 효과는 이번 측정 범위 밖이다.

## 원본과 재현 계약

- [최종 실행 전 등록 v7](../bench/protocols/stairs_agent_repair_s200_20261106_v7.json), [실제 모델 단계 등록 v5](../bench/protocols/stairs_agent_repair_s200_20261106_v5.json).
- [고정 과제 재생기](../bench/run_go2_stairs_agent_play_v1.py), [학습 실행기](../bench/train_stairs_agent_adaptation_v3.py), [새 인용 계약](../evals/stairs_agent_repair_v2.py).
- [평가 공통 설정](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/evaluation-common-config.json), [원 정책 trace](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/baseline-state-trace.jsonl), [추가 학습 후 trace](../evals/results/go2_stairs_agent_adaptation_s200_20261106_v1/adapted-state-trace.jsonl).
- [영상 출처·프레임·전체 디코딩](media/go2-stairs-agent-adaptation-s200-20261106-v3/source_manifest.json). 15초 비교는 양쪽 원본의 첫 750프레임을 같은 확대 구도로 사용하며 재생 속도 변경·길이 연장용 동작 반복·정지 화면은 없다. 통과 표시는 전체 30초 trace 판정에 따른다.
