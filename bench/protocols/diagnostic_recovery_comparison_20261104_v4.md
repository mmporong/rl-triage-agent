# 후속 교정 비교의 학습 시작 전 경로 오류 재실행 사전등록 v4

비교 조건·판정·예측은 [v2](diagnostic_recovery_comparison_20261104_v2.md), 저장된 모델 HOLD 처리는 [v3](diagnostic_recovery_comparison_20261104_v3.md)를 유지한다. 이 판은 S4b v3가 첫 학습을 시작하기 전에 멈춘 실행기 경로 오류를 고친 인프라 재실행이다.

## 보존하는 이전 기록

v3 첫 case-05/fixed_sequence는 교정 요청을 소비한 뒤 FileNotFoundError로 unknown 마감됐다. 모터 이어 학습 adapter가 owned 폴더를 준비하지 않고 owned/launch.json을 쓰는 기존 실행 함수를 호출했다. 원문에는 training/binding.json과 manifest.json만 있고 owned/launch.json·학습·고정평가 실행 영수증은 없다. 같은 실패를 실제 Windows Job CPU 실행 경계에서 재현했다. 학습용 SDK 자식과 모델 API는 시작하지 않았으며 회복 결과도 없다.

v3의 원본 요청·원장·출력 폴더·실패 요약은 보존한다. 그 요청을 다시 소비하거나 기존 stage에 재실행하지 않는다. v1·v2는 미실행 판이며, v3는 학습 시작 전 중단 판으로 구분한다.

## 고정하는 수정과 재실행

- 모터 이어 학습은 기존 V2 prepared-owned adapter를 재사용해 배타적인 새 owned 폴더를 만든 뒤 기존 실행 함수를 호출한다. 학습·교정·회복 판정 로직은 바꾸지 않는다.
- 기본 실행 경로와 주입 실행 경로 모두 실제 CPU 자식의 Windows Job 소유·자연 종료·정리를 확인한다. 실제 SDK Python 3.10의 짧은 CPU 실행도 통과해야 한다. 이 CPU 확인을 GPU 학습 증거로 세지 않는다.
- 새 source commit·새 v4 계획·새 출력 namespace를 실행 전에 등록한다. 원천 source/reference·등록 교정·모드·seed·300갱신을 유지한다. restart seed 미사용 검사는 새 jobs 준비 시 다시 수행한다.
- S4a의 원래 모델 응답과 진단 영수증을 사용한다. 새 모델 요청·의미 재시도·새 진단 검사·정답 기반 선택은 없다. 16arm 계획에서 확정15arm은 등록 교정, model_hold1arm은 교정 없이 남기는 기존 경로다.

## 상한과 결과 문구

Simulator 최대32·CPU checkpoint16·CPU telemetry16·전체 owned64, arm당5000초·API0을 유지한다. v3에서 Simulator/이들 CPU 작업은 시작하지 않았으므로 실행 수 상한을 추가하지 않는다. v3 준비 CPU wall253.0417314초/준비 프로세스 CPU99.609375초 및 실패 배치 wall26.8225587초는 별도 비용으로 보존한다. 새 시도의 수치만 골라 전체 비용으로 표시하지 않는다.

F2는 이어 학습, epoch 교정은 재학습이다. COMPOUND의 보상 결함을 보존하며 epoch만 교정해 행동 실패가 남을 수 있다는 예측을 유지한다. 실제 회복·미회복·HOLD·인프라 실패를 구분하고, 그 결과와 전체 시간·자원을 보고한다. 이미 진단 결과를 확인한 뒤 만든 실행 경로 수정이며 새 결함 일반화나 모델 성능 개선의 근거로 쓰지 않는다.
