# Nemotron이 작성한 PPO 설정 패치

Nemotron 3.5가 설정·코드·실제 PPO 갱신 기록을 읽고 새 설정 어댑터를 작성하는 개발 측정이다. 모델에 정답 패치나 결함 이름은 제공하지 않았다. 호스트는 생성 코드를 제한된 AST 계약으로 검사하고, 실제 RSL-RL PPO가 해당 설정을 소비하는지 확인한다.

기존 seed 2311은 모델이 검사를 고르고 호스트가 등록 교정을 적용한 사례다. 이 측정은 수정 소스를 모델이 작성한다는 점에서 구분한다. 계단 seed 200의 정책 적응 결과도 여기에 합산하지 않는다.

## 생성 패치와 CPU 검사

알려진 PPO epoch 결함 계열에서 인자 누락·상수 덮어쓰기·설정 후 재덮어쓰기 세 구현과 정상 구현 하나를 만들었다. 모두 개발용 합성 설정 어댑터이며, 처음 보는 결함군을 찾아낸 결과로 해석하지 않는다.

각 사례에 한 번씩 요청했다. 응답 네 개는 모두 첫 응답에서 출력 계약을 통과했으며, 의미 오류를 고치기 위한 재요청은 없었다.

| 검사 | 원래 결함 구현의 optimizer.step / update | 생성 패치 적용 후 | 기대값 |
| --- | ---: | ---: | ---: |
| epoch 5 · minibatch 4 | 4 | 20 | 20 |
| epoch 3 · minibatch 2 | 2 | 6 | 6 |
| epoch 2 · minibatch 3 | 3 | 6 | 6 |

세 결함 패치 모두 위 세 설정에서 실제 CPU PPO 갱신 횟수를 정상화했다. 설정마다 세 update를 관측했으며, 추가 algorithm 설정도 보존했다. 텐서로 만든 PPO 입출력 환경을 사용했으므로 이 수치는 로봇 행동 회복을 뜻하지 않는다.

정상 코드에서도 모델은 패치를 제안했다. 정상 무수정 판단은 **0/1**이다. 호스트는 기존 코드의 갱신이 이미 정상인 것을 확인해 이 수정의 승격을 차단했다. 이 차단은 모델의 올바른 판단으로 세지 않는다.

## 결과를 본 뒤 추가한 기본값 검사

주 측정을 마친 뒤 `num_learning_epochs`를 생략한 설정을 추가로 검사했다. 설치된 RSL-RL PPO의 기본값은 1이다. 세 결함 패치 중 두 개는 `config.get("num_learning_epochs", 5)`를 넣어 생략된 설정을 5로 바꿨고, 한 개만 설치된 기본값을 보존했다. 이 검사는 사후 개발 검사이며 주 측정의 성공 분모를 바꾸지 않는다.

GPU 행동 비교에 선택한 패치는 모델 원문의 다음 소스다. 호스트가 정답 코드로 치환하지 않았다.

```python
def build_algorithm_kwargs(config):
    options = dict(config)
    return options
```

모델의 실제 응답은 [상수 덮어쓰기 사례 원문](../evals/results/ppo_generated_patch_cpu_20261106_v1/model/literal_override.response-01.json)에 보존했다. 요청한 모델은 `nvidia/nemotron-3.5-super-vl-120b-a12b`, 반환된 provider 별칭은 `private/nvidia/nemotron-3.5-super-120b-a12b`였다. 이번 입력은 텍스트이며 VL 영상 효과나 3 Super 대비 우위를 측정하지 않았다.

## 실행 경계

생성 소스는 `build_algorithm_kwargs(config)` 함수 하나로 제한한다. 설정 dictionary 복사·상수 조회·epoch 항목 대입·반환만 허용하며 import, 파일·네트워크·셸 접근, 반복문, 임의 함수 호출은 거부한다. 여러 설정에서 실제 갱신 횟수와 나머지 설정의 보존을 확인한 후보만 별도 GPU 등록 대상으로 삼는다.

정상·결함·선택된 패치의 GPU 비교는 같은 task·seed·환경 수·학습 update 예산과 같은 고정 평가를 사용한다. 정상 코드는 유지하고, reward·termination 설정 해시와 실제 PPO epoch·minibatch 값을 각각 확인한다. 패치 생성, 검사 정상화, 행동 회복, 정상 대조는 별개 항목으로 보고한다.

## 근거와 재현 코드

- [CPU 공개 요약](../evals/results/ppo_generated_patch_cpu_20261106_v1/summary.json) · [주 측정 판정](../evals/results/ppo_generated_patch_cpu_20261106_v1/primary-verdicts.json)
- [주 측정 사전등록](../bench/protocols/ppo_generated_patch_cpu_20261106_v2.json) · [미실행 이전 판](../bench/protocols/ppo_generated_patch_cpu_20261106_v1.json)
- [CPU 적용 전](../evals/results/ppo_generated_patch_cpu_20261106_v1/cpu-before.json) · [적용 후](../evals/results/ppo_generated_patch_cpu_20261106_v1/cpu-after.json) · [사후 기본값 검사](../evals/results/ppo_generated_patch_cpu_20261106_v1/posthoc-default-edge.json)
- [호스트 입장 판정](../evals/results/ppo_generated_patch_cpu_20261106_v1/host-admission.json)
- [제한 소스 계약](../evals/ppo_patch_contract_v1.py) · [실제 CPU PPO 검사](../bench/inspect_ppo_patch_cpu_v1.py) · [GPU 학습 실행기](../bench/train_generated_ppo_adapter_v1.py)

이 측정의 성공 여부는 등록한 개발 사례의 결과다. 임의 저장소의 오류 수정, 미지 결함 일반화, 모델 버전 우위, GPU 비용 절감을 검증한 결과로 확대하지 않는다.
