# 공식 자료 재확인

확인: 2026-10-25T15:31:46.559960+09:00

- https://docs.tokenfactory.nebius.com/other-capabilities/billing-new : 가입 체험 $1/30일, 카드 계정의 음수 잔액·청구 threshold 도달 시 자동 청구, 프로모션 코드 적용 방법을 설명한다. 달러 hard cap 또는 자동 청구 중지 방법은 이 문서에서 확인하지 못했다.
- https://docs.tokenfactory.nebius.com/other-capabilities/billing : 과거 선불/선택적 auto top-up 안내가 별도로 남아 있어 현재 billing-new 및 계정 UI보다 우선하지 않았다. 그 문서의 음수 잔액 API 차단을 현재 계정의 보장으로 쓰지 않는다.
- https://nebiusglobalaihackathon.devpost.com/resources : 행사 활성화 코드 NEBIUS-DEVPOST-GLOBAL26의 신청 폼으로 Token Factory $25, Builders Program 가입으로 추가 $25를 안내한다.
- https://nebiusglobalaihackathon.devpost.com/details/faqs : 두 혜택은 Token Factory 크레딧이며 별도의 공개 AI Cloud 크레딧은 안내되지 않았다고 설명한다. 발급/국가 온보딩 문제는 Nebius의 개별 처리 대상이다.
- https://dev.nebius.com/builders : 무료 가입, 이메일 확인과 혜택 청구 절차, Token Factory $25를 안내한다. 실제 UI는 필수 약관 동의를 포함한다. 신청·동의는 하지 않았다.
- https://docs.tokenfactory.nebius.com/api-reference/examples/list-of-models : OpenAI 호환 API와 verbose 모델 카탈로그의 토큰당 가격 필드를 확인했다. 이번 가격은 model-catalog.json의 실제 인증 조회값을 사용했다.
- https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling : 명시적 function tool_choice와 호출 결과를 후속 메시지로 돌려주는 흐름을 확인했다.

계정 UI 관찰은 account-safety.json에 개인정보·카드 번호·키 원문 없이 기록했다. 금액은 공식 제공 혜택과 실제 수령/청구를 구분한다.
