# AGENTS.md — rl-triage-agent

이 저장소는 NVIDIA Korea Agentic AI Hackathon 2026(팀 Mollet) 제출작이다. 작업을 시작하기 전에 `docs/HANDOFF.md`를 먼저 읽는다.

## 작업 규칙

- 응답과 커밋 메시지 설명은 한글. 커밋은 Conventional Commits, 변경 파일만 경로별 스테이징.
- 평가 공정성: 에이전트 프롬프트·도구를 바꾸면 **평가 전에 커밋**하고, 이미 본 세트로 성능을 주장하지 않는다. 새 수치는 새 seed 보류 세트로만 낸다. 재시도는 인프라 오류(과부하·빈 응답·잘림)에만 허용한다.
- 결과 파일(`evals/results/**`)은 덮어쓰지 않는다. 새 실행은 새 `--tag` 폴더로 남긴다.
- README·제출 문안의 수치는 결과 파일에서 계산한 값만 쓴다. 유의하지 않은 차이를 유의하다고 쓰지 않는다.
- API 키는 저장소에 두지 않는다. WSL `~/.config/nvidia/env`에만 있다.
- 외부 공개 행동(push 외 배포·폼 제출·메시지)은 사용자 확인 후 한다.
