"""발표 재생 화면의 D3 장면: 변경 8개 중 원인 하나를 확정하는 데 든 확인 재학습을 방법별로 보인다.

근거는 세 종류다. 사전등록 프로토콜(bench/protocols/d3_experiment_v2.json)의 사례·위치·예측, 요약
(docs/replay/slots.json의 d3_summary 경로)의 방법별 합·우위 조건·결과, 요약이 가리키는 사례 결과(case_result.json)의
모델 순위 1위. 요약과 사례 결과가 그 프로토콜의 것이 아니면(프로토콜 SHA 불일치) 그 근거는 불일치로 둔다.
요약이 아직 승인되지 않았으면 사전등록 표만 채우고 결과 자리는 슬롯으로 남긴다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import hashlib
from typing import Any

PROTOCOL = "bench/protocols/d3_experiment_v2.json"
SLOT = "d3_summary"
SLOT_LABEL = "D3 방법 비교 요약(본실험 3사례가 닫힌 뒤 summarize 결과)"
METHODS = ("B1", "B2", "B3", "B4", "B5")
METHOD_NAMES = {"B1": "B1 이분 탐색", "B2": "B2 하나씩", "B3": "B3 probe 범주", "B4": "B4 모델 순위",
                "B5": "B5 모델(probe 없음)"}
CONDITIONS = {"fewer_than_every_deterministic_sum": "① B4 합 < B1·B2·B3 합의 최소",
              "never_more_than_b3_in_a_case": "② 어느 사례에서도 B4 ≤ B3",
              "no_wrong_or_unresolved": "③ B4 틀린 확정·미확정 0",
              "three_valid_cases": "④ 유효 사례 3개"}
OUTCOME_NAMES = {"agent_fewer_retrains": "재학습 감소", "no_advantage": "우위 없음", "agent_worse": "열세",
                 "invalid": "판정 없음"}
SCOPE_D3 = ("승인된 공개 결과만 읽는 발표 재생 화면. 변경 8개·사전등록 사례 3건의 방법 비교이며 결함·무해 변경·검사를 같은 팀이 "
            "만들었다. 진단 정확도·일반화·GPU 절감의 근거가 아니다.")


def _lf_sha(path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _bind(sources: Any, relative: str | None, protocol_sha: str | None, *, kind: str) -> tuple[Any, str]:
    """파일을 근거로 읽는다. 그 프로토콜의 산출물이 아니면 불일치로 바꾸고 값을 쓰지 않는다."""
    data, sid = sources.load(relative, kind=kind) if relative else (None, sources.file(None, kind=kind)["id"])
    if data is not None and protocol_sha is not None and data.get("protocol_sha256_lf") != protocol_sha:
        sources.rows[sid].update(status="mismatch", reason="protocol_mismatch")
        return None, sid
    return data, sid


def scene_d3(sources: Any, slots: dict) -> dict:
    from build_replay_page import cell
    protocol, protocol_id = sources.load(PROTOCOL, kind="protocol")
    protocol_sha = _lf_sha(sources.root / PROTOCOL) if protocol is not None else None
    cases = protocol["cases"] if protocol else {}
    computed = protocol["predictions"]["computed"] if protocol else {}
    prereg = [[case, cell(sources, spec["seed"], protocol_id, 0), cell(sources, spec["harmful"], protocol_id, None),
               cell(sources, f"{spec['position']} / {len(spec['order'])}", protocol_id, None),
               cell(sources, " · ".join(str(computed[case][m]) for m in ("B1", "B2", "B3")), protocol_id, None),
               "집계" if spec["aggregate"] else "개발(집계 제외)"]
              for case, spec in cases.items()]
    blocks = [{"type": "table", "title": "사례와 사전 예측(상호작용 없음·검사가 원인 범주를 맞힌다는 전제)",
               "columns": ["사례", "seed", "원인 변경", "위치", "예측 B1 · B2 · B3", "집계"], "rows": prereg,
               "status_sources": [protocol_id]}]
    summary, summary_id = _bind(sources, slots.get(SLOT), protocol_sha, kind=SLOT)
    if summary is None:
        blocks.append({"type": "slot", "label": SLOT_LABEL, "source": summary_id})
    else:
        rows = []
        for case, relative in summary["cases"].items():
            result, result_id = _bind(sources, relative, protocol_sha, kind="d3_case_result")
            costs = summary["costs"].get(case)
            if costs is None:  # 무효·개발 사례는 요약 비용표에 없다. 값이 없는 칸이 "대기"로 보이지 않게 "-"로 둔다
                reason = (result or {}).get("invalid_reason") or ("개발 사례" if result and not result["aggregate"] else None)
                rows.append([case, *["-" for _ in METHODS], cell(sources, reason and f"집계 제외 · {reason}", result_id, None)])
                continue
            model = ((result or {}).get("methods", {}).get("B4") or {}).get("model") or {}
            top = (model.get("ranking_labels") or [None])[0]
            shown_top = top and f"{top} = {result['labels'][top]}{' (원인)' if result['labels'][top] == result['harmful'] else ''}"
            rows.append([case, *[cell(sources, costs[m]["retrains"], summary_id, 0) for m in METHODS],
                         cell(sources, shown_top or (model and model.get("status")), result_id, None)])
        conditions = [[CONDITIONS[key], cell(sources, "충족" if ok else "미충족", summary_id, None)]
                      for key, ok in summary["conditions"].items()]
        blocks += [
            {"type": "table", "title": "확인 재학습 수(그 방법이 혼자 돌았을 때, 미확정은 상한 8로 센다)",
             "columns": ["사례", *[METHOD_NAMES[m] for m in METHODS], "B4 1위"], "rows": rows,
             "status_sources": [summary_id]},
            {"type": "table", "title": "합과 우위 조건", "columns": ["항목", "값"], "rows": [
                ["합 B1 · B2 · B3 · B4 · B5", cell(sources, " · ".join(str(summary["sums"][m]) for m in METHODS), summary_id, None)],
                *conditions,
                ["틀린 확정 B1 · B2 · B3 · B4 · B5",
                 cell(sources, " · ".join(str(summary["wrong_confirmations"][m]) for m in METHODS), summary_id, None)],
                ["결과", cell(sources, f"{OUTCOME_NAMES[summary['outcome']]}({summary['outcome']})", summary_id, None)],
                ["결과 문구(사전등록)", cell(sources, summary["wording"], summary_id, None)]],
             "status_sources": [summary_id]},
        ]
    return {
        "id": "d3", "title": "변경 여덟 개 중 원인 하나를 확정하는 데 든 확인 재학습", "dense": True, "scope": SCOPE_D3,
        "lead": ("무해 변경 일곱 개와 섞인 원인 변경 하나를, 그 변경 하나만 되돌린 재학습이 정상일 때만 확정한다. 다섯 방법이 같은 "
                 "상한(8회)과 같은 확정 규칙으로 쓴 재학습 수를 센다."),
        "blocks": blocks,
        "notes": ["모델 순위(B4)가 재학습을 줄였다고 말하는 것은 사전등록한 네 조건을 모두 채울 때뿐이다. 사례가 세 건이라 통계적 "
                  "우위나 일반화를 말하지 않는다.",
                  "무해 변경 목록은 무해성 smoke v1 결과를 보고 한 번 바꿨다(v2에서 보상 가중치 변경을 같은 수식 리팩터링으로 교체)."],
    }
