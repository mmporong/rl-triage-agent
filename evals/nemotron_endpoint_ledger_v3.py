"""v2 전역 원장을 보존하며 등록 provider endpoint alias만 검증용 view에서 정규화한다.

원시 reservation/response는 부모 ``Ledger``가 그대로 기록한다. 이 facade의 ``audit``과
``call``/``async_call`` 반환값만 phase label을 벗기고 provider identity view를 만든다.
정규화는 동일 weights의 증명이 아니라, 관찰된 endpoint 식별자를 요청 모델의 허용 목록에
결합하는 실행 계약이다.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))
import nemotron_bottleneck_ledger_v2 as V2

PHASE_PREFIX = "v3::"
# 상위 실행기가 provider 통계 필터에 쓰는 짧은 공개 이름.
PREFIX = PHASE_PREFIX
PRIVATE_CANDIDATE_ALIAS = "private/nvidia/nemotron-3.5-super-120b-a12b"
ALLOWED_PROVIDER_IDS = {
    V2.BASELINE: frozenset({V2.BASELINE}),
    V2.CANDIDATE: frozenset({V2.CANDIDATE, PRIVATE_CANDIDATE_ALIAS}),
}
IDENTITY_POLICY = "registered_endpoint_alias_v3_not_weights_attestation"


def provider_view(result, requested_model):
    """원시 결과를 바꾸지 않고 등록된 provider ID에 한해 검증용 모델 ID를 만든다."""
    view = copy.deepcopy(result)
    raw = view.get("provider_model")
    view["provider_model_raw"] = raw
    view["identity_policy"] = IDENTITY_POLICY
    allowed = ALLOWED_PROVIDER_IDS.get(requested_model, frozenset())
    if raw not in allowed:
        view["provider_identity_registered"] = False
        return view
    view["provider_identity_registered"] = True
    view["provider_model"] = requested_model
    # B3 v2의 exact provider guard가 alias 응답에 붙인 오류만 view에서 해제한다.
    # 다른 전송·내용 오류는 alias가 등록되어도 그대로 둔다.
    if raw != requested_model and view.get("error_type") == "ProviderModelMismatch":
        view.pop("error_type", None)
        view.pop("infra_error", None)
    return view


class PhaseLedger:
    """v2 전역 원장을 쓰되 ``v3::`` label의 검증용 view만 노출한다."""

    def __init__(self, parent):
        self.parent = parent

    @staticmethod
    def _raw_label(label):
        if not isinstance(label, str) or not label or label.startswith(PHASE_PREFIX):
            raise ValueError("v3 phase 요청 label이 잘못됐다")
        return PHASE_PREFIX + label

    def global_raw_audit(self):
        """부모가 검증한 전역 원장 전체의 복사본."""
        return copy.deepcopy(self.parent.audit())

    def raw_audit(self):
        """v3 phase 행만 원시 label·provider 값 그대로 반환한다."""
        return [row for row in self.global_raw_audit()
                if row["reservation"]["label"].startswith(PHASE_PREFIX)]

    def audit(self):
        """기존 v2 소비자가 읽는 v3 전용 view. 원시 원장 연결값은 증거로 쓰지 않는다."""
        rows = self.raw_audit()
        for row in rows:
            reservation = row["reservation"]
            reservation["label_raw"] = reservation["label"]
            reservation["label"] = reservation["label"][len(PHASE_PREFIX):]
            if row["response"] is not None:
                row["response"]["result"] = provider_view(
                    row["response"]["result"], reservation["model"])
        return rows

    def call(self, stage, label, request, complete):
        result = self.parent.call(stage, self._raw_label(label), request, complete)
        return provider_view(result, request.get("model"))

    async def async_call(self, stage, label, request, complete):
        result, value = await self.parent.async_call(stage, self._raw_label(label), request, complete)
        return provider_view(result, request.get("model")), value
