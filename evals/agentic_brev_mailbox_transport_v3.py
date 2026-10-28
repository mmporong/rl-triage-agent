"""Transport2의 인프라 원문 보존과 Host bound-method 소스 계약을 함께 지킨다."""
from __future__ import annotations

from typing import Any

import agentic_brev_mailbox_transport_v2 as V2

SCHEMA = V2.SCHEMA
SOURCE = "evals/agentic_brev_mailbox_transport_v3.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_transport_v3.py"


class BrevMailboxTransport(V2.BrevMailboxTransport):
    """Host가 검사하는 공개 메서드를 이 등록 소스에 명시적으로 결합한다."""

    def prepare_pair(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().prepare_pair(operation_id, intent, registration_ref)

    def start_reference(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().start_reference(operation_id, intent, registration_ref)

    def lookup_reference(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().lookup_reference(operation_id, intent, registration_ref)

    def dispatch_case(
        self, operation_id: str, intent: dict, registration_ref: dict, authorization: dict,
    ) -> dict:
        return super().dispatch_case(operation_id, intent, registration_ref, authorization)

    def lookup_pair(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().lookup_pair(operation_id, intent, registration_ref)


def __getattr__(name: str) -> Any:
    return getattr(V2, name)
