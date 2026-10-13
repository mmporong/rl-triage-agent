"""N3: NAT HITL 미들웨어 원장 어댑터. 명시적 approve만 원장 approve로 이어지고 정본은 원장이다."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nat.builder.context import ContextState  # noqa: E402
from nat.builder.workflow_builder import WorkflowBuilder  # noqa: E402
from nat.data_models.config import Config  # noqa: E402
from nat.data_models.interactive import (BinaryHumanPromptOption, HumanResponseBinary, HumanResponseCheckbox,  # noqa: E402
                                         HumanResponseDropdown, HumanResponseNotification, HumanResponseRadio,
                                         HumanResponseText, InteractionResponse, InteractionStatus,
                                         MultipleChoiceOption)
from nat.runtime.loader import PluginTypes, discover_and_register_plugins  # noqa: E402
from nat.utils.data_models.schema_validator import validate_schema  # noqa: E402

from rl_triage import nat_hitl_ledger as H  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_registry import REGISTRY, protocol_case_arm  # noqa: E402

CASE = "h02_s2099"
PREREG = {"case": CASE, "checkpoint_sha256": "c" * 64}
LOOP_STATE = "loop-state-digest"


def _response(content) -> InteractionResponse:
    return InteractionResponse(id="r", status=InteractionStatus.COMPLETED, timestamp="2026-10-13T00:00:00Z",
                               content=content)


def _binary(value) -> InteractionResponse:
    return _response(HumanResponseBinary(selected_option=BinaryHumanPromptOption(id=str(value), label="x", value=value)))


def _choice(cls, value: str) -> InteractionResponse:
    return _response(cls(selected_option=MultipleChoiceOption(id=value, label=value, value=value, description=value)))


NOT_APPROVE = {
    "binary_reject": _binary("reject"), "binary_hold": _binary("hold"), "binary_true": _binary(True),
    "binary_upper": _binary("APPROVE"), "binary_yes": _binary("yes"),
    "text_approve": _response(HumanResponseText(text="approve")), "notification": _response(HumanResponseNotification()),
    "radio_approve": _choice(HumanResponseRadio, "approve"), "checkbox_approve": _choice(HumanResponseCheckbox, "approve"),
    "dropdown_approve": _choice(HumanResponseDropdown, "approve"), "no_content": SimpleNamespace(content=None),
}


def _propose(kind: str, loops: Path) -> tuple[str, L.Ledger]:
    ledger = H.open_ledger(kind, loops, CASE)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "probe":
        return ledger.propose(PREREG, "P_reward", 100)["request_id"], ledger
    entry = REGISTRY["reward"]
    return ledger.propose(LOOP_STATE, "reward", "receipts", entry["correction_id"], "restart",
                          protocol_case_arm("reward"), 100)["request_id"], ledger


def _consume(kind: str, ledger: L.Ledger, rid: str) -> dict:
    return ledger.consume(rid, PREREG, "P_reward") if kind == "probe" else ledger.consume(rid, LOOP_STATE)


def test_only_an_explicit_binary_approve_is_read_as_approval():
    assert H.decision_from_response(_binary("approve")) == H.APPROVE
    assert {name: H.decision_from_response(r) for name, r in NOT_APPROVE.items()} == dict.fromkeys(NOT_APPROVE, H.NO_DECISION)


@pytest.mark.parametrize("kind", H.KINDS)
def test_approve_writes_one_ledger_approval_that_consume_then_accepts(tmp_path, kind):
    rid, ledger = _propose(kind, tmp_path)
    shown = H.pending_request(kind, tmp_path, CASE)
    assert shown["request_id"] == rid and shown["pending_count"] == 1
    result = H.approve_shown_request(shown, H.APPROVE, tmp_path, "nat_hitl:operator")
    assert result == {"hitl_decision": "approve", "ledger_event": "approved", "ledger_state": "approved"}
    events = ledger.events()
    assert [e["event"] for e in events] == ["proposed", "approved"] and events[-1]["approver"] == "nat_hitl:operator"
    assert _consume(kind, ledger, rid)["event"] == "consumed"


@pytest.mark.parametrize("kind", H.KINDS)
@pytest.mark.parametrize("name", sorted(NOT_APPROVE))
def test_any_other_response_leaves_the_ledger_and_consume_refuses(tmp_path, kind, name):
    rid, ledger = _propose(kind, tmp_path)
    before = ledger.path.read_bytes()
    decision = H.decision_from_response(NOT_APPROVE[name])
    result = H.approve_shown_request(H.pending_request(kind, tmp_path, CASE), decision, tmp_path, "nat_hitl:operator")
    assert result == {"hitl_decision": H.NO_DECISION, "ledger_event": None, "ledger_state": "pending"}
    assert ledger.path.read_bytes() == before
    with pytest.raises(L.LedgerError):
        _consume(kind, ledger, rid)


@pytest.mark.parametrize("kind", H.KINDS)
def test_the_ledger_stays_canonical(tmp_path, kind):
    _, ledger = _propose(kind, tmp_path)
    shown = H.pending_request(kind, tmp_path, CASE)
    with pytest.raises(ValueError, match="다르다"):
        H.approve_shown_request({**shown, "effect_key": "0" * 16}, H.APPROVE, tmp_path, "nat_hitl:operator")
    for decision in (H.APPROVE, H.NO_DECISION):
        with pytest.raises(ValueError, match="원장에 없는"):
            H.approve_shown_request({**shown, "request_id": "000000000000"}, decision, tmp_path, "nat_hitl:operator")
    for output in ("not json", "[1, 2]", None):
        with pytest.raises(ValueError, match="JSON"):
            H.shown_request(output)
    H.approve_shown_request(shown, H.APPROVE, tmp_path, "nat_hitl:operator")
    after_first = ledger.path.read_bytes()
    with pytest.raises(L.LedgerError):  # 대기가 아니면 원장이 거부한다. HITL 응답은 이를 넘지 못한다
        H.approve_shown_request(shown, H.APPROVE, tmp_path, "nat_hitl:operator")
    assert ledger.path.read_bytes() == after_first
    with pytest.raises(ValueError, match="케이스"):
        H.pending_request(kind, tmp_path, "../escape")


def test_no_single_pending_request_means_nothing_to_approve(tmp_path):
    ledger = H.open_ledger("probe", tmp_path, CASE)
    ledger.path.parent.mkdir(parents=True)
    ledger.propose(PREREG, "P_reward", 100)
    ledger.propose(PREREG, "P_noise", 100)
    shown = H.pending_request("probe", tmp_path, CASE)
    assert shown["request_id"] is None and shown["pending_count"] == 2
    before = ledger.path.read_bytes()
    assert H.approve_shown_request(shown, H.APPROVE, tmp_path, "x")["ledger_event"] is None
    assert ledger.path.read_bytes() == before


def test_the_adapter_has_a_single_ledger_approve_call_site():
    tree = ast.parse((ROOT / "src/rl_triage/nat_hitl_ledger.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr in ("approve", "reject", "consume", "propose", "_append", "hold", "receipt", "recover")]
    assert [c.func.attr for c in calls] == ["approve"]
    owner = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and calls[0] in ast.walk(f))
    assert owner.name == "approve_shown_request"


def _build_config(loops: Path, pre_invoke: bool = False) -> Config:
    raw = yaml.safe_load((ROOT / "configs/nat_hitl_ledger_example.yml").read_text(encoding="utf-8"))
    for section in ("functions", "middleware"):
        for value in raw[section].values():
            value["loops_dir"] = str(loops)
    if pre_invoke:
        raw["middleware"]["ledger_hitl"]["pre_invoke_prompt"] = raw["middleware"]["ledger_hitl"]["post_invoke_prompt"]
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    return validate_schema(raw, Config)


def _invoke(config: Config, function: str, answers: list | None):
    """NAT 빌더로 함수를 만들고 부른다. answers가 None이면 사람 응답 콜백을 등록하지 않는다."""
    prompts = []

    async def callback(prompt):
        prompts.append(prompt.content.text)
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def run():
        if answers is not None:
            ContextState.get().user_input_callback.set(callback)
        async with WorkflowBuilder(general_config=config.general) as builder:
            await builder.populate_builder(config, skip_workflow=True)
            fn = await builder.get_function(function)
            return await fn.ainvoke(CASE)
    return asyncio.run(run()), prompts


@pytest.mark.parametrize("kind,function", [("probe", "pending_probe"), ("correction", "pending_correction")])
def test_nat_middleware_maps_only_approve_to_the_ledger(tmp_path, kind, function):
    rid, ledger = _propose(kind, tmp_path)
    config = _build_config(tmp_path)
    before = ledger.path.read_bytes()
    out, prompts = _invoke(config, function, [HumanResponseBinary(selected_option=BinaryHumanPromptOption(
        id="hold", label="hold", value="hold"))])
    assert '"hitl_decision": "no_decision"' in out and ledger.path.read_bytes() == before
    assert rid in prompts[0]  # 운영자에게 승인할 요청 내용을 보여 준다
    out, _ = _invoke(config, function, [HumanResponseBinary(selected_option=BinaryHumanPromptOption(
        id="approve", label="approve", value="approve"))])
    assert '"ledger_event": "approved"' in out and ledger.requests()[rid]["state"] == "approved"
    approver = next(e for e in ledger.events() if e["event"] == "approved")["approver"]
    assert approver.startswith("nat_hitl:operator:") and approver != "nat_hitl:operator:None"  # 상호작용 id


@pytest.mark.parametrize("answers,error", [(None, NotImplementedError), ([TimeoutError("no answer")], TimeoutError)])
def test_nat_middleware_fails_closed_without_an_answer(tmp_path, answers, error):
    rid, ledger = _propose("probe", tmp_path)
    before = ledger.path.read_bytes()
    with pytest.raises(error):
        _invoke(_build_config(tmp_path), "pending_probe", answers)
    assert ledger.path.read_bytes() == before and ledger.requests()[rid]["state"] == "pending"


def test_pre_invoke_gate_skips_the_tool_unless_approved(tmp_path):
    _, ledger = _propose("probe", tmp_path)
    before = ledger.path.read_bytes()
    hold = HumanResponseBinary(selected_option=BinaryHumanPromptOption(id="hold", label="hold", value="hold"))
    out, prompts = _invoke(_build_config(tmp_path, pre_invoke=True), "pending_probe", [hold])
    assert out is None and len(prompts) == 1 and ledger.path.read_bytes() == before
