# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import run_prospective_reasoning_confirmation_v3 as V


BASE_V2 = V.ROOT / "bench/protocols/prospective_reasoning_confirmation_execution_20261101_v2.json"
BASE_V1 = V.ROOT / "bench/protocols/prospective_reasoning_confirmation_execution_20261101_v1.json"
DESIGN = V.ROOT / "bench/protocols/prospective_reasoning_confirmation_design_20261101_v2.json"
BRIDGE = V.ROOT / "bench/protocols/prospective_reasoning_confirmation_bridge_20261101_v2.json"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _failed_root(tmp_path: Path) -> tuple[Path, list, dict]:
    root = tmp_path / "bridge"
    messages = [{"role": "user", "content": "registered input"}]
    built = V.R.BRIDGE.MODE.build_request(messages, "legacy_none")
    assert built.pop("messages") == messages
    params = built
    bridge_sha = V._sha(BRIDGE)
    request = {
        "schema": V.R.BRIDGE.REQUEST_SCHEMA, "index": 1, "slot": V.FIRST_SLOT,
        "case_id": "c02", "mode": "legacy_none", "messages": messages,
        "messages_sha256": V._digest(messages), "params": params,
        "params_sha256": V._digest(params), "bridge_protocol_sha256": bridge_sha,
        "context_sha256": V._digest({"slot": V.FIRST_SLOT, "messages": messages,
                                      "params": params}),
    }
    request_path = root / V.ARTIFACTS["request"]
    _write(request_path, request)
    response = {"schema": V.R.BRIDGE.RESPONSE_SCHEMA,
                "request_sha256": V._sha(request_path),
                "result": {"content": "{}", "finish_reason": "stop",
                           "provider_model": "private/nvidia/nemotron-3.5-super-120b-a12b",
                           "request_params": params}}
    _write(root / V.ARTIFACTS["response"], response)
    _write(root / V.ARTIFACTS["session"], {
        "schema": V.R.BRIDGE.SESSION_SCHEMA,
        "protocol": {"path": V._relative(BRIDGE), "sha256": bridge_sha},
        "anchor": V.V2.STAGE_A_ANCHOR, "max_requests": V.R.BRIDGE.MAX_REQUESTS})
    claim = {"schema": "registered_reasoning_worker_claim_v2", "index": 1,
             "slot": V.FIRST_SLOT, "request_sha256": V._sha(request_path),
             "bridge_protocol_sha256": bridge_sha, "pid": 7, "start_ticks": 9}
    _write(root / V.ARTIFACTS["claim"], claim)
    _write(root / V.ARTIFACTS["slot_claim"], claim)
    return root, messages, params


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    source, corpus = tmp_path / "source.json", tmp_path / "corpus.json"
    _write(source, {"source": "fixture"})
    _write(corpus, {"corpus": "fixture"})
    return source, corpus


def test_actual_host_context_boundary_accepts_only_exact_true() -> None:
    def boundary(*_args, verify_context, **_kwargs):
        return V.R.HOST._context(
            verify_context, phase="before_role", role="case",
            state={"state_digest": "s"}, action={"action": "run_probe", "target": "P_reward"},
            spec={"schema": "request"})

    adapted = V.approved_host_context(boundary)
    assert adapted(verify_context=lambda _request: True) == {"approved": True}
    for value in (False, None, {"approved": True}):
        with pytest.raises(ValueError, match="정확한 True"):
            adapted(verify_context=lambda _request, value=value: value)

    def explode(_request):
        raise RuntimeError("context failed")

    with pytest.raises(RuntimeError, match="context failed"):
        adapted(verify_context=explode)


def test_host_adapter_revalidates_every_actual_boundary_call() -> None:
    calls = []

    def boundary(*_args, verify_context, **_kwargs):
        first = V.R.HOST._context(verify_context, phase="before_role", role="case",
                                  state={"state_digest": "s"}, action={}, spec={})
        second = V.R.HOST._context(verify_context, phase="before_execution", role="case",
                                   state={"state_digest": "s"}, action={}, spec={})
        return {"first": first, "second": second}

    result = V.approved_host_context(boundary)(
        verify_context=lambda request: calls.append(request["phase"]) or True)
    assert calls == ["before_role", "before_execution"]
    assert result["first"] == result["second"] == {"approved": True}


def test_replay_first_slot_is_post_free_and_later_slot_delegates(tmp_path: Path) -> None:
    root, messages, params = _failed_root(tmp_path)
    delegated = []

    def delegate(root_arg, slot, sent, request_params):
        delegated.append((root_arg, slot, sent, request_params))
        return {"delegated": True}

    replay = V.ReplayCompleteFactory(root, BRIDGE, delegate)
    first = replay(V.FIRST_SLOT, {})(deepcopy(messages), deepcopy(params))
    assert first["transport_ref"]["sha256"] == V._sha(root / V.ARTIFACTS["response"])
    assert replay.replayed == 1 and delegated == []
    later = replay("c02.legacy_none.d2.a1", {})(deepcopy(messages), deepcopy(params))
    assert later == {"delegated": True} and len(delegated) == 1


def test_replay_rejects_wrong_first_slot_messages_and_tamper(tmp_path: Path) -> None:
    root, messages, params = _failed_root(tmp_path)
    replay = V.ReplayCompleteFactory(root, BRIDGE, lambda *_args: {})
    with pytest.raises(ValueError, match="첫 continuation"):
        replay("c02.enabled_low.d1.a1", {})(messages, params)

    root, messages, params = _failed_root(tmp_path / "messages")
    replay = V.ReplayCompleteFactory(root, BRIDGE, lambda *_args: {})
    with pytest.raises(ValueError, match="messages/params"):
        replay(V.FIRST_SLOT, {})(messages + [{"role": "user", "content": "changed"}], params)

    root, messages, params = _failed_root(tmp_path / "tamper")
    replay = V.ReplayCompleteFactory(root, BRIDGE, lambda *_args: {})
    response = root / V.ARTIFACTS["response"]
    response.write_text(response.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="원문이 바뀌었다"):
        replay(V.FIRST_SLOT, {})(messages, params)


def test_prepare_protocol_preserves_denominators_and_has_no_private_paths(tmp_path: Path) -> None:
    root, _messages, _params = _failed_root(tmp_path)
    source, corpus = _inputs(tmp_path)
    value = V.prepare_protocol(BASE_V2, BASE_V1, DESIGN, BRIDGE, source, corpus, root)
    assert value["inherited"] == V.V2.prepare_protocol(BASE_V1)["inherited"]
    assert value["failed_attempt"]["api_requests_consumed"] == 1
    assert value["failed_attempt"]["sdk_role_executions"] == 0
    assert value["failed_attempt"]["assessment"] == "indeterminate"
    assert value["continuation"]["policy_prompt_threshold_case_or_budget_changed"] is False
    assert value["claims"] == {
        "llm_superiority": False, "default_mode_adoption": False,
        "new_fault_family_generalization": False, "repair": False,
        "recovery": False, "pure_unexposed_performance": False}
    encoded = json.dumps(value)
    assert str(tmp_path) not in encoded and ("bench" + "/private") not in encoded


def test_execute_reuses_v2_lease_and_real_host_boundary(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    root, messages, params = _failed_root(tmp_path)
    source, corpus = _inputs(tmp_path)
    wrapper = V.prepare_protocol(BASE_V2, BASE_V1, DESIGN, BRIDGE, source, corpus, root)
    protocol = tmp_path / "outer.json"
    _write(protocol, wrapper)
    monkeypatch.setattr(V, "verify_protocol", lambda *_args: deepcopy(wrapper))

    def boundary(*_args, verify_context, **_kwargs):
        return V.R.HOST._context(verify_context, phase="before_role", role="case",
                                 state={"state_digest": "s"}, action={}, spec={})

    def fake_r_execute(*args, complete_factory, execute_host):
        complete_factory(V.FIRST_SLOT, {})(deepcopy(messages), deepcopy(params))
        assert execute_host(verify_context=lambda _request: True) == {"approved": True}
        out = Path(args[-1])
        _write(out / "result.json", {"schema": V.R.RESULT_SCHEMA})
        return {"schema": V.R.RESULT_SCHEMA, "status": "complete", "model_requests": 1,
                "paired_probes": 1, "sdk_role_executions_upper_bound": 2,
                "assessment": {"outcome": "indeterminate"}}

    monkeypatch.setattr(V.R, "execute", fake_r_execute)

    def fake_v2_execute(_protocol, anchor, base, *args, base_execute):
        assert anchor == V.V2_ANCHOR and base == BASE_V1
        return base_execute(base, V.V2.STAGE_B_ANCHOR, *args)

    out = tmp_path / "out"
    result = V.execute(protocol, "c" * 40, BASE_V2, BASE_V1, DESIGN, BRIDGE,
                       source, corpus, tmp_path / "lab", root, out,
                       v2_execute=fake_v2_execute, execute_host=boundary,
                       bridge_complete=lambda *_args: pytest.fail("첫 슬롯에서 POST delegate 호출"))
    assert result["replayed_provider_requests"] == 1
    assert result["new_provider_requests"] == 0
    assert result["classification"] == "mechanical_infrastructure_continuation"
    assert (out / "v3-result.json").is_file()


def test_execute_writes_terminal_error_once(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    root, _messages, _params = _failed_root(tmp_path)
    source, corpus = _inputs(tmp_path)
    wrapper = V.prepare_protocol(BASE_V2, BASE_V1, DESIGN, BRIDGE, source, corpus, root)
    protocol = tmp_path / "outer.json"
    _write(protocol, wrapper)
    monkeypatch.setattr(V, "verify_protocol", lambda *_args: deepcopy(wrapper))

    def fail(*_args, **_kwargs):
        raise RuntimeError("terminal failure")

    out = tmp_path / "failed-out"
    with pytest.raises(RuntimeError, match="terminal failure"):
        V.execute(protocol, "c" * 40, BASE_V2, BASE_V1, DESIGN, BRIDGE,
                  source, corpus, tmp_path / "lab", root, out, v2_execute=fail)
    error = json.loads((out / "error.json").read_text(encoding="utf-8"))
    assert error["schema"] == V.ERROR_SCHEMA and error["prior_api_requests"] == 1
    original = (out / "error.json").read_bytes()
    V._write_error(out, RuntimeError("other"))
    assert (out / "error.json").read_bytes() == original
