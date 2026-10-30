"""Nemotron 3.5 A/B 호출 실행 고리의 CPU 회귀. synthetic stub만 쓰며 외부 연결을 막는다."""
from __future__ import annotations

import json
import shutil
import socket
import sys
from copy import deepcopy
from pathlib import Path

import httpx
import openai
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import nemotron35_value_cases_v1 as P5
import nemotron35_value_compare_call_v1 as G

F2A, F2B, F1A, F1B = "go2-flat-f2-s2033", "go2-flat-f2-s2046", "go2-flat-f1-s2034", "go2-flat-f1-s2080"
FACT = "telemetry:case:Train/mean_reward"


class Crash(BaseException):
    """intent 기록 뒤 프로세스가 끊긴 상황을 흉내 낸다."""


class Stub:
    kind = G.PROVIDER_KINDS[G.SYNTHETIC]

    def __init__(self, decide, crash_at: int | None = None):
        self.decide, self.crash_at, self.calls = decide, crash_at, []

    def __call__(self, request: dict) -> dict:
        self.calls.append(request["slot_id"])
        if self.crash_at is not None and len(self.calls) == self.crash_at:
            raise Crash
        return self.decide(request)


class RealKind(Stub):
    kind = G.PROVIDER_KINDS[G.REAL]


def _body(model: str, content: str | None, finish: str = "stop", usage=(1000, 20, 1020)) -> bytes:
    return json.dumps({"id": "synthetic-cpu", "object": "chat.completion", "model": model,
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                    "finish_reason": finish}],
                       "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                                 "total_tokens": usage[2]}}).encode("utf-8")


def _answer(probe: str | None, facts=(FACT,)) -> str:
    return json.dumps({"action": "continue" if probe else "hold", "probe": probe, "fact_ids": list(facts)})


def _ok(content: str, model: str | None = None):
    return lambda request: {"status_code": 200, "body": _body(model or request["model"], content), "error_type": None}


def _table(table: dict, default=None):
    """(계보, 조건) → 응답 함수. 표에 없으면 default."""
    def decide(request: dict) -> dict:
        slot = request["slot_id"]
        return table.get((slot[3:-2], slot[-1]), default)(request)
    return decide


def _expected():
    """A는 모두 P_reward, B는 F2에 P_torque·F1에 P_reward를 고르는 synthetic 응답."""
    return _table({(F2A, "B"): _ok(_answer("P_torque")), (F2B, "B"): _ok(_answer("P_torque"))},
                  _ok(_answer("P_reward")))


def _registration(run: Path, kind: str = G.SYNTHETIC, **override) -> Path:
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    value = {"schema": G.REGISTRATION_SCHEMA, "kind": kind, "protocol_sha256": G.file_sha256(G.PROTOCOL),
             "plan_digest": plan["plan_digest"], "p5_manifest_digest": plan["p5_manifest_digest"],
             "source_commit": "0" * 40, "approval_granted": True, "approver": "synthetic-cpu-test",
             "request_cap": 24, "logical_calls": 8, "host": {"key_env": "NVIDIA_API_KEY", "platform": sys.platform},
             **override}
    path = run.parent / f"{run.name}.registration.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _rewrite_journal(run: Path, mutate) -> None:
    """journal 해시 연결을 다시 계산하는 정교한 변조. 원문 재생 검사가 따로 잡아야 한다."""
    events = [json.loads(line) for line in (run / "journal.jsonl").read_bytes().splitlines()]
    mutate(events)
    lines, previous = [], G.GENESIS
    for index, event in enumerate(events):
        event = {**{k: v for k, v in event.items() if k not in ("seq", "prev")}, "seq": index, "prev": previous}
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        lines.append(line)
        previous = G.raw_sha(line)
    (run / "journal.jsonl").write_bytes(b"".join(lines))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("외부 연결 시도")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("p6") / "run"
    G.prepare(run)
    return run


@pytest.fixture(scope="module")
def completed_run(base_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("p6done") / "run"
    shutil.copytree(base_run, run)
    assert G.execute(run, _registration(run), Stub(_expected()))["status"] == "completed"
    return run


def _copy(source: Path, tmp_path: Path, name: str = "run") -> Path:
    target = tmp_path / name
    shutil.copytree(source, target)
    for path in source.parent.glob(f"{source.name}.registration.json"):
        shutil.copyfile(path, tmp_path / f"{name}.registration.json")
    return target


def _plan(run: Path) -> dict:
    return json.loads((run / "plan.json").read_text(encoding="utf-8"))


def test_prepare_makes_eight_counterbalanced_identical_pairs(base_run: Path) -> None:
    plan = _plan(base_run)
    slots = plan["slots"]
    assert [slot["arm"] for slot in slots] == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert [pair[0]["arm"] for pair in zip(slots[::2], slots[1::2], strict=True)].count("A") == 2
    for first, second in zip(slots[::2], slots[1::2], strict=True):
        assert first["lineage_id"] == second["lineage_id"]
        assert {key for key in first if first[key] != second[key]} == {
            "slot_id", "order", "arm", "model", "accepted_provider_models"}
        assert (base_run / first["capsule_file"]).read_bytes() == (base_run / second["capsule_file"]).read_bytes()
    assert set(plan["blocked_arms"].values()) == {"blocked_new_collection_required"}
    assert plan["budget"] == {"logical_calls": 8, "request_cap": 24, "max_requests_per_slot": 1,
                              "retries_registered": False}
    assert plan["request_params"] == {key: value for key, value in G.F.request_params("x").items() if key != "model"}
    with pytest.raises(ValueError, match="이미 있다"):
        G.prepare(base_run)


def test_stub_responses_reach_choices_and_p5_score(completed_run: Path) -> None:
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((completed_run / "requests").glob("*"))]
    assert len(receipts) == 8 and all(row["synthetic"] and row["http_status"] == 200 for row in receipts)
    assert all(row["raw_sha256"] == G.raw_sha((completed_run / f"raw/{row['logical_call_id']}.attempt1.response")
                                              .read_bytes()) for row in receipts)
    assert all(row["usage"]["total_tokens"] == 1020 and row["latency_s"] >= 0 for row in receipts)
    assert {row["provider_model"] for row in receipts} == {G.load_protocol()["arms"][arm]["model"] for arm in "AB"}
    score = G.score_run(completed_run)
    assert score["axes"] == {"model_change": "development_observation_only", "image_contribution": "inconclusive"}
    assert score["run_kind"] == "synthetic_cpu_stub" and score["measurement_kind"] == "synthetic_cpu"
    assert score["requests_used"] == 8 and score["usage_totals"]["total_tokens"] == 8 * 1020
    assert not any(score["claims"].values()) and score["selection_change_alone_counted"] is False
    choices = json.loads((completed_run / score["files"]["choices"]).read_text(encoding="utf-8"))
    assert choices["schema"] == P5.CHOICES_SCHEMA and choices["choices"][F2A]["B"]["probe"] == "P_torque"
    assert G.score_run(completed_run)["files"] == score["files"]


def test_completed_slots_are_not_recalled_and_intent_only_holds(completed_run: Path, base_run: Path,
                                                               tmp_path: Path) -> None:
    run = _copy(completed_run, tmp_path)
    again = Stub(_expected())
    assert G.execute(run, _registration(run), again)["status"] == "completed" and again.calls == []
    resumed = G.resume(run)
    assert resumed["status"] == "verified" and resumed["provider_calls_this_invocation"] == 0

    crashed = _copy(base_run, tmp_path, "crashed")
    with pytest.raises(Crash):
        G.execute(crashed, _registration(crashed), Stub(_expected(), crash_at=3))
    later = Stub(_expected())
    held = G.execute(crashed, _registration(crashed), later)
    assert (held["status"], held["reason"], later.calls) == ("hold", "unknown_pending_intent", [])
    slots = G.resume(crashed)["slots"]
    assert list(slots.values()).count("unknown_pending") == 1 and list(slots.values()).count("not_started") == 5
    assert G.score_run(crashed)["axes"]["model_change"] == "inconclusive"


def test_capsule_protocol_params_raw_receipt_and_journal_changes_are_rejected(completed_run: Path,
                                                                              tmp_path: Path) -> None:
    slot = _plan(completed_run)["slots"][0]["slot_id"]

    def rejected(name: str, change, match: str, protocol: Path = G.PROTOCOL) -> None:
        run = _copy(completed_run, tmp_path, name)
        change(run)
        with pytest.raises(ValueError, match=match):
            G.resume(run, protocol)

    def capsule(run: Path) -> None:
        path = run / _plan(run)["slots"][0]["capsule_file"]
        path.write_bytes(path.read_bytes().replace(b"Train/mean_reward", b"Train/mean_rewarx", 1))

    rejected("capsule", capsule, "capsule")
    changed = tmp_path / "protocol.json"
    value = G.load_protocol()
    changed.write_text(json.dumps({**value, "purpose": "changed"}, ensure_ascii=False), encoding="utf-8")
    rejected("protocol", lambda run: None, "plan 결합", changed)

    def params(run: Path) -> None:
        plan = _plan(run)
        plan["request_params"]["max_tokens"] = 1
        (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    rejected("params", params, "plan이 바뀌었다")

    def model(run: Path) -> None:
        path = run / f"requests/{slot}.attempt1.json"
        path.write_text(path.read_text(encoding="utf-8").replace("120b-a12b", "120b-a12c", 1), encoding="utf-8")

    rejected("model", model, "receipt가 journal")

    def raw(run: Path) -> None:
        path = run / f"raw/{slot}.attempt1.response"
        path.write_bytes(path.read_bytes() + b" ")

    rejected("raw", raw, "원문이 바뀌었다")

    def usage(run: Path) -> None:
        path = run / f"requests/{slot}.attempt1.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["usage"]["total_tokens"] = 1
        data = G._json_bytes(receipt)
        path.write_bytes(data)
        _rewrite_journal(run, lambda events: events[1].update(receipt_sha256=G.raw_sha(data)))

    rejected("usage", usage, "usage")

    def journal(run: Path) -> None:
        path = run / "journal.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"status":"valid"', b'"status":"semantic_invalid"', 1))

    rejected("journal", journal, "journal 연결")


def test_plan_caps_duplicates_order_c_and_foreign_lineages_are_rejected(base_run: Path) -> None:
    protocol = G.load_protocol()
    manifest = P5.check(P5.PROTOCOL, base_run / "p5/manifest.json")
    sha = G.file_sha256(G.PROTOCOL)

    def check(mutate, match: str) -> None:
        plan = _plan(base_run)
        mutate(plan["slots"])
        plan["plan_digest"] = G.L.digest({k: v for k, v in plan.items() if k != "plan_digest"})
        with pytest.raises(ValueError, match=match):
            G.validate_plan(plan, manifest, protocol, sha)

    check(lambda slots: slots.append({**deepcopy(slots[0]), "slot_id": "09-extra"}), "8개를 넘는")
    check(lambda slots: [slot.update(max_requests=4) for slot in slots], "요청 상한 24")
    check(lambda slots: slots[0].update(max_requests=2), "재시도")
    check(lambda slots: slots.__setitem__(1, deepcopy(slots[0])), "중복 slot")
    check(lambda slots: slots[1].update(arm="C"), "C 또는 등록 밖")
    check(lambda slots: [slot.update(lineage_id="go2-flat-f9-s9999") for slot in slots[:2]], "P5 밖 계보")
    check(lambda slots: slots.insert(0, slots.pop(1)), "순서")
    plan = _plan(base_run)
    plan["slots"][0]["model"] = "nvidia/other-model"
    with pytest.raises(ValueError, match="plan이 바뀌었다"):
        G.validate_plan(plan, manifest, protocol, sha)
    with_c = deepcopy(protocol)
    with_c["arms"]["C"] = deepcopy(with_c["arms"]["B"])
    with pytest.raises(ValueError, match="A·B 외"):
        G.verify_protocol(with_c)


def test_unavailable_a_is_model_unavailable_and_model_axis_inconclusive(base_run: Path, tmp_path: Path) -> None:
    run = _copy(base_run, tmp_path)
    missing = {"status_code": 404, "body": b'{"status":404,"title":"Not Found"}', "error_type": "NotFoundError"}
    decide = _table({(lineage, "A"): (lambda request: missing) for lineage in (F2A, F2B, F1A, F1B)},
                    _expected())
    result = G.execute(run, _registration(run), Stub(decide))
    assert result["status"] == "completed"
    assert {status for slot, status in result["slots"].items() if slot.endswith("-A")} == {"model_unavailable"}
    score = G.score_run(run)
    assert score["axes"]["model_change"] == "inconclusive"
    assert {row["A"] for row in score["arm_status"].values()} == {"model_unavailable"}


def test_invalid_outputs_hold_without_rule_fallback(base_run: Path, tmp_path: Path) -> None:
    run = _copy(base_run, tmp_path)
    decide = _table({(F2A, "B"): _ok("not json"), (F2B, "B"): _ok(_answer("P_torque", ("telemetry:case:unseen",))),
                     (F1A, "B"): _ok(_answer("P_unregistered")),
                     (F1B, "B"): lambda r: {"status_code": 200, "body": _body(r["model"], "", "length"),
                                            "error_type": None}}, _ok(_answer("P_reward")))
    G.execute(run, _registration(run), Stub(decide))
    results = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (run / "results").glob("*")}
    rows = {row["lineage_id"]: row for row in results.values() if row["arm"] == "B"}
    assert {lineage: (row["status"], row["codes"]) for lineage, row in rows.items()} == {
        F2A: ("semantic_invalid", ["F_JSON"]), F2B: ("semantic_invalid", ["S_FACT_UNKNOWN"]),
        F1A: ("semantic_invalid", ["S_PROBE_INELIGIBLE"]), F1B: ("infrastructure_failed", ["EMPTY_OR_TRUNCATED"])}
    assert all(row["probe"] is None and row["rule_fallback_used"] is False for row in rows.values())
    score = G.score_run(run)
    p5 = json.loads((run / score["files"]["score"]).read_text(encoding="utf-8"))
    assert p5["results"][F2A]["B"]["hold"] is True and p5["results"][F2A]["B"]["followup"] is None
    assert score["axes"]["model_change"] == "inconclusive"


def test_foreign_contract_records_and_historical_answers_are_not_accepted(base_run: Path, tmp_path: Path) -> None:
    run = _copy(base_run, tmp_path)
    score = G.score_run(run)
    assert {row[arm] for row in score["arm_status"].values() for arm in "AB"} == {"pending_api"}
    assert score["axes"] == {"model_change": "inconclusive", "image_contribution": "inconclusive"}
    slot = _plan(run)["slots"][0]["slot_id"]
    (run / "raw").mkdir()
    (run / f"raw/{slot}.attempt1.response").write_bytes(_body("nvidia/nemotron-3-super-120b-a12b", _answer("P_reward")))
    with pytest.raises(ValueError, match="intent 없는 raw"):
        G.resume(run)
    other = _copy(base_run, tmp_path, "forged")
    (other / "journal.jsonl").write_bytes(b"")
    _rewrite_journal(other, lambda events: events.append({"event": "response", "slot_id": slot, "attempt": 1}))
    with pytest.raises(ValueError, match="intent 없는 응답"):
        G.resume(other)


def test_execute_holds_before_any_call_without_registration_or_host(base_run: Path, tmp_path: Path) -> None:
    run = _copy(base_run, tmp_path)
    never = RealKind(lambda request: pytest.fail("호출되면 안 된다"))
    assert G.execute(run, None, Stub(_expected())) == {"status": "hold", "reason": "execution_registration_missing",
                                                         "provider_calls_this_invocation": 0}
    assert G.execute(run, _registration(run, G.REAL), never)["reason"] == "host_key_missing"
    mismatch = _registration(run, G.REAL, host={"key_env": "NVIDIA_API_KEY", "platform": "other-os"})
    assert G.execute(run, mismatch, never)["reason"] == "host_platform_mismatch"
    with pytest.raises(ValueError, match="provider 종류"):
        G.execute(run, _registration(run), never)
    with pytest.raises(ValueError, match="승인"):
        G.execute(run, _registration(run, approval_granted=False), Stub(_expected()))
    assert G.main(["--execute", "--run", str(run)]) == 2
    assert never.calls == [] and not (run / "journal.jsonl").exists() and not (run / "registration.json").exists()


def test_cli_prepare_resume_score_need_no_key_or_network(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run = tmp_path / "cli"
    assert G.main(["--prepare", "--run", str(run)]) == 0
    assert G.main(["--resume", "--run", str(run)]) == 0
    assert G.main(["--score", "--run", str(run)]) == 0
    output = capsys.readouterr().out
    assert '"pending_api"' in output and '"inconclusive"' in output
    assert G.main(["--prepare", "--run", str(run)]) == 1


def test_real_provider_unit_returns_raw_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sentinel-not-a-real-key"
    seen = {}

    class Raw:
        status_code, content = 200, _body("nvidia/nemotron-3-super-120b-a12b", _answer("P_reward"))

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = self
            self.completions = self
            self.with_raw_response = self

        def create(self, **kwargs):
            seen["request"] = kwargs
            return seen.pop("reply")() if callable(seen.get("reply")) else Raw()

    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    monkeypatch.setattr(openai, "OpenAI", Client)
    provider = G.NvidiaChatProvider(G.load_protocol()["provider"])
    request = {"slot_id": "01-x-A", "model": "nvidia/nemotron-3-super-120b-a12b", "messages": [],
               "params": G.load_protocol()["request_params"]}
    reply = provider(request)
    assert reply["status_code"] == 200 and secret not in repr(reply)
    assert seen["api_key"] == secret and seen["max_retries"] == 0 and seen["timeout"] == 300
    assert seen["request"] == {"messages": [], **G.F.request_params("nvidia/nemotron-3-super-120b-a12b")}
    url = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")

    def not_found():
        raise openai.NotFoundError("not found", response=httpx.Response(404, request=url, content=b"{}"), body=None)

    seen["reply"] = not_found
    assert provider(request) == {"status_code": 404, "body": b"{}", "error_type": "NotFoundError"}

    def broken():
        raise openai.APIConnectionError(request=url)

    seen["reply"] = broken
    assert provider(request) == {"status_code": None, "body": None, "error_type": "APIConnectionError"}
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert provider(request)["error_type"] == "KeyUnavailable"


def test_protocol_is_unexecuted_draft_and_sources_avoid_network_imports() -> None:
    protocol = G.load_protocol()
    assert (protocol["status"], protocol["executed"], protocol["approval_granted"]) == ("DRAFT_not_executed", False,
                                                                                         False)
    assert protocol["expected_without_calls"] == {"A": "pending_api", "B": "pending_api", "outcome": "inconclusive"}
    assert not any(protocol["claims"].values())
    source = (ROOT / "evals/nemotron35_value_compare_call_v1.py").read_text(encoding="utf-8")
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any(name in line for line in top for name in ("openai", "httpx", "requests", "urllib", "socket"))
    banned = ("Team " + "Mollet", chr(0xC9C1) + chr(0xC811))
    for path in (G.PROTOCOL, ROOT / "docs/NEMOTRON35-VALUE-COMPARISON.md"):
        assert not any(word in path.read_text(encoding="utf-8") for word in banned)
