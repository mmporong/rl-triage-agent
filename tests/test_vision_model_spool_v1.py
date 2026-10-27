from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import grounded_vision_observer_v1 as OBSERVER
import vision_model_spool_v1 as S


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    spool, ledger = tmp_path / "spool", tmp_path / "ledger"
    spool.mkdir(parents=True)
    ledger.mkdir(parents=True)
    return spool, ledger


def _messages(index: int = 1) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": f"frame-{index}"}]}]


def _provider(index: int = 1) -> dict:
    return {
        "response_id": f"response-{index}",
        "provider_model": "private/nvidia/nemotron-3.5-super-120b-a12b",
        "finish_reason": "stop",
        "content": json.dumps({"index": index}),
        "reasoning_content": None,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "wall_s": 0.25,
    }


def _request(spool: Path, job: str = "job-a", sequence: int = 1,
             *, params: dict | None = None) -> tuple[Path, Path]:
    with pytest.raises(S.SpoolPendingError):
        S.remote_complete(
            _messages(sequence), params or deepcopy(OBSERVER.REQUEST_PARAMS),
            spool=spool, job_id=job, sequence=sequence, timeout_s=0,
        )
    return spool / "requests" / job / f"{sequence:04d}.json", (
        spool / "responses" / job / f"{sequence:04d}.json")


def test_roundtrip_preserves_raw_provider_and_replays_without_resend(tmp_path):
    spool, ledger = _roots(tmp_path)
    request, response = _request(spool)
    response.parent.rmdir()
    request_before = request.read_bytes()
    calls = []
    raw = _provider()

    def provider(messages, params):
        calls.append((messages, params))
        return deepcopy(raw)

    assert S.process_one(request, response, ledger, provider) == raw
    assert calls == [(_messages(), OBSERVER.REQUEST_PARAMS)]
    assert request.read_bytes() == request_before
    assert S.remote_complete(
        _messages(), deepcopy(OBSERVER.REQUEST_PARAMS), spool=spool,
        job_id="job-a", sequence=1, timeout_s=0,
    ) == raw
    assert request.read_bytes() == request_before

    response.unlink()
    assert S.process_one(request, response, ledger, provider) == raw
    assert response.is_file()
    assert len(calls) == 1


def test_timeout_and_unknown_pending_never_resend(tmp_path):
    spool, _ledger = _roots(tmp_path)
    request, _response = _request(spool)
    before = request.read_bytes()
    sleeps = []
    ticks = iter((0.0, 0.0, 0.1, 0.2))

    with pytest.raises(S.SpoolPendingError):
        S.remote_complete(
            _messages(), deepcopy(OBSERVER.REQUEST_PARAMS), spool=spool,
            job_id="job-a", sequence=1, timeout_s=0.2, poll_interval_s=0.1,
            monotonic=lambda: next(ticks), sleep=sleeps.append,
        )
    assert request.read_bytes() == before
    assert sleeps == [0.1, 0.1]
    assert len(list((spool / "requests/job-a").iterdir())) == 1
    with pytest.raises(S.SpoolIntegrityError, match="기존 request"):
        S.remote_complete(
            _messages(2), deepcopy(OBSERVER.REQUEST_PARAMS), spool=spool,
            job_id="job-a", sequence=1, timeout_s=0,
        )
    assert request.read_bytes() == before


def test_provider_exception_records_class_only_and_pending_blocks_duplicates(tmp_path):
    spool, ledger = _roots(tmp_path)
    request, response = _request(spool)
    calls = 0

    def provider(_messages, _params):
        nonlocal calls
        calls += 1
        raise RuntimeError("SECRET api-key header value")

    with pytest.raises(S.ProviderCallError) as caught:
        S.process_one(request, response, ledger, provider)
    assert caught.value.provider_error_type == "builtins.RuntimeError"
    assert calls == 1 and not response.exists()
    text = "\n".join(path.read_text(encoding="utf-8") for path in ledger.rglob("*.json"))
    assert "builtins.RuntimeError" in text
    assert "SECRET" not in text and "api-key" not in text and "header value" not in text
    assert S._audit(ledger)["pending"] == 1

    with pytest.raises(S.SpoolPendingError):
        S.process_one(request, response, ledger, provider)
    assert calls == 1


def test_params_request_and_response_tampering_fail_closed(tmp_path):
    spool, ledger = _roots(tmp_path)
    wrong = {**OBSERVER.REQUEST_PARAMS, "max_tokens": 2048}
    request, response = _request(spool, params=wrong)
    with pytest.raises(S.SpoolIntegrityError, match="params"):
        S.process_one(request, response, ledger, lambda *_: _provider())

    spool2, ledger2 = _roots(tmp_path / "request-tamper")
    request2, response2 = _request(spool2)
    body = json.loads(request2.read_text(encoding="utf-8"))
    body["request_id"] = "forged"
    request2.write_text(json.dumps(body) + "\n", encoding="utf-8")
    with pytest.raises(S.SpoolIntegrityError, match="hash 또는 id"):
        S.process_one(request2, response2, ledger2, lambda *_: _provider())

    spool3, ledger3 = _roots(tmp_path / "response-tamper")
    request3, response3 = _request(spool3)
    S.process_one(request3, response3, ledger3, lambda *_: _provider())
    reply = json.loads(response3.read_text(encoding="utf-8"))
    reply["request_id"] = "forged"
    response3.write_text(json.dumps(reply) + "\n", encoding="utf-8")
    with pytest.raises(S.SpoolIntegrityError, match="결합"):
        S.remote_complete(
            _messages(), deepcopy(OBSERVER.REQUEST_PARAMS), spool=spool3,
            job_id="job-a", sequence=1, timeout_s=0,
        )


def test_other_pending_reservation_blocks_new_provider(tmp_path):
    spool, ledger = _roots(tmp_path)
    request1, response1 = _request(spool, sequence=1)
    with pytest.raises(S.ProviderCallError):
        S.process_one(request1, response1, ledger,
                      lambda *_: (_ for _ in ()).throw(OSError("private detail")))
    request2, response2 = _request(spool, sequence=2)
    called = False

    def provider(*_args):
        nonlocal called
        called = True
        return _provider(2)

    with pytest.raises(S.SpoolPendingError, match="다른 pending"):
        S.process_one(request2, response2, ledger, provider)
    assert not called


def test_global_cap_32_stops_before_provider(tmp_path):
    spool, ledger = _roots(tmp_path)
    calls = 0

    def provider(_messages, _params):
        nonlocal calls
        calls += 1
        return _provider(calls)

    for sequence in range(1, 33):
        request, response = _request(spool, sequence=sequence)
        S.process_one(request, response, ledger, provider)
    audit = S._audit(ledger)
    assert calls == 32
    assert (audit["reserved"], audit["pending"], audit["remaining"]) == (32, 0, 0)
    request, response = _request(spool, sequence=33)
    with pytest.raises(S.SpoolCapError):
        S.process_one(request, response, ledger, provider)
    assert calls == 32


def test_path_escape_symlink_and_identity_mismatch_are_rejected(tmp_path):
    spool, ledger = _roots(tmp_path)
    with pytest.raises(ValueError, match="job_id"):
        S.remote_complete(_messages(), deepcopy(OBSERVER.REQUEST_PARAMS), spool=spool,
                          job_id="../escape", sequence=1, timeout_s=0)

    request, _response = _request(spool)
    outside = tmp_path / "outside.json"
    with pytest.raises(S.SpoolIntegrityError, match="경로와 내부 identity"):
        S.process_one(request, outside, ledger, lambda *_: _provider())

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "spool-link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("이 호스트에서 directory symlink를 만들 수 없다")
    with pytest.raises(S.SpoolIntegrityError, match="symlink"):
        S.remote_complete(_messages(), deepcopy(OBSERVER.REQUEST_PARAMS), spool=link,
                          job_id="job-a", sequence=1, timeout_s=0)


def test_cli_uses_existing_default_complete_once(tmp_path, monkeypatch, capsys):
    spool, ledger = _roots(tmp_path)
    request, response = _request(spool)
    calls = []

    def complete(messages, params):
        calls.append((messages, params))
        return _provider()

    monkeypatch.setattr(S.OBSERVER, "default_complete", complete)
    assert S.main(["process-one", "--request", str(request), "--response", str(response),
                   "--ledger", str(ledger)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}
    assert calls == [(_messages(), OBSERVER.REQUEST_PARAMS)]
