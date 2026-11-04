from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "public_export_v2", HERE.parent / "evals" / "summarize_registered_recovery_comparison_v2.py"
)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _method(name: str, *, hold: int = 0, not_recovered: int = 4) -> dict:
    return {"method": name, "outcomes": {"recovered": 0, "not_recovered": not_recovered,
                                           "hold": hold, "unknown": 0}}


def _summary() -> dict:
    methods = [
        _method("fixed_sequence"), _method("static_rule"), _method("one_shot_llm"),
        _method("iterative_agent", hold=1, not_recovered=3),
    ]
    arms = []
    for index in range(16):
        arms.append({
            "method": methods[index % 4]["method"],
            "execution_origin": "verified_prior_effect" if index < 8 else "continuation_dispatch",
            "recovery_attempted": index != 9,
            "recovery_outcome": "hold" if index == 9 else "not_recovered",
        })
    return {"arms": arms, "methods": methods}


def _ledger() -> dict:
    return {
        "model_api_totals": {"known_calls": 25, "known_input_tokens": 21472.0,
                             "known_output_tokens": 1063.0},
        "buckets": [{"bucket_id": "diagnostic.legacy_v3.known_events",
                     "metrics": {"model_calls": 4.0, "input_tokens": 2614.0,
                                 "output_tokens": 128.0}}],
    }


def _verified() -> tuple[dict, dict, dict]:
    fixed = {"path": "fixed", "sha256": "a" * 64}
    checkpoint = {"path": "checkpoint", "sha256": "b" * 64}
    binding = {"source_anchor": "c" * 40}
    verified = {
        "historical_source_validation": {"source_anchor": binding["source_anchor"],
                                         "historical_source_authoritative": True},
        "backend_receipt_validation": {"request_id": "request-1",
                                       "receipt_ref": {"sha256": "d" * 64}},
        "post_observation": {"fixed_report": fixed, "new_checkpoint": checkpoint},
        "old_refs": {"fixed_report": fixed, "new_checkpoint": checkpoint},
    }
    done = {"request_id": "request-1", "receipt": {"result_sha": "d" * 64}}
    return binding, verified, done


def test_public_metrics_separate_diagnostic_and_recovery() -> None:
    value = EXPORT._public_metrics(_summary(), _ledger())
    assert value["recovery"] == {
        "planned_arms": 16, "attempted_arms": 15, "imported_completed_arms": 8,
        "fresh_attempted_arms": 7, "hold_arms": 1, "recovered_arms": 0,
        "not_recovered_arms": 15, "fresh_sdk_launches": 14, "api_calls": 0,
        "hold_is_not_a_saving_or_success": True,
    }
    assert value["diagnostic_current"] == {
        "planned_arms": 24, "api_calls": 21, "input_tokens": 18858,
        "output_tokens": 935,
        "derivation": "whole_model_totals_minus_separate_legacy_diagnostic",
    }


def test_historical_cache_binds_source_request_and_endpoint(tmp_path: Path) -> None:
    binding, verified, done = _verified()
    folder = tmp_path / "historical"
    folder.mkdir()
    posts: dict = {}
    EXPORT._cache_historical_post(posts, folder=folder, binding=binding,
                                  verified=verified, done=done)
    fallback_calls: list[tuple] = []

    def fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return {"fallback": True}

    assert EXPORT._historical_post(posts, folder, "request-1", fallback) == verified["post_observation"]
    assert fallback_calls == []
    with pytest.raises(EXPORT.ExportError, match="request identity"):
        EXPORT._historical_post(posts, folder, "request-2", fallback)


@pytest.mark.parametrize("field", ["source", "receipt", "endpoint"])
def test_historical_cache_rejects_tamper(tmp_path: Path, field: str) -> None:
    binding, verified, done = _verified()
    if field == "source":
        verified["historical_source_validation"]["source_anchor"] = "e" * 40
    elif field == "receipt":
        done["receipt"]["result_sha"] = "e" * 64
    else:
        verified["post_observation"]["new_checkpoint"] = {
            "path": "other", "sha256": "e" * 64}
    with pytest.raises(EXPORT.ExportError):
        EXPORT._cache_historical_post({}, folder=tmp_path, binding=binding,
                                      verified=verified, done=done)


def test_cached_post_digest_tamper_is_rejected(tmp_path: Path) -> None:
    binding, verified, done = _verified()
    posts: dict = {}
    EXPORT._cache_historical_post(posts, folder=tmp_path, binding=binding,
                                  verified=verified, done=done)
    key = (tmp_path.resolve(), "request-1")
    posts[key]["post"] = deepcopy(posts[key]["post"])
    posts[key]["post"]["new_checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(EXPORT.ExportError, match="cached historical post changed"):
        EXPORT._historical_post(posts, tmp_path, "request-1", lambda *_a, **_k: {})


def test_public_guard_rejects_paths() -> None:
    with pytest.raises(EXPORT.ExportError, match="path"):
        EXPORT._assert_public({"source": "C:/private/result.json"})
